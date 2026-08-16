from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from caster.bridge import (
    alternate_ARCHIVE_MOMENT,
    PREDICTIVE_CONTRACTS,
    read_bridge_config,
    score_draw_rows,
)
from caster.data import add_task_columns, filter_ledger_archive_for_task, task_from_args, task_metadata
from caster.filter import (
    availability_validation_metadata,
    compute_log_evidence,
    evidence_availability_by_model,
    hierarchical_update_from_log_evidence,
    initialize_hierarchical_weights,
    native_forecast_rows,
    posterior_predictive_readout_asof,
    summarize_hierarchical_posterior,
    write_json,
)
from caster.forecast import validate_draw_kernel_inputs
from caster.models import read_registry
from caster.utils import RuntimeLogger, write_timing_log
from run_caster_from_archive import (
    POSTERIOR_READOUT_POLICY_AVAILABLE,
    POSTERIOR_UPDATE_POLICY_HOLDOUT,
    POSTERIOR_UPDATE_POLICY_PREQUENTIAL,
    RELEASE_AVAILABILITY_RULE,
    _build_asof_posterior_readout_validation,
    _canonical_posterior_update_policy,
    _calibration_source_split,
    _draw_kernel_calibration_metadata,
    _enforce_formal_asof_validation,
    _evidence_unit_metadata,
    _posterior_update_scope,
    _project_formal_endpoint_inputs,
    _parse_csv,
    _read_bridge_metadata,
    _readout_predictive_interval_source,
    _score_update_rows,
    _selected_registry,
    _validate_posterior_update_policy,
    _validate_predictive_contract_identity,
    _validate_archive_contract,
    _validate_embargo_update_scope,
    _validate_scored_update_ledger,
    _write_asof_posterior_readout_validation,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = ArgumentParser(description="Run hierarchical CASTER from an existing forecast archive and frozen bridge config.")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--draws", default="")
    ap.add_argument("--registry", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--bridge-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--update-splits", default="train,val")
    ap.add_argument(
        "--update-release-cutoff",
        default="",
        help=(
            "Optional inclusive release-time cutoff for update evidence. "
            "Used by the frozen-pretest ablation so embargo rows released "
            "after the first test origin cannot update the posterior."
        ),
    )
    ap.add_argument("--readout-split", default="test")
    ap.add_argument(
        "--posterior-update-policy",
        choices=[POSTERIOR_UPDATE_POLICY_HOLDOUT, POSTERIOR_UPDATE_POLICY_PREQUENTIAL],
        default=POSTERIOR_UPDATE_POLICY_HOLDOUT,
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-id", default="", help="Optional Benchmark task id, e.g. benchmark_b_covid, benchmark_b_flu, or benchmark_b_pooled.")
    ap.add_argument("--target-components", default="", help="Comma-separated target components for this task.")
    ap.add_argument("--posterior-scope", default="", choices=["", "component_stratified", "pooled_sensitivity", "pooled_shared_posterior"])
    ap.add_argument("--score-source", default="archive_moment", choices=["archive_moment", "draw_kernel"])
    ap.add_argument(
        "--predictive-contract",
        choices=PREDICTIVE_CONTRACTS,
        default=alternate_ARCHIVE_MOMENT,
        help="Readout contract frozen in --bridge-config.",
    )
    ap.add_argument(
        "--method-id",
        default="",
        choices=["", "caster_hierarchical", "caster_hierarchical_draw_kernel"],
    )
    ap.add_argument("--draw-kernel-bandwidth-source", default="bridge_sigma_validation_frozen")
    args = ap.parse_args()
    if args.score_source == "draw_kernel" and not args.draws:
        raise SystemExit("--score-source draw_kernel requires --draws")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = RuntimeLogger()
    update_splits = _parse_csv(args.update_splits)
    readout_split = str(args.readout_split)
    _validate_posterior_update_policy(args.posterior_update_policy, update_splits, readout_split)
    posterior_update_policy = _canonical_posterior_update_policy(args.posterior_update_policy)

    with timer.measure("load_inputs"):
        ledger = pd.read_csv(args.ledger)
        archive = pd.read_csv(args.archive)
        draws = (
            pd.read_csv(
                args.draws,
                usecols=[
                    "forecast_id",
                    "model_id",
                    "particle_id",
                    "draw_id",
                    "draw",
                ],
            )
            if args.score_source == "draw_kernel"
            else pd.DataFrame()
        )
        task = task_from_args(
            task_id=args.task_id,
            target_components=args.target_components,
            posterior_scope=args.posterior_scope,
            dataset=ledger["dataset"].dropna().astype(str).iloc[0] if "dataset" in ledger.columns and not ledger.empty else "",
        )
        ledger, archive = filter_ledger_archive_for_task(ledger, archive, task)
        ledger, archive, formal_endpoint_metadata = (
            _project_formal_endpoint_inputs(ledger, archive, task)
        )
        if args.score_source == "draw_kernel" and not draws.empty:
            ledger_ids = set(ledger["forecast_id"].astype(str))
            draws = draws[draws["forecast_id"].astype(str).isin(ledger_ids)].copy()
        registry_all = read_registry(args.registry)
        selection = pd.read_csv(args.selection)
        registry, model_ids = _selected_registry(registry_all, selection)
        archive = archive[archive["model_id"].astype(str).isin(model_ids)].copy()
        if archive.empty:
            raise SystemExit("archive has no rows for selected model ids")
        bridge, rho = read_bridge_config(args.bridge_config)
        if rho is None:
            raise SystemExit("bridge config must contain frozen rho; run NEW-BRIDGE validation-only calibration first")
        bridge_metadata = _read_bridge_metadata(args.bridge_config)
        predictive_contract = _validate_predictive_contract_identity(
            args.predictive_contract,
            bridge,
            bridge_metadata,
        )
        registry.to_csv(out_dir / "model_registry.csv", index=False)
        selection.to_csv(out_dir / "candidate_selection_log.csv", index=False)

    _validate_embargo_update_scope(ledger, update_splits, readout_split)

    with timer.measure("archive_contract_validation"):
        _validate_archive_contract(archive, ledger, model_ids, out_dir)

    split = ledger["split"].astype(str)
    update_ledger = ledger[split.isin(update_splits)].copy()
    update_rows_before_release_cutoff = int(len(update_ledger))
    update_release_cutoff = pd.NaT
    if str(args.update_release_cutoff).strip():
        update_release_cutoff = pd.Timestamp(args.update_release_cutoff)
        release_time = pd.to_datetime(update_ledger["release_time"], errors="coerce")
        update_ledger = update_ledger[
            release_time.notna() & release_time.le(update_release_cutoff)
        ].copy()
    readout_ledger = ledger[split == readout_split].copy()
    if update_ledger.empty:
        raise SystemExit(f"no ledger rows for update splits {update_splits}")
    if readout_ledger.empty:
        raise SystemExit(f"no ledger rows for readout split {readout_split!r}")
    _validate_scored_update_ledger(update_ledger, out_dir)
    readout_ids = set(readout_ledger["forecast_id"].astype(str))
    update_archive = archive[archive["forecast_id"].astype(str).isin(set(update_ledger["forecast_id"].astype(str)))].copy()
    native_update_archive = native_forecast_rows(
        update_archive, require_provenance=True
    )
    readout_archive = archive[archive["forecast_id"].astype(str).isin(readout_ids)].copy()

    with timer.measure("score_update_rows"):
        if args.score_source == "draw_kernel":
            update_draws = draws[
                draws["forecast_id"].astype(str).isin(set(update_ledger["forecast_id"].astype(str)))
                & draws["model_id"].astype(str).isin(set(model_ids))
            ].copy()
            native_pairs = native_update_archive[["forecast_id", "model_id"]].drop_duplicates()
            update_draws = update_draws.merge(
                native_pairs, on=["forecast_id", "model_id"], how="inner"
            )
            active_model_ids = sorted(
                native_update_archive["model_id"].astype(str).unique()
            )
            violations = validate_draw_kernel_inputs(
                update_draws, update_ledger, native_update_archive, active_model_ids
            )
            if not violations.empty:
                violations_path = out_dir / "draw_kernel_input_violations.csv"
                violations.to_csv(violations_path, index=False)
                raise SystemExit(f"draw-kernel input validation failed; see {violations_path}")
            scored_update = score_draw_rows(update_ledger, update_draws, bridge)
            release_meta = update_ledger[["forecast_id", "release_time"]].drop_duplicates("forecast_id")
            scored_update = scored_update.merge(release_meta, on="forecast_id", how="left")
            scored_update["release_time"] = pd.to_datetime(scored_update["release_time"])
        else:
            scored_update = _score_update_rows(update_ledger, native_update_archive, bridge)

    hp = initialize_hierarchical_weights(registry)
    initial_model_weights = hp.model_weights[["model_id", "family", "weight"]].copy()
    posterior_rows: list[pd.DataFrame] = []
    family_rows: list[pd.DataFrame] = []
    inner_rows: list[pd.DataFrame] = []
    evidence_rows: list[pd.DataFrame] = []
    with timer.measure("hierarchical_filter"):
        for release_time in sorted(pd.to_datetime(update_ledger["release_time"]).unique()):
            current = scored_update[scored_update["release_time"] == pd.Timestamp(release_time)]
            batch_ledger = update_ledger[
                pd.to_datetime(update_ledger["release_time"]).eq(pd.Timestamp(release_time))
            ].copy()
            availability = evidence_availability_by_model(
                current,
                batch_ledger,
                model_ids,
                structural_unavailable_rows=update_archive,
            )
            log_evidence = pd.DataFrame(
                [
                    {
                        "release_time": pd.Timestamp(release_time),
                        "model_id": str(model_id),
                        "log_evidence": compute_log_evidence(current, model_id=str(model_id))
                        if availability[str(model_id)] else 0.0,
                        "evidence_available": availability[str(model_id)],
                    }
                    for model_id in registry["model_id"].astype(str)
                ]
            )
            hp = hierarchical_update_from_log_evidence(
                hp.family_weights,
                hp.inner_weights,
                log_evidence,
                rho=float(rho),
            )
            model_snapshot = hp.model_weights.copy()
            model_snapshot["release_time"] = pd.Timestamp(release_time)
            model_snapshot["rho"] = float(rho)
            model_snapshot = add_task_columns(model_snapshot, task)
            family_snapshot = hp.family_weights.copy()
            family_snapshot["release_time"] = pd.Timestamp(release_time)
            family_snapshot["rho"] = float(rho)
            family_snapshot = add_task_columns(family_snapshot, task)
            inner_snapshot = hp.inner_weights.copy()
            inner_snapshot["release_time"] = pd.Timestamp(release_time)
            inner_snapshot["rho"] = float(rho)
            inner_snapshot = add_task_columns(inner_snapshot, task)
            log_evidence = add_task_columns(log_evidence, task)
            posterior_rows.append(model_snapshot)
            family_rows.append(family_snapshot)
            inner_rows.append(inner_snapshot)
            evidence_rows.append(log_evidence)

    posterior = pd.concat(posterior_rows, ignore_index=True)
    family_posterior = pd.concat(family_rows, ignore_index=True)
    inner_weights = pd.concat(inner_rows, ignore_index=True)
    evidence = pd.concat(evidence_rows, ignore_index=True)
    posterior.to_csv(out_dir / "hierarchical_posterior_path.csv", index=False)
    family_posterior.to_csv(out_dir / "family_posterior.csv", index=False)
    inner_weights.to_csv(out_dir / "inner_weights.csv", index=False)
    evidence.to_csv(out_dir / "hierarchical_evidence_log.csv", index=False)
    final_model = posterior.sort_values("release_time").groupby("model_id").tail(1)
    final_family = family_posterior.sort_values("release_time").groupby("family").tail(1)
    weight_cols = ["model_id", "family", "weight", "log_evidence", "model_ess", "task_id", "target_component", "posterior_scope"]
    posterior_weights = final_model[[c for c in weight_cols if c in final_model.columns]].copy()
    posterior_weights.to_csv(out_dir / "hierarchical_posterior_weights.csv", index=False)
    method_id = args.method_id or (
        "caster_hierarchical_draw_kernel"
        if args.score_source == "draw_kernel"
        else "caster_hierarchical"
    )
    readout_draws = (
        draws[
            draws["forecast_id"].astype(str).isin(readout_ids)
            & draws["model_id"].astype(str).isin(model_ids)
        ].copy()
        if args.score_source == "draw_kernel"
        else None
    )
    with timer.measure("hierarchical_forecast_readout"):
        readout = posterior_predictive_readout_asof(
            readout_ledger,
            readout_archive,
            posterior,
            initial_model_weights,
            posterior_update_policy=posterior_update_policy,
            release_availability_rule=RELEASE_AVAILABILITY_RULE,
            bridge_config=bridge,
            score_source=args.score_source,
            draws=readout_draws,
        )
        predictive_interval_source = _readout_predictive_interval_source(
            readout,
            predictive_contract,
        )
        readout = add_task_columns(readout, task)
        readout, asof_validation, validation_fields = _build_asof_posterior_readout_validation(
            readout=readout,
            update_ledger=update_ledger,
            method=method_id,
            bridge_metadata=bridge_metadata,
            posterior_update_policy=posterior_update_policy,
        )
        readout.to_csv(out_dir / "hierarchical_forecast_readout.csv", index=False)
        validation_path = _write_asof_posterior_readout_validation(out_dir, asof_validation, method_id)
        _enforce_formal_asof_validation(validation_fields, bridge_metadata, method=method_id)
    summary = summarize_hierarchical_posterior(final_model, final_family)
    write_json(summary, out_dir / "hierarchical_model_distribution.json")
    metadata = {
        "bridge_config": str(args.bridge_config),
        "bridge_distribution": bridge.distribution,
        "kernel_distribution": bridge.kernel_distribution,
        "predictive_contract": predictive_contract,
        "predictive_interval_source": predictive_interval_source,
        "gaussian_as_student_t_limit": bool(bridge_metadata.get("gaussian_as_student_t_limit", False)),
        "formal_student_t_nu": bridge_metadata.get("formal_student_t_nu", ""),
        "gamma_selection_policy": bridge_metadata.get("gamma_selection_policy", ""),
        "fixed_gamma": bridge_metadata.get("fixed_gamma", ""),
        "bridge_calibration_split": _calibration_source_split(bridge_metadata),
        "rho": float(rho),
        "filter_dynamics": "hierarchical_bayesian_evidence_update",
        "posterior_update_policy": posterior_update_policy,
        "posterior_update_policy_cli": args.posterior_update_policy,
        "posterior_update_scope": _posterior_update_scope(args.posterior_update_policy, update_splits, readout_split),
        "posterior_update_splits": update_splits,
        "readout_split": readout_split,
        "ledger_rows": int(len(ledger)),
        "posterior_update_rows": int(len(update_ledger)),
        "update_release_cutoff": (
            "" if pd.isna(update_release_cutoff) else update_release_cutoff.isoformat()
        ),
        "update_rows_before_release_cutoff": update_rows_before_release_cutoff,
        "update_rows_excluded_after_release_cutoff": (
            update_rows_before_release_cutoff - int(len(update_ledger))
        ),
        "readout_rows": int(len(readout_ledger)),
        "archive_rows": int(len(archive)),
        "ledger_sha256": _sha256_file(Path(args.ledger)),
        "archive_sha256": _sha256_file(Path(args.archive)),
        "registry_sha256": _sha256_file(Path(args.registry)),
        "selection_sha256": _sha256_file(Path(args.selection)) if args.selection else "",
        "bridge_config_sha256": _sha256_file(Path(args.bridge_config)),
        "update_archive_rows": int(len(update_archive)),
        "readout_archive_rows": int(len(readout_archive)),
        "families": int(registry["family"].astype(str).nunique()),
        "models": int(len(registry)),
        "test_rows_used_for_bridge_calibration": int(bridge_metadata.get("test_rows_used_for_tuning", 0)),
        "embargo_rows_in_ledger": int(ledger["split"].astype(str).eq("embargo").sum()),
        "embargo_rows_used_for_selection": 0,
        "embargo_rows_used_for_bridge_calibration": int(bridge_metadata.get("embargo_rows_used_for_tuning", 0)),
        "embargo_rows_used_for_reported_metrics": 0,
        "embargo_rows_used_for_posterior_update": int(update_ledger["split"].astype(str).eq("embargo").sum()),
        "embargo_posterior_update_policy": "released_evidence_only_asof",
        "test_rows_used_for_posterior_update": int((update_ledger["split"].astype(str) == readout_split).sum()),
        "test_rows_used_for_posterior_update_policy": (
            "released_evidence_only" if args.posterior_update_policy == POSTERIOR_UPDATE_POLICY_PREQUENTIAL else "not_used"
        ),
        "posterior_readout_policy": POSTERIOR_READOUT_POLICY_AVAILABLE,
        "release_availability_rule": RELEASE_AVAILABILITY_RULE,
        "asof_posterior_readout_validation": str(validation_path),
        "hierarchical_temperature_scope": "outer_family_only",
        "native_likelihoods_compared": False,
        "score_source": args.score_source,
        "score_update_basis": "draw_kernel_bridge" if args.score_source == "draw_kernel" else "archive_moment_bridge",
    }
    metadata.update(formal_endpoint_metadata)
    metadata.update(_evidence_unit_metadata(update_ledger))
    metadata.update(availability_validation_metadata(evidence))
    if args.score_source == "draw_kernel":
        draw_counts = draws.groupby("forecast_id")["draw_id"].nunique() if not draws.empty and "draw_id" in draws.columns else pd.Series(dtype=float)
        metadata.update(
            {
                "draw_kernel_bandwidth_source": args.draw_kernel_bandwidth_source,
                "draw_kernel_variance_source": "not_used_by_kernel_score",
                "draw_specific_parameter_selection": bool(
                    bridge_metadata.get("tau_selection_policy")
                    == "direct_continuous_log_scale_exact_joint_risk"
                ),
                "draws_path": str(args.draws),
                "draws_sha256": _sha256_file(Path(args.draws)),
                "draw_rows": int(len(draws)),
                "n_draws_per_forecast_id_min": int(draw_counts.min()) if not draw_counts.empty else 0,
                "n_draws_per_forecast_id_max": int(draw_counts.max()) if not draw_counts.empty else 0,
                "diagnostic_only": False,
            }
        )
        metadata.update(_draw_kernel_calibration_metadata(bridge_metadata))
        if method_id == "caster_hierarchical_draw_kernel":
            metadata.update(
                {
                    "ablation_id": "caster_hierarchical_draw_kernel",
                    "ablation_reference": "hierarchical_full",
                    "ablation_row_label": "Hierarchical draw-kernel evidence",
                }
            )
    metadata.update(task_metadata(task, ledger))
    metadata["selected_particles"] = model_ids
    metadata["candidate_count"] = int(len(model_ids))
    metadata["variant"] = "hierarchical"
    metadata["result_method_id"] = method_id
    metadata.update(validation_fields)
    write_json(metadata, out_dir / "hierarchical_run_metadata.json")
    if args.score_source == "draw_kernel" and method_id == "caster_hierarchical_draw_kernel":
        posterior_weights.to_csv(out_dir / "posterior_weights.csv", index=False)
        posterior.to_csv(out_dir / "posterior_path.csv", index=False)
        evidence.to_csv(out_dir / "evidence_log.csv", index=False)
        readout.to_csv(out_dir / "forecast_readout.csv", index=False)
        write_json(metadata, out_dir / "ablation_metadata.json")
        write_json(metadata, out_dir / "caster_run_metadata.json")
    timing = timer.summary(seed=args.seed)
    write_timing_log(timing, out_dir / "hierarchical_timing.json")
    if args.score_source == "draw_kernel":
        write_timing_log(timing, out_dir / "timing.json")
    print(
        f"ok out={out_dir} update_rows={len(update_ledger)} readout_rows={len(readout_ledger)} models={len(registry)} "
        f"families={registry['family'].nunique()} rho={rho} top_family={summary['top_family']} "
        f"top_model={summary['top_model']} family_ess={summary['family_ess']:.6f}"
    )


if __name__ == "__main__":
    main()
