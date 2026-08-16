from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from math import lgamma, log, pi
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from caster.bridge import (
    COHERENT_MEAN_PRESERVING_TRUNCATED_T,
    alternate_ARCHIVE_MOMENT,
    PREDICTIVE_CONTRACTS,
    read_bridge_config,
    score_archive_rows,
    score_draw_rows,
)
from caster.data import (
    add_task_columns,
    benchmark_b_context_binding,
    benchmark_b_input_fingerprint,
    build_benchmark_b_canonical_context,
    build_benchmark_b_posterior_update_batch,
    filter_ledger_archive_for_task,
    load_benchmark_b_context_contract,
    task_from_args,
    task_metadata,
)
from caster.filter import (
    availability_validation_metadata,
    compute_log_evidence,
    compute_model_uniform_prior,
    discovered_model_distribution,
    evidence_availability_by_model,
    native_forecast_rows,
    posterior_predictive_readout_asof,
    summarize_model_distribution,
    update_outer_weights,
    validate_sleeping_model_archive,
    write_json,
)
from caster.forecast import validate_draw_kernel_inputs
from caster.models import read_registry
from caster.tasks import filter_rows_to_task_spec, load_task_spec
from caster.utils import RuntimeLogger, write_timing_log

POSTERIOR_UPDATE_POLICY_HOLDOUT = "holdout_train_val"
POSTERIOR_UPDATE_POLICY_PREQUENTIAL = "prequential_asof"
POSTERIOR_UPDATE_POLICY_PREQUENTIAL_CANONICAL = "prequential_asof_release_lte_origin"
POSTERIOR_READOUT_POLICY_AVAILABLE = "asof_release_time_lte_forecast_origin"
RELEASE_AVAILABILITY_RULE = "date_only_release_time_no_later_than_forecast_origin"
CASTER_PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAL_TASK_SPEC_PATH = CASTER_PROJECT_ROOT / "configs/caster_task_specs_v20.yaml"


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _formal_endpoint_task_id(ledger: pd.DataFrame, task) -> str:
    if task is not None and str(getattr(task, "task_id", "")):
        return str(task.task_id)
    if "dataset" not in ledger.columns or ledger.empty:
        return ""
    datasets = ledger["dataset"].dropna().astype(str).unique().tolist()
    if len(datasets) != 1:
        return ""
    dataset = str(datasets[0])
    if dataset.startswith("benchmark_a"):
        return "benchmark_a"
    if dataset.startswith("benchmark_b"):
        return "benchmark_b_pooled"
    return ""


def _project_formal_endpoint_inputs(
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    task,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    ""






    task_id = _formal_endpoint_task_id(ledger, task)
    before = int(len(ledger))
    if not task_id:
        return ledger.copy(), archive.copy(), {
            "formal_endpoint_projection_task_id": "",
            "formal_endpoint_projection_policy": "not_applicable_unknown_task",
            "ledger_rows_before_formal_endpoint_projection": before,
            "ledger_rows_after_formal_endpoint_projection": before,
            "intermediate_recursive_rows_excluded": 0,
        }
    spec = load_task_spec(FORMAL_TASK_SPEC_PATH, task_id)
    working_ledger = ledger.copy()
    strategy_inferred = "forecast_strategy" not in working_ledger.columns
    if strategy_inferred:
        horizons = pd.to_numeric(
            working_ledger["horizon"], errors="raise"
        ).astype(int)
        direct = set(spec.direct_horizons)
        recursive = set(spec.recursive_horizons)
        working_ledger["forecast_strategy"] = [
            spec.forecast_strategies[0]
            if horizon in direct
            else spec.forecast_strategies[1]
            if horizon in recursive
            else ""
            for horizon in horizons
        ]
    projected_ledger = filter_rows_to_task_spec(
        working_ledger, spec, require_complete=False
    )
    ids = set(projected_ledger["forecast_id"].astype(str))
    projected_archive = archive[
        archive["forecast_id"].astype(str).isin(ids)
    ].copy()
    if projected_archive.empty:
        raise ValueError(
            f"formal endpoint projection removed every archive row for {task_id}"
        )
    after = int(len(projected_ledger))
    return projected_ledger, projected_archive, {
        "formal_endpoint_projection_task_id": task_id,
        "formal_endpoint_projection_policy": (
            "formal_direct_and_recursive_endpoints_only"
        ),
        "ledger_rows_before_formal_endpoint_projection": before,
        "ledger_rows_after_formal_endpoint_projection": after,
        "intermediate_recursive_rows_excluded": int(before - after),
        "forecast_strategy_inferred_from_disjoint_formal_horizons": bool(
            strategy_inferred
        ),
        "intermediate_recursive_steps_used_for_posterior_evidence": False,
        "intermediate_recursive_steps_used_for_reported_metrics": False,
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    import hashlib

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_benchmark_b_runtime_context(
    *,
    panel_path: str | Path,
    contract_path: str | Path,
    frozen_selection_manifest_path: str | Path,
) -> tuple[pd.DataFrame, object, dict[str, object], str]:
    ""

    panel_file = Path(panel_path)
    manifest_file = Path(frozen_selection_manifest_path)
    if not panel_file.is_file():
        raise SystemExit(f"Benchmark B panel is missing: {panel_file}")
    if not manifest_file.is_file():
        raise SystemExit(f"frozen selection manifest is missing: {manifest_file}")
    panel = pd.read_csv(panel_file, low_memory=False)
    contract = load_benchmark_b_context_contract(
        contract_path,
        panel_columns=panel.columns,
    )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("task_id") != "benchmark_b_pooled":
        raise SystemExit("frozen selection packet is not bound to benchmark_b_pooled")
    selection_context = manifest.get("selection_context", {})
    if (
        not isinstance(selection_context, dict)
        or selection_context.get("schema")
        != "caster_benchmark_b_canonical_context_v1"
        or selection_context.get("context_contract_sha256") != contract.sha256
        or manifest.get("canonical_context_schema")
        != "caster_benchmark_b_canonical_context_v1"
        or manifest.get("benchmark_b_context_contract_sha256") != contract.sha256
        or manifest.get("input_change_invalidates_forecast_posterior_agent_results")
        is not True
    ):
        raise SystemExit(
            "frozen selection packet predates the canonical Benchmark B context; "
            "old forecast/posterior/Agent artifacts cannot be reused"
        )
    return panel, contract, manifest, _sha256_file(manifest_file)


def _bind_benchmark_b_context(
    frame: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    contract: object,
    frozen_selection_manifest: dict[str, object],
    frozen_selection_manifest_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ""

    bound_groups: list[pd.DataFrame] = []
    bindings: list[dict[str, object]] = []
    for origin, group in frame.groupby("forecast_origin", sort=True):
        payload, context_sha256 = build_benchmark_b_canonical_context(
            panel,
            ledger,
            forecast_origin=origin,
            contract=contract,
            frozen_selection_manifest=frozen_selection_manifest,
            frozen_selection_manifest_sha256=frozen_selection_manifest_sha256,
        )
        binding = benchmark_b_context_binding(payload, context_sha256)
        binding["canonical_context_schema"] = payload["schema"]
        binding["context_contract_sha256"] = payload["context_contract_sha256"]
        binding["origin_query_forecast_count"] = payload[
            "origin_query_forecast_count"
        ]
        binding["origin_query_forecast_ids_sha256"] = payload[
            "origin_query_forecast_ids_sha256"
        ]
        binding["released_event_count"] = payload["released_event_count"]
        binding["canonical_context_json"] = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        binding["canonical_context_consumed_by_caster"] = True
        query_ids = sorted(group["forecast_id"].astype(str).unique())
        if (
            len(group) != int(payload["origin_query_forecast_count"])
            or len(query_ids) != len(group)
            or _sha256_json(query_ids)
            != str(payload["origin_query_forecast_ids_sha256"])
        ):
            raise ValueError(
                "Benchmark B CASTER readout differs from the origin-bound ledger query "
                f"at forecast_origin={origin}"
            )
        bindings.append(binding)
        group_bound = group.copy()
        for key, value in binding.items():
            if key not in {"task_id", "forecast_origin", "canonical_context_json"}:
                group_bound[key] = value
        bound_groups.append(group_bound)
    if not bound_groups:
        raise ValueError("cannot bind an empty Benchmark B forecast readout")
    binding_frame = pd.DataFrame(bindings).sort_values(
        "forecast_origin", kind="mergesort"
    )
    if binding_frame["forecast_origin"].astype(str).duplicated().any():
        raise ValueError("Benchmark B context binding must be unique per forecast origin")
    return pd.concat(bound_groups, ignore_index=True), binding_frame.reset_index(drop=True)


def _canonical_posterior_update_policy(policy: str) -> str:
    if policy == POSTERIOR_UPDATE_POLICY_PREQUENTIAL:
        return POSTERIOR_UPDATE_POLICY_PREQUENTIAL_CANONICAL
    if policy == POSTERIOR_UPDATE_POLICY_HOLDOUT:
        return POSTERIOR_UPDATE_POLICY_HOLDOUT
    raise ValueError(f"unknown posterior update policy {policy!r}")


def _posterior_update_scope(policy: str, update_splits: list[str], readout_split: str) -> str:
    if policy == POSTERIOR_UPDATE_POLICY_PREQUENTIAL:
        ordered = [s for s in update_splits if s != readout_split]
        ordered.append(f"{readout_split}_prequential")
        return ";".join(ordered)
    return ";".join(update_splits)


def _validate_posterior_update_policy(policy: str, update_splits: list[str], readout_split: str) -> None:
    if policy == POSTERIOR_UPDATE_POLICY_HOLDOUT:
        if readout_split in update_splits:
            raise SystemExit("readout split must not be included in posterior update splits")
        return
    if policy == POSTERIOR_UPDATE_POLICY_PREQUENTIAL:
        if readout_split not in update_splits:
            raise SystemExit("prequential_asof posterior update requires readout split in posterior update splits")
        return
    raise SystemExit(f"unknown posterior update policy {policy!r}")


def _validate_embargo_update_scope(
    ledger: pd.DataFrame,
    update_splits: list[str],
    readout_split: str,
) -> None:
    ""
    declared = ledger["split"].astype(str) if "split" in ledger.columns else pd.Series(dtype=str)
    if not declared.eq("embargo").any():
        return
    if str(readout_split) == "embargo":
        raise SystemExit("embargo is update-only and cannot be the readout/metric split")
    if "embargo" not in update_splits:
        raise SystemExit(
            "ledger contains embargo forecast origins; posterior update splits must include embargo"
        )


def _read_bridge_metadata(path: str | Path) -> dict[str, object]:
    p = Path(path)
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml

            data = yaml.safe_load(p.read_text()) or {}
        except Exception:
            data = {}
    else:
        data = json.loads(p.read_text())
    meta = data.get("calibration_metadata", {})
    return meta if isinstance(meta, dict) else {}


def _validate_predictive_contract_identity(
    requested_contract: object,
    bridge: object,
    bridge_metadata: dict[str, object],
) -> str:
    ""

    requested = str(requested_contract)
    frozen = str(
        getattr(bridge, "predictive_contract", alternate_ARCHIVE_MOMENT)
    )
    metadata_contract = str(
        bridge_metadata.get("predictive_contract", alternate_ARCHIVE_MOMENT)
    )
    if requested not in PREDICTIVE_CONTRACTS:
        raise SystemExit(f"unsupported predictive contract {requested!r}")
    if frozen != requested or metadata_contract != requested:
        raise SystemExit(
            "--predictive-contract must match the validation-frozen bridge "
            f"config and calibration metadata: requested={requested!r}, "
            f"config={frozen!r}, metadata={metadata_contract!r}"
        )
    if (
        requested == COHERENT_MEAN_PRESERVING_TRUNCATED_T
        and str(getattr(bridge, "distribution", "")) != "student_t"
    ):
        raise SystemExit(
            "coherent_mean_preserving_truncated_t requires a frozen Student-t bridge"
        )
    return requested


def _readout_predictive_interval_source(
    readout: pd.DataFrame,
    predictive_contract: str,
) -> str:
    ""

    if "predictive_contract" in readout.columns:
        contracts = sorted(
            value
            for value in readout["predictive_contract"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            if value
        )
        if contracts != [predictive_contract]:
            raise SystemExit(
                "forecast readout predictive contract differs from the frozen "
                f"identity: expected={predictive_contract!r}, found={contracts!r}"
            )
    elif predictive_contract != alternate_ARCHIVE_MOMENT:
        raise SystemExit("nonalternate forecast readout omitted predictive_contract")

    if "predictive_interval_source" in readout.columns:
        sources = sorted(
            value
            for value in readout["predictive_interval_source"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            if value
        )
        if len(sources) != 1:
            raise SystemExit(
                "forecast readout must declare one predictive_interval_source; "
                f"found={sources!r}"
            )
        return sources[0]
    if predictive_contract != alternate_ARCHIVE_MOMENT:
        raise SystemExit(
            "nonalternate forecast readout omitted predictive_interval_source"
        )
    return "alternate_gaussian_archive_moment_interval"


def _draw_kernel_calibration_metadata(
    bridge_metadata: dict[str, object],
) -> dict[str, object]:
    ""







    explicit_tau_source = bridge_metadata.get("draw_kernel_tau_source")
    if explicit_tau_source in (None, ""):
        tau_source: object = (
            "tau_equals_computed_sigma"
            if bridge_metadata.get("tau_equals_computed_sigma")
            else "validation_selected_component_horizon_tau"
        )
    else:
        tau_source = explicit_tau_source

    metadata: dict[str, object] = {"draw_kernel_tau_source": tau_source}
    if "dimensional_binding" in bridge_metadata:
        metadata["dimensional_binding"] = bridge_metadata["dimensional_binding"]
    return metadata


def _calibration_source_split(bridge_metadata: dict[str, object]) -> str:
    return str(
        bridge_metadata.get(
            "calibration_source_split",
            bridge_metadata.get("selection_source_split", bridge_metadata.get("calibration_split", "val")),
        )
    )


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "t", "yes", "y"})


def _evidence_unit_metadata(ledger: pd.DataFrame) -> dict[str, object]:
    natural_count = int(ledger["natural_event_id"].astype(str).nunique()) if "natural_event_id" in ledger.columns else int(len(ledger))
    mode_counts = (
        {str(k): int(v) for k, v in ledger["mode"].astype(str).value_counts().sort_index().items()}
        if "mode" in ledger.columns else {}
    )
    strategy_counts = (
        {str(k): int(v) for k, v in ledger["forecast_strategy"].astype(str).value_counts().sort_index().items()}
        if "forecast_strategy" in ledger.columns else {}
    )
    return {
        "evidence_unit": "ledger_task_row",
        "posterior_natural_event_normalization": False,
        "same_outcome_multi_strategy_scores_accumulated": True,
        "posterior_evidence_task_rows": int(len(ledger)),
        "posterior_evidence_natural_events": natural_count,
        "posterior_evidence_strategy_duplicate_rows": int(len(ledger) - natural_count),
        "posterior_evidence_mode_counts": mode_counts,
        "posterior_evidence_strategy_counts": strategy_counts,
        "evidence_interpretation": (
            "Each direct/recursive-rollout horizon task row contributes one bridge score; "
            "shared natural outcomes are intentionally not deduplicated."
        ),
    }


def _asof_readout_validation_fields(readout: pd.DataFrame) -> dict[str, object]:
    if readout.empty:
        return {
            "readout_rows_using_prior_snapshot": 0,
            "readout_rows_future_snapshot_violation": 0,
            "readout_rows_self_target_update_violation": 0,
            "max_snapshot_after_origin_days": 0.0,
            "max_stale_posterior_age_days": 0.0,
            "p95_stale_posterior_age_days": 0.0,
        }
    required = {"forecast_origin", "posterior_snapshot_time", "used_prior_snapshot"}
    missing = sorted(required - set(readout.columns))
    if missing:
        raise ValueError(f"as-of readout missing validation columns {missing}")
    origin = pd.to_datetime(readout["forecast_origin"], errors="coerce")
    snapshot = pd.to_datetime(readout["posterior_snapshot_time"], errors="coerce")
    used_prior = _bool_series(readout["used_prior_snapshot"])
    if "future_snapshot_violation" in readout.columns:
        violations = _bool_series(readout["future_snapshot_violation"])
    else:
        violations = snapshot.notna() & (snapshot > origin) & ~used_prior
    self_violations = (
        _bool_series(readout["self_target_update_violation"])
        if "self_target_update_violation" in readout.columns
        else pd.Series(False, index=readout.index)
    )
    if violations.any():
        max_after = float(((snapshot - origin).dt.total_seconds() / 86400.0).where(violations).max())
    else:
        max_after = 0.0
    if "stale_posterior_age_days" in readout.columns:
        stale = pd.to_numeric(readout["stale_posterior_age_days"], errors="coerce")
    else:
        stale = ((origin - snapshot).dt.total_seconds() / 86400.0).where(snapshot.notna() & ~used_prior)
    stale_valid = stale.dropna()
    return {
        "readout_rows_using_prior_snapshot": int(used_prior.sum()),
        "readout_rows_future_snapshot_violation": int(violations.sum()),
        "readout_rows_self_target_update_violation": int(self_violations.sum()),
        "max_snapshot_after_origin_days": max_after,
        "max_stale_posterior_age_days": float(stale_valid.max()) if not stale_valid.empty else 0.0,
        "p95_stale_posterior_age_days": float(stale_valid.quantile(0.95)) if not stale_valid.empty else 0.0,
    }


def _build_asof_posterior_readout_validation(
    *,
    readout: pd.DataFrame,
    update_ledger: pd.DataFrame,
    method: str,
    bridge_metadata: dict[str, object],
    posterior_update_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if readout.empty:
        validation = pd.DataFrame(
            columns=[
                "dataset",
                "method",
                "forecast_id",
                "forecast_origin",
                "target_time",
                "release_time",
                "split",
                "posterior_snapshot_time",
                "used_prior_snapshot",
                "stale_posterior_age_days",
                "evidence_release_time_lte_origin",
                "evidence_release_available_before_origin",
                "future_snapshot_violation",
                "self_target_update_violation",
                "bridge_calibration_split",
                "test_rows_used_for_bridge_calibration",
                "posterior_update_policy",
                "release_availability_rule",
                "validation_status",
            ]
        )
        return readout, validation, _asof_readout_validation_fields(readout)

    required = {"forecast_id", "forecast_origin", "posterior_snapshot_time", "used_prior_snapshot"}
    missing = sorted(required - set(readout.columns))
    if missing:
        raise ValueError(f"readout missing validation columns {missing}")

    out = readout.copy()
    out["forecast_id"] = out["forecast_id"].astype(str)
    origin = pd.to_datetime(out["forecast_origin"], errors="coerce")
    snapshot = pd.to_datetime(out["posterior_snapshot_time"], errors="coerce")
    used_prior = _bool_series(out["used_prior_snapshot"])

    update_meta = update_ledger[["forecast_id", "release_time"]].copy()
    update_meta["forecast_id"] = update_meta["forecast_id"].astype(str)
    update_meta["__update_release_time__"] = pd.to_datetime(update_meta["release_time"], errors="coerce")
    update_release = update_meta.drop_duplicates("forecast_id").set_index("forecast_id")["__update_release_time__"]
    self_release = out["forecast_id"].map(update_release)
    self_target_violation = snapshot.notna() & self_release.notna() & (self_release <= snapshot)
    evidence_lte_origin = used_prior | (snapshot.notna() & (snapshot <= origin))
    evidence_available = evidence_lte_origin
    future_violation = ~evidence_lte_origin

    out["future_snapshot_violation"] = future_violation
    out["self_target_update_violation"] = self_target_violation
    if "stale_posterior_age_days" not in out.columns:
        out["stale_posterior_age_days"] = ((origin - snapshot).dt.total_seconds() / 86400.0).where(snapshot.notna())
    out["posterior_update_policy"] = posterior_update_policy
    out["release_availability_rule"] = RELEASE_AVAILABILITY_RULE

    bridge_split = _calibration_source_split(bridge_metadata)
    test_calib = int(bridge_metadata.get("test_rows_used_for_tuning", 0))
    invalid_calibration_source = bridge_split not in {"train", "val"}
    validation_status = np.where(
        future_violation | self_target_violation | invalid_calibration_source | (test_calib != 0),
        "FAIL",
        "PASS",
    )
    validation_cols = [
        "dataset",
        "forecast_id",
        "forecast_origin",
        "target_time",
        "release_time",
        "split",
        "posterior_snapshot_time",
        "used_prior_snapshot",
        "stale_posterior_age_days",
        "future_snapshot_violation",
        "self_target_update_violation",
        "posterior_update_policy",
        "release_availability_rule",
    ]
    validation = out[[c for c in validation_cols if c in out.columns]].copy()
    if "dataset" not in validation.columns:
        validation["dataset"] = ""
    validation.insert(1, "method", method)
    validation["evidence_release_time_lte_origin"] = evidence_lte_origin.to_numpy(dtype=bool)
    validation["evidence_release_available_before_origin"] = evidence_available.to_numpy(dtype=bool)
    validation["bridge_calibration_split"] = bridge_split
    validation["test_rows_used_for_bridge_calibration"] = test_calib
    validation["validation_status"] = validation_status
    ordered = [
        "dataset",
        "method",
        "forecast_id",
        "forecast_origin",
        "target_time",
        "release_time",
        "split",
        "posterior_snapshot_time",
        "used_prior_snapshot",
        "stale_posterior_age_days",
        "evidence_release_time_lte_origin",
        "evidence_release_available_before_origin",
        "future_snapshot_violation",
        "self_target_update_violation",
        "bridge_calibration_split",
        "test_rows_used_for_bridge_calibration",
        "posterior_update_policy",
        "release_availability_rule",
        "validation_status",
    ]
    validation = validation.reindex(columns=ordered)
    return out, validation, _asof_readout_validation_fields(out)


def _write_asof_posterior_readout_validation(out_dir: Path, validation: pd.DataFrame, method: str) -> Path:
    path = out_dir / "asof_posterior_readout_validation.csv"
    if path.exists():
        existing = pd.read_csv(path)
        if "method" in existing.columns:
            existing = existing[existing["method"].astype(str) != str(method)]
        validation = pd.concat([existing, validation], ignore_index=True, sort=False)
    validation.to_csv(path, index=False)
    return path


def _enforce_formal_asof_validation(validation_fields: dict[str, object], bridge_metadata: dict[str, object], *, method: str) -> None:
    if int(bridge_metadata.get("test_rows_used_for_tuning", 0)) != 0:
        raise SystemExit(f"{method} bridge calibration used test rows")
    calibration_source = _calibration_source_split(bridge_metadata)
    if calibration_source not in {"train", "val"}:
        raise SystemExit(f"{method} bridge calibration source must be train or val, got {calibration_source!r}")
    if int(validation_fields.get("readout_rows_future_snapshot_violation", 0)) != 0:
        raise SystemExit(f"{method} as-of validation failed: future snapshot violation")
    if int(validation_fields.get("readout_rows_self_target_update_violation", 0)) != 0:
        raise SystemExit(f"{method} as-of validation failed: self-target posterior update violation")


def _selected_registry(registry_all: pd.DataFrame, selection: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if "model_id" not in selection.columns:
        raise SystemExit("selection must contain model_id column")
    model_ids = selection["model_id"].astype(str).tolist()
    reg = registry_all.copy()
    reg["model_id"] = reg["model_id"].astype(str)
    missing = [mid for mid in model_ids if mid not in set(reg["model_id"])]
    if missing:
        raise SystemExit(f"selection contains model_id not in registry: {missing}")
    order = {mid: i for i, mid in enumerate(model_ids)}
    selected = reg[reg["model_id"].isin(model_ids)].copy()
    selected["__selection_order__"] = selected["model_id"].map(order)
    selected = selected.sort_values("__selection_order__").drop(columns=["__selection_order__"]).reset_index(drop=True)
    return selected, model_ids


def _validate_archive_contract(
    archive: pd.DataFrame,
    ledger: pd.DataFrame,
    model_ids: list[str],
    out_dir: Path,
) -> None:
    required_archive = {
        "forecast_id",
        "model_id",
        "particle_id",
        "forecast_origin",
        "target_time",
        "component",
        "pred_mean",
        "pred_var",
        "generated_at",
        "features_available_until",
        "forecast_status",
        "forecast_fallback_used",
        "forecast_failure_reason",
        "forecast_fallback_method",
        "proxy_fallback_used",
        "unsafe_native_proxy_executed",
    }
    required_ledger = {"forecast_id", "forecast_origin", "target_time", "component", "observed_mask", "split"}
    missing_archive = sorted(required_archive - set(archive.columns))
    missing_ledger = sorted(required_ledger - set(ledger.columns))
    if missing_archive or missing_ledger:
        raise SystemExit(f"missing required columns archive={missing_archive} ledger={missing_ledger}")

    violations: list[dict[str, object]] = []
    dup = archive.duplicated(["forecast_id", "model_id", "particle_id"])
    if dup.any():
        violations.append({"violation": "duplicate_prediction", "details": int(dup.sum())})
    if not np.isfinite(pd.to_numeric(archive["pred_mean"], errors="coerce")).all():
        violations.append({"violation": "nonfinite_pred_mean", "details": "pred_mean must be finite"})
    pred_var = pd.to_numeric(archive["pred_var"], errors="coerce")
    if (pred_var < 0).any() or not np.isfinite(pred_var).all():
        violations.append({"violation": "invalid_pred_var", "details": "pred_var must be finite and nonnegative"})

    sleeping_violations = validate_sleeping_model_archive(archive)
    if not sleeping_violations.empty:
        violations.extend(sleeping_violations.to_dict(orient="records"))

    ledger_meta = ledger[["forecast_id", "forecast_origin", "target_time", "component"]].drop_duplicates("forecast_id")
    joined = archive[["forecast_id", "forecast_origin", "target_time", "component", "generated_at", "features_available_until"]].merge(
        ledger_meta,
        on="forecast_id",
        how="left",
        suffixes=("", "_ledger"),
        indicator=True,
    )
    if (joined["_merge"] == "left_only").any():
        sample = joined.loc[joined["_merge"] == "left_only", "forecast_id"].astype(str).unique()[:10]
        violations.append({"violation": "forecast_id_not_in_ledger", "details": ",".join(sample)})
    known = joined[joined["_merge"] == "both"].copy()
    if not known.empty:
        origin = pd.to_datetime(known["forecast_origin"], format="mixed")
        generated = pd.to_datetime(known["generated_at"], format="mixed")
        features = pd.to_datetime(known["features_available_until"], format="mixed")
        target = pd.to_datetime(known["target_time"], format="mixed")
        target_ledger = pd.to_datetime(known["target_time_ledger"], format="mixed")
        checks = {
            "generated_after_origin": generated > origin,
            "features_after_origin": features > origin,
            "target_mismatch": target != target_ledger,
            "component_mismatch": known["component"].astype(str) != known["component_ledger"].astype(str),
        }
        for name, mask in checks.items():
            if mask.any():
                sample = known.loc[mask, "forecast_id"].astype(str).unique()[:10]
                violations.append({"violation": name, "details": ",".join(sample)})

    ledger_ids = set(ledger["forecast_id"].astype(str))
    for model_id in model_ids:
        rows = archive[archive["model_id"].astype(str) == str(model_id)]
        ids = set(rows["forecast_id"].astype(str))
        missing = sorted(ledger_ids - ids)
        extra = sorted(ids - ledger_ids)
        if missing:
            violations.append({"violation": "missing_ledger_predictions", "details": f"{model_id}:{','.join(missing[:10])}"})
        if extra:
            violations.append({"violation": "forecast_id_not_in_ledger_for_model", "details": f"{model_id}:{','.join(extra[:10])}"})

    if violations:
        path = out_dir / "archive_contract_violations.csv"
        pd.DataFrame(violations).to_csv(path, index=False)
        raise SystemExit(f"archive contract validation failed; see {path}")


def _validate_scored_update_ledger(update_ledger: pd.DataFrame, out_dir: Path) -> None:
    required = {"forecast_id", "forecast_origin", "target_time", "release_time", "observed_mask"}
    missing = sorted(required - set(update_ledger.columns))
    if missing:
        raise SystemExit(f"update ledger missing required no-leakage columns {missing}")
    rows = update_ledger.copy()
    observed = _bool_series(rows["observed_mask"])
    origin = pd.to_datetime(rows["forecast_origin"], errors="coerce")
    target = pd.to_datetime(rows["target_time"], errors="coerce")
    release = pd.to_datetime(rows["release_time"], errors="coerce")
    violations = rows[
        observed
        & (
            origin.isna()
            | target.isna()
            | release.isna()
            | ~(origin < target)
            | ~(target <= release)
        )
    ].copy()
    if not violations.empty:
        path = out_dir / "update_ledger_invariant_violations.csv"
        violations.to_csv(path, index=False)
        raise SystemExit(f"update ledger invariant failed for scored rows; see {path}")


def _transform(values: pd.Series, transform: str) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if transform == "identity":
        return arr
    if transform == "log1p":
        return np.log1p(np.maximum(arr, 0.0))
    raise ValueError(f"unknown transform {transform!r}")


def _delta_transform_var(mean: pd.Series, var: pd.Series, transform: str) -> np.ndarray:
    mu = np.maximum(pd.to_numeric(mean, errors="coerce").to_numpy(dtype=float), 0.0)
    v = np.maximum(pd.to_numeric(var, errors="coerce").to_numpy(dtype=float), 0.0)
    if transform == "identity":
        return v
    if transform == "log1p":
        return v / np.square(1.0 + mu)
    raise ValueError(f"unknown transform {transform!r}")


def _score_update_rows(update_ledger: pd.DataFrame, archive: pd.DataFrame, bridge) -> pd.DataFrame:
    scored = score_archive_rows(update_ledger, archive, bridge)
    release_meta = update_ledger[["forecast_id", "release_time"]].drop_duplicates("forecast_id")
    rows = scored.merge(release_meta, on="forecast_id", how="left")
    rows["release_time"] = pd.to_datetime(rows["release_time"])
    return rows


def main() -> None:
    ap = ArgumentParser(description="Run one-layer CASTER from an existing forecast archive and frozen validation-only bridge config.")
    ap.add_argument("--ledger", required=True)
    ap.add_argument(
        "--panel",
        default="",
        help="Required for benchmark_b_pooled canonical context binding.",
    )
    ap.add_argument("--archive", required=True)
    ap.add_argument("--draws", default="")
    ap.add_argument("--registry", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument(
        "--frozen-selection-manifest",
        default="",
        help="Shared CASTER--Agent frozen selection packet manifest for Benchmark B.",
    )
    ap.add_argument(
        "--benchmark-b-context-contract",
        default=str(
            Path(__file__).resolve().parents[3]
            / "configs/benchmark_b_context_v26_1.yaml"
        ),
    )
    ap.add_argument("--bridge-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--update-splits", default="train,val")
    ap.add_argument("--readout-split", default="test")
    ap.add_argument(
        "--posterior-update-policy",
        choices=[POSTERIOR_UPDATE_POLICY_HOLDOUT, POSTERIOR_UPDATE_POLICY_PREQUENTIAL],
        default=POSTERIOR_UPDATE_POLICY_HOLDOUT,
        help=(
            "holdout_train_val keeps the readout split out of posterior updates; "
            "prequential_asof maintains posterior snapshots through readout-period "
            "released evidence and reads them out when release_time is no later than forecast origin."
        ),
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
        choices=["", "caster_one_layer", "caster_one_layer_draw_kernel"],
        help="Optional result method identity; permits the selected result family to use draw-kernel evidence without becoming an ablation row.",
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
        benchmark_b_panel = pd.DataFrame()
        benchmark_b_contract = None
        frozen_selection_manifest: dict[str, object] | None = None
        frozen_selection_manifest_sha256 = ""
        if task is not None and task.task_id == "benchmark_b_pooled":
            if not args.panel or not args.frozen_selection_manifest:
                raise SystemExit(
                    "benchmark_b_pooled requires --panel and "
                    "--frozen-selection-manifest for shared CASTER--Agent "
                    "causal context binding"
                )
            (
                benchmark_b_panel,
                benchmark_b_contract,
                frozen_selection_manifest,
                frozen_selection_manifest_sha256,
            ) = _load_benchmark_b_runtime_context(
                panel_path=args.panel,
                contract_path=args.benchmark_b_context_contract,
                frozen_selection_manifest_path=args.frozen_selection_manifest,
            )
        ledger, archive = filter_ledger_archive_for_task(ledger, archive, task)
        ledger, archive, formal_endpoint_metadata = (
            _project_formal_endpoint_inputs(ledger, archive, task)
        )
        if args.score_source == "draw_kernel":
            draws = draws[draws["forecast_id"].astype(str).isin(set(ledger["forecast_id"].astype(str)))].copy()
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
    readout_ledger = ledger[split == readout_split].copy()
    if update_ledger.empty:
        raise SystemExit(f"no ledger rows for update splits {update_splits}")
    if readout_ledger.empty:
        raise SystemExit(f"no ledger rows for readout split {readout_split!r}")
    _validate_scored_update_ledger(update_ledger, out_dir)
    update_ids = set(update_ledger["forecast_id"].astype(str))
    readout_ids = set(readout_ledger["forecast_id"].astype(str))
    update_archive = archive[archive["forecast_id"].astype(str).isin(update_ids)].copy()
    native_update_archive = native_forecast_rows(
        update_archive, require_provenance=True
    )
    readout_archive = archive[archive["forecast_id"].astype(str).isin(readout_ids)].copy()

    with timer.measure("score_update_rows"):
        if args.score_source == "draw_kernel":
            update_draws = draws[
                draws["forecast_id"].astype(str).isin(update_ids)
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

    initial_weights = compute_model_uniform_prior(registry).rename(columns={"prior_weight": "weight"})[["model_id", "family", "weight"]]
    initial_weights.to_csv(out_dir / "initial_prior.csv", index=False)
    weights = initial_weights.copy()
    posterior_rows: list[pd.DataFrame] = []
    evidence_rows: list[pd.DataFrame] = []
    posterior_batch_validation_rows: list[dict[str, object]] = []
    with timer.measure("sequential_filter"):
        previous_release_time: pd.Timestamp | None = None
        for release_time in sorted(pd.to_datetime(update_ledger["release_time"]).unique()):
            batch_ledger = update_ledger[
                pd.to_datetime(update_ledger["release_time"]).eq(pd.Timestamp(release_time))
            ].copy()
            current = scored_update[scored_update["release_time"] == pd.Timestamp(release_time)]
            availability = evidence_availability_by_model(
                current,
                batch_ledger,
                model_ids,
                structural_unavailable_rows=update_archive,
            )
            if task is not None and task.task_id == "benchmark_b_pooled":
                posterior_batch = build_benchmark_b_posterior_update_batch(
                    native_update_archive,
                    update_ledger,
                    current_update_time=release_time,
                    previous_update_time=previous_release_time,
                )
                batch_ids = set(
                    posterior_batch.archived_forecasts["forecast_id"].astype(str)
                )
                scored_ids = set(
                    scored_update.loc[
                        scored_update["release_time"].eq(pd.Timestamp(release_time)),
                        "forecast_id",
                    ].astype(str)
                )
                if batch_ids != scored_ids:
                    raise SystemExit(
                        "Benchmark B posterior update differs from frozen archive + D_t "
                        f"at release_time={pd.Timestamp(release_time).date()}"
                    )
                posterior_batch_validation_rows.append(
                    {
                        "previous_update_time": posterior_batch.previous_update_time,
                        "current_update_time": posterior_batch.current_update_time,
                        "released_event_count": int(
                            len(posterior_batch.released_events)
                        ),
                        "released_event_sha256": posterior_batch.released_event_sha256,
                        "archived_forecast_row_count": int(
                            len(posterior_batch.archived_forecasts)
                        ),
                        "posterior_input_schema": posterior_batch.schema,
                        "posterior_current_x_forbidden": True,
                    }
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
            weights = update_outer_weights(
                weights,
                log_evidence,
                rho=float(rho),
            )
            snapshot = weights.copy()
            snapshot["release_time"] = pd.Timestamp(release_time)
            snapshot["rho"] = float(rho)
            snapshot = add_task_columns(snapshot, task)
            log_evidence = add_task_columns(log_evidence, task)
            posterior_rows.append(snapshot)
            evidence_rows.append(log_evidence)
            weights = weights[["model_id", "family", "weight"]]
            previous_release_time = pd.Timestamp(release_time)

    posterior = pd.concat(posterior_rows, ignore_index=True)
    evidence = pd.concat(evidence_rows, ignore_index=True)
    posterior.to_csv(out_dir / "posterior_path.csv", index=False)
    evidence.to_csv(out_dir / "evidence_log.csv", index=False)
    posterior_weights = posterior.sort_values("release_time").groupby("model_id").tail(1)
    weight_cols = ["model_id", "family", "weight", "log_evidence", "model_ess", "task_id", "target_component", "posterior_scope"]
    posterior_weights = posterior_weights[[c for c in weight_cols if c in posterior_weights.columns]]
    posterior_weights.to_csv(out_dir / "posterior_weights.csv", index=False)

    method_id = args.method_id or (
        "caster_one_layer_draw_kernel"
        if args.score_source == "draw_kernel"
        else "caster_one_layer"
    )
    context_bindings = pd.DataFrame()
    readout_draws = (
        draws[
            draws["forecast_id"].astype(str).isin(readout_ids)
            & draws["model_id"].astype(str).isin(model_ids)
        ].copy()
        if args.score_source == "draw_kernel"
        else None
    )
    with timer.measure("forecast_readout"):
        readout = posterior_predictive_readout_asof(
            readout_ledger,
            readout_archive,
            posterior,
            initial_weights,
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
        if task is not None and task.task_id == "benchmark_b_pooled":
            assert benchmark_b_contract is not None
            assert frozen_selection_manifest is not None
            readout, context_bindings = _bind_benchmark_b_context(
                readout,
                panel=benchmark_b_panel,
                ledger=ledger,
                contract=benchmark_b_contract,
                frozen_selection_manifest=frozen_selection_manifest,
                frozen_selection_manifest_sha256=frozen_selection_manifest_sha256,
            )
        readout, asof_validation, validation_fields = _build_asof_posterior_readout_validation(
            readout=readout,
            update_ledger=update_ledger,
            method=method_id,
            bridge_metadata=bridge_metadata,
            posterior_update_policy=posterior_update_policy,
        )
        if not context_bindings.empty:
            context_bindings.to_csv(
                out_dir / "benchmark_b_context_bindings.csv", index=False
            )
            context_columns = [
                "forecast_origin",
                "canonical_context_schema",
                "canonical_context_sha256",
                "visible_panel_sha256",
                "released_event_sha256",
                "frozen_selection_packet_sha256",
                "forecast_origin_binding_sha256",
                "context_contract_sha256",
            ]
            validation_context_bindings = context_bindings[context_columns].copy()
            asof_validation["forecast_origin"] = pd.to_datetime(
                asof_validation["forecast_origin"], errors="raise"
            )
            validation_context_bindings["forecast_origin"] = pd.to_datetime(
                validation_context_bindings["forecast_origin"], errors="raise"
            )
            asof_validation = asof_validation.merge(
                validation_context_bindings,
                on="forecast_origin",
                how="left",
                validate="many_to_one",
            )
        readout.to_csv(out_dir / "forecast_readout.csv", index=False)
        validation_path = _write_asof_posterior_readout_validation(out_dir, asof_validation, method_id)
        _enforce_formal_asof_validation(validation_fields, bridge_metadata, method=method_id)
    dist = discovered_model_distribution(posterior_weights)
    write_json(dist, out_dir / "model_distribution.json")
    draw_kernel_calibration_metadata = (
        _draw_kernel_calibration_metadata(bridge_metadata)
        if args.score_source == "draw_kernel"
        else {}
    )
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
        "prior_policy": "uniform_model_prior",
        "initial_prior": str(out_dir / "initial_prior.csv"),
        "initial_prior_family_mass": initial_weights.groupby("family")["weight"].sum().sort_index().to_dict(),
        "initial_prior_model_ess": float(summarize_model_distribution(initial_weights)["model_ess"]),
        "rho": float(rho),
        "filter_dynamics": "bayesian_evidence_update",
        "posterior_update_policy": posterior_update_policy,
        "posterior_update_policy_cli": args.posterior_update_policy,
        "posterior_update_scope": _posterior_update_scope(args.posterior_update_policy, update_splits, readout_split),
        "posterior_update_splits": update_splits,
        "readout_split": readout_split,
        "ledger_rows": int(len(ledger)),
        "posterior_update_rows": int(len(update_ledger)),
        "readout_rows": int(len(readout_ledger)),
        "archive_rows": int(len(archive)),
        "ledger_sha256": _sha256_file(Path(args.ledger)),
        "archive_sha256": _sha256_file(Path(args.archive)),
        "registry_sha256": _sha256_file(Path(args.registry)),
        "selection_sha256": _sha256_file(Path(args.selection)) if args.selection else "",
        "bridge_config_sha256": _sha256_file(Path(args.bridge_config)),
        "update_archive_rows": int(len(update_archive)),
        "readout_archive_rows": int(len(readout_archive)),
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
        "native_likelihoods_compared": False,
        "score_source": args.score_source,
        "score_update_basis": "draw_kernel_bridge" if args.score_source == "draw_kernel" else "archive_moment_bridge",
        "draw_kernel_bandwidth_source": args.draw_kernel_bandwidth_source if args.score_source == "draw_kernel" else "",
        "draw_specific_parameter_selection": bool(
            args.score_source == "draw_kernel"
            and bridge_metadata.get("tau_selection_policy")
            == "direct_continuous_log_scale_exact_joint_risk"
        ),
        "draw_kernel_tau_source": draw_kernel_calibration_metadata.get(
            "draw_kernel_tau_source", ""
        ),
        "draw_kernel_variance_source": "not_used_by_kernel_score" if args.score_source == "draw_kernel" else "",
        "draw_kernel_rho_source": "joint_validation_selected_global_rho" if args.score_source == "draw_kernel" else "",
        "draw_kernel_gamma_used": False if args.score_source == "draw_kernel" else True,
    }
    metadata.update(draw_kernel_calibration_metadata)
    metadata.update(formal_endpoint_metadata)
    if task is not None and task.task_id == "benchmark_b_pooled":
        assert benchmark_b_contract is not None
        panel_sha256 = _sha256_file(Path(args.panel))
        posterior_batch_validation_path = out_dir / "benchmark_b_posterior_batch_validation.csv"
        pd.DataFrame(posterior_batch_validation_rows).to_csv(
            posterior_batch_validation_path, index=False
        )
        metadata.update(
            {
                "benchmark_b_context_contract": str(
                    args.benchmark_b_context_contract
                ),
                "benchmark_b_context_contract_sha256": benchmark_b_contract.sha256,
                "frozen_selection_manifest": str(args.frozen_selection_manifest),
                "frozen_selection_packet_sha256": frozen_selection_manifest_sha256,
                "panel_sha256": panel_sha256,
                "benchmark_b_input_fingerprint": benchmark_b_input_fingerprint(
                    contract_sha256=benchmark_b_contract.sha256,
                    panel_sha256=panel_sha256,
                    ledger_sha256=_sha256_file(Path(args.ledger)),
                    frozen_selection_packet_sha256=frozen_selection_manifest_sha256,
                    candidate_registry_sha256=_sha256_file(Path(args.registry)),
                ),
                "canonical_context_shared_by_caster_and_agents": True,
                "canonical_context_schema": "caster_benchmark_b_canonical_context_v1",
                "context_binding_rows": int(len(context_bindings)),
                "context_binding_artifact": str(
                    out_dir / "benchmark_b_context_bindings.csv"
                ),
                "posterior_batch_validation": str(posterior_batch_validation_path),
                "posterior_update_inputs": "frozen_forecast_archive_and_released_batch_D_t_only",
                "posterior_current_x_forbidden": True,
                "no_learning_algorithm_difference_preserved": True,
                "old_forecast_posterior_agent_result_reuse_allowed": False,
            }
        )
    metadata.update(_evidence_unit_metadata(update_ledger))
    metadata.update(availability_validation_metadata(evidence))
    metadata.update(task_metadata(task, ledger))
    metadata["selected_particles"] = model_ids
    metadata["candidate_count"] = int(len(model_ids))
    metadata["variant"] = (
        "one_layer_draw_kernel"
        if method_id == "caster_one_layer_draw_kernel"
        else "one_layer"
    )
    metadata["result_method_id"] = method_id
    if args.score_source == "draw_kernel":
        draw_counts = draws.groupby(["forecast_id", "model_id", "particle_id"])["draw_id"].nunique() if "draw_id" in draws.columns else pd.Series(dtype=float)
        metadata.update(
            {
                "draws_path": str(args.draws),
                "draws_sha256": _sha256_file(Path(args.draws)),
                "draw_rows": int(len(draws)),
                "n_draws_per_particle_min": int(draw_counts.min()) if not draw_counts.empty else 0,
                "n_draws_per_particle_max": int(draw_counts.max()) if not draw_counts.empty else 0,
            }
        )
        if method_id == "caster_one_layer_draw_kernel":
            metadata["ablation_id"] = "caster_one_layer_draw_kernel"
    metadata.update(validation_fields)
    write_json(metadata, out_dir / "caster_run_metadata.json")
    write_timing_log(timer.summary(seed=args.seed), out_dir / "timing.json")
    summary = summarize_model_distribution(posterior_weights)
    print(
        f"ok out={out_dir} update_rows={len(update_ledger)} readout_rows={len(readout_ledger)} "
        f"models={len(registry)} rho={rho} "
        f"top_model={dist['top_model']} model_ess={summary['model_ess']:.6f}"
    )


if __name__ == "__main__":
    main()
