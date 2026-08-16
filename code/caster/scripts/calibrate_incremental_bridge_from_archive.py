from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from caster.bridge import (
    evaluate_temperature_grid,
    fit_bridge_config,
    read_bridge_config,
    selected_rho,
    write_bridge_config,
)
from caster.filter import native_forecast_rows, validate_sleeping_model_archive
from caster.utils import RuntimeLogger, write_timing_log


def _forecast_id_sha256(frame: pd.DataFrame) -> str:
    values = sorted(frame["forecast_id"].astype(str).unique())
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _read_selection(path: str | Path) -> pd.DataFrame:
    selection = pd.read_csv(path)
    if "model_id" not in selection.columns:
        raise SystemExit("selection must contain model_id")
    if selection.empty:
        raise SystemExit("selection is empty")
    return selection.copy()


def _split_slice(ledger: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if "split" not in ledger.columns:
        raise SystemExit("ledger must contain split column")
    rows = ledger[ledger["split"].astype(str).eq(str(split_name))].copy()
    if rows.empty:
        raise SystemExit(f"ledger has no calibration rows for split {split_name!r}")
    return rows


def _parse_rho_grid(value: str) -> list[float]:
    grid = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not grid:
        raise SystemExit("--rho-grid must contain at least one numeric value")
    if any(rho <= 0 for rho in grid):
        raise SystemExit("--rho-grid values must be positive")
    return grid


def _selected_registry(registry: pd.DataFrame, model_ids: list[str]) -> pd.DataFrame:
    if "model_id" not in registry.columns:
        raise SystemExit("registry must contain model_id")
    reg = registry.copy()
    reg["model_id"] = reg["model_id"].astype(str)
    missing = [model_id for model_id in model_ids if model_id not in set(reg["model_id"])]
    if missing:
        raise SystemExit(f"selection model_id not present in registry: {missing}")
    order = {model_id: i for i, model_id in enumerate(model_ids)}
    selected = reg[reg["model_id"].isin(model_ids)].copy()
    selected["__order__"] = selected["model_id"].map(order)
    return selected.sort_values("__order__").drop(columns=["__order__"]).reset_index(drop=True)


def main() -> None:
    parser = ArgumentParser(description="Calibrate component sigma/gamma/nu bridge parameters (or load them), then optionally select global rho.")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--distribution", default="student_t", choices=["student_t", "gaussian"])
    parser.add_argument("--transform", default="log1p", choices=["log1p", "identity"])
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--rho-grid", default="")
    parser.add_argument("--target-ess-fraction", type=float, default=0.5)
    parser.add_argument("--prior-policy", default="uniform_model", choices=["uniform_model", "family_balanced"])
    parser.add_argument("--base-bridge", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--temperature-report", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    timer = RuntimeLogger()
    with timer.measure("load_inputs"):
        ledger = pd.read_csv(args.ledger)
        archive_all = pd.read_csv(args.archive)
        availability_violations = validate_sleeping_model_archive(archive_all)
        if not availability_violations.empty:
            violation_path = Path(args.out).with_name(
                "bridge_availability_violations.csv"
            )
            violation_path.parent.mkdir(parents=True, exist_ok=True)
            availability_violations.to_csv(violation_path, index=False)
            raise SystemExit(
                "incremental bridge archive violates native-availability protocol; "
                f"see {violation_path}"
            )
        registry = pd.read_csv(args.registry)
        selection = _read_selection(args.selection)
        model_ids = selection["model_id"].dropna().astype(str).tolist()
        selected_registry = _selected_registry(registry, model_ids)
        available = set(archive_all["model_id"].dropna().astype(str))
        missing = [model_id for model_id in model_ids if model_id not in available]
        if missing:
            raise SystemExit(f"selection model_id not present in archive: {missing}")
        archive_with_placeholders = archive_all[
            archive_all["model_id"].astype(str).isin(model_ids)
        ].copy()
        archive = native_forecast_rows(
            archive_with_placeholders, require_provenance=True
        )
        if archive.empty:
            raise SystemExit("selected incremental bridge archive has no native rows")
        cal_ledger = _split_slice(ledger, args.calibration_split)

    with timer.measure("bridge_fit"):
        base_metadata: dict[str, object] = {}
        if args.base_bridge:
            base_payload = json.loads(Path(args.base_bridge).read_text(encoding="utf-8"))
            raw_base_metadata = base_payload.get("calibration_metadata", {})
            base_metadata = raw_base_metadata if isinstance(raw_base_metadata, dict) else {}
            config, base_rho = read_bridge_config(args.base_bridge)
            if config.distribution != args.distribution:
                raise SystemExit(
                    f"base bridge distribution {config.distribution!r} does not match --distribution {args.distribution!r}"
                )
            if config.transform != args.transform:
                raise SystemExit(f"base bridge transform {config.transform!r} does not match --transform {args.transform!r}")
            expected_kernel = "student_t" if config.distribution == "student_t" else "gaussian"
            if config.kernel_distribution != expected_kernel:
                raise SystemExit(f"formal incremental bridge requires kernel_distribution={expected_kernel}")
            family = str(base_metadata.get("selected_bridge_family", "moment_t"))
            score_source = str(base_metadata.get("score_source", "archive_moment"))
            if (family, score_source) not in {
                ("moment_t", "archive_moment"),
                ("draw_kernel_t", "draw_kernel"),
            }:
                raise SystemExit(
                    "base bridge has inconsistent P1 family/score source: "
                    f"{family!r}/{score_source!r}"
                )
            active_scales = (
                config.sigma_by_component
                if family == "moment_t"
                else config.tau_by_component
            )
            if not active_scales:
                coordinate = "sigma" if family == "moment_t" else "tau"
                raise SystemExit(
                    "formal incremental bridge requires materialized "
                    f"component-horizon {coordinate}"
                )
            if family == "moment_t":
                if config.tau_by_component:
                    raise SystemExit("moment-t incremental bridge must leave tau inactive")
                if set(config.sigma_by_component) != set(config.gamma_by_component):
                    raise SystemExit("moment-t incremental bridge requires materialized component-horizon sigma/gamma")
            else:
                if config.sigma_by_component:
                    raise SystemExit("draw-kernel incremental bridge must leave sigma inactive")
                if config.gamma_by_component:
                    raise SystemExit("draw-kernel incremental bridge must leave gamma inactive")
            if config.distribution == "student_t" and set(config.nu_by_component) != set(active_scales):
                raise SystemExit("formal incremental Student-t bridge requires materialized component-horizon nu")
        else:
            config = fit_bridge_config(
                cal_ledger,
                archive,
                distribution=args.distribution,
                transform=args.transform,
                nu=5.0,
                min_sigma=0.04,
                default_sigma=0.20,
                gamma_grid=[0.25, 0.5, 1.0, 2.0, 4.0],
                nu_grid=[5.0, 10.0, float("inf")],
            )
            base_rho = None

    temperature_report_path = Path(args.temperature_report) if args.temperature_report else None
    rho_grid: list[float] = []
    if args.rho_grid:
        rho_grid = _parse_rho_grid(args.rho_grid)
        with timer.measure("rho_grid_selection"):
            temperature_report = evaluate_temperature_grid(
                cal_ledger,
                archive,
                selected_registry,
                config,
                grid=rho_grid,
                target_ess_fraction=float(args.target_ess_fraction),
                prior_policy=str(args.prior_policy),
            )
            rho_value = selected_rho(temperature_report)
            if temperature_report_path is not None:
                temperature_report_path.parent.mkdir(parents=True, exist_ok=True)
                temperature_report.to_csv(temperature_report_path, index=False)
    else:
        rho_value = float(args.rho if args.rho is not None else (base_rho if base_rho is not None else 1.0))

    metadata = dict(base_metadata)
    metadata.update({
        "calibration_split": str(args.calibration_split),
        "rho_selection_split": str(args.calibration_split),
        "calibration_policy": "validation_only_shared_bridge_tuned_rho" if args.rho_grid else "validation_only_shared_bridge_fixed_rho",
        "distribution": str(args.distribution),
        "gaussian_as_student_t_limit": args.distribution == "gaussian",
        "formal_student_t_nu": "infinity" if args.distribution == "gaussian" else "component_horizon_selected",
        "nu_used": args.distribution == "student_t",
        "kernel_distribution": str(config.kernel_distribution),
        "gamma_selection_policy": "reused_materialized_bridge" if args.base_bridge else "component_horizon_joint_validation_grid",
        "fixed_gamma": None,
        "gamma_selection_performed": not bool(args.base_bridge),
        "sigma_selection_performed": not bool(args.base_bridge),
        "tau_selection_performed": False,
        "nu_selection_performed": not bool(args.base_bridge) and args.distribution == "student_t",
        "negative_binomial_phi_implemented": False,
        "transform": str(args.transform),
        "rho": float(rho_value),
        "base_bridge": str(args.base_bridge) if args.base_bridge else "",
        "base_bridge_rho": None if base_rho is None else float(base_rho),
        "rho_grid": rho_grid,
        "filter_dynamics": {"kind": "bayesian_evidence_update", "scope": "model"},
        "target_ess_fraction": float(args.target_ess_fraction),
        "rho_selection_prior_policy": "uniform_model_prior" if args.prior_policy == "uniform_model" else str(args.prior_policy),
        "prior_policy": "uniform_model_prior" if args.prior_policy == "uniform_model" else str(args.prior_policy),
        "temperature_report": str(temperature_report_path) if temperature_report_path is not None else "",
        "selection_path": str(args.selection),
        "calibration_model_set": model_ids,
        "calibration_model_count": int(len(model_ids)),
        "registry_model_count": int(len(registry)),
        "ledger_rows_available": int(len(ledger)),
        "calibration_rows_used": int(len(cal_ledger)),
        "observed_calibration_rows_used": int(cal_ledger["observed_mask"].astype(bool).sum()) if "observed_mask" in cal_ledger.columns else int(len(cal_ledger)),
        "train_rows_available": int((ledger["split"].astype(str) == "train").sum()) if "split" in ledger.columns else 0,
        "embargo_rows_available": int((ledger["split"].astype(str) == "embargo").sum()) if "split" in ledger.columns else 0,
        "embargo_rows_used_for_tuning": 0,
        "embargo_rows_used_for_bridge_calibration": 0,
        "test_rows_available": int((ledger["split"].astype(str) == "test").sum()) if "split" in ledger.columns else 0,
        "test_rows_used_for_tuning": 0,
        "test_rows_used_for_bridge_calibration": 0,
        "calibration_forecast_ids_sha256": _forecast_id_sha256(cal_ledger),
        "archive_rows_available_all_models": int(len(archive_all)),
        "archive_rows_after_selection_before_native_mask": int(len(archive_with_placeholders)),
        "archive_rows_after_selection": int(len(archive)),
        "archive_structural_placeholder_rows_excluded": int(
            len(archive_with_placeholders) - len(archive)
        ),
        "bridge_calibration_availability_policy": "native_rows_only",
        "eta_selection_performed": not bool(args.base_bridge),
        "rho_selection_performed": bool(args.rho_grid),
        "parameter_selection_performed": (not bool(args.base_bridge)) or bool(args.rho_grid),
        "fixed_parameter_materialization": bool(args.base_bridge) and not bool(args.rho_grid),
        "parameters_reused_from_frozen_base": bool(args.base_bridge),
        "base_parameter_selection_protocol": base_metadata.get("parameter_selection_protocol", ""),
        "base_selected_bridge_family": base_metadata.get("selected_bridge_family", ""),
        "base_score_source": base_metadata.get("score_source", ""),
        "fixed_rho": float(rho_value) if args.base_bridge and not args.rho_grid else None,
    })
    if args.base_bridge:
        family = str(base_metadata.get("selected_bridge_family", "moment_t"))
        metadata["selected_bridge_family"] = family
        metadata["score_source"] = (
            "draw_kernel" if family == "draw_kernel_t" else "archive_moment"
        )
        metadata["sigma_selection_policy"] = "reused_frozen_base" if family == "moment_t" else "inactive_for_draw_kernel_family"
        metadata["gamma_selection_policy"] = "reused_frozen_base" if family == "moment_t" else "inactive_for_draw_kernel_family"
        metadata["tau_selection_policy"] = "reused_frozen_base" if family == "draw_kernel_t" else "inactive_for_moment_family"
    out_path = Path(args.out)
    write_bridge_config(config, out_path, rho=float(rho_value), metadata=metadata)
    timing_path = out_path.with_name(out_path.stem + "_timing.json")
    write_timing_log(timer.summary(seed=args.seed), timing_path)
    print(f"bridge_config={out_path}")
    print(f"selected_rho={float(rho_value)}")
    if temperature_report_path is not None:
        print(f"temperature_report={temperature_report_path}")
    print(f"calibration_model_count={len(model_ids)}")
    print(f"sigma_by_component={config.sigma_by_component}")
    print(f"tau_by_component={config.tau_by_component}")
    print(f"gamma_by_component={config.gamma_by_component}")
    print(f"nu_by_component={config.nu_by_component}")


if __name__ == "__main__":
    main()
