from __future__ import annotations
from argparse import ArgumentParser
from dataclasses import fields
from pathlib import Path
import hashlib
import json
import math
import sys
CASTER_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(CASTER_ROOT / "scripts"))

import pandas as pd

from caster.bridge import (
    COHERENT_CENSORED_STUDENT_T,
    COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T,
    COHERENT_MEAN_PRESERVING_TRUNCATED_T,
    alternate_ARCHIVE_MOMENT,
    PREDICTIVE_CONTRACTS,
    BridgeConfig,
    DEFAULT_GAMMA_GRID,
    DEFAULT_NU_GRID,
    FORMAL_FIXED_GAMMA,
    DEFAULT_GAMMA_PRIOR_STRENGTH,
    DEFAULT_GAMMA_SCALE_RATIO_FLOOR,
    DEFAULT_GAMMA_SCALE_RATIO_PENALTY_STRENGTH,
    DEFAULT_RHO_GRID,
    DEFAULT_RHO_FAMILY_ESS_PENALTY,
    DEFAULT_RHO_MODEL_ESS_PENALTY,
    DEFAULT_RHO_TOP1_PENALTY,
    DEFAULT_RHO_TOP1_TARGET,
    clean_validation_rho_grid,
    calibrate_component_sigma,
    calibrate_truncation_upper_bounds,
    evaluate_hierarchical_temperature_grid,
    evaluate_temperature_grid,
    fit_bridge_config,
    bridge_lookup_keys,
    selected_rho,
    write_bridge_config,
    BRIDGE_FAMILIES,
    DEFAULT_OBJECTIVE_WEIGHTS,
    DRAW_KERNEL_T,
    MOMENT_T,
    ExactValidationReplay,
    JointParameterSelector,
    JointSelectionSettings,
    JOINT_METRICS,
    RHO_ONLY_OVERALL_RMSE_METRICS,
    RHO_ONLY_OBJECTIVE_WEIGHTS,
    RhoOnlySelectionSettings,
    RhoOnlySelector,
    deterministic_small_validation_manifest,
    settings_payload,
)
from caster.data import add_task_columns, filter_ledger_archive_for_task, task_from_args, task_metadata
from caster.filter import native_forecast_rows, validate_sleeping_model_archive
from caster.models import read_registry
from caster.tasks import selection_fold_manifest_sha256
from caster.utils import RuntimeLogger, write_timing_log
from result_metric_contract import (
    filter_to_formal_horizon_grid,
    metric_slices_from_scored_rows,
    strategy_macro_values,
)


RHO_ONLY_CENSORED_CONTRACTS = {
    COHERENT_CENSORED_STUDENT_T,
    COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T,
}
DEFAULT_RHO_ONLY_FIXED_C_U = 1.25
RHO_ONLY_CENSORED_BOUND_POLICY = (
    "train_only_max_observed_or_mean_plus_4sd_times_fixed_c_u"
)
RHO_ONLY_ELIGIBLE27_CENSORED_BOUND_POLICY = (
    "eligible27_train_max_observed_or_mean_plus_4sd_times_fixed_c_u"
)
RHO_ONLY_CENSORED_NLL_MEASURE_BASIS = (
    "log1p_transform_mixed_measure_atoms_at_zero_and_upper_bound"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_float_list(text: str, *, option_name: str) -> list[float]:
    try:
        values = [float(x) for x in str(text).split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"{option_name} must be a comma-separated list of floats") from exc
    if not values:
        raise SystemExit(f"{option_name} must contain at least one value")
    return values


def _parse_validation_rho_grid(text: str, *, allow_outside_result_range: bool = False) -> list[float]:
    try:
        return clean_validation_rho_grid(
            _parse_float_list(text, option_name="--rho-grid"),
            enforce_result_range=not allow_outside_result_range,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _parse_gamma_grid(text: str) -> list[float]:
    try:
        values = [float(x) for x in str(text).split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit("--gamma-grid must be a comma-separated list of floats") from exc
    if not values:
        raise SystemExit("--gamma-grid must contain at least one value")
    if any((not math.isfinite(x)) or x < 0.0 for x in values):
        raise SystemExit("--gamma-grid values must be finite and nonnegative")
    return sorted(set(values))


def _parse_nu_grid(text: str) -> list[float]:
    aliases = {"inf": float("inf"), "infinity": float("inf"), "∞": float("inf")}
    values: list[float] = []
    try:
        for raw in str(text).split(","):
            item = raw.strip().lower()
            if item:
                values.append(aliases[item] if item in aliases else float(item))
    except ValueError as exc:
        raise SystemExit("--nu-grid must contain positive numbers or infinity") from exc
    if not values or any(math.isnan(value) or value <= 0.0 for value in values):
        raise SystemExit("--nu-grid must contain positive numbers or infinity")
    return sorted(set(values), key=lambda value: (math.isinf(value), value))


def _parse_sigma_multipliers(text: str) -> list[float]:
    try:
        values = [float(x) for x in str(text).split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit("--sigma-multipliers must be a comma-separated list of floats") from exc
    if not values:
        raise SystemExit("--sigma-multipliers must contain at least one value")
    if any((not math.isfinite(x)) or x <= 0.0 for x in values):
        raise SystemExit("--sigma-multipliers values must be finite and positive")
    return sorted(set(values))


def _rho_only_objective_weights(args) -> dict[str, float]:
    ""

    attributes = {
        "nll": "rho_objective_weight_nll",
        "wis": "rho_objective_weight_wis",
        "short_rmse": "rho_objective_weight_short_rmse",
        "long_rmse": "rho_objective_weight_long_rmse",
        "mae": "rho_objective_weight_mae",
        "coverage_penalty": "rho_objective_weight_coverage_penalty",
    }
    declared = {
        name: float(getattr(args, attribute))
        for name, attribute in attributes.items()
    }
    overall_raw = getattr(args, "rho_objective_weight_overall_rmse", None)
    if overall_raw is not None:
        declared["overall_rmse"] = float(overall_raw)
    if any(not math.isfinite(value) or value < 0.0 for value in declared.values()):
        raise SystemExit(
            "rho-only objective weights must be finite and nonnegative"
        )
    mode = str(getattr(args, "rho_objective_rmse_mode", "short-long"))
    if mode == "short-long":
        if overall_raw is not None and not math.isclose(
            float(overall_raw), 0.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise SystemExit(
                "--rho-objective-weight-overall-rmse is only active in overall mode"
            )
        weights = {name: float(declared[name]) for name in JOINT_METRICS}
    elif mode == "overall":
        overall_weight = (
            float(declared["short_rmse"] + declared["long_rmse"])
            if overall_raw is None
            else float(overall_raw)
        )
        values = {
            "nll": declared["nll"],
            "overall_rmse": overall_weight,
            "mae": declared["mae"],
            "wis": declared["wis"],
            "coverage_penalty": declared["coverage_penalty"],
        }
        weights = {
            name: float(values[name]) for name in RHO_ONLY_OVERALL_RMSE_METRICS
        }
    else:
        raise SystemExit(f"unknown rho-only RMSE objective mode: {mode!r}")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("rho-only objective weights must sum to one")
    return weights


def _nonnegative_float(value: str, option_name: str) -> float:
    try:
        x = float(value)
    except ValueError as exc:
        raise SystemExit(f"{option_name} must be a float") from exc
    if not math.isfinite(x) or x < 0.0:
        raise SystemExit(f"{option_name} must be finite and nonnegative")
    return x


def _positive_float(value: str, option_name: str) -> float:
    try:
        x = float(value)
    except ValueError as exc:
        raise SystemExit(f"{option_name} must be a float") from exc
    if not math.isfinite(x) or x <= 0.0:
        raise SystemExit(f"{option_name} must be finite and positive")
    return x


def _float_list_text(values: list[float]) -> str:
    return ",".join(f"{float(v):g}" for v in values)


def _variant_names(value: str) -> list[str]:
    if value == "both":
        return ["one_layer", "hierarchical"]
    return [value]


def _variant_config_path(base: str | Path, variant: str) -> Path:
    path = Path(base)
    suffix = path.suffix or ".json"
    return path.with_name(f"{path.stem}.{variant}{suffix}")


def _variant_report_path(base: str | Path, variant: str) -> Path:
    path = Path(base)
    suffix = path.suffix or ".csv"
    return path.with_name(f"{path.stem}.{variant}{suffix}")


def _variant_update_equation(variant: str) -> str:
    if variant == "one_layer":
        return "model_level_tempered_outer_update"
    if variant == "hierarchical":
        return "family_outer_tempered_inner_untempered_update"
    raise ValueError(f"unknown rho selection variant {variant!r}")


def _variant_metadata(
    base: dict[str, object],
    *,
    mode: str,
    variant: str,
    rho_grid: list[float] | None = None,
    fixed_rho: float | None = None,
    fixed_rho_outside_result_range: bool = False,
) -> dict[str, object]:
    meta = dict(base)
    meta["rho_selection_mode"] = mode
    meta["rho_selection_variant"] = variant
    meta["rho_selection_update_equation"] = _variant_update_equation(variant)
    meta["rho_selection_split"] = (
        "val" if mode == "validation_grid" else "not_applicable_fixed_input"
    )
    meta["rho_selection_replay"] = (
        "validation_replay" if mode == "validation_grid" else "not_simulated_fixed_input"
    )
    meta["fixed_rho_outside_result_range"] = bool(fixed_rho_outside_result_range)
    meta["filter_dynamics"] = {
        "kind": (
            "bayesian_evidence_update"
            if variant == "one_layer"
            else "hierarchical_bayesian_evidence_update"
        ),
        "scope": "model" if variant == "one_layer" else "outer_family",
    }
    if mode == "validation_grid":
        meta["rho_grid"] = _float_list_text(rho_grid or DEFAULT_RHO_GRID)
        meta["fixed_rho"] = ""
        meta["rho_run_label"] = "formal_validation_grid"
    else:
        meta["rho_grid"] = ""
        meta["fixed_rho"] = float(fixed_rho if fixed_rho is not None else 0.0)
        meta["rho_run_label"] = "fixed_input"
    return meta


def _augment_report(
    report: pd.DataFrame,
    *,
    task,
    metadata: dict[str, object],
    mode: str,
    variant: str,
) -> pd.DataFrame:
    out = report.copy()
    if "calibration_split" not in out.columns:
        out.insert(0, "calibration_split", "val")
    if "rho_selection_split" not in out.columns:
        split = "val" if mode == "validation_grid" else "not_applicable_fixed_input"
        out.insert(1, "rho_selection_split", split)
    out["rho_selection_mode"] = mode
    out["rho_selection_variant"] = variant
    out["rho_selection_update_equation"] = _variant_update_equation(variant)
    out["rho_selection_replay"] = "validation_replay" if mode == "validation_grid" else "not_simulated_fixed_input"
    if task is not None:
        out = add_task_columns(out, task)
    out["validation_rows_used"] = metadata["validation_rows_used"]
    out["observed_validation_rows_used"] = metadata["observed_validation_rows_used"]
    out["test_rows_used_for_tuning"] = 0
    return out


def _forecast_id_sha256(frame: pd.DataFrame) -> str:
    values = sorted(frame["forecast_id"].astype(str).unique())
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validation_only_slice(
    ledger: pd.DataFrame,
    *,
    selection_fold_manifest: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if "split" not in ledger.columns:
        raise SystemExit("ledger must contain split column; bridge calibration requires split == 'val'")
    split = ledger["split"].astype(str)
    val = ledger[split == "val"].copy()
    if val.empty:
        raise SystemExit("ledger has no validation rows")
    fold_path = Path(selection_fold_manifest) if selection_fold_manifest else None
    if fold_path is not None:
        if not fold_path.is_file():
            raise SystemExit(f"selection fold manifest does not exist: {fold_path}")
        folds = pd.read_csv(fold_path)
        required = {"forecast_id", "fold_id", "fold_manifest_sha256"}
        if missing := sorted(required - set(folds.columns)):
            raise SystemExit(
                f"selection fold manifest missing columns {missing}: {fold_path}"
            )
        if folds.empty:
            raise SystemExit("selection fold manifest is empty")
        declared_hashes = folds["fold_manifest_sha256"].dropna().astype(str).unique()
        if len(declared_hashes) != 1:
            raise SystemExit(
                "selection fold manifest must declare exactly one fold_manifest_sha256"
            )
        computed_hash = selection_fold_manifest_sha256(
            folds.drop(columns=["fold_manifest_sha256"])
        )
        if str(declared_hashes[0]) != computed_hash:
            raise SystemExit(
                "selection fold manifest content/hash mismatch: "
                f"declared={declared_hashes[0]} computed={computed_hash}"
            )
        fold_map = folds[["forecast_id", "fold_id"]].copy()
        fold_map["forecast_id"] = fold_map["forecast_id"].astype(str)
        if fold_map["forecast_id"].duplicated().any():
            raise SystemExit(
                "a selection-fold forecast_id may belong to only one fold"
            )
        val["forecast_id"] = val["forecast_id"].astype(str)
        manifest_ids = set(fold_map["forecast_id"])
        ledger_ids = set(val["forecast_id"])
        if missing_ids := sorted(manifest_ids - ledger_ids):
            raise SystemExit(
                "selection fold manifest contains forecast IDs absent from the "
                "validation ledger: " + ",".join(missing_ids[:10])
            )
        val = val[val["forecast_id"].isin(manifest_ids)].merge(
            fold_map, on="forecast_id", how="inner", validate="one_to_one"
        )
        if val.empty:
            raise SystemExit("selection fold projection removed every validation row")
        fold_manifest_hash = computed_hash
    else:
        fold_manifest_hash = ""
    observed_val = val[val["observed_mask"].astype(bool)].copy()
    metadata = {
        "calibration_split": "val",
        "rho_selection_split": "val",
        "ledger_rows_available": int(len(ledger)),
        "validation_rows_used": int(len(val)),
        "observed_validation_rows_used": int(len(observed_val)),
        "train_rows_available": int((split == "train").sum()),
        "embargo_rows_available": int((split == "embargo").sum()),
        "embargo_rows_used_for_tuning": 0,
        "embargo_rows_used_for_bridge_calibration": 0,
        "test_rows_available": int((split == "test").sum()),
        "test_rows_used_for_tuning": 0,
        "validation_forecast_ids_sha256": _forecast_id_sha256(val),
        "selection_fold_manifest_path": (
            str(fold_path.resolve()) if fold_path is not None else ""
        ),
        "selection_fold_manifest_sha256": fold_manifest_hash,
        "validation_fold_count": int(
            val["fold_id"].astype(str).nunique() if "fold_id" in val.columns else 1
        ),
        "validation_fold_replay_policy": (
            "independent_same_W0_per_fold"
            if "fold_id" in val.columns
            else "single_fold_same_W0"
        ),
    }
    return val, metadata


def _read_selected_model_ids(path: str | None) -> list[str]:
    if not path:
        return []
    selection = pd.read_csv(path)
    if "model_id" not in selection.columns:
        raise SystemExit("selection must contain model_id column")
    model_ids = selection["model_id"].dropna().astype(str).tolist()
    if not model_ids:
        raise SystemExit("selection contains no model_id rows")
    return model_ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_blocker(out_report: Path, mode: str, exc: Exception) -> Path:
    path = out_report.with_name("calibration_blocker_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "blocked",
                "calibration_mode": mode,
                "reason": str(exc),
                "test_based_fallback_used": False,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return path


def _read_serialized(path: Path) -> dict[str, object]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("fixed bridge template must contain an object")
    return data


def _fixed_variant_config(path: Path, task_id: str, variant: str) -> tuple[BridgeConfig, float]:
    data = _read_serialized(path)
    if "tasks" in data:
        tasks = data.get("tasks")
        if not isinstance(tasks, dict) or task_id not in tasks:
            raise ValueError(f"fixed bridge manifest has no task {task_id!r}")
        task_entry = tasks[task_id]
        if not isinstance(task_entry, dict):
            raise ValueError(f"fixed bridge task {task_id!r} is invalid")
        variants = task_entry.get("variants", task_entry)
        if not isinstance(variants, dict) or variant not in variants:
            raise ValueError(f"fixed bridge task {task_id!r} has no variant {variant!r}")
        entry = variants[variant]
        if not isinstance(entry, dict):
            raise ValueError(f"fixed bridge variant {task_id}/{variant} is invalid")
    else:
        entry = data
    rho = entry.get("rho")
    if rho is None or not math.isfinite(float(rho)) or float(rho) < 0:
        raise ValueError(f"fixed bridge variant {task_id}/{variant} requires a finite nonnegative rho")
    allowed = {field.name for field in fields(BridgeConfig)}
    config = BridgeConfig(**{key: value for key, value in entry.items() if key in allowed})
    if not config.sigma_by_component:
        raise ValueError(f"fixed bridge variant {task_id}/{variant} has no materialized sigma_by_component")
    if not config.gamma_by_component:
        raise ValueError(f"fixed bridge variant {task_id}/{variant} has no materialized gamma_by_component")
    expected_rho = 0.5 if task_id == "benchmark_a" else (0.05 if task_id.startswith("benchmark_b") else None)
    if expected_rho is not None and not math.isclose(float(rho), expected_rho, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"fix_parameter requires rho={expected_rho:g} for {task_id}, found {float(rho):g}"
        )
    if config.distribution not in {"student_t", "gaussian"}:
        raise ValueError("fix_parameter supports the materialized Student-t bridge only")
    return config, float(rho)


def _validate_fixed_materialization(config: BridgeConfig, ledger: pd.DataFrame, *, task_id: str, variant: str) -> None:
    columns = ["component", "horizon"] + (["mode"] if "mode" in ledger.columns else [])
    missing: list[str] = []
    for row in ledger[columns].drop_duplicates().itertuples(index=False):
        mode = getattr(row, "mode", "")
        component = str(getattr(row, "component"))
        horizon = int(getattr(row, "horizon"))
        keys = bridge_lookup_keys(mode, component, horizon)
        if not any(key in config.sigma_by_component for key in keys):
            missing.append(f"sigma:{component}__h{horizon}")
        if not any(key in config.gamma_by_component for key in keys):
            missing.append(f"gamma:{component}__h{horizon}")
        if config.distribution == "student_t" and not any(key in config.nu_by_component for key in keys):
            missing.append(f"nu:{component}__h{horizon}")
    if missing:
        raise ValueError(
            f"fixed bridge variant {task_id}/{variant} is not fully materialized; missing {sorted(set(missing))}"
        )


def _fixed_input_report(variant: str, rho: float) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "rho": float(rho),
            "validation_mixture_nll": float("nan"),
            "selected": True,
            "source": "fixed_input_materialized_template",
            "rho_selection_mode": "fixed_input",
            "rho_selection_variant": variant,
            "test_rows_used_for_tuning": 0,
        }]
    )


def _run_fixed_input(args, ledger: pd.DataFrame, task_id: str, timer: RuntimeLogger) -> None:
    template = Path(args.fixed_bridge_config_template)
    if not template.is_file():
        raise SystemExit("fixed_input requires an existing --fixed-bridge-config-template")
    template_hash = _sha256(template)
    outputs: dict[str, tuple[BridgeConfig, float, dict[str, object]]] = {}
    with timer.measure("fixed_input_load"):
        for variant in ("one_layer", "hierarchical"):
            config, rho = _fixed_variant_config(template, task_id, variant)
            _validate_fixed_materialization(config, ledger, task_id=task_id, variant=variant)
            metadata = {
                "calibration_mode": "fixed_input",
                "eta_selection_performed": False,
                "rho_selection_performed": False,
                "test_rows_used_for_tuning": 0,
                "embargo_rows_used_for_tuning": 0,
                "embargo_rows_used_for_bridge_calibration": 0,
                "distribution": config.distribution,
                "gaussian_as_student_t_limit": config.distribution == "gaussian",
                "formal_student_t_nu": "component_horizon_materialized",
                "nu_used": config.distribution == "student_t",
                "kernel_distribution": config.kernel_distribution,
                "transform": config.transform,
                "gamma_selection_policy": "fixed_input",
                "fixed_gamma": "component_horizon_materialized",
                "gamma_selection_performed": False,
                "nu_selection_performed": False,
                "fix_parameter": True,
                "negative_binomial_phi_implemented": False,
                "task_id": task_id,
                "parameter_source": "fixed_input_materialized_template",
                "fixed_bridge_config_template_path": str(template.resolve()),
                "fixed_bridge_config_template_sha256": template_hash,
                "rho_selection_variant": variant,
                "rho_selection_update_equation": _variant_update_equation(variant),
                "filter_dynamics": {
                    "kind": (
                        "bayesian_evidence_update"
                        if variant == "one_layer"
                        else "hierarchical_bayesian_evidence_update"
                    ),
                    "scope": "model" if variant == "one_layer" else "outer_family",
                },
                "official_split_modified": False,
            }
            config_path = _variant_config_path(args.out_config, variant)
            report_path = _variant_report_path(args.out_report, variant)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            _fixed_input_report(variant, rho).to_csv(report_path, index=False)
            write_bridge_config(config, config_path, rho=rho, metadata=metadata)
            outputs[variant] = (config, rho, metadata)
    config, rho, metadata = outputs["one_layer"]
    write_bridge_config(config, args.out_config, rho=rho, metadata=metadata)
    _fixed_input_report("one_layer", rho).to_csv(args.out_report, index=False)


def _task_id(task, ledger: pd.DataFrame) -> str:
    if task is not None:
        return str(task.task_id)
    dataset = ""
    if "dataset" in ledger.columns and not ledger.empty:
        dataset = str(ledger["dataset"].dropna().astype(str).iloc[0]).lower()
    return "benchmark_a" if "benchmark_a" in dataset or "epillm" in dataset else dataset or "task"


def _joint_metric_evaluator(
    readout: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, object], pd.DataFrame]:
    formal = filter_to_formal_horizon_grid(readout, strict=True)
    slices = metric_slices_from_scored_rows(
        formal,
        source="exact_validation_causal_replay",
        y_col="observed_value",
        pred_col="predictive_mean",
        median_col=(
            "predictive_median"
            if "predictive_median" in formal.columns
            else None
        ),
        lower_50_col="lower_50",
        upper_50_col="upper_50",
        lower_90_col="lower_90",
        upper_90_col="upper_90",
        nll_col="bridge_nll",
        method_group="caster",
    )
    short = slices[slices["horizon_group"].astype(str).eq("short")].copy()
    long = slices[slices["horizon_group"].astype(str).eq("long")].copy()
    short_values, short_validation = strategy_macro_values(short, ["rmse"])
    long_values, long_validation = strategy_macro_values(long, ["rmse"])
    all_values, all_validation = strategy_macro_values(
        slices, ["rmse", "mae", "nll", "wis", "coverage_90"]
    )
    metrics = {
        "overall_rmse": float(all_values.get("rmse", float("nan"))),
        "short_rmse": float(short_values.get("rmse", float("nan"))),
        "long_rmse": float(long_values.get("rmse", float("nan"))),
        "mae": float(all_values.get("mae", float("nan"))),
        "nll": float(all_values.get("nll", float("nan"))),
        "wis": float(all_values.get("wis", float("nan"))),
        "coverage_90": float(all_values.get("coverage_90", float("nan"))),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"joint validation metrics contain non-finite values: {metrics}")
    validation = {
        "metric_contract": "result_metric_contract_v05_short_direct_long_recursive_endpoints",
        "metric_rows": int(len(slices)),
        "short_metric_rows": int(len(short)),
        "long_metric_rows": int(len(long)),
        "overall_metric_rows": int(len(slices)),
        "short_aggregation_order": short_validation.get("aggregation_order", ""),
        "long_aggregation_order": long_validation.get("aggregation_order", ""),
        "overall_aggregation_order": all_validation.get("aggregation_order", ""),
        "all_aggregation_order": all_validation.get("aggregation_order", ""),
        "rmse_formula": short_validation.get("rmse_formula", ""),
        "validation_fold_count": int(all_validation.get("fold_count", 1)),
    }
    return metrics, validation, slices


def _family_config_path(base: str | Path, variant: str, family: str) -> Path:
    path = Path(base)
    return path.with_name(f"{path.stem}.{variant}.{family}{path.suffix}")


def _joint_artifact_path(base: str | Path, stem: str, variant: str, suffix: str) -> Path:
    path = Path(base)
    return path.with_name(f"{stem}.{variant}.{suffix}")


def _joint_metadata(
    *,
    base: dict[str, object],
    outcome,
    settings: JointSelectionSettings,
    variant: str,
    task_id: str,
) -> dict[str, object]:
    state = outcome.selected_state
    family = state.family
    metadata = dict(base)
    metadata.update(
        {
            "calibration_mode": "validation_joint_multicriterion_causal_replay",
            "parameter_selection_protocol": "frozen_joint_multicriterion_causal_replay_v1",
            "result_parameter_selection_equations": "P1-P10",
            "rho_selection_variant": variant,
            "rho_selection_update_equation": _variant_update_equation(variant),
            "filter_dynamics": {
                "kind": (
                    "bayesian_evidence_update"
                    if variant == "one_layer"
                    else "hierarchical_bayesian_evidence_update"
                ),
                "scope": "model" if variant == "one_layer" else "outer_family",
            },
            "selected_bridge_family": family,
            "bridge_family_selection_performed": True,
            "bridge_family_candidates": list(BRIDGE_FAMILIES),
            "score_source": "draw_kernel" if family == DRAW_KERNEL_T else "archive_moment",
            "distribution": "student_t",
            "kernel_distribution": "student_t",
            "transform": "log1p",
            "formal_student_t_nu": "component_horizon_outer_discrete_selected",
            "eta_parameter_scope": "component_horizon",
            "eta_selection_performed": True,
            "sigma_selection_performed": family == MOMENT_T,
            "tau_selection_performed": family == DRAW_KERNEL_T,
            "gamma_selection_performed": family == MOMENT_T,
            "nu_selection_performed": True,
            "rho_selection_performed": True,
            "sigma_selection_policy": (
                "direct_continuous_log_scale_exact_joint_risk"
                if family == MOMENT_T
                else "inactive_for_draw_kernel_family"
            ),
            "tau_selection_policy": (
                "direct_continuous_log_scale_exact_joint_risk"
                if family == DRAW_KERNEL_T
                else "inactive_for_moment_family"
            ),
            "gamma_selection_policy": (
                "direct_continuous_log_scale_exact_joint_risk"
                if family == MOMENT_T
                else "inactive_for_draw_kernel_family"
            ),
            "nu_selection_policy": "outer_discrete_exact_joint_risk",
            "rho_selection_policy": "global_continuous_log_scale_exact_joint_risk",
            "rho_selection_objective": "P1_joint_multicriterion_validation_risk",
            "objective_weights": dict(settings.objective_weights),
            "objective_metric_order": [
                "exact_asof_mixture_nll",
                "short_rmse",
                "long_rmse",
                "mae",
                "wis",
                "coverage_penalty",
            ],
            "coverage_target": float(settings.coverage_target),
            "coverage_tolerance": float(settings.coverage_tolerance),
            "coverage_upper_penalty_weight": float(settings.coverage_upper_weight),
            "validation_standardization": "fixed_reference_and_validation_only_robust_scales",
            "reference_metrics": outcome.reference_metrics,
            "reference_transformed_metrics": outcome.reference_transformed,
            "normalization_scales": outcome.normalization_scales,
            "selected_joint_risk": float(outcome.selected_objective),
            "selected_validation_metrics": outcome.selected_metrics,
            "selection_settings": settings_payload(settings),
            "online_update_uses_likelihood_only": True,
            "point_metrics_used_in_online_update": False,
            "fixed_share_used": False,
            "ess_used_as_selection_target": False,
            "top1_mass_used_as_selection_target": False,
            "target_ess_fraction": "not_applicable",
            "rho_model_ess_penalty": 0.0,
            "rho_family_ess_penalty": 0.0,
            "rho_top1_penalty": 0.0,
            "rho_top1_target": "not_applicable",
            "gamma_prior_strength": 0.0,
            "gamma_scale_ratio_penalty_strength": 0.0,
            "residual_rmse_role": "initialization_anchor_only",
            "model_specific_rho": False,
            "rho_global_within_filtering_task": True,
            "test_rows_used_for_tuning": 0,
            "embargo_rows_used_for_tuning": 0,
            "embargo_rows_used_for_bridge_calibration": 0,
            "all_choices_frozen_before_test": True,
            "negative_binomial_phi_implemented": False,
            "task_id": task_id,
        }
    )
    return metadata


def _metadata_for_family(
    metadata: dict[str, object], family: str
) -> dict[str, object]:
    ""

    out = dict(metadata)
    is_moment = family == MOMENT_T
    is_draw = family == DRAW_KERNEL_T
    if not (is_moment or is_draw):
        raise ValueError(f"unknown joint bridge family {family!r}")
    out.update(
        {
            "selected_bridge_family": family,
            "score_source": "draw_kernel" if is_draw else "archive_moment",
            "sigma_selection_performed": is_moment,
            "tau_selection_performed": is_draw,
            "gamma_selection_performed": is_moment,
            "sigma_selection_policy": (
                "direct_continuous_log_scale_exact_joint_risk"
                if is_moment
                else "inactive_for_draw_kernel_family"
            ),
            "tau_selection_policy": (
                "direct_continuous_log_scale_exact_joint_risk"
                if is_draw
                else "inactive_for_moment_family"
            ),
            "gamma_selection_policy": (
                "direct_continuous_log_scale_exact_joint_risk"
                if is_moment
                else "inactive_for_draw_kernel_family"
            ),
        }
    )
    return out


def _formal_parameter_selection_ledger(
    validation_ledger: pd.DataFrame,
    *,
    task_id: str,
) -> tuple[pd.DataFrame, int]:
    ""







    projected = validation_ledger.copy()
    projected["dataset"] = str(task_id)
    projected = filter_to_formal_horizon_grid(projected, strict=True)
    return projected, int(len(validation_ledger) - len(projected))


def _causal_fold_replay_ledger(metric_ledger: pd.DataFrame) -> pd.DataFrame:
    ""

    fold_column = next(
        (
            column
            for column in ("validation_fold", "fold_id", "fold")
            if column in metric_ledger.columns
        ),
        "",
    )
    if not fold_column:
        return metric_ledger.copy()
    base = metric_ledger.drop(columns=[fold_column]).copy()
    base["forecast_origin"] = pd.to_datetime(
        base["forecast_origin"], errors="raise"
    )
    base["release_time"] = pd.to_datetime(base["release_time"], errors="raise")
    prefixes: list[pd.DataFrame] = []
    for fold_name, metric_rows in metric_ledger.groupby(fold_column, sort=True):
        origins = pd.to_datetime(
            metric_rows["forecast_origin"], errors="raise"
        ).drop_duplicates()
        if len(origins) != 1:
            raise ValueError(
                "formal selection requires exactly one metric forecast origin "
                f"per fold; fold={fold_name!r} origins={len(origins)}"
            )
        origin = pd.Timestamp(origins.iloc[0])
        prefix = base[base["release_time"].le(origin)].copy()
        if not prefix.empty:
            prefix[fold_column] = str(fold_name)
            prefixes.append(prefix)
    if prefixes:
        return pd.concat(prefixes, ignore_index=True)
    empty = base.iloc[0:0].copy()
    empty[fold_column] = pd.Series(dtype=str)
    return empty


def _run_joint_selection(
    *,
    args,
    ledger: pd.DataFrame,
    val: pd.DataFrame,
    archive: pd.DataFrame,
    draws: pd.DataFrame,
    registry: pd.DataFrame,
    archive_models: list[str],
    metadata: dict[str, object],
    task,
    gamma_grid: list[float],
    nu_grid: list[float],
    rho_grid: list[float],
    timer: RuntimeLogger,
) -> None:
    if args.distribution != "student_t":
        raise SystemExit(
            "Joint selection enumerates moment-t and draw-kernel-t; "
            "use --distribution student_t (nu=inf is the Gaussian limit)"
        )
    if args.fixed_gamma is not None or args.fixed_nu is not None:
        raise SystemExit(
            "Joint selection does not accept --fixed-gamma/--fixed-nu; "
            "the active coordinates are selected by P1-P10"
        )
    if draws.empty:
        raise SystemExit(
            "Joint bridge-family selection requires --draws; "
            "draw-kernel candidates may not reuse moment-only surrogates"
        )
    active_registry = registry[
        registry["model_id"].astype(str).isin(archive_models)
    ].copy()
    selected_pairs = archive[["forecast_id", "model_id"]].drop_duplicates()
    draws = draws.copy()
    draws["forecast_id"] = draws["forecast_id"].astype(str)
    draws["model_id"] = draws["model_id"].astype(str)
    draws = draws.merge(
        selected_pairs, on=["forecast_id", "model_id"], how="inner"
    )
    if draws.empty:
        raise SystemExit("selected native archive has no matching forecast draws")
    task_id = _task_id(task, ledger)
    selection_ledger, excluded_intermediate_rows = (
        _formal_parameter_selection_ledger(val, task_id=task_id)
    )
    replay_ledger = _causal_fold_replay_ledger(selection_ledger)
    metadata.update(
        {
            "validation_rows_before_formal_endpoint_projection": int(len(val)),
            "validation_rows_after_formal_endpoint_projection": int(
                len(selection_ledger)
            ),
            "validation_intermediate_recursive_rows_excluded": (
                excluded_intermediate_rows
            ),
            "parameter_selection_evidence_endpoint_policy": (
                "formal_direct_and_recursive_endpoints_only"
            ),
            "validation_replay_evidence_policy": (
                "independent_same_W0_causal_released_prefix_per_metric_origin"
            ),
            "validation_replay_evidence_rows_across_folds": int(
                len(replay_ledger)
            ),
            "intermediate_recursive_steps_used_for_parameter_selection": False,
            "intermediate_recursive_steps_used_for_bridge_calibration": False,
            "intermediate_recursive_steps_used_for_posterior_evidence": False,
            "intermediate_recursive_steps_used_for_metrics": False,
        }
    )
    positive_gamma = sorted({float(value) for value in gamma_grid if float(value) > 0.0})
    if not positive_gamma:
        raise SystemExit("joint continuous gamma optimization requires a positive gamma range")
    rho_min, rho_max = min(rho_grid), max(rho_grid)
    settings = JointSelectionSettings(
        objective_weights=dict(DEFAULT_OBJECTIVE_WEIGHTS),
        nu_values=tuple(float(value) for value in nu_grid),
        rho_bounds=(float(rho_min), float(rho_max)),
        gamma_bounds=(float(min(positive_gamma)), float(max(positive_gamma))),
        scale_bound_multiplier=float(args.scale_bound_multiplier),
        min_scale=float(args.min_sigma),
        coordinate_passes=int(args.coordinate_passes),
        refinement_passes=int(args.refinement_passes),
        multi_starts=int(args.multi_starts),
        initial_log_step=float(args.initial_log_step),
        exact_tie_tolerance=float(args.exact_tie_tol),
        metric_epsilon=float(args.metric_epsilon),
        robust_scale_floor=float(args.robust_scale_floor),
        robust_relative_floor=float(args.robust_relative_floor),
        coverage_target=0.90,
        coverage_tolerance=float(args.coverage_tolerance),
        coverage_upper_weight=float(args.coverage_upper_weight),
        seed=int(args.seed),
    ).validate()
    replay = ExactValidationReplay(
        validation_ledger=replay_ledger,
        metric_ledger=selection_ledger,
        archive=archive,
        draws=draws,
        registry=active_registry,
        dataset_key=task_id,
        metric_evaluator=_joint_metric_evaluator,
    )
    selector = JointParameterSelector(replay, settings=settings, transform="log1p")
    outcomes = {}
    with timer.measure("joint_multicriterion_exact_selection"):
        for variant in _variant_names(args.rho_selection_variant):
            outcomes[variant] = selector.select(variant=variant)

    family_reports: list[pd.DataFrame] = []
    component_reports: list[pd.DataFrame] = []
    reference_payload: dict[str, object] = {
        "schema": "caster_parameter_selection_reference_v1",
        "task_id": task_id,
        "objective_weights": dict(settings.objective_weights),
        "selection_settings": settings_payload(settings),
        "test_rows_used_for_tuning": 0,
    }
    written_paths: list[Path] = []
    for variant, outcome in outcomes.items():
        variant_metadata = _joint_metadata(
            base=metadata,
            outcome=outcome,
            settings=settings,
            variant=variant,
            task_id=task_id,
        )
        config_path = _variant_config_path(args.out_config, variant)
        report_path = _variant_report_path(args.out_report, variant)
        report = outcome.trace.copy()
        report["validation_mixture_nll"] = pd.to_numeric(
            report.get("nll"), errors="coerce"
        )
        report["objective"] = -pd.to_numeric(
            report.get("joint_risk"), errors="coerce"
        )
        report["regularized_objective"] = report["objective"]
        report["rho_selection_objective"] = (
            "P1_joint_multicriterion_validation_risk"
        )
        report["rho_regularization_used_for_selection"] = False
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False)
        write_bridge_config(
            outcome.selected_config,
            config_path,
            rho=float(outcome.selected_state.rho),
            metadata=variant_metadata,
        )
        written_paths.extend([config_path, report_path])

        for family, state in outcome.family_best_states.items():
            family_metadata = _metadata_for_family(variant_metadata, family)
            family_row = outcome.family_report[
                outcome.family_report["bridge_family"].astype(str).eq(family)
            ].iloc[0]
            family_metrics = {
                name: float(family_row[name])
                for name in (
                    "nll",
                    "short_rmse",
                    "long_rmse",
                    "mae",
                    "wis",
                    "coverage_90",
                )
            }
            family_metadata.update(
                {
                    "outer_bridge_family_selected": (
                        family == outcome.selected_state.family
                    ),
                    "family_specific_frozen_config": True,
                    "selected_joint_risk": float(family_row["joint_risk"]),
                    "selected_validation_metrics": family_metrics,
                }
            )
            family_path = _family_config_path(args.out_config, variant, family)
            write_bridge_config(
                state.config(transform="log1p"),
                family_path,
                rho=float(state.rho),
                metadata=family_metadata,
            )
            written_paths.append(family_path)

        posterior_path = _joint_artifact_path(
            args.out_report, "parameter_selection_posterior_path", variant, "csv"
        )
        slice_path = _joint_artifact_path(
            args.out_report, "parameter_selection_metric_slices", variant, "csv"
        )
        readout_path = _joint_artifact_path(
            args.out_report, "parameter_selection_scored_readout", variant, "csv"
        )
        outcome.replay_artifacts.posterior_path.to_csv(posterior_path, index=False)
        outcome.replay_artifacts.metric_slices.to_csv(slice_path, index=False)
        outcome.replay_artifacts.scored_readout.to_csv(readout_path, index=False)
        written_paths.extend([posterior_path, slice_path, readout_path])
        family_reports.append(outcome.family_report)
        component_reports.append(outcome.component_report)
        reference_payload[variant] = {
            "reference_metrics": outcome.reference_metrics,
            "reference_transformed_metrics": outcome.reference_transformed,
            "normalization_scales": outcome.normalization_scales,
            "selected_bridge_family": outcome.selected_state.family,
            "selected_joint_risk": float(outcome.selected_objective),
            "selected_parameters": outcome.selected_state.serializable(),
        }

    if "one_layer" not in outcomes:
        raise SystemExit("formal joint selection must produce a one_layer configuration")
    one = outcomes["one_layer"]
    one_metadata = _joint_metadata(
        base=metadata,
        outcome=one,
        settings=settings,
        variant="one_layer",
        task_id=task_id,
    )
    write_bridge_config(
        one.selected_config,
        args.out_config,
        rho=float(one.selected_state.rho),
        metadata=one_metadata,
    )
    one_report = _variant_report_path(args.out_report, "one_layer")
    pd.read_csv(one_report).to_csv(args.out_report, index=False)
    written_paths.extend([Path(args.out_config), Path(args.out_report)])

    component_report_path = Path(args.out_report).with_name(
        "bridge_component_calibration_report.csv"
    )
    family_report_path = Path(args.out_report).with_name(
        "bridge_family_selection_report.csv"
    )
    trace_path = Path(args.out_report).with_name("parameter_selection_trace.csv")
    reference_path = Path(args.out_report).with_name(
        "parameter_selection_reference.json"
    )
    freeze_path = Path(args.out_report).with_name(
        "parameter_selection_freeze_manifest.json"
    )
    pd.concat(component_reports, ignore_index=True).to_csv(
        component_report_path, index=False
    )
    pd.concat(family_reports, ignore_index=True).to_csv(
        family_report_path, index=False
    )
    pd.concat([outcome.trace for outcome in outcomes.values()], ignore_index=True).to_csv(
        trace_path, index=False
    )
    reference_path.write_text(
        json.dumps(reference_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    written_paths.extend(
        [component_report_path, family_report_path, trace_path, reference_path]
    )
    freeze_manifest = {
        "schema": "caster_parameter_selection_freeze_manifest_v1",
        "task_id": task_id,
        "parameter_selection_protocol": "frozen_joint_multicriterion_causal_replay_v1",
        "ledger_path": str(Path(args.ledger).resolve()),
        "ledger_sha256": _sha256_file(Path(args.ledger)),
        "archive_path": str(Path(args.archive).resolve()),
        "archive_sha256": _sha256_file(Path(args.archive)),
        "draws_path": str(Path(args.draws).resolve()),
        "draws_sha256": _sha256_file(Path(args.draws)),
        "registry_path": str(Path(args.registry).resolve()),
        "registry_sha256": _sha256_file(Path(args.registry)),
        "selection_path": str(Path(args.selection).resolve()) if args.selection else "",
        "selection_sha256": _sha256_file(Path(args.selection)) if args.selection else "",
        "selection_fold_manifest_path": (
            str(Path(args.selection_fold_manifest).resolve())
            if args.selection_fold_manifest
            else ""
        ),
        "selection_fold_manifest_file_sha256": (
            _sha256_file(Path(args.selection_fold_manifest))
            if args.selection_fold_manifest
            else ""
        ),
        "selection_fold_manifest_sha256": str(
            metadata.get("selection_fold_manifest_sha256", "")
        ),
        "validation_fold_count": int(metadata.get("validation_fold_count", 1)),
        "validation_fold_replay_policy": str(
            metadata.get("validation_fold_replay_policy", "single_fold_same_W0")
        ),
        "objective_weights": dict(settings.objective_weights),
        "settings": settings_payload(settings),
        "validation_rows": int(len(selection_ledger)),
        "validation_replay_evidence_rows_across_folds": int(
            len(replay_ledger)
        ),
        "validation_replay_evidence_policy": (
            "independent_same_W0_causal_released_prefix_per_metric_origin"
        ),
        "validation_rows_before_formal_endpoint_projection": int(len(val)),
        "formal_validation_metric_rows": int(len(selection_ledger)),
        "validation_intermediate_recursive_rows_excluded": int(
            excluded_intermediate_rows
        ),
        "parameter_selection_evidence_endpoint_policy": (
            "formal_direct_and_recursive_endpoints_only"
        ),
        "embargo_rows_used_for_tuning": 0,
        "test_rows_used_for_tuning": 0,
        "all_choices_frozen_before_test": True,
        "selected": {
            variant: outcome.selected_state.serializable()
            for variant, outcome in outcomes.items()
        },
        "artifacts": {
            str(path.name): {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
            }
            for path in written_paths
            if path.is_file()
        },
    }
    freeze_path.write_text(
        json.dumps(freeze_manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(f"bridge_config={args.out_config}")
    print(f"report={args.out_report}")
    print(f"component_report={component_report_path}")
    print(f"family_report={family_report_path}")
    print(f"selection_trace={trace_path}")
    print(f"freeze_manifest={freeze_path}")
    print(f"selected_rho={one.selected_state.rho}")
    print(f"selected_bridge_family={one.selected_state.family}")


def _fixed_rho_only_family(task_id: str, requested: str) -> str:
    if requested == "moment_t":
        return MOMENT_T
    if requested == "draw_kernel_t":
        return DRAW_KERNEL_T
    if requested != "auto":
        raise ValueError(f"unknown fixed rho-only family {requested!r}")
    if task_id in {
        "benchmark_a",
        "benchmark_b_covid",
        "benchmark_b_flu",
        "benchmark_b_pooled",
    }:
        return MOMENT_T
    raise ValueError(
        f"fixed rho-only family auto mapping is undefined for task {task_id!r}"
    )


def _rho_only_anchors_within_bounds(
    rho_min: float,
    rho_max: float,
) -> tuple[tuple[float, ...], float]:
    ""

    lower, upper = float(rho_min), float(rho_max)
    base = RhoOnlySelectionSettings().anchor_rhos
    anchors = tuple(
        sorted(
            {
                lower,
                upper,
                *(float(value) for value in base if lower <= float(value) <= upper),
            }
        )
    )
    reference = float(min(max(0.50, lower), upper))
    return anchors, reference


def _load_censoring_support_manifest(
    path_text: str,
    *,
    task_id: str,
) -> tuple[dict[str, float], dict[str, object]]:
    path = Path(str(path_text))
    if not path.is_file():
        raise SystemExit(f"censoring support manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_upper = payload["base_upper_raw_by_component"]
        upper = {
            str(key): float(value)
            for key, value in raw_upper.items()
        }
        eligible_model_ids = [
            str(value) for value in payload["eligible_model_ids"]
        ]
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(
            f"invalid censoring support manifest: {path}"
        ) from exc
    sha_fields = (
        "ledger_sha256",
        "archive_sha256",
        "eligibility_sha256",
        "candidate_manifest_sha256",
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema")
        != "caster_censoring_support_manifest_v1"
        or str(payload.get("task_id", "")) != str(task_id)
        or str(payload.get("bound_scope", "")) != "eligible27_train"
        or str(payload.get("source_split", "")) != "train"
        or int(payload.get("eligible_model_count", -1)) != 27
        or len(eligible_model_ids) != 27
        or len(set(eligible_model_ids)) != 27
        or int(payload.get("archive_train_rows_used", -1))
        != int(payload.get("expected_archive_train_rows", -2))
        or int(payload.get("validation_rows_used", -1)) != 0
        or int(payload.get("test_rows_used", -1)) != 0
        or int(payload.get("test_targets_used", -1)) != 0
        or int(payload.get("selection_outcomes_used", -1)) != 0
        or not math.isclose(
            float(payload.get("predictive_standard_deviations", math.nan)),
            4.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(payload.get("minimum_upper_raw", math.nan)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not upper
        or any(
            not math.isfinite(value) or value <= 0.0
            for value in upper.values()
        )
        or any(
            len(str(payload.get(field, ""))) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(payload.get(field, ""))
            )
            for field in sha_fields
        )
    ):
        raise SystemExit(
            "censoring support manifest violates the eligible27 train-only "
            f"contract: {path}"
        )
    metadata = {
        "censoring_bound_scope": "eligible27_train",
        "censoring_support_manifest_path": str(path.resolve()),
        "censoring_support_manifest_sha256": _sha256_file(path),
        "censoring_support_source_split": "train",
        "censoring_support_eligible_model_count": 27,
        "censoring_support_validation_rows_used": 0,
        "censoring_support_test_rows_used": 0,
        "censoring_support_test_targets_used": 0,
        "censoring_support_archive_sha256": str(
            payload["archive_sha256"]
        ),
        "censoring_support_eligibility_sha256": str(
            payload["eligibility_sha256"]
        ),
    }
    return upper, metadata


def _rho_only_bounded_support(
    *,
    predictive_contract: str,
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    fixed_c_u: float = DEFAULT_RHO_ONLY_FIXED_C_U,
    base_upper_override: dict[str, float] | None = None,
) -> tuple[
    dict[str, float],
    dict[str, float],
    str,
    float | None,
    float | None,
]:
    ""






    if predictive_contract == COHERENT_MEAN_PRESERVING_TRUNCATED_T:
        upper = calibrate_truncation_upper_bounds(
            ledger,
            archive,
            source_split="train",
            predictive_standard_deviations=4.0,
            safety_multiplier=1.25,
            minimum_upper_raw=1.0,
        )
        return (
            upper,
            {},
            "train_only_max_observed_or_mean_plus_4sd_times_1p25",
            None,
            1.25,
        )
    if predictive_contract not in RHO_ONLY_CENSORED_CONTRACTS:
        return {}, {}, "none", None, 1.25

    c_u = float(fixed_c_u)
    if not math.isfinite(c_u) or c_u < 1.0:
        raise SystemExit(
            "fixed censored rho-only support requires finite --fixed-c-u >= 1"
        )
    selected_topk_base_upper = calibrate_truncation_upper_bounds(
        ledger,
        archive,
        source_split="train",
        predictive_standard_deviations=4.0,
        safety_multiplier=1.0,
        minimum_upper_raw=1.0,
    )
    if base_upper_override is None:
        base_upper = selected_topk_base_upper
        bound_policy = RHO_ONLY_CENSORED_BOUND_POLICY
    else:
        base_upper = {
            str(key): float(value)
            for key, value in sorted(base_upper_override.items())
        }
        missing = sorted(set(selected_topk_base_upper) - set(base_upper))
        if missing:
            raise SystemExit(
                "eligible27 censoring support is missing active Top-10 "
                f"component/horizon keys: {missing}"
            )
        bound_policy = RHO_ONLY_ELIGIBLE27_CENSORED_BOUND_POLICY
    upper = {
        str(key): float(value) * c_u
        for key, value in sorted(base_upper.items())
    }
    values = list(upper.values())
    if (
        not values
        or any(not math.isfinite(value) or value <= 0.0 for value in values)
    ):
        raise SystemExit(
            "fixed censored rho-only support calibration produced invalid upper bounds"
        )
    return (
        upper,
        {str(key): float(value) for key, value in sorted(base_upper.items())},
        bound_policy,
        c_u,
        None,
    )


def _rho_only_family_metadata(
    base: dict[str, object],
    *,
    task_id: str,
    variant: str,
    family: str,
    outcome,
    settings: RhoOnlySelectionSettings,
    smallval_manifest_path: Path,
    smallval_manifest_sha256: str,
    small_validation_seed: int = 42,
    sigma_by_component: dict[str, float],
    fixed_gamma: float,
    fixed_nu: float | None = None,
    distribution: str = "student_t",
    predictive_contract: str = alternate_ARCHIVE_MOMENT,
    truncation_upper_raw_by_component: dict[str, float] | None = None,
    base_truncation_upper_raw_by_component: dict[str, float] | None = None,
    truncation_bound_policy: str | None = None,
    fixed_c_u: float | None = None,
    truncation_support_expansion_multiplier: float | None = 1.25,
) -> dict[str, object]:
    is_moment = family == MOMENT_T
    is_censored = predictive_contract in RHO_ONLY_CENSORED_CONTRACTS
    score_source = "archive_moment" if is_moment else "draw_kernel"
    active_gamma = float(fixed_gamma) if is_moment else None
    active_nu: float | None = None
    if distribution == "student_t":
        active_nu = 5.0 if fixed_nu is None else float(fixed_nu)
    nu_label = (
        f"fixed_{active_nu:g}"
        if active_nu is not None
        else "not_applicable_gaussian"
    )
    active_bound_policy = (
        str(truncation_bound_policy)
        if truncation_bound_policy is not None
        else (
            "train_only_max_observed_or_mean_plus_4sd_times_1p25"
            if predictive_contract == COHERENT_MEAN_PRESERVING_TRUNCATED_T
            else "none"
        )
    )
    if is_censored:
        active_c_u = (
            DEFAULT_RHO_ONLY_FIXED_C_U
            if fixed_c_u is None
            else float(fixed_c_u)
        )
        point_readout_source = (
            "censored_latent_student_t_posterior_mixture_expectation"
        )
        interval_readout_source = (
            f"censored_{score_source}_posterior_mixture_quantiles"
        )
        censored_metadata: dict[str, object] = {
            "base_truncation_upper_raw_by_component": dict(
                base_truncation_upper_raw_by_component or {}
            ),
            "fixed_c_u": active_c_u,
            "c_u_selection_performed": False,
            "truncation_support_expansion_multiplier": (
                truncation_support_expansion_multiplier
            ),
            "nll_measure_basis": RHO_ONLY_CENSORED_NLL_MEASURE_BASIS,
            "posterior_predictive_measure": (
                "continuous interior plus censoring atoms at zero and "
                "the frozen upper bound"
            ),
        }
    else:
        point_readout_source = (
            "coherent_mean_constrained_truncated_bridge_mixture"
            if predictive_contract == COHERENT_MEAN_PRESERVING_TRUNCATED_T
            else "posterior_weighted_archived_raw_means"
        )
        interval_readout_source = (
            "archived_raw_total_variance_gaussian"
            if predictive_contract == alternate_ARCHIVE_MOMENT
            else (
                "mean_constrained_truncated_bridge_mixture_quantiles"
                if predictive_contract
                == COHERENT_MEAN_PRESERVING_TRUNCATED_T
                else "untruncated_bridge_mixture_quantiles"
            )
        )
        censored_metadata = {}
    return {
        **base,
        "calibration_mode": "fixed_bridge_rho_only_small_validation",
        "parameter_selection_protocol": "fixed_formula_rho_only_smallval_newton_v1",
        "rho_selection_variant": variant,
        "rho_selection_update_equation": _variant_update_equation(variant),
        "rho_selection_performed": True,
        "rho_selection_policy": "safeguarded_bounded_newton_log_rho",
        "rho_selection_objective": (
            "five_metric_overall_rmse_validation_risk_fixed_normalization"
            if "overall_rmse" in settings.objective_weights
            else "six_metric_short_long_rmse_validation_risk_fixed_normalization"
        ),
        "rho_objective_rmse_mode": (
            "overall"
            if "overall_rmse" in settings.objective_weights
            else "short-long"
        ),
        "rho_bounds": [float(value) for value in settings.rho_bounds],
        "normalization_anchor_rhos": [
            float(value) for value in settings.anchor_rhos
        ],
        "reference_rho": float(settings.reference_rho),
        "objective_weights": dict(settings.objective_weights),
        "objective_metric_order": list(settings.objective_metric_order),
        "coverage_target": float(settings.coverage_target),
        "coverage_tolerance": float(settings.coverage_tolerance),
        "coverage_upper_penalty_weight": float(settings.coverage_upper_weight),
        "optimizer": {
            "name": "safeguarded_bounded_newton_log_rho",
            "derivatives": "central_finite_difference",
            "projection": "closed_rho_bounds",
            "global_safeguards": "multi_start_backtracking_largest_gap_visited_best",
            "multi_starts": int(settings.multi_starts),
            "finite_difference_log_step": float(settings.finite_difference_log_step),
            "maximum_newton_log_step": float(settings.maximum_newton_log_step),
            "max_iterations": int(settings.max_iterations),
            "max_evaluations": int(settings.max_evaluations),
            "max_backtracks": int(settings.max_backtracks),
            "hessian_floor": float(settings.hessian_floor),
            "evaluation_count": int(outcome.optimizer.evaluation_count),
            "fallback_count": int(outcome.optimizer.fallback_count),
        },
        "selected_joint_risk": float(outcome.selected_objective),
        "selected_validation_metrics": dict(outcome.selected_metrics),
        "reference_metrics": dict(outcome.reference_metrics),
        "reference_transformed_metrics": dict(outcome.reference_transformed),
        "normalization_scales": dict(outcome.normalization_scales),
        "validation_standardization": "anchors_frozen_before_newton_robust_MAD_with_floors",
        "selected_bridge_family": family,
        "bridge_family_selection_performed": False,
        "bridge_family_source": f"preregistered_fixed_{family}_pilot",
        "score_source": score_source,
        "distribution": distribution,
        "kernel_distribution": distribution,
        "predictive_contract": predictive_contract,
        "point_readout_source": point_readout_source,
        "interval_readout_source": interval_readout_source,
        "truncation_upper_raw_by_component": dict(
            truncation_upper_raw_by_component or {}
        ),
        "truncation_bound_policy": active_bound_policy,
        **censored_metadata,
        "gaussian_as_student_t_limit": False,
        "explicit_gaussian_likelihood": distribution == "gaussian",
        "nu": active_nu,
        "nu_grid": (
            _float_list_text([active_nu]) if active_nu is not None else ""
        ),
        "nu_used": distribution == "student_t",
        "transform": "log1p",
        "formal_student_t_nu": (
            nu_label
        ),
        "nu_selection_performed": False,
        "nu_parameter_active": distribution == "student_t",
        "nu_parameter_status": (
            nu_label if distribution == "student_t" else "inactive_gaussian"
        ),
        "fixed_nu": active_nu,
        "gamma_selection_performed": False,
        "gamma_parameter_status": (
            "fixed_input" if is_moment else "inactive_for_draw_kernel_family"
        ),
        "gamma_parameter_active": bool(is_moment),
        "fixed_gamma": active_gamma,
        "default_gamma": active_gamma,
        "sigma_calculation_performed": True,
        "sigma_selection_performed": False,
        "sigma_formula": "alternate_log1p_transform_residual_rmse",
        "sigma_formula_gamma_dependence": "none",
        "sigma_formula_equivalent_fixed_input_name": "residual_rmse_sqrt_gamma",
        "sigma_grouping": "component_horizon",
        "sigma_nonfinite_policy": "drop_nonfinite_then_default_if_empty",
        "sigma_parameter_count": int(len(sigma_by_component)),
        "tau_calculation_performed": True,
        "tau_selection_performed": False,
        "tau_formula": "tau_equals_computed_sigma",
        "tau_equals_computed_sigma": True,
        "eta_selection_performed": False,
        "bridge_parameter_scope": "component_horizon",
        "fixed_parameter_coordinates": [
            "family",
            "distribution",
            "sigma",
            "tau",
            *(["gamma"] if is_moment else []),
            *(["nu"] if distribution == "student_t" else []),
            *(["c_u"] if is_censored else []),
        ],
        "only_optimized_coordinate": "rho",
        "small_validation_manifest_path": str(smallval_manifest_path.resolve()),
        "small_validation_manifest_sha256": smallval_manifest_sha256,
        "small_validation_seed": int(small_validation_seed),
        "distinct_validation_endpoint_rows": 360,
        "validation_fold_count": 10,
        "validation_fold_replay_policy": "independent_same_W0_per_fold",
        "validation_sampling_uses_observed_values": False,
        "validation_sampling_policy": "10_evenly_spaced_folds_metadata_hash_entity_blocks_v1",
        "validation_replay_evidence_policy": "independent_same_W0_causal_released_prefix_per_metric_origin",
        "online_update_uses_likelihood_only": True,
        "point_metrics_used_in_online_update": False,
        "fixed_share_used": False,
        "model_specific_rho": False,
        "rho_global_within_filtering_task": True,
        "test_rows_used_for_tuning": 0,
        "embargo_rows_used_for_tuning": 0,
        "embargo_rows_used_for_bridge_calibration": 0,
        "all_choices_frozen_before_test": True,
        "task_id": task_id,
    }


def _run_fixed_bridge_rho_only(
    *,
    args,
    ledger: pd.DataFrame,
    val: pd.DataFrame,
    archive: pd.DataFrame,
    draws: pd.DataFrame,
    registry: pd.DataFrame,
    archive_models: list[str],
    metadata: dict[str, object],
    task,
    rho_grid: list[float],
    timer: RuntimeLogger,
) -> None:
    task_id = _task_id(task, ledger)
    predictive_contract = str(args.predictive_contract)
    if predictive_contract not in PREDICTIVE_CONTRACTS:
        raise SystemExit(f"unknown predictive contract {predictive_contract!r}")
    if args.distribution not in {"student_t", "gaussian"} or args.transform != "log1p":
        raise SystemExit(
            "fixed bridge rho-only requires Student-t or Gaussian with log1p transform"
        )
    fixed_gamma = 1.0 if args.fixed_gamma is None else float(args.fixed_gamma)
    if not math.isfinite(fixed_gamma) or fixed_gamma <= 0.0:
        raise SystemExit("fixed bridge rho-only requires a finite positive --fixed-gamma")
    if args.distribution == "student_t":
        fixed_nu = 5.0 if args.fixed_nu is None else float(args.fixed_nu)
        if math.isnan(fixed_nu) or fixed_nu <= 0.0:
            raise SystemExit(
                "fixed Student-t bridge rho-only requires positive --fixed-nu"
            )
    elif args.fixed_nu is not None and not math.isinf(float(args.fixed_nu)):
        raise SystemExit(
            "fixed Gaussian bridge rho-only does not use finite --fixed-nu"
        )
    else:
        fixed_nu = None
    family = _fixed_rho_only_family(task_id, args.fixed_bridge_family)
    if family == DRAW_KERNEL_T and draws.empty:
        raise SystemExit("fixed draw-kernel rho-only selection requires --draws")
    bounded_contracts = {
        COHERENT_MEAN_PRESERVING_TRUNCATED_T,
        *RHO_ONLY_CENSORED_CONTRACTS,
    }
    if predictive_contract in bounded_contracts:
        if args.distribution != "student_t":
            raise SystemExit(
                "bounded coherent rho-only contracts require Student-t"
            )
    censoring_support_metadata: dict[str, object] = {}
    base_upper_override: dict[str, float] | None = None
    if str(getattr(args, "censoring_support_manifest", "")).strip():
        if predictive_contract not in RHO_ONLY_CENSORED_CONTRACTS:
            raise SystemExit(
                "--censoring-support-manifest is only valid for censored "
                "Student-t predictive contracts"
            )
        (
            base_upper_override,
            censoring_support_metadata,
        ) = _load_censoring_support_manifest(
            str(args.censoring_support_manifest),
            task_id=task_id,
        )
    elif predictive_contract in RHO_ONLY_CENSORED_CONTRACTS:
        censoring_support_metadata = {
            "censoring_bound_scope": "selected_topk_train",
            "censoring_support_manifest_path": "",
            "censoring_support_manifest_sha256": "",
            "censoring_support_source_split": "train",
            "censoring_support_eligible_model_count": len(archive_models),
            "censoring_support_validation_rows_used": 0,
            "censoring_support_test_rows_used": 0,
            "censoring_support_test_targets_used": 0,
        }
    (
        truncation_upper_raw_by_component,
        base_truncation_upper_raw_by_component,
        truncation_bound_policy,
        fixed_c_u,
        truncation_support_expansion_multiplier,
    ) = _rho_only_bounded_support(
        predictive_contract=predictive_contract,
        ledger=ledger,
        archive=archive,
        fixed_c_u=(
            DEFAULT_RHO_ONLY_FIXED_C_U
            if getattr(args, "fixed_c_u", None) is None
            else float(args.fixed_c_u)
        ),
        base_upper_override=base_upper_override,
    )

    formal, excluded_intermediate_rows = _formal_parameter_selection_ledger(
        val, task_id=task_id
    )
    smallval = deterministic_small_validation_manifest(
        formal,
        task_id=task_id,
        seed=int(args.small_validation_seed),
        fold_count=int(args.small_validation_folds),
    )
    smallval_path = Path(args.out_report).with_name("small_validation_manifest.csv")
    smallval_path.parent.mkdir(parents=True, exist_ok=True)
    smallval.to_csv(smallval_path, index=False)
    smallval_hash = _sha256_file(smallval_path)
    replay_ledger = _causal_fold_replay_ledger(smallval)
    small_ids = set(smallval["forecast_id"].astype(str))
    small_archive = archive[archive["forecast_id"].astype(str).isin(small_ids)].copy()
    small_draws = (
        draws[draws["forecast_id"].astype(str).isin(small_ids)].copy()
        if not draws.empty
        else draws.copy()
    )
    sigma_by_component = calibrate_component_sigma(
        smallval,
        small_archive,
        transform="log1p",
        min_sigma=float(args.min_sigma),
    )
    if not sigma_by_component:
        raise SystemExit("fixed bridge rho-only could not compute residual-RMSE sigma")

    active_registry = registry[
        registry["model_id"].astype(str).isin(archive_models)
    ].copy()
    rho_min, rho_max = min(rho_grid), max(rho_grid)
    anchor_rhos, reference_rho = _rho_only_anchors_within_bounds(
        rho_min, rho_max
    )
    settings = RhoOnlySelectionSettings(
        objective_weights=_rho_only_objective_weights(args),
        rho_bounds=(float(rho_min), float(rho_max)),
        anchor_rhos=anchor_rhos,
        reference_rho=reference_rho,
        multi_starts=int(args.rho_newton_multi_starts),
        finite_difference_log_step=float(args.rho_newton_fd_log_step),
        maximum_newton_log_step=float(args.rho_newton_max_log_step),
        max_iterations=int(args.rho_newton_max_iterations),
        max_evaluations=int(args.rho_newton_max_evaluations),
        max_backtracks=int(args.rho_newton_max_backtracks),
        hessian_floor=float(args.rho_newton_hessian_floor),
    ).validate()
    replay = ExactValidationReplay(
        validation_ledger=replay_ledger,
        metric_ledger=smallval,
        archive=small_archive,
        draws=small_draws,
        registry=active_registry,
        dataset_key=task_id,
        metric_evaluator=_joint_metric_evaluator,
    )
    selector = RhoOnlySelector(
        replay,
        family=family,
        scales=sigma_by_component,
        settings=settings,
        fixed_gamma=fixed_gamma,
        fixed_nu=fixed_nu,
        distribution=args.distribution,
        predictive_contract=predictive_contract,
        truncation_upper_raw_by_component=truncation_upper_raw_by_component,
        default_truncation_upper_raw=(
            max(truncation_upper_raw_by_component.values())
            if truncation_upper_raw_by_component
            else float("inf")
        ),
        truncation_bound_policy=truncation_bound_policy,
        truncation_quadrature_order=int(args.truncation_quadrature_order),
        truncation_zero_mean_epsilon=float(args.truncation_zero_mean_epsilon),
        truncation_support_expansion_multiplier=(
            truncation_support_expansion_multiplier
        ),
    )
    outcomes = {}
    with timer.measure("fixed_bridge_rho_only_smallval_newton"):
        for variant in _variant_names(args.rho_selection_variant):
            outcomes[variant] = selector.select(variant=variant)

    written_paths: list[Path] = [smallval_path]
    references: dict[str, object] = {
        "schema": "caster_rho_only_smallval_reference_v1",
        "task_id": task_id,
        "fixed_family": family,
        "distribution": args.distribution,
        "fixed_nu": fixed_nu,
        "fixed_gamma": fixed_gamma if family == MOMENT_T else None,
        "predictive_contract": predictive_contract,
        "truncation_upper_raw_by_component": truncation_upper_raw_by_component,
        "truncation_bound_policy": truncation_bound_policy,
        "gamma_parameter_status": (
            "fixed_input"
            if family == MOMENT_T
            else "inactive_for_draw_kernel_family"
        ),
        "objective_weights": dict(settings.objective_weights),
        "rho_objective_rmse_mode": (
            "overall"
            if "overall_rmse" in settings.objective_weights
            else "short-long"
        ),
    }
    if predictive_contract in RHO_ONLY_CENSORED_CONTRACTS:
        references.update(
            {
                "base_truncation_upper_raw_by_component": (
                    base_truncation_upper_raw_by_component
                ),
                "fixed_c_u": fixed_c_u,
                "truncation_support_expansion_multiplier": (
                    truncation_support_expansion_multiplier
                ),
                **censoring_support_metadata,
            }
        )
    component_rows: list[dict[str, object]] = []
    for variant, outcome in outcomes.items():
        variant_meta = _rho_only_family_metadata(
            metadata,
            task_id=task_id,
            variant=variant,
            family=family,
            distribution=args.distribution,
            outcome=outcome,
            settings=settings,
            smallval_manifest_path=smallval_path,
            smallval_manifest_sha256=smallval_hash,
            small_validation_seed=int(args.small_validation_seed),
            sigma_by_component=sigma_by_component,
            fixed_gamma=fixed_gamma,
            fixed_nu=fixed_nu,
            predictive_contract=predictive_contract,
            truncation_upper_raw_by_component=truncation_upper_raw_by_component,
            base_truncation_upper_raw_by_component=(
                base_truncation_upper_raw_by_component
            ),
            truncation_bound_policy=truncation_bound_policy,
            fixed_c_u=fixed_c_u,
            truncation_support_expansion_multiplier=(
                truncation_support_expansion_multiplier
            ),
        )
        variant_meta.update(
            {
                "validation_rows_used": int(len(smallval)),
                "observed_validation_rows_used": int(
                    smallval["observed_mask"].astype(bool).sum()
                ),
                "validation_forecast_ids_sha256": _forecast_id_sha256(smallval),
                "validation_rows_before_formal_endpoint_projection": int(len(val)),
                "validation_rows_after_formal_endpoint_projection": int(len(formal)),
                "validation_intermediate_recursive_rows_excluded": int(excluded_intermediate_rows),
                "validation_replay_evidence_rows_across_folds": int(len(replay_ledger)),
                "archive_model_event_rows_in_small_validation": int(len(small_archive)),
                "archive_draw_rows_in_small_validation": int(len(small_draws)),
                "selected_rho": float(outcome.selected_state.rho),
                **censoring_support_metadata,
            }
        )
        selected_path = _variant_config_path(args.out_config, variant)
        report_path = _variant_report_path(args.out_report, variant)
        outcome.trace.to_csv(report_path, index=False)
        write_bridge_config(
            outcome.selected_state.config(transform="log1p"),
            selected_path,
            rho=float(outcome.selected_state.rho),
            metadata=variant_meta,
        )
        written_paths.extend([selected_path, report_path])

        family_meta = dict(variant_meta)
        family_meta.update(
            {
                "family_specific_fixed_formula_config": True,
                "outer_bridge_family_selected": True,
            }
        )
        family_path = _family_config_path(args.out_config, variant, family)
        write_bridge_config(
            outcome.selected_state.config(transform="log1p"),
            family_path,
            rho=float(outcome.selected_state.rho),
            metadata=family_meta,
        )
        written_paths.append(family_path)

        posterior_path = _joint_artifact_path(
            args.out_report, "parameter_selection_posterior_path", variant, "csv"
        )
        slices_path = _joint_artifact_path(
            args.out_report, "parameter_selection_metric_slices", variant, "csv"
        )
        readout_path = _joint_artifact_path(
            args.out_report, "parameter_selection_scored_readout", variant, "csv"
        )
        outcome.replay_artifacts.posterior_path.to_csv(posterior_path, index=False)
        outcome.replay_artifacts.metric_slices.to_csv(slices_path, index=False)
        outcome.replay_artifacts.scored_readout.to_csv(readout_path, index=False)
        written_paths.extend([posterior_path, slices_path, readout_path])
        references[variant] = {
            "reference_metrics": outcome.reference_metrics,
            "reference_transformed_metrics": outcome.reference_transformed,
            "normalization_scales": outcome.normalization_scales,
            "selected_joint_risk": float(outcome.selected_objective),
            "selected_parameters": outcome.selected_state.serializable(),
        }
        for key, sigma in sigma_by_component.items():
            component_rows.append(
                {
                    "rho_selection_variant": variant,
                    "bridge_r_key": key,
                    "computed_sigma": float(sigma),
                    "fixed_tau": float(sigma),
                    "fixed_gamma": fixed_gamma if family == MOMENT_T else None,
                    "gamma_parameter_status": (
                        "fixed_input"
                        if family == MOMENT_T
                        else "inactive_for_draw_kernel_family"
                    ),
                    "fixed_nu": (
                        fixed_nu
                    ),
                    "distribution": args.distribution,
                    "predictive_contract": predictive_contract,
                    "truncation_upper_raw": truncation_upper_raw_by_component.get(
                        key, math.nan
                    ),
                    **(
                        {
                            "base_truncation_upper_raw": (
                                base_truncation_upper_raw_by_component.get(
                                    key, math.nan
                                )
                            ),
                            "fixed_c_u": fixed_c_u,
                            "truncation_bound_policy": (
                                truncation_bound_policy
                            ),
                            "truncation_support_expansion_multiplier": (
                                truncation_support_expansion_multiplier
                            ),
                            "censoring_bound_scope": (
                                censoring_support_metadata.get(
                                    "censoring_bound_scope",
                                    "selected_topk_train",
                                )
                            ),
                            "censoring_support_manifest_sha256": (
                                censoring_support_metadata.get(
                                    "censoring_support_manifest_sha256", ""
                                )
                            ),
                        }
                        if predictive_contract
                        in RHO_ONLY_CENSORED_CONTRACTS
                        else {}
                    ),
                    "selected_rho": float(outcome.selected_state.rho),
                    "sigma_formula": "alternate_log1p_transform_residual_rmse",
                    "tau_formula": "tau_equals_computed_sigma",
                    "sigma_grouping": "component_horizon",
                }
            )

    if "one_layer" not in outcomes:
        raise SystemExit("fixed bridge rho-only must produce a one-layer config")
    one = outcomes["one_layer"]
    one_path = _variant_config_path(args.out_config, "one_layer")
    Path(args.out_config).write_bytes(one_path.read_bytes())
    pd.read_csv(_variant_report_path(args.out_report, "one_layer")).to_csv(
        args.out_report, index=False
    )
    written_paths.extend([Path(args.out_config), Path(args.out_report)])

    component_path = Path(args.out_report).with_name(
        "bridge_component_calibration_report.csv"
    )
    reference_path = Path(args.out_report).with_name(
        "parameter_selection_reference.json"
    )
    trace_path = Path(args.out_report).with_name("parameter_selection_trace.csv")
    freeze_path = Path(args.out_report).with_name(
        "parameter_selection_freeze_manifest.json"
    )
    pd.DataFrame(component_rows).to_csv(component_path, index=False)
    pd.concat([outcome.trace for outcome in outcomes.values()], ignore_index=True).to_csv(
        trace_path, index=False
    )
    reference_path.write_text(
        json.dumps(references, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    written_paths.extend([component_path, reference_path, trace_path])
    freeze_manifest = {
        "schema": "caster_rho_only_smallval_freeze_manifest_v1",
        "task_id": task_id,
        "parameter_selection_protocol": "fixed_formula_rho_only_smallval_newton_v1",
        "source_selection_fold_manifest_path": str(Path(args.selection_fold_manifest).resolve()),
        "selection_fold_manifest_sha256": str(metadata.get("selection_fold_manifest_sha256", "")),
        "small_validation_manifest_path": str(smallval_path.resolve()),
        "small_validation_manifest_sha256": smallval_hash,
        "small_validation_seed": int(args.small_validation_seed),
        "distinct_validation_endpoint_rows": 360,
        "validation_fold_count": 10,
        "validation_fold_replay_policy": "independent_same_W0_per_fold",
        "objective_weights": dict(settings.objective_weights),
        "rho_objective_rmse_mode": (
            "overall"
            if "overall_rmse" in settings.objective_weights
            else "short-long"
        ),
        "objective_metric_order": list(settings.objective_metric_order),
        "fixed_parameters": {
            "family": family,
            "sigma_formula": "alternate_log1p_transform_residual_rmse",
            "tau": "sigma",
            "gamma": fixed_gamma if family == MOMENT_T else None,
            "gamma_parameter_status": (
                "fixed_input"
                if family == MOMENT_T
                else "inactive_for_draw_kernel_family"
            ),
            "nu": fixed_nu,
            "distribution": args.distribution,
            "predictive_contract": predictive_contract,
            "truncation_upper_raw_by_component": truncation_upper_raw_by_component,
            "truncation_bound_policy": truncation_bound_policy,
            "truncation_quadrature_order": int(args.truncation_quadrature_order),
            "truncation_zero_mean_epsilon": float(
                args.truncation_zero_mean_epsilon
            ),
            **(
                {
                    "base_truncation_upper_raw_by_component": (
                        base_truncation_upper_raw_by_component
                    ),
                    "fixed_c_u": fixed_c_u,
                    "truncation_support_expansion_multiplier": (
                        truncation_support_expansion_multiplier
                    ),
                    **censoring_support_metadata,
                }
                if predictive_contract in RHO_ONLY_CENSORED_CONTRACTS
                else {}
            ),
        },
        "optimizer": "safeguarded_bounded_newton_log_rho",
        "optimizer_settings": {
            "multi_starts": int(settings.multi_starts),
            "finite_difference_log_step": float(settings.finite_difference_log_step),
            "maximum_newton_log_step": float(settings.maximum_newton_log_step),
            "max_iterations": int(settings.max_iterations),
            "max_evaluations": int(settings.max_evaluations),
            "max_backtracks": int(settings.max_backtracks),
            "hessian_floor": float(settings.hessian_floor),
        },
        "rho_bounds": [float(value) for value in settings.rho_bounds],
        "normalization_anchor_rhos": [
            float(value) for value in settings.anchor_rhos
        ],
        "reference_rho": float(settings.reference_rho),
        "test_rows_used_for_tuning": 0,
        "embargo_rows_used_for_tuning": 0,
        "all_choices_frozen_before_test": True,
        **censoring_support_metadata,
        "selected": {
            variant: outcome.selected_state.serializable()
            for variant, outcome in outcomes.items()
        },
        "artifacts": {
            path.name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for path in written_paths
            if path.is_file()
        },
    }
    freeze_path.write_text(
        json.dumps(freeze_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"bridge_config={args.out_config}")
    print(f"report={args.out_report}")
    print(f"small_validation_manifest={smallval_path}")
    print(f"freeze_manifest={freeze_path}")
    print(f"selected_family={family}")
    for variant, outcome in outcomes.items():
        print(f"selected_rho_{variant}={outcome.selected_state.rho}")


def main() -> None:
    ap = ArgumentParser(description="Select bridge/filter parameters by exact validation-only causal replay.")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument(
        "--draws",
        default="",
        help="Forecast draws required by the formal moment-t versus draw-kernel-t outer search.",
    )
    ap.add_argument("--registry", required=True)
    ap.add_argument(
        "--selection",
        default="",
        help=(
            "Optional selected Top-K CSV. When supplied, bridge calibration and "
            "rho selection are restricted to these selected model_id rows even "
            "if the forecast archive stores all enabled candidates."
        ),
    )
    ap.add_argument(
        "--selection-fold-manifest",
        default="",
        help=(
            "Frozen validation-fold manifest. Formal P1-P10 runs pass the "
            "task-local selection_fold_manifest.csv so every fold restarts "
            "from the same W0."
        ),
    )
    ap.add_argument("--out-config", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument(
        "--calibration-mode",
        default="validation_grid",
        choices=["validation_grid", "fixed_input", "fixed_bridge_rho_only"],
    )
    ap.add_argument(
        "--parameter-selection-protocol",
        default="frozen_joint_multicriterion_causal_replay",
        choices=["frozen_joint_multicriterion_causal_replay", "alternate_nll_grid"],
        help="Formal runs use the frozen multi-criterion joint objective; alternate_nll_grid is diagnostic compatibility only.",
    )
    ap.add_argument(
        "--fix_parameter",
        action="store_true",
        help="Use the fully materialized fixed-parameter entry (Benchmark A rho=.5; Benchmark B rho=.05).",
    )
    ap.add_argument(
        "--fixed-bridge-config-template",
        default="configs/fixed_bridge_defaults.current_project.yaml",
    )
    ap.add_argument("--distribution", default="student_t", choices=["student_t", "gaussian"])
    ap.add_argument(
        "--predictive-contract",
        default=alternate_ARCHIVE_MOMENT,
        choices=list(PREDICTIVE_CONTRACTS),
        help=(
            "Freeze the point/interval/NLL predictive contract. The default "
            "is the fixed archived-moment readout."
        ),
    )
    ap.add_argument("--truncation-quadrature-order", type=int, default=128)
    ap.add_argument("--truncation-zero-mean-epsilon", type=float, default=1e-10)
    ap.add_argument(
        "--fixed-c-u",
        type=float,
        default=DEFAULT_RHO_ONLY_FIXED_C_U,
        help=(
            "Fixed train-only upper-support multiplier for censored rho-only "
            "contracts; it is not optimized."
        ),
    )
    ap.add_argument(
        "--censoring-support-manifest",
        default="",
        help=(
            "Optional eligible-27, train-only upper-support manifest. When "
            "omitted, censored rho-only runs retain the fixed selected "
            "Top-10 train-only support."
        ),
    )
    ap.add_argument("--transform", default="log1p", choices=["log1p", "identity"])
    ap.add_argument("--nu", type=float, default=5.0)
    ap.add_argument("--nu-grid", default=_float_list_text(DEFAULT_NU_GRID))
    ap.add_argument("--fixed-nu", type=float, default=None)
    ap.add_argument("--min-sigma", type=float, default=0.04)
    ap.add_argument("--default-sigma", type=float, default=0.20)
    ap.add_argument("--gamma-grid", default=_float_list_text(DEFAULT_GAMMA_GRID))
    ap.add_argument(
        "--fixed-gamma",
        type=float,
        default=None,
        help="Use one declared gamma without validation selection (diagnostic/backward-compatible path).",
    )
    ap.add_argument(
        "--allow-zero-gamma",
        action="store_true",
        help="Allow gamma=0 in --gamma-grid for explicit diagnostic/sensitivity runs only.",
    )
    ap.add_argument("--gamma-nll-tie-tol", type=float, default=1e-12)
    ap.add_argument("--gamma-prior-strength", default="0")
    ap.add_argument("--gamma-scale-ratio-floor", default=str(DEFAULT_GAMMA_SCALE_RATIO_FLOOR))
    ap.add_argument("--gamma-scale-ratio-penalty-strength", default="0")
    ap.add_argument("--sigma-multipliers", default="0.5,1,2,4")
    ap.add_argument("--min-gamma-rows", type=int, default=5)
    ap.add_argument("--rho-grid", default=_float_list_text(DEFAULT_RHO_GRID))
    ap.add_argument(
        "--allow-rho-grid-outside-result-range",
        action="store_true",
        help="Allow validation-grid rho values outside the formal result grid range for explicit diagnostics only.",
    )
    ap.add_argument("--rho-selection-variant", default="both", choices=["one_layer", "hierarchical", "both"])
    ap.add_argument(
        "--fixed-bridge-family",
        default="auto",
        choices=["auto", "moment_t", "draw_kernel_t"],
        help="Family frozen before the rho-only pilot; auto currently fixes moment-t for all four tasks.",
    )
    ap.add_argument("--small-validation-folds", type=int, default=10)
    ap.add_argument("--small-validation-seed", type=int, default=42)
    ap.add_argument("--rho-newton-multi-starts", type=int, default=3)
    ap.add_argument("--rho-newton-fd-log-step", type=float, default=0.10)
    ap.add_argument("--rho-newton-max-log-step", type=float, default=0.50)
    ap.add_argument("--rho-newton-max-iterations", type=int, default=10)
    ap.add_argument("--rho-newton-max-evaluations", type=int, default=48)
    ap.add_argument("--rho-newton-max-backtracks", type=int, default=8)
    ap.add_argument("--rho-newton-hessian-floor", type=float, default=1e-6)
    ap.add_argument(
        "--rho-objective-weight-nll",
        type=float,
        default=RHO_ONLY_OBJECTIVE_WEIGHTS["nll"],
    )
    ap.add_argument(
        "--rho-objective-weight-wis",
        type=float,
        default=RHO_ONLY_OBJECTIVE_WEIGHTS["wis"],
    )
    ap.add_argument(
        "--rho-objective-weight-short-rmse",
        type=float,
        default=RHO_ONLY_OBJECTIVE_WEIGHTS["short_rmse"],
    )
    ap.add_argument(
        "--rho-objective-weight-long-rmse",
        type=float,
        default=RHO_ONLY_OBJECTIVE_WEIGHTS["long_rmse"],
    )
    ap.add_argument(
        "--rho-objective-rmse-mode",
        choices=["short-long", "overall"],
        default="short-long",
    )
    ap.add_argument(
        "--rho-objective-weight-overall-rmse",
        type=float,
        default=None,
        help=(
            "Overall-RMSE weight in overall mode; defaults to the declared "
            "short- plus long-RMSE weights."
        ),
    )
    ap.add_argument(
        "--rho-objective-weight-mae",
        type=float,
        default=RHO_ONLY_OBJECTIVE_WEIGHTS["mae"],
    )
    ap.add_argument(
        "--rho-objective-weight-coverage-penalty",
        type=float,
        default=RHO_ONLY_OBJECTIVE_WEIGHTS["coverage_penalty"],
    )
    ap.add_argument("--target-ess-fraction", type=float, default=0.5)
    ap.add_argument("--rho-model-ess-penalty", type=float, default=None)
    ap.add_argument("--rho-family-ess-penalty", type=float, default=0.0)
    ap.add_argument("--rho-top1-penalty", type=float, default=0.0)
    ap.add_argument("--rho-top1-target", type=float, default=0.0)
    ap.add_argument("--scale-bound-multiplier", type=float, default=4.0)
    ap.add_argument("--coordinate-passes", type=int, default=2)
    ap.add_argument("--refinement-passes", type=int, default=1)
    ap.add_argument("--multi-starts", type=int, default=2)
    ap.add_argument("--initial-log-step", type=float, default=math.log(2.0))
    ap.add_argument("--exact-tie-tol", type=float, default=1e-10)
    ap.add_argument("--metric-epsilon", type=float, default=1e-8)
    ap.add_argument("--robust-scale-floor", type=float, default=1e-3)
    ap.add_argument("--robust-relative-floor", type=float, default=0.05)
    ap.add_argument("--coverage-tolerance", type=float, default=0.03)
    ap.add_argument("--coverage-upper-weight", type=float, default=0.5)
    ap.add_argument("--task-id", default="", help="Optional Benchmark task id, e.g. benchmark_b_covid, benchmark_b_flu, or benchmark_b_pooled.")
    ap.add_argument("--target-components", default="", help="Comma-separated target components for this task.")
    ap.add_argument(
        "--posterior-scope",
        default="",
        choices=["", "component_stratified", "pooled_sensitivity", "pooled_shared_posterior"],
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.fix_parameter:
        args.calibration_mode = "fixed_input"
    if args.calibration_mode == "fixed_input":
        timer = RuntimeLogger()
        try:
            with timer.measure("load_inputs"):
                ledger = pd.read_csv(args.ledger)
                task = task_from_args(
                    task_id=args.task_id,
                    target_components=args.target_components,
                    posterior_scope=args.posterior_scope,
                    dataset=(
                        ledger["dataset"].dropna().astype(str).iloc[0]
                        if "dataset" in ledger.columns and not ledger.empty
                        else ""
                    ),
                )
                if task is not None:
                    ledger = ledger[ledger["component"].astype(str).isin(task.components)].copy()
                    if ledger.empty:
                        raise ValueError(f"ledger has no rows for fixed-input task {task.task_id}")
            _run_fixed_input(args, ledger, _task_id(task, ledger), timer)
        except Exception as exc:
            blocker = _write_blocker(Path(args.out_report), "fixed_input", exc)
            print(f"calibration blocked: {exc}")
            print(f"blocker_report={blocker}")
            raise SystemExit(2) from exc
        finally:
            write_timing_log(timer.summary(seed=args.seed), Path(args.out_report).with_name("calibrate_likelihood_bridge_timing.json"))
        print(f"bridge_config={args.out_config}")
        print(f"report={args.out_report}")
        return
    args.rho_selection_mode = "validation_grid"
    gamma_grid = _parse_gamma_grid(args.gamma_grid)
    nu_grid = _parse_nu_grid(args.nu_grid)
    if args.fixed_nu is not None and (math.isnan(args.fixed_nu) or args.fixed_nu <= 0.0):
        raise SystemExit("--fixed-nu must be positive or infinity")
    if args.fixed_gamma is not None and (not math.isfinite(args.fixed_gamma) or args.fixed_gamma < 0.0):
        raise SystemExit("--fixed-gamma must be finite and nonnegative")
    if args.fixed_gamma == 0.0 and not args.allow_zero_gamma:
        raise SystemExit("--fixed-gamma 0 requires --allow-zero-gamma and is diagnostic only")
    if any(x == 0.0 for x in gamma_grid) and not args.allow_zero_gamma:
        raise SystemExit("--gamma-grid contains 0; pass --allow-zero-gamma only for diagnostic/sensitivity runs")
    gamma_nll_tie_tol = float(args.gamma_nll_tie_tol)
    if not math.isfinite(gamma_nll_tie_tol) or gamma_nll_tie_tol < 0.0:
        raise SystemExit("--gamma-nll-tie-tol must be finite and nonnegative")
    gamma_prior_strength = _nonnegative_float(args.gamma_prior_strength, "--gamma-prior-strength")
    gamma_scale_ratio_floor = _positive_float(args.gamma_scale_ratio_floor, "--gamma-scale-ratio-floor")
    gamma_scale_ratio_penalty_strength = _nonnegative_float(
        args.gamma_scale_ratio_penalty_strength,
        "--gamma-scale-ratio-penalty-strength",
    )
    sigma_multipliers = _parse_sigma_multipliers(args.sigma_multipliers)
    if int(args.min_gamma_rows) < 1:
        raise SystemExit("--min-gamma-rows must be >= 1")
    rho_model_ess_penalty = (
        float(args.rho_model_ess_penalty)
        if args.rho_model_ess_penalty is not None
        else (
            0.0
            if args.parameter_selection_protocol
            == "frozen_joint_multicriterion_causal_replay"
            else DEFAULT_RHO_MODEL_ESS_PENALTY
        )
    )
    if not math.isfinite(rho_model_ess_penalty) or rho_model_ess_penalty < 0.0:
        raise SystemExit("--rho-model-ess-penalty must be finite and nonnegative")
    rho_family_ess_penalty = float(args.rho_family_ess_penalty)
    if not math.isfinite(rho_family_ess_penalty) or rho_family_ess_penalty < 0.0:
        raise SystemExit("--rho-family-ess-penalty must be finite and nonnegative")
    rho_top1_penalty = float(args.rho_top1_penalty)
    if not math.isfinite(rho_top1_penalty) or rho_top1_penalty < 0.0:
        raise SystemExit("--rho-top1-penalty must be finite and nonnegative")
    rho_top1_target = float(args.rho_top1_target)
    if not math.isfinite(rho_top1_target) or not (0.0 <= rho_top1_target < 1.0):
        raise SystemExit("--rho-top1-target must be finite and in [0,1)")
    if args.parameter_selection_protocol == "frozen_joint_multicriterion_causal_replay":
        forbidden_terms = {
            "--gamma-prior-strength": gamma_prior_strength,
            "--gamma-scale-ratio-penalty-strength": gamma_scale_ratio_penalty_strength,
            "--rho-model-ess-penalty": rho_model_ess_penalty,
            "--rho-family-ess-penalty": rho_family_ess_penalty,
            "--rho-top1-penalty": rho_top1_penalty,
            "--rho-top1-target": rho_top1_target,
        }
        nonzero = [
            option
            for option, value in forbidden_terms.items()
            if not math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=0.0)
        ]
        if nonzero:
            raise SystemExit(
                "frozen multi-criterion selection forbids alternate regularization/ESS/top-1 "
                "objective terms: " + ", ".join(nonzero)
            )
        if args.allow_zero_gamma:
            raise SystemExit(
                "frozen multi-criterion selection requires positive gamma coordinates; "
                "--allow-zero-gamma is diagnostic-only"
            )
        if args.transform != "log1p":
            raise SystemExit("frozen multi-criterion selection requires --transform log1p")
    variants = _variant_names(args.rho_selection_variant)
    rho_grid: list[float] | None = None
    rho_grid = _parse_validation_rho_grid(
        args.rho_grid,
        allow_outside_result_range=bool(args.allow_rho_grid_outside_result_range),
    )
    timer = RuntimeLogger()
    with timer.measure("load_inputs"):
        ledger = pd.read_csv(args.ledger)
        archive = pd.read_csv(args.archive)
        draws = pd.read_csv(args.draws) if args.draws else pd.DataFrame()
        task = task_from_args(
            task_id=args.task_id,
            target_components=args.target_components,
            posterior_scope=args.posterior_scope,
            dataset=ledger["dataset"].dropna().astype(str).iloc[0] if "dataset" in ledger.columns and not ledger.empty else "",
        )
        ledger, archive = filter_ledger_archive_for_task(ledger, archive, task)
        if not draws.empty:
            draws = draws[
                draws["forecast_id"].astype(str).isin(
                    set(ledger["forecast_id"].astype(str))
                )
            ].copy()
        availability_violations = validate_sleeping_model_archive(archive)
        if not availability_violations.empty:
            violation_path = Path(args.out_report).with_name(
                "bridge_availability_violations.csv"
            )
            violation_path.parent.mkdir(parents=True, exist_ok=True)
            availability_violations.to_csv(violation_path, index=False)
            raise SystemExit(
                "bridge calibration archive violates native-availability protocol; "
                f"see {violation_path}"
            )
        registry = read_registry(args.registry)
        all_archive_rows = int(len(archive))
        all_archive_models = sorted(archive["model_id"].dropna().astype(str).unique().tolist())
        selected_model_ids = _read_selected_model_ids(args.selection)
        if selected_model_ids:
            available = set(archive["model_id"].dropna().astype(str))
            missing = [model_id for model_id in selected_model_ids if model_id not in available]
            if missing:
                raise SystemExit(f"selection model_id not present in archive: {missing}")
            archive = archive[archive["model_id"].astype(str).isin(selected_model_ids)].copy()
            if archive.empty:
                raise SystemExit("selected bridge archive has no rows")
        selected_archive_rows_with_placeholders = int(len(archive))
        archive = native_forecast_rows(archive, require_provenance=True)
        if archive.empty:
            raise SystemExit("selected bridge archive has no native forecast rows")
        val, metadata = _validation_only_slice(
            ledger,
            selection_fold_manifest=args.selection_fold_manifest or None,
        )
        if (
            args.parameter_selection_protocol
            == "frozen_joint_multicriterion_causal_replay"
            and _task_id(task, ledger)
            in {
                "benchmark_a",
                "benchmark_b_covid",
                "benchmark_b_flu",
                "benchmark_b_pooled",
            }
            and not any(
                column in val.columns
                for column in ("validation_fold", "fold_id", "fold")
            )
        ):
            raise SystemExit(
                "formal P1-P10 selection requires a frozen validation-fold "
                "assignment; pass --selection-fold-manifest"
            )
        archive_models = archive["model_id"].astype(str).unique()
        metadata["archive_rows_available_all_enabled"] = all_archive_rows
        metadata["archive_model_count_all_enabled"] = len(all_archive_models)
        metadata["archive_rows_available"] = int(len(archive))
        metadata["archive_rows_after_selection_before_native_mask"] = selected_archive_rows_with_placeholders
        metadata["archive_rows_after_selection"] = int(len(archive))
        metadata["archive_structural_placeholder_rows_excluded"] = int(
            selected_archive_rows_with_placeholders - len(archive)
        )
        metadata["bridge_calibration_availability_policy"] = "native_rows_only"
        metadata["calibration_model_scope"] = "selected_topk" if selected_model_ids else "archive_models"
        metadata["selection_path"] = args.selection if selected_model_ids else ""
        metadata["selected_model_ids"] = selected_model_ids
        metadata["archive_rows_joined_to_validation"] = int(archive[archive["forecast_id"].astype(str).isin(set(val["forecast_id"].astype(str)))].shape[0])
        metadata["calibration_mode"] = "validation_grid"
        metadata["official_split_modified"] = False
        metadata["distribution"] = args.distribution
        metadata["nu"] = float(args.nu)
        metadata["nu_grid"] = _float_list_text(nu_grid)
        metadata["nu_used"] = args.distribution == "student_t"
        metadata["gaussian_as_student_t_limit"] = args.distribution == "gaussian"
        metadata["formal_student_t_nu"] = "infinity" if args.distribution == "gaussian" else "component_horizon_selected"
        metadata["kernel_distribution"] = "gaussian" if args.distribution == "gaussian" else "student_t"
        metadata["transform"] = args.transform
        component_report_path = Path(args.out_report).with_name("bridge_component_calibration_report.csv")
        metadata["component_calibration_report_path"] = str(component_report_path)
        metadata["gamma_calibration_enabled"] = (
            args.fixed_gamma is None and args.distribution in {"student_t", "gaussian"}
        )
        metadata["gamma_selection_performed"] = bool(metadata["gamma_calibration_enabled"])
        metadata["gamma_selection_policy"] = "fixed_input" if args.fixed_gamma is not None else "validation_grid"
        metadata["fixed_gamma"] = "" if args.fixed_gamma is None else float(args.fixed_gamma)
        metadata["gamma_grid"] = "" if args.fixed_gamma is not None else _float_list_text(gamma_grid)
        metadata["allow_zero_gamma"] = bool(args.allow_zero_gamma)
        metadata["gamma_nll_tie_tol"] = float(gamma_nll_tie_tol)
        metadata["gamma_selection_objective"] = (
            "not_applicable_fixed_input" if args.fixed_gamma is not None else "joint_component_horizon_validation_nll"
        )
        metadata["nu_selection_performed"] = args.distribution == "student_t" and args.fixed_nu is None
        metadata["eta_selection_performed"] = True
        metadata["eta_parameter_scope"] = "component_horizon"
        metadata["negative_binomial_phi_implemented"] = False
        metadata["gamma_prior_strength"] = float(gamma_prior_strength)
        metadata["gamma_scale_ratio_floor"] = float(gamma_scale_ratio_floor)
        metadata["gamma_scale_ratio_penalty_strength"] = float(gamma_scale_ratio_penalty_strength)
        metadata["sigma_multipliers"] = _float_list_text(sigma_multipliers)
        metadata["sigma_selection_policy"] = "residual_rmse_frozen"
        metadata["bridge_parameter_scope"] = "component_horizon"
        metadata["ledger_sha256"] = _sha256_file(Path(args.ledger))
        metadata["archive_sha256"] = _sha256_file(Path(args.archive))
        metadata["draws_path"] = str(Path(args.draws).resolve()) if args.draws else ""
        metadata["draws_sha256"] = _sha256_file(Path(args.draws)) if args.draws else ""
        metadata["registry_sha256"] = _sha256_file(Path(args.registry))
        metadata["selection_sha256"] = _sha256_file(Path(args.selection)) if args.selection else ""
        metadata["sigma_multipliers_used_for_formal_selection"] = False
        metadata["min_gamma_rows"] = int(args.min_gamma_rows)
        metadata["default_gamma"] = 1.0
        metadata["rho_selection_objective"] = "prequential_untempered_validation_mixture_nll"
        metadata["target_ess_fraction"] = float(args.target_ess_fraction)
        metadata["rho_model_ess_penalty"] = float(rho_model_ess_penalty)
        metadata["rho_family_ess_penalty"] = float(rho_family_ess_penalty)
        metadata["rho_top1_penalty"] = float(rho_top1_penalty)
        metadata["rho_top1_target"] = float(rho_top1_target)
        metadata.update(task_metadata(task, ledger))
    if args.calibration_mode == "fixed_bridge_rho_only":
        _run_fixed_bridge_rho_only(
            args=args,
            ledger=ledger,
            val=val,
            archive=archive,
            draws=draws,
            registry=registry,
            archive_models=archive_models.tolist(),
            metadata=metadata,
            task=task,
            rho_grid=rho_grid,
            timer=timer,
        )
        timing_path = Path(args.out_report).with_name(
            "calibrate_likelihood_bridge_timing.json"
        )
        write_timing_log(timer.summary(seed=args.seed), timing_path)
        return
    if args.parameter_selection_protocol == "frozen_joint_multicriterion_causal_replay":
        _run_joint_selection(
            args=args,
            ledger=ledger,
            val=val,
            archive=archive,
            draws=draws,
            registry=registry,
            archive_models=archive_models.tolist(),
            metadata=metadata,
            task=task,
            gamma_grid=gamma_grid,
            nu_grid=nu_grid,
            rho_grid=rho_grid,
            timer=timer,
        )
        timing_path = Path(args.out_report).with_name(
            "calibrate_likelihood_bridge_timing.json"
        )
        write_timing_log(timer.summary(seed=args.seed), timing_path)
        return
    with timer.measure("bridge_fit"):
        config, component_report = fit_bridge_config(
            val,
            archive,
            distribution=args.distribution,
            transform=args.transform,
            nu=args.nu,
            min_sigma=args.min_sigma,
            default_sigma=args.default_sigma,
            gamma_grid=gamma_grid,
            nu_grid=nu_grid,
            fixed_gamma=args.fixed_gamma,
            fixed_nu=args.fixed_nu,
            sigma_multipliers=sigma_multipliers,
            min_rows_per_component=args.min_gamma_rows,
            allow_zero_gamma=args.allow_zero_gamma,
            gamma_nll_tie_tol=gamma_nll_tie_tol,
            gamma_prior_strength=gamma_prior_strength,
            gamma_scale_ratio_floor=gamma_scale_ratio_floor,
            gamma_scale_ratio_penalty_strength=gamma_scale_ratio_penalty_strength,
            return_report=True,
        )
        component_report_path.parent.mkdir(parents=True, exist_ok=True)
        component_report.to_csv(component_report_path, index=False)
    outputs: dict[str, dict[str, object]] = {}
    with timer.measure("rho_selection"):
        active_registry = registry[registry["model_id"].astype(str).isin(archive_models)].copy()
        metadata["active_registry_models"] = int(len(active_registry))
        for variant in variants:
            if variant == "one_layer":
                report = evaluate_temperature_grid(
                    val,
                    archive,
                    active_registry,
                    config,
                    grid=rho_grid,
                    target_ess_fraction=args.target_ess_fraction,
                    ess_penalty=rho_model_ess_penalty,
                    family_ess_penalty=0.0,
                    top1_penalty=rho_top1_penalty,
                    top1_target=rho_top1_target,
                )
            else:
                report = evaluate_hierarchical_temperature_grid(
                    val,
                    archive,
                    active_registry,
                    config,
                    grid=rho_grid,
                    target_ess_fraction=args.target_ess_fraction,
                    ess_penalty=rho_model_ess_penalty,
                    family_ess_penalty=rho_family_ess_penalty,
                    top1_penalty=rho_top1_penalty,
                    top1_target=rho_top1_target,
                )
            rho = selected_rho(report)
            variant_meta = _variant_metadata(
                metadata,
                mode="validation_grid",
                variant=variant,
                rho_grid=rho_grid,
            )

            report = _augment_report(
                report,
                task=task,
                metadata=variant_meta,
                mode="validation_grid",
                variant=variant,
            )
            config_path = _variant_config_path(args.out_config, variant)
            report_path = _variant_report_path(args.out_report, variant)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report.to_csv(report_path, index=False)
            written_config = write_bridge_config(config, config_path, rho=rho, metadata=variant_meta)
            outputs[variant] = {
                "config": written_config,
                "report": report_path,
                "rho": float(rho),
                "metadata": variant_meta,
                "report_frame": report,
            }

        if "one_layer" in outputs:
            one = outputs["one_layer"]
            Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
            one["report_frame"].to_csv(args.out_report, index=False)                                   
            write_bridge_config(
                config,
                args.out_config,
                rho=float(one["rho"]),
                metadata=one["metadata"],                          
            )
    timing_path = Path(args.out_report).with_name("calibrate_likelihood_bridge_timing.json")
    write_timing_log(timer.summary(seed=args.seed), timing_path)
    print(f"bridge_config={args.out_config}")
    print(f"report={args.out_report}")
    print(f"component_report={component_report_path}")
    if "one_layer" in outputs:
        print(f"selected_rho={outputs['one_layer']['rho']}")
    for variant, paths in outputs.items():
        print(f"bridge_config_{variant}={paths['config']}")
        print(f"report_{variant}={paths['report']}")
        print(f"selected_rho_{variant}={paths['rho']}")
    print(f"sigma_by_component={config.sigma_by_component}")
    print(f"gamma_by_component={config.gamma_by_component}")
    print(f"nu_by_component={config.nu_by_component}")

if __name__ == "__main__":
    main()
