from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import math
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
CASTER_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CASTER_ROOT / "scripts"))

import pandas as pd

from caster.bridge import (
    alternate_ARCHIVE_MOMENT,
    PREDICTIVE_CONTRACTS,
    read_bridge_config,
    score_archive_rows,
    score_draw_rows,
)
from caster.eval import posterior_diagnostics
from caster.filter.availability import native_forecast_rows
from caster.filter.hierarchical import initialize_hierarchical_weights
from result_metric_contract import RESULT_GROUP_COLS, apply_result_metric_contract, metric_slices_from_scored_rows


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARKS = {
    "benchmark_a": {
        "root": REPO_ROOT / "artifacts" / "real_full" / "benchmark_a",
        "ledger": CASTER_ROOT / "data/benchmark_a/curated_full_v3_direct_rollout7/event_ledger.csv",
    },
    "benchmark_b": {
        "root": REPO_ROOT / "artifacts" / "real_full" / "benchmark_b",
        "ledger": CASTER_ROOT / "data/benchmark_b/curated_v26_1_release_lag1/benchmark_b_pooled/event_ledger.csv",
    },
}
ABLATIONS = [
    "top1_collapse",
    "no_temperature",
    "no_hierarchy",
    "half_bank",
    "caster_hierarchical_draw_kernel",
]
ABLATION_DISPLAY_LABELS = {
    "top1_collapse": "Top-1 collapse",
    "no_temperature": "No temperature",
    "no_hierarchy": "No hierarchy",
    "half_bank": "Half-size candidate bank",
    "caster_hierarchical_draw_kernel": "Hierarchical draw-kernel evidence",
}
ABLATION_REFERENCE_METHODS = {
    "caster_hierarchical_draw_kernel": "caster_hierarchical",
}
ABLATION_DELTA_METRICS = [
    "mae",
    "rmse",
    "nll",
    "bridge_nll",
    "coverage_90",
    "width_90",
    "wis",
    "model_ess",
    "structural_entropy",
    "top1_mass",
]
Z50 = 0.67448975
Z90 = 1.64485363
EPS = 1e-12
EXACT_NLL_SCORE_BASIS = "exact_asof_posterior_mixture_bridge"
EXACT_DRAW_KERNEL_NLL_SCORE_BASIS = "exact_asof_posterior_draw_kernel_mixture"
BLOCKED_EXACT_NLL_SCORE_BASIS = "blocked_missing_exact_asof_posterior_mixture_bridge"
DIAGNOSTIC_MOMENT_MATCHED_BASIS = "diagnostic_moment_matched_bridge_excluded_from_formal_nll"
alternate_ASOF_AVAILABLE_BEFORE_POLICY = "asof_release_time_available_before_forecast_origin"
ASOF_LTE_POLICY = "asof_release_time_lte_forecast_origin"
SNAPSHOT_TIME_ALIASES = [
    "release_time",
    "as_of",
    "as_of_time",
    "asof_time",
    "posterior_snapshot_time",
    "snapshot_time",
]
FORECAST_ORIGIN_ALIASES = ["forecast_origin", "origin", "origin_date", "as_of_date"]
FORECAST_ID_ALIASES = ["forecast_id", "event_id", "target_id"]
MODEL_ID_ALIASES = ["model_id", "model", "candidate_id", "adapter_id"]
WEIGHT_ALIASES = ["weight", "posterior_weight", "model_weight", "prior_weight"]
FAMILY_ALIASES = ["family", "model_family", "mechanism_family"]
FAMILY_WEIGHT_ALIASES = ["family_weight", "posterior_family_weight", "weight"]
INNER_WEIGHT_ALIASES = ["inner_weight", "within_family_weight", "conditional_weight", "weight"]

RESULT_READY_COLUMNS = [
    "dataset",
    "method",
    "method_group",
    "task_id",
    "target_component",
    "components",
    "posterior_scope",
    "bridge_calibration_scope",
    "metric_group",
    "metric",
    "value",
    "value_text",
    "status",
    "reason",
    "source_artifact",
    "bridge_calibration_split",
    "rho",
    "n_validation_rows",
    "n_test_rows",
    "candidate_count",
    "variant",
    "posterior_update_splits",
    "test_rows_used_for_bridge_calibration",
    "test_rows_used_for_posterior_update",
    "posterior_update_policy",
    "posterior_update_scope",
    "test_rows_used_for_posterior_update_policy",
    "posterior_readout_policy",
    "release_availability_rule",
    "readout_rows_future_snapshot_violation",
    "readout_rows_self_target_update_violation",
    "max_stale_posterior_age_days",
    "p95_stale_posterior_age_days",
    "native_likelihoods_compared",
    "adapter_native_likelihoods_compared",
    "adapter_native_likelihood_executed",
    "diagnostic_only",
    "diagnostic_values_executed",
    "used_for_h5_positive_claim",
    "unsafe_diagnostic",
    "not_for_positive_h5_claim",
    "score_update_basis",
    "evaluation_basis",
    "unsafe_native_proxy_executed",
    "unsafe_no_event_ledger_executed",
    "no_event_ledger_enforcement_relaxed",
    "diagnostic_readout_policy",
    "native_likelihood_scores_path",
    "native_likelihood_evidence_log_path",
    "true_native_coverage_summary",
    "blocked_selected_model_count",
    "unavailable_selected_model_count",
    "proxy_fallback_used",
    "proxy_fallback_row_count",
    "adapter_loglik_row_count",
    "native_likelihoods_comparable",
    "topk_selection_stage",
    "topk_selection_rerun_per_algorithm",
    "topk_selection_reused_across_algorithms",
    "topk_selection_timing_sec",
    "topk_selection_charged_sec",
    "topk_selection_path",
    "topk_selection_timing_path",
    "topk_selection_metadata_path",
    "topk_selection_algorithm_id",
    "topk_selection_benchmark",
    "topk_selection_query",
    "topk_selection_top_k",
    "topk_selection_selected_model_count",
    "topk_selection_selected_model_ids",
    "nll_score_basis",
    "formal_nll_status",
    "nll_source_kind",
    "probability_score_basis",
    "diagnostic_moment_matched_nll_available",
    "diagnostic_moment_matched_nll",
    "asof_mixture_weight_validation_path",
]
CONTRACT_COLUMNS = [
    "bridge_calibration_split",
    "rho",
    "task_id",
    "target_component",
    "components",
    "posterior_scope",
    "bridge_calibration_scope",
    "n_validation_rows",
    "n_test_rows",
    "candidate_count",
    "variant",
    "posterior_update_splits",
    "test_rows_used_for_bridge_calibration",
    "test_rows_used_for_posterior_update",
    "posterior_update_policy",
    "posterior_update_scope",
    "test_rows_used_for_posterior_update_policy",
    "posterior_readout_policy",
    "release_availability_rule",
    "readout_rows_future_snapshot_violation",
    "readout_rows_self_target_update_violation",
    "max_stale_posterior_age_days",
    "p95_stale_posterior_age_days",
    "native_likelihoods_compared",
    "adapter_native_likelihoods_compared",
    "adapter_native_likelihood_executed",
    "diagnostic_only",
    "diagnostic_values_executed",
    "used_for_h5_positive_claim",
    "unsafe_diagnostic",
    "not_for_positive_h5_claim",
    "score_update_basis",
    "evaluation_basis",
    "unsafe_native_proxy_executed",
    "unsafe_no_event_ledger_executed",
    "no_event_ledger_enforcement_relaxed",
    "diagnostic_readout_policy",
    "native_likelihood_scores_path",
    "native_likelihood_evidence_log_path",
    "true_native_coverage_summary",
    "blocked_selected_model_count",
    "unavailable_selected_model_count",
    "proxy_fallback_used",
    "proxy_fallback_row_count",
    "adapter_loglik_row_count",
    "native_likelihoods_comparable",
    "topk_selection_stage",
    "topk_selection_rerun_per_algorithm",
    "topk_selection_reused_across_algorithms",
    "topk_selection_timing_sec",
    "topk_selection_charged_sec",
    "topk_selection_path",
    "topk_selection_timing_path",
    "topk_selection_metadata_path",
    "topk_selection_algorithm_id",
    "topk_selection_benchmark",
    "topk_selection_query",
    "topk_selection_top_k",
    "topk_selection_selected_model_count",
    "topk_selection_selected_model_ids",
]
PROTOCOL_COLUMNS = [
    "mode",
    "mode_kind",
    "country",
    "country_code",
    "jurisdiction",
    "entity_id",
]
ASOF_COLUMNS = [
    "posterior_snapshot_time",
    "used_prior_snapshot",
    "stale_posterior_age_days",
    "future_snapshot_violation",
    "self_target_update_violation",
    "posterior_update_policy",
    "release_availability_rule",
]
FORECAST_INTERVAL_COLUMNS = [
    "dataset",
    "method",
    "forecast_id",
    *PROTOCOL_COLUMNS,
    "forecast_origin",
    "target_time",
    "component",
    "horizon",
    "observed_value",
    "predictive_mean",
    "predictive_median",
    "predictive_var",
    "lower_50",
    "upper_50",
    "lower_90",
    "upper_90",
    "lower_95",
    "upper_95",
    *ASOF_COLUMNS,
    "split",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_parameter_selection_freeze(root: Path) -> dict[str, Any]:
    ""

    path = root / "parameter_selection_freeze_manifest.json"
    payload = _read_json(path)
    if payload.get("schema") != "caster_parameter_selection_freeze_manifest_v1":
        raise SystemExit(f"{path} is missing or has the wrong freeze-manifest schema")
    if payload.get("parameter_selection_protocol") != "frozen_joint_multicriterion_causal_replay_v1":
        raise SystemExit(f"{path} does not describe the frozen multi-criterion protocol")
    if payload.get("all_choices_frozen_before_test") is not True:
        raise SystemExit(f"{path} does not freeze every parameter choice before test")
    if int(payload.get("test_rows_used_for_tuning", -1)) != 0:
        raise SystemExit(f"{path} records test rows used for parameter selection")
    if int(payload.get("embargo_rows_used_for_tuning", -1)) != 0:
        raise SystemExit(f"{path} records embargo rows used for parameter selection")
    if int(payload.get("validation_fold_count", 0)) <= 1:
        raise SystemExit(f"{path} does not bind the declared independent validation folds")
    if payload.get("validation_fold_replay_policy") != "independent_same_W0_per_fold":
        raise SystemExit(f"{path} does not restart every validation fold from W0")
    if not str(payload.get("selection_fold_manifest_sha256", "")):
        raise SystemExit(f"{path} is missing the frozen validation-fold hash")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SystemExit(f"{path} is missing the frozen artifact map")
    required = [
        "bridge_config.json",
        "bridge_config.one_layer.json",
        "bridge_config.hierarchical.json",
        "bridge_config.one_layer.moment_t.json",
        "bridge_config.one_layer.draw_kernel_t.json",
        "bridge_config.hierarchical.moment_t.json",
        "bridge_config.hierarchical.draw_kernel_t.json",
    ]
    for name in required:
        config_path = root / name
        record = artifacts.get(name)
        if not config_path.is_file() or not isinstance(record, dict):
            raise SystemExit(f"{path} is missing frozen artifact evidence for {name}")
        expected = str(record.get("sha256", ""))
        actual = _sha256_file(config_path)
        if not expected or expected != actual:
            raise SystemExit(f"{config_path} no longer matches the pre-test freeze manifest")
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _ablation_reference_method(ablation: str) -> str:
    return ABLATION_REFERENCE_METHODS.get(str(ablation), "caster_one_layer")


def _ablation_export_fields(ablation: str) -> dict[str, Any]:
    reference = _ablation_reference_method(ablation)
    return {
        "display_label": ABLATION_DISPLAY_LABELS.get(ablation, ablation),
        "ablation_row_label": ABLATION_DISPLAY_LABELS.get(ablation, ablation),
        "reference_method": reference,
        "delta_reference_method": reference,
        "delta_convention": "ablated_score_minus_reference_score",
        "ablation_display_order": ABLATIONS.index(ablation) + 1 if ablation in ABLATIONS else 999,
    }


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _attach_ablation_deltas(row: dict[str, Any], reference_row: dict[str, Any] | None) -> None:
    if not reference_row:
        for metric in ABLATION_DELTA_METRICS:
            row[f"{metric}_delta_vs_reference"] = math.nan
        return
    for metric in ABLATION_DELTA_METRICS:
        value = _float_or_nan(row.get(metric))
        reference = _float_or_nan(reference_row.get(metric))
        row[f"{metric}_delta_vs_reference"] = value - reference if math.isfinite(value) and math.isfinite(reference) else math.nan


def _rows(path: Path) -> int | None:
    if not path.exists() or path.suffix.lower() != ".csv":
        return None
    try:
        return int(sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
    except OSError:
        return None


def _bridge_config_path(root: Path, variant: str = "one_layer") -> Path:
    candidate = root / f"bridge_config.{variant}.json"
    return candidate if candidate.exists() else root / "bridge_config.json"


def _require_frozen_bridge(root: Path, variant: str = "one_layer") -> dict[str, Any]:
    bridge_path = _bridge_config_path(root, variant)
    payload = _read_json(bridge_path)
    if "rho" not in payload:
        raise SystemExit(f"{bridge_path} must contain frozen rho from validation-only NEW-BRIDGE")
    meta = payload.get("calibration_metadata", {})
    if not isinstance(meta, dict):
        raise SystemExit(f"{bridge_path} calibration_metadata must be an object")
    split = str(meta.get("calibration_split", "val"))
    if split != "val":
        raise SystemExit(f"{bridge_path} calibration_split must be val, got {split!r}")
    expected_variant = "one_layer" if variant == "one_layer" else "hierarchical"
    got_variant = str(meta.get("rho_selection_variant", expected_variant))
    if got_variant != expected_variant:
        raise SystemExit(f"{bridge_path} rho_selection_variant must be {expected_variant}, got {got_variant!r}")
    if str(meta.get("parameter_selection_protocol", "")) != "frozen_joint_multicriterion_causal_replay_v1":
        raise SystemExit(f"{bridge_path} is not a frozen frozen multi-criterion selection artifact")
    if int(meta.get("test_rows_used_for_tuning", -1)) != 0:
        raise SystemExit(f"{bridge_path} used test rows for parameter selection")
    if meta.get("all_choices_frozen_before_test") is not True:
        raise SystemExit(f"{bridge_path} does not declare all choices frozen before test")
    if int(meta.get("validation_fold_count", 0)) <= 1:
        raise SystemExit(f"{bridge_path} does not bind independent validation folds")
    if meta.get("validation_fold_replay_policy") != "independent_same_W0_per_fold":
        raise SystemExit(f"{bridge_path} does not restart every validation fold from W0")
    if not str(meta.get("selection_fold_manifest_sha256", "")):
        raise SystemExit(f"{bridge_path} is missing the frozen validation-fold hash")
    return payload


def _require_family_bridge(
    path: Path, *, variant: str, family: str
) -> dict[str, Any]:
    payload = _read_json(path)
    if "rho" not in payload:
        raise SystemExit(f"{path} must contain independently selected frozen rho")
    meta = payload.get("calibration_metadata", {})
    if not isinstance(meta, dict):
        raise SystemExit(f"{path} calibration_metadata must be an object")
    if meta.get("parameter_selection_protocol") != "frozen_joint_multicriterion_causal_replay_v1":
        raise SystemExit(f"{path} is not a frozen frozen multi-criterion family artifact")
    if meta.get("rho_selection_variant") != variant:
        raise SystemExit(f"{path} has the wrong filtering variant")
    if meta.get("selected_bridge_family") != family:
        raise SystemExit(f"{path} has the wrong bridge family")
    expected_source = "draw_kernel" if family == "draw_kernel_t" else "archive_moment"
    if meta.get("score_source") != expected_source:
        raise SystemExit(f"{path} has an inconsistent score source")
    if int(meta.get("test_rows_used_for_tuning", -1)) != 0:
        raise SystemExit(f"{path} used test rows for parameter selection")
    if meta.get("all_choices_frozen_before_test") is not True:
        raise SystemExit(f"{path} does not freeze all choices before test")
    if int(meta.get("validation_fold_count", 0)) <= 1:
        raise SystemExit(f"{path} does not bind independent validation folds")
    if meta.get("validation_fold_replay_policy") != "independent_same_W0_per_fold":
        raise SystemExit(f"{path} does not restart every validation fold from W0")
    if not str(meta.get("selection_fold_manifest_sha256", "")):
        raise SystemExit(f"{path} is missing the frozen validation-fold hash")
    expected_flags = {
        "sigma_selection_performed": family == "moment_t",
        "gamma_selection_performed": family == "moment_t",
        "tau_selection_performed": family == "draw_kernel_t",
    }
    for key, expected in expected_flags.items():
        if meta.get(key) is not expected:
            raise SystemExit(f"{path} has invalid active-coordinate flag {key}")
    sigma = payload.get("sigma_by_component", {})
    tau = payload.get("tau_by_component", {})
    gamma = payload.get("gamma_by_component", {})
    nu = payload.get("nu_by_component", {})
    if not all(isinstance(values, dict) for values in (sigma, tau, gamma, nu)):
        raise SystemExit(f"{path} has malformed component-parameter maps")
    active_scale = sigma if family == "moment_t" else tau
    if not active_scale or set(nu) != set(active_scale):
        raise SystemExit(f"{path} has incomplete active scale/nu coordinates")
    if family == "moment_t":
        if tau or set(gamma) != set(sigma):
            raise SystemExit(f"{path} must materialize sigma/gamma and leave tau inactive")
    elif sigma or gamma:
        raise SystemExit(f"{path} must materialize tau and leave sigma/gamma inactive")
    for coordinate, values in (("scale", active_scale), ("nu", nu)):
        if any(float(value) <= 0.0 or math.isnan(float(value)) for value in values.values()):
            raise SystemExit(f"{path} has invalid positive {coordinate} coordinates")
    return payload


def _selected_score_source(payload: dict[str, Any], *, label: str) -> str:
    meta = payload.get("calibration_metadata", {})
    family = str(meta.get("selected_bridge_family", ""))
    source = str(meta.get("score_source", ""))
    expected = {
        "moment_t": "archive_moment",
        "draw_kernel_t": "draw_kernel",
    }.get(family)
    if expected is None or source != expected:
        raise SystemExit(
            f"{label} has inconsistent selected bridge family/score source: "
            f"family={family!r} source={source!r}"
        )
    return source


def _join_splits(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(x) for x in value)
    if value is None:
        return ""
    return str(value)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return default
        return int(float(text))
    except Exception:
        return default


def _canonical_meta_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(str(x) for x in value)
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else str(value)
    return str(value)


def _contract_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ";".join(str(x) for x in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _predictive_contract_from_artifacts(
    run_metadata: dict[str, Any],
    bridge_payload: dict[str, Any],
    *,
    source: str,
) -> str:
    bridge_meta = bridge_payload.get("calibration_metadata", {})
    if not isinstance(bridge_meta, dict):
        bridge_meta = {}
    config_contract = str(
        bridge_payload.get("predictive_contract", alternate_ARCHIVE_MOMENT)
    )
    calibration_contract = str(
        bridge_meta.get("predictive_contract", alternate_ARCHIVE_MOMENT)
    )
    run_contract = str(
        run_metadata.get("predictive_contract", alternate_ARCHIVE_MOMENT)
    )
    if config_contract not in PREDICTIVE_CONTRACTS:
        raise SystemExit(
            f"{source} has unsupported predictive_contract={config_contract!r}"
        )
    if len({config_contract, calibration_contract, run_contract}) != 1:
        raise SystemExit(
            f"{source} mixes predictive contracts across bridge config, "
            "calibration metadata, and run metadata"
        )
    return config_contract


def _contract(root: Path, metadata_name: str, bridge_payload: dict[str, Any], *, diagnostic_only: bool = False) -> dict[str, Any]:
    meta = _read_json(root / metadata_name)
    bridge_meta = bridge_payload.get("calibration_metadata", {})
    if not isinstance(bridge_meta, dict):
        bridge_meta = {}
    predictive_contract = _predictive_contract_from_artifacts(
        meta,
        bridge_payload,
        source=str(root / metadata_name),
    )
    return {
        "bridge_calibration_split": meta.get("bridge_calibration_split", bridge_meta.get("calibration_split", "val")),
        "rho": meta.get("rho", bridge_payload.get("rho", math.nan)),
        "task_id": meta.get("task_id", bridge_meta.get("task_id", "")),
        "target_component": meta.get("target_component", bridge_meta.get("target_component", "")),
        "components": meta.get("components", bridge_meta.get("components", "")),
        "posterior_scope": meta.get("posterior_scope", bridge_meta.get("posterior_scope", "")),
        "bridge_calibration_scope": meta.get("bridge_calibration_scope", bridge_meta.get("bridge_calibration_scope", "")),
        "n_validation_rows": _safe_int(meta.get("n_validation_rows", bridge_meta.get("n_validation_rows", 0))),
        "n_test_rows": _safe_int(meta.get("n_test_rows", bridge_meta.get("n_test_rows", 0))),
        "candidate_count": _safe_int(meta.get("candidate_count", 0)),
        "variant": meta.get("variant", ""),
        "predictive_contract": predictive_contract,
        "posterior_update_splits": _join_splits(meta.get("posterior_update_splits", ["train", "val"] if not diagnostic_only else [])),
        "test_rows_used_for_bridge_calibration": int(meta.get("test_rows_used_for_bridge_calibration", bridge_meta.get("test_rows_used_for_tuning", 0))),
        "test_rows_used_for_posterior_update": int(meta.get("test_rows_used_for_posterior_update", 0)),
        "posterior_update_policy": meta.get("posterior_update_policy", "holdout_train_val"),
        "posterior_update_scope": meta.get("posterior_update_scope", _join_splits(meta.get("posterior_update_splits", []))),
        "test_rows_used_for_posterior_update_policy": meta.get("test_rows_used_for_posterior_update_policy", ""),
        "posterior_readout_policy": meta.get("posterior_readout_policy", ""),
        "release_availability_rule": meta.get("release_availability_rule", ""),
        "readout_rows_future_snapshot_violation": int(meta.get("readout_rows_future_snapshot_violation", -1 if not diagnostic_only else 0)),
        "readout_rows_self_target_update_violation": int(meta.get("readout_rows_self_target_update_violation", -1 if not diagnostic_only else 0)),
        "max_stale_posterior_age_days": float(meta.get("max_stale_posterior_age_days", 0.0) or 0.0),
        "p95_stale_posterior_age_days": float(meta.get("p95_stale_posterior_age_days", 0.0) or 0.0),
        "native_likelihoods_compared": _boolish(meta.get("native_likelihoods_compared", False)),
        "adapter_native_likelihoods_compared": _boolish(meta.get("adapter_native_likelihoods_compared", False)),
        "adapter_native_likelihood_executed": _boolish(meta.get("adapter_native_likelihood_executed", False)),
        "diagnostic_only": bool(diagnostic_only),
        "diagnostic_values_executed": _boolish(meta.get("diagnostic_values_executed", False)),
        "used_for_h5_positive_claim": _boolish(meta.get("used_for_h5_positive_claim", False)),
        "unsafe_diagnostic": _boolish(meta.get("unsafe_diagnostic", False)),
        "not_for_positive_h5_claim": _boolish(meta.get("not_for_positive_h5_claim", False)),
        "score_update_basis": meta.get("score_update_basis", ""),
        "evaluation_basis": meta.get("evaluation_basis", ""),
        "unsafe_native_proxy_executed": _boolish(meta.get("unsafe_native_proxy_executed", False)),
        "unsafe_no_event_ledger_executed": _boolish(meta.get("unsafe_no_event_ledger_executed", False)),
        "no_event_ledger_enforcement_relaxed": _boolish(meta.get("no_event_ledger_enforcement_relaxed", False)),
        "diagnostic_readout_policy": meta.get("diagnostic_readout_policy", ""),
        "native_likelihood_scores_path": meta.get("native_likelihood_scores_path", ""),
        "native_likelihood_evidence_log_path": meta.get("native_likelihood_evidence_log_path", ""),
        "true_native_coverage_summary": meta.get("true_native_coverage_summary", ""),
        "blocked_selected_model_count": _safe_int(meta.get("blocked_selected_model_count", 0)),
        "unavailable_selected_model_count": _safe_int(meta.get("unavailable_selected_model_count", 0)),
        "proxy_fallback_used": _boolish(meta.get("proxy_fallback_used", False)),
        "proxy_fallback_row_count": _safe_int(meta.get("proxy_fallback_row_count", 0)),
        "adapter_loglik_row_count": _safe_int(meta.get("adapter_loglik_row_count", 0)),
        "native_likelihoods_comparable": _boolish(meta.get("native_likelihoods_comparable", meta.get("native_likelihoods_compared", False))),
        "topk_selection_stage": meta.get("topk_selection_stage", ""),
        "topk_selection_rerun_per_algorithm": _boolish(meta.get("topk_selection_rerun_per_algorithm", False)),
        "topk_selection_reused_across_algorithms": _boolish(meta.get("topk_selection_reused_across_algorithms", False)),
        "topk_selection_timing_sec": float(meta.get("topk_selection_timing_sec", 0.0) or 0.0),
        "topk_selection_charged_sec": float(meta.get("topk_selection_charged_sec", meta.get("topk_selection_timing_sec", 0.0)) or 0.0),
        "topk_selection_path": meta.get("topk_selection_path", ""),
        "topk_selection_timing_path": meta.get("topk_selection_timing_path", ""),
        "topk_selection_metadata_path": meta.get("topk_selection_metadata_path", ""),
        "topk_selection_algorithm_id": meta.get("topk_selection_algorithm_id", ""),
        "topk_selection_benchmark": meta.get("topk_selection_benchmark", ""),
        "topk_selection_query": meta.get("topk_selection_query", ""),
        "topk_selection_top_k": _safe_int(meta.get("topk_selection_top_k", 0)),
        "topk_selection_selected_model_count": _safe_int(meta.get("topk_selection_selected_model_count", 0)),
        "topk_selection_selected_model_ids": meta.get("topk_selection_selected_model_ids", ""),
    }


def _contract_from_ablation(meta: dict[str, Any], bridge_payload: dict[str, Any]) -> dict[str, Any]:
    bridge_meta = bridge_payload.get("calibration_metadata", {})
    if not isinstance(bridge_meta, dict):
        bridge_meta = {}
    predictive_contract = _predictive_contract_from_artifacts(
        meta,
        bridge_payload,
        source="ablation metadata",
    )
    return {
        "bridge_calibration_split": meta.get("bridge_calibration_split", bridge_meta.get("calibration_split", "val")),
        "rho": meta.get("rho_override", meta.get("rho", bridge_payload.get("rho", math.nan))),
        "task_id": meta.get("task_id", bridge_meta.get("task_id", "")),
        "target_component": meta.get("target_component", bridge_meta.get("target_component", "")),
        "components": meta.get("components", bridge_meta.get("components", "")),
        "posterior_scope": meta.get("posterior_scope", bridge_meta.get("posterior_scope", "")),
        "bridge_calibration_scope": meta.get("bridge_calibration_scope", bridge_meta.get("bridge_calibration_scope", "")),
        "n_validation_rows": _safe_int(meta.get("n_validation_rows", bridge_meta.get("n_validation_rows", 0))),
        "n_test_rows": _safe_int(meta.get("n_test_rows", bridge_meta.get("n_test_rows", 0))),
        "candidate_count": _safe_int(meta.get("candidate_count", 0)),
        "variant": meta.get("variant", ""),
        "predictive_contract": predictive_contract,
        "posterior_update_splits": _join_splits(meta.get("posterior_update_splits", [])),
        "test_rows_used_for_bridge_calibration": int(meta.get("test_rows_used_for_bridge_calibration", bridge_meta.get("test_rows_used_for_tuning", 0))),
        "test_rows_used_for_posterior_update": int(meta.get("test_rows_used_for_posterior_update", 0)),
        "posterior_update_policy": meta.get("posterior_update_policy", "holdout_train_val"),
        "posterior_update_scope": meta.get("posterior_update_scope", _join_splits(meta.get("posterior_update_splits", []))),
        "test_rows_used_for_posterior_update_policy": meta.get("test_rows_used_for_posterior_update_policy", ""),
        "posterior_readout_policy": meta.get("posterior_readout_policy", ""),
        "release_availability_rule": meta.get("release_availability_rule", ""),
        "readout_rows_future_snapshot_violation": int(meta.get("readout_rows_future_snapshot_violation", -1 if not bool(meta.get("diagnostic_only", False)) else 0)),
        "readout_rows_self_target_update_violation": int(meta.get("readout_rows_self_target_update_violation", -1 if not bool(meta.get("diagnostic_only", False)) else 0)),
        "max_stale_posterior_age_days": float(meta.get("max_stale_posterior_age_days", 0.0) or 0.0),
        "p95_stale_posterior_age_days": float(meta.get("p95_stale_posterior_age_days", 0.0) or 0.0),
        "native_likelihoods_compared": _boolish(meta.get("native_likelihoods_compared", False)),
        "adapter_native_likelihoods_compared": _boolish(meta.get("adapter_native_likelihoods_compared", False)),
        "adapter_native_likelihood_executed": _boolish(meta.get("adapter_native_likelihood_executed", False)),
        "diagnostic_only": _boolish(meta.get("diagnostic_only", False)),
        "diagnostic_values_executed": _boolish(meta.get("diagnostic_values_executed", False)),
        "used_for_h5_positive_claim": _boolish(meta.get("used_for_h5_positive_claim", False)),
        "unsafe_diagnostic": _boolish(meta.get("unsafe_diagnostic", False)),
        "not_for_positive_h5_claim": _boolish(meta.get("not_for_positive_h5_claim", False)),
        "score_update_basis": meta.get("score_update_basis", ""),
        "evaluation_basis": meta.get("evaluation_basis", ""),
        "unsafe_native_proxy_executed": _boolish(meta.get("unsafe_native_proxy_executed", False)),
        "unsafe_no_event_ledger_executed": _boolish(meta.get("unsafe_no_event_ledger_executed", False)),
        "no_event_ledger_enforcement_relaxed": _boolish(meta.get("no_event_ledger_enforcement_relaxed", False)),
        "diagnostic_readout_policy": meta.get("diagnostic_readout_policy", ""),
        "native_likelihood_scores_path": meta.get("native_likelihood_scores_path", ""),
        "native_likelihood_evidence_log_path": meta.get("native_likelihood_evidence_log_path", ""),
        "true_native_coverage_summary": meta.get("true_native_coverage_summary", ""),
        "blocked_selected_model_count": _safe_int(meta.get("blocked_selected_model_count", 0)),
        "unavailable_selected_model_count": _safe_int(meta.get("unavailable_selected_model_count", 0)),
        "proxy_fallback_used": _boolish(meta.get("proxy_fallback_used", False)),
        "proxy_fallback_row_count": _safe_int(meta.get("proxy_fallback_row_count", 0)),
        "adapter_loglik_row_count": _safe_int(meta.get("adapter_loglik_row_count", 0)),
        "native_likelihoods_comparable": _boolish(meta.get("native_likelihoods_comparable", meta.get("native_likelihoods_compared", False))),
        "topk_selection_stage": meta.get("topk_selection_stage", ""),
        "topk_selection_rerun_per_algorithm": _boolish(meta.get("topk_selection_rerun_per_algorithm", False)),
        "topk_selection_reused_across_algorithms": _boolish(meta.get("topk_selection_reused_across_algorithms", False)),
        "topk_selection_timing_sec": float(meta.get("topk_selection_timing_sec", 0.0) or 0.0),
        "topk_selection_charged_sec": float(meta.get("topk_selection_charged_sec", meta.get("topk_selection_timing_sec", 0.0)) or 0.0),
        "topk_selection_path": meta.get("topk_selection_path", ""),
        "topk_selection_timing_path": meta.get("topk_selection_timing_path", ""),
        "topk_selection_metadata_path": meta.get("topk_selection_metadata_path", ""),
        "topk_selection_algorithm_id": meta.get("topk_selection_algorithm_id", ""),
        "topk_selection_benchmark": meta.get("topk_selection_benchmark", ""),
        "topk_selection_query": meta.get("topk_selection_query", ""),
        "topk_selection_top_k": _safe_int(meta.get("topk_selection_top_k", 0)),
        "topk_selection_selected_model_count": _safe_int(meta.get("topk_selection_selected_model_count", 0)),
        "topk_selection_selected_model_ids": meta.get("topk_selection_selected_model_ids", ""),
    }


def _check_contract(contract: dict[str, Any], source: str) -> None:
    if str(contract["bridge_calibration_split"]) != "val":
        raise SystemExit(f"{source} violates validation-only bridge calibration: {contract['bridge_calibration_split']}")
    if int(contract["test_rows_used_for_bridge_calibration"]) != 0:
        raise SystemExit(f"{source} used test rows for bridge calibration")
    diagnostic = bool(contract.get("diagnostic_only", False))
    if not diagnostic:
        policy = str(contract.get("posterior_update_policy", "holdout_train_val"))
        readout_policy = str(contract.get("posterior_readout_policy", ""))
        if policy in {"", "holdout_train_val"}:
            if int(contract["test_rows_used_for_posterior_update"]) != 0:
                raise SystemExit(f"{source} used test rows for posterior update under holdout policy")
            if readout_policy not in {
                "asof_release_time_lte_forecast_origin",
                alternate_ASOF_AVAILABLE_BEFORE_POLICY,
            }:
                raise SystemExit(f"{source} missing as-of posterior readout policy")
        elif policy in {"prequential_asof_release_lte_origin", "prequential_asof_release_available_before_origin"}:
            if readout_policy not in {
                "asof_release_time_lte_forecast_origin",
                alternate_ASOF_AVAILABLE_BEFORE_POLICY,
            }:
                raise SystemExit(f"{source} missing prequential as-of posterior readout policy")
            if str(contract.get("test_rows_used_for_posterior_update_policy", "")) != "released_evidence_only":
                raise SystemExit(f"{source} missing released-evidence-only posterior update policy")
            if int(contract.get("readout_rows_self_target_update_violation", -1)) != 0:
                raise SystemExit(f"{source} has self-target posterior update violations")
        else:
            raise SystemExit(f"{source} has unknown posterior_update_policy={policy!r}")
        if int(contract.get("readout_rows_future_snapshot_violation", -1)) != 0:
            raise SystemExit(f"{source} has future posterior snapshot readout violations")
    elif bool(contract.get("diagnostic_values_executed", False)):
        if bool(contract.get("unsafe_no_event_ledger_executed", False)):
            if not bool(contract.get("no_event_ledger_enforcement_relaxed", False)):
                raise SystemExit(f"{source} no-event diagnostic missing relaxed-ledger validation flag")
            if int(contract.get("test_rows_used_for_posterior_update", 0)) <= 0:
                raise SystemExit(f"{source} no-event diagnostic did not record test posterior-update rows")
            if int(contract.get("readout_rows_future_snapshot_violation", 0)) <= 0:
                raise SystemExit(f"{source} no-event diagnostic did not record future snapshot readout violations")
            if contract.get("diagnostic_readout_policy") != "latest_snapshot":
                raise SystemExit(f"{source} no-event diagnostic missing latest_snapshot readout policy")
        else:
            if int(contract.get("test_rows_used_for_posterior_update", 0)) != 0:
                raise SystemExit(f"{source} diagnostic used test rows for posterior update without no-event unsafe flag")
            if int(contract.get("readout_rows_future_snapshot_violation", 0)) != 0:
                raise SystemExit(f"{source} diagnostic has readout violations without no-event unsafe flag")
    native_compared = bool(contract.get("native_likelihoods_compared", False))
    adapter_compared = bool(contract.get("adapter_native_likelihoods_compared", False))
    proxy_fallback_used = bool(contract.get("proxy_fallback_used", False))
    native_comparable = bool(contract.get("native_likelihoods_comparable", native_compared))
    if native_compared != adapter_compared and not (adapter_compared and proxy_fallback_used and not native_comparable):
        raise SystemExit(f"{source} has inconsistent native likelihood comparison flags")
    if native_compared or adapter_compared:
        allowed_adapter_native = (
            "ablation_adapter_native_likelihood" in source
            and diagnostic
            and bool(contract.get("unsafe_diagnostic", False))
            and bool(contract.get("not_for_positive_h5_claim", False))
            and not bool(contract.get("used_for_h5_positive_claim", False))
            and bool(contract.get("adapter_native_likelihood_executed", False))
            and contract.get("score_update_basis") in {"adapter_log_likelihood", "adapter_log_likelihood_with_proxy_fallback"}
            and adapter_compared
            and (native_compared or (proxy_fallback_used and not native_comparable))
        )
        if not allowed_adapter_native:
            raise SystemExit(f"{source} compared native likelihoods outside adapter-native unsafe diagnostic contract")


RUN_CONTRACT_FIELDS = [
    "bridge_calibration_split",
    "rho",
    "task_id",
    "target_component",
    "components",
    "posterior_scope",
    "bridge_calibration_scope",
    "n_validation_rows",
    "n_test_rows",
    "candidate_count",
    "variant",
    "predictive_contract",
    "posterior_update_splits",
    "test_rows_used_for_bridge_calibration",
    "test_rows_used_for_posterior_update",
    "posterior_update_policy",
    "posterior_update_scope",
    "test_rows_used_for_posterior_update_policy",
    "posterior_readout_policy",
    "release_availability_rule",
    "readout_rows_future_snapshot_violation",
    "readout_rows_self_target_update_violation",
    "max_stale_posterior_age_days",
    "p95_stale_posterior_age_days",
    "native_likelihoods_compared",
    "adapter_native_likelihoods_compared",
    "adapter_native_likelihood_executed",
    "diagnostic_only",
    "diagnostic_values_executed",
    "used_for_h5_positive_claim",
    "unsafe_diagnostic",
    "not_for_positive_h5_claim",
    "score_update_basis",
    "evaluation_basis",
    "unsafe_native_proxy_executed",
    "unsafe_no_event_ledger_executed",
    "no_event_ledger_enforcement_relaxed",
    "diagnostic_readout_policy",
    "native_likelihood_scores_path",
    "native_likelihood_evidence_log_path",
    "true_native_coverage_summary",
    "blocked_selected_model_count",
    "unavailable_selected_model_count",
    "proxy_fallback_used",
    "proxy_fallback_row_count",
    "adapter_loglik_row_count",
    "native_likelihoods_comparable",
    "topk_selection_stage",
    "topk_selection_rerun_per_algorithm",
    "topk_selection_reused_across_algorithms",
    "topk_selection_timing_sec",
    "topk_selection_charged_sec",
    "topk_selection_path",
    "topk_selection_timing_path",
    "topk_selection_metadata_path",
    "topk_selection_algorithm_id",
    "topk_selection_benchmark",
    "topk_selection_query",
    "topk_selection_top_k",
    "topk_selection_selected_model_count",
    "topk_selection_selected_model_ids",
]


def _merged_ablation_metadata(ablation_meta: dict[str, Any], run_meta: dict[str, Any], source: str) -> dict[str, Any]:
    merged = dict(ablation_meta)
    if not run_meta:
        return merged
    for key in RUN_CONTRACT_FIELDS:
        if key not in ablation_meta or key not in run_meta:
            continue
        if _canonical_meta_value(ablation_meta[key]) != _canonical_meta_value(run_meta[key]):
            raise SystemExit(
                f"{source} ablation_metadata.json conflicts with caster_run_metadata.json for {key}: "
                f"{ablation_meta[key]!r} != {run_meta[key]!r}"
            )
    for key in RUN_CONTRACT_FIELDS:
        if key in run_meta:
            merged[key] = run_meta[key]
    return merged


def _diagnostic_values_executed(meta: dict[str, Any], ablation: str) -> bool:
    if ablation == "native_proxy_gaussian_archive_score":
        return _boolish(meta.get("unsafe_native_proxy_executed", False))
    if ablation == "adapter_native_likelihood":
        return _boolish(meta.get("adapter_native_likelihood_executed", False))
    if ablation == "no_event_ledger_diagnostic":
        return _boolish(meta.get("unsafe_no_event_ledger_executed", False))
    return _boolish(meta.get("diagnostic_values_executed", False))


def _require_executed_diagnostic_artifacts(run_dir: Path, dataset: str, ablation: str) -> None:
    required = [
        "forecast_readout.csv",
        "posterior_weights.csv",
        "timing.json",
        "ablation_metadata.json",
        "caster_run_metadata.json",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise SystemExit(f"{dataset}:ablation_{ablation} missing executed diagnostic artifact(s): {', '.join(missing)}")
    if ablation == "native_proxy_gaussian_archive_score":
        meta = _read_json(run_dir / "ablation_metadata.json")
        run_meta = _read_json(run_dir / "caster_run_metadata.json")
        merged = {**meta, **run_meta}
        if not _boolish(merged.get("unsafe_native_proxy_executed", False)):
            raise SystemExit(f"{dataset}:ablation_{ablation} missing unsafe_native_proxy_executed=true")
        if str(merged.get("score_update_basis", "")) != "native_proxy_gaussian_archive_score":
            raise SystemExit(f"{dataset}:ablation_{ablation} must use score_update_basis=native_proxy_gaussian_archive_score")
        if _boolish(merged.get("native_likelihoods_compared", False)):
            raise SystemExit(f"{dataset}:ablation_{ablation} must not set native_likelihoods_compared")
        if _boolish(merged.get("adapter_native_likelihoods_compared", False)):
            raise SystemExit(f"{dataset}:ablation_{ablation} must not set adapter_native_likelihoods_compared")
    if ablation == "adapter_native_likelihood":
        extra_required = ["evidence_log.csv", "adapter_native_likelihood_blockers.csv"]
        extra_missing = [name for name in extra_required if not (run_dir / name).exists()]
        if extra_missing:
            raise SystemExit(f"{dataset}:ablation_{ablation} missing adapter-native provenance artifact(s): {', '.join(extra_missing)}")
        meta = _read_json(run_dir / "ablation_metadata.json")
        run_meta = _read_json(run_dir / "caster_run_metadata.json")
        merged = {**meta, **run_meta}
        required_nonempty = [
            "native_likelihood_scores_path",
            "native_likelihood_evidence_log_path",
            "true_native_coverage_summary",
        ]
        missing_provenance = [key for key in required_nonempty if not str(merged.get(key, "")).strip()]
        if missing_provenance:
            raise SystemExit(f"{dataset}:ablation_{ablation} missing adapter-native metadata provenance: {', '.join(missing_provenance)}")
        if not _boolish(merged.get("adapter_native_likelihood_executed", False)):
            raise SystemExit(f"{dataset}:ablation_{ablation} missing adapter_native_likelihood_executed=true")
        if not _boolish(merged.get("adapter_native_likelihoods_compared", False)):
            raise SystemExit(f"{dataset}:ablation_{ablation} missing adapter_native_likelihoods_compared=true")
        proxy_fallback_used = _boolish(merged.get("proxy_fallback_used", False))
        native_comparable = _boolish(merged.get("native_likelihoods_comparable", merged.get("native_likelihoods_compared", False)))
        if not proxy_fallback_used and not _boolish(merged.get("native_likelihoods_compared", False)):
            raise SystemExit(f"{dataset}:ablation_{ablation} missing native_likelihoods_compared=true")
        if proxy_fallback_used and native_comparable:
            raise SystemExit(f"{dataset}:ablation_{ablation} proxy fallback must record native_likelihoods_comparable=false")
        if str(merged.get("score_update_basis", "")) not in {"adapter_log_likelihood", "adapter_log_likelihood_with_proxy_fallback"}:
            raise SystemExit(f"{dataset}:ablation_{ablation} must use adapter log-likelihood score basis")
        if _safe_int(merged.get("blocked_selected_model_count", 0), -1) != 0:
            raise SystemExit(f"{dataset}:ablation_{ablation} has blocked selected models")
        if _safe_int(merged.get("unavailable_selected_model_count", 0), -1) != 0:
            raise SystemExit(f"{dataset}:ablation_{ablation} has unavailable selected models")
        evidence = pd.read_csv(run_dir / "evidence_log.csv")
        required_cols = {
            "score_update_basis",
            "native_likelihood_scores_path",
            "native_likelihood_evidence_log_path",
            "true_native_coverage_summary",
            "blocked_selected_model_count",
            "unavailable_selected_model_count",
            "proxy_fallback_used",
            "native_likelihoods_comparable",
        }
        missing_cols = sorted(required_cols - set(evidence.columns))
        if missing_cols:
            raise SystemExit(f"{dataset}:ablation_{ablation} evidence_log.csv missing adapter-native provenance columns: {missing_cols}")
        allowed_basis = {"adapter_log_likelihood", "adapter_log_likelihood_with_proxy_fallback"}
        evidence_basis = evidence["score_update_basis"].astype(str)
        if not evidence_basis.isin(allowed_basis).all():
            raise SystemExit(
                f"{dataset}:ablation_{ablation} evidence_log.csv must use adapter_log_likelihood "
                "or adapter_log_likelihood_with_proxy_fallback"
            )


def _missing_nll() -> dict[str, Any]:
    return {
        "nll": math.nan,
        "nll_status": "not_available",
        "nll_reason": "No comparable NLL export in this tracker row; native likelihood comparison remains disabled.",
    }


def _interval_score(y: pd.Series, lower: pd.Series, upper: pd.Series, alpha: float) -> pd.Series:
    yy = y.astype(float)
    lo = lower.astype(float)
    hi = upper.astype(float)
    return (hi - lo) + (2.0 / alpha) * (lo - yy) * (yy < lo) + (2.0 / alpha) * (yy - hi) * (yy > hi)


def _weighted_interval_score_50_90(df: pd.DataFrame) -> float:
    center_column = (
        "predictive_median"
        if "predictive_median" in df.columns
        else "predictive_mean"
    )
    rows = df.dropna(subset=["observed_value", center_column, "lower_50", "upper_50", "lower_90", "upper_90"]).copy()
    if rows.empty:
        return math.nan
    numerator = (
        0.5 * (rows["observed_value"].astype(float) - rows[center_column].astype(float)).abs()
        + 0.50 / 2.0 * _interval_score(rows["observed_value"], rows["lower_50"], rows["upper_50"], 0.50)
        + 0.10 / 2.0 * _interval_score(rows["observed_value"], rows["lower_90"], rows["upper_90"], 0.10)
    )
    return float((numerator / 2.5).mean())


def _with_contract_intervals(readout: pd.DataFrame) -> pd.DataFrame:
    out = readout.copy()
    if "predictive_contract" not in out.columns:
        predictive_contract = alternate_ARCHIVE_MOMENT
    else:
        contracts = sorted(
            value
            for value in out["predictive_contract"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            if value
        )
        if len(contracts) != 1 or contracts[0] not in PREDICTIVE_CONTRACTS:
            raise SystemExit(
                "forecast readout must contain exactly one supported "
                f"predictive_contract; found={contracts!r}"
            )
        predictive_contract = contracts[0]
    if predictive_contract != alternate_ARCHIVE_MOMENT:
        quantile_columns = (
            "lower_50",
            "upper_50",
            "lower_90",
            "upper_90",
            "lower_95",
            "upper_95",
        )
        missing = sorted(set(quantile_columns) - set(out.columns))
        if missing:
            raise SystemExit(
                "nonalternate forecast readout is missing bridge-mixture quantiles: "
                f"{missing}"
            )
        if "predictive_interval_source" not in out.columns:
            raise SystemExit(
                "nonalternate forecast readout is missing predictive_interval_source"
            )
        sources = sorted(
            value
            for value in out["predictive_interval_source"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            if value
        )
        if len(sources) != 1:
            raise SystemExit(
                "nonalternate forecast readout must declare exactly one "
                f"predictive_interval_source; found={sources!r}"
            )
        numeric = out[list(quantile_columns)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        finite = numeric.apply(lambda column: column.map(math.isfinite))
        if not finite.all(axis=None):
            raise SystemExit("nonalternate bridge-mixture quantiles must be finite")
        if (
            (numeric["lower_50"] > numeric["upper_50"]).any()
            or (numeric["lower_90"] > numeric["upper_90"]).any()
            or (numeric["lower_95"] > numeric["upper_95"]).any()
        ):
            raise SystemExit("nonalternate bridge-mixture quantile bounds are inverted")
                                                                               
                                                                              
        return out
    sigma = out["predictive_var"].astype(float).clip(lower=0.0).pow(0.5)
    out["lower_50"] = (out["predictive_mean"].astype(float) - Z50 * sigma).clip(lower=0.0)
    out["upper_50"] = (out["predictive_mean"].astype(float) + Z50 * sigma).clip(lower=0.0)
    out["lower_90"] = (out["predictive_mean"].astype(float) - Z90 * sigma).clip(lower=0.0)
    out["upper_90"] = (out["predictive_mean"].astype(float) + Z90 * sigma).clip(lower=0.0)
    return out


def _with_protocol_columns(readout: pd.DataFrame, *, dataset: str, source: Path) -> pd.DataFrame:
    out = readout.copy()
    missing_protocol: list[str] = []
    if "jurisdiction" not in out.columns and "entity_id" in out.columns:
        out["jurisdiction"] = out["entity_id"]
    if dataset == "benchmark_a" and "country" not in out.columns and "entity_id" in out.columns:
        out["country"] = out["entity_id"]
    for col in PROTOCOL_COLUMNS:
        if col not in out.columns:
            out[col] = "unknown"
            missing_protocol.append(col)
    for col in ASOF_COLUMNS:
        if col not in out.columns:
            raise SystemExit(f"{source} missing post-fix as-of column {col}")
    out["protocol_slice_status"] = "ok" if not missing_protocol else "missing_protocol_fields"
    out["protocol_slice_reason"] = "" if not missing_protocol else "missing: " + ",".join(sorted(missing_protocol))
    return out


def _column_for(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lower = {str(col).lower(): str(col) for col in df.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def _normalize_required_column(df: pd.DataFrame, aliases: list[str], canonical: str, label: str) -> pd.DataFrame:
    col = _column_for(df, aliases)
    if col is None:
        raise SystemExit(f"{label} missing required column; accepted aliases={aliases}")
    out = df.copy()
    if col != canonical:
        out[canonical] = out[col]
    return out


def _normalize_optional_column(df: pd.DataFrame, aliases: list[str], canonical: str) -> pd.DataFrame:
    col = _column_for(df, aliases)
    out = df.copy()
    if col is not None and col != canonical:
        out[canonical] = out[col]
    return out


def _normalize_forecast_ids_and_origins(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = _normalize_required_column(df, FORECAST_ID_ALIASES, "forecast_id", label)
    out = _normalize_required_column(out, FORECAST_ORIGIN_ALIASES, "forecast_origin", label)
    out["forecast_id"] = out["forecast_id"].astype(str)
    out["forecast_origin"] = pd.to_datetime(out["forecast_origin"], errors="coerce")
    if out["forecast_origin"].isna().any():
        raise SystemExit(f"{label} contains invalid forecast_origin values")
    return out


def _normalize_model_weights(df: pd.DataFrame, label: str, *, require_snapshot: bool = False) -> pd.DataFrame:
    out = _normalize_required_column(df, MODEL_ID_ALIASES, "model_id", label)
    out = _normalize_required_column(out, WEIGHT_ALIASES, "weight", label)
    out = _normalize_optional_column(out, FAMILY_ALIASES, "family")
    out = _normalize_optional_column(out, SNAPSHOT_TIME_ALIASES, "snapshot_time")
    if require_snapshot and "snapshot_time" not in out.columns:
        raise SystemExit(f"{label} missing snapshot time column; accepted aliases={SNAPSHOT_TIME_ALIASES}")
    out["model_id"] = out["model_id"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    if out["weight"].isna().any():
        raise SystemExit(f"{label} contains invalid weight values")
    if "family" not in out.columns:
        out["family"] = ""
    else:
        out["family"] = out["family"].astype(str)
    if "snapshot_time" in out.columns:
        out["snapshot_time"] = pd.to_datetime(out["snapshot_time"], errors="coerce")
        if out["snapshot_time"].isna().any():
            raise SystemExit(f"{label} contains invalid snapshot time values")
    return out


def _normalize_family_weights(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = _normalize_required_column(df, FAMILY_ALIASES, "family", label)
    out = _normalize_required_column(out, FAMILY_WEIGHT_ALIASES, "family_weight", label)
    out = _normalize_required_column(out, SNAPSHOT_TIME_ALIASES, "snapshot_time", label)
    out["family"] = out["family"].astype(str)
    out["family_weight"] = pd.to_numeric(out["family_weight"], errors="coerce")
    out["snapshot_time"] = pd.to_datetime(out["snapshot_time"], errors="coerce")
    if out["family_weight"].isna().any() or out["snapshot_time"].isna().any():
        raise SystemExit(f"{label} contains invalid family_weight or snapshot_time values")
    return out


def _normalize_inner_weights(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = _normalize_required_column(df, FAMILY_ALIASES, "family", label)
    out = _normalize_required_column(out, MODEL_ID_ALIASES, "model_id", label)
    out = _normalize_required_column(out, INNER_WEIGHT_ALIASES, "inner_weight", label)
    out = _normalize_required_column(out, SNAPSHOT_TIME_ALIASES, "snapshot_time", label)
    out["family"] = out["family"].astype(str)
    out["model_id"] = out["model_id"].astype(str)
    out["inner_weight"] = pd.to_numeric(out["inner_weight"], errors="coerce")
    out["snapshot_time"] = pd.to_datetime(out["snapshot_time"], errors="coerce")
    if out["inner_weight"].isna().any() or out["snapshot_time"].isna().any():
        raise SystemExit(f"{label} contains invalid inner_weight or snapshot_time values")
    return out


def _normalize_weights_sum(weights: pd.DataFrame, label: str) -> pd.DataFrame:
    out = weights.copy()
    total = float(pd.to_numeric(out["weight"], errors="coerce").sum())
    if not math.isfinite(total) or total <= 0.0:
        raise SystemExit(f"{label} has non-positive total weight")
    out["weight"] = out["weight"].astype(float) / total
    return out


def _normalize_weights_by_group(weights: pd.DataFrame, group_cols: list[str], label: str) -> pd.DataFrame:
    out = weights.copy()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    totals = out.groupby(group_cols, dropna=False)["weight"].transform("sum")
    bad = out["weight"].isna() | totals.isna() | (totals <= 0.0)
    if bad.any():
        raise SystemExit(f"{label} has invalid or non-positive grouped weights")
    out["weight"] = out["weight"].astype(float) / totals.astype(float)
    return out


def _selected_registry(root: Path, selection_path: Path | None = None) -> pd.DataFrame:
    candidates = [selection_path, root / "candidate_selection_log.csv", root / "model_registry.csv"]
    for path in candidates:
        if path is None or not path.exists():
            continue
        df = pd.read_csv(path)
        df = _normalize_required_column(df, MODEL_ID_ALIASES, "model_id", str(path))
        df = _normalize_optional_column(df, FAMILY_ALIASES, "family")
        if "family" not in df.columns and (root / "model_registry.csv").exists() and path != root / "model_registry.csv":
            registry = pd.read_csv(root / "model_registry.csv")
            registry = _normalize_required_column(registry, MODEL_ID_ALIASES, "model_id", str(root / "model_registry.csv"))
            registry = _normalize_optional_column(registry, FAMILY_ALIASES, "family")
            if "family" in registry.columns:
                df = df.merge(registry[["model_id", "family"]].drop_duplicates("model_id"), on="model_id", how="left")
        if "family" not in df.columns:
            df["family"] = "unknown"
        df["model_id"] = df["model_id"].astype(str)
        df["family"] = df["family"].fillna("unknown").astype(str)
        return df[["model_id", "family"]].drop_duplicates("model_id")
    raise SystemExit(f"{root} missing selected model registry for prior reconstruction")


def _one_layer_initial_prior(root: Path, selection_path: Path | None) -> tuple[pd.DataFrame, str]:
    prior_path = root / "initial_prior.csv"
    if prior_path.exists():
        prior = _normalize_model_weights(pd.read_csv(prior_path), str(prior_path), require_snapshot=False)
        return _normalize_weights_sum(prior[["model_id", "family", "weight"]], str(prior_path)), str(prior_path)
    selected = _selected_registry(root, selection_path)
    n = len(selected)
    if n <= 0:
        raise SystemExit(f"{root} has no selected models for uniform prior reconstruction")
    selected = selected.copy()
    selected["weight"] = 1.0 / float(n)
    return selected[["model_id", "family", "weight"]], "uniform_selected_topk_reconstructed"


def _hierarchical_initial_prior(root: Path, selection_path: Path | None) -> tuple[pd.DataFrame, str]:
    registry = _selected_registry(root, selection_path)
    prior = initialize_hierarchical_weights(registry).model_weights
    return prior[["model_id", "family", "weight"]].copy(), "initialize_hierarchical_weights_from_selected_registry"


def _one_layer_snapshot_weights(root: Path, weights_path: Path | None = None) -> tuple[pd.DataFrame, str]:
    candidates: list[Path | None] = []
    if weights_path is not None and weights_path.name != "posterior_weights.csv":
        candidates.append(weights_path)
    candidates.extend([root / "posterior_path.csv", root / "posterior_weights.csv"])
    if weights_path is not None and weights_path.name == "posterior_weights.csv":
        candidates.append(weights_path)
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        df = pd.read_csv(path)
        if _column_for(df, SNAPSHOT_TIME_ALIASES) is None:
            continue
        weights = _normalize_model_weights(df, str(path), require_snapshot=True)
        return weights[["snapshot_time", "model_id", "family", "weight"]].copy(), str(path)
    return pd.DataFrame(columns=["snapshot_time", "model_id", "family", "weight"]), "no_one_layer_posterior_snapshot_available"


def _hierarchical_snapshot_weights(root: Path) -> tuple[pd.DataFrame, str]:
    family_path = root / "family_posterior.csv"
    inner_path = root / "inner_weights.csv"
    if not family_path.exists() and not inner_path.exists():
        return pd.DataFrame(columns=["snapshot_time", "model_id", "family", "weight"]), (
            "no_hierarchical_posterior_snapshot_available"
        )
    if not family_path.exists() or not inner_path.exists():
        raise SystemExit(f"{root} missing hierarchical family_posterior.csv or inner_weights.csv")
    family = _normalize_family_weights(pd.read_csv(family_path), str(family_path))
    inner = _normalize_inner_weights(pd.read_csv(inner_path), str(inner_path))
    merged = inner.merge(
        family[["snapshot_time", "family", "family_weight"]],
        on=["snapshot_time", "family"],
        how="inner",
    )
    if merged.empty:
        raise SystemExit(f"{root} hierarchical snapshots do not align by family and snapshot time")
    merged["weight"] = merged["inner_weight"].astype(float) * merged["family_weight"].astype(float)
    merged = merged.groupby(["snapshot_time", "model_id", "family"], as_index=False)["weight"].sum()
    merged = _normalize_weights_by_group(merged, ["snapshot_time"], f"{family_path}+{inner_path}")
    return merged[["snapshot_time", "model_id", "family", "weight"]].reset_index(drop=True), f"{family_path};{inner_path}"


def _asof_search_side(policy: str) -> str:
    if policy == ASOF_LTE_POLICY:
        return "right"
    raise SystemExit(f"unsupported posterior_readout_policy for formal mixture scoring: {policy!r}")


def _asof_weights_for_forecasts(
    *,
    test: pd.DataFrame,
    root: Path,
    dataset: str,
    method: str,
    policy: str,
    hierarchical: bool,
    selection_path: Path | None = None,
    weights_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_norm = _normalize_forecast_ids_and_origins(test, f"{method}:test ledger")
    if hierarchical:
        snapshots, snapshot_source = _hierarchical_snapshot_weights(root)
        prior, prior_source = _hierarchical_initial_prior(root, selection_path)
    else:
        snapshots, snapshot_source = _one_layer_snapshot_weights(root, weights_path)
        prior, prior_source = _one_layer_initial_prior(root, selection_path)
    if selection_path is not None and selection_path.exists():
        selected_df = _normalize_required_column(pd.read_csv(selection_path), MODEL_ID_ALIASES, "model_id", str(selection_path))
        selected = set(selected_df["model_id"].astype(str))
        prior = prior[prior["model_id"].astype(str).isin(selected)].copy()
        snapshots = snapshots[snapshots["model_id"].astype(str).isin(selected)].copy()
    prior = _normalize_weights_sum(prior[["model_id", "family", "weight"]], f"{method}:initial prior")
    if snapshots.empty:
        snapshot_times = pd.Series([], dtype="datetime64[ns]").to_numpy(dtype="datetime64[ns]")
        positions = pd.Series([-1] * len(test_norm), index=test_norm.index)
    else:
        snapshots = _normalize_weights_by_group(snapshots, ["snapshot_time"], method).reset_index(drop=True)
        snapshots = snapshots.copy()
        snapshots["snapshot_time"] = pd.to_datetime(snapshots["snapshot_time"], errors="coerce")
        snapshot_times = pd.Series(snapshots["snapshot_time"].dropna().unique()).sort_values().to_numpy(dtype="datetime64[ns]")
        if snapshot_times.size == 0:
            positions = pd.Series([-1] * len(test_norm), index=test_norm.index)
        else:
            positions = pd.Series(
                pd.Series(snapshot_times).searchsorted(
                    test_norm["forecast_origin"].to_numpy(dtype="datetime64[ns]"),
                    side=_asof_search_side(policy),
                )
                - 1,
                index=test_norm.index,
            )
    weight_rows: list[pd.DataFrame] = []
    validation_rows: list[dict[str, Any]] = []
    for idx, row in test_norm.drop_duplicates("forecast_id").iterrows():
        fid = str(row["forecast_id"])
        origin = pd.Timestamp(row["forecast_origin"])
        pos = int(positions.loc[idx])
        used_prior = pos < 0
        if used_prior:
            weights = prior.copy()
            selected_snapshot_time: pd.Timestamp | pd.NaT = pd.NaT
            stale_age_days = math.nan
            source = ""
        else:
            selected_snapshot_time = pd.Timestamp(snapshot_times[pos])
            weights = snapshots[snapshots["snapshot_time"].eq(selected_snapshot_time)][["model_id", "family", "weight"]].copy()
            weights = _normalize_weights_sum(weights, f"{method}:{selected_snapshot_time}")
            stale_age_days = float((origin - selected_snapshot_time).total_seconds() / 86400.0)
            source = snapshot_source
        weights = weights.copy()
        weights["forecast_id"] = fid
        weight_rows.append(weights[["forecast_id", "model_id", "family", "weight"]])
        validation_rows.append(
            {
                "dataset": str(dataset),
                "method": method,
                "forecast_id": fid,
                "forecast_origin": origin.isoformat(),
                "target_component": str(row.get("target_component", row.get("component", ""))),
                "selected_snapshot_time": "" if pd.isna(selected_snapshot_time) else pd.Timestamp(selected_snapshot_time).isoformat(),
                "policy": policy,
                "used_prior": bool(used_prior),
                "stale_age_days": stale_age_days,
                "selected_model_count": int(weights["model_id"].nunique()),
                "prior_source": prior_source if used_prior else "",
                "snapshot_source": source,
                "validation_status": "PASS",
            }
        )
    all_weights = pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame()
    validation = pd.DataFrame(validation_rows)
    return all_weights, validation


def _logsumexp(values: pd.Series) -> float:
    arr = [float(value) for value in values]
    if not arr:
        return float("-inf")
    if any(math.isnan(value) for value in arr):
        raise ValueError("log-sum-exp input contains NaN")
    m = max(arr)
    if m == float("-inf") or m == float("inf"):
        return m
    return float(m + math.log(sum(math.exp(value - m) for value in arr)))


def _exact_log_probability_weight(value: float) -> float:
    ""

    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("mixture weights must be finite and nonnegative")
    return float("-inf") if weight == 0.0 else math.log(weight)


def _bridge_asof_mixture_scores(
    *,
    dataset: str,
    method: str,
    ledger_path: Path,
    archive_path: Path,
    bridge_path: Path,
    root: Path,
    policy: str,
    hierarchical: bool = False,
    selection_path: Path | None = None,
    weights_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bridge, rho = read_bridge_config(bridge_path)
    if rho is None:
        raise SystemExit(f"{bridge_path} must contain frozen rho from validation-only NEW-BRIDGE")
    ledger = pd.read_csv(ledger_path)
    if "split" not in ledger.columns:
        raise SystemExit(f"{ledger_path} missing split column")
    test = ledger[ledger["split"].astype(str) == "test"].copy()
    test = test[test.get("observed_mask", True).astype(bool)].copy()
    test = _normalize_forecast_ids_and_origins(test, str(ledger_path))
    archive = pd.read_csv(archive_path)
    archive = _normalize_required_column(archive, MODEL_ID_ALIASES, "model_id", str(archive_path))
    archive = _normalize_optional_column(archive, FORECAST_ORIGIN_ALIASES, "forecast_origin")
    archive["model_id"] = archive["model_id"].astype(str)
    weights, validation = _asof_weights_for_forecasts(
        test=test,
        root=root,
        dataset=dataset,
        method=method,
        policy=policy,
        hierarchical=hierarchical,
        selection_path=selection_path,
        weights_path=weights_path,
    )
    if selection_path is not None and selection_path.exists():
        selected_df = _normalize_required_column(pd.read_csv(selection_path), MODEL_ID_ALIASES, "model_id", str(selection_path))
        selected = set(selected_df["model_id"].astype(str))
        archive = archive[archive["model_id"].astype(str).isin(selected)].copy()
        weights = weights[weights["model_id"].astype(str).isin(selected)].copy()
        weights = _normalize_weights_by_group(weights, ["forecast_id"], f"{dataset}/{method}").reset_index(drop=True)
    archive = archive[archive["model_id"].astype(str).isin(set(weights["model_id"].astype(str)))].copy()
    archive = native_forecast_rows(
        archive, require_provenance="forecast_fallback_used" in archive.columns
    )
    native_pairs = archive[["forecast_id", "model_id"]].drop_duplicates()
    weights = weights.merge(
        native_pairs, on=["forecast_id", "model_id"], how="inner"
    )
    weights = _normalize_weights_by_group(weights, ["forecast_id"], f"{dataset}/{method}:native").reset_index(drop=True)
    scored = score_archive_rows(test, archive, bridge)
    if scored.empty:
        raise SystemExit(f"no bridge score rows for {dataset}/{method}")
    scored = scored[scored["observed_mask"].astype(bool)].copy()
    scored["forecast_id"] = scored["forecast_id"].astype(str)
    scored["model_id"] = scored["model_id"].astype(str)
    scored = scored.merge(weights[["forecast_id", "model_id", "weight"]], on=["forecast_id", "model_id"], how="inner")
    if scored.empty:
        raise SystemExit(f"no bridge score rows after as-of weight join for {dataset}/{method}")
    particle_counts = scored.groupby(["forecast_id", "model_id"])["particle_id"].transform("nunique").astype(float)
    scored["mixture_weight"] = scored["weight"].astype(float) / particle_counts
    scored["log_term"] = scored["log_score"].astype(float) + scored["mixture_weight"].map(_exact_log_probability_weight)
    mixture = scored.groupby("forecast_id", as_index=False)["log_term"].agg(_logsumexp).rename(columns={"log_term": "bridge_log_score"})
    expected_forecast_ids = set(test["forecast_id"].astype(str))
    scored_forecast_ids = set(mixture["forecast_id"].astype(str))
    missing_forecast_ids = sorted(expected_forecast_ids - scored_forecast_ids)
    if missing_forecast_ids:
        raise SystemExit(
            f"exact as-of mixture bridge NLL unavailable for {dataset}/{method}; missing forecast_ids={missing_forecast_ids[:8]}"
        )
    meta_cols = [
        "forecast_id",
        *PROTOCOL_COLUMNS,
        "forecast_origin",
        "target_time",
        "component",
        "horizon",
        "observed_value",
        "observed_mask",
        "split",
    ]
    meta = test[[c for c in meta_cols if c in test.columns]].drop_duplicates("forecast_id")
    out = meta.merge(mixture, on="forecast_id", how="inner")
    out.insert(0, "method", method)
    out.insert(0, "dataset", dataset)
    out["bridge_nll"] = -out["bridge_log_score"].astype(float)
    out["nll_score_basis"] = EXACT_NLL_SCORE_BASIS
    out["formal_nll_status"] = "formal_exact_asof_posterior_mixture_bridge"
    out["nll_source_kind"] = "exact_asof_posterior_mixture_bridge"
    out["probability_score_basis"] = "exact_asof_posterior_mixture_bridge"
    out["posterior_readout_policy"] = policy
    return out, validation


def _draw_kernel_asof_mixture_scores(
    *,
    dataset: str,
    method: str,
    ledger_path: Path,
    draws_path: Path,
    archive_path: Path,
    bridge_path: Path,
    root: Path,
    policy: str,
    selection_path: Path | None = None,
    weights_path: Path | None = None,
    hierarchical: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bridge, rho = read_bridge_config(bridge_path)
    if rho is None:
        raise SystemExit(f"{bridge_path} must contain frozen one-layer rho")
    ledger = pd.read_csv(ledger_path)
    if "split" not in ledger.columns:
        raise SystemExit(f"{ledger_path} missing split column")
    test = ledger[ledger["split"].astype(str) == "test"].copy()
    test = test[test.get("observed_mask", True).astype(bool)].copy()
    test = _normalize_forecast_ids_and_origins(test, str(ledger_path))
    draws = pd.read_csv(draws_path)
    draws = _normalize_required_column(draws, MODEL_ID_ALIASES, "model_id", str(draws_path))
    draws["model_id"] = draws["model_id"].astype(str)
    availability_archive = pd.read_csv(archive_path)
    availability_archive = _normalize_required_column(availability_archive, MODEL_ID_ALIASES, "model_id", str(archive_path))
    availability_archive = native_forecast_rows(availability_archive, require_provenance=True)
    native_pairs = availability_archive[["forecast_id", "model_id"]].drop_duplicates()
    draws = draws.merge(native_pairs, on=["forecast_id", "model_id"], how="inner")
    weights, validation = _asof_weights_for_forecasts(
        test=test,
        root=root,
        dataset=dataset,
        method=method,
        policy=policy,
        hierarchical=hierarchical,
        selection_path=selection_path,
        weights_path=weights_path,
    )
    if selection_path is not None and selection_path.exists():
        selected_df = _normalize_required_column(pd.read_csv(selection_path), MODEL_ID_ALIASES, "model_id", str(selection_path))
        selected = set(selected_df["model_id"].astype(str))
        draws = draws[draws["model_id"].isin(selected)].copy()
        weights = weights[weights["model_id"].astype(str).isin(selected)].copy()
    weights = _normalize_weights_by_group(weights, ["forecast_id"], f"{dataset}/{method}").reset_index(drop=True)
    weights = weights.merge(
        native_pairs, on=["forecast_id", "model_id"], how="inner"
    )
    weights = _normalize_weights_by_group(weights, ["forecast_id"], f"{dataset}/{method}:native").reset_index(drop=True)
    scored = score_draw_rows(test, draws, bridge)
    if scored.empty:
        raise SystemExit(f"no draw-kernel score rows for {dataset}/{method}")
    scored = scored[scored["observed_mask"].astype(bool)].copy()
    scored["forecast_id"] = scored["forecast_id"].astype(str)
    scored["model_id"] = scored["model_id"].astype(str)
    scored = scored.merge(weights[["forecast_id", "model_id", "weight"]], on=["forecast_id", "model_id"], how="inner")
    if scored.empty:
        raise SystemExit(f"no draw-kernel rows after as-of weight join for {dataset}/{method}")
    particle_counts = scored.groupby(["forecast_id", "model_id"])["particle_id"].transform("nunique").astype(float)
    scored["mixture_weight"] = scored["weight"].astype(float) / particle_counts
    scored["log_term"] = scored["log_score"].astype(float) + scored["mixture_weight"].map(_exact_log_probability_weight)
    mixture = scored.groupby("forecast_id", as_index=False)["log_term"].agg(_logsumexp).rename(columns={"log_term": "bridge_log_score"})
    expected = set(test["forecast_id"].astype(str))
    missing = sorted(expected - set(mixture["forecast_id"].astype(str)))
    if missing:
        raise SystemExit(f"draw-kernel mixture NLL unavailable for {dataset}/{method}; missing forecast_ids={missing[:8]}")
    meta_cols = [
        "forecast_id", *PROTOCOL_COLUMNS, "forecast_origin", "target_time", "component", "horizon",
        "observed_value", "observed_mask", "split",
    ]
    meta = test[[c for c in meta_cols if c in test.columns]].drop_duplicates("forecast_id")
    out = meta.merge(mixture, on="forecast_id", how="inner")
    out.insert(0, "method", method)
    out.insert(0, "dataset", dataset)
    out["bridge_nll"] = -out["bridge_log_score"].astype(float)
    out["nll_score_basis"] = EXACT_DRAW_KERNEL_NLL_SCORE_BASIS
    out["formal_nll_status"] = "formal_exact_asof_posterior_draw_kernel_mixture"
    out["nll_source_kind"] = EXACT_DRAW_KERNEL_NLL_SCORE_BASIS
    out["probability_score_basis"] = EXACT_DRAW_KERNEL_NLL_SCORE_BASIS
    out["posterior_readout_policy"] = policy
    return out, validation


def _bridge_readout_scores(
    *,
    dataset: str,
    method: str,
    ledger_path: Path,
    readout_path: Path,
    bridge_path: Path,
) -> pd.DataFrame:
    bridge, rho = read_bridge_config(bridge_path)
    if rho is None:
        raise SystemExit(f"{bridge_path} must contain frozen rho from validation-only NEW-BRIDGE")
    ledger = pd.read_csv(ledger_path)
    readout = _with_protocol_columns(_read_csv(readout_path), dataset=dataset, source=readout_path)
    readout = _normalize_optional_column(readout, FORECAST_ORIGIN_ALIASES, "forecast_origin")
    if "split" in readout.columns:
        readout = readout[readout["split"].astype(str) == "test"].copy()
    if "forecast_id" in ledger.columns and "split" in ledger.columns:
        test_ids = set(ledger.loc[ledger["split"].astype(str) == "test", "forecast_id"].astype(str))
        readout = readout[readout["forecast_id"].astype(str).isin(test_ids)].copy()
    readout = readout[readout.get("observed_mask", True).astype(bool)].copy()
    archive = readout[["forecast_id", "component", "horizon", "predictive_mean", "predictive_var"]].copy()
    archive["model_id"] = method
    archive["particle_id"] = 0
    archive = archive.rename(columns={"predictive_mean": "pred_mean", "predictive_var": "pred_var"})
    scored = score_archive_rows(readout, archive, bridge)
    if scored.empty:
        raise SystemExit(f"no bridge score rows for {dataset}/{method}")
    scores = scored[["forecast_id", "log_score"]].rename(columns={"log_score": "bridge_log_score"})
    meta_cols = ["forecast_id", *PROTOCOL_COLUMNS, "forecast_origin", "target_time", "component", "horizon", "observed_value", "observed_mask", *ASOF_COLUMNS, "split"]
    meta = readout[[c for c in meta_cols if c in readout.columns]].drop_duplicates("forecast_id")
    out = meta.merge(scores, on="forecast_id", how="inner")
    out.insert(0, "method", method)
    out.insert(0, "dataset", dataset)
    out = out.rename(columns={"bridge_log_score": "diagnostic_moment_matched_log_score"})
    out["diagnostic_moment_matched_nll"] = -out["diagnostic_moment_matched_log_score"].astype(float)
    out["diagnostic_moment_matched_nll_available"] = True
    out["diagnostic_score_basis"] = DIAGNOSTIC_MOMENT_MATCHED_BASIS
    return out


def _with_diagnostic_scores(formal_scores: pd.DataFrame, diagnostic_scores: pd.DataFrame) -> pd.DataFrame:
    diag_cols = [
        "forecast_id",
        "diagnostic_moment_matched_log_score",
        "diagnostic_moment_matched_nll",
        "diagnostic_moment_matched_nll_available",
        "diagnostic_score_basis",
    ]
    if diagnostic_scores.empty:
        out = formal_scores.copy()
        out["diagnostic_moment_matched_nll_available"] = False
        out["diagnostic_moment_matched_nll"] = math.nan
        return out
    out = formal_scores.merge(diagnostic_scores[[c for c in diag_cols if c in diagnostic_scores.columns]], on="forecast_id", how="left")
    if "diagnostic_moment_matched_nll_available" in out.columns:
        out["diagnostic_moment_matched_nll_available"] = out["diagnostic_moment_matched_nll_available"].fillna(False).astype(bool)
    return out


def _blocked_scores_from_diagnostic(diagnostic_scores: pd.DataFrame, reason: str) -> pd.DataFrame:
    out = diagnostic_scores.copy()
    out["bridge_log_score"] = math.nan
    out["bridge_nll"] = math.nan
    out["nll_score_basis"] = BLOCKED_EXACT_NLL_SCORE_BASIS
    out["formal_nll_status"] = "blocked_missing_exact_asof_posterior_mixture_bridge"
    out["nll_source_kind"] = "diagnostic_moment_matched_blocked"
    out["probability_score_basis"] = DIAGNOSTIC_MOMENT_MATCHED_BASIS
    out["nll_blocker_reason"] = reason
    return out


def _metric_frame(
    dataset: str,
    method: str,
    method_group: str,
    readout_path: Path,
    scores: pd.DataFrame,
    contract: dict[str, Any],
    *,
    nll_status: str = "ok",
    nll_reason: str = "exact as-of posterior-mixture bridge evaluated on test rows only",
    asof_weight_validation_path: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contract = {key: _contract_scalar(value) for key, value in contract.items()}
    def _single_score_value(column: str, default: str) -> str:
        if column not in scores.columns:
            return default
        values = [str(value) for value in scores[column].dropna().astype(str).unique() if str(value)]
        if len(values) > 1:
            raise SystemExit(f"{dataset}/{method} has mixed {column}: {values}")
        return values[0] if values else default

    formal_basis = _single_score_value("nll_score_basis", EXACT_NLL_SCORE_BASIS)
    formal_status = _single_score_value("formal_nll_status", "formal_exact_asof_posterior_mixture_bridge")
    formal_source_kind = _single_score_value("nll_source_kind", EXACT_NLL_SCORE_BASIS)
    probability_basis = _single_score_value("probability_score_basis", EXACT_NLL_SCORE_BASIS)
    readout = _with_protocol_columns(_with_contract_intervals(_read_csv(readout_path)), dataset=dataset, source=readout_path)
    readout = readout[readout.get("observed_mask", True).astype(bool)].copy()
    scores = scores.copy()
    for col in ["bridge_log_score", "bridge_nll"]:
        if col not in scores.columns:
            scores[col] = math.nan
    score_cols = [
        "forecast_id",
        "bridge_log_score",
        "bridge_nll",
        "nll_score_basis",
        "formal_nll_status",
        "nll_source_kind",
        "probability_score_basis",
        "diagnostic_moment_matched_nll_available",
        "diagnostic_moment_matched_nll",
    ]
    readout = readout.merge(scores[[c for c in score_cols if c in scores.columns]], on="forecast_id", how="inner")
    readout["dataset"] = dataset
    readout["method"] = method
    readout["method_group"] = method_group
    readout["split"] = readout["split"].astype(str)
    for col, default in [
        ("nll_score_basis", EXACT_NLL_SCORE_BASIS if pd.to_numeric(readout["bridge_nll"], errors="coerce").notna().any() else BLOCKED_EXACT_NLL_SCORE_BASIS),
        ("formal_nll_status", "formal_exact_asof_posterior_mixture_bridge" if pd.to_numeric(readout["bridge_nll"], errors="coerce").notna().any() else "blocked_missing_exact_asof_posterior_mixture_bridge"),
        ("nll_source_kind", "exact_asof_posterior_mixture_bridge" if pd.to_numeric(readout["bridge_nll"], errors="coerce").notna().any() else "missing_exact_asof_posterior_mixture_bridge"),
        ("probability_score_basis", "exact_asof_posterior_mixture_bridge" if pd.to_numeric(readout["bridge_nll"], errors="coerce").notna().any() else DIAGNOSTIC_MOMENT_MATCHED_BASIS),
        ("diagnostic_moment_matched_nll_available", False),
        ("diagnostic_moment_matched_nll", math.nan),
    ]:
        if col not in readout.columns:
            readout[col] = default
    slice_metrics = metric_slices_from_scored_rows(
        readout,
        source=readout_path,
        y_col="observed_value",
        pred_col="predictive_mean",
        median_col=(
            "predictive_median"
            if "predictive_median" in readout.columns
            else None
        ),
        lower_50_col="lower_50",
        upper_50_col="upper_50",
        lower_90_col="lower_90",
        upper_90_col="upper_90",
        nll_col="bridge_nll",
        method_group=method_group,
    )
    contracted_readout = apply_result_metric_contract(readout, method_group=method_group)
    diag_grouped = contracted_readout.groupby(RESULT_GROUP_COLS, dropna=False).agg(
        diagnostic_moment_matched_nll=("diagnostic_moment_matched_nll", "mean"),
        diagnostic_moment_matched_nll_available=("diagnostic_moment_matched_nll_available", "any"),
    ).reset_index()
    slice_metrics = slice_metrics.drop(columns=[c for c in diag_grouped.columns if c in slice_metrics.columns and c not in RESULT_GROUP_COLS], errors="ignore")
    slice_metrics = slice_metrics.merge(diag_grouped, on=RESULT_GROUP_COLS, how="left")
    has_formal_nll = pd.to_numeric(slice_metrics["bridge_nll"], errors="coerce").notna()
    slice_metrics["nll_status"] = nll_status
    slice_metrics["nll_reason"] = nll_reason
    slice_metrics["nll_score_basis"] = formal_basis
    slice_metrics.loc[~has_formal_nll, "nll_score_basis"] = BLOCKED_EXACT_NLL_SCORE_BASIS
    slice_metrics["formal_nll_status"] = formal_status
    slice_metrics.loc[~has_formal_nll, "formal_nll_status"] = "blocked_missing_exact_asof_posterior_mixture_bridge"
    slice_metrics["nll_source_kind"] = formal_source_kind
    slice_metrics.loc[~has_formal_nll, "nll_source_kind"] = "missing_exact_asof_posterior_mixture_bridge"
    slice_metrics["probability_score_basis"] = probability_basis
    slice_metrics.loc[~has_formal_nll, "probability_score_basis"] = DIAGNOSTIC_MOMENT_MATCHED_BASIS
    slice_metrics["diagnostic_moment_matched_nll_available"] = slice_metrics["diagnostic_moment_matched_nll_available"].fillna(False).astype(bool)
    slice_metrics["asof_mixture_weight_validation_path"] = "" if asof_weight_validation_path is None else str(asof_weight_validation_path)
    for key, value in contract.items():
        slice_metrics[key] = value
    value_cols = ["mae", "rmse", "nll", "bridge_nll", "coverage_90", "width_90", "coverage_50", "width_50", "wis"]
    macro = {c: float(pd.to_numeric(slice_metrics[c], errors="coerce").mean()) for c in value_cols}
    macro.update(
        {
            "dataset": dataset,
            "method": method,
            "method_group": method_group,
            "split": "test",
            "n": int(slice_metrics["n"].sum()),
            "macro_groups": int(len(slice_metrics)),
            "nll_status": nll_status,
            "nll_reason": nll_reason,
            "source_artifact": str(readout_path),
            "status": "ok",
            "macro_slice": ",".join(RESULT_GROUP_COLS),
            "nll_score_basis": formal_basis if math.isfinite(float(macro["bridge_nll"])) else BLOCKED_EXACT_NLL_SCORE_BASIS,
            "formal_nll_status": formal_status
            if math.isfinite(float(macro["bridge_nll"]))
            else "blocked_missing_exact_asof_posterior_mixture_bridge",
            "nll_source_kind": formal_source_kind
            if math.isfinite(float(macro["bridge_nll"]))
            else "missing_exact_asof_posterior_mixture_bridge",
            "probability_score_basis": probability_basis
            if math.isfinite(float(macro["bridge_nll"]))
            else DIAGNOSTIC_MOMENT_MATCHED_BASIS,
            "diagnostic_moment_matched_nll_available": bool(slice_metrics["diagnostic_moment_matched_nll_available"].any()),
            "diagnostic_moment_matched_nll": float(pd.to_numeric(slice_metrics["diagnostic_moment_matched_nll"], errors="coerce").mean()),
            "asof_mixture_weight_validation_path": "" if asof_weight_validation_path is None else str(asof_weight_validation_path),
            **contract,
        }
    )
    return pd.DataFrame([macro]), slice_metrics


def _pooled_result_alias_slices(dataset: str, method: str, slices: pd.DataFrame) -> pd.DataFrame:
    ""
    if dataset != "benchmark_b_pooled" or method not in {
        "caster_one_layer",
        "caster_hierarchical",
        "caster_one_layer_draw_kernel",
        "caster_hierarchical_draw_kernel",
    } or slices.empty:
        return pd.DataFrame()
    if "component" not in slices.columns:
        return pd.DataFrame()
    method_alias = {
        "caster_one_layer": "caster_one_layer_pooled",
        "caster_hierarchical": "caster_hierarchical_pooled",
        "caster_one_layer_draw_kernel": "caster_one_layer_draw_kernel_pooled",
        "caster_hierarchical_draw_kernel": "caster_hierarchical_draw_kernel_pooled",
    }[method]
    component_to_dataset = {
        "covid_adm_per100k": "benchmark_b_covid",
        "flu_adm_per100k": "benchmark_b_flu",
    }
    frames: list[pd.DataFrame] = []
    for component, alias_dataset in component_to_dataset.items():
        alias = slices[slices["component"].astype(str).eq(component)].copy()
        if alias.empty:
            continue
        alias["dataset"] = alias_dataset
        alias["task_id"] = alias_dataset
        alias["method"] = method_alias
        alias["method_group"] = "caster"
        alias["pooled_source_dataset"] = dataset
        alias["pooled_source_method"] = method
        alias["posterior_scope"] = "pooled_shared_posterior"
        alias["source_note"] = (
            "Benchmark B pooled shared-posterior result projected to the corresponding "
            "component result block; forecast scores remain from benchmark_b_pooled."
        )
        frames.append(alias)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _forecast_metric_row(dataset: str, method: str, method_group: str, readout_path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    readout = _read_csv(readout_path)
    row: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "method_group": method_group,
        "source_artifact": str(readout_path),
        "status": "ok" if not readout.empty else "empty",
        **point_metrics(readout),
        **interval_metrics(readout),
        **_missing_nll(),
        **contract,
    }
    return row


def _weights_for_hierarchy(root: Path) -> pd.DataFrame:
    family = _read_csv(root / "family_posterior.csv")
    inner = _read_csv(root / "inner_weights.csv")
    family_final = family.sort_values("release_time").groupby("family", as_index=False).tail(1)
    inner_final = inner.sort_values("release_time").groupby(["family", "model_id"], as_index=False).tail(1)
    merged = inner_final.merge(family_final[["family", "family_weight"]], on="family", how="left")
    merged["weight"] = merged["inner_weight"].astype(float) * merged["family_weight"].astype(float)
    return merged[["model_id", "family", "weight"]]


def _discovery_row(
    dataset: str,
    method: str,
    method_group: str,
    weights: pd.DataFrame,
    contract: dict[str, Any],
    source_artifact: Path,
) -> dict[str, Any]:
    row = {
        "dataset": dataset,
        "method": method,
        "method_group": method_group,
        "source_artifact": str(source_artifact),
        "status": "ok" if not weights.empty else "empty",
        **posterior_diagnostics(weights),
        **contract,
    }
    return row


def _runtime_rows(dataset: str, method: str, method_group: str, timing_path: Path, contract: dict[str, Any]) -> list[dict[str, Any]]:
    data = _read_json(timing_path)
    rows: list[dict[str, Any]] = []
    for rec in data.get("records", []) if isinstance(data.get("records"), list) else []:
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "method_group": method_group,
                "stage": rec.get("name", "unknown"),
                "seconds": float(rec.get("seconds", 0.0)),
                "source_artifact": str(timing_path),
                **contract,
            }
        )
    total = data.get("total_sec", data.get("total_seconds"))
    if total is not None:
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "method_group": method_group,
                "stage": "total",
                "seconds": float(total),
                "source_artifact": str(timing_path),
                **contract,
            }
        )
    return rows


def _forecast_intervals(dataset: str, method: str, readout_path: Path) -> pd.DataFrame:
    readout = _with_protocol_columns(_with_contract_intervals(_read_csv(readout_path)), dataset=dataset, source=readout_path)
    out = readout.copy()
    out["dataset"] = dataset
    out["method"] = method
    for col in FORECAST_INTERVAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[FORECAST_INTERVAL_COLUMNS]


def _long_rows_from_wide(row: dict[str, Any], metric_group: str, metrics: list[str]) -> list[dict[str, Any]]:
    base = {col: row.get(col, "") for col in RESULT_READY_COLUMNS if col not in {"metric_group", "metric", "value", "value_text", "status", "reason"}}
    status = str(row.get("status", "ok"))
    out: list[dict[str, Any]] = []
    for metric in metrics:
        value = row.get(metric, math.nan)
        metric_status = status
        reason = ""
        if metric == "nll":
            metric_status = str(row.get("nll_status", "not_available"))
            reason = str(row.get("nll_reason", ""))
        if pd.isna(value):
            value_text = reason or "NA"
        else:
            value_text = f"{float(value):.12g}" if isinstance(value, (int, float)) else str(value)
        out.append(
            {
                **base,
                "metric_group": metric_group,
                "metric": metric,
                "value": value,
                "value_text": value_text,
                "status": metric_status,
                "reason": reason,
            }
        )
    return out


def _manifest_row(kind: str, dataset: str, method: str, path: Path, status: str = "ok", note: str = "") -> dict[str, Any]:
    return {
        "artifact_kind": kind,
        "dataset": dataset,
        "method": method,
        "path": str(path),
        "exists": path.exists(),
        "rows": _rows(path),
        "status": status if path.exists() else "missing",
        "note": note,
    }


def _benchmark_roots(args: Any) -> dict[str, dict[str, Path]]:
    def component_ledger(root: Path, fallback: Path) -> Path:
        local = root / "event_ledger.csv"
        return local if local.exists() else fallback

    a_root = Path(args.benchmark_a_root) if args.benchmark_a_root else DEFAULT_BENCHMARKS["benchmark_a"]["root"]
    a_fallback_ledger = (
        Path(args.benchmark_a_ledger)
        if args.benchmark_a_ledger
        else DEFAULT_BENCHMARKS["benchmark_a"]["ledger"]
    )
                                                                             
                                                                           
                                                                           
                                                                         
                                                        
    roots = {
        "benchmark_a": {
            "root": a_root,
            "ledger": component_ledger(a_root, a_fallback_ledger),
        }
    }
    b_ledger = Path(args.benchmark_b_ledger) if args.benchmark_b_ledger else DEFAULT_BENCHMARKS["benchmark_b"]["ledger"]
    explicit_component_roots = {
        "benchmark_b_covid": getattr(args, "benchmark_b_covid_root", None),
        "benchmark_b_flu": getattr(args, "benchmark_b_flu_root", None),
        "benchmark_b_pooled": getattr(args, "benchmark_b_pooled_root", None),
    }
    any_explicit = any(explicit_component_roots.values())
    if any_explicit:
        for dataset, value in explicit_component_roots.items():
            if value:
                root = Path(value)
                roots[dataset] = {"root": root, "ledger": component_ledger(root, b_ledger)}
        return roots

    b_root = Path(args.benchmark_b_root) if args.benchmark_b_root else DEFAULT_BENCHMARKS["benchmark_b"]["root"]
    component_children = {
        "benchmark_b_covid": b_root / "benchmark_b_covid",
        "benchmark_b_flu": b_root / "benchmark_b_flu",
        "benchmark_b_pooled": b_root / "benchmark_b_pooled",
    }
    if (component_children["benchmark_b_covid"] / "caster_run_metadata.json").exists() and (
        component_children["benchmark_b_flu"] / "caster_run_metadata.json"
    ).exists():
        roots["benchmark_b_covid"] = {
            "root": component_children["benchmark_b_covid"],
            "ledger": component_ledger(component_children["benchmark_b_covid"], b_ledger),
        }
        roots["benchmark_b_flu"] = {
            "root": component_children["benchmark_b_flu"],
            "ledger": component_ledger(component_children["benchmark_b_flu"], b_ledger),
        }
        if (component_children["benchmark_b_pooled"] / "caster_run_metadata.json").exists():
            roots["benchmark_b_pooled"] = {
                "root": component_children["benchmark_b_pooled"],
                "ledger": component_ledger(component_children["benchmark_b_pooled"], b_ledger),
            }
    else:
        roots["benchmark_b"] = {"root": b_root, "ledger": b_ledger}
    return roots


def export_results(args: Any) -> dict[str, Path]:
    out_dir = Path(args.out_dir)
    asof_validation_output_path = out_dir / "asof_mixture_weight_validation.csv"

    forecast_rows: list[dict[str, Any]] = []
    metric_slice_frames: list[pd.DataFrame] = []
    discovery_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    bridge_score_frames: list[pd.DataFrame] = []
    asof_weight_validation_frames: list[pd.DataFrame] = []
    long_rows: list[dict[str, Any]] = []
    interval_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []

    for dataset, paths in _benchmark_roots(args).items():
        root = paths["root"]
        ledger_path = paths["ledger"]
        if not root.exists():
            raise SystemExit(f"missing benchmark root: {root}")
        if not ledger_path.exists():
            raise SystemExit(f"missing event ledger for no-leakage validation: {ledger_path}")
        _require_parameter_selection_freeze(root)
        one_bridge_path = _bridge_config_path(root, "one_layer")
        hier_bridge_path = _bridge_config_path(root, "hierarchical")
        bridge_payload = _require_frozen_bridge(root, "one_layer")
        hier_bridge_payload = _require_frozen_bridge(root, "hierarchical")
        one_score_source = _selected_score_source(
            bridge_payload, label=str(one_bridge_path)
        )
        hier_score_source = _selected_score_source(
            hier_bridge_payload, label=str(hier_bridge_path)
        )
        one_moment_bridge_path = root / "bridge_config.one_layer.moment_t.json"
        hier_moment_bridge_path = root / "bridge_config.hierarchical.moment_t.json"
        if not one_moment_bridge_path.is_file() or not hier_moment_bridge_path.is_file():
            raise SystemExit(f"{root} is missing family-specific moment-t bridge configs")
        _require_family_bridge(
            one_moment_bridge_path, variant="one_layer", family="moment_t"
        )
        _require_family_bridge(
            hier_moment_bridge_path, variant="hierarchical", family="moment_t"
        )
        bridge_meta = bridge_payload.get("calibration_metadata", {})
        if isinstance(bridge_meta, dict) and int(bridge_meta.get("test_rows_used_for_tuning", 0)) != 0:
            raise SystemExit(f"{one_bridge_path} used test rows for bridge tuning")
        hier_bridge_meta = hier_bridge_payload.get("calibration_metadata", {})
        if isinstance(hier_bridge_meta, dict) and int(hier_bridge_meta.get("test_rows_used_for_tuning", 0)) != 0:
            raise SystemExit(f"{hier_bridge_path} used test rows for bridge tuning")

        manifest_rows.extend(
            [
                _manifest_row("source", dataset, "event_ledger", ledger_path, note="event-ledger/no-leakage source"),
                _manifest_row("source", dataset, "bridge_config_one_layer", one_bridge_path, note="frozen validation-only one-layer bridge"),
                _manifest_row("source", dataset, "bridge_config_hierarchical", hier_bridge_path, note="frozen validation-only hierarchical bridge"),
                _manifest_row("source", dataset, "forecast_archive_manifest", root / "forecast_archive_manifest.json"),
            ]
        )

        one_contract = _contract(root, "caster_run_metadata.json", bridge_payload)
        hier_contract = _contract(root, "hierarchical_run_metadata.json", hier_bridge_payload)
        _check_contract(one_contract, f"{dataset}:caster_one_layer")
        _check_contract(hier_contract, f"{dataset}:caster_hierarchical")
        one_contract = {
            **one_contract,
            "posterior_readout_policy": ASOF_LTE_POLICY,
            "release_availability_rule": "release_time_no_later_than_forecast_origin",
        }
        hier_contract = {
            **hier_contract,
            "posterior_readout_policy": ASOF_LTE_POLICY,
            "release_availability_rule": "release_time_no_later_than_forecast_origin",
        }

        one_readout = root / "forecast_readout.csv"
        one_weights = root / "posterior_weights.csv"
        one_diag_scores = _bridge_readout_scores(
            dataset=dataset,
            method="caster_one_layer",
            ledger_path=ledger_path,
            readout_path=one_readout,
            bridge_path=one_moment_bridge_path,
        )
        if one_score_source == "draw_kernel":
            one_scores, one_validation = _draw_kernel_asof_mixture_scores(
                dataset=dataset,
                method="caster_one_layer",
                ledger_path=ledger_path,
                draws_path=root / "forecast_draws.csv",
                archive_path=root / "forecast_archive.csv",
                bridge_path=one_bridge_path,
                root=root,
                policy=ASOF_LTE_POLICY,
                selection_path=root / "candidate_selection_log.csv",
                weights_path=one_weights,
                hierarchical=False,
            )
        else:
            one_scores, one_validation = _bridge_asof_mixture_scores(
                dataset=dataset,
                method="caster_one_layer",
                ledger_path=ledger_path,
                archive_path=root / "forecast_archive.csv",
                bridge_path=one_bridge_path,
                root=root,
                policy=ASOF_LTE_POLICY,
                hierarchical=False,
                selection_path=root / "candidate_selection_log.csv",
                weights_path=one_weights,
            )
        one_scores = _with_diagnostic_scores(one_scores, one_diag_scores)
        asof_weight_validation_frames.append(one_validation)
        bridge_score_frames.append(one_scores)
        forecast_df, slices = _metric_frame(
            dataset,
            "caster_one_layer",
            "caster",
            one_readout,
            one_scores,
            one_contract,
            asof_weight_validation_path=asof_validation_output_path,
        )
        forecast = forecast_df.iloc[0].to_dict()
        forecast_rows.append(forecast)
        metric_slice_frames.append(slices)
        pooled_alias = _pooled_result_alias_slices(dataset, "caster_one_layer", slices)
        if not pooled_alias.empty:
            metric_slice_frames.append(pooled_alias)
        long_rows.extend(_long_rows_from_wide(forecast, "forecast", ["n", "mae", "rmse", "nll", "coverage_90", "width_90", "wis"]))
        interval_frames.append(_forecast_intervals(dataset, "caster_one_layer", one_readout))
        discovery = _discovery_row(dataset, "caster_one_layer", "caster", _read_csv(one_weights), one_contract, one_weights)
        discovery_rows.append(discovery)
        long_rows.extend(_long_rows_from_wide(discovery, "discovery", ["model_ess", "structural_entropy", "top1_mass", "family_ess", "family_entropy"]))
        runtime_rows.extend(_runtime_rows(dataset, "caster_one_layer", "caster", root / "timing.json", one_contract))

        hier_weights = _weights_for_hierarchy(root)
        hier_discovery = _discovery_row(dataset, "caster_hierarchical", "caster", hier_weights, hier_contract, root / "inner_weights.csv")
        discovery_rows.append(hier_discovery)
        long_rows.extend(_long_rows_from_wide(hier_discovery, "discovery", ["model_ess", "structural_entropy", "top1_mass", "family_ess", "family_entropy"]))
        runtime_rows.extend(_runtime_rows(dataset, "caster_hierarchical", "caster", root / "hierarchical_timing.json", hier_contract))
        hier_readout = root / "hierarchical_forecast_readout.csv"
        hier_forecast: dict[str, Any] | None = None
        if hier_readout.exists():
            hier_diag_scores = _bridge_readout_scores(
                dataset=dataset,
                method="caster_hierarchical",
                ledger_path=ledger_path,
                readout_path=hier_readout,
                bridge_path=hier_moment_bridge_path,
            )
            if hier_score_source == "draw_kernel":
                hier_scores, hier_validation = _draw_kernel_asof_mixture_scores(
                    dataset=dataset,
                    method="caster_hierarchical",
                    ledger_path=ledger_path,
                    draws_path=root / "forecast_draws.csv",
                    archive_path=root / "forecast_archive.csv",
                    bridge_path=hier_bridge_path,
                    root=root,
                    policy=ASOF_LTE_POLICY,
                    hierarchical=True,
                    selection_path=root / "candidate_selection_log.csv",
                )
            else:
                hier_scores, hier_validation = _bridge_asof_mixture_scores(
                    dataset=dataset,
                    method="caster_hierarchical",
                    ledger_path=ledger_path,
                    archive_path=root / "forecast_archive.csv",
                    bridge_path=hier_bridge_path,
                    root=root,
                    policy=ASOF_LTE_POLICY,
                    hierarchical=True,
                    selection_path=root / "candidate_selection_log.csv",
                )
            hier_scores = _with_diagnostic_scores(hier_scores, hier_diag_scores)
            asof_weight_validation_frames.append(hier_validation)
            bridge_score_frames.append(hier_scores)
            hier_forecast_df, hier_slices = _metric_frame(
                dataset,
                "caster_hierarchical",
                "caster",
                hier_readout,
                hier_scores,
                hier_contract,
                asof_weight_validation_path=asof_validation_output_path,
            )
            hier_forecast = hier_forecast_df.iloc[0].to_dict()
            forecast_rows.append(hier_forecast)
            metric_slice_frames.append(hier_slices)
            hier_pooled_alias = _pooled_result_alias_slices(dataset, "caster_hierarchical", hier_slices)
            if not hier_pooled_alias.empty:
                metric_slice_frames.append(hier_pooled_alias)
            long_rows.extend(_long_rows_from_wide(hier_forecast, "forecast", ["n", "mae", "rmse", "nll", "coverage_90", "width_90", "wis"]))
            interval_frames.append(_forecast_intervals(dataset, "caster_hierarchical", hier_readout))

        draw_root = root / "ablations/caster_one_layer_draw_kernel"
        draw_readout = draw_root / "forecast_readout.csv"
        draw_weights = draw_root / "posterior_weights.csv"
        draw_metadata = draw_root / "caster_run_metadata.json"
        draw_selection = draw_root / "candidate_selection_log.csv"
        draw_source = root / "forecast_draws.csv"
        draw_bridge_path = root / "bridge_config.one_layer.draw_kernel_t.json"
        draw_bridge_payload = _require_family_bridge(
            draw_bridge_path, variant="one_layer", family="draw_kernel_t"
        )
        if _selected_score_source(draw_bridge_payload, label=str(draw_bridge_path)) != "draw_kernel":
            raise SystemExit(f"{draw_bridge_path} must be the frozen draw-kernel family config")
        for required in [draw_readout, draw_weights, draw_metadata, draw_selection, draw_source, draw_bridge_path]:
            if not required.exists():
                raise SystemExit(f"missing formal one-layer draw-kernel artifact: {required}")
        draw_contract = _contract(draw_root, "caster_run_metadata.json", draw_bridge_payload)
        _check_contract(draw_contract, f"{dataset}:caster_one_layer_draw_kernel")
        draw_contract = {
            **draw_contract,
            "posterior_readout_policy": ASOF_LTE_POLICY,
            "release_availability_rule": "release_time_no_later_than_forecast_origin",
        }
        draw_diag_scores = _bridge_readout_scores(
            dataset=dataset,
            method="caster_one_layer_draw_kernel",
            ledger_path=ledger_path,
            readout_path=draw_readout,
            bridge_path=one_moment_bridge_path,
        )
        draw_scores, draw_validation = _draw_kernel_asof_mixture_scores(
            dataset=dataset,
            method="caster_one_layer_draw_kernel",
            ledger_path=ledger_path,
            draws_path=draw_source,
            archive_path=root / "forecast_archive.csv",
            bridge_path=draw_bridge_path,
            root=draw_root,
            policy=ASOF_LTE_POLICY,
            selection_path=draw_selection,
            weights_path=draw_weights,
        )
        draw_scores = _with_diagnostic_scores(draw_scores, draw_diag_scores)
        asof_weight_validation_frames.append(draw_validation)
        bridge_score_frames.append(draw_scores)
        draw_forecast_df, draw_slices = _metric_frame(
            dataset,
            "caster_one_layer_draw_kernel",
            "caster",
            draw_readout,
            draw_scores,
            draw_contract,
            nll_reason="exact as-of posterior draw-kernel mixture evaluated on test rows only",
            asof_weight_validation_path=asof_validation_output_path,
        )
        draw_forecast = draw_forecast_df.iloc[0].to_dict()
        forecast_rows.append(draw_forecast)
        metric_slice_frames.append(draw_slices)
        draw_pooled_alias = _pooled_result_alias_slices(dataset, "caster_one_layer_draw_kernel", draw_slices)
        if not draw_pooled_alias.empty:
            metric_slice_frames.append(draw_pooled_alias)
        long_rows.extend(_long_rows_from_wide(draw_forecast, "forecast", ["n", "mae", "rmse", "nll", "coverage_90", "width_90", "wis"]))
        interval_frames.append(_forecast_intervals(dataset, "caster_one_layer_draw_kernel", draw_readout))
        draw_discovery = _discovery_row(
            dataset,
            "caster_one_layer_draw_kernel",
            "caster",
            _read_csv(draw_weights),
            draw_contract,
            draw_weights,
        )
        discovery_rows.append(draw_discovery)
        long_rows.extend(_long_rows_from_wide(draw_discovery, "discovery", ["model_ess", "structural_entropy", "top1_mass", "family_ess", "family_entropy"]))
        runtime_rows.extend(_runtime_rows(dataset, "caster_one_layer_draw_kernel", "caster", draw_root / "timing.json", draw_contract))
        manifest_rows.extend(
            [
                _manifest_row("source", dataset, "one_layer_draw_kernel_readout", draw_readout),
                _manifest_row("source", dataset, "one_layer_draw_kernel_posterior", draw_weights),
                _manifest_row("source", dataset, "one_layer_draw_kernel_draws", draw_source),
                _manifest_row("source", dataset, "one_layer_draw_kernel_metadata", draw_metadata),
            ]
        )

        ablation_root = root / "ablations"
        for ablation in ABLATIONS:
            run_dir = ablation_root / ablation
            meta_path = run_dir / "ablation_metadata.json"
            meta = _read_json(meta_path)
            hierarchical_draw_ablation = ablation == "caster_hierarchical_draw_kernel"
            ablation_bridge_path = one_bridge_path
            ablation_diagnostic_bridge_path = one_moment_bridge_path
            ablation_bridge_payload = bridge_payload
            if hierarchical_draw_ablation:
                ablation_bridge_path = root / "bridge_config.hierarchical.draw_kernel_t.json"
                ablation_diagnostic_bridge_path = hier_moment_bridge_path
                ablation_bridge_payload = _require_family_bridge(
                    ablation_bridge_path,
                    variant="hierarchical",
                    family="draw_kernel_t",
                )
                if _selected_score_source(
                    ablation_bridge_payload, label=str(ablation_bridge_path)
                ) != "draw_kernel":
                    raise SystemExit(
                        f"{ablation_bridge_path} must be the frozen hierarchical draw-kernel config"
                    )
            reference_row = hier_forecast if _ablation_reference_method(ablation) == "caster_hierarchical" else forecast
            if not meta:
                method = f"ablation_{ablation}"
                contract = _contract(root, "caster_run_metadata.json", bridge_payload, diagnostic_only=True)
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "method": method,
                    "ablation": ablation,
                    "method_group": "ablation",
                    "source_artifact": str(meta_path),
                    "status": "not_available",
                    "reason": f"missing ablation metadata: {meta_path}",
                    "selected_model_count": 0,
                    "diagnostic_values_executed": False,
                    "used_for_h5_positive_claim": False,
                    **_ablation_export_fields(ablation),
                    **_missing_nll(),
                    **contract,
                }
                for col in [
                    "n",
                    "mae",
                    "rmse",
                    "nll",
                    "bridge_nll",
                    "coverage_90",
                    "width_90",
                    "coverage_50",
                    "width_50",
                    "wis",
                    "model_ess",
                    "structural_entropy",
                    "top1_mass",
                    "family_ess",
                    "family_entropy",
                ]:
                    row[col] = math.nan
                _attach_ablation_deltas(row, reference_row)
                ablation_rows.append(row)
                long_rows.extend(
                    _long_rows_from_wide(
                        row,
                        "ablation",
                        ["n", "mae", "rmse", "nll", "coverage_90", "width_90", "wis", "model_ess", "structural_entropy", "top1_mass"],
                    )
                )
                continue
            run_meta = _read_json(run_dir / "caster_run_metadata.json")
            diagnostic_only = bool(meta.get("diagnostic_only", False))
            diagnostic_executed = False
            if diagnostic_only:
                diagnostic_executed = _diagnostic_values_executed(meta, ablation)
                if diagnostic_executed:
                    _require_executed_diagnostic_artifacts(run_dir, dataset, ablation)
                    meta = _merged_ablation_metadata(meta, run_meta, f"{dataset}:ablation_{ablation}")
                    meta["diagnostic_values_executed"] = True
                    meta["used_for_h5_positive_claim"] = False
                    meta.setdefault("evaluation_basis", "frozen_result_bridge_metrics")
                    if ablation == "native_proxy_gaussian_archive_score":
                        meta.setdefault("score_update_basis", "native_proxy_gaussian_archive_score")
                    elif ablation == "adapter_native_likelihood":
                        meta.setdefault("score_update_basis", "adapter_log_likelihood")
                        meta.setdefault("adapter_native_likelihoods_compared", True)
                    elif ablation == "no_event_ledger_diagnostic":
                        meta.setdefault("diagnostic_readout_policy", "latest_snapshot")
                else:
                    raise SystemExit(f"{dataset}:ablation_{ablation} diagnostic not executed; missing executed diagnostic artifacts")
            contract = _contract_from_ablation(meta, ablation_bridge_payload)
            _check_contract(contract, f"{dataset}:ablation_{ablation}")
            if not contract["diagnostic_only"]:
                contract = {
                    **contract,
                    "posterior_readout_policy": ASOF_LTE_POLICY,
                    "release_availability_rule": "release_time_no_later_than_forecast_origin",
                }
            method = f"ablation_{ablation}"
            row: dict[str, Any] = {
                "dataset": dataset,
                "method": method,
                "ablation": ablation,
                "method_group": "ablation",
                "source_artifact": str(meta_path),
                "status": "diagnostic_executed" if diagnostic_executed else ("diagnostic_only" if contract["diagnostic_only"] else "ok"),
                "selected_model_count": len(meta.get("selected_models", [])),
                "unsafe_native_proxy_warning": meta.get("unsafe_native_proxy_warning", ""),
                "adapter_native_likelihood_warning": meta.get("adapter_native_likelihood_warning", ""),
                "no_event_ledger_refusal": meta.get("no_event_ledger_refusal", ""),
                "proxy_status": meta.get("proxy_status", ""),
                "diagnostic_values_executed": diagnostic_executed,
                "used_for_h5_positive_claim": False,
                "score_update_basis": meta.get("score_update_basis", ""),
                "evaluation_basis": meta.get("evaluation_basis", ""),
                "unsafe_native_proxy_executed": bool(meta.get("unsafe_native_proxy_executed", False)),
                "unsafe_no_event_ledger_executed": bool(meta.get("unsafe_no_event_ledger_executed", False)),
                "no_event_ledger_enforcement_relaxed": bool(meta.get("no_event_ledger_enforcement_relaxed", False)),
                "diagnostic_readout_policy": meta.get("diagnostic_readout_policy", ""),
                **_ablation_export_fields(ablation),
                **_missing_nll(),
                **contract,
            }
            if (not contract["diagnostic_only"]) or diagnostic_executed:
                readout_path = run_dir / "forecast_readout.csv"
                weights_path = run_dir / "posterior_weights.csv"
                diag_scores = _bridge_readout_scores(
                    dataset=dataset,
                    method=method,
                    ledger_path=ledger_path,
                    readout_path=readout_path,
                    bridge_path=ablation_diagnostic_bridge_path,
                )
                nll_status = "ok"
                nll_reason = "exact as-of posterior-mixture bridge evaluated on test rows only"
                if contract["diagnostic_only"]:
                    scores = _blocked_scores_from_diagnostic(
                        diag_scores,
                        "diagnostic-only ablation; exact as-of mixture NLL excluded from formal claims",
                    )
                    nll_status = "blocked_missing_exact_asof_posterior_mixture_bridge"
                    nll_reason = "diagnostic-only ablation; moment-matched readout NLL is diagnostic-only"
                else:
                    try:
                        ablation_archive = run_dir / "forecast_archive.csv"
                        if not ablation_archive.exists():
                            ablation_archive = root / "forecast_archive.csv"
                        selection_path = run_dir / "candidate_selection_log.csv"
                        if not selection_path.exists():
                            selection_path = root / "candidate_selection_log.csv"
                        if hierarchical_draw_ablation:
                            scores, validation = _draw_kernel_asof_mixture_scores(
                                dataset=dataset,
                                method=method,
                                ledger_path=ledger_path,
                                draws_path=root / "forecast_draws.csv",
                                archive_path=ablation_archive,
                                bridge_path=ablation_bridge_path,
                                root=run_dir,
                                policy=ASOF_LTE_POLICY,
                                hierarchical=True,
                                selection_path=selection_path,
                            )
                            nll_reason = (
                                "exact as-of posterior draw-kernel mixture "
                                "evaluated on test rows only"
                            )
                        else:
                            scores, validation = _bridge_asof_mixture_scores(
                                dataset=dataset,
                                method=method,
                                ledger_path=ledger_path,
                                archive_path=ablation_archive,
                                bridge_path=ablation_bridge_path,
                                root=run_dir,
                                policy=ASOF_LTE_POLICY,
                                hierarchical=False,
                                selection_path=selection_path,
                                weights_path=weights_path,
                            )
                        scores = _with_diagnostic_scores(scores, diag_scores)
                        asof_weight_validation_frames.append(validation)
                    except SystemExit as exc:
                        scores = _blocked_scores_from_diagnostic(diag_scores, str(exc))
                        nll_status = "blocked_missing_exact_asof_posterior_mixture_bridge"
                        nll_reason = str(exc)
                bridge_score_frames.append(scores)
                metric_df, slices = _metric_frame(
                    dataset,
                    method,
                    "ablation",
                    readout_path,
                    scores,
                    contract,
                    nll_status=nll_status,
                    nll_reason=nll_reason,
                    asof_weight_validation_path=asof_validation_output_path if nll_status == "ok" else None,
                )
                row.update(metric_df.iloc[0].to_dict())
                if diagnostic_executed:
                    row["status"] = "diagnostic_executed"
                    row["diagnostic_values_executed"] = True
                    row["used_for_h5_positive_claim"] = False
                    row["score_update_basis"] = meta.get("score_update_basis", row.get("score_update_basis", ""))
                    row["evaluation_basis"] = meta.get("evaluation_basis", row.get("evaluation_basis", ""))
                    row["unsafe_native_proxy_executed"] = bool(meta.get("unsafe_native_proxy_executed", False))
                    row["unsafe_no_event_ledger_executed"] = bool(meta.get("unsafe_no_event_ledger_executed", False))
                    row["no_event_ledger_enforcement_relaxed"] = bool(meta.get("no_event_ledger_enforcement_relaxed", False))
                    row["diagnostic_readout_policy"] = meta.get("diagnostic_readout_policy", "")
                row.update(posterior_diagnostics(_read_csv(weights_path)))
                metric_slice_frames.append(slices)
                interval_frames.append(_forecast_intervals(dataset, method, readout_path))
                runtime_rows.extend(_runtime_rows(dataset, method, "ablation", run_dir / "timing.json", contract))
            else:
                for col in ["n", "mae", "rmse", "nll", "bridge_nll", "coverage_90", "width_90", "coverage_50", "width_50", "wis", "model_ess", "structural_entropy", "top1_mass", "family_ess", "family_entropy"]:
                    row[col] = math.nan
            _attach_ablation_deltas(row, reference_row)
            ablation_rows.append(row)
            long_rows.extend(_long_rows_from_wide(row, "ablation", ["n", "mae", "rmse", "nll", "coverage_90", "width_90", "wis", "model_ess", "structural_entropy", "top1_mass"]))

    outputs = {
        "result_ready_results": out_dir / "result_ready_results.csv",
        "caster_metrics": out_dir / "caster_metrics.csv",
        "metric_slices": out_dir / "metric_slices.csv",
        "test_bridge_scores": out_dir / "test_bridge_scores.csv",
        "asof_mixture_weight_validation": asof_validation_output_path,
        "discovery_metrics": out_dir / "discovery_metrics.csv",
        "ablation_metrics": out_dir / "ablation_metrics.csv",
        "runtime_metrics": out_dir / "runtime_metrics.csv",
        "forecast_intervals": out_dir / "forecast_intervals.csv",
        "result_result_manifest": out_dir / "result_result_manifest.csv",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    result_ready = pd.DataFrame(long_rows)
    for col in RESULT_READY_COLUMNS:
        if col not in result_ready.columns:
            result_ready[col] = pd.NA
    result_ready[RESULT_READY_COLUMNS].to_csv(outputs["result_ready_results"], index=False)

    pd.DataFrame(forecast_rows).to_csv(outputs["caster_metrics"], index=False)
    pd.concat(metric_slice_frames, ignore_index=True).to_csv(outputs["metric_slices"], index=False)
    pd.concat(bridge_score_frames, ignore_index=True).to_csv(outputs["test_bridge_scores"], index=False)
    if asof_weight_validation_frames:
        pd.concat(asof_weight_validation_frames, ignore_index=True).to_csv(outputs["asof_mixture_weight_validation"], index=False)
    else:
        pd.DataFrame(
            columns=[
                "dataset",
                "method",
                "forecast_id",
                "forecast_origin",
                "selected_snapshot_time",
                "policy",
                "used_prior",
                "stale_age_days",
                "selected_model_count",
                "prior_source",
                "snapshot_source",
                "validation_status",
            ]
        ).to_csv(outputs["asof_mixture_weight_validation"], index=False)
    pd.DataFrame(discovery_rows).to_csv(outputs["discovery_metrics"], index=False)
    pd.DataFrame(ablation_rows).to_csv(outputs["ablation_metrics"], index=False)
    pd.DataFrame(runtime_rows).to_csv(outputs["runtime_metrics"], index=False)
    pd.concat(interval_frames, ignore_index=True)[FORECAST_INTERVAL_COLUMNS].to_csv(outputs["forecast_intervals"], index=False)

    for name, path in outputs.items():
        if name == "result_result_manifest":
            manifest_rows.append(
                {
                    "artifact_kind": "output",
                    "dataset": "all",
                    "method": name,
                    "path": str(path),
                    "exists": True,
                    "rows": len(manifest_rows) + 1,
                    "status": "ok",
                    "note": "",
                }
            )
        else:
            manifest_rows.append(_manifest_row("output", "all", name, path))
    pd.DataFrame(manifest_rows).to_csv(outputs["result_result_manifest"], index=False)
    return outputs


def main() -> None:
    ap = ArgumentParser(description="Export real-full result-ready CASTER result CSVs without recalibration or native likelihood comparison.")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "real_full_result_inputs"))
    ap.add_argument("--benchmark-a-root", default=None)
    ap.add_argument("--benchmark-b-root", default=None)
    ap.add_argument("--benchmark-b-covid-root", default=None)
    ap.add_argument("--benchmark-b-flu-root", default=None)
    ap.add_argument("--benchmark-b-pooled-root", default=None)
    ap.add_argument("--benchmark-a-ledger", default=None)
    ap.add_argument("--benchmark-b-ledger", default=None)
    args = ap.parse_args()
    outputs = export_results(args)
    print(f"result_results={Path(args.out_dir)}")
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
