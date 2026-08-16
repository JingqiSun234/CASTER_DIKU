#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CASTER_ROOT = ROOT.parents[1]
NEW_METHOD_SRC = CASTER_ROOT / "code" / "caster" / "src"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(NEW_METHOD_SRC))
sys.path.insert(0, str(CASTER_ROOT / "scripts"))

from aggregate_results import load_manifest_json, scan_run_dirs
from caster.bridge import COHERENT_CENSORED_STUDENT_T, read_bridge_config
from eval.metrics import mae, rmse
from eval.prob_metrics import (
    Z90,
    coverage,
    crps_gaussian,
    gaussian_nll,
    interval_width,
    sigma_from_interval,
    weighted_interval_score,
)
from result_metric_contract import (
    RESULT_GROUP_COLS,
    SCHEMA_VERSION,
    apply_result_metric_contract,
    filter_to_formal_horizon_grid,
)


NUMERIC_METRIC_COLUMNS = (
    "n",
    "mae",
    "rmse",
    "gaussian_nll",
    "coverage_50",
    "coverage_90",
    "width_50",
    "width_90",
)

EXPECTED_METHODS: tuple[dict[str, object], ...] = (
    {"method": "last_value", "tracker_id": "BL-NAIVE-01", "optional": False},
    {"method": "seasonal_naive", "tracker_id": "BL-NAIVE-02", "optional": False},
    {"method": "autoarima", "tracker_id": "BL-SF-AUTOARIM", "optional": False},
    {"method": "autoets", "tracker_id": "BL-SF-AUTOETS", "optional": False},
    {"method": "autotheta", "tracker_id": "BL-SF-AUTOTHET", "optional": False},
    {"method": "autoces", "tracker_id": "BL-SF-AUTOCES-", "optional": False},
    {"method": "prophet", "tracker_id": "BL-PROPHET-01", "optional": False},
    {"method": "nbeats", "tracker_id": "BL-NF-NBEATS", "optional": False},
    {"method": "nhits", "tracker_id": "BL-NF-NHITS", "optional": False},
    {"method": "deepar", "tracker_id": "BL-NF-DEEPAR", "optional": False},
    {"method": "patchtst", "tracker_id": "BL-NF-PATCHTST", "optional": False},
    {"method": "tft", "tracker_id": "BL-NF-TFT", "optional": False},
    {"method": "chronos_bolt_small", "tracker_id": "BL-FND-CHRONOS", "optional": False},
    {"method": "timesfm_2_0", "tracker_id": "BL-FND-TIMESFM", "optional": False},
    {"method": "timegpt", "tracker_id": "BL-FND-TIMEGPT", "optional": True},
    {"method": "agentic_top_one", "tracker_id": "BL-AGENT-01", "optional": False},
    {"method": "agent_react", "tracker_id": "BL-AGENT-REACT", "optional": False},
    {"method": "agentic_full_recovery", "tracker_id": "BL-AGENT-02", "optional": False},
    {"method": "offline_bma", "tracker_id": "BL-INTERNAL-BMA", "optional": True},
)
AGENT_METHODS = {"agentic_top_one", "agent_react", "agentic_full_recovery"}
BENCHMARK_B_TASK_KEYS = {"benchmark_b_covid", "benchmark_b_flu", "benchmark_b_pooled"}
POOLED_METHOD_ALIASES = {
    "agentic_top_one": "agentic_top_one_pooled",
    "agent_react": "agent_react_pooled",
    "agentic_full_recovery": "agentic_full_recovery_pooled",
    "offline_bma": "offline_bma_pooled",
}
COHERENT_CENSORED_NLL_MEASURE_BASIS = (
    "log1p_transform_mixed_measure_atoms_at_zero_and_upper_bound"
)
COHERENT_CENSORED_SCORE_BASIS = (
    "validation_frozen_single_forecast_archive_moment_"
    "coherent_censored_student_t_mixed_measure"
)
MEAN_PRESERVING_CENSORED_STUDENT_T = (
    "coherent_mean_preserving_censored_student_t"
)
COHERENT_CENSORED_CONTRACTS = {
    COHERENT_CENSORED_STUDENT_T,
    MEAN_PRESERVING_CENSORED_STUDENT_T,
}
UNIFIED_MIXTURE_POINT_SCORE_BASIS = (
    "unified_posterior_predictive_mixture_mean"
)
UNIFIED_MIXTURE_WIS_CENTER_BASIS = (
    "unified_posterior_predictive_mixture_median"
)
EXACT_STATIC_MIXTURE_SCORE_BASIS = (
    "exact_static_pretest_evidence_weighted_mixture_bridge"
)
CANONICAL_OFFLINE_INTERVAL_SOURCE = (
    "canonical_frozen_bridge_posterior_mixture_quantiles"
)
CANONICAL_OFFLINE_EVALUATION_PROVENANCE = (
    "canonical_a3_rho1_exact_static_mixture_bridge"
)
CANONICAL_OFFLINE_STAGE = "A3_offline_one_layer_caster"
CANONICAL_OFFLINE_FORECAST_SOURCE = (
    "canonical_a3_immutable_archive_frozen_pretest_mixture"
)


def _coherent_censored_score_basis(contract: str) -> str:
    return (
        "validation_frozen_single_forecast_archive_moment_"
        f"{contract}_mixed_measure"
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _path_family(run_root: Path, run_dir: Path) -> str:
    try:
        rel = run_dir.relative_to(run_root)
    except ValueError:
        rel = run_dir
    return str(rel.parts[0]) if rel.parts else "NA"


def _as_string_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _join_list(values: list[str]) -> str:
    return ";".join(sorted({str(value) for value in values if str(value)}))


def _dataset_keys_from_metrics(metrics: pd.DataFrame) -> list[str]:
    col = "dataset_key" if "dataset_key" in metrics.columns else "dataset" if "dataset" in metrics.columns else ""
    if not col:
        return []
    return sorted(_canonical_dataset_key(value) for value in metrics[col].dropna().unique())


def _dataset_scope(methods: list[str], manifest: dict[str, Any], dataset_keys: list[str]) -> tuple[str, list[str], str]:
    selected = _as_string_list(manifest.get("selected_dataset_keys"))
    excluded = _as_string_list(manifest.get("excluded_dataset_keys"))
    scope = str(manifest.get("dataset_scope", "") or "")
    if not scope and selected:
        scope = "custom"
    skip_reason = ""
    if scope == "benchmark_b_only" and "benchmark_a" in excluded:
        skip_reason = "benchmark_a_excluded_by_design"
    return scope or "all_available", selected or dataset_keys, skip_reason


def _validate_numeric_metrics(metrics: pd.DataFrame, run_dir: Path) -> None:
    missing = [col for col in NUMERIC_METRIC_COLUMNS if col not in metrics.columns]
    if missing:
        raise ValueError(f"{run_dir} metrics.csv missing numeric columns: {missing}")
    for col in NUMERIC_METRIC_COLUMNS:
        vals = pd.to_numeric(metrics[col], errors="coerce")
        if vals.isna().any() or not vals.map(math.isfinite).all():
            raise ValueError(f"{run_dir} metrics.csv has non-finite values in {col}")


def _result_group_cols(forecast: pd.DataFrame) -> list[str]:
    wanted = RESULT_GROUP_COLS
    return [col for col in wanted if col in forecast.columns]


def _ensure_protocol_slices_ok(metrics: pd.DataFrame, run_dir: Path) -> None:
    if metrics.empty:
        return
    if "protocol_slice_status" not in metrics.columns:
        raise ValueError(f"{run_dir} result metric slices missing protocol_slice_status")
    status = metrics["protocol_slice_status"].fillna("").astype(str)
    bad = metrics[~status.eq("ok")]
    if bad.empty:
        return
    reason_counts = (
        bad.get("protocol_slice_reason", pd.Series([""] * len(bad), index=bad.index))
        .fillna("")
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
    )
    sample_cols = [
        col
        for col in [
            "dataset",
            "method",
            "split",
            "mode",
            "mode_kind",
            "country",
            "country_code",
            "jurisdiction",
            "entity_id",
            "component",
            "horizon",
            "protocol_slice_status",
            "protocol_slice_reason",
        ]
        if col in bad.columns
    ]
    samples = bad[sample_cols].head(5).to_dict(orient="records")
    raise ValueError(
        f"{run_dir} has non-ok result protocol slices; "
        f"count={len(bad)} reasons={json.dumps(reason_counts, sort_keys=True)} "
        f"samples={json.dumps(samples, sort_keys=True)}"
    )


def _canonical_dataset_key(value: object) -> str:
    text = str(value)
    if text.startswith("benchmark_a"):
        return "benchmark_a"
    if text in BENCHMARK_B_TASK_KEYS:
        return text
    if text.startswith("benchmark_b"):
        return "benchmark_b"
    return text


def _is_benchmark_b_pooled_scope(
    manifest: dict[str, Any],
    timing: dict[str, Any],
    dataset_keys: list[str],
) -> bool:
    values: list[object] = [
        manifest.get("agent_selection_scope", ""),
        timing.get("agent_selection_scope", ""),
        manifest.get("task_id", ""),
        manifest.get("dataset_key", ""),
        manifest.get("dataset", ""),
        *dataset_keys,
    ]
    values.extend(_as_string_list(manifest.get("selected_dataset_keys")))
    values.extend(_as_string_list(timing.get("selected_dataset_keys")))
    return any(_canonical_dataset_key(value) == "benchmark_b_pooled" for value in values if str(value).strip())


def _component_dataset_key(component: object) -> str:
    comp = str(component or "").strip().lower()
    if "covid" in comp:
        return "benchmark_b_covid"
    if "flu" in comp:
        return "benchmark_b_flu"
    return ""


def _alias_pooled_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    ""






    if metrics.empty or "method" not in metrics.columns:
        return metrics
    out = metrics.copy()
    mask = out["method"].astype(str).isin(POOLED_METHOD_ALIASES)
    if not mask.any():
        return out
    original_method = out.loc[mask, "method"].astype(str)
    out.loc[mask, "pooled_source_dataset"] = "benchmark_b_pooled"
    out.loc[mask, "pooled_source_method"] = original_method.to_numpy()
    out.loc[mask, "method"] = original_method.map(POOLED_METHOD_ALIASES).to_numpy()
    agent_mask = mask & original_method.reindex(out.index).fillna("").isin(AGENT_METHODS)
    out.loc[agent_mask, "agent_selection_scope"] = "benchmark_b_pooled"
    out.loc[mask, "source_note"] = (
        "Benchmark B pooled result projected to the corresponding "
        "component result block; forecast scores remain from benchmark_b_pooled."
    )
    if "component" in out.columns:
        projected = out.loc[mask, "component"].map(_component_dataset_key)
        projected_mask = mask & projected.reindex(out.index).fillna("").astype(str).ne("")
        out.loc[projected_mask, "dataset"] = projected.reindex(out.index).loc[projected_mask].to_numpy()
        out.loc[projected_mask, "dataset_key"] = projected.reindex(out.index).loc[projected_mask].to_numpy()
        out.loc[projected_mask, "task_id"] = projected.reindex(out.index).loc[projected_mask].to_numpy()
    return out


def _bridge_dataset_key_for_row(dataset_value: object, component_value: object | None = None) -> str:
    ""







    ds = _canonical_dataset_key(dataset_value)
    comp = str(component_value or "").strip().lower()
    if ds == "benchmark_b_pooled":
        return ds
    if ds == "benchmark_b":
        if "covid" in comp:
            return "benchmark_b_covid"
        if "flu" in comp:
            return "benchmark_b_flu"
    return ds


def _benchmark_b_metric_task(row: pd.Series) -> str:
    ds = str(row.get("dataset", "")).strip()
    comp = str(row.get("component", "")).strip().lower()
    task_id = str(row.get("task_id", "")).strip()
    scope = str(row.get("agent_selection_scope", "")).strip()
    for value in [task_id, scope, ds]:
        if value in {"benchmark_b_covid", "benchmark_b_flu", "benchmark_b_pooled"}:
            return value
    if ds.startswith("benchmark_b"):
        if "covid" in comp:
            return "benchmark_b_covid"
        if "flu" in comp:
            return "benchmark_b_flu"
    return ds


def _is_task_local_agent_metric(row: pd.Series, task: str) -> bool:
    scope = str(row.get("agent_selection_scope", "")).strip()
    keys = {x.strip() for x in str(row.get("dataset_keys_present", "")).replace(",", ";").split(";") if x.strip()}
    source = str(row.get("source_run_dir", ""))
    return scope == task or task in keys or f"/archive_backed/{task}/" in source


def _prefer_task_local_agent_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    ""
    if metrics.empty or "method" not in metrics.columns:
        return metrics
    work = metrics.copy()
    work["_metric_task_key"] = [_benchmark_b_metric_task(row) for _, row in work.iterrows()]
    task_local_pairs: set[tuple[str, str]] = set()
    for _, row in work.iterrows():
        method = str(row.get("method", ""))
        task = str(row.get("_metric_task_key", ""))
        if method in AGENT_METHODS and task in {"benchmark_b_covid", "benchmark_b_flu"} and _is_task_local_agent_metric(row, task):
            task_local_pairs.add((method, task))
    if not task_local_pairs:
        return metrics
    drop = []
    for _, row in work.iterrows():
        method = str(row.get("method", ""))
        task = str(row.get("_metric_task_key", ""))
        drop.append((method, task) in task_local_pairs and not _is_task_local_agent_metric(row, task))
    out = work.loc[~pd.Series(drop, index=work.index)].drop(columns=["_metric_task_key"])
    return out.reset_index(drop=True)


def _candidate_bridge_roots(*paths: Path | None) -> list[Path]:
    roots: list[Path] = []
    for path in paths:
        if path is None:
            continue
        current = path.resolve()
        if current.is_file():
            current = current.parent
        for root in [current, *current.parents]:
            for candidate in [
                root / "new_method" / "artifacts",
                root / "artifacts" / "real_full",
            ]:
                if candidate not in roots:
                    roots.append(candidate)
    return roots


def _bridge_config_candidates(
    dataset_key: object,
    *,
    run_dir: Path | None = None,
    run_root: Path | None = None,
    bridge_config_root: Path | None = None,
) -> list[Path]:
    ds = _canonical_dataset_key(dataset_key)
    candidates: list[Path] = []
    override_root = bridge_config_root or os.environ.get("BASELINE_BRIDGE_CONFIG_ROOT") or os.environ.get("BRIDGE_CONFIG_ROOT")
    if override_root:
        root = Path(override_root)
        if ds in BENCHMARK_B_TASK_KEYS:
            candidates.append(root / "benchmark_b" / ds / "bridge_config.json")
            candidates.append(root / ds / "bridge_config.json")
            if ds != "benchmark_b_pooled":
                candidates.append(root / "benchmark_b" / "bridge_config.json")
        else:
            candidates.append(root / ds / "bridge_config.json")
                                                                           
                                                                            
                                                                   
        return candidates
    for root in _candidate_bridge_roots(run_dir, run_root):
        if ds in BENCHMARK_B_TASK_KEYS:
            candidates.append(root / "benchmark_b" / ds / "bridge_config.json")
            candidates.append(root / ds / "bridge_config.json")
            if ds != "benchmark_b_pooled":
                candidates.append(root / "benchmark_b" / "bridge_config.json")
        else:
            candidates.append(root / ds / "bridge_config.json")
    fallback_root = CASTER_ROOT / "code" / "caster" / "artifacts" / "real_full"
    if ds in BENCHMARK_B_TASK_KEYS:
        candidates.extend(
            [
                fallback_root / "benchmark_b" / ds / "bridge_config.json",
                fallback_root / ds / "bridge_config.json",
            ]
        )
        if ds != "benchmark_b_pooled":
            candidates.append(fallback_root / "benchmark_b" / "bridge_config.json")
    else:
        candidates.append(fallback_root / ds / "bridge_config.json")
    out: list[Path] = []
    for path in candidates:
        if path not in out:
            out.append(path)
    return out


def _bridge_config_path(
    dataset_key: object,
    *,
    run_dir: Path | None = None,
    run_root: Path | None = None,
    bridge_config_root: Path | None = None,
) -> Path:
    for path in _bridge_config_candidates(
        dataset_key,
        run_dir=run_dir,
        run_root=run_root,
        bridge_config_root=bridge_config_root,
    ):
        if path.exists():
            return path
    return _bridge_config_candidates(
        dataset_key,
        run_dir=run_dir,
        run_root=run_root,
        bridge_config_root=bridge_config_root,
    )[0]


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_bridge_metadata(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("calibration_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    metadata["rho"] = payload.get("rho", "")
    return metadata


def _bridge_core_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"bridge config is not a JSON object: {path}")
    payload = dict(payload)
    payload.pop("calibration_metadata", None)
    payload.pop("rho", None)
    return payload


def _resolve_recorded_path(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (CASTER_ROOT / path).resolve()


def _ensure_forecast_id(forecast: pd.DataFrame) -> pd.DataFrame:
    forecast = forecast.copy()
    if "forecast_id" not in forecast.columns:
        forecast["forecast_id"] = [f"baseline_generated_{i}" for i in range(len(forecast))]
    return forecast


def _bool_series(values: pd.Series, default: bool = True) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=bool)
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    text = values.astype(str).str.strip().str.lower()
    truthy = text.isin({"1", "true", "t", "yes", "y"})
    falsey = text.isin({"0", "false", "f", "no", "n", "", "nan", "none", "na"})
    out = pd.Series(default, index=values.index, dtype=bool)
    out.loc[truthy] = True
    out.loc[falsey] = False
    return out


def _observed_mask(group: pd.DataFrame) -> np.ndarray:
    y = pd.to_numeric(group["y_true"], errors="coerce")
    finite = np.isfinite(y.astype(float).to_numpy())
    if "observed_mask" not in group.columns:
        return finite
    return _bool_series(group["observed_mask"], default=True).to_numpy(dtype=bool) & finite


def _bridge_log_scores_vectorized(group: pd.DataFrame, bridge) -> np.ndarray:
    ""
                                                                            
                                                                           
                                                                             
                                                                           
                                                                             
                                                  
    lower_90 = group["pred_lower_90"].astype(float)
    upper_90 = group["pred_upper_90"].astype(float)
    sigma_interval = sigma_from_interval(lower_90, upper_90, Z90)
    pred_var = sigma_interval * sigma_interval
    observed = pd.to_numeric(group["y_true"], errors="coerce").astype(float).to_numpy()
    pred_mean_raw = pd.to_numeric(group["pred_mean"], errors="coerce").astype(float).to_numpy()
    observed_mask = _observed_mask(group)

    if bridge.transform == "identity":
        y = observed.astype(float)
        mu = pred_mean_raw.astype(float)
        pred_v = np.maximum(pred_var.astype(float), 0.0)
    elif bridge.transform == "log1p":
        y = np.log1p(np.maximum(observed.astype(float), 0.0))
        mu_nonnegative = np.maximum(pred_mean_raw.astype(float), 0.0)
        mu = np.log1p(mu_nonnegative)
        pred_v = np.maximum(pred_var.astype(float), 0.0) / np.square(1.0 + mu_nonnegative)
    else:
        raise ValueError(f"unknown bridge transform {bridge.transform!r}")

    components = group["component"].astype(str)
    sigma = components.map(lambda comp: float(bridge.sigma_by_component.get(comp, bridge.default_sigma))).astype(float).to_numpy()
    gamma = components.map(lambda comp: float(bridge.gamma_by_component.get(comp, bridge.default_gamma))).astype(float).to_numpy()
    gamma = np.maximum(gamma, 0.0)
    scale = np.sqrt(np.maximum(gamma * pred_v, 0.0) + sigma * sigma + float(bridge.min_scale) * float(bridge.min_scale))

    if bridge.distribution == "gaussian":
        z = (y - mu) / np.maximum(scale, 1e-12)
        log_scores = -0.5 * math.log(2.0 * math.pi) - np.log(np.maximum(scale, 1e-12)) - 0.5 * z * z
    elif bridge.distribution == "student_t":
        nu = float(bridge.nu)
        z2 = np.square((y - mu) / np.maximum(scale, 1e-12))
        const = math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0) - 0.5 * math.log(nu * math.pi)
        log_scores = const - np.log(np.maximum(scale, 1e-12)) - ((nu + 1.0) / 2.0) * np.log1p(z2 / nu)
    else:
        raise ValueError(f"unsupported baseline bridge distribution {bridge.distribution!r}")

    log_scores = np.asarray(log_scores, dtype=float)
    log_scores[~observed_mask] = 0.0
    return log_scores


def _coherent_censored_single_model_readout(
    group: pd.DataFrame,
    bridge,
) -> pd.DataFrame:
    ""








    required = {
        "forecast_id",
        "method",
        "component",
        "horizon",
        "y_true",
        "pred_mean",
        "pred_lower_90",
        "pred_upper_90",
    }
    if missing := sorted(required - set(group.columns)):
        raise ValueError(
            "coherent censored baseline readout missing columns "
            f"{missing}"
        )
    keys = group[["forecast_id", "method"]].astype(str)
    if keys.duplicated().any():
        examples = keys.loc[keys.duplicated(keep=False)].head(5).to_dict(
            orient="records"
        )
        raise ValueError(
            "coherent censored baseline readout requires unique "
            f"forecast_id/method rows; examples={examples}"
        )

    lower_90 = pd.to_numeric(
        group["pred_lower_90"], errors="raise"
    ).to_numpy(dtype=float)
    upper_90 = pd.to_numeric(
        group["pred_upper_90"], errors="raise"
    ).to_numpy(dtype=float)
    sigma_interval = sigma_from_interval(lower_90, upper_90, Z90)
    observed_mask = _observed_mask(group)

    ledger_columns = [
        "forecast_id",
        "protocol_version",
        "natural_event_id",
        "dataset",
        "entity_id",
        "mode",
        "mode_kind",
        "forecast_strategy",
        "revision_version",
        "country",
        "country_code",
        "jurisdiction",
        "pathogen",
        "target_variable",
        "target",
        "anchor_week",
        "forecast_origin",
        "target_time",
        "release_time",
        "features_available_until",
        "component",
        "horizon",
        "split",
    ]
    ledger = group[
        [column for column in ledger_columns if column in group.columns]
    ].copy()
    ledger["forecast_id"] = group["forecast_id"].astype(str).to_numpy()
    ledger["observed_value"] = pd.to_numeric(
        group["y_true"], errors="coerce"
    ).to_numpy(dtype=float)
    ledger["observed_mask"] = observed_mask
    ledger = ledger.drop_duplicates("forecast_id")

    archive = pd.DataFrame(
        {
            "forecast_id": group["forecast_id"].astype(str).to_numpy(),
            "model_id": group["method"].astype(str).to_numpy(),
            "particle_id": "0",
            "pred_mean": pd.to_numeric(
                group["pred_mean"], errors="raise"
            ).to_numpy(dtype=float),
            "pred_var": np.square(sigma_interval),
            "component": group["component"].astype(str).to_numpy(),
            "horizon": pd.to_numeric(
                group["horizon"], errors="raise"
            ).astype(int).to_numpy(),
        }
    )
    if "mode" in group.columns:
        archive["mode"] = group["mode"].astype(str).to_numpy()

                                                                          
                                                                            
                                 
    from caster.filter import single_model_predictive_readout

    readout = single_model_predictive_readout(
        ledger,
        archive,
        bridge,
        score_source="archive_moment",
    ).copy()
    required_output = {
        "forecast_id",
        "model_id",
        "predictive_mean",
        "predictive_median",
        "predictive_var",
        "lower_50",
        "upper_50",
        "lower_90",
        "upper_90",
        "log_score",
        "predictive_contract",
        "predictive_mean_source",
        "predictive_interval_source",
        "nll_measure_basis",
        "boundary_atom_source",
    }
    if missing := sorted(required_output - set(readout.columns)):
        raise ValueError(
            "core coherent censored single-model readout omitted columns "
            f"{missing}"
        )
    readout["forecast_id"] = readout["forecast_id"].astype(str)
    readout["model_id"] = readout["model_id"].astype(str)
    if readout.duplicated(["forecast_id", "model_id"]).any():
        raise ValueError(
            "core coherent censored single-model readout returned duplicate "
            "forecast_id/model_id rows"
        )
    contracts = sorted(
        readout["predictive_contract"].dropna().astype(str).unique()
    )
    expected_contract = str(bridge.predictive_contract)
    if contracts != [expected_contract]:
        raise ValueError(
            "core coherent censored single-model readout contract mismatch: "
            f"{contracts}"
        )
    measures = sorted(
        readout["nll_measure_basis"].dropna().astype(str).unique()
    )
    if measures != [COHERENT_CENSORED_NLL_MEASURE_BASIS]:
        raise ValueError(
            "core coherent censored single-model readout measure mismatch: "
            f"{measures}"
        )
    for column in (
        "predictive_mean",
        "predictive_median",
        "predictive_var",
        "lower_50",
        "upper_50",
        "lower_90",
        "upper_90",
        "log_score",
    ):
        values = pd.to_numeric(readout[column], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all():
            raise ValueError(
                "core coherent censored single-model readout produced "
                f"non-finite {column}"
            )
    for lower, upper in (
        ("lower_50", "upper_50"),
        ("lower_90", "upper_90"),
    ):
        if (
            pd.to_numeric(readout[lower], errors="raise")
            > pd.to_numeric(readout[upper], errors="raise")
        ).any():
            raise ValueError(
                "core coherent censored single-model readout produced "
                f"inverted {lower}/{upper} intervals"
            )
    return readout


def _attach_bridge_scores(
    forecast: pd.DataFrame,
    *,
    run_dir: Path | None = None,
    run_root: Path | None = None,
    bridge_config_root: Path | None = None,
    strict_bridge_config: bool = False,
) -> pd.DataFrame:
    forecast = _ensure_forecast_id(forecast)
    original_bridge_nll = pd.to_numeric(
        forecast.get("bridge_nll", pd.Series(math.nan, index=forecast.index)), errors="coerce"
    )
    original_bridge_log_score = pd.to_numeric(
        forecast.get("bridge_log_score", pd.Series(math.nan, index=forecast.index)), errors="coerce"
    )
    original_score_basis = forecast.get(
        "nll_score_basis", pd.Series("", index=forecast.index, dtype="object")
    ).fillna("").astype(str)
    precomputed_exact = (
        original_score_basis.eq(EXACT_STATIC_MIXTURE_SCORE_BASIS)
        & original_bridge_nll.notna()
    )
    original_columns = {
        name: forecast.get(name, pd.Series("", index=forecast.index, dtype="object")).copy()
        for name in [
            "formal_nll_status",
            "nll_source_kind",
            "probability_score_basis",
            "evaluation_bridge_provenance",
            "bridge_config_path",
            "bridge_config_hash",
            "bridge_calibration_split",
        ]
    }
    original_contract = forecast.get(
        "predictive_contract",
        pd.Series("", index=forecast.index, dtype="object"),
    ).fillna("").astype(str)
    original_interval_source = forecast.get(
        "predictive_interval_source",
        pd.Series("", index=forecast.index, dtype="object"),
    ).fillna("").astype(str)
    original_method = forecast.get(
        "method", pd.Series("", index=forecast.index, dtype="object")
    ).fillna("").astype(str)
    original_stage = forecast.get(
        "canonical_ablation_stage",
        pd.Series("", index=forecast.index, dtype="object"),
    ).fillna("").astype(str)
    original_forecast_source = forecast.get(
        "forecast_source",
        pd.Series("", index=forecast.index, dtype="object"),
    ).fillna("").astype(str)
    original_selected_model = forecast.get(
        "selected_model_id",
        pd.Series("", index=forecast.index, dtype="object"),
    ).fillna("").astype(str)
    original_online_update = _bool_series(
        forecast.get(
            "online_posterior_update",
            pd.Series(True, index=forecast.index, dtype=bool),
        ),
        default=True,
    )
    original_test_weight_rows = pd.to_numeric(
        forecast.get(
            "test_rows_used_for_weighting",
            pd.Series(math.nan, index=forecast.index),
        ),
        errors="coerce",
    )
    original_bridge_path = (
        original_columns["bridge_config_path"].fillna("").astype(str)
    )
    canonical_offline_mixture = (
        original_method.eq("offline_bma")
        & original_score_basis.eq(EXACT_STATIC_MIXTURE_SCORE_BASIS)
        & original_contract.eq(MEAN_PRESERVING_CENSORED_STUDENT_T)
        & original_interval_source.eq(CANONICAL_OFFLINE_INTERVAL_SOURCE)
        & original_columns["evaluation_bridge_provenance"].fillna("").astype(str).eq(
            CANONICAL_OFFLINE_EVALUATION_PROVENANCE
        )
        & original_stage.eq(CANONICAL_OFFLINE_STAGE)
        & original_forecast_source.eq(CANONICAL_OFFLINE_FORECAST_SOURCE)
        & original_selected_model.eq("canonical_a3_frozen_pretest_posterior")
        & original_bridge_path.ne("")
        & ~original_online_update
        & original_test_weight_rows.eq(0)
    )
    canonical_readout_columns = {
        "bridge_predictive_mean": "pred_mean",
        "bridge_predictive_median": "pred_median",
        "bridge_lower_50": "pred_lower_50",
        "bridge_upper_50": "pred_upper_50",
        "bridge_lower_90": "pred_lower_90",
        "bridge_upper_90": "pred_upper_90",
    }
    missing_canonical_readout = sorted(
        set(canonical_readout_columns.values()) - set(forecast.columns)
    )
    if missing_canonical_readout:
        canonical_offline_mixture[:] = False
    else:
        canonical_values = {
            target: pd.to_numeric(forecast[source], errors="coerce")
            for target, source in canonical_readout_columns.items()
        }
        if canonical_offline_mixture.any():
            for target, values in canonical_values.items():
                if not np.isfinite(
                    values.loc[canonical_offline_mixture].to_numpy(dtype=float)
                ).all():
                    raise ValueError(
                        "canonical Offline A3 mixture contains non-finite "
                        f"{target}"
                    )
            for lower, upper in (
                ("bridge_lower_50", "bridge_upper_50"),
                ("bridge_lower_90", "bridge_upper_90"),
            ):
                if (
                    canonical_values[lower].loc[canonical_offline_mixture]
                    > canonical_values[upper].loc[canonical_offline_mixture]
                ).any():
                    raise ValueError(
                        "canonical Offline A3 mixture contains inverted "
                        f"{lower}/{upper}"
                    )
    observed = pd.to_numeric(forecast["y_true"], errors="coerce") if "y_true" in forecast.columns else pd.Series(math.nan, index=forecast.index)
    if "observed_mask" in forecast.columns:
        observed_mask = _bool_series(forecast["observed_mask"], default=True) & np.isfinite(observed.astype(float))
    else:
        observed_mask = pd.Series(np.isfinite(observed.astype(float)), index=forecast.index)
    forecast["observed_mask"] = observed_mask.astype(bool)
    forecast["bridge_log_score"] = math.nan
    forecast["bridge_nll"] = math.nan
    forecast["bridge_config_path"] = ""
    forecast["bridge_config_hash"] = ""
    forecast["evaluation_bridge_provenance"] = "missing_bridge_config"
    forecast["bridge_calibration_split"] = ""
    forecast["rho"] = ""
    forecast["test_rows_used_for_bridge_calibration"] = 0
    forecast["nll_score_basis"] = "validation_frozen_single_forecast_bridge"
    forecast["formal_nll_status"] = "formal_validation_frozen_bridge"
    forecast["nll_source_kind"] = "validation_frozen_bridge"
    forecast["probability_score_basis"] = "validation_frozen_single_forecast_bridge"
    forecast["predictive_contract"] = ""
    forecast["point_score_basis"] = ""
    forecast["wis_center_basis"] = ""
    forecast["nll_measure_basis"] = ""
    forecast["interval_basis"] = ""
    forecast["boundary_atom_source"] = ""
    forecast["predictive_mean_source"] = ""
    forecast["evaluation_bridge_score_source"] = ""
    forecast["canonical_offline_mixture_preserved"] = False
    forecast["bridge_predictive_mean"] = math.nan
    forecast["bridge_predictive_median"] = math.nan
    forecast["bridge_predictive_var"] = math.nan
    forecast["bridge_lower_50"] = math.nan
    forecast["bridge_upper_50"] = math.nan
    forecast["bridge_lower_90"] = math.nan
    forecast["bridge_upper_90"] = math.nan
    coherent_contract_rows = pd.Series(False, index=forecast.index, dtype=bool)
    if "dataset_key" in forecast.columns:
        dataset_values = forecast["dataset_key"]
    elif "dataset" in forecast.columns:
        dataset_values = forecast["dataset"]
    else:
        dataset_values = pd.Series(["unknown"] * len(forecast), index=forecast.index)
    if "component" in forecast.columns:
        component_values = forecast["component"]
    else:
        component_values = pd.Series([""] * len(forecast), index=forecast.index)
    forecast["_bridge_dataset_key"] = [
        _bridge_dataset_key_for_row(dataset_value, component_value)
        for dataset_value, component_value in zip(dataset_values, component_values)
    ]

    for dataset_key, idx in forecast.groupby("_bridge_dataset_key", dropna=False).groups.items():
        path = _bridge_config_path(
            dataset_key,
            run_dir=run_dir,
            run_root=run_root,
            bridge_config_root=bridge_config_root,
        )
        if not path.exists():
            if strict_bridge_config:
                checked = ", ".join(
                    str(candidate)
                    for candidate in _bridge_config_candidates(
                        dataset_key,
                        run_dir=run_dir,
                        run_root=run_root,
                        bridge_config_root=bridge_config_root,
                    )
                )
                raise FileNotFoundError(f"missing frozen bridge_config.json for {dataset_key}; checked: {checked}")
            continue
        bridge, rho = read_bridge_config(path)
        metadata = _read_bridge_metadata(path)
        digest = _sha256_file(path)
        group = forecast.loc[list(idx)].copy()
        mask = _observed_mask(group)
        canonical_group = canonical_offline_mixture.loc[list(idx)]
        if canonical_group.all():
            recorded_paths = sorted(
                set(original_bridge_path.loc[list(idx)].astype(str))
            )
            recorded_hashes = sorted(
                set(
                    original_columns["bridge_config_hash"]
                    .loc[list(idx)]
                    .fillna("")
                    .astype(str)
                )
            )
            recorded_path = (
                _resolve_recorded_path(recorded_paths[0])
                if len(recorded_paths) == 1
                else Path("")
            )
            core_matches = (
                len(recorded_paths) == 1
                and len(recorded_hashes) == 1
                and recorded_path.is_file()
                and recorded_hashes[0] == _sha256_file(recorded_path)
                and _bridge_core_payload(recorded_path)
                == _bridge_core_payload(path)
            )
            if not core_matches:
                canonical_offline_mixture.loc[list(idx)] = False
                canonical_group = canonical_offline_mixture.loc[list(idx)]
        if (
            str(getattr(bridge, "predictive_contract", ""))
            in COHERENT_CENSORED_CONTRACTS
            and canonical_group.all()
        ):
                                                                         
                                                                            
                                                                            
                                                                           
            coherent_contract_rows.loc[list(idx)] = True
            continue
        if (
            str(getattr(bridge, "predictive_contract", ""))
            in COHERENT_CENSORED_CONTRACTS
        ):
            readout = _coherent_censored_single_model_readout(group, bridge)
            lookup = readout.set_index(["forecast_id", "model_id"])
            group_keys = pd.MultiIndex.from_arrays(
                [
                    group["forecast_id"].astype(str),
                    group["method"].astype(str),
                ],
                names=["forecast_id", "model_id"],
            )
            matched = lookup.reindex(group_keys)
            if matched["log_score"].isna().any():
                raise ValueError(
                    f"{path} coherent censored baseline readout did not cover "
                    "every forecast row"
                )
            log_scores = pd.to_numeric(
                matched["log_score"], errors="raise"
            ).to_numpy(dtype=float)
            coherent_contract_rows.loc[list(idx)] = True
            for source, destination in (
                ("predictive_mean", "bridge_predictive_mean"),
                ("predictive_median", "bridge_predictive_median"),
                ("predictive_var", "bridge_predictive_var"),
                ("lower_50", "bridge_lower_50"),
                ("upper_50", "bridge_upper_50"),
                ("lower_90", "bridge_lower_90"),
                ("upper_90", "bridge_upper_90"),
            ):
                forecast.loc[list(idx), destination] = pd.to_numeric(
                    matched[source], errors="raise"
                ).to_numpy(dtype=float)
            forecast.loc[list(idx), "predictive_contract"] = (
                str(bridge.predictive_contract)
            )
            mean_preserving_censored = (
                str(bridge.predictive_contract)
                == MEAN_PRESERVING_CENSORED_STUDENT_T
            )
            forecast.loc[list(idx), "point_score_basis"] = (
                UNIFIED_MIXTURE_POINT_SCORE_BASIS
                if mean_preserving_censored
                else "archived_raw_point"
            )
            if mean_preserving_censored:
                forecast.loc[list(idx), "wis_center_basis"] = (
                    UNIFIED_MIXTURE_WIS_CENTER_BASIS
                )
            forecast.loc[list(idx), "nll_measure_basis"] = (
                COHERENT_CENSORED_NLL_MEASURE_BASIS
            )
            forecast.loc[list(idx), "interval_basis"] = (
                matched["predictive_interval_source"].astype(str).to_numpy()
            )
            forecast.loc[list(idx), "boundary_atom_source"] = (
                matched["boundary_atom_source"].astype(str).to_numpy()
            )
            forecast.loc[list(idx), "predictive_mean_source"] = (
                matched["predictive_mean_source"].astype(str).to_numpy()
            )
            forecast.loc[list(idx), "nll_score_basis"] = (
                _coherent_censored_score_basis(
                    str(bridge.predictive_contract)
                )
            )
            forecast.loc[list(idx), "probability_score_basis"] = (
                str(bridge.predictive_contract)
            )
            forecast.loc[list(idx), "evaluation_bridge_score_source"] = (
                "archive_moment"
            )
        else:
            log_scores = _bridge_log_scores_vectorized(group, bridge)
        forecast.loc[list(idx), "bridge_log_score"] = log_scores
        forecast.loc[list(idx), "bridge_nll"] = np.where(mask, -log_scores, math.nan)
        forecast.loc[list(idx), "bridge_config_path"] = str(path)
        forecast.loc[list(idx), "bridge_config_hash"] = digest
        forecast.loc[list(idx), "evaluation_bridge_provenance"] = "validation_frozen_bridge_config"
        forecast.loc[list(idx), "bridge_calibration_split"] = str(metadata.get("calibration_split", "val"))
        forecast.loc[list(idx), "rho"] = "" if rho is None else float(rho)
        forecast.loc[list(idx), "test_rows_used_for_bridge_calibration"] = int(metadata.get("test_rows_used_for_tuning", 0) or 0)
    restore_canonical_offline = (
        canonical_offline_mixture & coherent_contract_rows
    )
    if restore_canonical_offline.any():
        restore_exact_score = (
            restore_canonical_offline & original_bridge_nll.notna()
        )
        forecast.loc[restore_exact_score, "bridge_nll"] = (
            original_bridge_nll.loc[restore_exact_score]
        )
        has_original_log_score = (
            restore_exact_score & original_bridge_log_score.notna()
        )
        forecast.loc[has_original_log_score, "bridge_log_score"] = (
            original_bridge_log_score.loc[has_original_log_score]
        )
        derive_log_score = restore_exact_score & ~original_bridge_log_score.notna()
        forecast.loc[derive_log_score, "bridge_log_score"] = (
            -original_bridge_nll.loc[derive_log_score]
        )
        for target, values in canonical_values.items():
            forecast.loc[restore_canonical_offline, target] = (
                values.loc[restore_canonical_offline].to_numpy(dtype=float)
            )
        forecast.loc[
            restore_canonical_offline, "predictive_contract"
        ] = original_contract.loc[restore_canonical_offline]
        forecast.loc[
            restore_canonical_offline, "point_score_basis"
        ] = UNIFIED_MIXTURE_POINT_SCORE_BASIS
        forecast.loc[
            restore_canonical_offline, "wis_center_basis"
        ] = UNIFIED_MIXTURE_WIS_CENTER_BASIS
        forecast.loc[
            restore_canonical_offline, "nll_measure_basis"
        ] = COHERENT_CENSORED_NLL_MEASURE_BASIS
        forecast.loc[
            restore_canonical_offline, "boundary_atom_source"
        ] = "latent_student_t_censoring"
        forecast.loc[
            restore_canonical_offline, "interval_basis"
        ] = original_interval_source.loc[restore_canonical_offline]
        forecast.loc[
            restore_canonical_offline, "predictive_mean_source"
        ] = "canonical_frozen_bridge_posterior_mixture_expectation"
        forecast.loc[
            restore_canonical_offline, "evaluation_bridge_score_source"
        ] = "canonical_static_pretest_posterior_mixture"
        forecast.loc[
            restore_canonical_offline, "canonical_offline_mixture_preserved"
        ] = True
        forecast.loc[restore_canonical_offline, "rho"] = 1.0
        forecast.loc[
            restore_canonical_offline, "nll_score_basis"
        ] = original_score_basis.loc[restore_canonical_offline]
        for name, values in original_columns.items():
            forecast.loc[restore_canonical_offline, name] = values.loc[
                restore_canonical_offline
            ]
    restore_precomputed = precomputed_exact & ~coherent_contract_rows
    if restore_precomputed.any():
        forecast.loc[restore_precomputed, "bridge_nll"] = original_bridge_nll.loc[restore_precomputed]
        if original_bridge_log_score.loc[restore_precomputed].notna().all():
            forecast.loc[restore_precomputed, "bridge_log_score"] = original_bridge_log_score.loc[restore_precomputed]
        else:
            forecast.loc[restore_precomputed, "bridge_log_score"] = -original_bridge_nll.loc[restore_precomputed]
        forecast.loc[restore_precomputed, "nll_score_basis"] = original_score_basis.loc[restore_precomputed]
        for name, values in original_columns.items():
            forecast.loc[restore_precomputed, name] = values.loc[restore_precomputed]
    return forecast.drop(columns=["_bridge_dataset_key"])


def _filter_to_evaluation_split(
    forecast: pd.DataFrame,
    evaluation_split: str,
) -> pd.DataFrame:
    ""

    selected = str(evaluation_split).strip().lower()
    if selected == "all":
        return forecast.copy()
    if selected != "test":
        raise ValueError(
            "evaluation_split must be one of {'all', 'test'}; "
            f"received {evaluation_split!r}"
        )
    if "split" not in forecast.columns:
        raise ValueError(
            "evaluation_split='test' requires forecast.csv column 'split'"
        )
    projected = forecast[
        forecast["split"].astype(str).str.strip().str.lower().eq("test")
    ].copy()
    if projected.empty:
        raise ValueError(
            "evaluation_split='test' selected no forecast rows"
        )
    if "result_metric_eligible" in projected.columns:
        projected = projected[
            _bool_series(projected["result_metric_eligible"], default=False)
        ].copy()
        if projected.empty:
            raise ValueError(
                "evaluation_split='test' selected no result_metric_eligible rows"
            )
    return filter_to_formal_horizon_grid(projected, strict=True)


def _recompute_result_metrics_from_forecast(
    run_dir: Path,
    *,
    run_root: Path | None = None,
    bridge_config_root: Path | None = None,
    strict_bridge_config: bool = False,
    evaluation_split: str = "all",
) -> pd.DataFrame:
    raw_forecast = pd.read_csv(
        run_dir / "forecast.csv", low_memory=False
    )
    raw_forecast = _filter_to_evaluation_split(
        raw_forecast, evaluation_split
    )
    forecast = _attach_bridge_scores(
        raw_forecast,
        run_dir=run_dir,
        run_root=run_root,
        bridge_config_root=bridge_config_root,
        strict_bridge_config=strict_bridge_config,
    )
    required = {
        "method",
        "component",
        "horizon",
        "split",
        "y_true",
        "pred_mean",
        "pred_lower_50",
        "pred_upper_50",
        "pred_lower_90",
        "pred_upper_90",
    }
    missing = required - set(forecast.columns)
    if missing:
        raise ValueError(f"{run_dir} forecast.csv missing result metric columns: {sorted(missing)}")
    forecast = apply_result_metric_contract(forecast, method_group="baseline")
    group_cols = _result_group_cols(forecast)
    if not group_cols:
        raise ValueError(f"{run_dir} forecast.csv has no result grouping columns")
    rows: list[dict[str, object]] = []
    for keys, group in forecast.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        contracts = sorted(
            {
                str(value)
                for value in group["predictive_contract"].dropna().unique()
                if str(value)
            }
        )
        coherent_censored = (
            len(contracts) == 1
            and contracts[0] in COHERENT_CENSORED_CONTRACTS
        )
        mean_preserving_censored = (
            coherent_censored
            and contracts[0] == MEAN_PRESERVING_CENSORED_STUDENT_T
        )
        canonical_offline_mixture = (
            "canonical_offline_mixture_preserved" in group.columns
            and _bool_series(
                group["canonical_offline_mixture_preserved"],
                default=False,
            ).all()
        )
        if (
            any(value in COHERENT_CENSORED_CONTRACTS for value in contracts)
            and not coherent_censored
        ):
            raise ValueError(
                f"{run_dir} result metric group mixes predictive contracts: "
                f"{contracts}"
            )
        if coherent_censored:
            probability_mean = pd.to_numeric(
                group["bridge_predictive_mean"], errors="raise"
            ).astype(float)
            probability_median = pd.to_numeric(
                group["bridge_predictive_median"], errors="raise"
            ).astype(float)
            lower_50 = pd.to_numeric(
                group["bridge_lower_50"], errors="raise"
            ).astype(float)
            upper_50 = pd.to_numeric(
                group["bridge_upper_50"], errors="raise"
            ).astype(float)
            lower_90 = pd.to_numeric(
                group["bridge_lower_90"], errors="raise"
            ).astype(float)
            upper_90 = pd.to_numeric(
                group["bridge_upper_90"], errors="raise"
            ).astype(float)
            if not mean_preserving_censored:
                lower_50 = lower_50.clip(lower=0.0)
                upper_50 = upper_50.clip(lower=0.0)
                lower_90 = lower_90.clip(lower=0.0)
                upper_90 = upper_90.clip(lower=0.0)
            probability_columns = {
                "bridge_predictive_mean": probability_mean,
                "bridge_predictive_median": probability_median,
                "bridge_lower_50": lower_50,
                "bridge_upper_50": upper_50,
                "bridge_lower_90": lower_90,
                "bridge_upper_90": upper_90,
            }
            for column, values in probability_columns.items():
                if not np.isfinite(values.to_numpy(dtype=float)).all():
                    raise ValueError(
                        f"{run_dir} coherent censored readout has non-finite "
                        f"{column}"
                    )
        else:
            probability_mean = group["pred_mean"].astype(float)
            probability_median = probability_mean
            lower_50 = group["pred_lower_50"].astype(float).clip(lower=0.0)
            upper_50 = group["pred_upper_50"].astype(float).clip(lower=0.0)
            lower_90 = group["pred_lower_90"].astype(float).clip(lower=0.0)
            upper_90 = group["pred_upper_90"].astype(float).clip(lower=0.0)
        sigma = sigma_from_interval(lower_90, upper_90, Z90)
        observed_group_mask = _observed_mask(group)
        observed_rows = int(observed_group_mask.sum())
        bridge_nll = pd.to_numeric(group["bridge_nll"], errors="coerce")
        observed_bridge_nll = bridge_nll[observed_group_mask]
        bridge_available = observed_rows > 0 and observed_bridge_nll.notna().all()
        row = dict(zip(group_cols, keys))
        row["horizon"] = int(row["horizon"])
        point_prediction = (
            probability_mean
            if mean_preserving_censored
            else group["pred_mean"].astype(float)
        )
        wis_center = (
            probability_median
            if mean_preserving_censored
            else probability_mean
        )
        diagnostic_gaussian_nll = gaussian_nll(
            group["y_true"], probability_mean, sigma
        )
        crps_value = crps_gaussian(
            group["y_true"], probability_mean, sigma
        )
        row.update({
            "n": observed_rows,
            "mae": mae(group["y_true"], point_prediction),
            "rmse": rmse(group["y_true"], point_prediction),
            "nll": float(observed_bridge_nll.mean()) if bridge_available else math.nan,
            "bridge_nll": float(observed_bridge_nll.mean()) if bridge_available else math.nan,
            "gaussian_nll": diagnostic_gaussian_nll,
            "diagnostic_gaussian_nll": diagnostic_gaussian_nll,
            "coverage_50": coverage(group["y_true"], lower_50, upper_50),
            "coverage_90": coverage(group["y_true"], lower_90, upper_90),
            "width_50": interval_width(lower_50, upper_50),
            "width_90": interval_width(lower_90, upper_90),
            "wis": weighted_interval_score(
                group["y_true"],
                wis_center,
                [
                    (0.50, lower_50, upper_50),
                    (0.10, lower_90, upper_90),
                ],
            ),
            "crps": crps_value,
            "crps_gaussian": crps_value,
            "metric_harness": "canonical_declared_slice_horizon",
            "macro_slice": ",".join(RESULT_GROUP_COLS),
            "metric_slice_schema": SCHEMA_VERSION,
            "aggregation_schema": ",".join(RESULT_GROUP_COLS),
            "interval_policy": (
                (
                    f"{contracts[0]}_canonical_static_posterior_"
                    "mixture_quantiles"
                )
                if canonical_offline_mixture
                else f"{contracts[0]}_single_model_mixture_quantiles"
                if coherent_censored
                else "nonnegative_clip_for_result_metric_contract"
            ),
            "nll_status": "ok" if bridge_available else "bridge_config_missing",
            "nll_reason": (
                (
                    "shared validation-calibrated coherent censored Student-t "
                    + (
                        "canonical frozen A3 posterior mixture; exact mixture "
                        "NLL and mixture quantiles preserved; "
                        if canonical_offline_mixture
                        else
                        "single-model predictive; baseline pred_var derived "
                        "from the raw 90% interval before target-domain "
                        "clipping; "
                    )
                    + "NLL uses the log1p mixed measure with explicit atoms at "
                    "zero and the frozen upper bound; intervals, WIS, and "
                    "coverage use the same predictive distribution; "
                    + (
                        "RMSE/MAE use its predictive mean and WIS uses its "
                        "predictive median; "
                        if mean_preserving_censored
                        else "raw archived point retained for RMSE/MAE; "
                    )
                    + "non-finite targets "
                    "marked observed_mask=False; no native likelihood comparison"
                    if coherent_censored
                    else
                    "shared validation-calibrated bridge evaluated through CASTER bridge.score_archive_rows; "
                    "baseline pred_var derived from the raw 90% interval before target-domain clipping; "
                    "non-finite targets marked observed_mask=False; "
                    "no native likelihood comparison"
                )
                if bridge_available
                else "No frozen shared bridge_config.json was available for this dataset; nll/bridge_nll left missing."
            ),
            "nll_score_basis": _join_list([str(value) for value in group["nll_score_basis"].dropna().unique()]),
            "formal_nll_status": _join_list([str(value) for value in group["formal_nll_status"].dropna().unique()]),
            "nll_source_kind": _join_list([str(value) for value in group["nll_source_kind"].dropna().unique()]),
            "probability_score_basis": _join_list([str(value) for value in group["probability_score_basis"].dropna().unique()]),
            "predictive_contract": _join_list([str(value) for value in group["predictive_contract"].dropna().unique()]),
            "point_score_basis": _join_list([str(value) for value in group["point_score_basis"].dropna().unique()]),
            "wis_center_basis": _join_list([str(value) for value in group["wis_center_basis"].dropna().unique()]),
            "nll_measure_basis": _join_list([str(value) for value in group["nll_measure_basis"].dropna().unique()]),
            "interval_basis": _join_list([str(value) for value in group["interval_basis"].dropna().unique()]),
            "boundary_atom_source": _join_list([str(value) for value in group["boundary_atom_source"].dropna().unique()]),
            "predictive_mean_source": _join_list([str(value) for value in group["predictive_mean_source"].dropna().unique()]),
            "evaluation_bridge_score_source": _join_list([str(value) for value in group["evaluation_bridge_score_source"].dropna().unique()]),
            "diagnostic_gaussian_nll_available": not coherent_censored,
            "diagnostic_moment_matched_nll_available": False,
            "canonical_offline_mixture_preserved": bool(
                canonical_offline_mixture
            ),
            "h2_evidence_eligible": bool(bridge_available),
            "bridge_config_path": _join_list([str(value) for value in group["bridge_config_path"].dropna().unique()]),
            "bridge_config_hash": _join_list([str(value) for value in group["bridge_config_hash"].dropna().unique()]),
            "evaluation_bridge_provenance": _join_list([str(value) for value in group["evaluation_bridge_provenance"].dropna().unique()]),
            "bridge_calibration_split": _join_list([str(value) for value in group["bridge_calibration_split"].dropna().unique()]),
            "rho": _join_list([str(value) for value in group["rho"].dropna().unique()]),
            "observed_mask_false_rows": int((~observed_group_mask).sum()),
            "posterior_update_splits": "train;val",
            "test_rows_used_for_bridge_calibration": int(pd.to_numeric(group["test_rows_used_for_bridge_calibration"], errors="coerce").fillna(0).max()),
            "test_rows_used_for_posterior_update": 0,
            "native_likelihoods_compared": False,
            "diagnostic_only": False,
            "method_group": "baseline",
        })
        rows.append(row)
    metrics = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    _ensure_protocol_slices_ok(metrics, run_dir)
    return metrics


def aggregate_for_result(
    *,
    run_root: Path,
    out_metrics: Path,
    out_manifest: Path,
    report_path: Path,
    out_metric_slices: Path | None = None,
    bridge_config_root: Path | None = None,
    strict_bridge_config: bool = False,
    evaluation_split: str = "all",
    allow_missing_agents: bool = False,
    extra_run_roots: list[Path] | None = None,
    exclude_run_prefixes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete, skipped = scan_run_dirs(run_root)
    normalised_excludes = [
        value.strip().strip("/")
        for value in (exclude_run_prefixes or [])
        if value.strip().strip("/")
    ]

    def included(summary: dict[str, object], root: Path) -> bool:
        run_dir = Path(str(summary["run_dir"]))
        try:
            relative = run_dir.relative_to(root).as_posix()
        except ValueError:
            relative = run_dir.as_posix()
        return not any(
            relative == prefix or relative.startswith(prefix + "/")
            for prefix in normalised_excludes
        )

    complete = [row for row in complete if included(row, run_root)]
    seen_run_dirs = {str(row["run_dir"]) for row in complete}
    for extra_root in extra_run_roots or []:
        extra_complete, extra_skipped = scan_run_dirs(extra_root)
        extra_complete = [
            row for row in extra_complete if included(row, extra_root)
        ]
        skipped.extend(extra_skipped)
        for row in extra_complete:
            if str(row["run_dir"]) not in seen_run_dirs:
                complete.append(row)
                seen_run_dirs.add(str(row["run_dir"]))
    metric_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    for summary in complete:
        run_dir = Path(str(summary["run_dir"]))
        source_metrics = pd.read_csv(run_dir / "metrics.csv")
        _validate_numeric_metrics(source_metrics, run_dir)
        metrics = _recompute_result_metrics_from_forecast(
            run_dir,
            run_root=run_root,
            bridge_config_root=bridge_config_root,
            strict_bridge_config=strict_bridge_config,
            evaluation_split=evaluation_split,
        )
        _validate_numeric_metrics(metrics, run_dir)
        manifest = load_manifest_json(run_dir)
        timing = _load_json(run_dir / "timing.json")
        family = _path_family(run_root, run_dir)
        metric_dataset_keys = _dataset_keys_from_metrics(metrics)
        source_methods = sorted(str(m) for m in metrics["method"].dropna().unique())
        pooled_scope = _is_benchmark_b_pooled_scope(manifest, timing, metric_dataset_keys)
        if pooled_scope:
            metrics = _alias_pooled_metrics(metrics)
            metric_dataset_keys = _dataset_keys_from_metrics(metrics)
        methods = sorted(str(m) for m in metrics["method"].dropna().unique())
                                                                           
                                                                             
                                                                   
        manifest_methods = source_methods if pooled_scope else methods
        dataset_scope, dataset_keys_present, dataset_skip_reason = _dataset_scope(methods, manifest, metric_dataset_keys)
        excluded_dataset_keys = _as_string_list(manifest.get("excluded_dataset_keys"))
        if not excluded_dataset_keys and dataset_scope == "benchmark_b_only":
            excluded_dataset_keys = ["benchmark_a"]
        metrics = metrics.copy()
        metrics["status"] = "numeric"
        metrics["baseline_family"] = family
        metrics["source_run_dir"] = str(run_dir)
        metrics["source_metrics_path"] = str(run_dir / "metrics.csv")
        metrics["source_forecast_path"] = str(run_dir / "forecast.csv")
        metrics["dataset_scope"] = dataset_scope
        metrics["dataset_keys_present"] = _join_list(dataset_keys_present)
        metrics["excluded_dataset_keys"] = _join_list(excluded_dataset_keys)
        metrics["agent_selection_scope"] = manifest.get("agent_selection_scope", timing.get("agent_selection_scope", ""))
        if pooled_scope:
            pooled_agent_mask = metrics["method"].astype(str).isin(
                [POOLED_METHOD_ALIASES[method] for method in AGENT_METHODS]
            )
            metrics.loc[pooled_agent_mask, "agent_selection_scope"] = "benchmark_b_pooled"
        metric_frames.append(metrics)
        manifest_rows.append({
            "status": "numeric",
            "method": ";".join(manifest_methods),
            "methods": ";".join(manifest_methods),
            "baseline_family": family,
            "run_dir": str(run_dir),
            "forecast_path": str(run_dir / "forecast.csv"),
            "metrics_path": str(run_dir / "metrics.csv"),
            "timing_path": str(run_dir / "timing.json"),
            "run_manifest_path": str(run_dir / "run_manifest.json"),
            "forecast_rows": int(summary["forecast_rows"]),
            "metrics_rows": int(summary["metrics_rows"]),
            "dataset_key": _join_list(dataset_keys_present) or manifest.get("dataset_key", "NA"),
            "dataset_keys_present": _join_list(dataset_keys_present),
            "dataset_scope": dataset_scope,
            "agent_selection_scope": manifest.get("agent_selection_scope", timing.get("agent_selection_scope", "")),
            "excluded_dataset_keys": _join_list(excluded_dataset_keys),
            "dataset": manifest.get("dataset", "NA"),
            "mode": manifest.get("mode", "NA"),
            "backend": manifest.get("backend", "NA"),
            "total_seconds": timing.get("total_seconds", ""),
            "artifact_reuse": timing.get("artifact_reuse", manifest.get("artifact_reuse", "")),
            "restart_type": timing.get("restart_type", manifest.get("restart_type", "")),
            "skip_reason": dataset_skip_reason,
        })

    metrics_out = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    metrics_out = _prefer_task_local_agent_rows(metrics_out)
    _ensure_protocol_slices_ok(metrics_out, run_root)
    present_methods = set(metrics_out["method"].astype(str).unique()) if not metrics_out.empty else set()
    required_missing: list[str] = []
    for spec in EXPECTED_METHODS:
        method = str(spec["method"])
        if method in present_methods:
            continue
        deferred_agent = bool(allow_missing_agents and method in AGENT_METHODS)
        optional = bool(spec["optional"])
        status = "deferred" if deferred_agent else "skipped" if optional else "missing"
        if not optional and not deferred_agent:
            required_missing.append(method)
        manifest_rows.append({
            "status": status,
            "method": method,
            "methods": method,
            "baseline_family": "expected_method",
            "run_dir": "",
            "forecast_path": "",
            "metrics_path": "",
            "timing_path": "",
            "run_manifest_path": "",
            "forecast_rows": 0,
            "metrics_rows": 0,
            "dataset_key": "NA",
            "dataset_keys_present": "",
            "dataset_scope": "not_run",
            "agent_selection_scope": "",
            "excluded_dataset_keys": "",
            "dataset": "NA",
            "mode": "NA",
            "backend": "NA",
            "total_seconds": "",
            "artifact_reuse": "",
            "restart_type": "",
            "skip_reason": (
                "deferred_until_full26_archive"
                if deferred_agent
                else "optional_not_implemented"
                if optional
                else "required_method_not_found"
            ),
            "tracker_id": spec["tracker_id"],
        })

    manifest_out = pd.DataFrame(manifest_rows)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.to_csv(out_metrics, index=False)
    if out_metric_slices is not None:
        out_metric_slices.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.to_csv(out_metric_slices, index=False)
    manifest_out.to_csv(out_manifest, index=False)
    _write_summary_packet(report_path, run_root, out_metrics, out_manifest, metrics_out, manifest_out, skipped)
    if required_missing:
        raise RuntimeError(f"required methods missing from aggregated metrics: {required_missing}")
    return metrics_out, manifest_out


def _write_summary_packet(
    path: Path,
    run_root: Path,
    out_metrics: Path,
    out_manifest: Path,
    metrics: pd.DataFrame,
    manifest: pd.DataFrame,
    skipped_scan_rows: list[dict],
) -> None:
    status_counts = manifest["status"].value_counts(dropna=False).to_dict() if not manifest.empty else {}
    method_count = int(metrics["method"].nunique()) if not metrics.empty and "method" in metrics.columns else 0
    sample = metrics.iloc[0].to_dict() if not metrics.empty else {}
    lines = [
        "# Baseline summary packet",
        "",
        f"- run_root: `{run_root}`",
        f"- baseline_metrics: `{out_metrics}`",
        f"- baseline_run_manifest: `{out_manifest}`",
        f"- metric_rows: {len(metrics)}",
        f"- manifest_rows: {len(manifest)}",
        f"- numeric_method_count: {method_count}",
        f"- manifest_status_counts: `{json.dumps(status_counts, sort_keys=True)}`",
        f"- incomplete_or_invalid_scan_rows: {len(skipped_scan_rows)}",
        "",
        "## Validation sample row",
        "",
    ]
    if sample:
        for key in sorted(sample):
            lines.append(f"- {key}: `{sample[key]}`")
    else:
        lines.append("- no numeric metric rows were aggregated")
    lines.extend([
        "",
        "## Notes",
        "",
        "- `baseline_metrics.csv` and `baseline_metric_slices.csv` contain canonical declared dataset-slice/horizon macro slices and include source artifact paths.",
        "- Baseline pred_var is derived from the raw 90% interval. alternate contracts retain the archived result intervals; the coherent censored contract obtains NLL, intervals, WIS, and coverage from one core single-model predictive readout.",
        "- `nll` and `bridge_nll` are scored through the frozen CASTER validation-calibrated likelihood bridge; `gaussian_nll` is retained only as a diagnostic.",
        "- Rows with non-finite targets are retained with `observed_mask=False` and excluded from bridge NLL denominators.",
        "- `baseline_run_manifest.csv` contains numeric runs plus optional skipped expected methods.",
        "- Result table scripts should read `results/baseline_metrics.csv` directly.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CASTER baseline results into result-readable CSVs.")
    parser.add_argument("--run-root", default="runs_v3_full/baselines")
    parser.add_argument("--out-metrics", default="results/baseline_metrics.csv")
    parser.add_argument("--out-metric-slices", default="results/baseline_metric_slices.csv")
    parser.add_argument("--out-manifest", default="results/baseline_run_manifest.csv")
    parser.add_argument("--report", default="reports/baseline_summary_packet.md")
    parser.add_argument("--bridge-config-root", default=None, help="Root containing <dataset>/bridge_config.json files.")
    parser.add_argument("--strict-bridge-config", action="store_true", help="Fail if any dataset bridge_config.json is missing.")
    parser.add_argument(
        "--evaluation-split",
        choices=("all", "test"),
        default="all",
        help=(
            "Rows to score. 'all' preserves the fixed aggregation; "
            "'test' projects to formal test rows before bridge scoring."
        ),
    )
    parser.add_argument(
        "--extra-run-root",
        action="append",
        default=[],
        help="Additional baseline run tree to aggregate (repeatable).",
    )
    parser.add_argument(
        "--allow-missing-agents",
        action="store_true",
        help="Allow the shared pre-archive baseline stage to defer agents until the immutable full-26 archive exists.",
    )
    parser.add_argument(
        "--exclude-run-prefix",
        action="append",
        default=[],
        help=(
            "Run-root-relative directory prefix to exclude from result "
            "metrics (repeatable), for example agentic/archive_backed."
        ),
    )
    args = parser.parse_args()
    metrics, manifest = aggregate_for_result(
        run_root=Path(args.run_root),
        out_metrics=Path(args.out_metrics),
        out_metric_slices=Path(args.out_metric_slices),
        out_manifest=Path(args.out_manifest),
        report_path=Path(args.report),
        bridge_config_root=Path(args.bridge_config_root) if args.bridge_config_root else None,
        strict_bridge_config=bool(args.strict_bridge_config),
        evaluation_split=str(args.evaluation_split),
        allow_missing_agents=bool(args.allow_missing_agents),
        extra_run_roots=[Path(path) for path in args.extra_run_root],
        exclude_run_prefixes=list(args.exclude_run_prefix),
    )
    print(
        "ok out_metrics={out_metrics} metric_rows={metric_rows} out_manifest={out_manifest} "
        "manifest_rows={manifest_rows} report={report}".format(
            out_metrics=args.out_metrics,
            metric_rows=len(metrics),
            out_manifest=args.out_manifest,
            manifest_rows=len(manifest),
            report=args.report,
        )
    )


if __name__ == "__main__":
    main()
