#!/usr/bin/env python3
""







from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import yaml


SCHEMA_VERSION = "result_metric_contract_v05_short_direct_long_recursive_endpoints"
FILL_VALUE = "all"
Z90 = 1.64485363
METRIC_EPS = 1e-8

PROTOCOL_METADATA_COLUMNS = [
    "forecast_strategy",
    "mode",
    "mode_kind",
    "country",
    "country_code",
    "jurisdiction",
    "entity_id",
    "component",
    "horizon",
]

REQUIRED_PROTOCOL_METADATA_COLUMNS = [
    "split",
    "mode",
    "mode_kind",
    "jurisdiction",
    "entity_id",
    "component",
    "horizon",
]

RESULT_SLICE_KEYS = [
    "dataset",
    "method",
    "method_group",
    "split",
    "mode",
    "mode_kind",
    "forecast_strategy",
    "country",
    "country_code",
    "jurisdiction",
    "entity_id",
    "component",
    "horizon",
    "horizon_group",
]

RESULT_GROUP_COLS = [
    *RESULT_SLICE_KEYS,
    "protocol_slice_status",
    "protocol_slice_reason",
]

REQUIRED_METRIC_COLUMNS = [
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
]


def bridge_nll_reason_for_split(value: object) -> str:
    ""

    split = str(value).strip().lower()
    if split in {"val", "validation"}:
        return (
            "shared bridge evaluated by causal full-validation replay; "
            "no test rows used for tuning"
        )
    if split == "test":
        return (
            "shared validation-calibrated bridge evaluated on test rows only"
        )
    label = split or "unspecified"
    return f"shared validation-calibrated bridge evaluated on {label} rows"


def formal_horizon_grid(config_path: str | Path | None = None) -> dict[str, dict[str, tuple[int, ...]]]:
    path = Path(config_path) if config_path is not None else Path(__file__).resolve().parents[1] / "configs/caster_task_specs_v20.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    by_task = {str(row["task_id"]): row for row in payload.get("tasks", [])}
    task_by_dataset = {
        "benchmark_a": "benchmark_a",
        "benchmark_b": "benchmark_b_pooled",
        "benchmark_b_covid": "benchmark_b_covid",
        "benchmark_b_flu": "benchmark_b_flu",
        "benchmark_b_pooled": "benchmark_b_pooled",
    }
    grid: dict[str, dict[str, tuple[int, ...]]] = {}
    for dataset, task_id in task_by_dataset.items():
        row = by_task[task_id]
        strategies = [str(value) for value in row["forecast_strategies"]]
        grid[dataset] = {
            strategies[0]: tuple(int(value) for value in row["direct_horizons"]),
            strategies[1]: tuple(int(value) for value in row["recursive_horizons"]),
        }
    return grid


def normalize_dataset(value: object) -> str:
    text = str(value)
    if text in {"benchmark_b_covid", "benchmark_b_flu", "benchmark_b_pooled"}:
        return text
    if text.startswith("benchmark_a"):
        return "benchmark_a"
    if text.startswith("benchmark_b"):
        return "benchmark_b"
    return text


def _clean_text(value: object, default: str = FILL_VALUE) -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na"}:
        return default
    return text


def _is_missing_text(value: object, *, allow_fill: bool = False) -> bool:
    text = _clean_text(value, "")
    if not text:
        return True
    if text.lower() == "unknown":
        return True
    if not allow_fill and text == FILL_VALUE:
        return True
    return False


def _as_int(value: object, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def horizon_group_for(dataset: object, mode: object, mode_kind: object, horizon: object) -> str:
    ""






    ds = normalize_dataset(dataset)
    h = _as_int(horizon)
    if ds.startswith("benchmark_b"):
        return "short" if h in {1, 2} else "long"
    return "short" if h in {1, 3} else "long"


def _infer_mode_kind(mode: object, mode_kind: object) -> str:
    current = _clean_text(mode_kind, "")
    if current and current != FILL_VALUE and current != "unknown":
        return current
    text = _clean_text(mode, "").lower()
    if text.startswith("direct"):
        return "direct"
    if text.startswith("multi") or text.startswith("rollout"):
        return "rollout"
    return ""


def _infer_forecast_strategy(mode: object, mode_kind: object, strategy: object) -> str:
    current = _clean_text(strategy, "").lower()
    if current in {"direct", "recursive_rollout"}:
        return current
    if current in {"rollout", "recursive"}:
        return "recursive_rollout"
    kind = _infer_mode_kind(mode, mode_kind).lower()
    return "recursive_rollout" if kind == "rollout" else "direct"


def _declared_benchmark_a_entity(row: pd.Series) -> str:
    for col in ["country", "country_code", "jurisdiction", "entity_id"]:
        value = _clean_text(row.get(col), "")
        if value and value != FILL_VALUE:
            return value
    return ""


def _declared_benchmark_b_entity(row: pd.Series) -> str:
    for col in ["jurisdiction", "entity_id"]:
        value = _clean_text(row.get(col), "")
        if value and value != FILL_VALUE:
            return value
    return ""


def _missing_protocol_fields(row: pd.Series) -> list[str]:
    missing: list[str] = []
    for col in REQUIRED_PROTOCOL_METADATA_COLUMNS:
        if col == "horizon":
            if _as_int(row.get(col), 0) <= 0:
                missing.append(col)
            continue
        if _is_missing_text(row.get(col), allow_fill=False):
            missing.append(col)
    return missing


def apply_result_metric_contract(df: pd.DataFrame, *, method_group: str | None = None) -> pd.DataFrame:
    ""







    out = df.copy()
    if "dataset" not in out.columns and "dataset_key" in out.columns:
        out["dataset"] = out["dataset_key"]
    if "dataset" not in out.columns:
        out["dataset"] = FILL_VALUE
    out["dataset"] = out["dataset"].map(normalize_dataset)
    if method_group is not None:
        out["method_group"] = method_group
    if "method_group" not in out.columns:
        out["method_group"] = FILL_VALUE
    if "method" not in out.columns:
        out["method"] = FILL_VALUE
    if "split" not in out.columns:
        out["split"] = FILL_VALUE

    if "raw_entity_id" not in out.columns and "entity_id" in out.columns:
        out["raw_entity_id"] = out["entity_id"]
    for col in PROTOCOL_METADATA_COLUMNS:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].map(lambda value: _clean_text(value, ""))
    out["mode_kind"] = [_infer_mode_kind(mode, kind) for mode, kind in zip(out["mode"], out["mode_kind"])]
    out["forecast_strategy"] = [
        _infer_forecast_strategy(mode, kind, strategy)
        for mode, kind, strategy in zip(out["mode"], out["mode_kind"], out["forecast_strategy"])
    ]
    out["horizon"] = pd.to_numeric(out["horizon"], errors="coerce").fillna(0).astype(int)

    for idx, row in out.iterrows():
        dataset = normalize_dataset(row["dataset"])
        if dataset == "benchmark_a":
            declared = _declared_benchmark_a_entity(row)
            if declared:
                out.at[idx, "entity_id"] = declared
                out.at[idx, "jurisdiction"] = declared
            if _clean_text(row.get("country"), "") == "":
                out.at[idx, "country"] = declared or FILL_VALUE
            if _clean_text(row.get("country_code"), "") == "":
                out.at[idx, "country_code"] = FILL_VALUE
        elif dataset.startswith("benchmark_b"):
            declared = _declared_benchmark_b_entity(row)
            if declared:
                out.at[idx, "entity_id"] = declared
                out.at[idx, "jurisdiction"] = declared
            out.at[idx, "country"] = FILL_VALUE
            out.at[idx, "country_code"] = FILL_VALUE

    for col in PROTOCOL_METADATA_COLUMNS:
        if col == "horizon":
            continue
        out[col] = out[col].map(lambda value: _clean_text(value, FILL_VALUE))

    out["horizon_group"] = [
        horizon_group_for(ds, mode, kind, horizon)
        for ds, mode, kind, horizon in zip(out["dataset"], out["mode"], out["mode_kind"], out["horizon"])
    ]
    missing_by_row = [_missing_protocol_fields(row) for _, row in out.iterrows()]
    out["protocol_slice_status"] = [
        "ok" if not missing else "missing_protocol_fields"
        for missing in missing_by_row
    ]
    out["protocol_slice_reason"] = [
        "" if not missing else "missing: " + ",".join(sorted(missing))
        for missing in missing_by_row
    ]
    out["metric_slice_schema"] = SCHEMA_VERSION
    out["aggregation_schema"] = ",".join(RESULT_SLICE_KEYS)
    return out


def filter_to_formal_horizon_grid(
    df: pd.DataFrame,
    *,
    config_path: str | Path | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    ""

    rows = apply_result_metric_contract(df)
    grid = formal_horizon_grid(config_path)
    recognized = rows["dataset"].astype(str).isin(grid)
    if strict and not recognized.all():
        unknown = sorted(rows.loc[~recognized, "dataset"].astype(str).unique())
        raise ValueError(f"formal horizon grid has no task declaration for datasets={unknown}")
    eligible = pd.Series(False, index=rows.index)
    labels = pd.Series("", index=rows.index, dtype=str)
    for dataset, strategies in grid.items():
        dataset_mask = rows["dataset"].astype(str).eq(dataset)
        labels.loc[dataset_mask] = ";".join(
            f"{strategy}=" + ",".join(str(horizon) for horizon in horizons)
            for strategy, horizons in strategies.items()
        )
        for strategy, horizons in strategies.items():
            eligible |= (
                dataset_mask
                & rows["forecast_strategy"].astype(str).eq(strategy)
                & rows["horizon"].astype(int).isin(horizons)
            )
    rows["formal_horizon_eligible"] = eligible
    rows["formal_horizon_grid"] = labels
    projected = rows.loc[eligible].copy()
    if strict and not rows.empty and projected.empty:
        raise ValueError("formal horizon projection removed every metric row")
    return projected


def numeric_series(df: pd.DataFrame, column: str, default: float = math.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def truthy_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes", "y"})


def interval_score(y: pd.Series, lower: pd.Series, upper: pd.Series, alpha: float) -> pd.Series:
    yy = y.astype(float)
    lo = lower.astype(float)
    hi = upper.astype(float)
    return (hi - lo) + (2.0 / alpha) * (lo - yy) * (yy < lo) + (2.0 / alpha) * (yy - hi) * (yy > hi)


def canonical_forecast_strategy(row: pd.Series) -> str:
    declared = str(row.get("forecast_strategy", "")).strip().lower()
    if declared in {"recursive_rollout", "rollout", "recursive"}:
        return "recursive_rollout"
    if declared == "direct":
        return "direct"
    kind = str(row.get("mode_kind", "")).strip().lower()
    mode = str(row.get("mode", "")).strip().lower()
    return (
        "recursive_rollout"
        if kind in {"rollout", "autoregressive"}
        or mode.startswith(("rollout", "multi"))
        else "direct"
    )


def strategy_macro_values(
    group: pd.DataFrame,
    metric_columns: Sequence[str],
) -> tuple[dict[str, float], dict[str, object]]:
    ""







    data = group.copy()
    data["_strategy"] = [
        canonical_forecast_strategy(row) for _, row in data.iterrows()
    ]
    data["_horizon"] = (
        pd.to_numeric(data.get("horizon", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    fold_column = next(
        (
            column
            for column in ("validation_fold", "fold_id", "fold")
            if column in data.columns
        ),
        "",
    )
    if fold_column:
        data["_fold"] = data[fold_column].fillna("fold_0").astype(str)
    else:
        data["_fold"] = "fold_0"

    entity_column = next(
        (column for column in ("entity_id", "jurisdiction") if column in data.columns),
        "",
    )
    unit_columns: list[str] = []
    if entity_column:
        data[entity_column] = data[entity_column].fillna("all").astype(str)
        unit_columns.append(entity_column)
    if "component" in data.columns:
        data["component"] = data["component"].fillna("all").astype(str)
        unit_columns.append("component")

    usable_metrics = [column for column in metric_columns if column in data.columns]
    linear_metrics = [column for column in usable_metrics if column != "rmse"]
    aggregate_metrics = list(linear_metrics)
    for column in linear_metrics:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if "rmse" in usable_metrics:
        data["_mse"] = pd.to_numeric(data["rmse"], errors="coerce").pow(2)
        aggregate_metrics.append("_mse")
    if not aggregate_metrics:
        return {}, {
            "forecast_strategies": "",
            "strategy_count": 0,
            "fold_count": int(data["_fold"].nunique()),
            "aggregation_order": (
                "origin_mean_within_unit_then_entity_macro_then_horizon_macro_"
                "then_equal_strategy_macro_then_equal_fold_macro"
            ),
        }

    unit_keys = ["_fold", "_strategy", "_horizon", *unit_columns]
    unit_origin_macro = (
        data.groupby(unit_keys, dropna=False)[aggregate_metrics]
        .mean()
        .reset_index()
    )
    entity_macro = (
        unit_origin_macro.groupby(
            ["_fold", "_strategy", "_horizon"], dropna=False
        )[aggregate_metrics]
        .mean()
        .reset_index()
    )
    horizon_macro = (
        entity_macro.groupby(["_fold", "_strategy"], dropna=False)[
            aggregate_metrics
        ]
        .mean()
        .reset_index()
    )
    strategy_macro = (
        horizon_macro.groupby("_fold", dropna=False)[aggregate_metrics]
        .mean()
        .reset_index()
    )

    def finite_mean(values: pd.Series) -> float:
        numeric = pd.to_numeric(values, errors="coerce")
        numeric = numeric[numeric.map(math.isfinite)]
        return float(numeric.mean()) if not numeric.empty else float("nan")

    values = {column: finite_mean(strategy_macro[column]) for column in linear_metrics}
    if "_mse" in strategy_macro.columns:
        mse = finite_mean(strategy_macro["_mse"])
        values["rmse"] = (
            float(math.sqrt(mse))
            if math.isfinite(mse) and mse >= 0.0
            else float("nan")
        )

    strategies = sorted(horizon_macro["_strategy"].astype(str).unique())
    weights = {
        strategy: 1.0 / len(strategies) for strategy in strategies
    } if strategies else {}
    validation = {
        "forecast_strategies": ";".join(strategies),
        "strategy_count": int(len(strategies)),
        "strategy_weights": ";".join(
            f"{strategy}={weights[strategy]:.6f}" for strategy in strategies
        ),
        "fold_count": int(strategy_macro["_fold"].nunique()),
        "unit_origin_macro_rows": int(len(unit_origin_macro)),
        "entity_macro_rows": int(len(entity_macro)),
        "strategy_horizon_rows": int(len(horizon_macro)),
        "aggregation_order": (
            "origin_mean_within_unit_then_entity_macro_then_horizon_macro_"
            "then_equal_strategy_macro_then_equal_fold_macro"
        ),
        "rmse_formula": (
            "sqrt(equal_fold_mean(equal_strategy_mean(horizon_mean("
            "entity_mean(unit_origin_mean_squared_error)))))"
        ),
    }
    return values, validation


def metric_slices_from_scored_rows(
    rows: pd.DataFrame,
    *,
    source: str | Path,
    y_col: str,
    pred_col: str,
    median_col: str | None = None,
    lower_50_col: str,
    upper_50_col: str,
    lower_90_col: str,
    upper_90_col: str,
    nll_col: str = "bridge_nll",
    method_group: str | None = None,
) -> pd.DataFrame:
    ""

    data = apply_result_metric_contract(rows, method_group=method_group)
    if "observed_mask" in data.columns:
        data = data[truthy_series(data["observed_mask"])].copy()
    y = numeric_series(data, y_col)
    pred = numeric_series(data, pred_col)
    median = numeric_series(
        data, pred_col if median_col is None else median_col
    )
    lower_50 = numeric_series(data, lower_50_col).clip(lower=0.0)
    upper_50 = numeric_series(data, upper_50_col).clip(lower=0.0)
    lower_90 = numeric_series(data, lower_90_col).clip(lower=0.0)
    upper_90 = numeric_series(data, upper_90_col).clip(lower=0.0)
    err = pred - y
    data["_abs_err"] = err.abs()
    data["_sq_err"] = err * err
    data["_nll"] = numeric_series(data, nll_col)
    data["_coverage_90"] = ((y >= lower_90) & (y <= upper_90)).astype(float)
    data["_width_90"] = upper_90 - lower_90
    data["_coverage_50"] = ((y >= lower_50) & (y <= upper_50)).astype(float)
    data["_width_50"] = upper_50 - lower_50
                                                                           
                                                                        
                                                                     
                                                                              
                          
    sigma = ((upper_90 - lower_90).abs() / (2.0 * Z90)).clip(lower=METRIC_EPS)
    z = (y - pred) / sigma
    data["_gaussian_nll"] = 0.5 * math.log(2.0 * math.pi) + sigma.map(math.log) + 0.5 * z.pow(2)
    density = (-0.5 * z.pow(2)).map(math.exp) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + (z / math.sqrt(2.0)).map(math.erf))
    data["_crps_gaussian"] = sigma * (
        z * (2.0 * cdf - 1.0) + 2.0 * density - 1.0 / math.sqrt(math.pi)
    )
    data["_wis"] = (
        0.5 * (median - y).abs()
        + 0.50 / 2.0 * interval_score(y, lower_50, upper_50, 0.50)
        + 0.10 / 2.0 * interval_score(y, lower_90, upper_90, 0.10)
    ) / 2.5
                                                                    
                                                                           
                                        
    fold_column = next(
        (
            column
            for column in ("validation_fold", "fold_id", "fold")
            if column in data.columns
        ),
        "",
    )
    group_columns = list(RESULT_GROUP_COLS)
    if fold_column:
        group_columns.append(fold_column)
    grouped = data.groupby(group_columns, dropna=False).agg(
        n=("_abs_err", "size"),
        mae=("_abs_err", "mean"),
        mse=("_sq_err", "mean"),
        nll=("_nll", "mean"),
        bridge_nll=("_nll", "mean"),
        gaussian_nll=("_gaussian_nll", "mean"),
        diagnostic_gaussian_nll=("_gaussian_nll", "mean"),
        crps=("_crps_gaussian", "mean"),
        crps_gaussian=("_crps_gaussian", "mean"),
        coverage_90=("_coverage_90", "mean"),
        width_90=("_width_90", "mean"),
        coverage_50=("_coverage_50", "mean"),
        width_50=("_width_50", "mean"),
        wis=("_wis", "mean"),
    ).reset_index()
    grouped["rmse"] = grouped["mse"].pow(0.5)
    grouped["nll_status"] = "ok"
    grouped["nll_reason"] = grouped["split"].map(bridge_nll_reason_for_split)
    grouped["source_artifact"] = str(source)
    grouped["status"] = "ok"
    grouped["metric_slice_schema"] = SCHEMA_VERSION
    grouped["aggregation_schema"] = ",".join(RESULT_SLICE_KEYS)
    return grouped.drop(columns=["mse"])


def combine_metric_slices(slices: pd.DataFrame) -> pd.DataFrame:
    ""

    if slices.empty:
        return slices.copy()
    rows = apply_result_metric_contract(slices)
    rows["n"] = numeric_series(rows, "n", 1.0).fillna(1.0).clip(lower=0.0)
    weight = rows["n"].replace(0, math.nan)
    rows["_abs_sum"] = numeric_series(rows, "mae") * weight
    rows["_sq_sum"] = numeric_series(rows, "rmse").pow(2) * weight
    for col in ["nll", "bridge_nll", "coverage_90", "width_90", "coverage_50", "width_50", "wis"]:
        rows[f"_{col}_sum"] = numeric_series(rows, col) * weight

    grouped = rows.groupby(RESULT_GROUP_COLS, dropna=False).agg(
        n=("n", "sum"),
        mae_sum=("_abs_sum", "sum"),
        sq_sum=("_sq_sum", "sum"),
        nll_sum=("_nll_sum", "sum"),
        bridge_nll_sum=("_bridge_nll_sum", "sum"),
        coverage_90_sum=("_coverage_90_sum", "sum"),
        width_90_sum=("_width_90_sum", "sum"),
        coverage_50_sum=("_coverage_50_sum", "sum"),
        width_50_sum=("_width_50_sum", "sum"),
        wis_sum=("_wis_sum", "sum"),
        source_artifact=("source_artifact", lambda s: ";".join(sorted({str(x) for x in s if str(x)}))),
        status=("status", lambda s: "ok" if set(map(str, s)) <= {"ok", "numeric"} else ";".join(sorted(set(map(str, s))))),
    ).reset_index()
    denom = grouped["n"].replace(0, math.nan)
    grouped["mae"] = grouped["mae_sum"] / denom
    grouped["rmse"] = (grouped["sq_sum"] / denom).pow(0.5)
    for col in ["nll", "bridge_nll", "coverage_90", "width_90", "coverage_50", "width_50", "wis"]:
        grouped[col] = grouped[f"{col}_sum"] / denom
    grouped["nll_status"] = "ok"
    grouped["nll_reason"] = grouped["split"].map(bridge_nll_reason_for_split)
    grouped["metric_slice_schema"] = SCHEMA_VERSION
    grouped["aggregation_schema"] = ",".join(RESULT_SLICE_KEYS)
    drop_cols = [c for c in grouped.columns if c.endswith("_sum") or c in {"mae_sum", "sq_sum"}]
    return grouped.drop(columns=drop_cols)


def schema_payload() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "result_slice_keys": RESULT_SLICE_KEYS,
        "result_group_cols": RESULT_GROUP_COLS,
        "required_metric_columns": REQUIRED_METRIC_COLUMNS,
        "horizon_group_rule": {
            "benchmark_a": "direct h1,h3 -> short; recursive_rollout h7 -> long",
            "benchmark_b*": "direct h1,h2 -> short; recursive_rollout h4 -> long",
        },
        "formal_scoring_grid": {
            "benchmark_a": "direct h1,h3; recursive_rollout h7",
            "benchmark_b*": "direct h1,h2; recursive_rollout h4",
        },
        "strategy_weight_rule": "entity macro, then horizon macro within strategy, then direct/recursive_rollout 50/50",
        "declared_unit_rule": {
            "benchmark_a": "country is the declared reporting slice; raw graph nodes are collapsed",
            "benchmark_b*": "jurisdiction x target component is the declared reporting slice",
        },
        "global_rmse": "not used for the primary result summary; extended/sensitivity only",
    }


def missing_required_columns(columns: Iterable[str]) -> list[str]:
    present = set(columns)
    return sorted(set(RESULT_GROUP_COLS + REQUIRED_METRIC_COLUMNS) - present)
