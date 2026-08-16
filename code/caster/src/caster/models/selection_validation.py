from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from caster.tasks.spec import TaskSpec, filter_rows_to_task_spec


SCORE_SOURCE = "task_specific_official_validation_folds"
REPORTING_MACRO_UNIT_ENTITY = "entity"
REPORTING_MACRO_UNIT_COUNTRY = "country"
SUPPORTED_REPORTING_MACRO_UNITS = {
    REPORTING_MACRO_UNIT_ENTITY,
    REPORTING_MACRO_UNIT_COUNTRY,
}


def robust_normalize_utility(raw_utility: pd.Series) -> tuple[pd.Series, float, float, str]:
    values = pd.to_numeric(raw_utility, errors="raise").astype(float)
    center = float(values.median())
    scale = float(1.4826 * (values - center).abs().median())
    source = "mad"
    if scale < 1e-12:
        scale = float((values.quantile(0.75) - values.quantile(0.25)) / 1.349)
        source = "iqr"
    if scale < 1e-12:
        return pd.Series(0.0, index=values.index), center, 0.0, "degenerate_zero"
    return (values - center) / scale, center, scale, source


def _normalized_weight(values: dict[object, float], keys: Iterable[object], name: str) -> dict[object, float]:
    selected = {key: float(values[key]) for key in keys}
    total = sum(selected.values())
    if total <= 0:
        raise ValueError(f"{name} weights must have positive total")
    return {key: value / total for key, value in selected.items()}


def _eligible_registry(
    registry: pd.DataFrame,
    excluded_model_ids: Iterable[str],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if "model_id" not in registry.columns:
        raise ValueError("candidate registry missing model_id")
    candidates = registry.copy()
    if "enabled" in candidates.columns:
        candidates = candidates[
            candidates["enabled"].astype(str).str.lower().isin({"true", "1", "t", "yes"})
        ]
    enabled_ids = set(candidates["model_id"].astype(str))
    excluded = tuple(sorted({str(value).strip() for value in excluded_model_ids if str(value).strip()}))
    if unknown := sorted(set(excluded) - enabled_ids):
        raise ValueError(f"excluded candidates are not enabled registry members: {unknown}")
    if excluded:
        candidates = candidates[~candidates["model_id"].astype(str).isin(excluded)].copy()
    if candidates.empty:
        raise ValueError("candidate eligibility policy removed every enabled candidate")
    return candidates, excluded


def _reporting_unit_column(
    rows: pd.DataFrame,
    spec: TaskSpec,
    reporting_macro_unit: str,
) -> tuple[pd.DataFrame, str]:
    unit = str(reporting_macro_unit).strip().lower()
    if unit not in SUPPORTED_REPORTING_MACRO_UNITS:
        raise ValueError(
            f"unsupported reporting_macro_unit={reporting_macro_unit!r}; "
            f"expected one of {sorted(SUPPORTED_REPORTING_MACRO_UNITS)}"
        )
    frame = rows.copy()
    if unit == REPORTING_MACRO_UNIT_ENTITY:
        frame["_reporting_macro_id"] = frame["entity_id"].astype(str)
        return frame, unit
    if str(spec.dataset) != "benchmark_a":
        raise ValueError("country reporting macro is supported only for Benchmark A")
    if "country" not in frame.columns:
        raise ValueError("Benchmark A country-macro scoring requires a country column")
    country = frame["country"].fillna("").astype(str).str.strip()
    if country.eq("").any() or country.str.lower().isin({"nan", "none", "null", "unknown", "all"}).any():
        raise ValueError("Benchmark A country-macro scoring requires a declared country for every row")
    frame["_reporting_macro_id"] = country
    return frame, unit


def _rmse_aggregation(reporting_macro_unit: str, *, equal_fold: bool) -> str:
    inner = f"component_strategy_horizon_{reporting_macro_unit}_mean_squared_error"
    return f"sqrt(equal_fold_mean({inner}))" if equal_fold else f"sqrt({inner})"


def task_macro_rmse(
    rows: pd.DataFrame,
    spec: TaskSpec,
    *,
    reporting_macro_unit: str = REPORTING_MACRO_UNIT_ENTITY,
) -> float:
    ""





    required = {"entity_id", "component", "forecast_strategy", "horizon", "observed_value", "pred_mean"}
    if missing := sorted(required - set(rows.columns)):
        raise ValueError(f"scoring rows missing columns {missing}")
    frame = filter_rows_to_task_spec(rows, spec, require_complete=True)
    frame, _ = _reporting_unit_column(frame, spec, reporting_macro_unit)
    frame["squared_error"] = np.square(
        pd.to_numeric(frame["pred_mean"], errors="raise") - pd.to_numeric(frame["observed_value"], errors="raise")
    )
    entity = frame.groupby(
        ["component", "forecast_strategy", "horizon", "_reporting_macro_id"], as_index=False
    )["squared_error"].mean()
    macro = entity.groupby(
        ["component", "forecast_strategy", "horizon"], as_index=False
    )["squared_error"].mean()
    component_mse: dict[str, float] = {}
    for component in spec.target_components:
        strategy_mse: dict[str, float] = {}
        for strategy in spec.forecast_strategies:
            part = macro[
                macro["component"].astype(str).eq(component)
                & macro["forecast_strategy"].astype(str).eq(strategy)
            ]
            expected = list(spec.horizon_weights[strategy])
            present = sorted(pd.to_numeric(part["horizon"], errors="raise").astype(int).unique().tolist())
            if present != sorted(expected):
                raise ValueError(
                    f"incomplete horizons for component={component} strategy={strategy}: {present} != {sorted(expected)}"
                )
            by_horizon = part.set_index(part["horizon"].astype(int))["squared_error"].astype(float).to_dict()
            hweights = _normalized_weight(dict(spec.horizon_weights[strategy]), expected, "horizon")
            strategy_mse[strategy] = sum(hweights[h] * float(by_horizon[h]) for h in expected)
        sweights = _normalized_weight(dict(spec.strategy_weights), spec.forecast_strategies, "strategy")
        component_mse[component] = sum(sweights[s] * strategy_mse[s] for s in spec.forecast_strategies)
    cweights = _normalized_weight(dict(spec.component_weights), spec.target_components, "component")
    mse = float(sum(cweights[c] * component_mse[c] for c in spec.target_components))
    return float(np.sqrt(max(mse, 0.0)))


def _point_archive(archive: pd.DataFrame, forecast_ids: set[str], model_ids: list[str]) -> pd.DataFrame:
    required = {"forecast_id", "model_id", "particle_id", "pred_mean"}
    if missing := sorted(required - set(archive.columns)):
        raise ValueError(f"archive missing columns {missing}")
    projected = archive[
        archive["forecast_id"].astype(str).isin(forecast_ids)
        & archive["model_id"].astype(str).isin(model_ids)
    ].copy()
    return projected.groupby(["forecast_id", "model_id"], as_index=False)["pred_mean"].mean()


def build_candidate_validation_scores(
    fold_manifest: pd.DataFrame,
    archive: pd.DataFrame,
    registry: pd.DataFrame,
    spec: TaskSpec,
    *,
    reporting_macro_unit: str = REPORTING_MACRO_UNIT_ENTITY,
    excluded_model_ids: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    required_manifest = {
        "task_id", "fold_id", "forecast_id", "entity_id", "component", "forecast_strategy", "horizon",
        "observed_value", "observed_mask", "release_time", "first_test_origin", "fold_manifest_sha256",
    }
    if missing := sorted(required_manifest - set(fold_manifest.columns)):
        raise ValueError(f"fold manifest missing columns {missing}")
    if set(fold_manifest["task_id"].astype(str)) != {spec.task_id}:
        raise ValueError("fold manifest task_id does not match TaskSpec")
    if set(fold_manifest.get("split", pd.Series(dtype=str)).astype(str)) != {"val"}:
        raise ValueError("candidate validation accepts official val folds only")
    manifest = fold_manifest[
        fold_manifest["observed_mask"].astype(str).str.lower().isin({"true", "1", "t", "yes"})
    ].copy()
    candidates, excluded = _eligible_registry(registry, excluded_model_ids)
    model_ids = sorted(candidates["model_id"].astype(str).unique().tolist())
    point = _point_archive(archive, set(manifest["forecast_id"].astype(str)), model_ids)
    expected_support = int(manifest["forecast_id"].nunique())
    support = point.groupby("model_id")["forecast_id"].nunique().reindex(model_ids, fill_value=0)
    if not support[support.ne(expected_support)].empty:
        raise ValueError(f"incomplete validation coverage: {support[support.ne(expected_support)].to_dict()}")
    joined = manifest.merge(point, on="forecast_id", how="inner", validate="many_to_many")
    rows: list[dict[str, object]] = []
    for (fold_id, model_id), group in joined.groupby(["fold_id", "model_id"], sort=True):
        rows.append({
            "task_id": spec.task_id,
            "model_id": str(model_id),
            "fold_id": str(fold_id),
            "n_events": int(group["forecast_id"].nunique()),
            "task_macro_rmse": task_macro_rmse(
                group,
                spec,
                reporting_macro_unit=reporting_macro_unit,
            ),
            "rmse_aggregation": _rmse_aggregation(reporting_macro_unit, equal_fold=False),
            "reporting_macro_unit": str(reporting_macro_unit),
            "score_source": SCORE_SOURCE,
            "max_label_release_time": str(pd.to_datetime(group["release_time"]).max().date()),
            "first_test_origin": str(group["first_test_origin"].iloc[0]),
            "fold_manifest_sha256": str(group["fold_manifest_sha256"].iloc[0]),
            "task_spec_sha256": spec.task_spec_sha256,
        })
    by_fold = pd.DataFrame(rows)
    by_fold["task_macro_mse"] = np.square(by_fold["task_macro_rmse"].astype(float))
    if by_fold.groupby("model_id")["fold_id"].nunique().nunique() != 1:
        raise ValueError("candidates do not have equal validation fold coverage")
    summary = by_fold.groupby(["task_id", "model_id"], as_index=False).agg(
        n_folds=("fold_id", "nunique"),
        n_events=("n_events", "sum"),
        task_macro_mse=("task_macro_mse", "mean"),
        max_label_release_time=("max_label_release_time", "max"),
        first_test_origin=("first_test_origin", "first"),
        fold_manifest_sha256=("fold_manifest_sha256", "first"),
        task_spec_sha256=("task_spec_sha256", "first"),
        score_source=("score_source", "first"),
        rmse_aggregation=("rmse_aggregation", "first"),
        reporting_macro_unit=("reporting_macro_unit", "first"),
    )
    summary["task_macro_rmse"] = np.sqrt(summary["task_macro_mse"].clip(lower=0.0))
    summary["raw_utility"] = -np.log1p(summary["task_macro_rmse"].astype(float))
    normalized, center, scale, scale_source = robust_normalize_utility(summary["raw_utility"])
    summary["robust_center"] = center
    summary["robust_scale"] = scale
    summary["validation_utility_robust_norm"] = normalized
    normalization = {
        "task_id": spec.task_id,
        "center": center,
        "scale": scale,
        "scale_source": scale_source,
        "mad_multiplier": 1.4826,
        "iqr_divisor": 1.349,
        "clipped": False,
        "score_source": SCORE_SOURCE,
        "rmse_aggregation": _rmse_aggregation(reporting_macro_unit, equal_fold=True),
        "reporting_macro_unit": str(reporting_macro_unit),
        "source_candidate_count": int(len(model_ids) + len(excluded)),
        "eligible_candidate_count": int(len(model_ids)),
        "excluded_model_ids": list(excluded),
    }
    validation = pd.DataFrame([
        {"task_id": spec.task_id, "check": "complete_eligible_candidate_bank", "status": "PASS", "value": len(model_ids)},
        {"task_id": spec.task_id, "check": "excluded_before_normalization", "status": "PASS", "value": ",".join(excluded)},
        {"task_id": spec.task_id, "check": "official_validation_only", "status": "PASS", "value": "true"},
        {"task_id": spec.task_id, "check": "test_rows_used", "status": "PASS", "value": 0},
        {"task_id": spec.task_id, "check": "rmse_one_final_square_root", "status": "PASS", "value": "true"},
        {"task_id": spec.task_id, "check": "reporting_macro_unit", "status": "PASS", "value": str(reporting_macro_unit)},
    ])
    return by_fold, summary, normalization, validation


def build_test_rmse_ranking(
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    registry: pd.DataFrame,
    spec: TaskSpec,
    *,
    reporting_macro_unit: str = REPORTING_MACRO_UNIT_ENTITY,
    excluded_model_ids: Iterable[str] = (),
) -> pd.DataFrame:
    test = ledger[
        ledger["split"].astype(str).eq("test")
        & ledger["component"].astype(str).isin(spec.target_components)
        & ledger["observed_mask"].astype(str).str.lower().isin({"true", "1", "t", "yes"})
    ].copy()
    test = filter_rows_to_task_spec(test, spec, require_complete=True)
    if test.empty:
        raise ValueError(f"official test split is empty for {spec.task_id}")
    candidates, _ = _eligible_registry(registry, excluded_model_ids)
    model_ids = sorted(candidates["model_id"].astype(str).unique().tolist())
    point = _point_archive(archive, set(test["forecast_id"].astype(str)), model_ids)
    expected_support = int(test["forecast_id"].nunique())
    support = point.groupby("model_id")["forecast_id"].nunique().reindex(model_ids, fill_value=0)
    if not support[support.ne(expected_support)].empty:
        raise ValueError(f"incomplete test coverage: {support[support.ne(expected_support)].to_dict()}")
    joined = test.merge(point, on="forecast_id", how="inner", validate="many_to_many")
    ranking = pd.DataFrame([
        {
            "task_id": spec.task_id,
            "model_id": model_id,
            "test_task_macro_rmse": task_macro_rmse(
                group,
                spec,
                reporting_macro_unit=reporting_macro_unit,
            ),
            "test_events": int(group["forecast_id"].nunique()),
            "reporting_macro_unit": str(reporting_macro_unit),
            "rmse_aggregation": _rmse_aggregation(reporting_macro_unit, equal_fold=False),
        }
        for model_id, group in joined.groupby("model_id", sort=True)
    ]).sort_values(["test_task_macro_rmse", "model_id"], kind="mergesort").reset_index(drop=True)
    ranking.insert(0, "test_rmse_rank", range(1, len(ranking) + 1))
    return ranking
