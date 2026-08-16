from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd


TASKS = ("benchmark_a", "benchmark_b_covid", "benchmark_b_flu")
LINEAR_METRICS = (
    "mae",
    "nll",
    "wis",
    "coverage_90",
    "width_90",
    "coverage_50",
    "width_50",
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SystemExit(f"{label} missing columns: {missing}")


def finite_mean(values: pd.Series) -> float:
    numbers = pd.to_numeric(values, errors="coerce")
    numbers = numbers[numbers.map(lambda value: pd.notna(value) and math.isfinite(float(value)))]
    return float(numbers.mean()) if not numbers.empty else float("nan")


def task_column(frame: pd.DataFrame) -> str:
    for name in ("task", "task_id", "dataset_key", "dataset"):
        if name in frame.columns:
            return name
    raise SystemExit("input has no task column")


def method_column(frame: pd.DataFrame) -> str:
    for name in ("method", "method_key", "method_id", "included_methods"):
        if name in frame.columns:
            return name
    raise SystemExit("input has no method column")


def method_group_column(frame: pd.DataFrame) -> str:
    for name in ("method_group", "source_method_group", "family"):
        if name in frame.columns:
            return name
    return ""


def strategy(frame: pd.DataFrame) -> pd.Series:
    if "forecast_strategy" in frame.columns:
        declared = frame["forecast_strategy"].fillna("").astype(str).str.strip().str.lower()
    else:
        declared = pd.Series("", index=frame.index, dtype=str)
    if "mode_kind" in frame.columns:
        kind = frame["mode_kind"].fillna("").astype(str).str.strip().str.lower()
    else:
        kind = pd.Series("", index=frame.index, dtype=str)
    if "mode" in frame.columns:
        mode = frame["mode"].fillna("").astype(str).str.strip().str.lower()
    else:
        mode = pd.Series("", index=frame.index, dtype=str)
    recursive = (
        declared.isin({"recursive_rollout", "rollout", "recursive"})
        | kind.isin({"rollout", "autoregressive"})
        | mode.str.startswith(("rollout", "multi", "recursive"))
    )
    return pd.Series("direct", index=frame.index).where(~recursive, "recursive_rollout")


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    method_name = method_column(data)
    if "dataset" in data.columns:
        tasks = data["dataset"].fillna("").astype(str).str.strip()
    else:
        tasks = pd.Series("", index=data.index, dtype=str)
    for name in ("dataset_key", "task_id", "task"):
        if name in data.columns:
            values = data[name].fillna("").astype(str).str.strip()
            usable = ~values.str.lower().isin({"", "nan", "none", "null"})
            tasks = tasks.where(~usable, values)
    tasks = tasks.map(
        lambda value: (
            "benchmark_a"
            if str(value).startswith("benchmark_a")
            else str(value)
        )
    )
    component_name = "component" if "component" in data.columns else "target_component" if "target_component" in data.columns else ""
    if component_name:
        components = data[component_name].fillna("").astype(str).str.strip().str.lower()
        tasks = tasks.where(~(tasks.eq("benchmark_b") & components.isin({"covid", "covid_adm", "covid_admissions", "covid_adm_per100k"})), "benchmark_b_covid")
        tasks = tasks.where(~(tasks.eq("benchmark_b") & components.isin({"flu", "influenza", "flu_adm", "flu_admissions", "flu_adm_per100k"})), "benchmark_b_flu")
    data["task"] = tasks
    data["method"] = (
        data[method_name]
        .astype(str)
        .str.strip()
        .replace({"agent_react": "react"})
    )
    if "split" in data.columns:
        data = data[data["split"].astype(str).str.strip().str.lower().eq("test")].copy()
    excluded = (
        data["task"].eq("benchmark_b")
        | data["task"].str.contains("pooled", case=False, na=False)
        | data["method"].str.contains("pooled", case=False, na=False)
    )
    data = data[~excluded].copy()
    unknown = sorted(set(data["task"]) - set(TASKS))
    if unknown:
        raise SystemExit(f"unsupported tasks: {unknown}")
    if "nll" not in data.columns:
        data["nll"] = float("nan")
    for name in ("bridge_nll", "gaussian_nll"):
        if name in data.columns:
            data["nll"] = pd.to_numeric(data["nll"], errors="coerce").fillna(
                pd.to_numeric(data[name], errors="coerce")
            )
    require(data, ["horizon", "rmse", "mae", "nll", "wis", "coverage_90", "width_90"], "input")
    data["horizon"] = pd.to_numeric(data["horizon"], errors="raise").astype(int)
    data["_strategy"] = strategy(data)
    keep = (
        (
            data["task"].eq("benchmark_a")
            & (
                (data["_strategy"].eq("direct") & data["horizon"].isin({1, 3}))
                | (data["_strategy"].eq("recursive_rollout") & data["horizon"].eq(7))
            )
        )
        | (
            data["task"].isin({"benchmark_b_covid", "benchmark_b_flu"})
            & (
                (data["_strategy"].eq("direct") & data["horizon"].isin({1, 2}))
                | (data["_strategy"].eq("recursive_rollout") & data["horizon"].eq(4))
            )
        )
    )
    data = data[keep].copy()
    data["horizon_group"] = "long"
    data.loc[
        data["task"].eq("benchmark_a") & data["horizon"].isin({1, 3}),
        "horizon_group",
    ] = "short"
    data.loc[
        data["task"].isin({"benchmark_b_covid", "benchmark_b_flu"})
        & data["horizon"].isin({1, 2}),
        "horizon_group",
    ] = "short"
    group_name = method_group_column(data)
    data["method_group"] = data[group_name].fillna("").astype(str) if group_name else ""
    if "n" not in data.columns:
        data["n"] = 1.0
    data["n"] = pd.to_numeric(data["n"], errors="raise").astype(float)
    numeric_metrics = ["rmse", *LINEAR_METRICS]
    for metric in numeric_metrics:
        if metric in data.columns:
            data[metric] = pd.to_numeric(data[metric], errors="coerce")
    identity = ["task", "method", "_strategy", "horizon", *unit_columns(data)]
    fold_name = fold_column(data)
    if fold_name:
        identity.append(fold_name)
    duplicates = data[data.duplicated(identity, keep=False)].copy()
    for key, group in duplicates.groupby(identity, dropna=False):
        for name in ["n", *numeric_metrics]:
            if name in group.columns:
                values = pd.to_numeric(group[name], errors="coerce").dropna().to_numpy(dtype=float)
                if values.size and not all(math.isclose(float(value), float(values[0]), rel_tol=0.0, abs_tol=1e-12) for value in values):
                    raise SystemExit(f"conflicting duplicate metric rows for {key}: {name}")
    return data.drop_duplicates(identity, keep="first").reset_index(drop=True)


def fold_column(frame: pd.DataFrame) -> str:
    for name in ("validation_fold", "fold_id", "fold"):
        if name in frame.columns:
            return name
    return ""


def unit_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for name in ("entity_id", "macro_entity_id", "jurisdiction", "country"):
        if name in frame.columns:
            columns.append(name)
            break
    for name in ("component", "target_component"):
        if name in frame.columns:
            columns.append(name)
            break
    return columns


def macro(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        raise SystemExit("empty metric slice")
    data = frame.copy()
    fold_name = fold_column(data)
    data["_fold"] = data[fold_name].fillna("fold_0").astype(str) if fold_name else "fold_0"
    units = unit_columns(data)
    for name in units:
        data[name] = data[name].fillna("all").astype(str)
    metrics = [name for name in LINEAR_METRICS if name in data.columns]
    for name in ("rmse", *metrics):
        values = pd.to_numeric(data[name], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise SystemExit(f"non-finite values in {name}")
        data[name] = values.astype(float)
    data["_mse"] = data["rmse"].pow(2)
    aggregate = ["_mse", *metrics]
    unit = data.groupby(["_fold", "_strategy", "horizon", *units], dropna=False)[aggregate].mean().reset_index()
    entities = unit.groupby(["_fold", "_strategy", "horizon"], dropna=False)[aggregate].mean().reset_index()
    horizons = entities.groupby(["_fold", "_strategy"], dropna=False)[aggregate].mean().reset_index()
    strategies = horizons.groupby("_fold", dropna=False)[aggregate].mean().reset_index()
    result = {name: finite_mean(strategies[name]) for name in metrics}
    mse = finite_mean(strategies["_mse"])
    result["rmse"] = float(math.sqrt(mse)) if math.isfinite(mse) and mse >= 0.0 else float("nan")
    return result


def endpoint_macro(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        raise SystemExit("empty endpoint slice")
    data = frame.copy()
    fold_name = fold_column(data)
    data["_fold"] = data[fold_name].fillna("fold_0").astype(str) if fold_name else "fold_0"
    units = unit_columns(data)
    for name in units:
        data[name] = data[name].fillna("all").astype(str)
    metrics = [name for name in LINEAR_METRICS if name in data.columns]
    for name in ("rmse", *metrics):
        values = pd.to_numeric(data[name], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise SystemExit(f"non-finite values in {name}")
        data[name] = values.astype(float)
    data["_mse"] = data["rmse"].pow(2)
    aggregate = ["_mse", *metrics]
    unit = data.groupby(["_fold", *units], dropna=False)[aggregate].mean().reset_index()
    folds = unit.groupby("_fold", dropna=False)[aggregate].mean().reset_index()
    result = {name: finite_mean(folds[name]) for name in metrics}
    mse = finite_mean(folds["_mse"])
    result["rmse"] = float(math.sqrt(mse)) if math.isfinite(mse) and mse >= 0.0 else float("nan")
    return result


def unique_text(values: pd.Series) -> str:
    return ";".join(sorted({str(value).strip() for value in values if str(value).strip()}))


def weighted(values: pd.Series, weights: pd.Series) -> float:
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "weight": pd.to_numeric(weights, errors="coerce"),
        }
    )
    frame = frame[
        frame["value"].map(lambda value: pd.notna(value) and math.isfinite(float(value)))
        & frame["weight"].map(lambda value: pd.notna(value) and math.isfinite(float(value)) and float(value) > 0.0)
    ]
    if frame.empty:
        return float("nan")
    return float((frame["value"] * frame["weight"]).sum() / frame["weight"].sum())


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")
