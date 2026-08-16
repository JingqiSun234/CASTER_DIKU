from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


COVID_COMPONENT = "covid_adm_per100k"
FLU_COMPONENT = "flu_adm_per100k"

COMPONENT_COLUMNS = ("component", "target_component", "target_name", "target", "pathogen")
COMPONENT_ALIASES = {
    "covid": COVID_COMPONENT,
    "covid19": COVID_COMPONENT,
    "covid_19": COVID_COMPONENT,
    "sars_cov_2": COVID_COMPONENT,
    "sars-cov-2": COVID_COMPONENT,
    "covid_adm": COVID_COMPONENT,
    "covid_admissions": COVID_COMPONENT,
    "covid_adm_per100k": COVID_COMPONENT,
    "flu": FLU_COMPONENT,
    "influenza": FLU_COMPONENT,
    "influenza_a": FLU_COMPONENT,
    "flu_adm": FLU_COMPONENT,
    "flu_admissions": FLU_COMPONENT,
    "flu_adm_per100k": FLU_COMPONENT,
}

BENCHMARK_B_TASKS = {
    "benchmark_b_covid": (COVID_COMPONENT,),
    "benchmark_b_flu": (FLU_COMPONENT,),
    "benchmark_b_pooled": (COVID_COMPONENT, FLU_COMPONENT),
}


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    dataset: str
    components: tuple[str, ...]
    posterior_scope: str

    @property
    def component_label(self) -> str:
        return ";".join(self.components)


def canonical_component(value: object) -> str:
    text = str(value).strip()
    key = text.lower().replace(" ", "_")
    return COMPONENT_ALIASES.get(key, text)


def parse_components(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [x.strip() for x in value.replace(";", ",").split(",") if x.strip()]
    else:
        parts = [str(x).strip() for x in value if str(x).strip()]
    return tuple(canonical_component(x) for x in parts)


def component_column(frame: pd.DataFrame) -> str | None:
    for col in COMPONENT_COLUMNS:
        if col in frame.columns:
            return col
    return None


def infer_dataset(frame: pd.DataFrame, default: str = "") -> str:
    if "dataset" not in frame.columns or frame.empty:
        return default
    values = frame["dataset"].dropna().astype(str).unique()
    return str(values[0]) if len(values) else default


def task_from_args(
    *,
    task_id: str | None = None,
    target_components: str | Iterable[str] | None = None,
    posterior_scope: str | None = None,
    dataset: str | None = None,
) -> BenchmarkTask | None:
    tid = str(task_id or "").strip()
    components = parse_components(target_components)
    if tid in BENCHMARK_B_TASKS:
        canonical = BENCHMARK_B_TASKS[tid]
        if components and components != canonical:
            raise ValueError(f"task_id={tid!r} conflicts with target_components={components!r}")
        components = canonical
        ds = "benchmark_b"
    else:
        ds = str(dataset or "").strip()
    if not tid and not components:
        return None
    if not components:
        raise ValueError("target components are required when task_id is not a known Benchmark B task")
    if not tid:
        tid = f"{ds or 'task'}_{'_'.join(components)}"
    if not ds:
        ds = "benchmark_b" if tid.startswith("benchmark_b") else "unknown"
    scope = str(posterior_scope or "").strip()
    if not scope:
        scope = "pooled_shared_posterior" if tid == "benchmark_b_pooled" or len(components) > 1 else "component_stratified"
    if scope == "pooled_sensitivity":
        scope = "pooled_shared_posterior"
    if scope not in {"component_stratified", "pooled_shared_posterior"}:
        raise ValueError(f"unknown posterior_scope={scope!r}")
    if scope == "component_stratified" and len(components) != 1:
        raise ValueError("component_stratified task must have exactly one target component")
    return BenchmarkTask(task_id=tid, dataset=ds, components=components, posterior_scope=scope)


def filter_frame_for_task(frame: pd.DataFrame, task: BenchmarkTask | None, *, frame_name: str) -> pd.DataFrame:
    if task is None:
        return frame.copy()
    col = component_column(frame)
    if col is None:
        raise ValueError(f"{frame_name} has no component/target column for task {task.task_id}")
    out = frame.copy()
    canonical = out[col].map(canonical_component)
    out = out[canonical.isin(set(task.components))].copy()
    if out.empty:
        raise ValueError(f"{frame_name} has no rows for task {task.task_id} components={task.components}")
    return out


def filter_ledger_archive_for_task(
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    task: BenchmarkTask | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if task is None:
        return ledger.copy(), archive.copy()
    filtered_ledger = filter_frame_for_task(ledger, task, frame_name="ledger")
    ids = set(filtered_ledger["forecast_id"].astype(str)) if "forecast_id" in filtered_ledger.columns else set()
    filtered_archive = filter_frame_for_task(archive, task, frame_name="archive")
    if ids:
        filtered_archive = filtered_archive[filtered_archive["forecast_id"].astype(str).isin(ids)].copy()
    if filtered_archive.empty:
        raise ValueError(f"archive has no rows after task/forecast_id filter for {task.task_id}")
    return filtered_ledger, filtered_archive


def add_task_columns(frame: pd.DataFrame, task: BenchmarkTask | None) -> pd.DataFrame:
    out = frame.copy()
    if task is None:
        return out
    out["task_id"] = task.task_id
    out["posterior_scope"] = task.posterior_scope
    out["task_components"] = task.component_label
    col = component_column(out)
    if col is None:
        out["target_component"] = task.component_label
    else:
        out["target_component"] = out[col].map(canonical_component)
    return out


def task_metadata(task: BenchmarkTask | None, ledger: pd.DataFrame | None = None) -> dict[str, object]:
    if task is None:
        return {}
    meta: dict[str, object] = {
        "task_id": task.task_id,
        "dataset": task.dataset,
        "target_component": task.components[0] if len(task.components) == 1 else "",
        "components": list(task.components),
        "posterior_scope": task.posterior_scope,
        "bridge_calibration_scope": task.posterior_scope,
    }
    if ledger is not None and "split" in ledger.columns:
        split = ledger["split"].astype(str)
        meta["n_validation_rows"] = int((split == "val").sum())
        meta["n_test_rows"] = int((split == "test").sum())
    return meta
