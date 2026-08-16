from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASTER_SOURCE = ROOT / "code/caster/src"
SCRIPT_SOURCE = ROOT / "scripts"
for source in (CASTER_SOURCE, SCRIPT_SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from caster.bridge import read_bridge_config
from caster.filter import single_model_predictive_readout
from result_metric_contract import (
    filter_to_formal_horizon_grid,
    metric_slices_from_scored_rows,
)


TASKS = ("benchmark_a", "benchmark_b_covid", "benchmark_b_flu")
EXPECTED_MODELS = 27
MODEL_GROUP = "model_pool"
MEAN_PRESERVING = "coherent_mean_preserving_censored_student_t"
REQUIRED_LEDGER = {
    "forecast_id",
    "split",
    "observed_value",
    "component",
    "horizon",
}
REQUIRED_ARCHIVE = {
    "forecast_id",
    "model_id",
    "particle_id",
    "pred_mean",
    "pred_var",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score every candidate model on one task and write metric slices."
    )
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def truthy(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna("").astype(str).str.strip().str.lower().isin(
        {"true", "1", "t", "yes", "y"}
    )


def require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise SystemExit(f"{label} missing columns: {missing}")


def canonical_component(values: pd.Series) -> pd.Series:
    aliases = {
        "covid": "covid_adm_per100k",
        "covid_adm": "covid_adm_per100k",
        "covid_admissions": "covid_adm_per100k",
        "flu": "flu_adm_per100k",
        "influenza": "flu_adm_per100k",
        "flu_adm": "flu_adm_per100k",
        "flu_admissions": "flu_adm_per100k",
    }
    clean = values.fillna("").astype(str).str.strip().str.lower()
    return clean.map(lambda value: aliases.get(value, value))


def task_component(task: str) -> str:
    return {
        "benchmark_b_covid": "covid_adm_per100k",
        "benchmark_b_flu": "flu_adm_per100k",
    }.get(task, "")


def load_registry(path: Path) -> pd.DataFrame:
    registry = read_csv(path)
    require(registry, {"model_id"}, "registry")
    registry = registry.copy()
    registry["model_id"] = registry["model_id"].fillna("").astype(str).str.strip()
    if registry["model_id"].eq("").any():
        raise SystemExit("registry contains an empty model identifier")
    if registry["model_id"].duplicated().any():
        raise SystemExit("registry contains duplicate model identifiers")
    if len(registry) != EXPECTED_MODELS:
        raise SystemExit(
            f"registry must contain {EXPECTED_MODELS} models; found {len(registry)}"
        )
    order_name = next(
        (
            name
            for name in ("candidate_registry_order", "registry_order")
            if name in registry.columns
        ),
        "",
    )
    if order_name:
        order = pd.to_numeric(registry[order_name], errors="coerce")
        if order.isna().any() or order.duplicated().any():
            raise SystemExit("registry contains invalid ordering values")
        registry["candidate_registry_order"] = order.astype(int)
    else:
        registry["candidate_registry_order"] = np.arange(1, len(registry) + 1)
    return registry


def load_ledger(path: Path, task: str) -> pd.DataFrame:
    ledger = read_csv(path)
    require(ledger, REQUIRED_LEDGER, "ledger")
    ledger = ledger.copy()
    if task.startswith("benchmark_b_"):
        wanted = task_component(task)
        ledger = ledger[canonical_component(ledger["component"]).eq(wanted)].copy()
    ledger = ledger[ledger["split"].astype(str).str.strip().str.lower().eq("test")].copy()
    for name in ("result_metric_eligible", "metric_eligible"):
        if name in ledger.columns:
            ledger = ledger[truthy(ledger[name])].copy()
            break
    if "observed_mask" in ledger.columns:
        ledger = ledger[truthy(ledger["observed_mask"])].copy()
    ledger["observed_value"] = pd.to_numeric(
        ledger["observed_value"], errors="coerce"
    )
    ledger = ledger[ledger["observed_value"].map(math.isfinite)].copy()
    if ledger.empty:
        raise SystemExit("ledger has no eligible test rows")
    ledger["dataset"] = task
    ledger["forecast_id"] = ledger["forecast_id"].astype(str)
    ledger = filter_to_formal_horizon_grid(
        ledger,
        config_path=ROOT / "configs/caster_task_specs_v20.yaml",
        strict=True,
    )
    if ledger.empty:
        raise SystemExit("ledger has no rows on the declared horizon grid")
    if ledger["forecast_id"].duplicated().any():
        raise SystemExit("ledger contains duplicate forecast identifiers")
    return ledger


def load_archive(
    path: Path, ledger: pd.DataFrame, registry: pd.DataFrame
) -> pd.DataFrame:
    archive = read_csv(path)
    require(archive, REQUIRED_ARCHIVE, "archive")
    archive = archive.copy()
    for name in ("forecast_id", "model_id", "particle_id"):
        archive[name] = archive[name].astype(str)
    wanted_forecasts = set(ledger["forecast_id"])
    archive = archive[archive["forecast_id"].isin(wanted_forecasts)].copy()
    if archive.empty:
        raise SystemExit("archive has no matching forecast rows")
    registered = set(registry["model_id"])
    observed_models = set(archive["model_id"])
    missing_models = sorted(registered - observed_models)
    extra_models = sorted(observed_models - registered)
    if missing_models or extra_models:
        raise SystemExit(
            "archive model set differs from registry; "
            f"missing={missing_models}, extra={extra_models}"
        )
    duplicate_key = ["forecast_id", "model_id", "particle_id"]
    if archive.duplicated(duplicate_key).any():
        raise SystemExit("archive contains duplicate forecast/model/particle rows")
    expected_pairs = len(wanted_forecasts) * len(registered)
    observed_pairs = archive[["forecast_id", "model_id"]].drop_duplicates()
    if len(observed_pairs) != expected_pairs:
        counts = observed_pairs.groupby("model_id")["forecast_id"].nunique()
        incomplete = {
            model: int(counts.get(model, 0))
            for model in sorted(registered)
            if int(counts.get(model, 0)) != len(wanted_forecasts)
        }
        raise SystemExit(f"archive coverage is incomplete: {incomplete}")
    return archive


def attach_registry_fields(
    slices: pd.DataFrame, registry: pd.DataFrame
) -> pd.DataFrame:
    columns = ["model_id", "candidate_registry_order"]
    rename: dict[str, str] = {"model_id": "method"}
    if "family" in registry.columns:
        columns.append("family")
        rename["family"] = "candidate_family"
    fields = registry[columns].rename(columns=rename)
    return slices.merge(fields, on="method", how="left", validate="many_to_one")


def score(args: argparse.Namespace) -> pd.DataFrame:
    registry = load_registry(args.registry)
    ledger = load_ledger(args.ledger, args.task)
    archive = load_archive(args.archive, ledger, registry)
    bridge, _ = read_bridge_config(args.bridge)
    readout = single_model_predictive_readout(
        ledger,
        archive,
        bridge,
        score_source="archive_moment",
    ).copy()
    mean_preserving = str(bridge.predictive_contract) == MEAN_PRESERVING
    required = {
        "forecast_id",
        "model_id",
        "predictive_mean",
        "lower_50",
        "upper_50",
        "lower_90",
        "upper_90",
        "log_score",
    }
    if mean_preserving:
        required.add("predictive_median")
    require(readout, required, "candidate readout")
    readout["forecast_id"] = readout["forecast_id"].astype(str)
    readout["model_id"] = readout["model_id"].astype(str)
    expected_rows = len(ledger) * len(registry)
    if len(readout) != expected_rows:
        raise SystemExit(
            f"candidate readout row count mismatch: expected {expected_rows}, found {len(readout)}"
        )
    if readout.duplicated(["forecast_id", "model_id"]).any():
        raise SystemExit("candidate readout contains duplicate forecast/model rows")
    numeric = [
        "predictive_mean",
        "lower_50",
        "upper_50",
        "lower_90",
        "upper_90",
        "log_score",
    ]
    if mean_preserving:
        numeric.append("predictive_median")
    for name in numeric:
        values = pd.to_numeric(readout[name], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise SystemExit(f"candidate readout contains non-finite {name} values")
        readout[name] = values
    readout["dataset"] = args.task
    readout["method"] = readout["model_id"]
    readout["bridge_nll"] = -readout["log_score"]
    prediction = "predictive_mean"
    median = "predictive_median" if mean_preserving else None
    slices = metric_slices_from_scored_rows(
        readout,
        source=args.archive.name,
        y_col="observed_value",
        pred_col=prediction,
        median_col=median,
        lower_50_col="lower_50",
        upper_50_col="upper_50",
        lower_90_col="lower_90",
        upper_90_col="upper_90",
        nll_col="bridge_nll",
        method_group=MODEL_GROUP,
    )
    slices = attach_registry_fields(slices, registry)
    return slices.sort_values(
        ["dataset", "candidate_registry_order", "method", "forecast_strategy", "horizon"],
        kind="stable",
    ).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    if args.output.suffix.lower() != ".csv":
        raise SystemExit("output path must end in .csv")
    result = score(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
