from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from _shared import TASKS, method_column, read_csv, task_column, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize selection distributions at test forecast origins.")
    parser.add_argument(
        "--lane",
        action="append",
        nargs=6,
        metavar=("TASK", "METHOD", "WEIGHTS", "READOUT", "CATEGORY", "WEIGHT"),
        default=[],
    )
    parser.add_argument(
        "--one-hot",
        action="append",
        nargs=2,
        metavar=("TASK", "METHOD"),
        default=[],
    )
    parser.add_argument("--forecast-summary", type=Path)
    parser.add_argument("--shift-scores", type=Path)
    parser.add_argument(
        "--shift-input",
        action="append",
        nargs=6,
        metavar=("TASK", "METHOD", "FORECAST", "LEDGER", "EVENTS", "BRIDGE"),
        default=[],
    )
    parser.add_argument(
        "--mixture-shift-input",
        action="append",
        nargs=8,
        metavar=("TASK", "METHOD", "WEIGHTS", "EVIDENCE", "EVENTS", "CATEGORY", "WEIGHT", "LOG_SCORE"),
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def check_task(task: str) -> None:
    if task not in TASKS:
        raise SystemExit(f"unsupported task: {task}")


def component_name(task: str) -> str:
    return {
        "benchmark_a": "cases",
        "benchmark_b_covid": "covid_adm_per100k",
        "benchmark_b_flu": "flu_adm_per100k",
    }[task]


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


def filter_task(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    data = frame.copy()
    for name in ("task", "task_id", "dataset_key", "dataset"):
        if name not in data.columns:
            continue
        values = data[name].fillna("").astype(str).str.strip()
        allowed = {task}
        if task.startswith("benchmark_b_"):
            allowed.add("benchmark_b")
        matched = values.isin(allowed)
        if matched.any():
            data = data[matched].copy()
            break
    if task.startswith("benchmark_b_") and "component" in data.columns:
        components = canonical_component(data["component"])
        blank = components.isin({"", "nan", "none", "all", "all_components"})
        data = data[blank | components.eq(component_name(task))].copy()
    return data


def select_column(frame: pd.DataFrame, names: tuple[str, ...], label: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise SystemExit(f"missing {label} column")


def enrich_forecast(forecast: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    data = forecast.copy()
    if "forecast_id" in data.columns and "forecast_id" in ledger.columns:
        source = ledger.drop_duplicates("forecast_id", keep="last").set_index("forecast_id")
        ids = data["forecast_id"].astype(str)
        source.index = source.index.astype(str)
        for name in (
            "observed_value",
            "observed_mask",
            "mode",
            "component",
            "horizon",
            "forecast_origin",
            "target_time",
        ):
            if name not in source.columns:
                continue
            values = ids.map(source[name])
            if name not in data.columns:
                data[name] = values
            else:
                missing = data[name].isna() | data[name].astype(str).str.strip().isin({"", "nan", "None"})
                data.loc[missing, name] = values.loc[missing]
    return data


def score_forecast(frame: pd.DataFrame, bridge_path: Path, method: str) -> pd.DataFrame:
    source_root = Path(__file__).resolve().parents[1] / "code/caster/src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from caster.bridge import read_bridge_config, score_archive_rows

    data = frame.copy().reset_index(drop=True)
    observed_name = select_column(data, ("observed_value", "y_true"), "observed value")
    mean_name = select_column(data, ("predictive_mean", "pred_mean", "y_pred"), "predictive mean")
    lower_name = select_column(data, ("lower_90", "pred_lower_90"), "lower interval")
    upper_name = select_column(data, ("upper_90", "pred_upper_90"), "upper interval")
    for name in ("component", "horizon", "forecast_origin", "target_time"):
        if name not in data.columns:
            raise SystemExit(f"forecast missing column: {name}")
    lower = pd.to_numeric(data[lower_name], errors="raise").clip(lower=0.0)
    upper = pd.to_numeric(data[upper_name], errors="raise").clip(lower=0.0)
    interval_var = ((upper - lower).abs() / (2.0 * 1.6448536269514722)).pow(2)
    variance_name = next((name for name in ("pred_var", "predictive_var") if name in data.columns), "")
    if variance_name:
        variance = pd.to_numeric(data[variance_name], errors="coerce")
        variance = variance.where(variance.notna() & variance.ge(0.0), interval_var)
    else:
        variance = interval_var
    row_ids = pd.Series([f"shift_{index}" for index in range(len(data))], dtype=str)
    observed = pd.to_numeric(data[observed_name], errors="coerce")
    observed_mask = observed.notna()
    if "observed_mask" in data.columns:
        declared = data["observed_mask"]
        if pd.api.types.is_bool_dtype(declared):
            declared_mask = declared.fillna(False).astype(bool)
        else:
            declared_mask = (
                declared.fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "t", "yes", "y"})
            )
        observed_mask &= declared_mask
    mode = data["mode"].astype(str) if "mode" in data.columns else pd.Series("", index=data.index)
    score_ledger = pd.DataFrame(
        {
            "forecast_id": row_ids,
            "observed_value": observed,
            "observed_mask": observed_mask,
            "mode": mode,
            "component": data["component"].astype(str),
            "horizon": pd.to_numeric(data["horizon"], errors="raise").astype(int),
            "forecast_origin": data["forecast_origin"],
            "target_time": data["target_time"],
        }
    )
    if "selected_model_id" in data.columns:
        model = data["selected_model_id"].fillna(method).astype(str)
    else:
        model = pd.Series(method, index=data.index, dtype=str)
    archive = pd.DataFrame(
        {
            "forecast_id": row_ids,
            "model_id": model,
            "particle_id": "0",
            "pred_mean": pd.to_numeric(data[mean_name], errors="raise"),
            "pred_var": variance,
            "mode": mode,
            "component": data["component"].astype(str),
            "horizon": pd.to_numeric(data["horizon"], errors="raise").astype(int),
            "forecast_origin": data["forecast_origin"],
            "target_time": data["target_time"],
            "features_available_until": data["forecast_origin"],
        }
    )
    bridge, _ = read_bridge_config(bridge_path)
    scored = score_archive_rows(score_ledger, archive, bridge)
    if "log_score" not in scored.columns:
        raise SystemExit("bridge scorer returned no log score")
    by_id = scored.set_index(scored["forecast_id"].astype(str))["log_score"]
    data["_log_score"] = row_ids.map(by_id).to_numpy(dtype=float)
    data.loc[~observed_mask, "_log_score"] = np.nan
    return data


def shift_value(spec: list[str]) -> tuple[tuple[str, str], float]:
    task, method, forecast_name, ledger_name, events_name, bridge_name = spec
    check_task(task)
    forecast = filter_task(read_csv(Path(forecast_name)), task)
    ledger = filter_task(read_csv(Path(ledger_name)), task)
    events = filter_task(read_csv(Path(events_name)), task)
    if forecast.empty or ledger.empty or events.empty:
        raise SystemExit(f"empty shift input for {(task, method)}")
    forecast = enrich_forecast(forecast, ledger)
    forecast = score_forecast(forecast, Path(bridge_name), method)
    if "target_time" not in forecast.columns:
        raise SystemExit(f"forecast has no target time: {forecast_name}")
    forecast["target_time"] = pd.to_datetime(forecast["target_time"], errors="raise").dt.normalize()
    required = {"event_id", "event_time"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise SystemExit(f"{events_name} missing columns: {missing}")
    events = events.copy()
    events["event_time"] = pd.to_datetime(events["event_time"], errors="raise").dt.normalize()
    rows: list[dict[str, object]] = []
    for event in events.sort_values(["event_time", "event_id"], kind="stable").itertuples(index=False):
        event_id = str(getattr(event, "event_id"))
        event_time = pd.Timestamp(getattr(event, "event_time"))
        event_component = str(getattr(event, "component", "")).strip()
        for relative_week in range(9):
            release_time = event_time + pd.Timedelta(days=7 * relative_week)
            current = forecast[forecast["target_time"].eq(release_time)].copy()
            if event_component and event_component.lower() not in {"nan", "none", "all", "all_components"}:
                current = current[canonical_component(current["component"]).eq(canonical_component(pd.Series([event_component])).iloc[0])]
            scores = pd.to_numeric(current.get("_log_score", pd.Series(dtype=float)), errors="coerce").dropna()
            if not scores.empty:
                rows.append({"event_id": event_id, "nll": -float(scores.mean())})
    if not rows:
        raise SystemExit(f"no event-aligned score rows for {(task, method)}")
    detail = pd.DataFrame(rows)
    value = float(detail.groupby("event_id", dropna=False)["nll"].mean().mean())
    if not math.isfinite(value):
        raise SystemExit(f"non-finite shift score for {(task, method)}")
    return (task, method), value


def stable_log_sum(values: np.ndarray) -> float:
    if values.size == 0 or not np.isfinite(values).all():
        return float("nan")
    maximum = float(values.max())
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def mixture_shift_value(spec: list[str]) -> tuple[tuple[str, str], float]:
    task, method, weights_name, evidence_name, events_name, category, weight, log_score = spec
    check_task(task)
    weights = filter_task(read_csv(Path(weights_name)), task)
    evidence = filter_task(read_csv(Path(evidence_name)), task)
    events = filter_task(read_csv(Path(events_name)), task)
    for frame, label, columns in (
        (weights, weights_name, ("release_time", category, weight)),
        (evidence, evidence_name, ("release_time", category, log_score)),
        (events, events_name, ("event_id", "event_time")),
    ):
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise SystemExit(f"{label} missing columns: {missing}")
    weights = weights.copy()
    evidence = evidence.copy()
    events = events.copy()
    weights["release_time"] = pd.to_datetime(weights["release_time"], errors="raise").dt.normalize()
    evidence["release_time"] = pd.to_datetime(evidence["release_time"], errors="raise").dt.normalize()
    events["event_time"] = pd.to_datetime(events["event_time"], errors="raise").dt.normalize()
    weights[category] = weights[category].astype(str)
    evidence[category] = evidence[category].astype(str)
    weights[weight] = pd.to_numeric(weights[weight], errors="raise").astype(float)
    evidence[log_score] = pd.to_numeric(evidence[log_score], errors="raise").astype(float)
    categories = sorted(weights[category].unique())
    snapshot_times = sorted(pd.Timestamp(value) for value in weights["release_time"].unique())
    rows: list[dict[str, object]] = []
    for event in events.sort_values(["event_time", "event_id"], kind="stable").itertuples(index=False):
        event_id = str(getattr(event, "event_id"))
        event_time = pd.Timestamp(getattr(event, "event_time"))
        for relative_week in range(9):
            release_time = event_time + pd.Timedelta(days=7 * relative_week)
            previous = [value for value in snapshot_times if value < release_time]
            if not previous:
                continue
            weight_rows = weights[weights["release_time"].eq(max(previous))]
            evidence_rows = evidence[evidence["release_time"].eq(release_time)]
            if weight_rows.empty or evidence_rows.empty:
                continue
            mass = weight_rows.drop_duplicates(category, keep="last").set_index(category)[weight].reindex(categories)
            scores = evidence_rows.drop_duplicates(category, keep="last").set_index(category)[log_score].reindex(categories)
            mass_values = pd.to_numeric(mass, errors="coerce").to_numpy(dtype=float)
            score_values = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(mass_values).all() or not np.isfinite(score_values).all() or float(mass_values.sum()) <= 0.0:
                continue
            mass_values = np.clip(mass_values, 0.0, None)
            mass_values = mass_values / float(mass_values.sum())
            value = stable_log_sum(np.log(np.clip(mass_values, 1e-300, None)) + score_values)
            if math.isfinite(value):
                rows.append({"event_id": event_id, "nll": -value})
    if not rows:
        raise SystemExit(f"no event-aligned mixture score rows for {(task, method)}")
    detail = pd.DataFrame(rows)
    value = float(detail.groupby("event_id", dropna=False)["nll"].mean().mean())
    if not math.isfinite(value):
        raise SystemExit(f"non-finite shift score for {(task, method)}")
    return (task, method), value


def distribution_rows(path: Path, category: str, weight: str) -> pd.DataFrame:
    frame = read_csv(path)
    required = {"release_time", category, weight}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{path} missing columns: {missing}")
    frame = frame[["release_time", category, weight]].copy()
    frame["release_time"] = pd.to_datetime(frame["release_time"], errors="raise")
    frame[category] = frame[category].astype(str)
    frame[weight] = pd.to_numeric(frame[weight], errors="raise").astype(float)
    if frame.duplicated(["release_time", category]).any():
        raise SystemExit(f"duplicate support rows: {path}")
    support = sorted(frame[category].unique())
    rows: list[dict[str, object]] = []
    for release_time, group in frame.groupby("release_time", sort=True):
        if sorted(group[category].unique()) != support:
            raise SystemExit(f"support changes across snapshots: {path}")
        values = group.set_index(category)[weight].reindex(support).to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0.0) or float(values.sum()) <= 0.0:
            raise SystemExit(f"invalid weights: {path}")
        values = values / float(values.sum())
        positive = values[values > 0.0]
        rows.append(
            {
                "posterior_snapshot_time": release_time,
                "top1_mass": float(values.max()),
                "entropy": float(-np.sum(positive * np.log(positive))),
                "ess": float(1.0 / np.square(values).sum()),
                "support_count": int(len(values)),
            }
        )
    if not rows:
        raise SystemExit(f"no weight snapshots: {path}")
    return pd.DataFrame(rows)


def lane_summary(spec: list[str]) -> dict[str, object]:
    task, method, weights_name, readout_name, category, weight = spec
    check_task(task)
    snapshots = distribution_rows(Path(weights_name), category, weight)
    readout = read_csv(Path(readout_name))
    if "method" in readout.columns and readout["method"].astype(str).eq(method).any():
        readout = readout[readout["method"].astype(str).eq(method)].copy()
    required = {"forecast_origin", "posterior_snapshot_time"}
    missing = sorted(required - set(readout.columns))
    if missing:
        raise SystemExit(f"{readout_name} missing columns: {missing}")
    if "split" in readout.columns:
        readout = readout[readout["split"].astype(str).str.lower().eq("test")].copy()
    if readout.empty:
        raise SystemExit(f"no test rows: {readout_name}")
    readout["forecast_origin"] = pd.to_datetime(readout["forecast_origin"], errors="raise")
    readout["posterior_snapshot_time"] = pd.to_datetime(readout["posterior_snapshot_time"], errors="raise")
    mapping = readout[["forecast_origin", "posterior_snapshot_time"]].drop_duplicates()
    if mapping["forecast_origin"].duplicated().any():
        raise SystemExit(f"an origin maps to multiple snapshots: {readout_name}")
    if (mapping["posterior_snapshot_time"] > mapping["forecast_origin"]).any():
        raise SystemExit(f"future snapshot in {readout_name}")
    values = mapping.merge(snapshots, on="posterior_snapshot_time", how="left", validate="many_to_one")
    if values[["top1_mass", "entropy", "ess"]].isna().any().any():
        raise SystemExit(f"readout references an unknown snapshot: {readout_name}")
    return {
        "task": task,
        "method": method,
        "top1_mass": float(values["top1_mass"].mean()),
        "entropy": float(values["entropy"].mean()),
        "ess": float(values["ess"].mean()),
        "n_test_origins": int(len(values)),
        "support_count": int(values["support_count"].max()),
    }


def keyed_values(path: Path | None, value_names: tuple[str, ...]) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    frame = read_csv(path)
    task_name = task_column(frame)
    method_name = method_column(frame)
    value_name = next((name for name in value_names if name in frame.columns), "")
    if not value_name:
        raise SystemExit(f"{path} has no supported value column")
    frame = frame.copy()
    frame["_value"] = pd.to_numeric(frame[value_name], errors="coerce")
    if "event_id" in frame.columns:
        per_event = (
            frame.groupby([task_name, method_name, "event_id"], dropna=False)["_value"]
            .mean()
            .reset_index()
        )
        frame = per_event.groupby([task_name, method_name], dropna=False)["_value"].mean().reset_index()
    values: dict[tuple[str, str], float] = {}
    for (task, method), group in frame.groupby([task_name, method_name], dropna=False):
        numbers = pd.to_numeric(group["_value"], errors="coerce").dropna()
        if numbers.empty:
            continue
        value = float(numbers.mean())
        if not math.isfinite(value):
            raise SystemExit(f"non-finite value for {(task, method)} in {path}")
        values[(str(task), str(method))] = value
    return values


def main() -> int:
    args = parse_args()
    rows = [lane_summary(spec) for spec in args.lane]
    for task, method in args.one_hot:
        check_task(task)
        rows.append(
            {
                "task": task,
                "method": method,
                "top1_mass": 1.0,
                "entropy": 0.0,
                "ess": 1.0,
                "n_test_origins": "",
                "support_count": 1,
            }
        )
    if not rows:
        raise SystemExit("at least one --lane or --one-hot entry is required")
    coverage = keyed_values(args.forecast_summary, ("coverage_90",))
    shift = keyed_values(args.shift_scores, ("shift_nll", "post_shift_nll_numeric", "post_shift_nll"))
    for spec in args.shift_input:
        key, value = shift_value(spec)
        if key in shift and not math.isclose(shift[key], value, rel_tol=0.0, abs_tol=1e-10):
            raise SystemExit(f"conflicting shift scores for {key}")
        shift[key] = value
    for spec in args.mixture_shift_input:
        key, value = mixture_shift_value(spec)
        if key in shift and not math.isclose(shift[key], value, rel_tol=0.0, abs_tol=1e-10):
            raise SystemExit(f"conflicting shift scores for {key}")
        shift[key] = value
    for row in rows:
        key = (str(row["task"]), str(row["method"]))
        row["coverage_90"] = coverage.get(key, float("nan"))
        row["shift_nll"] = shift.get(key, float("nan"))
    result = pd.DataFrame(rows)
    if result[["task", "method"]].duplicated().any():
        raise SystemExit("duplicate task and method entries")
    result = result[
        [
            "task",
            "method",
            "coverage_90",
            "top1_mass",
            "entropy",
            "ess",
            "shift_nll",
            "n_test_origins",
            "support_count",
        ]
    ].sort_values(["task", "method"], kind="stable")
    write_csv(result, args.output)
    print(f"rows={len(result)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
