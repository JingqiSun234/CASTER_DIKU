#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


TASKS = ("benchmark_a", "benchmark_b_covid", "benchmark_b_flu")
METHODS = ("caster_one_layer", "caster_hierarchical")
LANES = ("moment_t", "draw_kernel_t")
TASK_RELATIVE = {
    "benchmark_a": Path("benchmark_a"),
    "benchmark_b_covid": Path("benchmark_b/benchmark_b_covid"),
    "benchmark_b_flu": Path("benchmark_b/benchmark_b_flu"),
}
TASK_COMPONENT = {
    "benchmark_a": "cases",
    "benchmark_b_covid": "covid_adm_per100k",
    "benchmark_b_flu": "flu_adm_per100k",
}
EVENT_TYPES = {
    "benchmark_a": ("declared_wave_onset",),
    "benchmark_b_covid": ("variant_turnover", "winter_onset"),
    "benchmark_b_flu": ("winter_onset",),
}
KNOWN_EVENT_TYPES = {
    "declared_wave_onset",
    "variant_turnover",
    "winter_onset",
    "large_revision_spike",
}
MODEL_LABELS = {
    "last_value": "Last value",
    "seasonal_naive": "SeasonalNaive-7d",
    "drift": "Drift",
    "covariate_drift": "Covariate drift",
    "sir_tau": "SIR (tau)",
    "seir_tau": "SEIR (tau)",
    "seirs_tau": "SEIRS (tau)",
    "tv_seir_rt": "Time-varying SEIR (R_t)",
    "renewal_rt": "Renewal (R_t)",
    "local_level": "Local level",
    "covariate_dynamic_linear_trend": "Covariate dynamic linear trend",
    "statsforecast_autoarima": "AutoARIMA",
    "statsforecast_autotheta": "AutoTheta",
    "statsforecast_autoces": "AutoCES",
    "rnn_simple": "Fixed-weight RNN",
    "lstm_style": "LSTM",
    "gru_style": "Fixed-weight GRU",
    "deepar_style": "DeepAR",
    "nbeats_basis": "N-BEATS",
    "nhits_hinterp": "NHITS",
    "patchtst_patched": "PatchTST",
    "tft_gated": "TFT",
    "chronos_external": "Chronos",
    "timesfm_external": "TimesFM",
}
FAMILY_COLUMNS = (
    "dataset",
    "method",
    "as_of_date",
    "family",
    "posterior_mass",
)
MODEL_COLUMNS = (
    "dataset",
    "method",
    "as_of_date",
    "selection_rank",
    "model_id",
    "model_label",
    "family",
    "posterior_mass",
)
EVENT_COLUMNS = ("dataset", "event_type", "event_time")
EVENT_ALIGNED_COLUMNS = (
    "dataset",
    "method",
    "event_type",
    "family",
    "relative_week",
    "mean_posterior_mass",
    "se_posterior_mass",
    "n_events_total",
)
MASS_TOLERANCE = 1.0e-8


class InputContractError(RuntimeError):
    pass


def require_columns(frame: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise InputContractError(f"{path} is missing columns: {', '.join(missing)}")


def task_root(run_root: Path, lane: str, task: str) -> Path:
    base = run_root
    if lane == "draw_kernel_t":
        base = run_root / "branches/draw_kernel"
    return base / "new_method/artifacts" / TASK_RELATIVE[task]


def resolve_existing(candidates: Sequence[Path], role: str) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    tried = "\n".join(f"  {path}" for path in candidates)
    raise InputContractError(f"missing {role}; checked:\n{tried}")


def selection_source_candidates(
    root: Path,
    task: str,
    filename: str,
) -> list[Path]:
    relative = TASK_RELATIVE[task]
    return [
        root / "new_method/artifacts" / relative / filename,
        root / "new_method/artifacts/shared_selections" / task / filename,
        root / "shared_selections" / task / filename,
        root / relative / filename,
        root / task / filename,
        root / filename,
    ]


def registry_label(row: pd.Series) -> str:
    model_id = str(row["model_id"])
    if model_id in MODEL_LABELS:
        return MODEL_LABELS[model_id]
    for column in ("model_label", "display_name", "candidate_type"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip().replace("_", " ")
    return model_id.replace("_", " ")


def load_selection(
    run_root: Path,
    selection_root: Path,
    task: str,
) -> pd.DataFrame:
    selection_path = resolve_existing(
        selection_source_candidates(selection_root, task, "candidate_selection_log.csv")
        + selection_source_candidates(run_root, task, "candidate_selection_log.csv"),
        f"Top-10 selection for {task}",
    )
    registry_path = resolve_existing(
        selection_source_candidates(selection_root, task, "model_registry.csv")
        + selection_source_candidates(run_root, task, "model_registry.csv"),
        f"model registry for {task}",
    )
    selected = pd.read_csv(selection_path, low_memory=False)
    registry = pd.read_csv(registry_path, low_memory=False)
    require_columns(selected, {"rank", "model_id", "family"}, selection_path)
    require_columns(registry, {"model_id", "family"}, registry_path)
    if "task_id" in selected.columns:
        selected = selected[selected["task_id"].astype(str).eq(task)].copy()
    selected = selected[["rank", "model_id", "family"]].copy()
    selected["rank"] = pd.to_numeric(selected["rank"], errors="raise").astype(int)
    selected["model_id"] = selected["model_id"].astype(str)
    selected["family"] = selected["family"].astype(str)
    selected = selected[selected["rank"].between(1, 10)].copy()
    if len(selected) != 10 or sorted(selected["rank"].tolist()) != list(range(1, 11)):
        raise InputContractError(f"{task} selection must contain ranks 1 through 10")
    if selected["model_id"].duplicated().any():
        raise InputContractError(f"{task} selection contains duplicate model IDs")

    registry = registry.drop_duplicates("model_id", keep="first").copy()
    registry["model_id"] = registry["model_id"].astype(str)
    registry["family"] = registry["family"].astype(str)
    merged = selected.merge(registry, on="model_id", how="left", suffixes=("", "_registry"), validate="one_to_one")
    if merged["family_registry"].isna().any():
        missing = sorted(merged.loc[merged["family_registry"].isna(), "model_id"].astype(str))
        raise InputContractError(f"{task} registry is missing selected models: {missing}")
    mismatch = merged["family"].ne(merged["family_registry"])
    if mismatch.any():
        models = sorted(merged.loc[mismatch, "model_id"].astype(str))
        raise InputContractError(f"{task} selection and registry families differ for: {models}")
    merged["model_label"] = merged.apply(registry_label, axis=1)
    return (
        merged.rename(columns={"rank": "selection_rank"})
        [["selection_rank", "model_id", "model_label", "family"]]
        .sort_values("selection_rank", kind="stable")
        .reset_index(drop=True)
    )


def validate_simplex(frame: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    values = pd.to_numeric(frame["posterior_mass"], errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise InputContractError(f"{label} contains invalid posterior masses")
    sums = frame.groupby(list(keys), observed=True)["posterior_mass"].sum()
    if sums.empty:
        raise InputContractError(f"{label} has no posterior updates")
    error = float((sums - 1.0).abs().max())
    if not math.isfinite(error) or error > MASS_TOLERANCE:
        raise InputContractError(f"{label} posterior normalization error is {error:.12g}")


def model_source_path(root: Path, lane: str, method: str) -> Path:
    if lane == "moment_t":
        filename = "posterior_path.csv" if method == "caster_one_layer" else "hierarchical_posterior_path.csv"
        return root / filename
    if method == "caster_one_layer":
        return root / "one_layer/posterior_path.csv"
    return root / "hierarchical/hierarchical_posterior_path.csv"


def family_source_path(root: Path, lane: str) -> Path:
    if lane == "moment_t":
        return root / "family_posterior.csv"
    return root / "hierarchical/family_posterior.csv"


def load_model_panel(
    path: Path,
    task: str,
    method: str,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    require_columns(frame, {"model_id", "family", "weight", "release_time"}, path)
    if "task_id" in frame.columns and not frame["task_id"].astype(str).eq(task).all():
        raise InputContractError(f"{path} contains rows from another task")
    frame = frame.copy()
    frame["model_id"] = frame["model_id"].astype(str)
    frame["family"] = frame["family"].astype(str)
    selected_ids = set(selection["model_id"])
    if set(frame["model_id"]) != selected_ids:
        raise InputContractError(f"{task}/{method} posterior model set differs from the Top-10 selection")
    joined = frame.merge(selection, on="model_id", how="left", suffixes=("_source", ""), validate="many_to_one")
    if joined["family_source"].ne(joined["family"]).any():
        raise InputContractError(f"{task}/{method} posterior family labels differ from the selection")
    if method == "caster_hierarchical" and {"family_weight", "inner_weight"}.issubset(joined.columns):
        joint = pd.to_numeric(joined["weight"], errors="raise")
        factorized = pd.to_numeric(joined["family_weight"], errors="raise") * pd.to_numeric(
            joined["inner_weight"], errors="raise"
        )
        error = float((joint - factorized).abs().max())
        if not math.isfinite(error) or error > MASS_TOLERANCE:
            raise InputContractError(f"{task}/{method} hierarchical masses do not factorize")
    out = pd.DataFrame(
        {
            "dataset": task,
            "method": method,
            "as_of_date": pd.to_datetime(joined["release_time"], errors="raise").dt.normalize(),
            "selection_rank": joined["selection_rank"].astype(int),
            "model_id": joined["model_id"].astype(str),
            "model_label": joined["model_label"].astype(str),
            "family": joined["family"].astype(str),
            "posterior_mass": pd.to_numeric(joined["weight"], errors="raise"),
        }
    )
    duplicate_key = ["dataset", "method", "as_of_date", "model_id"]
    if out.duplicated(duplicate_key).any():
        raise InputContractError(f"{task}/{method} posterior contains duplicate model/date rows")
    counts = out.groupby("as_of_date", observed=True)["model_id"].nunique()
    if not counts.eq(10).all():
        raise InputContractError(f"{task}/{method} does not contain all ten models at every update")
    validate_simplex(out, ("dataset", "method", "as_of_date"), f"{task}/{method}")
    return out.loc[:, MODEL_COLUMNS].sort_values(["as_of_date", "selection_rank"], kind="stable")


def one_layer_families(models: pd.DataFrame) -> pd.DataFrame:
    return (
        models.groupby(["dataset", "method", "as_of_date", "family"], observed=True, as_index=False)[
            "posterior_mass"
        ]
        .sum()
        .loc[:, FAMILY_COLUMNS]
    )


def load_hierarchical_families(path: Path, task: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    require_columns(frame, {"family", "family_weight", "release_time"}, path)
    if "task_id" in frame.columns and not frame["task_id"].astype(str).eq(task).all():
        raise InputContractError(f"{path} contains rows from another task")
    out = pd.DataFrame(
        {
            "dataset": task,
            "method": "caster_hierarchical",
            "as_of_date": pd.to_datetime(frame["release_time"], errors="raise").dt.normalize(),
            "family": frame["family"].astype(str),
            "posterior_mass": pd.to_numeric(frame["family_weight"], errors="raise"),
        }
    )
    key = ["dataset", "method", "as_of_date", "family"]
    if out.duplicated(key).any():
        raise InputContractError(f"{task} hierarchical family posterior contains duplicate rows")
    validate_simplex(out, ("dataset", "method", "as_of_date"), f"{task}/caster_hierarchical families")
    return out.loc[:, FAMILY_COLUMNS].sort_values(["as_of_date", "family"], kind="stable")


def build_lane(
    run_root: Path,
    selection_root: Path,
    lane: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_pieces: list[pd.DataFrame] = []
    family_pieces: list[pd.DataFrame] = []
    for task in TASKS:
        selection = load_selection(run_root, selection_root, task)
        root = task_root(run_root, lane, task)
        one_models = load_model_panel(
            model_source_path(root, lane, "caster_one_layer"),
            task,
            "caster_one_layer",
            selection,
        )
        hierarchical_models = load_model_panel(
            model_source_path(root, lane, "caster_hierarchical"),
            task,
            "caster_hierarchical",
            selection,
        )
        one_dates = set(one_models["as_of_date"])
        hierarchical_dates = set(hierarchical_models["as_of_date"])
        if one_dates != hierarchical_dates:
            raise InputContractError(f"{task}/{lane} posterior update dates differ by method")
        hierarchical_families = load_hierarchical_families(family_source_path(root, lane), task)
        if set(hierarchical_families["as_of_date"]) != hierarchical_dates:
            raise InputContractError(f"{task}/{lane} hierarchical family and model dates differ")
        grouped_joint = (
            hierarchical_models.groupby(["as_of_date", "family"], observed=True)["posterior_mass"].sum().sort_index()
        )
        direct_family = (
            hierarchical_families.set_index(["as_of_date", "family"])["posterior_mass"].sort_index()
        )
        if not grouped_joint.index.equals(direct_family.index):
            raise InputContractError(f"{task}/{lane} hierarchical family coverage differs from model coverage")
        family_error = float((grouped_joint - direct_family).abs().max())
        if not math.isfinite(family_error) or family_error > MASS_TOLERANCE:
            raise InputContractError(f"{task}/{lane} hierarchical family and model masses differ")
        model_pieces.extend((one_models, hierarchical_models))
        family_pieces.extend((one_layer_families(one_models), hierarchical_families))
    models = pd.concat(model_pieces, ignore_index=True).loc[:, MODEL_COLUMNS]
    families = pd.concat(family_pieces, ignore_index=True).loc[:, FAMILY_COLUMNS]
    models = models.sort_values(
        ["dataset", "method", "as_of_date", "selection_rank"], kind="stable"
    ).reset_index(drop=True)
    families = families.sort_values(
        ["dataset", "method", "as_of_date", "family"], kind="stable"
    ).reset_index(drop=True)
    return families, models


def event_source_path(events_root: Path, task: str) -> Path:
    folder = "benchmark_a" if task == "benchmark_a" else "benchmark_b"
    return events_root / folder / "events.csv"


def load_events(events_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    for task in TASKS:
        path = event_source_path(events_root, task)
        frame = pd.read_csv(path, low_memory=False)
        require_columns(frame, {"event_id", "event_time", "event_type", "component"}, path)
        frame = frame.copy()
        if task != "benchmark_a":
            frame = frame[frame["component"].astype(str).eq(TASK_COMPONENT[task])].copy()
        frame["dataset"] = task
        frame["event_id"] = frame["event_id"].astype(str)
        frame["event_type"] = frame["event_type"].astype(str)
        frame["event_time"] = pd.to_datetime(frame["event_time"], errors="raise").dt.normalize()
        frame["window_before"] = pd.to_numeric(
            frame.get("window_weeks_before", pd.Series(4, index=frame.index)), errors="raise"
        ).astype(int)
        frame["window_after"] = pd.to_numeric(
            frame.get("window_weeks_after", pd.Series(8, index=frame.index)), errors="raise"
        ).astype(int)
        if not frame["window_before"].eq(4).all() or not frame["window_after"].eq(8).all():
            raise InputContractError(f"{path} must use the declared -4 through +8 week window")
        unknown = sorted(set(frame["event_type"]) - KNOWN_EVENT_TYPES)
        if unknown:
            raise InputContractError(f"{path} contains unknown event types: {unknown}")
        pieces.append(
            frame[["dataset", "event_id", "event_type", "event_time", "window_before", "window_after"]]
        )
    raw = pd.concat(pieces, ignore_index=True)
    if raw.duplicated(["dataset", "event_id"]).any():
        raise InputContractError("event IDs must be unique within each task")
    markers = (
        raw.loc[:, EVENT_COLUMNS]
        .sort_values(["dataset", "event_time", "event_type"], kind="stable")
        .reset_index(drop=True)
    )
    return raw, markers


def standard_error(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if array.size <= 1:
        return 0.0
    return float(array.std(ddof=1) / math.sqrt(array.size))


def build_event_aligned(families: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    aligned_pieces: list[pd.DataFrame] = []
    totals: dict[tuple[str, str], int] = {}
    for task in TASKS:
        task_events = events[
            events["dataset"].eq(task) & events["event_type"].isin(EVENT_TYPES[task])
        ].copy()
        for event_type, group in task_events.groupby("event_type", observed=True):
            totals[(task, str(event_type))] = int(group["event_id"].nunique())
        task_families = families[families["dataset"].eq(task)].copy()
        for event in task_events.itertuples(index=False):
            relative_days = (task_families["as_of_date"] - pd.Timestamp(event.event_time)).dt.days
            keep = (
                relative_days.between(-7 * int(event.window_before), 7 * int(event.window_after))
                & relative_days.mod(7).eq(0)
            )
            panel = task_families.loc[keep].copy()
            if panel.empty:
                continue
            panel["event_id"] = str(event.event_id)
            panel["event_type"] = str(event.event_type)
            panel["relative_week"] = (relative_days.loc[keep] // 7).astype(int)
            aligned_pieces.append(panel)
    if not aligned_pieces:
        raise InputContractError("no posterior updates fall inside the declared event windows")
    aligned = pd.concat(aligned_pieces, ignore_index=True)
    group_columns = ["dataset", "method", "event_type", "family", "relative_week"]
    grouped = aligned.groupby(group_columns, observed=True)["posterior_mass"]
    result = grouped.agg(mean_posterior_mass="mean", se_posterior_mass=standard_error).reset_index()
    result["n_events_total"] = [
        totals[(str(task), str(event_type))]
        for task, event_type in zip(result["dataset"], result["event_type"], strict=True)
    ]
    expected = {
        (task, method, event_type)
        for task in TASKS
        for method in METHODS
        for event_type in EVENT_TYPES[task]
    }
    observed = set(zip(result["dataset"], result["method"], result["event_type"], strict=True))
    if observed != expected:
        raise InputContractError(
            f"event-aligned coverage differs: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return (
        result.loc[:, EVENT_ALIGNED_COLUMNS]
        .sort_values(["dataset", "method", "event_type", "family", "relative_week"], kind="stable")
        .reset_index(drop=True)
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare posterior and event CSV inputs for the trajectory renderers."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--events-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.resolve()
    events_root = args.events_root.resolve()
    selection_root = (args.selection_root or run_root).resolve()
    lane_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for lane in LANES:
        lane_data[lane] = build_lane(run_root, selection_root, lane)
    event_rows, markers = load_events(events_root)
    event_aligned = build_event_aligned(lane_data["moment_t"][0], event_rows)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(lane_data["moment_t"][0], output_dir / "moment_t_family.csv")
    write_csv(lane_data["draw_kernel_t"][0], output_dir / "draw_kernel_t_family.csv")
    write_csv(lane_data["moment_t"][1], output_dir / "moment_t_models.csv")
    write_csv(lane_data["draw_kernel_t"][1], output_dir / "draw_kernel_t_models.csv")
    write_csv(event_aligned, output_dir / "event_aligned.csv")
    write_csv(markers, output_dir / "events.csv")
    for filename in (
        "moment_t_family.csv",
        "draw_kernel_t_family.csv",
        "moment_t_models.csv",
        "draw_kernel_t_models.csv",
        "event_aligned.csv",
        "events.csv",
    ):
        print(output_dir / filename)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputContractError as exc:
        raise SystemExit(f"input error: {exc}") from exc
