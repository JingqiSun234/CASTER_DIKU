#!/usr/bin/env python
""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import hashlib

import pandas as pd


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _counts_text(series: pd.Series) -> str:
    counts = series.astype(str).value_counts(dropna=False).sort_index()
    return ";".join(f"{k}:{int(v)}" for k, v in counts.items())


def _source_dataset_key(task: str) -> str:
    if task.startswith("benchmark_a"):
        return "benchmark_a"
    if task.startswith("benchmark_b"):
        return "benchmark_b"
    return task


def main() -> int:
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--baseline-manifest", required=True)
    ap.add_argument("--task-ledger", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument(
        "--selection-log",
        default="",
        help=(
            "Optional previous agent selection log. If omitted, only the task-local "
            "manifest is written; true Qwen selection can then run against a "
            "prebuilt immutable forecast archive."
        ),
    )
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-replay", default="")
    ap.add_argument(
        "--out-selection",
        default="",
        help=(
            "Optional task-local model selection CSV for building the immutable "
            "forecast archive used by the archive-backed agent replay. The file "
            "contains the union of replay-selected model_id values."
        ),
    )
    args = ap.parse_args()

    task = str(args.task)
    source_key = _source_dataset_key(task)
    baseline_manifest = Path(args.baseline_manifest)
    task_ledger = Path(args.task_ledger)
    selection_log = Path(args.selection_log) if str(args.selection_log).strip() else None
    out_manifest = Path(args.out_manifest)
    out_replay = Path(args.out_replay) if str(args.out_replay).strip() else None
    out_selection = Path(args.out_selection) if str(args.out_selection).strip() else None

    manifest = pd.read_csv(baseline_manifest, keep_default_na=False)
    matches = manifest[manifest["dataset_key"].astype(str).eq(source_key)].copy()
    if matches.empty:
        raise SystemExit(f"baseline manifest has no source dataset_key={source_key}")
    row = matches.iloc[0].copy()
                                                                             
                                                             
    ledger = pd.read_csv(task_ledger, keep_default_na=False, low_memory=False)
    if ledger.empty:
        raise SystemExit(f"task ledger is empty: {task_ledger}")

    row["dataset_key"] = task
    row["ledger_path"] = str(task_ledger)
    row["panel_path"] = str(Path(args.panel))
    row["ledger_rows"] = int(len(ledger))
    row["ledger_sha256"] = _sha256(task_ledger)
    if "component" in ledger.columns:
        components = sorted(ledger["component"].astype(str).dropna().unique())
        row["components"] = ";".join(components)
        row["component_counts"] = _counts_text(ledger["component"])
    if "split" in ledger.columns:
        row["splits"] = ";".join(sorted(ledger["split"].astype(str).dropna().unique()))
        row["split_counts"] = _counts_text(ledger["split"])
    if "horizon" in ledger.columns:
        row["horizons"] = ";".join(str(x) for x in sorted(pd.to_numeric(ledger["horizon"], errors="coerce").dropna().astype(int).unique()))
        row["horizon_counts"] = _counts_text(pd.to_numeric(ledger["horizon"], errors="coerce").dropna().astype(int))
    for col, out_col in [
        ("forecast_origin", "ledger_origin"),
        ("target_time", "ledger_target"),
    ]:
        if col in ledger.columns:
            ts = pd.to_datetime(ledger[col], errors="coerce").dropna()
            if not ts.empty:
                row[f"{out_col}_min"] = str(ts.min().date())
                row[f"{out_col}_max"] = str(ts.max().date())

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row])[manifest.columns].to_csv(out_manifest, index=False)

    print(f"archive_backed_agent_manifest={out_manifest}")

    if selection_log is None:
        if out_replay is not None:
            out_replay.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=["dataset_key", "forecast_origin", "selected_model_id"]).to_csv(out_replay, index=False)
            print(f"archive_backed_agent_replay={out_replay}")
        print("archive_backed_agent_replay_mode=not_written_true_selection_expected")
        print(f"task={task} replay_rows=0 ledger_rows={len(ledger)}")
        return 0
    if out_replay is None:
        raise SystemExit("--out-replay is required when --selection-log is provided")

    replay = pd.read_csv(selection_log, keep_default_na=False)
    required = {"dataset_key", "forecast_origin"}
    missing = sorted(required - set(replay.columns))
    if missing:
        raise SystemExit(f"selection log missing columns: {missing}")
    origins = set(ledger["forecast_origin"].astype(str))
    replay = replay[replay["dataset_key"].astype(str).eq(source_key)].copy()
    replay = replay[replay["forecast_origin"].astype(str).isin(origins)].copy()
    if replay.empty:
        raise SystemExit(f"selection log has no replay rows for source={source_key} task={task}")
    replay["dataset_key"] = task
    out_replay.parent.mkdir(parents=True, exist_ok=True)
    replay.to_csv(out_replay, index=False)
    selected_col = "selected_model_id" if "selected_model_id" in replay.columns else "model_id"
    if selected_col not in replay.columns:
        raise SystemExit("selection log must contain selected_model_id or model_id")
    replay_model_ids = []
    seen: set[str] = set()
    for model_id in replay[selected_col].astype(str):
        model_id = model_id.strip()
        if model_id and model_id not in seen:
            replay_model_ids.append(model_id)
            seen.add(model_id)
    if not replay_model_ids:
        raise SystemExit(f"selection log has no selected models for source={source_key} task={task}")
    if out_selection is not None:
        out_selection.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "rank": list(range(1, len(replay_model_ids) + 1)),
                "model_id": replay_model_ids,
                "selection_source": "archive_backed_agent_replay_union",
                "source_dataset_key": source_key,
                "task": task,
            }
        ).to_csv(out_selection, index=False)
    print(f"archive_backed_agent_replay={out_replay}")
    if out_selection is not None:
        print(f"archive_backed_agent_selection={out_selection}")
        print(f"archive_backed_agent_selected_models={','.join(replay_model_ids)}")
    print(f"task={task} replay_rows={len(replay)} ledger_rows={len(ledger)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
