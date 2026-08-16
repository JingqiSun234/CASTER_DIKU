#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caster_baselines.external_forecasting import (
    STATSFORECAST_MODELS,
    finite_metric_check,
    run_external_forecaster_from_manifest,
)
from caster_baselines.ledger_runner import forecast_strategy_manifest_fields
from caster_baselines.metrics import summarize_forecasts


PARALLEL_DATASET_KEYS = ("benchmark_a", "benchmark_b")
FORECAST_NUMERIC_COLUMNS = (
    "horizon",
    "y_true",
    "pred_mean",
    "pred_lower_50",
    "pred_upper_50",
    "pred_lower_90",
    "pred_upper_90",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_forecast(path: Path) -> pd.DataFrame:
    ""

    frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
    for column in FORECAST_NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _copy_dataset_checkpoint(
    source_out: Path,
    destination_out: Path,
    *,
    model_name: str,
    dataset_key: str,
    overwrite: bool = False,
) -> None:
    ""

    source = source_out / ".checkpoints"
    destination = destination_out / ".checkpoints"
    for suffix in ("forecast.csv", "training.csv", "json"):
        source_path = source / f"{model_name}.{dataset_key}.{suffix}"
        if not source_path.is_file():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        destination_path = destination / source_path.name
                                                                           
                                                                   
                                                                           
        if destination_path.is_file() and not overwrite:
            continue
        shutil.copy2(source_path, destination_path)


def _merge_dataset_runs(
    part_dirs: list[Path],
    out_dir: Path,
    *,
    model_name: str,
    total_seconds: float,
) -> Path:
    if len(part_dirs) != len(PARALLEL_DATASET_KEYS):
        raise ValueError(
            f"expected {len(PARALLEL_DATASET_KEYS)} StatsForecast parts, got {len(part_dirs)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = [_read_json(path / "run_manifest.json") for path in part_dirs]
    timings = [_read_json(path / "timing.json") for path in part_dirs]
    forecasts = [_read_forecast(path / "forecast.csv") for path in part_dirs]
    for key, frame in zip(PARALLEL_DATASET_KEYS, forecasts):
        if set(frame["dataset_key"].astype(str)) != {key}:
            raise RuntimeError(
                f"StatsForecast dataset part mismatch for {key}: "
                f"{sorted(set(frame['dataset_key'].astype(str)))}"
            )
        if set(frame["method"].astype(str)) != {model_name}:
            raise RuntimeError(
                f"StatsForecast model part mismatch for {key}: "
                f"{sorted(set(frame['method'].astype(str)))}"
            )
    forecast = pd.concat(forecasts, ignore_index=True)
    if "forecast_id" in forecast.columns and forecast["forecast_id"].astype(str).duplicated().any():
        duplicate_ids = (
            forecast.loc[forecast["forecast_id"].astype(str).duplicated(), "forecast_id"]
            .astype(str)
            .head(10)
            .tolist()
        )
        raise RuntimeError(
            "parallel StatsForecast merge produced duplicate forecast_id values: "
            f"{duplicate_ids}"
        )

    expected_rows = int(sum(int(item["expected_rows"]) for item in timings))
    if len(forecast) != expected_rows:
        raise RuntimeError(
            f"parallel StatsForecast row-count mismatch: expected={expected_rows} "
            f"actual={len(forecast)}"
        )

    training = pd.concat(
        [pd.read_csv(path / "training_log.csv", keep_default_na=False) for path in part_dirs],
        ignore_index=True,
    )
    failed_frames = [
        pd.read_csv(path / "failed_series.csv", keep_default_na=False)
        for path in part_dirs
        if (path / "failed_series.csv").is_file()
    ]

    metrics = summarize_forecasts(forecast)
    finite_failures = finite_metric_check(metrics)
    if finite_failures:
        raise RuntimeError(
            f"parallel StatsForecast metrics contain non-finite values: {finite_failures}"
        )

    dependency_reports = [
        _read_json(path / "dependency_report.json") for path in part_dirs
    ]
    if any(report != dependency_reports[0] for report in dependency_reports[1:]):
        raise RuntimeError(
            "parallel StatsForecast subprocesses reported different dependencies"
        )

                                                                             
                                                                                
    for name, frame in (
        ("forecast.csv", forecast),
        ("training_log.csv", training),
        ("metrics.csv", metrics),
    ):
        temporary = out_dir / f".{name}.tmp"
        frame.to_csv(temporary, index=False)
        temporary.replace(out_dir / name)
    failed_path = out_dir / "failed_series.csv"
    if failed_frames:
        temporary = out_dir / ".failed_series.csv.tmp"
        pd.concat(failed_frames, ignore_index=True).to_csv(temporary, index=False)
        temporary.replace(failed_path)
    elif failed_path.exists():
        failed_path.unlink()
    shutil.copyfile(
        part_dirs[0] / "dependency_report.json",
        out_dir / "dependency_report.json",
    )

                                                                          
                                                                             
    for key, part_dir in zip(PARALLEL_DATASET_KEYS, part_dirs):
        _copy_dataset_checkpoint(
            part_dir,
            out_dir,
            model_name=model_name,
            dataset_key=key,
            overwrite=True,
        )

    fallback_group_count = int(
        sum(int(item["fallback_group_count"]) for item in timings)
    )
    timing = {
        "total_seconds": round(float(total_seconds), 6),
        "forecast_rows": int(len(forecast)),
        "metric_rows": int(len(metrics)),
        "expected_rows": expected_rows,
        "group_rows": int(len(training)),
        "fallback_group_count": fallback_group_count,
        "dataset_part_total_seconds": {
            key: float(item["total_seconds"])
            for key, item in zip(PARALLEL_DATASET_KEYS, timings)
        },
    }
    (out_dir / "timing.json").write_text(
        json.dumps(timing, indent=2, sort_keys=True), encoding="utf-8"
    )

    manifest = dict(manifests[0])
    dataset_summaries = []
    for key, item in zip(PARALLEL_DATASET_KEYS, manifests):
        summaries = list(item.get("dataset_summaries", []))
        if len(summaries) != 1 or str(summaries[0].get("dataset_key")) != key:
            raise RuntimeError(
                f"parallel StatsForecast manifest for {key} has invalid dataset_summaries"
            )
        dataset_summaries.extend(summaries)
    manifest.update(
        {
            "selected_dataset_keys": [],
            "expected_rows": expected_rows,
            "forecast_rows": int(len(forecast)),
            "dataset_summaries": dataset_summaries,
            "dependency_report": dependency_reports[0],
            "fallback_group_count": fallback_group_count,
            "failed_series_path": "failed_series.csv" if failed_frames else "",
            "dataset_execution_mode": "parallel_subprocesses",
            "parallel_dataset_keys": list(PARALLEL_DATASET_KEYS),
            "native_sidecars_enabled": False,
            "native_sidecar_root": "",
            **forecast_strategy_manifest_fields(forecast),
        }
    )
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out_dir


def _run_parallel_datasets(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    parts_root = out_dir.parent / f".{out_dir.name}.dataset_parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    part_dirs = [parts_root / key for key in PARALLEL_DATASET_KEYS]
    for key, part_dir in zip(PARALLEL_DATASET_KEYS, part_dirs):
        _copy_dataset_checkpoint(
            out_dir,
            part_dir,
            model_name=args.model,
            dataset_key=key,
        )

    processes: list[subprocess.Popen] = []
    for key, part_dir in zip(PARALLEL_DATASET_KEYS, part_dirs):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model",
            args.model,
            "--manifest",
            str(Path(args.manifest).resolve()),
            "--out",
            str(part_dir),
            "--dataset-key",
            key,
            "--min-train-rows",
            str(int(args.min_train_rows)),
        ]
        if args.allow_fallback:
            command.append("--allow-fallback")
        print(
            f"statsforecast_parallel_launch model={args.model} "
            f"dataset={key} out={part_dir}",
            flush=True,
        )
        processes.append(subprocess.Popen(command))
    try:
        while True:
            return_codes = [process.poll() for process in processes]
            failed = [
                (key, code)
                for key, code in zip(PARALLEL_DATASET_KEYS, return_codes)
                if code not in (None, 0)
            ]
            if failed:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    process.wait()
                raise RuntimeError(
                    f"parallel StatsForecast subprocess failed for {args.model}: {failed}"
                )
            if all(code == 0 for code in return_codes):
                break
            time.sleep(0.5)
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise
    return _merge_dataset_runs(
        part_dirs,
        out_dir,
        model_name=args.model,
        total_seconds=time.time() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StatsForecast baselines from event ledger.")
    parser.add_argument("--model", required=True, choices=sorted(STATSFORECAST_MODELS))
    parser.add_argument("--manifest", default="data/subset_manifest.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--dataset-key",
        action="append",
        choices=["all", "benchmark_a", "benchmark_b"],
        default=None,
        help="Dataset key to run. Repeat for multiple datasets, or omit/use all for every manifest dataset.",
    )
    parser.add_argument(
        "--parallel-datasets",
        action="store_true",
        help=(
            "Run Benchmark A and Benchmark B in two subprocesses, then merge "
            "them into one canonical StatsForecast artifact."
        ),
    )
    parser.add_argument("--min-train-rows", type=int, default=3)
    parser.add_argument("--enable-native-sidecars", action="store_true")
    parser.add_argument("--native-sidecar-root", default="")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Diagnostic only: allow provenance-marked last-value fallback groups.",
    )
    args = parser.parse_args()
    dataset_keys = None if not args.dataset_key or "all" in args.dataset_key else args.dataset_key
    if args.parallel_datasets:
        if dataset_keys is not None:
            parser.error("--parallel-datasets requires --dataset-key all (or no dataset key)")
        if args.enable_native_sidecars:
            parser.error("--parallel-datasets does not support native sidecars")
        out = _run_parallel_datasets(args)
    else:
        out = run_external_forecaster_from_manifest(
            manifest_path=args.manifest,
            out_dir=args.out,
            model_name=args.model,
            backend="statsforecast",
            dataset_keys=dataset_keys,
            min_train_rows=args.min_train_rows,
            required_packages=["statsforecast"],
            enable_native_sidecars=args.enable_native_sidecars,
            native_sidecar_root=args.native_sidecar_root or None,
            fail_on_fallback=not args.allow_fallback,
        )
    timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    print(
        "ok out={out} model={model} forecast_rows={forecast_rows} expected_rows={expected_rows} "
        "metric_rows={metric_rows} group_rows={group_rows} fallback_group_count={fallback_group_count}".format(
            out=out,
            model=args.model,
            **timing,
        ),
        flush=True,
    )
                                                                             
                                                                              
    os._exit(0)


if __name__ == "__main__":
    main()
