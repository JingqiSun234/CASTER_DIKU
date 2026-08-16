#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caster_baselines.external_forecasting import (
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


def _merge_dataset_runs(
    part_dirs: list[Path],
    out_dir: Path,
    *,
    total_seconds: float,
) -> Path:
    if len(part_dirs) != len(PARALLEL_DATASET_KEYS):
        raise ValueError(
            f"expected {len(PARALLEL_DATASET_KEYS)} Prophet parts, got {len(part_dirs)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = [_read_json(path / "run_manifest.json") for path in part_dirs]
    timings = [_read_json(path / "timing.json") for path in part_dirs]
    forecasts = [_read_forecast(path / "forecast.csv") for path in part_dirs]
    for key, frame in zip(PARALLEL_DATASET_KEYS, forecasts):
        if set(frame["dataset_key"].astype(str)) != {key}:
            raise RuntimeError(
                f"Prophet dataset part mismatch for {key}: "
                f"{sorted(set(frame['dataset_key'].astype(str)))}"
            )
    forecast = pd.concat(forecasts, ignore_index=True)
    if "forecast_id" in forecast.columns and forecast["forecast_id"].astype(str).duplicated().any():
        duplicate_ids = (
            forecast.loc[forecast["forecast_id"].astype(str).duplicated(), "forecast_id"]
            .astype(str)
            .head(10)
            .tolist()
        )
        raise RuntimeError(f"parallel Prophet merge produced duplicate forecast_id values: {duplicate_ids}")

    expected_rows = int(sum(int(item["expected_rows"]) for item in timings))
    if len(forecast) != expected_rows:
        raise RuntimeError(
            f"parallel Prophet row-count mismatch: expected={expected_rows} actual={len(forecast)}"
        )
    forecast.to_csv(out_dir / "forecast.csv", index=False)

    training = pd.concat(
        [pd.read_csv(path / "training_log.csv", keep_default_na=False) for path in part_dirs],
        ignore_index=True,
    )
    training.to_csv(out_dir / "training_log.csv", index=False)
    failed_frames = [
        pd.read_csv(path / "failed_series.csv", keep_default_na=False)
        for path in part_dirs
        if (path / "failed_series.csv").is_file()
    ]
    if failed_frames:
        pd.concat(failed_frames, ignore_index=True).to_csv(
            out_dir / "failed_series.csv", index=False
        )

    metrics = summarize_forecasts(forecast)
    finite_failures = finite_metric_check(metrics)
    if finite_failures:
        raise RuntimeError(
            f"parallel Prophet metrics contain non-finite values: {finite_failures}"
        )
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    dependency_reports = [
        _read_json(path / "dependency_report.json") for path in part_dirs
    ]
    if any(report != dependency_reports[0] for report in dependency_reports[1:]):
        raise RuntimeError("parallel Prophet subprocesses reported different dependencies")
    shutil.copyfile(
        part_dirs[0] / "dependency_report.json",
        out_dir / "dependency_report.json",
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
    model_config_by_dataset = {}
    for key, item in zip(PARALLEL_DATASET_KEYS, manifests):
        summaries = list(item.get("dataset_summaries", []))
        if len(summaries) != 1 or str(summaries[0].get("dataset_key")) != key:
            raise RuntimeError(
                f"parallel Prophet manifest for {key} has invalid dataset_summaries"
            )
        dataset_summaries.extend(summaries)
        config_by_dataset = item.get("model_config_by_dataset", {})
        if isinstance(config_by_dataset, dict) and isinstance(
            config_by_dataset.get(key), dict
        ):
            config = dict(config_by_dataset[key])
        elif isinstance(item.get("model_config"), dict):
            config = dict(item["model_config"])
        else:
                                                                         
                                                                     
            config = {"yearly_seasonality_mode": "auto"}
        model_config_by_dataset[key] = config
    uniform_model_config = (
        next(iter(model_config_by_dataset.values()))
        if len(
            {
                json.dumps(value, sort_keys=True)
                for value in model_config_by_dataset.values()
            }
        )
        == 1
        else None
    )
    if uniform_model_config is None:
        manifest.pop("model_config", None)
    else:
        manifest["model_config"] = uniform_model_config
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
            "model_config_scope": (
                "uniform" if uniform_model_config is not None else "dataset_specific"
            ),
            "model_config_by_dataset": model_config_by_dataset,
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
    processes: list[subprocess.Popen] = []
    for key, part_dir in zip(PARALLEL_DATASET_KEYS, part_dirs):
        yearly_mode = (
            args.benchmark_b_yearly_seasonality_mode
            if key == "benchmark_b"
            else "auto"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--manifest",
            str(Path(args.manifest).resolve()),
            "--out",
            str(part_dir),
            "--dataset-key",
            key,
            "--min-train-rows",
            str(int(args.min_train_rows)),
            "--benchmark-b-yearly-seasonality-mode",
            yearly_mode,
        ]
        if args.allow_fallback:
            command.append("--allow-fallback")
        print(
            f"prophet_parallel_launch dataset={key} out={part_dir}",
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
                raise RuntimeError(f"parallel Prophet subprocess failed: {failed}")
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
        total_seconds=time.time() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prophet baseline from selected or all event ledgers.")
    parser.add_argument("--manifest", default="data/subset_manifest.csv")
    parser.add_argument("--out", default="runs/baselines/prophet")
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
            "them into the same canonical Prophet artifact."
        ),
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=0,
        help=(
            "Explicit adapter-input minimum; 0 uses Prophet's native "
            "feasibility minimum after protocol-ledger filtering (2 rows)."
        ),
    )
    parser.add_argument(
        "--benchmark-b-yearly-seasonality-mode",
        choices=["auto", "off"],
        default="auto",
        help=(
            "Benchmark B annual-seasonality policy. 'off' disables both "
            "Prophet's built-in yearly term and the custom season-length "
            "fallback; Benchmark A remains on the alternate auto policy."
        ),
    )
    parser.add_argument("--enable-native-sidecars", action="store_true")
    parser.add_argument("--native-sidecar-root", default="")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help=(
            "Allow last-value fallback rows. Disabled by default because formal "
            "Prophet artifacts must contain only native Prophet forecasts."
        ),
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
        selected = set(dataset_keys or PARALLEL_DATASET_KEYS)
        if (
            args.benchmark_b_yearly_seasonality_mode == "off"
            and selected != {"benchmark_b"}
        ):
            parser.error(
                "Benchmark B yearly-seasonality off requires either "
                "--dataset-key benchmark_b or --parallel-datasets so "
                "Benchmark A keeps its auto configuration"
            )
        yearly_mode = (
            args.benchmark_b_yearly_seasonality_mode
            if selected == {"benchmark_b"}
            else "auto"
        )
        out = run_external_forecaster_from_manifest(
            manifest_path=args.manifest,
            out_dir=args.out,
            model_name="prophet",
            backend="prophet",
            dataset_keys=dataset_keys,
            min_train_rows=args.min_train_rows,
            required_packages=["prophet"],
            enable_native_sidecars=args.enable_native_sidecars,
            native_sidecar_root=args.native_sidecar_root or None,
            fail_on_fallback=not args.allow_fallback,
            model_config={"yearly_seasonality_mode": yearly_mode},
        )
    timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    print(
        "ok out={out} model=prophet forecast_rows={forecast_rows} expected_rows={expected_rows} "
        "metric_rows={metric_rows} group_rows={group_rows} fallback_group_count={fallback_group_count}".format(
            out=out,
            **timing,
        )
    )


if __name__ == "__main__":
    main()
