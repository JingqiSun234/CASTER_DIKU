#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caster_baselines.artifacts import validate_run_dir

REQUIRED = ("forecast.csv", "metrics.csv", "timing.json", "run_manifest.json")


def find_candidate_run_dirs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    candidates = set()
    for name in REQUIRED:
        for path in run_root.rglob(name):
            try:
                relative_parts = path.relative_to(run_root).parts
            except ValueError:
                relative_parts = path.parts
                                                                              
                                                                            
            if "_invalidated" in relative_parts:
                continue
                                                                            
                                                                             
                                                                              
                                        
            if any(part.startswith(".") for part in relative_parts):
                continue
            candidates.add(path.parent)
    if all((run_root / name).exists() for name in REQUIRED):
        candidates.add(run_root)
    return sorted(candidates)


def scan_run_dirs(run_root: Path) -> tuple[list[dict], list[dict]]:
    complete: list[dict] = []
    skipped: list[dict] = []
    for run_dir in find_candidate_run_dirs(run_root):
        missing = [name for name in REQUIRED if not (run_dir / name).exists()]
        if missing:
            skipped.append({"run_dir": str(run_dir), "missing": ",".join(missing)})
            continue
        try:
            summary = validate_run_dir(run_dir)
        except Exception as exc:                                                              
            skipped.append({"run_dir": str(run_dir), "missing": f"invalid:{exc}"})
            continue
        complete.append(summary)
    return complete, skipped


def load_manifest_json(run_dir: Path) -> dict:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_complete_runs(complete: list[dict], out_metrics: Path, out_manifest: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    manifest_rows = []
    for summary in complete:
        run_dir = Path(summary["run_dir"])
        metrics = pd.read_csv(run_dir / "metrics.csv")
        manifest = load_manifest_json(run_dir)
        for col in ("dataset_key", "dataset", "mode"):
            if col in manifest and col not in metrics.columns:
                metrics[col] = manifest[col]
        metrics["run_dir"] = str(run_dir)
        metric_frames.append(metrics)
        manifest_rows.append({
            "run_dir": str(run_dir),
            "methods": ";".join(summary["methods"]),
            "forecast_rows": summary["forecast_rows"],
            "metrics_rows": summary["metrics_rows"],
            "dataset_key": manifest.get("dataset_key", "NA"),
            "dataset": manifest.get("dataset", "NA"),
            "mode": manifest.get("mode", "NA"),
        })
    metrics_out = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    manifest_out = pd.DataFrame(manifest_rows)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.to_csv(out_metrics, index=False)
    manifest_out.to_csv(out_manifest, index=False)
    return metrics_out, manifest_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate baseline run metrics from completed run folders.")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--out-metrics", default="reports/baseline_metrics.csv")
    parser.add_argument("--out-manifest", default="reports/baseline_run_manifest.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    complete, skipped = scan_run_dirs(Path(args.run_root))
    print(f"scan run_root={args.run_root} complete={len(complete)} skipped={len(skipped)}")
    for row in complete:
        print(f"complete run_dir={row['run_dir']} forecast_rows={row['forecast_rows']} metrics_rows={row['metrics_rows']} methods={row['methods']}")
    for row in skipped:
        print(f"skipped run_dir={row['run_dir']} missing={row['missing']}")
    if args.dry_run:
        print("dry_run=true wrote_metrics=false")
        return
    metrics_out, manifest_out = aggregate_complete_runs(complete, Path(args.out_metrics), Path(args.out_manifest))
    print(f"ok out_metrics={args.out_metrics} metric_rows={len(metrics_out)} out_manifest={args.out_manifest} run_rows={len(manifest_out)}")


if __name__ == "__main__":
    main()
