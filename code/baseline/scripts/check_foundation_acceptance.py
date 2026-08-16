#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REQUIRED_ARTIFACTS = (
    "forecast.csv",
    "metrics.csv",
    "timing.json",
    "run_manifest.json",
    "training_log.csv",
    "dependency_report.json",
)
NON_NUMERIC_METRIC_COLUMNS = {
    "dataset_key",
    "dataset",
    "method",
    "mode",
    "mode_kind",
    "forecast_strategy",
    "component",
    "split",
}
MODEL_DIRS = {"chronos": "chronos", "timesfm": "timesfm"}


class FoundationAcceptanceError(RuntimeError):
    pass


def finite_metrics(metrics: pd.DataFrame) -> None:
    for col in metrics.columns:
        if col in NON_NUMERIC_METRIC_COLUMNS:
            continue
        values = pd.to_numeric(metrics[col], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise FoundationAcceptanceError(f"metrics.csv numeric column is non-finite or unparsable: {col}")


def expected_rows(manifest_path: Path) -> int:
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    return int(pd.to_numeric(manifest["ledger_rows"], errors="raise").sum())


def check_one(run_dir: Path, model: str, expected: int) -> dict[str, object]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).exists()]
    if missing:
        raise FoundationAcceptanceError(f"{run_dir} missing required artifacts: {missing}")
    if (run_dir / "blocker_report.csv").exists():
        raise FoundationAcceptanceError(f"{run_dir} contains blocker_report.csv")

    forecast = pd.read_csv(run_dir / "forecast.csv", low_memory=False)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    timing = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    dependency_report = json.loads((run_dir / "dependency_report.json").read_text(encoding="utf-8"))
    checks = {
        "forecast.csv": int(len(forecast)),
        "timing.expected_rows": int(timing.get("expected_rows", -1)),
        "timing.forecast_rows": int(timing.get("forecast_rows", -1)),
        "run_manifest.expected_rows": int(run_manifest.get("expected_rows", -1)),
        "run_manifest.forecast_rows": int(run_manifest.get("forecast_rows", -1)),
    }
    mismatches = {name: value for name, value in checks.items() if value != expected}
    if mismatches:
        raise FoundationAcceptanceError(f"{run_dir} row-count mismatch expected={expected} actual={mismatches}")
    if int(len(metrics)) <= 0:
        raise FoundationAcceptanceError(f"{run_dir} metrics.csv is empty")
    finite_metrics(metrics)

    for key in ("python_executable", "python_version", "sys_prefix", "package_versions", "checkpoint_id", "checkpoint_path", "device"):
        if key not in dependency_report:
            raise FoundationAcceptanceError(f"{run_dir} dependency_report.json missing {key}")
        if key not in run_manifest:
            raise FoundationAcceptanceError(f"{run_dir} run_manifest.json missing {key}")
    if run_manifest.get("no_leakage_rule") != "history uses only panel_time <= forecast_origin":
        raise FoundationAcceptanceError(f"{run_dir} run_manifest.json missing no-leakage rule")
    if "checkpoint_sha256" not in run_manifest:
        raise FoundationAcceptanceError(f"{run_dir} run_manifest.json missing checkpoint_sha256")

    return {
        "model": model,
        "run_dir": str(run_dir),
        "forecast_rows": int(len(forecast)),
        "metric_rows": int(len(metrics)),
        "python_executable": str(run_manifest["python_executable"]),
        "checkpoint_id": str(run_manifest["checkpoint_id"]),
        "checkpoint_path": str(run_manifest["checkpoint_path"]),
        "interval_source": str(run_manifest.get("interval_source", "")),
        "status": "PASS",
    }


def render(rows: list[dict[str, object]], *, status: str = "PASS") -> str:
    lines = [
        "# Foundation Acceptance",
        "",
        f"- status: `{status}`",
        "",
        "| Model | Forecast rows | Metric rows | Python | Checkpoint ID | Checkpoint path | Interval source | Run dir | Status | Reason |",
        "|---|---:|---:|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['forecast_rows']} | {row['metric_rows']} | "
            f"`{row['python_executable']}` | `{row['checkpoint_id']}` | `{row['checkpoint_path']}` | "
            f"`{row['interval_source']}` | `{row['run_dir']}` | {row['status']} | {row.get('reason', '')} |"
        )
    return "\n".join(lines) + "\n"


def check_foundation_acceptance(manifest: Path, runs_root: Path, models: list[str], out: Path) -> list[dict[str, object]]:
    expected = expected_rows(manifest)
    rows = []
    for model in models:
        key = model.strip().lower()
        if key not in MODEL_DIRS:
            raise FoundationAcceptanceError(f"unknown foundation model for acceptance: {model}")
        rows.append(check_one(runs_root / MODEL_DIRS[key], key, expected))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check foundation model artifact acceptance.")
    parser.add_argument("--manifest", default="data/subset_manifest.csv")
    parser.add_argument("--runs-root", default="runs_v3_full/baselines/foundation")
    parser.add_argument("--models", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    models = [part.strip() for part in args.models.split(",") if part.strip()]
    manifest = Path(args.manifest)
    runs_root = Path(args.runs_root)
    out = Path(args.out)
    expected = expected_rows(manifest)
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for model in models:
        key = model.strip().lower()
        try:
            if key not in MODEL_DIRS:
                raise FoundationAcceptanceError(f"unknown foundation model for acceptance: {model}")
            rows.append(check_one(runs_root / MODEL_DIRS[key], key, expected))
        except Exception as exc:
            failures.append(f"{key}: {exc}")
            rows.append({
                "model": key,
                "run_dir": str(runs_root / MODEL_DIRS.get(key, key)),
                "forecast_rows": "",
                "metric_rows": "",
                "python_executable": "",
                "checkpoint_id": "",
                "checkpoint_path": "",
                "interval_source": "",
                "status": "FAIL",
                "reason": str(exc).replace("|", "/"),
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows, status="FAIL" if failures else "PASS"), encoding="utf-8")
    if failures:
        print(f"FAIL out={args.out} failures={failures}")
        raise SystemExit(1)
    print(f"PASS out={args.out} models={len(rows)}")


if __name__ == "__main__":
    main()
