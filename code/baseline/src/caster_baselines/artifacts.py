from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FORECAST_REQUIRED = {
    "method", "entity_id", "forecast_origin", "target_time", "component", "horizon",
    "y_true", "pred_mean", "pred_lower_50", "pred_upper_50", "pred_lower_90", "pred_upper_90", "split"
}
METRICS_REQUIRED = {
    "method", "component", "horizon", "split", "n", "mae", "rmse", "gaussian_nll",
    "coverage_50", "coverage_90", "width_50", "width_90"
}


def validate_run_dir(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    files = {
        "forecast": run_dir / "forecast.csv",
        "metrics": run_dir / "metrics.csv",
        "timing": run_dir / "timing.json",
        "manifest": run_dir / "run_manifest.json",
    }
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required artifacts in {run_dir}: {missing}")
    forecast = pd.read_csv(files["forecast"], low_memory=False)
    metrics = pd.read_csv(files["metrics"])
    miss_f = FORECAST_REQUIRED - set(forecast.columns)
    miss_m = METRICS_REQUIRED - set(metrics.columns)
    if miss_f:
        raise ValueError(f"forecast.csv missing columns: {sorted(miss_f)}")
    if miss_m:
        raise ValueError(f"metrics.csv missing columns: {sorted(miss_m)}")
    with open(files["timing"], "r", encoding="utf-8") as f:
        timing = json.load(f)
    with open(files["manifest"], "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if len(forecast) == 0 or len(metrics) == 0:
        raise ValueError("forecast.csv and metrics.csv must be non-empty")
    return {
        "run_dir": str(run_dir),
        "forecast_rows": int(len(forecast)),
        "metrics_rows": int(len(metrics)),
        "methods": sorted(forecast["method"].unique().tolist()),
        "timing_keys": sorted(timing.keys()),
        "manifest_keys": sorted(manifest.keys()),
    }
