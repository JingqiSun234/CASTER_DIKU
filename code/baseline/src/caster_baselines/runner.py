from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import summarize_forecasts
from .models import get_baselines

Z50 = 0.67448975
Z90 = 1.64485363


def prepare_panel(panel: pd.DataFrame, entity_cols: list[str], time_col: str, target_cols: list[str]) -> pd.DataFrame:
    df = panel.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df["entity_id"] = df[entity_cols].astype(str).agg("|".join, axis=1)
    needed = ["entity_id", time_col] + target_cols
    out = df[needed].sort_values(["entity_id", time_col]).reset_index(drop=True)
    return out.rename(columns={time_col: "time"})


def split_name(t: pd.Timestamp, train_end: pd.Timestamp, val_end: pd.Timestamp, test_start: pd.Timestamp) -> str:
    if t <= train_end:
        return "train"
    if t <= val_end:
        return "val"
    if t >= test_start:
        return "test"
    return "gap"


def run_baselines(
    panel: pd.DataFrame,
    out_dir: str | Path,
    entity_cols: list[str],
    time_col: str,
    target_cols: list[str],
    horizons: list[int],
    train_end: str,
    val_end: str,
    test_start: str,
    baseline_names: list[str] | None = None,
) -> Path:
    start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    test_start_ts = pd.Timestamp(test_start)
    df = prepare_panel(panel, entity_cols, time_col, target_cols)
    baselines = get_baselines(baseline_names)
    rows = []
    for entity_id, g in df.groupby("entity_id"):
        g = g.sort_values("time").reset_index(drop=True)
        times = list(g["time"])
        time_to_idx = {t: i for i, t in enumerate(times)}
        for origin_idx, origin_time in enumerate(times[:-max(horizons)]):
            split = split_name(origin_time, train_end_ts, val_end_ts, test_start_ts)
            if split == "gap":
                continue
            history = g.iloc[: origin_idx + 1]
            for h in horizons:
                target_idx = origin_idx + h
                if target_idx >= len(g):
                    continue
                target_time = g.loc[target_idx, "time"]
                for component in target_cols:
                    y_true = float(g.loc[target_idx, component])
                    if not np.isfinite(y_true):
                        continue
                    for baseline in baselines:
                        pred, sigma = baseline.predict(history, component, h)
                        sigma = max(float(sigma), 1e-6)
                        rows.append({
                            "method": baseline.name,
                            "entity_id": entity_id,
                            "forecast_origin": origin_time.strftime("%Y-%m-%d"),
                            "target_time": target_time.strftime("%Y-%m-%d"),
                            "component": component,
                            "horizon": int(h),
                            "y_true": y_true,
                            "pred_mean": float(pred),
                            "pred_lower_50": float(pred - Z50 * sigma),
                            "pred_upper_50": float(pred + Z50 * sigma),
                            "pred_lower_90": float(pred - Z90 * sigma),
                            "pred_upper_90": float(pred + Z90 * sigma),
                            "split": split,
                        })
    forecast = pd.DataFrame(rows)
    if forecast.empty:
        raise ValueError("no forecasts generated; check panel length, horizons, and split dates")
    forecast_path = out_dir / "forecast.csv"
    forecast.to_csv(forecast_path, index=False)
    metrics = summarize_forecasts(forecast)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    timing = {"total_seconds": round(time.time() - start, 6), "forecast_rows": int(len(forecast)), "metric_rows": int(len(metrics))}
    with open(out_dir / "timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    manifest = {
        "baseline_names": [b.name for b in baselines],
        "entity_cols": entity_cols,
        "time_col": time_col,
        "target_cols": target_cols,
        "horizons": horizons,
        "train_end": train_end,
        "val_end": val_end,
        "test_start": test_start,
    }
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out_dir
