from __future__ import annotations
import numpy as np
import pandas as pd


def point_metrics(readout: pd.DataFrame) -> dict[str, float]:
    rows = readout[readout.get("observed_mask", True).astype(bool)].copy() if "observed_mask" in readout else readout.copy()
    if rows.empty:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan")}
    y = rows["observed_value"].astype(float).to_numpy()
    pred = rows["predictive_mean"].astype(float).to_numpy()
    err = pred - y
    return {"n": int(len(rows)), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err * err)))}


def interval_metrics(readout: pd.DataFrame, *, alpha: float = 0.05) -> dict[str, float]:
    rows = readout[readout.get("observed_mask", True).astype(bool)].copy() if "observed_mask" in readout else readout.copy()
    required = {"observed_value", "lower_95", "upper_95"}
    missing = sorted(required - set(rows.columns))
    if rows.empty or missing:
        return {"coverage_95": float("nan"), "interval_width_95": float("nan"), "wis_95": float("nan")}
    y = rows["observed_value"].astype(float).to_numpy()
    lo = rows["lower_95"].astype(float).to_numpy()
    hi = rows["upper_95"].astype(float).to_numpy()
    width = hi - lo
    coverage = ((y >= lo) & (y <= hi)).mean()
    wis = width + (2.0 / alpha) * (lo - y) * (y < lo) + (2.0 / alpha) * (y - hi) * (y > hi)
    return {"coverage_95": float(coverage), "interval_width_95": float(np.mean(width)), "wis_95": float(np.mean(wis))}


def posterior_diagnostics(weights: pd.DataFrame) -> dict[str, float | str]:
    if weights.empty:
        return {"model_ess": 0.0, "structural_entropy": 0.0, "top_model": ""}
    w = weights["weight"].astype(float).to_numpy()
    w = w / max(w.sum(), 1e-300)
    out: dict[str, float | str] = {
        "model_ess": float(1.0 / np.square(w).sum()),
        "structural_entropy": float(-np.sum(w * np.log(np.maximum(w, 1e-300)))),
        "top1_mass": float(w.max()),
        "top_model": str(weights.iloc[int(np.argmax(w))]["model_id"]),
    }
    if "family" in weights:
        fam = weights.assign(weight=w).groupby("family")["weight"].sum()
        fw = fam.to_numpy()
        out["family_ess"] = float(1.0 / np.square(fw).sum())
        out["family_entropy"] = float(-np.sum(fw * np.log(np.maximum(fw, 1e-300))))
        out["top_family"] = str(fam.sort_values(ascending=False).index[0])
    return out
