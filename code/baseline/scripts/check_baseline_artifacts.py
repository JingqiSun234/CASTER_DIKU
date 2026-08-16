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
from caster_baselines.artifacts import validate_run_dir


TRUE_TEXT = {"1", "true", "t", "yes", "y"}


def _truthy(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin(TRUE_TEXT)


def validate_formal_reuse(run_dir: str | Path) -> dict:
    ""

    run_dir = Path(run_dir)
    summary = validate_run_dir(run_dir)
    forecast = pd.read_csv(run_dir / "forecast.csv", low_memory=False)
    timing = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    actual_rows = int(len(forecast))

    for document_name, document in (("timing.json", timing), ("run_manifest.json", manifest)):
        for field in ("forecast_rows", "expected_rows"):
            if field in document and int(document[field]) != actual_rows:
                raise ValueError(
                    f"{document_name} {field}={document[field]} does not match "
                    f"forecast.csv rows={actual_rows}"
                )

    if "forecast_id" not in forecast.columns:
        raise ValueError("formal reuse requires forecast_id")
    if forecast["forecast_id"].isna().any() or not forecast["forecast_id"].is_unique:
        raise ValueError("formal reuse requires non-null unique forecast_id values")

    for field in (
        "fallback_group_count",
        "scoring_fallback_group_count",
        "disallowed_fallback_rows",
    ):
        for document_name, document in (("timing.json", timing), ("run_manifest.json", manifest)):
            if field in document and int(document[field]) != 0:
                raise ValueError(
                    f"formal reuse rejected: {document_name} {field}={document[field]}"
                )

    for field in ("proxy_fallback_used", "unsafe_native_proxy_executed"):
        if field in forecast.columns and _truthy(forecast[field]).any():
            raise ValueError(f"formal reuse rejected: forecast.csv contains {field}=true")

    for field in (
        "pred_mean",
        "pred_lower_50",
        "pred_upper_50",
        "pred_lower_90",
        "pred_upper_90",
    ):
        values = pd.to_numeric(forecast[field], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"formal reuse requires finite {field} for every ledger row")

    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument(
        "--formal-reuse",
        action="store_true",
        help="also require complete row counts, unique IDs, finite predictions, and no formal fallback/proxy",
    )
    args = p.parse_args()
    summary = (
        validate_formal_reuse(args.run_dir)
        if args.formal_reuse
        else validate_run_dir(args.run_dir)
    )
    print(
        "ok run_dir={run_dir} forecast_rows={forecast_rows} metrics_rows={metrics_rows} "
        "methods={methods} formal_reuse={formal_reuse}".format(
            **summary, formal_reuse=int(args.formal_reuse)
        )
    )

if __name__ == "__main__":
    main()
