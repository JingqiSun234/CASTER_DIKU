#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caster_baselines.foundation_forecasting import run_foundation_from_manifest              


def main() -> None:
    parser = argparse.ArgumentParser(description="Run foundation-model baselines from event ledger.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default="data/subset_manifest.csv")
    parser.add_argument("--checkpoint-id", default="")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--external-forecast-csv", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out = run_foundation_from_manifest(
        manifest_path=args.manifest,
        out_dir=args.out,
        model=args.model,
        checkpoint_id=args.checkpoint_id,
        checkpoint_path=args.checkpoint_path,
        external_forecast_csv=args.external_forecast_csv or None,
        device=args.device,
    )
    timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    print(
        "ok out={out} model={model} forecast_rows={forecast_rows} expected_rows={expected_rows} "
        "metric_rows={metric_rows} group_rows={group_rows} fallback_group_count={fallback_group_count}".format(
            out=out,
            model=args.model,
            **timing,
        )
    )


if __name__ == "__main__":
    main()
