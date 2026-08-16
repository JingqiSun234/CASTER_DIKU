#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caster_baselines.ledger_runner import canonical_model_name, run_baseline_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple baseline from the event-ledger manifest.")
    parser.add_argument(
        "--model",
        required=True,
        help="lastvalue, seasonalnaive and underscore aliases",
    )
    parser.add_argument("--manifest", default="data/subset_manifest.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--enable-native-sidecars", action="store_true")
    parser.add_argument("--native-sidecar-root", default="")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help=(
            "Diagnostic only: allow non-sleeping fallback rows. Provenance-marked "
            "train structural prefixes are always retained for sleeping-mask alignment."
        ),
    )
    args = parser.parse_args()

    model = canonical_model_name(args.model)
    out = run_baseline_from_manifest(
        manifest_path=args.manifest,
        out_dir=args.out,
        model=model,
        enable_native_sidecars=args.enable_native_sidecars,
        native_sidecar_root=args.native_sidecar_root or None,
        fail_on_fallback=not args.allow_fallback,
    )
    with open(out / "timing.json", "r", encoding="utf-8") as f:
        timing = json.load(f)
    print(
        "ok out={out} model={model} forecast_rows={forecast_rows} expected_rows={expected_rows} "
        "metric_rows={metric_rows} fallback_counts={fallback_counts}".format(
            out=out,
            model=model,
            forecast_rows=timing["forecast_rows"],
            expected_rows=timing["expected_rows"],
            metric_rows=timing["metric_rows"],
            fallback_counts=json.dumps(timing["fallback_counts"], sort_keys=True),
        )
    )


if __name__ == "__main__":
    main()
