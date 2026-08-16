#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caster_baselines.neuralforecast_runner import MODEL_OUTPUT_DIRS, NEURAL_MODELS, run_neuralforecast_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded NeuralForecast baselines from event ledger.")
    parser.add_argument("--model", required=True, choices=sorted(NEURAL_MODELS))
    parser.add_argument("--manifest", default="data/subset_manifest.csv")
    parser.add_argument("--out", default="")
    parser.add_argument("--external-forecast-csv", default="")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--allow-no-test",
        action="store_true",
        help="Allow an update-only ledger (for example embargo-only archive extension) with no test rows.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Diagnostic only: allow provenance-marked last-value fallback rows.",
    )
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    out_dir = args.out or f"runs/baselines/neural/{MODEL_OUTPUT_DIRS[args.model]}"
    out = run_neuralforecast_from_manifest(
        manifest_path=args.manifest,
        out_dir=out_dir,
        model=args.model,
        external_forecast_csv=args.external_forecast_csv or None,
        max_steps=args.max_steps,
        seed=args.seed,
        device=args.device,
        require_test_rows=not args.allow_no_test,
        fail_on_fallback=not args.allow_fallback,
    )
    timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    print(
        "ok out={out} model={model} forecast_rows={forecast_rows} expected_rows={expected_rows} "
        "metric_rows={metric_rows} nonfallback_scoring_rows={nonfallback_scoring_rows} "
        "nonfallback_test_rows={nonfallback_test_rows} fallback_rows={fallback_rows}".format(
            out=out,
            model=args.model,
            **timing,
        )
    )


if __name__ == "__main__":
    main()
