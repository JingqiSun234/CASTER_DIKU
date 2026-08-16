#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "code" / "caster" / "scripts"
sys.path.insert(0, str(IMPL))

from run_incremental_ablation_from_archive import (
    _asof_validation,
    _exact_stage_scores,
    _stage_dataset,
    _stage_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--bridge-config", required=True, type=Path)
    parser.add_argument("--draws", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    readout_path = args.stage_root / "hierarchical_forecast_readout.csv"
    required = (
        readout_path,
        args.stage_root / "family_posterior.csv",
        args.stage_root / "inner_weights.csv",
        args.stage_root / "candidate_selection_log.csv",
        args.stage_root / "model_registry.csv",
        args.ledger,
        args.archive,
        args.bridge_config,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing inputs: " + ", ".join(missing))
    shutil.copy2(readout_path, args.stage_root / "forecast_readout.csv")
    shutil.copy2(
        args.stage_root / "hierarchical_posterior_path.csv",
        args.stage_root / "posterior_path.csv",
    )
    readout = pd.read_csv(readout_path)
    scores, validation_path = _exact_stage_scores(
        dataset=_stage_dataset(readout, args.ledger),
        method_id=args.method_id,
        ledger_path=args.ledger,
        archive_path=args.archive,
        bridge_config=args.bridge_config,
        stage_root=args.stage_root,
        readout_path=readout_path,
        draws_path=args.draws,
        hierarchical=True,
    )
    metrics = _stage_metrics(
        readout,
        args.stage,
        args.method_id,
        str(readout_path),
        bridge_config=args.bridge_config,
        formal_scores=scores,
        asof_mixture_weight_validation_path=validation_path,
    )
    metrics.to_csv(args.stage_root / "stage_metrics.csv", index=False)
    validation = _asof_validation(
        readout, stage=args.stage, method_id=args.method_id
    )
    validation.to_csv(
        args.stage_root / "asof_posterior_readout_validation.csv", index=False
    )
    if int(validation["future_snapshot_violation"].astype(bool).sum()) != 0:
        raise SystemExit("future snapshot detected")
    if int(validation["self_target_update_violation"].astype(bool).sum()) != 0:
        raise SystemExit("self target update detected")
    print(f"stage={args.stage} rows={len(readout)} metrics={len(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
