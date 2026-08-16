#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caster_baselines.data_validation import validation_event_ledgers_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate curated subset event ledger compatibility.")
    parser.add_argument("--manifest", default="data/subset_manifest.csv")
    parser.add_argument("--out", default="reports/event_ledger_validation.md")
    args = parser.parse_args()
    results = validation_event_ledgers_from_manifest(Path(args.manifest), Path(args.out), root=ROOT)
    datasets = ",".join(result.dataset_key for result in results)
    print(f"ok out={args.out} datasets={datasets} rows={sum(r.row_count for r in results)}")


if __name__ == "__main__":
    main()
