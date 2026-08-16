#!/usr/bin/env python
from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from caster.models import read_registry, validate_registry, write_registry


def main() -> None:
    ap = ArgumentParser(description="Validate a CASTER candidate model registry.")
    ap.add_argument("--registry", default="configs/model_registry.example.yaml")
    ap.add_argument("--out", default="reports/model_registry_validation.csv")
    ap.add_argument("--normalized-out", default="artifacts/model_registry.csv")
    args = ap.parse_args()
    registry = read_registry(args.registry)
    violations = validate_registry(registry)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    violations.to_csv(args.out, index=False)
    write_registry(registry, args.normalized_out)
    enabled = registry[registry["enabled"].astype(bool)] if "enabled" in registry else registry
    print(f"registry={args.registry}")
    print(f"rows={len(registry)} enabled={len(enabled)} families={registry['family'].nunique()}")
    print(f"violations={len(violations)} out={args.out}")
    if not violations.empty:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
