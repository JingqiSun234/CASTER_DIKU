#!/usr/bin/env python
from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from caster.models import read_registry, embed_registry


def main() -> None:
    ap = ArgumentParser(description="Create deterministic description embeddings for model-skill retrieval.")
    ap.add_argument("--registry", default="configs/model_registry.example.yaml")
    ap.add_argument("--out", default="artifacts/candidate_embeddings.csv")
    ap.add_argument("--dim", type=int, default=128)
    args = ap.parse_args()
    reg = read_registry(args.registry)
    emb = embed_registry(reg, dim=args.dim)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
                                                                                   
    emb.to_csv(out, index=False)
    print(f"registry_rows={len(reg)} embedding_rows={len(emb)} dim={args.dim} out={out}")

if __name__ == "__main__":
    main()
