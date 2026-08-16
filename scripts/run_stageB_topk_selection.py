#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
NEWMETHOD_ROOT = REPO_ROOT / "code" / "caster"
sys.path.insert(0, str(NEWMETHOD_ROOT / "src"))

from caster.models import RetrievalConfig, read_registry, select_top_k_candidates              
from formal_candidate_bank import FORMAL_RESULT_TOP_K


DEFAULT_QUERY = "short-term epidemic forecasting with hospital admissions and emergency department visits"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage B top-K selection and write timing/provenance."
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timing-out", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--algorithm-id", required=True)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=FORMAL_RESULT_TOP_K)
    parser.add_argument("--validation-weight", type=float, default=0.25)
    parser.add_argument("--priority-weight", type=float, default=0.05)
    parser.add_argument("--family-diversity-bonus", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-scope", default="per_algorithm_caster_run")
    parser.add_argument("--shared-selection", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry_path = Path(args.registry)
    embeddings_path = Path(args.embeddings)
    out_path = Path(args.out)
    timing_path = Path(args.timing_out)
    metadata_path = Path(args.metadata_out)

    start = time.perf_counter()
    registry = read_registry(registry_path)
    embeddings = pd.read_csv(embeddings_path)
    selected = select_top_k_candidates(
        registry,
        embeddings,
        query=str(args.query),
        top_k=int(args.top_k),
        config=RetrievalConfig(
            validation_weight=float(args.validation_weight),
            priority_weight=float(args.priority_weight),
            family_diversity_bonus=float(args.family_diversity_bonus),
        ),
    )
    elapsed = time.perf_counter() - start

    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_path, index=False)

    selected_ids = selected["model_id"].astype(str).tolist() if "model_id" in selected.columns else []
    metadata = {
        "stage": "stageB",
        "selection_scope": str(args.selection_scope),
        "selection_rerun_per_algorithm": not bool(args.shared_selection),
        "selection_reused_across_algorithms": bool(args.shared_selection),
        "benchmark": str(args.benchmark),
        "algorithm_id": str(args.algorithm_id),
        "selection_path": str(out_path),
        "registry_path": str(registry_path),
        "registry_sha256": _sha256(registry_path) if registry_path.exists() else "",
        "embeddings_path": str(embeddings_path),
        "embeddings_sha256": _sha256(embeddings_path) if embeddings_path.exists() else "",
        "query": str(args.query),
        "top_k": int(args.top_k),
        "validation_weight": float(args.validation_weight),
        "priority_weight": float(args.priority_weight),
        "family_diversity_bonus": float(args.family_diversity_bonus),
        "seed": int(args.seed),
        "selected_model_ids": selected_ids,
        "selected_model_count": int(len(selected_ids)),
    }
    timing = {
        "records": [{"name": "stageB_topk_selection", "seconds": float(elapsed)}],
        "total_sec": float(elapsed),
        "seed": int(args.seed),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "selection_metadata": metadata,
    }
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"stageB_topk_selection={out_path}")
    print(f"benchmark={args.benchmark} algorithm_id={args.algorithm_id} top_k={args.top_k}")
    print("selected=" + ",".join(selected_ids))
    print(f"timing_json={timing_path}")
    print(f"metadata_json={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
