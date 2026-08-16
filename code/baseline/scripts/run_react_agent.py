#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caster_baselines.agentic_llm import DEFAULT_QWEN_7B, QwenLocalEngine
from caster_baselines.agentic_react import REACT_SELECTION_POLICIES, run_react_agent_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReAct-style repair-and-execute agentic baseline.")
    parser.add_argument("--manifest", default="data/subset_manifest.csv")
    parser.add_argument("--registry", default="../caster/configs/model_registry.yaml")
    parser.add_argument("--out", default="runs_v3_full/baselines/agentic/react")
    parser.add_argument("--primary-model-path", default=DEFAULT_QWEN_7B)
    parser.add_argument("--fallback-model-path", default="")
    parser.add_argument("--allow-fallback", action="store_true", help="Diagnostic only: allow explicit fallback model use.")
    parser.add_argument("--runtime-budget-seconds", type=float, default=7200.0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--selection-policy",
        choices=REACT_SELECTION_POLICIES,
        default="llm_only",
        help="llm_only selects from dataset context only; no validation scores, guard, or prompt context are used.",
    )
    parser.add_argument("--selection-replay-log", default=None)
    parser.add_argument("--forecast-archive", default=None)
    parser.add_argument("--archive-mode", choices=["off", "required"], default="off")
    parser.add_argument("--charge-selection", action="store_true")
    parser.add_argument("--timing-mode", choices=["alternate_refit", "archive_backed"], default=None)
    parser.add_argument(
        "--exclude-llm-load-from-timing",
        action="store_true",
        help="Preload the local LLM before the timed ReAct loop; generation/control remains charged to update.",
    )
    parser.add_argument(
        "--dataset-key",
        action="append",
        choices=["all", "benchmark_a", "benchmark_b", "benchmark_b_covid", "benchmark_b_flu", "benchmark_b_pooled"],
        default=None,
        help="Dataset key to run. Repeat for multiple datasets, or omit/use all for every manifest dataset.",
    )
    args = parser.parse_args()
    dataset_keys = None if not args.dataset_key or "all" in args.dataset_key else args.dataset_key
    engine = None if args.selection_replay_log else QwenLocalEngine(
        primary_model_path=args.primary_model_path,
        fallback_model_path=args.fallback_model_path,
        allow_fallback=args.allow_fallback,
        runtime_budget_seconds=args.runtime_budget_seconds,
        max_new_tokens=args.max_new_tokens,
        required_model_path=DEFAULT_QWEN_7B,
        require_cuda=True,
    )
    out = run_react_agent_from_manifest(
        manifest_path=args.manifest,
        registry_path=args.registry,
        out_dir=args.out,
        engine=engine,
        dataset_keys=dataset_keys,
        selection_policy=args.selection_policy,
        selection_replay_log=args.selection_replay_log,
        forecast_archive=args.forecast_archive,
        archive_mode=args.archive_mode,
        charge_selection=args.charge_selection,
        timing_mode=args.timing_mode,
        exclude_llm_load_from_timing=args.exclude_llm_load_from_timing,
    )
    timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    print(
        "ok out={out} model=agent_react selection_score_source={selection_score_source} "
        "selection_policy={selection_policy} selection_replay_used={selection_replay_used} "
        "forecast_rows={forecast_rows} expected_rows={expected_rows} "
        "metric_rows={metric_rows} react_group_rows={react_group_rows} artifact_reuse={artifact_reuse}".format(
            out=out,
            **timing,
        )
    )


if __name__ == "__main__":
    main()
