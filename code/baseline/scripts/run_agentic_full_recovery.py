#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caster_baselines.agentic_full_recovery import (
    FULL_RECOVERY_SELECTION_POLICIES,
    run_agentic_full_recovery,
)
from caster_baselines.agentic_llm import DEFAULT_QWEN_7B, DeterministicNoModelComputeEngine, QwenLocalEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agentic Full Recovery baseline.")
    parser.add_argument("--manifest", default="data/subset_manifest.csv")
    parser.add_argument("--registry", default="../caster/configs/model_registry.yaml")
    parser.add_argument("--out", default="runs_v3_full/baselines/agentic/full_recovery")
    parser.add_argument("--primary-model-path", default=DEFAULT_QWEN_7B)
    parser.add_argument("--fallback-model-path", default="")
    parser.add_argument("--allow-fallback", action="store_true", help="Diagnostic only: allow explicit fallback model use.")
    parser.add_argument("--runtime-budget-seconds", type=float, default=7200.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--selection-policy",
        choices=FULL_RECOVERY_SELECTION_POLICIES,
        default="llm_only",
        help="llm_only selects from dataset context only; no validation scores, guard, or prompt context are used.",
    )
    parser.add_argument(
        "--method-name",
        default=None,
        help="Override forecast/run_manifest method name. Defaults to agentic_full_recovery.",
    )
    parser.add_argument("--selection-replay-log", default=None)
    parser.add_argument("--forecast-archive", default=None)
    parser.add_argument("--archive-mode", choices=["off", "required"], default="off")
    parser.add_argument("--charge-selection", action="store_true")
    parser.add_argument("--timing-mode", choices=["alternate_refit", "archive_backed"], default=None)
    parser.add_argument(
        "--selection-engine",
        choices=["qwen", "deterministic_no_model_compute"],
        default="qwen",
        help="Use deterministic_no_model_compute only for diagnostic timing scopes that exclude LLM inference.",
    )
    parser.add_argument(
        "--exclude-llm-load-from-timing",
        action="store_true",
        help="Preload the local LLM before the timed restart loop; generation/control remains charged to update.",
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
    if args.selection_replay_log:
        engine = None
    elif args.selection_engine == "deterministic_no_model_compute":
        engine = DeterministicNoModelComputeEngine()
    else:
        engine = QwenLocalEngine(
            primary_model_path=args.primary_model_path,
            fallback_model_path=args.fallback_model_path,
            allow_fallback=args.allow_fallback,
            runtime_budget_seconds=args.runtime_budget_seconds,
            max_new_tokens=args.max_new_tokens,
            required_model_path=DEFAULT_QWEN_7B,
            require_cuda=True,
        )
    out = run_agentic_full_recovery(
        manifest_path=args.manifest,
        registry_path=args.registry,
        out_dir=args.out,
        engine=engine,
        dataset_keys=dataset_keys,
        selection_policy=args.selection_policy,
        method_name=args.method_name,
        selection_replay_log=args.selection_replay_log,
        forecast_archive=args.forecast_archive,
        archive_mode=args.archive_mode,
        charge_selection=args.charge_selection,
        timing_mode=args.timing_mode,
        exclude_llm_load_from_timing=args.exclude_llm_load_from_timing,
    )
    timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    print(
        "ok out={out} model={model} selection_policy={selection_policy} "
        "forecast_rows={forecast_rows} expected_rows={expected_rows} "
        "metric_rows={metric_rows} restart_group_rows={restart_group_rows} artifact_reuse={artifact_reuse}".format(
            out=out,
            **timing,
        )
    )


if __name__ == "__main__":
    main()
