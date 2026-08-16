#!/usr/bin/env python3
""






from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NEWMETHOD_ROOT = ROOT / "code/caster"
if str(NEWMETHOD_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(NEWMETHOD_ROOT / "src"))

from caster.models import (              
    QWEN25_EMBEDDING_PROFILE,
    Qwen25EmbeddingConfig,
    Qwen25HiddenStateEmbedder,
    embed_registry_qwen25,
    fingerprint_local_checkpoint,
    read_registry,
)
from caster.tasks import (              
    QWEN25_MULTISCALE_CONTEXT_PROFILE,
    QWEN25_MULTISCALE_CONTEXT_SCHEMA,
    build_qwen25_multiscale_context,
    load_task_specs,
    selection_context_canonical_json,
    sequence_sketch_canonical_json,
)


TASKS = (
    "benchmark_a",
    "benchmark_b_covid",
    "benchmark_b_flu",
)
TOP_K = 10
ELIGIBLE_CANDIDATE_COUNT = 27
STRUCTURED_CONTEXT_PROFILE = "qwen25_structured_context_v1"
AUGMENTED_CONTEXT_PROFILE = QWEN25_MULTISCALE_CONTEXT_PROFILE
CONTEXT_PROFILES = (
    STRUCTURED_CONTEXT_PROFILE,
    AUGMENTED_CONTEXT_PROFILE,
)
DEFAULT_SHARED = (
    ROOT / "experiments/result_runs_v3_direct_rollout/shared_baseline"
)
DEFAULT_CHECKPOINT = Path("QWEN_CHECKPOINT_NOT_CONFIGURED")
alternate_PROFILE = "formal_27_country_macro_v1"
alternate_RANKING_NAME = "candidate_selection_all27.csv"
SOURCE_COPY_NAMES = (
    "candidate_validation_summary.csv",
    "candidate_validation_normalization.json",
    "candidate_validation_validation.csv",
    "candidate_validation_by_fold.csv",
    "selection_fold_manifest.csv",
    "selection_fold_validation.csv",
    "candidate_pool_eligibility.csv",
)
CANDIDATE_EMBEDDING_INSTRUCTION = (
    "Represent this forecasting candidate for semantic model retrieval. "
    "Emphasize its temporal inductive bias, useful history length, trend and "
    "seasonality behavior, nonlinear dynamics, cross-entity structure, "
    "uncertainty behavior, and suitable forecast horizons.\n"
    "scientific_description="
)
TASK_EMBEDDING_INSTRUCTION = (
    "Represent this forecasting task for matching compatible candidate "
    "models. Use only information available and released by the forecast "
    "origin. Emphasize cadence, "
    "forecast horizons, trend, seasonality, volatility, sparsity, recent "
    "trajectory, cross-entity heterogeneity, cross-stream behavior, and "
    "available auxiliary streams.\n"
    "task_context=\n"
)
TASK_PATHS = {
    "benchmark_a": (
        ROOT / "data/benchmark_a/curated_full_v3_direct_rollout7/daily_panel.csv",
        ROOT / "data/benchmark_a/curated_full_v3_direct_rollout7/event_ledger.csv",
    ),
    "benchmark_b_covid": (
        ROOT
        / "data/benchmark_b/curated_v26_1_release_lag1/benchmark_b_pooled/weekly_panel.csv",
        ROOT
        / "data/benchmark_b/curated_v26_1_release_lag1/benchmark_b_pooled/event_ledger.csv",
    ),
    "benchmark_b_flu": (
        ROOT
        / "data/benchmark_b/curated_v26_1_release_lag1/benchmark_b_pooled/weekly_panel.csv",
        ROOT
        / "data/benchmark_b/curated_v26_1_release_lag1/benchmark_b_pooled/event_ledger.csv",
    ),
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _enabled(value: object) -> bool:
    return (
        bool(value)
        if isinstance(value, bool)
        else str(value).strip().lower()
        not in {"0", "false", "f", "no", "n", "disabled"}
    )


def _load_alternate_root(shared: Path) -> tuple[Path, Path, str]:
    pointer = (
        shared
        / "caster_selections"
        / alternate_PROFILE
        / "active_selection.json"
    )
    if not pointer.is_file():
        raise FileNotFoundError(pointer)
    pointer_sha = _sha256_file(pointer)
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    root = Path(str(payload["selection_root"])).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if root.parent.name != alternate_PROFILE:
        raise ValueError(f"unexpected alternate selection root: {root}")
    manifest = root / "selection_manifest.json"
    if (
        not manifest.is_file()
        or str(payload.get("selection_manifest_sha256", ""))
        != _sha256_file(manifest)
    ):
        raise ValueError("alternate selection pointer/manifest hash mismatch")
    return pointer, root, pointer_sha


def _qwen_checkpoint_contract(checkpoint: Path) -> dict[str, object]:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        str(payload.get("model_type", "")) != "qwen2"
        or "Qwen2ForCausalLM"
        not in [str(value) for value in payload.get("architectures", [])]
        or int(payload.get("hidden_size", -1)) != 3584
        or int(payload.get("num_hidden_layers", -1)) != 28
        or int(payload.get("max_position_embeddings", -1)) < 32768
    ):
        raise ValueError(
            "checkpoint is not the expected Qwen2.5-7B-Instruct architecture"
        )
    return {
        "model_type": payload["model_type"],
        "architectures": payload["architectures"],
        "hidden_size": int(payload["hidden_size"]),
        "num_hidden_layers": int(payload["num_hidden_layers"]),
        "max_position_embeddings": int(payload["max_position_embeddings"]),
        "config_sha256": _sha256_file(config_path),
    }


def _candidate_embedding_text(row: pd.Series) -> str:
    skill = str(row.get("description", "") or "").strip()
    if not skill:
        raise ValueError(
            f"empty scientific description for model_id={row.get('model_id')}"
        )
    return CANDIDATE_EMBEDDING_INSTRUCTION + skill


def _ranking(
    registry: pd.DataFrame,
    candidate_vectors: np.ndarray,
    query_vector: np.ndarray,
    validation: pd.DataFrame,
    *,
    task: str,
    spec: Any,
    context_sha: str,
    validation_sha: str,
) -> pd.DataFrame:
    if candidate_vectors.shape != (len(registry), 3584):
        raise ValueError(
            f"unexpected candidate vector shape {candidate_vectors.shape}"
        )
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if query.shape != (3584,):
        raise ValueError(f"unexpected query vector shape {query.shape}")
    if not np.allclose(
        np.linalg.norm(candidate_vectors, axis=1), 1.0, atol=2e-4
    ) or not np.isclose(np.linalg.norm(query), 1.0, atol=2e-4):
        raise ValueError("Qwen embedding vectors must be L2 normalized")
    required_validation = {
        "model_id",
        "validation_utility_robust_norm",
        "score_source",
    }
    if missing := sorted(required_validation - set(validation.columns)):
        raise ValueError(f"validation summary missing columns {missing}")
    if set(validation["score_source"].astype(str)) != {
        "task_specific_official_validation_folds"
    }:
        raise ValueError("validation source is not the official fold score")

    scores = np.asarray(candidate_vectors, dtype=np.float32) @ query
    frame = registry[["model_id", "family"]].copy()
    frame["embedding_score"] = scores.astype(float)
    frame = frame.merge(
        validation[
            [
                "model_id",
                "validation_utility_robust_norm",
                "score_source",
            ]
        ],
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    if frame.isna().any(axis=None):
        missing_ids = frame.loc[
            frame.isna().any(axis=1), "model_id"
        ].astype(str).tolist()
        raise ValueError(f"ranking has missing values for {missing_ids}")
    frame["validation_utility_robust_norm"] = pd.to_numeric(
        frame["validation_utility_robust_norm"], errors="raise"
    )
    frame["beta_val"] = 1.0
    frame["validation_contribution"] = frame[
        "validation_utility_robust_norm"
    ]
    frame["retrieval_score"] = (
        frame["embedding_score"] + frame["validation_contribution"]
    )
    frame["embedding_backend"] = QWEN25_EMBEDDING_PROFILE
    frame["retrieval_profile"] = (
        "qwen25_hidden_state_plus_official_validation_v1"
    )
    frame["task_id"] = task
    frame["t_sel"] = spec.t_sel
    frame["task_spec_sha256"] = spec.task_spec_sha256
    frame["selection_context_sha256"] = context_sha
    frame["candidate_validation_sha256"] = validation_sha
    frame["runtime_score"] = 0.0
    frame["beta_runtime"] = 0.0
    frame["priority_weight"] = 0.0
    frame["family_diversity_bonus"] = 0.0
    frame = frame.sort_values(
        ["retrieval_score", "model_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def _comparison(
    old_ranking: pd.DataFrame,
    new_ranking: pd.DataFrame,
    *,
    task: str,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    old = old_ranking.head(top_k)[["rank", "model_id", "family"]].copy()
    new = new_ranking.head(top_k)[["rank", "model_id", "family"]].copy()
    old = old.rename(columns={"rank": "old_rank", "family": "old_family"})
    new = new.rename(columns={"rank": "new_rank", "family": "new_family"})
    joined = old.merge(new, on="model_id", how="outer")
    joined.insert(0, "task_id", task)
    joined["family"] = joined["new_family"].where(
        joined["new_family"].notna(), joined["old_family"]
    )
    joined["status"] = np.select(
        [
            joined["old_rank"].notna() & joined["new_rank"].notna(),
            joined["old_rank"].notna(),
        ],
        ["retained", "exited"],
        default="entered",
    )
    joined["rank_change_new_minus_old"] = (
        joined["new_rank"] - joined["old_rank"]
    )
    joined = joined[
        [
            "task_id",
            "model_id",
            "family",
            "old_rank",
            "new_rank",
            "rank_change_new_minus_old",
            "status",
        ]
    ].sort_values(
        ["status", "new_rank", "old_rank", "model_id"],
        kind="mergesort",
        na_position="last",
    )
    old_ids = set(old["model_id"].astype(str))
    new_ids = set(new["model_id"].astype(str))
    intersection = old_ids & new_ids
    union = old_ids | new_ids
    summary = {
        "task_id": task,
        "top_k": top_k,
        "overlap_count": len(intersection),
        "jaccard": len(intersection) / len(union),
        "unchanged_exact_order": old["model_id"].astype(str).tolist()
        == new["model_id"].astype(str).tolist(),
        "same_rank_count": int(
            joined.loc[
                joined["old_rank"].notna() & joined["new_rank"].notna()
            ]
            .eval("old_rank == new_rank")
            .sum()
        ),
        "mean_abs_rank_change_shared": float(
            joined.loc[
                joined["old_rank"].notna() & joined["new_rank"].notna(),
                "rank_change_new_minus_old",
            ]
            .abs()
            .mean()
        ),
        "entered": ";".join(sorted(new_ids - old_ids)),
        "exited": ";".join(sorted(old_ids - new_ids)),
    }
    return joined.reset_index(drop=True), summary


def _complete(root: Path, profile: str) -> bool:
    required = [
        root / "selection_manifest.json",
        root / "candidate_embeddings.npy",
        root / "candidate_embeddings.csv",
        root / "candidate_embedding_manifest.json",
        root / "task_embedding_batch_manifest.json",
        root / "top10_comparison.csv",
        root / "top10_comparison_summary.csv",
        root / "top10_comparison.md",
    ]
    for task in TASKS:
        task_root = root / task
        required.extend(
            [
                task_root / "candidate_ranking_all27.csv",
                task_root / "top10_selection.csv",
                task_root / "selection_context.json",
                task_root / "selection_context.txt",
                task_root / "selection_context_validation.csv",
                task_root / "task_embedding.npy",
                task_root / "task_embedding_manifest.json",
                task_root / "candidate_validation_summary.csv",
                task_root / "candidate_pool_eligibility.csv",
            ]
        )
        if profile == AUGMENTED_CONTEXT_PROFILE:
            required.extend(
                [
                    task_root / "sequence_sketch.json",
                    task_root / "sequence_sketch.txt",
                ]
            )
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        return False
    try:
        return all(
            len(pd.read_csv(root / task / "top10_selection.csv")) == TOP_K
            and len(
                pd.read_csv(root / task / "candidate_ranking_all27.csv")
            )
            == ELIGIBLE_CANDIDATE_COUNT
            for task in TASKS
        )
    except Exception:
        return False


def _write_active_selection_pointer(
    shared: Path,
    profile: str,
    root: Path,
) -> None:
    if not _complete(root, profile):
        raise RuntimeError(f"incomplete Qwen Top-10 artifacts: {root}")
    manifest = root / "selection_manifest.json"
    _write_json(
        shared / "caster_selections" / profile / "active_selection.json",
        {
            "profile": profile,
            "selection_input_hash": root.name,
            "selection_root": str(root.resolve()),
            "selection_manifest": str(manifest.resolve()),
            "selection_manifest_sha256": _sha256_file(manifest),
            "alternate_active_selection_replaced": False,
        },
    )


def _render_comparison_markdown(
    summary: pd.DataFrame, *, profile: str
) -> str:
    lines = [
        f"# Qwen2.5 Top-10 comparison: {profile}",
        "",
        "The candidate forecast archive and official validation utilities are "
        "unchanged. Only the semantic embedding term changed.",
        "",
        "| Task | Overlap | Jaccard | Exact order | Entered | Exited |",
        "|---|---:|---:|:---:|---|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['task_id']} | {int(row['overlap_count'])}/"
            f"{int(row['top_k'])} | {float(row['jaccard']):.3f} | "
            f"{'yes' if bool(row['unchanged_exact_order']) else 'no'} | "
            f"{row['entered'] or '—'} | {row['exited'] or '—'} |"
        )
    lines.extend(
        [
            "",
            "No candidate forecasts, posterior search, result summary, or visualization "
            "experiment was run by this command.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument(
        "--max-estimated-sequence-tokens", type=int, default=8000
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser.parse_args(argv)


def _write_profile_bank(
    *,
    shared: Path,
    profile: str,
    root: Path,
    identity: dict[str, Any],
    registry: pd.DataFrame,
    candidate_embedding_frame: pd.DataFrame,
    candidate_vectors: np.ndarray,
    candidate_manifest: Mapping[str, object],
    task_batch: Any,
    query_indices: Mapping[tuple[str, str], int],
    prepared: Mapping[str, dict[str, Any]],
    specs: Mapping[str, Any],
    alternate_pointer: Path,
    alternate_pointer_sha: str,
    candidate_cache: Path,
) -> pd.DataFrame:
    if _complete(root, profile):
        return pd.read_csv(
            root / "top10_comparison_summary.csv", keep_default_na=False
        )
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "candidate_embeddings.npy", candidate_vectors)
    candidate_embedding_frame.to_csv(
        root / "candidate_embeddings.csv", index=False
    )
    registry[
        ["model_id", "family", "candidate_type", "skill_embedding_text"]
    ].to_csv(root / "candidate_embedding_inputs.csv", index=False)
    _write_json(root / "candidate_embedding_manifest.json", candidate_manifest)
    _write_json(
        root / "task_embedding_batch_manifest.json", task_batch.manifest
    )

    comparisons: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    ranking_sha: dict[str, str] = {}
    for task in TASKS:
        common = prepared[task]
        state = common["variants"][profile]
        index = int(query_indices[(profile, task)])
        task_root = root / task
        task_root.mkdir(parents=True, exist_ok=True)
        ranking = _ranking(
            registry,
            candidate_vectors,
            task_batch.vectors[index],
            common["validation"],
            task=task,
            spec=specs[task],
            context_sha=state["context_sha"],
            validation_sha=common["validation_sha"],
        )
        ranking.to_csv(task_root / "candidate_ranking_all27.csv", index=False)
        ranking.head(TOP_K).to_csv(
            task_root / "top10_selection.csv", index=False
        )
        detail, summary = _comparison(
            common["old_ranking"], ranking, task=task, top_k=TOP_K
        )
        detail.insert(1, "qwen_context_profile", profile)
        summary["qwen_context_profile"] = profile
        comparisons.append(detail)
        summaries.append(summary)
        state["context_validation"].to_csv(
            task_root / "selection_context_validation.csv", index=False
        )
        (task_root / "selection_context.json").write_text(
            selection_context_canonical_json(state["context"]),
            encoding="utf-8",
        )
        (task_root / "selection_context.txt").write_text(
            state["context_text"], encoding="utf-8"
        )
        if profile == AUGMENTED_CONTEXT_PROFILE:
            (task_root / "sequence_sketch.json").write_text(
                sequence_sketch_canonical_json(state["sequence"]),
                encoding="utf-8",
            )
            (task_root / "sequence_sketch.txt").write_text(
                state["sequence_text"], encoding="utf-8"
            )
        (task_root / "task_embedding_input.txt").write_text(
            state["task_embedding_text"], encoding="utf-8"
        )
        np.save(
            task_root / "task_embedding.npy",
            task_batch.vectors[index : index + 1],
        )
        task_manifest = {
            "schema": "caster_qwen25_task_embedding_reference_v1",
            "embedding_contract_sha256": str(
                task_batch.manifest["embedding_contract_sha256"]
            ),
            "parent_batch_manifest_sha256": _sha256_file(
                root / "task_embedding_batch_manifest.json"
            ),
            "collection": {
                "kind": "task_context",
                "task_id": task,
                "qwen_context_profile": profile,
                "context_sha256": state["context_sha"],
                "sequence_sketch_sha256": state.get("sequence_sha", ""),
                "task_embedding_text_sha256": _sha256_bytes(
                    state["task_embedding_text"].encode("utf-8")
                ),
                "batch_row_index": index,
                "query_vector_sha256": _sha256_bytes(
                    np.ascontiguousarray(
                        task_batch.vectors[index], dtype=np.float32
                    ).tobytes()
                ),
            },
        }
        task_manifest["manifest_sha256"] = _canonical_sha256(task_manifest)
        _write_json(
            task_root / "task_embedding_manifest.json", task_manifest
        )
        for name in SOURCE_COPY_NAMES:
            source = common["old_task_root"] / name
            if source.is_file():
                shutil.copy2(source, task_root / name)
        ranking_sha[task] = _sha256_file(
            task_root / "candidate_ranking_all27.csv"
        )

    comparison_frame = pd.concat(comparisons, ignore_index=True)
    comparison_summary = pd.DataFrame(summaries)
    comparison_frame.to_csv(root / "top10_comparison.csv", index=False)
    comparison_summary.to_csv(
        root / "top10_comparison_summary.csv", index=False
    )
    comparison_text = _render_comparison_markdown(
        comparison_summary, profile=profile
    )
    (root / "top10_comparison.md").write_text(
        comparison_text, encoding="utf-8"
    )
    manifest: dict[str, Any] = {
        "schema": "caster_qwen25_top10_bank_v2_context_input_control",
        "created_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "profile": profile,
        "selection_input_hash": root.name,
        "identity": identity,
        "selection_root": str(root.resolve()),
        "candidate_embedding_manifest_sha256": _sha256_file(
            root / "candidate_embedding_manifest.json"
        ),
        "task_ranking_sha256": ranking_sha,
        "top10_comparison_summary_sha256": _sha256_file(
            root / "top10_comparison_summary.csv"
        ),
        "old_profile_preserved": True,
        "old_active_pointer_path": str(alternate_pointer),
        "old_active_pointer_sha256_before": alternate_pointer_sha,
        "candidate_forecast_archive_reused": str(candidate_cache),
        "candidate_forecast_generation_executed": False,
        "downstream_caster_experiment_executed": False,
    }
    _write_json(root / "selection_manifest.json", manifest)
    if _sha256_file(alternate_pointer) != alternate_pointer_sha:
        raise RuntimeError("alternate active selection pointer changed unexpectedly")
    if not _complete(root, profile):
        raise RuntimeError(f"incomplete Qwen Top-10 artifacts: {root}")
    print(f"qwen_top10=created profile={profile} root={root}", flush=True)
    print(comparison_text, flush=True)
    return comparison_summary


def _write_variant_closeness(
    shared: Path,
    roots: Mapping[str, Path],
) -> Path:
    frames: list[pd.DataFrame] = []
    manifests: dict[str, str] = {}
    for profile, root in roots.items():
        frame = pd.read_csv(
            root / "top10_comparison_summary.csv", keep_default_na=False
        )
        frame["qwen_context_profile"] = profile
        frames.append(frame)
        manifests[profile] = _sha256_file(root / "selection_manifest.json")
    by_task = pd.concat(frames, ignore_index=True)
    aggregates = (
        by_task.groupby("qwen_context_profile", as_index=False)
        .agg(
            tasks=("task_id", "nunique"),
            total_overlap=("overlap_count", "sum"),
            mean_overlap=("overlap_count", "mean"),
            mean_jaccard=("jaccard", "mean"),
            total_same_rank=("same_rank_count", "sum"),
            mean_abs_rank_change_shared=(
                "mean_abs_rank_change_shared",
                "mean",
            ),
        )
        .sort_values(
            [
                "total_overlap",
                "total_same_rank",
                "mean_abs_rank_change_shared",
                "qwen_context_profile",
            ],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    aggregates.insert(0, "closeness_rank", range(1, len(aggregates) + 1))
    best_profile = str(aggregates.iloc[0]["qwen_context_profile"])
    task_winners: list[dict[str, object]] = []
    for task, rows in by_task.groupby("task_id", sort=False):
        ordered = rows.sort_values(
            [
                "overlap_count",
                "same_rank_count",
                "mean_abs_rank_change_shared",
                "qwen_context_profile",
            ],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        top = ordered.iloc[0]
        task_winners.append(
            {
                "task_id": task,
                "closer_profile": str(top["qwen_context_profile"]),
                "overlap_count": int(top["overlap_count"]),
                "same_rank_count": int(top["same_rank_count"]),
                "mean_abs_rank_change_shared": float(
                    top["mean_abs_rank_change_shared"]
                ),
            }
        )
    comparison_identity = {
        "schema": "caster_qwen25_input_variant_closeness_identity_v1",
        "selection_manifest_sha256": manifests,
        "alternate_reference_profile": alternate_PROFILE,
        "primary_closeness_metric": (
            f"sum_of_{len(TASKS)}_task_top10_overlap_counts"
        ),
        "tie_breakers": [
            "sum_of_same_rank_counts_desc",
            "mean_absolute_rank_change_shared_asc",
            "profile_name_asc",
        ],
    }
    digest = _canonical_sha256(comparison_identity)
    root = (
        shared
        / "caster_selections/qwen25_context_input_comparison_v1"
        / digest
    )
    root.mkdir(parents=True, exist_ok=True)
    by_task.to_csv(root / "closeness_by_task.csv", index=False)
    aggregates.to_csv(root / "closeness_summary.csv", index=False)
    pd.DataFrame(task_winners).to_csv(
        root / "closeness_task_winners.csv", index=False
    )
    lines = [
        "# Which Qwen Top-10 is closer to the previous Top-10?",
        "",
        f"Overall closer profile: `{best_profile}`.",
        "",
        f"Primary rule: total Top-10 set overlap across {len(TASKS)} tasks. "
        "Ties use exact-rank matches, then mean absolute shared-model rank "
        "movement.",
        "",
        "| Rank | Profile | Total overlap | Mean Jaccard | "
        "Same rank | Mean |rank change| |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for _, row in aggregates.iterrows():
        lines.append(
            f"| {int(row['closeness_rank'])} | "
            f"{row['qwen_context_profile']} | "
            f"{int(row['total_overlap'])} / {len(TASKS) * TOP_K} | "
            f"{float(row['mean_jaccard']):.3f} | "
            f"{int(row['total_same_rank'])} | "
            f"{float(row['mean_abs_rank_change_shared']):.3f} |"
        )
    lines.append("")
    (root / "closeness_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    _write_json(
        root / "comparison_manifest.json",
        {
            "identity": comparison_identity,
            "comparison_input_hash": digest,
            "overall_closer_profile": best_profile,
            "profile_roots": {
                profile: str(path.resolve())
                for profile, path in roots.items()
            },
        },
    )
    profile_root = root.parent
    _write_json(
        profile_root / "active_comparison.json",
        {
            "comparison_input_hash": digest,
            "comparison_root": str(root.resolve()),
            "overall_closer_profile": best_profile,
            "comparison_manifest_sha256": _sha256_file(
                root / "comparison_manifest.json"
            ),
        },
    )
    print(f"qwen_top10_closeness={root}", flush=True)
    print("\n".join(lines), flush=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.top_k != TOP_K:
        raise SystemExit(f"this protocol fixes --top-k={TOP_K}")
    shared = args.shared.resolve()
    checkpoint = args.checkpoint.resolve()
    candidate_cache = (
        shared / "caster_candidates/k27__draws10__seed42"
    ).resolve()
    registry_path = candidate_cache / "model_registry.formal.csv"
    cache_manifest_path = candidate_cache / "manifest.json"
    task_specs_path = ROOT / "configs/caster_task_specs_v20.yaml"
    for path in (
        registry_path,
        cache_manifest_path,
        task_specs_path,
        *[item for pair in TASK_PATHS.values() for item in pair],
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    alternate_pointer, alternate_root, alternate_pointer_sha = _load_alternate_root(shared)
    registry_all = read_registry(registry_path)
    registry_all = registry_all[registry_all["enabled"].map(_enabled)].copy()
    if len(registry_all) != ELIGIBLE_CANDIDATE_COUNT:
        raise ValueError(
            "formal source registry is not exactly "
            f"{ELIGIBLE_CANDIDATE_COUNT} models"
        )
    specs = load_task_specs(task_specs_path)
    prepared: dict[str, dict[str, Any]] = {}
    eligible_order: list[str] | None = None
    panel_cache: dict[Path, pd.DataFrame] = {}
    ledger_cache: dict[Path, pd.DataFrame] = {}
    for task in TASKS:
        print(f"prepare_context task={task}", flush=True)
        old_task_root = alternate_root / task
        old_ranking_path = old_task_root / alternate_RANKING_NAME
        validation_path = old_task_root / "candidate_validation_summary.csv"
        old_ranking = pd.read_csv(old_ranking_path, keep_default_na=False)
        validation = pd.read_csv(validation_path, keep_default_na=False)
        if len(old_ranking) != ELIGIBLE_CANDIDATE_COUNT:
            raise ValueError(
                f"alternate eligible ranking is not 27 rows for task={task}"
            )
        task_ids = old_ranking["model_id"].astype(str).tolist()
        if eligible_order is None:
            eligible_order = task_ids
        elif set(task_ids) != set(eligible_order):
            raise ValueError("eligible candidate set differs across tasks")
        if set(validation["model_id"].astype(str)) != set(task_ids):
            raise ValueError(f"validation candidate set mismatch task={task}")

        panel_path, ledger_path = TASK_PATHS[task]
        panel = panel_cache.setdefault(
            panel_path, pd.read_csv(panel_path, low_memory=False)
        )
        ledger = ledger_cache.setdefault(
            ledger_path,
            pd.read_csv(ledger_path, keep_default_na=False, low_memory=False),
        )
        combined, combined_text, combined_sha, combined_validation = (
            build_qwen25_multiscale_context(
                panel,
                ledger,
                specs[task],
                max_estimated_sequence_tokens=int(
                    args.max_estimated_sequence_tokens
                ),
            )
        )
        if str(combined.get("schema", "")) != QWEN25_MULTISCALE_CONTEXT_SCHEMA:
            raise ValueError(f"wrong combined context schema task={task}")
        structured_marker = "--- structured_context ---\n"
        sequence_marker = "--- release_time_valid_multiscale_sequence ---\n"
        if (
            structured_marker not in combined_text
            or sequence_marker not in combined_text
        ):
            raise ValueError(f"combined context markers missing task={task}")
        structured_text = (
            combined_text.split(structured_marker, 1)[1]
            .split(sequence_marker, 1)[0]
            .rstrip()
            + "\n"
        )
        sequence_text = combined_text.split(sequence_marker, 1)[1]
        structured_validation = combined_validation[
            combined_validation["validation_stage"].astype(str).eq(
                "structured_context"
            )
        ].drop(columns=["validation_stage"])
        structured_state = {
            "context": combined["base_context"],
            "context_text": structured_text,
            "context_sha": str(combined["base_context_sha256"]),
            "context_validation": structured_validation,
            "task_embedding_text": TASK_EMBEDDING_INSTRUCTION
            + structured_text,
        }
        augmented_state = {
            "context": combined,
            "context_text": combined_text,
            "context_sha": combined_sha,
            "context_validation": combined_validation,
            "sequence": combined[
                "causal_multiscale_sequence_sketch"
            ],
            "sequence_text": sequence_text,
            "sequence_sha": str(combined["sequence_sketch_sha256"]),
            "task_embedding_text": TASK_EMBEDDING_INSTRUCTION
            + combined_text,
        }
        prepared[task] = {
            "old_task_root": old_task_root,
            "old_ranking": old_ranking,
            "validation": validation,
            "validation_sha": _sha256_file(validation_path),
            "panel_path": panel_path,
            "ledger_path": ledger_path,
            "variants": {
                STRUCTURED_CONTEXT_PROFILE: structured_state,
                AUGMENTED_CONTEXT_PROFILE: augmented_state,
            },
        }

    assert eligible_order is not None
    registry = (
        registry_all.set_index("model_id", drop=False)
        .loc[eligible_order]
        .reset_index(drop=True)
    )
    registry["skill_embedding_text"] = registry.apply(
        _candidate_embedding_text, axis=1
    )
    candidate_texts = registry["skill_embedding_text"].astype(str).tolist()
    checkpoint_contract = _qwen_checkpoint_contract(checkpoint)
    print("fingerprint_checkpoint", flush=True)
    checkpoint_identity = fingerprint_local_checkpoint(checkpoint)
    qwen_config = {
        "checkpoint_sha256": checkpoint_identity["checkpoint_sha256"],
        "device": str(args.device),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "padding_side": "left",
        "pooling": "last_nonpadding_token",
        "model_dtype": "bfloat16",
        "output_dtype": "float32_l2_normalized",
    }
    common_identity = {
        "schema": "caster_qwen25_top10_identity_common_v2",
        "source_candidate_cache": str(candidate_cache),
        "source_candidate_cache_manifest_sha256": _sha256_file(
            cache_manifest_path
        ),
        "source_registry_sha256": _sha256_file(registry_path),
        "alternate_selection_pointer_sha256": alternate_pointer_sha,
        "alternate_selection_manifest_sha256": _sha256_file(
            alternate_root / "selection_manifest.json"
        ),
        "eligible_model_ids": eligible_order,
        "source_candidate_count": ELIGIBLE_CANDIDATE_COUNT,
        "eligible_candidate_count": ELIGIBLE_CANDIDATE_COUNT,
        "excluded_model_ids": [],
        "top_k": TOP_K,
        "score_formula": "qwen_cosine+1.0*validation_utility_robust_norm",
        "beta_val": 1.0,
        "beta_runtime": 0.0,
        "candidate_embedding_instruction_sha256": _sha256_bytes(
            CANDIDATE_EMBEDDING_INSTRUCTION.encode("utf-8")
        ),
        "task_embedding_instruction_sha256": _sha256_bytes(
            TASK_EMBEDDING_INSTRUCTION.encode("utf-8")
        ),
        "candidate_embedding_texts_sha256": _canonical_sha256(candidate_texts),
        "qwen_checkpoint_contract": checkpoint_contract,
        "qwen_checkpoint_identity": checkpoint_identity,
        "qwen_inference": qwen_config,
        "implementation_sha256": {
            "qwen_embedding": _sha256_file(
                NEWMETHOD_ROOT / "src/caster/models/qwen25_embedding.py"
            ),
            "sequence_sketch": _sha256_file(
                NEWMETHOD_ROOT / "src/caster/tasks/sequence_sketch.py"
            ),
            "combined_context": _sha256_file(
                NEWMETHOD_ROOT / "src/caster/tasks/qwen_context.py"
            ),
            "selection_script": _sha256_file(Path(__file__).resolve()),
        },
        "candidate_forecasts_reused": True,
        "candidate_models_executed": False,
        "test_metrics_used_for_selection": False,
        "downstream_experiments_executed": False,
    }
    identities: dict[str, dict[str, Any]] = {}
    roots: dict[str, Path] = {}
    for profile in CONTEXT_PROFILES:
        identity = {
            **common_identity,
            "schema": "caster_qwen25_top10_identity_v2_context_input_control",
            "profile": profile,
            "sequence_sketch_in_task_input": profile
            == AUGMENTED_CONTEXT_PROFILE,
            "tasks": {
                task: {
                    "task_spec_sha256": specs[task].task_spec_sha256,
                    "panel_sha256": _sha256_file(
                        prepared[task]["panel_path"]
                    ),
                    "ledger_sha256": _sha256_file(
                        prepared[task]["ledger_path"]
                    ),
                    "context_sha256": prepared[task]["variants"][profile][
                        "context_sha"
                    ],
                    "sequence_sketch_sha256": prepared[task]["variants"][
                        profile
                    ].get("sequence_sha", ""),
                    "task_embedding_text_sha256": _sha256_bytes(
                        prepared[task]["variants"][profile][
                            "task_embedding_text"
                        ].encode("utf-8")
                    ),
                    "validation_summary_sha256": prepared[task][
                        "validation_sha"
                    ],
                    "old_ranking_sha256": _sha256_file(
                        prepared[task]["old_task_root"]
                        / alternate_RANKING_NAME
                    ),
                }
                for task in TASKS
            },
        }
        identities[profile] = identity
        digest = _canonical_sha256(identity)
        roots[profile] = (
            shared / "caster_selections" / profile / digest
        )

    incomplete = [
        profile
        for profile, root in roots.items()
        if not _complete(root, profile)
    ]
    if incomplete:
        print(f"load_qwen checkpoint={checkpoint}", flush=True)
        embedder = Qwen25HiddenStateEmbedder.from_local_checkpoint(
            Qwen25EmbeddingConfig(
                checkpoint_path=checkpoint,
                device=str(args.device),
                max_length=int(args.max_length),
                batch_size=int(args.batch_size),
            )
        )
        print("embed_candidates count=27", flush=True)
        candidate_embedding_frame, candidate_manifest = (
            embed_registry_qwen25(registry, embedder)
        )
        emb_cols = sorted(
            column
            for column in candidate_embedding_frame.columns
            if column.startswith("emb_")
        )
        candidate_vectors = candidate_embedding_frame[emb_cols].to_numpy(
            dtype=np.float32
        )
        query_keys = [
            (profile, task)
            for profile in CONTEXT_PROFILES
            for task in TASKS
        ]
        query_indices = {
            key: index for index, key in enumerate(query_keys)
        }
        print(
            "embed_tasks "
            f"count={len(query_keys)} "
            f"variants={len(CONTEXT_PROFILES)} tasks={len(TASKS)}",
            flush=True,
        )
        task_batch = embedder.embed_texts(
            [
                prepared[task]["variants"][profile][
                    "task_embedding_text"
                ]
                for profile, task in query_keys
            ]
        )
        if (
            str(candidate_manifest["embedding_contract_sha256"])
            != str(task_batch.manifest["embedding_contract_sha256"])
        ):
            raise ValueError("candidate/query embedding contract mismatch")
        for profile in incomplete:
            _write_profile_bank(
                shared=shared,
                profile=profile,
                root=roots[profile],
                identity=identities[profile],
                registry=registry,
                candidate_embedding_frame=candidate_embedding_frame,
                candidate_vectors=candidate_vectors,
                candidate_manifest=candidate_manifest,
                task_batch=task_batch,
                query_indices=query_indices,
                prepared=prepared,
                specs=specs,
                alternate_pointer=alternate_pointer,
                alternate_pointer_sha=alternate_pointer_sha,
                candidate_cache=candidate_cache,
            )
    else:
        for profile, root in roots.items():
            print(
                f"qwen_top10=skip_existing profile={profile} root={root}",
                flush=True,
            )
    for profile, root in roots.items():
        _write_active_selection_pointer(shared, profile, root)
    if _sha256_file(alternate_pointer) != alternate_pointer_sha:
        raise RuntimeError("alternate active selection pointer changed unexpectedly")
    _write_variant_closeness(shared, roots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
