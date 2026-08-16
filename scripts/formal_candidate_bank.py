""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FORMAL_REGISTRY = (
    ROOT
    / "code/caster/configs/"
    "model_registry.yaml"
)


@lru_cache(maxsize=1)
def formal_candidate_model_ids() -> tuple[str, ...]:
    payload = yaml.safe_load(FORMAL_REGISTRY.read_text(encoding="utf-8"))
    rows = payload.get("candidates", payload.get("models", []))
    enabled = [
        row
        for row in rows
        if str(row.get("enabled", True)).strip().lower()
        in {"true", "1", "t", "yes"}
    ]
    model_ids = tuple(
        str(row.get("model_id", row.get("id", ""))).strip()
        for row in enabled
    )
    if len(model_ids) != 27 or len(set(model_ids)) != 27:
        raise RuntimeError("formal registry must contain exactly 27 candidates")
    return model_ids


FORMAL_CANDIDATE_COUNT = len(formal_candidate_model_ids())
FORMAL_RESULT_CANDIDATE_PROFILE = "formal_27_country_macro_v1"
FORMAL_RESULT_ELIGIBLE_MODEL_IDS = formal_candidate_model_ids()
FORMAL_RESULT_EXCLUDED_MODEL_IDS: tuple[str, ...] = ()
FORMAL_RESULT_ELIGIBLE_CANDIDATE_COUNT = FORMAL_CANDIDATE_COUNT
FORMAL_RESULT_TOP_K = 10
FORMAL_CACHE_TAG = "k27__draws10__seed42"
FORMAL_SELECTION_DIR = "K_27"
FORMAL_ARCHIVE_NAME = "forecast_archive_all27.csv"
FORMAL_POOLED_MANIFEST_NAME = "forecast_archive_all27_manifest.json"
FORMAL_RANKING_NAME = "candidate_selection_all27.csv"
FORMAL_TEST_RANKING_NAME = "test_rmse_ranking_all27.csv"
FORMAL_AGENT_REGISTRY_SUFFIX = "agent_registry_all27.csv"


def formal_result_eligibility_profile() -> tuple[
    str, tuple[str, ...], tuple[str, ...]
]:
    return (
        FORMAL_RESULT_CANDIDATE_PROFILE,
        FORMAL_RESULT_ELIGIBLE_MODEL_IDS,
        FORMAL_RESULT_EXCLUDED_MODEL_IDS,
    )


def cache_tag(*, n_draws: int = 10, base_seed: int = 42) -> str:
    return f"k27__draws{int(n_draws)}__seed{int(base_seed)}"
