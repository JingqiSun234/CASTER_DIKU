from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PACKET_KIND = "caster_agent_common_selection_input_v26_1"
CANDIDATE_FIELDS = (
    "model_id", "family", "candidate_type", "description", "skill_embedding_text",
    "embedding_score", "validation_utility_robust_norm", "validation_contribution",
    "retrieval_score", "router_score", "score_source", "t_sel", "max_label_release_time",
)


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenSelectionInput:
    candidates: pd.DataFrame
    manifest: Mapping[str, Any]
    manifest_path: Path
    candidates_path: Path
    manifest_sha256: str
    candidates_sha256: str

    def prompt_context(self) -> str:
        payload = {
            "kind": PACKET_KIND,
            "task_id": self.manifest["task_id"],
            "t_sel": self.manifest["t_sel"],
            "selection_text": self.manifest["selection_text"],
            "selection_context": self.manifest["selection_context"],
            "candidate_fields": list(CANDIDATE_FIELDS),
            "candidate_count": int(self.manifest["candidate_count"]),
            "candidate_rows_sha256": str(self.manifest["candidate_rows_sha256"]),
            "packet_manifest_sha256": self.manifest_sha256,
            "information_policy": self.manifest["information_policy"],
        }
        return "common_selection_input=" + json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def provenance(self) -> dict[str, object]:
        return {
            "selection_input_packet_consumed": True,
            "selection_input_packet_kind": PACKET_KIND,
            "selection_input_packet_sha256": self.manifest_sha256,
            "selection_input_candidates_sha256": self.candidates_sha256,
            "selection_input_candidate_rows_sha256": str(self.manifest["candidate_rows_sha256"]),
            "selection_input_task_id": str(self.manifest["task_id"]),
            "selection_input_t_sel": str(self.manifest["t_sel"]),
            "selection_input_candidate_count": int(self.manifest["candidate_count"]),
            "selection_input_validation_visible": True,
            "selection_input_test_information_visible": False,
        }


def load_common_selection_input(candidates_path: str | Path, manifest_path: str | Path) -> FrozenSelectionInput:
    candidates_file, manifest_file = Path(candidates_path).resolve(), Path(manifest_path).resolve()
    if not candidates_file.is_file() or not manifest_file.is_file():
        raise ValueError("common selection packet files are missing")
    candidates = pd.read_csv(candidates_file, keep_default_na=False)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    required = set(CANDIDATE_FIELDS) | {"common_selection_candidate_row_sha256"}
    if manifest.get("kind") != PACKET_KIND or manifest.get("status") != "frozen":
        raise ValueError("common selection packet identity mismatch")
    if manifest.get("task_id") != "benchmark_b_pooled":
        raise ValueError("noLearning formal selection packet must bind benchmark_b_pooled")
    if manifest.get("canonical_context_schema") != "caster_benchmark_b_canonical_context_v1":
        raise ValueError("stale Benchmark B selection packet is forbidden")
    if manifest.get("input_change_invalidates_forecast_posterior_agent_results") is not True:
        raise ValueError("selection packet lacks input invalidation policy")
    if not required <= set(candidates.columns):
        raise ValueError(f"selection candidates lack fields {sorted(required - set(candidates.columns))}")
    if int(manifest.get("candidate_count", -1)) != len(candidates):
        raise ValueError("selection candidate count mismatch")
    if str(manifest.get("candidates_sha256", "")) != sha256_file(candidates_file):
        raise ValueError("selection candidate file hash mismatch")
    return FrozenSelectionInput(
        candidates=candidates,
        manifest=manifest,
        manifest_path=manifest_file,
        candidates_path=candidates_file,
        manifest_sha256=sha256_file(manifest_file),
        candidates_sha256=sha256_file(candidates_file),
    )


def merge_common_selection_input(registry: pd.DataFrame, frozen: FrozenSelectionInput) -> pd.DataFrame:
    output = registry.copy()
    enabled = output.get("enabled", pd.Series(True, index=output.index)).astype(str).str.lower().isin({"1", "true", "yes", "y", "t"})
    if set(output.loc[enabled, "model_id"].astype(str)) != set(frozen.candidates["model_id"].astype(str)):
        raise ValueError("noLearning Agent and CASTER candidate registries differ")
    attached = [field for field in CANDIDATE_FIELDS if field != "model_id"] + ["common_selection_candidate_row_sha256"]
    output = output.drop(columns=[field for field in attached if field in output.columns])
    output = output.merge(
        frozen.candidates[["model_id", *attached]], on="model_id", how="left", validate="one_to_one"
    )
    if output.loc[enabled, "common_selection_candidate_row_sha256"].astype(str).eq("").any():
        raise ValueError("selection packet merge lost an enabled candidate")
    return output
