from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import hashlib
import json

import pandas as pd

                                                                                      
REGISTRY_COLUMNS = [
    "model_id",
    "family",
    "candidate_type",
    "seed",
    "adapter_path",
    "hyperparams_hash",
    "hyperparams_json",
    "description",
    "repo_url",
    "enabled",
    "priority",
    "skill_embedding_text",
    "validation_score",
    "checkpoint_status",
]

REQUIRED_REGISTRY_COLUMNS = [
    "model_id",
    "family",
    "candidate_type",
    "seed",
    "adapter_path",
    "hyperparams_hash",
    "hyperparams_json",
    "description",
    "enabled",
    "priority",
]


@dataclass(frozen=True)
class CandidateSpec:
    model_id: str
    family: str
    candidate_type: str
    seed: int
    adapter_path: str
    hyperparams_hash: str
    description: str = ""
    repo_url: str = ""
    enabled: bool = True
    priority: int = 100
    skill_embedding_text: str = ""
    validation_score: float | None = None
    checkpoint_status: str = "unknown"


def hash_hyperparams(params: Mapping[str, object] | None) -> str:
    return hashlib.sha1(
        json.dumps(params or {}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


def apply_hyperparam_overrides(
    registry: pd.DataFrame,
    overrides: Mapping[str, Mapping[str, object]] | None,
) -> pd.DataFrame:
    ""







    result = registry.copy(deep=True)
    if overrides is None:
        return result
    if not isinstance(overrides, Mapping):
        raise ValueError("hyperparameter overrides must be a model-to-parameters mapping")
    if not overrides:
        return result
    required = {"model_id", "hyperparams_json", "hyperparams_hash"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(
            "registry cannot apply hyperparameter overrides; missing columns: "
            + ",".join(missing)
        )
    model_ids = result["model_id"].astype(str)
    duplicated = sorted(model_ids[model_ids.duplicated(keep=False)].unique())
    if duplicated:
        raise ValueError(
            "registry cannot apply hyperparameter overrides to duplicate model_id values: "
            + ",".join(duplicated)
        )

    registry_ids = set(model_ids)
    override_ids = [str(model_id) for model_id in overrides]
    if len(override_ids) != len(set(override_ids)):
        raise ValueError("hyperparameter overrides contain duplicate normalized model_id values")
    unknown_models = sorted(set(override_ids) - registry_ids)
    if unknown_models:
        raise ValueError(
            "hyperparameter overrides reference unknown model_id values: "
            + ",".join(unknown_models)
        )

    for raw_model_id, patch in overrides.items():
        model_id = str(raw_model_id)
        if not isinstance(patch, Mapping):
            raise ValueError(
                f"hyperparameter override for {model_id} must be a parameter mapping"
            )
        row_position = model_ids.tolist().index(model_id)
        params_column = result.columns.get_loc("hyperparams_json")
        hash_column = result.columns.get_loc("hyperparams_hash")
        raw_params = result.iat[row_position, params_column]
        if isinstance(raw_params, Mapping):
            params = dict(raw_params)
        else:
            text = "" if raw_params is None else str(raw_params).strip()
            try:
                decoded = json.loads(text or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid hyperparams_json for {model_id}: {text[:80]}"
                ) from exc
            if not isinstance(decoded, dict):
                raise ValueError(
                    f"hyperparams_json for {model_id} must decode to an object"
                )
            params = decoded

        patch_keys = {str(key) for key in patch}
        if len(patch_keys) != len(patch):
            raise ValueError(
                f"hyperparameter override for {model_id} contains duplicate normalized keys"
            )
        unknown_keys = sorted(patch_keys - set(params))
        if unknown_keys:
            raise ValueError(
                f"hyperparameter override for {model_id} contains unknown keys: "
                + ",".join(unknown_keys)
            )
        params.update({str(key): value for key, value in patch.items()})
        try:
            canonical = json.dumps(
                params, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            digest = hash_hyperparams(params)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"hyperparameter override for {model_id} is not JSON serializable"
            ) from exc
        result.iat[row_position, params_column] = canonical
        result.iat[row_position, hash_column] = digest

    return result


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    text = str(value).strip().lower()
    return text not in {"0", "false", "f", "no", "n", "disabled"}


def _default_description(row: Mapping[str, object]) -> str:
    parts = [
        str(row.get("candidate_type", "candidate model")),
        str(row.get("family", "unknown family")),
        str(row.get("model_id", "unnamed")),
    ]
    return " ".join(p for p in parts if p and p != "None")


def make_registry(specs: Sequence[CandidateSpec | Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        row = spec.__dict__.copy() if isinstance(spec, CandidateSpec) else dict(spec)
        if "hyperparams_hash" not in row or not row.get("hyperparams_hash"):
            row["hyperparams_hash"] = hash_hyperparams(row.get("hyperparams"))
        if not str(row.get("description", "")).strip():
            row["description"] = _default_description(row)
        row.setdefault("repo_url", "")
        row.setdefault("enabled", True)
        row.setdefault("priority", 100)
        if not str(row.get("checkpoint_status", "")).strip():
            row["checkpoint_status"] = "unknown"
        row.setdefault("validation_score", None)
        if not row.get("skill_embedding_text"):
            # Scientific retrieval text is deliberately separated from internal
            # execution and aggregation metadata.  In particular, model_id,
            # family, and candidate_type are not evidence about model behavior.
            row["skill_embedding_text"] = str(row.get("description", "")).strip()
        row["enabled"] = _coerce_bool(row.get("enabled", True))
        if not row.get("hyperparams_json"):
            row["hyperparams_json"] = json.dumps(row.get("hyperparams") or {}, sort_keys=True, separators=(",", ":"))
        try:
            row["priority"] = int(row.get("priority", 100))
        except Exception:
            row["priority"] = 100
        rows.append({col: row.get(col) for col in REGISTRY_COLUMNS})
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def validate_registry(registry: pd.DataFrame) -> pd.DataFrame:
    violations = []
    missing = sorted(set(REQUIRED_REGISTRY_COLUMNS) - set(registry.columns))
    if missing:
        return pd.DataFrame(
            [{"row": None, "violation": "missing_columns", "details": ",".join(missing)}]
        )
    if registry.empty:
        return pd.DataFrame(
            [{"row": None, "violation": "empty_registry", "details": "registry has no rows"}]
        )
    if registry["model_id"].duplicated().any():
        ids = registry.loc[registry["model_id"].duplicated(), "model_id"].astype(str).unique()[:10]
        violations.append({"row": None, "violation": "duplicate_model_id", "details": ",".join(ids)})
    for col in ["model_id", "family", "candidate_type", "adapter_path", "hyperparams_hash", "description"]:
        bad = registry[col].isna() | (registry[col].astype(str).str.strip() == "")
        for idx in registry.index[bad]:
            violations.append(
                {"row": int(idx), "violation": f"blank_{col}", "details": str(registry.loc[idx, "model_id"])}
            )
    if not pd.to_numeric(registry["seed"], errors="coerce").notna().all():
        violations.append({"row": None, "violation": "invalid_seed", "details": "seed must be numeric"})
    if not pd.to_numeric(registry["priority"], errors="coerce").notna().all():
        violations.append({"row": None, "violation": "invalid_priority", "details": "priority must be numeric"})
    if "enabled" in registry:
        invalid_enabled = registry["enabled"].map(lambda x: str(x).strip().lower() not in {"true", "false", "1", "0", "yes", "no", "t", "f"} if not isinstance(x, bool) else False)
        for idx in registry.index[invalid_enabled]:
            violations.append({"row": int(idx), "violation": "invalid_enabled", "details": str(registry.loc[idx, "model_id"])})
    return pd.DataFrame(violations, columns=["row", "violation", "details"])


def read_registry(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text()) or []
        if isinstance(data, dict):
            data = data.get("candidates", data.get("models", []))
        return make_registry(data)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = data.get("candidates", data.get("models", []))
        return make_registry(data)
    if path.suffix.lower() == ".csv":
        return make_registry(pd.read_csv(path).to_dict(orient="records"))
    raise ValueError(f"Unsupported registry file suffix: {path.suffix}")


def write_registry(registry: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(path, index=False)
    return path
