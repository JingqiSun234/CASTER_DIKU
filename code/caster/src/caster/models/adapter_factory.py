from __future__ import annotations

from importlib import import_module
import json
from typing import Iterable

import pandas as pd

from .base_adapter import BaseCandidateAdapter, validate_adapter_contract


def _parse_hyperparams(value: object) -> dict[str, object]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return dict(value)
    text = str(value).strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid hyperparams_json: {text[:80]}") from exc
    if not isinstance(obj, dict):
        raise ValueError("hyperparams_json must decode to a dictionary")
    return obj


def import_object(path: str):
    if not path or "." not in path:
        raise ValueError(f"adapter_path must be fully qualified, got {path!r}")
    module_name, class_name = str(path).rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def instantiate_adapter_from_row(row: pd.Series | dict[str, object]) -> BaseCandidateAdapter:
    mapping = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    cls = import_object(str(mapping.get("adapter_path", "")))
    params = _parse_hyperparams(mapping.get("hyperparams_json"))
    adapter = cls(**params)
                                                                                     
                                                                                       
    if str(getattr(adapter, "model_id", mapping.get("model_id", ""))) != str(mapping.get("model_id")):
        try:
            adapter.model_id = str(mapping.get("model_id"))
        except Exception:
            pass
    if str(getattr(adapter, "family", mapping.get("family", ""))) != str(mapping.get("family")):
        try:
            adapter.family = str(mapping.get("family"))
        except Exception:
            pass
    violations = validate_adapter_contract(adapter)
    if not violations.empty:
        raise ValueError(f"Adapter contract failed for {mapping.get('model_id')}: {violations.to_dict(orient='records')}")
    return adapter


def instantiate_adapters_from_registry(
    registry: pd.DataFrame,
    *,
    model_ids: Iterable[str] | None = None,
    enabled_only: bool = True,
) -> list[BaseCandidateAdapter]:
    df = registry.copy()
    if enabled_only and "enabled" in df.columns:
        df = df[df["enabled"].map(lambda x: bool(x) if isinstance(x, bool) else str(x).strip().lower() not in {"0", "false", "no", "disabled"})]
    if model_ids is not None:
        order = [str(x) for x in model_ids]
        df = df[df["model_id"].astype(str).isin(order)].copy()
        rank = {mid: i for i, mid in enumerate(order)}
        df["_rank"] = df["model_id"].astype(str).map(rank)
        df = df.sort_values("_rank")
    if df.empty:
        raise ValueError("No registry rows available for adapter instantiation")
    return [instantiate_adapter_from_row(row) for _, row in df.iterrows()]
