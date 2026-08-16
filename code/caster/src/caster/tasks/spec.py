from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

import pandas as pd


ALLOWED_POSTERIOR_SCOPES = {
    "one_task_posterior",
    "pooled_shared_posterior",
    "component_conditioned_posterior",
}


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    dataset: str
    role: str
    natural_language_query: str
    target_components: tuple[str, ...]
    cadence: str
    entity_scope: str
    direct_horizons: tuple[int, ...]
    recursive_horizons: tuple[int, ...]
    forecast_strategies: tuple[str, ...]
    posterior_scope: str
    component_weights: Mapping[str, float]
    strategy_weights: Mapping[str, float]
    horizon_weights: Mapping[str, Mapping[int, float]]
    selection_fold_policy: str
    candidate_selection_cutoff_policy: str
    initial_context_policy: Mapping[str, Any]
    t_sel: str
    t_test: str
    frozen_hash_role: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TaskSpec":
        spec = cls(
            task_id=str(payload["task_id"]),
            dataset=str(payload["dataset"]),
            role=str(payload["role"]),
            natural_language_query=str(payload["natural_language_query"]),
            target_components=tuple(str(x) for x in payload["target_components"]),
            cadence=str(payload["cadence"]),
            entity_scope=str(payload["entity_scope"]),
            direct_horizons=tuple(int(x) for x in payload["direct_horizons"]),
            recursive_horizons=tuple(int(x) for x in payload["recursive_horizons"]),
            forecast_strategies=tuple(str(x) for x in payload["forecast_strategies"]),
            posterior_scope=str(payload["posterior_scope"]),
            component_weights={str(k): float(v) for k, v in payload["component_weights"].items()},
            strategy_weights={str(k): float(v) for k, v in payload["strategy_weights"].items()},
            horizon_weights={
                str(strategy): {int(k): float(v) for k, v in weights.items()}
                for strategy, weights in payload["horizon_weights"].items()
            },
            selection_fold_policy=str(payload["selection_fold_policy"]),
            candidate_selection_cutoff_policy=str(payload["candidate_selection_cutoff_policy"]),
            initial_context_policy=dict(payload["initial_context_policy"]),
            t_sel=str(payload["t_sel"]),
            t_test=str(payload["t_test"]),
            frozen_hash_role=(
                str(payload["frozen_hash_role"])
                if payload.get("frozen_hash_role") is not None
                else None
            ),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.task_id or not self.dataset or not self.target_components:
            raise ValueError("TaskSpec requires task_id, dataset, and target_components")
        if self.posterior_scope not in ALLOWED_POSTERIOR_SCOPES:
            raise ValueError(f"unsupported posterior_scope={self.posterior_scope!r}")
        if self.posterior_scope == "pooled_shared_posterior" and len(self.target_components) < 2:
            raise ValueError("pooled_shared_posterior requires at least two target components")
        if self.posterior_scope == "component_conditioned_posterior" and len(self.target_components) != 1:
            raise ValueError("component_conditioned_posterior requires exactly one target component")
        if set(self.component_weights) != set(self.target_components):
            raise ValueError("component_weights must cover target_components exactly")
        if set(self.strategy_weights) != set(self.forecast_strategies):
            raise ValueError("strategy_weights must cover forecast_strategies exactly")
        for name, values in (("component", self.component_weights), ("strategy", self.strategy_weights)):
            if any(float(v) < 0 for v in values.values()) or abs(sum(values.values()) - 1.0) > 1e-12:
                raise ValueError(f"{name}_weights must be nonnegative and sum to one")
        if len(self.forecast_strategies) != 2:
            raise ValueError("formal TaskSpec requires exactly direct and recursive strategies")
        expected_horizons = {
            self.forecast_strategies[0]: set(self.direct_horizons),
            self.forecast_strategies[1]: set(self.recursive_horizons),
        }
        for strategy, expected in expected_horizons.items():
            weights = self.horizon_weights.get(strategy, {})
            if set(weights) != expected or abs(sum(weights.values()) - 1.0) > 1e-12:
                raise ValueError(f"horizon_weights[{strategy!r}] must cover horizons and sum to one")
        if self.initial_context_policy.get("history_scope") != "all_available_before_t_sel":
            raise ValueError("formal selection requires the full pre-t_sel history scope")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "candidate_selection_cutoff_policy": self.candidate_selection_cutoff_policy,
            "cadence": self.cadence,
            "component_weights": dict(self.component_weights),
            "dataset": self.dataset,
            "direct_horizons": list(self.direct_horizons),
            "entity_scope": self.entity_scope,
            "forecast_strategies": list(self.forecast_strategies),
            "horizon_weights": {
                strategy: {str(k): value for k, value in sorted(weights.items())}
                for strategy, weights in sorted(self.horizon_weights.items())
            },
            "initial_context_policy": dict(self.initial_context_policy),
            "natural_language_query": self.natural_language_query,
            "posterior_scope": self.posterior_scope,
            "recursive_horizons": list(self.recursive_horizons),
            "role": self.role,
            "selection_fold_policy": self.selection_fold_policy,
            "strategy_weights": dict(self.strategy_weights),
            "t_sel": self.t_sel,
            "t_test": self.t_test,
            "target_components": list(self.target_components),
            "task_id": self.task_id,
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.canonical_payload())

    @property
    def task_spec_sha256(self) -> str:
        ""







        payload = self.canonical_payload()
        if self.frozen_hash_role is not None:
            payload["role"] = self.frozen_hash_role
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    @property
    def horizons_by_strategy(self) -> dict[str, tuple[int, ...]]:
        return {
            self.forecast_strategies[0]: self.direct_horizons,
            self.forecast_strategies[1]: self.recursive_horizons,
        }


def filter_rows_to_task_spec(
    frame: pd.DataFrame,
    spec: TaskSpec,
    *,
    require_complete: bool = False,
) -> pd.DataFrame:
    ""

    required = {"component", "forecast_strategy", "horizon"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"task-grid projection missing columns {missing}")
    strategy = frame["forecast_strategy"].fillna("").astype(str).str.strip().str.lower()
    strategy = strategy.replace({"recursive": "recursive_rollout", "rollout": "recursive_rollout"})
    horizon = pd.to_numeric(frame["horizon"], errors="raise").astype(int)
    mask = frame["component"].astype(str).isin(spec.target_components)
    horizon_mask = pd.Series(False, index=frame.index)
    for strategy_name, declared_horizons in spec.horizons_by_strategy.items():
        horizon_mask |= strategy.eq(strategy_name) & horizon.isin(declared_horizons)
    projected = frame.loc[mask & horizon_mask].copy()
    if projected.empty:
        raise ValueError(f"formal task projection is empty for {spec.task_id}")
    projected["forecast_strategy"] = strategy.loc[projected.index]
    projected["horizon"] = horizon.loc[projected.index]
    if require_complete:
        for component in spec.target_components:
            for strategy_name, declared_horizons in spec.horizons_by_strategy.items():
                part = projected[
                    projected["component"].astype(str).eq(component)
                    & projected["forecast_strategy"].astype(str).eq(strategy_name)
                ]
                present = set(pd.to_numeric(part["horizon"], errors="raise").astype(int))
                expected = set(declared_horizons)
                if present != expected:
                    raise ValueError(
                        f"incomplete formal task grid for task={spec.task_id} component={component} "
                        f"strategy={strategy_name}: {sorted(present)} != {sorted(expected)}"
                    )
    return projected


def load_task_specs(path: str | Path) -> dict[str, TaskSpec]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if int(payload.get("version", 0)) != 20:
        raise ValueError("task-spec config must declare version: 20")
    specs = [TaskSpec.from_mapping(row) for row in payload.get("tasks", [])]
    out = {spec.task_id: spec for spec in specs}
    required = {"benchmark_a", "benchmark_b_pooled", "benchmark_b_covid", "benchmark_b_flu"}
    if len(out) != len(specs) or set(out) != required:
        raise ValueError(f"task-spec config must contain exactly {sorted(required)}")
    return out


def load_task_spec(path: str | Path, task_id: str) -> TaskSpec:
    try:
        return load_task_specs(path)[str(task_id)]
    except KeyError as exc:
        raise KeyError(f"unknown task_id={task_id!r}") from exc
