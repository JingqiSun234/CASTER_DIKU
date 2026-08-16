""















from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Iterable

import numpy as np
import pandas as pd

from .joint_selection import (
    DRAW_KERNEL_T,
    FILTER_VARIANTS,
    JOINT_METRICS,
    MOMENT_T,
    ExactValidationReplay,
    ParameterState,
    ReplayArtifacts,
)
from .likelihood import COHERENT_CENSORED_STUDENT_T, BridgeConfig
from .rho_only import transformed_rho_only_metrics


COHERENT_CENSORED_OBJECTIVE_WEIGHTS = {
    "nll": 0.20,
    "short_rmse": 0.20,
    "long_rmse": 0.20,
    "mae": 0.10,
    "wis": 0.20,
    "coverage_penalty": 0.10,
}
COHERENT_CENSORED_C_U_VALUES = (1.0, 1.25, 1.5, 2.0)
COHERENT_CENSORED_FAMILIES = (MOMENT_T, DRAW_KERNEL_T)
COHERENT_CENSORED_C_U_MODES = ("fixed", "final_sensitivity", "enumerate")
COHERENT_CENSORED_PREDICTIVE_CONTRACTS = (
    COHERENT_CENSORED_STUDENT_T,
    "coherent_mean_preserving_censored_student_t",
)


@dataclass(frozen=True)
class CoherentCensoredSelectionSettings:
    ""

    objective_weights: dict[str, float] = field(
        default_factory=lambda: dict(COHERENT_CENSORED_OBJECTIVE_WEIGHTS)
    )
    rho_bounds: tuple[float, float] = (0.005, 0.50)
    gamma_bounds: tuple[float, float] = (0.25, 4.0)
    scale_multiplier_bounds: tuple[float, float] = (0.5, 2.5)
    c_u_values: tuple[float, ...] = COHERENT_CENSORED_C_U_VALUES
    c_u_mode: str = "final_sensitivity"
    fixed_c_u: float = 1.25
    nu: float = 5.0
    coverage_floor: float | None = 0.87
    coverage_target: float = 0.90
    coverage_tolerance: float = 0.03
    coverage_upper_weight: float = 0.5
    metric_epsilon: float = 1e-8
    robust_scale_floor: float = 1e-3
    robust_relative_floor: float = 0.05
    initial_log_step_fraction: float = 0.50
    log_step_shrink: float = 0.50
    minimum_log_step: float = 0.025
    max_pattern_iterations: int = 12
    final_refinement_iterations: int = 4
    max_evaluations: int = 96
    exact_tie_tolerance: float = 1e-10
    quadrature_order: int = 128
    truncation_zero_mean_epsilon: float = 1e-10

    def validate(self) -> "CoherentCensoredSelectionSettings":
        if set(self.objective_weights) != set(JOINT_METRICS):
            raise ValueError(
                "objective_weights must contain exactly " + ",".join(JOINT_METRICS)
            )
        weights = np.asarray(list(self.objective_weights.values()), dtype=float)
        if not np.isfinite(weights).all() or (weights < 0.0).any():
            raise ValueError("objective weights must be finite and nonnegative")
        if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12):
            raise ValueError("objective weights must sum to one")
        for name, bounds in (
            ("rho", self.rho_bounds),
            ("gamma", self.gamma_bounds),
            ("scale multiplier", self.scale_multiplier_bounds),
        ):
            lower, upper = (float(value) for value in bounds)
            if not (
                math.isfinite(lower)
                and math.isfinite(upper)
                and 0.0 < lower < upper
            ):
                raise ValueError(
                    f"{name} bounds must be finite and satisfy 0 < lower < upper"
                )
        c_values = tuple(float(value) for value in self.c_u_values)
        if (
            not c_values
            or any(not math.isfinite(value) or value < 1.0 for value in c_values)
            or len(set(c_values)) != len(c_values)
        ):
            raise ValueError("c_u_values must be unique finite values >= 1")
        if self.c_u_mode not in COHERENT_CENSORED_C_U_MODES:
            raise ValueError(
                "c_u_mode must be one of "
                + ",".join(COHERENT_CENSORED_C_U_MODES)
            )
        if not any(
            math.isclose(float(self.fixed_c_u), value, abs_tol=1e-12)
            for value in c_values
        ):
            raise ValueError("fixed_c_u must be one of c_u_values")
        if not math.isfinite(float(self.nu)) or float(self.nu) <= 0.0:
            raise ValueError("nu must be finite and positive")
        if self.coverage_floor is not None and not (
            0.0 < float(self.coverage_floor) <= 1.0
        ):
            raise ValueError("coverage_floor must be None or lie in (0,1]")
        positive = (
            self.metric_epsilon,
            self.robust_scale_floor,
            self.robust_relative_floor,
            self.initial_log_step_fraction,
            self.log_step_shrink,
            self.minimum_log_step,
            self.exact_tie_tolerance,
            self.truncation_zero_mean_epsilon,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("search tolerances and numerical floors must be positive")
        if not 0.0 < float(self.log_step_shrink) < 1.0:
            raise ValueError("log_step_shrink must lie in (0,1)")
        if not 0.0 < float(self.initial_log_step_fraction) <= 1.0:
            raise ValueError("initial_log_step_fraction must lie in (0,1]")
        if (
            int(self.max_pattern_iterations) < 1
            or int(self.final_refinement_iterations) < 0
            or int(self.max_evaluations) < 8
        ):
            raise ValueError("invalid pattern-search iteration budget")
        if int(self.quadrature_order) < 32 or int(self.quadrature_order) % 4:
            raise ValueError(
                "quadrature_order must be at least 32 and divisible by 4"
            )
        return self

    def serializable(self) -> dict[str, object]:
        return {
            "objective_weights": {
                key: float(value)
                for key, value in sorted(self.objective_weights.items())
            },
            "rho_bounds": [float(value) for value in self.rho_bounds],
            "gamma_bounds": [float(value) for value in self.gamma_bounds],
            "scale_multiplier_bounds": [
                float(value) for value in self.scale_multiplier_bounds
            ],
            "c_u_values": [float(value) for value in self.c_u_values],
            "c_u_mode": self.c_u_mode,
            "fixed_c_u": float(self.fixed_c_u),
            "nu": float(self.nu),
            "coverage_floor": (
                None
                if self.coverage_floor is None
                else float(self.coverage_floor)
            ),
            "coverage_target": float(self.coverage_target),
            "coverage_tolerance": float(self.coverage_tolerance),
            "coverage_upper_weight": float(self.coverage_upper_weight),
            "metric_epsilon": float(self.metric_epsilon),
            "robust_scale_floor": float(self.robust_scale_floor),
            "robust_relative_floor": float(self.robust_relative_floor),
            "initial_log_step_fraction": float(self.initial_log_step_fraction),
            "log_step_shrink": float(self.log_step_shrink),
            "minimum_log_step": float(self.minimum_log_step),
            "max_pattern_iterations": int(self.max_pattern_iterations),
            "final_refinement_iterations": int(
                self.final_refinement_iterations
            ),
            "max_evaluations": int(self.max_evaluations),
            "exact_tie_tolerance": float(self.exact_tie_tolerance),
            "quadrature_order": int(self.quadrature_order),
            "truncation_zero_mean_epsilon": float(
                self.truncation_zero_mean_epsilon
            ),
        }


@dataclass(frozen=True)
class CoherentCensoredPoint:
    rho: float
    scale_multiplier: float
    gamma: float
    c_u: float

    def serializable(self, *, family: str) -> dict[str, float | None]:
        return {
            "rho": float(self.rho),
            "scale_multiplier": float(self.scale_multiplier),
            "gamma": float(self.gamma) if family == MOMENT_T else None,
            "c_u": float(self.c_u),
        }


@dataclass
class _Evaluation:
    point: CoherentCensoredPoint
    state: ParameterState
    artifacts: ReplayArtifacts | None
    metrics: dict[str, float]
    error: str = ""


@dataclass
class CoherentCensoredSelectionOutcome:
    variant: str
    family: str
    selected_point: CoherentCensoredPoint
    selected_state: ParameterState
    selected_config: BridgeConfig
    selected_metrics: dict[str, float]
    selected_objective: float
    selected_feasible: bool
    reference_metrics: dict[str, float]
    reference_transformed: dict[str, float]
    normalization_scales: dict[str, float]
    trace: pd.DataFrame
    c_u_sensitivity: pd.DataFrame
    component_report: pd.DataFrame
    replay_artifacts: ReplayArtifacts
    settings: CoherentCensoredSelectionSettings


def _geometric_midpoint(bounds: tuple[float, float]) -> float:
    lower, upper = (float(value) for value in bounds)
    return float(math.exp((math.log(lower) + math.log(upper)) / 2.0))


class CoherentCensoredSelector:
    ""

    def __init__(
        self,
        replay: ExactValidationReplay,
        *,
        family: str,
        base_scales: dict[str, float],
        base_upper_raw_by_component: dict[str, float],
        settings: CoherentCensoredSelectionSettings,
        predictive_contract: str = COHERENT_CENSORED_STUDENT_T,
    ) -> None:
        if family not in COHERENT_CENSORED_FAMILIES:
            raise ValueError(f"unknown coherent-censored family {family!r}")
        if predictive_contract not in COHERENT_CENSORED_PREDICTIVE_CONTRACTS:
            raise ValueError(
                "unsupported coherent-censored predictive contract "
                f"{predictive_contract!r}"
            )
        if not base_scales:
            raise ValueError("base_scales must be nonempty")
        if not base_upper_raw_by_component:
            raise ValueError("base_upper_raw_by_component must be nonempty")
        scales = {
            str(key): float(value)
            for key, value in sorted(base_scales.items())
        }
        uppers = {
            str(key): float(value)
            for key, value in sorted(base_upper_raw_by_component.items())
        }
        if set(scales) != set(uppers):
            raise ValueError(
                "base scale and train-frozen upper maps must use identical "
                "component-horizon keys"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in scales.values()):
            raise ValueError("base scales must be finite and positive")
        if any(not math.isfinite(value) or value <= 0.0 for value in uppers.values()):
            raise ValueError("base upper bounds must be finite and positive")
        self.replay = replay
        self.family = str(family)
        self.predictive_contract = str(predictive_contract)
        self.base_scales = scales
        self.base_upper_raw_by_component = uppers
        self.settings = settings.validate()
        self._cache: dict[tuple[str, float, float, float, float], _Evaluation] = {}
        self._events: list[dict[str, object]] = []
        self._evaluation_count = 0
        self._reference_transformed: dict[str, float] = {}
        self._normalization_scales: dict[str, float] = {}

    def _canonical_point(
        self, point: CoherentCensoredPoint
    ) -> CoherentCensoredPoint:
        rho_low, rho_high = self.settings.rho_bounds
        scale_low, scale_high = self.settings.scale_multiplier_bounds
        gamma_low, gamma_high = self.settings.gamma_bounds
        c_u = min(
            self.settings.c_u_values,
            key=lambda value: (abs(float(value) - float(point.c_u)), float(value)),
        )
        return CoherentCensoredPoint(
            rho=round(min(max(float(point.rho), rho_low), rho_high), 13),
            scale_multiplier=round(
                min(max(float(point.scale_multiplier), scale_low), scale_high),
                13,
            ),
            gamma=round(
                min(max(float(point.gamma), gamma_low), gamma_high), 13
            ),
            c_u=float(c_u),
        )

    def _point_key(
        self, point: CoherentCensoredPoint, variant: str
    ) -> tuple[str, float, float, float, float]:
        canonical = self._canonical_point(point)
        gamma = canonical.gamma if self.family == MOMENT_T else 1.0
        return (
            str(variant),
            canonical.rho,
            canonical.scale_multiplier,
            gamma,
            canonical.c_u,
        )

    def state(self, point: CoherentCensoredPoint) -> ParameterState:
        point = self._canonical_point(point)
        scales = {
            key: float(value) * float(point.scale_multiplier)
            for key, value in self.base_scales.items()
        }
        upper = {
            key: float(value) * float(point.c_u)
            for key, value in self.base_upper_raw_by_component.items()
        }
        return ParameterState(
            family=self.family,
            scales=scales,
            gammas=(
                {key: float(point.gamma) for key in scales}
                if self.family == MOMENT_T
                else {}
            ),
            nus={key: float(self.settings.nu) for key in scales},
            rho=float(point.rho),
            distribution="student_t",
            predictive_contract=self.predictive_contract,
            truncation_upper_raw_by_component=upper,
            default_truncation_upper_raw=max(upper.values()),
            truncation_bound_policy=(
                "train_only_max_observed_or_mean_plus_4sd_times_c_u"
            ),
            truncation_quadrature_order=int(self.settings.quadrature_order),
            truncation_zero_mean_epsilon=float(
                self.settings.truncation_zero_mean_epsilon
            ),
                                                                              
                                                                 
            truncation_support_expansion_multiplier=None,
        )

    def _request(
        self,
        point: CoherentCensoredPoint,
        *,
        variant: str,
        stage: str,
        iteration: int = 0,
        coordinate: str = "",
        direction: int = 0,
    ) -> _Evaluation | None:
        canonical = self._canonical_point(point)
        key = self._point_key(canonical, variant)
        cache_hit = key in self._cache
        if not cache_hit:
            if self._evaluation_count >= int(self.settings.max_evaluations):
                return None
            state = self.state(canonical)
            try:
                artifacts = self.replay.evaluate(state, variant=variant)
                metrics = {
                    name: float(value)
                    for name, value in artifacts.metrics.items()
                }
                required = {
                    "nll",
                    "short_rmse",
                    "long_rmse",
                    "mae",
                    "wis",
                    "coverage_90",
                }
                if missing := sorted(required - set(metrics)):
                    raise ValueError(
                        "validation replay omitted metrics " + ",".join(missing)
                    )
                if not all(math.isfinite(metrics[name]) for name in required):
                    raise ValueError(
                        f"validation replay produced non-finite metrics: {metrics}"
                    )
                evaluation = _Evaluation(
                    point=canonical,
                    state=state,
                    artifacts=artifacts,
                    metrics=metrics,
                )
            except Exception as exc:                                          
                evaluation = _Evaluation(
                    point=canonical,
                    state=state,
                    artifacts=None,
                    metrics={},
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._evaluation_count += 1
            self._cache[key] = evaluation
        evaluation = self._cache[key]
        self._events.append(
            {
                "event_id": int(len(self._events) + 1),
                "evaluation_id": int(
                    list(self._cache).index(key) + 1
                ),
                "stage": str(stage),
                "iteration": int(iteration),
                "coordinate": str(coordinate),
                "direction": int(direction),
                "cache_hit": bool(cache_hit),
                "variant": str(variant),
                "bridge_family": self.family,
                **canonical.serializable(family=self.family),
            }
        )
        return evaluation

    def _reference_point(self) -> CoherentCensoredPoint:
        return CoherentCensoredPoint(
            rho=_geometric_midpoint(self.settings.rho_bounds),
            scale_multiplier=1.0,
            gamma=1.0,
            c_u=float(self.settings.fixed_c_u),
        )

    def _coordinate_names(self) -> tuple[str, ...]:
        if self.family == MOMENT_T:
            return ("rho", "gamma", "scale_multiplier")
        return ("rho", "scale_multiplier")

    def _bounds(self, name: str) -> tuple[float, float]:
        if name == "rho":
            return self.settings.rho_bounds
        if name == "gamma":
            return self.settings.gamma_bounds
        if name == "scale_multiplier":
            return self.settings.scale_multiplier_bounds
        raise KeyError(name)

    def _with_coordinate(
        self,
        point: CoherentCensoredPoint,
        name: str,
        value: float,
    ) -> CoherentCensoredPoint:
        values = {
            "rho": point.rho,
            "scale_multiplier": point.scale_multiplier,
            "gamma": point.gamma,
            "c_u": point.c_u,
        }
        values[name] = float(value)
        return CoherentCensoredPoint(**values)

    def _normalization_anchors(
        self, *, variant: str
    ) -> tuple[CoherentCensoredPoint, list[_Evaluation]]:
        reference = self._reference_point()
        points = [reference]
        for name in self._coordinate_names():
            lower, upper = self._bounds(name)
            points.extend(
                [
                    self._with_coordinate(reference, name, lower),
                    self._with_coordinate(reference, name, upper),
                ]
            )
        if self.settings.c_u_mode != "fixed":
            points.extend(
                CoherentCensoredPoint(
                    rho=reference.rho,
                    scale_multiplier=reference.scale_multiplier,
                    gamma=reference.gamma,
                    c_u=float(c_u),
                )
                for c_u in self.settings.c_u_values
            )
        evaluations: list[_Evaluation] = []
        for point in points:
            result = self._request(
                point, variant=variant, stage="normalization_anchor"
            )
            if result is not None and not result.error:
                evaluations.append(result)
        reference_evaluation = self._cache.get(
            self._point_key(reference, variant)
        )
        if reference_evaluation is None or reference_evaluation.error:
            reason = (
                reference_evaluation.error
                if reference_evaluation is not None
                else "evaluation budget exhausted"
            )
            raise RuntimeError(
                "coherent-censored reference configuration is invalid: " + reason
            )
        return reference, evaluations

    def _initialize_normalization(
        self,
        reference: CoherentCensoredPoint,
        anchors: Iterable[_Evaluation],
        *,
        variant: str,
    ) -> None:
        reference_evaluation = self._cache[self._point_key(reference, variant)]
        transformed_reference = transformed_rho_only_metrics(
            reference_evaluation.metrics,
            metric_order=tuple(JOINT_METRICS),
            metric_epsilon=float(self.settings.metric_epsilon),
            coverage_target=float(self.settings.coverage_target),
            coverage_tolerance=float(self.settings.coverage_tolerance),
            coverage_upper_weight=float(self.settings.coverage_upper_weight),
        )
        transformed_rows = [
            transformed_rho_only_metrics(
                evaluation.metrics,
                metric_order=tuple(JOINT_METRICS),
                metric_epsilon=float(self.settings.metric_epsilon),
                coverage_target=float(self.settings.coverage_target),
                coverage_tolerance=float(self.settings.coverage_tolerance),
                coverage_upper_weight=float(self.settings.coverage_upper_weight),
            )
            for evaluation in anchors
            if not evaluation.error
        ]
        scales: dict[str, float] = {}
        for name in JOINT_METRICS:
            values = np.asarray(
                [
                    row[name]
                    for row in transformed_rows
                    if math.isfinite(float(row[name]))
                ],
                dtype=float,
            )
            median = float(np.median(values)) if values.size else 0.0
            mad = (
                float(1.4826 * np.median(np.abs(values - median)))
                if values.size
                else 0.0
            )
            relative_floor = float(self.settings.robust_relative_floor) * max(
                abs(float(transformed_reference[name])),
                float(self.settings.robust_scale_floor),
            )
            scales[name] = float(
                max(
                    mad,
                    relative_floor,
                    float(self.settings.robust_scale_floor),
                )
            )
        self._reference_transformed = transformed_reference
        self._normalization_scales = scales

    def _risk(self, evaluation: _Evaluation) -> float:
        if evaluation.error or not self._normalization_scales:
            return float("inf")
        transformed = transformed_rho_only_metrics(
            evaluation.metrics,
            metric_order=tuple(JOINT_METRICS),
            metric_epsilon=float(self.settings.metric_epsilon),
            coverage_target=float(self.settings.coverage_target),
            coverage_tolerance=float(self.settings.coverage_tolerance),
            coverage_upper_weight=float(self.settings.coverage_upper_weight),
        )
        return float(
            sum(
                float(self.settings.objective_weights[name])
                * (
                    float(transformed[name])
                    - float(self._reference_transformed[name])
                )
                / float(self._normalization_scales[name])
                for name in JOINT_METRICS
            )
        )

    def _coverage_violation(self, evaluation: _Evaluation) -> float:
        if evaluation.error:
            return float("inf")
        if self.settings.coverage_floor is None:
            return 0.0
        return float(
            max(
                0.0,
                float(self.settings.coverage_floor)
                - float(evaluation.metrics["coverage_90"]),
            )
        )

    def _tie_key(self, evaluation: _Evaluation) -> tuple[float, ...]:
        point = evaluation.point
        return (
            float(point.rho),
            float(point.gamma) if self.family == MOMENT_T else 1.0,
            float(point.scale_multiplier),
            float(point.c_u),
        )

    def _better(
        self, candidate: _Evaluation, incumbent: _Evaluation
    ) -> bool:
        candidate_valid = not candidate.error
        incumbent_valid = not incumbent.error
        if candidate_valid != incumbent_valid:
            return candidate_valid
        if not candidate_valid:
            return self._tie_key(candidate) < self._tie_key(incumbent)
        candidate_violation = self._coverage_violation(candidate)
        incumbent_violation = self._coverage_violation(incumbent)
        candidate_feasible = candidate_violation <= self.settings.exact_tie_tolerance
        incumbent_feasible = incumbent_violation <= self.settings.exact_tie_tolerance
        if candidate_feasible != incumbent_feasible:
            return candidate_feasible
        if not candidate_feasible and not math.isclose(
            candidate_violation,
            incumbent_violation,
            abs_tol=self.settings.exact_tie_tolerance,
        ):
            return candidate_violation < incumbent_violation
        candidate_risk = self._risk(candidate)
        incumbent_risk = self._risk(incumbent)
        if not math.isclose(
            candidate_risk,
            incumbent_risk,
            abs_tol=self.settings.exact_tie_tolerance,
        ):
            return candidate_risk < incumbent_risk
        return self._tie_key(candidate) < self._tie_key(incumbent)

    def _best(self, evaluations: Iterable[_Evaluation]) -> _Evaluation:
        iterator = iter(evaluations)
        try:
            best = next(iterator)
        except StopIteration as exc:
            raise RuntimeError("pattern search produced no candidates") from exc
        for candidate in iterator:
            if self._better(candidate, best):
                best = candidate
        return best

    def _pattern_search(
        self,
        start: CoherentCensoredPoint,
        *,
        variant: str,
        stage_prefix: str,
        max_iterations: int,
        initial_step_scale: float = 1.0,
    ) -> _Evaluation:
        current = self._request(
            start, variant=variant, stage=f"{stage_prefix}_start"
        )
        if current is None:
            raise RuntimeError("evaluation budget exhausted before pattern start")
        steps = {
            name: max(
                float(self.settings.minimum_log_step),
                float(self.settings.initial_log_step_fraction)
                * float(initial_step_scale)
                * (
                    math.log(self._bounds(name)[1])
                    - math.log(self._bounds(name)[0])
                ),
            )
            for name in self._coordinate_names()
        }
        for iteration in range(1, int(max_iterations) + 1):
            neighbours: list[_Evaluation] = [current]
            for name in self._coordinate_names():
                current_value = float(getattr(current.point, name))
                for direction in (-1, 1):
                    proposed = self._with_coordinate(
                        current.point,
                        name,
                        math.exp(
                            math.log(current_value)
                            + float(direction) * float(steps[name])
                        ),
                    )
                    evaluation = self._request(
                        proposed,
                        variant=variant,
                        stage=f"{stage_prefix}_pattern",
                        iteration=iteration,
                        coordinate=name,
                        direction=direction,
                    )
                    if evaluation is not None:
                        neighbours.append(evaluation)
            best = self._best(neighbours)
            if self._better(best, current):
                current = best
            else:
                steps = {
                    name: float(value) * float(self.settings.log_step_shrink)
                    for name, value in steps.items()
                }
            if max(steps.values()) <= float(self.settings.minimum_log_step):
                break
            if self._evaluation_count >= int(self.settings.max_evaluations):
                break
        return current

    def _continuous_anchor_start(
        self, *, variant: str, c_u: float
    ) -> CoherentCensoredPoint:
        candidates = [
            evaluation
            for (cached_variant, _, _, _, cached_c_u), evaluation in self._cache.items()
            if cached_variant == variant
            and math.isclose(cached_c_u, float(c_u), abs_tol=1e-12)
            and not evaluation.error
        ]
        return (
            self._best(candidates).point
            if candidates
            else CoherentCensoredPoint(
                rho=_geometric_midpoint(self.settings.rho_bounds),
                scale_multiplier=1.0,
                gamma=1.0,
                c_u=float(c_u),
            )
        )

    def _trace(self, selected: _Evaluation) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for event in self._events:
            point = CoherentCensoredPoint(
                rho=float(event["rho"]),
                scale_multiplier=float(event["scale_multiplier"]),
                gamma=(
                    float(event["gamma"])
                    if event["gamma"] is not None
                    else 1.0
                ),
                c_u=float(event["c_u"]),
            )
            evaluation = self._cache[
                self._point_key(point, str(event["variant"]))
            ]
            transformed: dict[str, float] = {}
            z: dict[str, float] = {}
            if not evaluation.error:
                transformed = transformed_rho_only_metrics(
                    evaluation.metrics,
                    metric_order=tuple(JOINT_METRICS),
                    metric_epsilon=float(self.settings.metric_epsilon),
                    coverage_target=float(self.settings.coverage_target),
                    coverage_tolerance=float(self.settings.coverage_tolerance),
                    coverage_upper_weight=float(
                        self.settings.coverage_upper_weight
                    ),
                )
                z = {
                    name: (
                        float(transformed[name])
                        - float(self._reference_transformed[name])
                    )
                    / float(self._normalization_scales[name])
                    for name in JOINT_METRICS
                }
            rows.append(
                {
                    **event,
                    **{
                        name: evaluation.metrics.get(name, math.nan)
                        for name in (
                            "overall_rmse",
                            "short_rmse",
                            "long_rmse",
                            "mae",
                            "nll",
                            "wis",
                            "coverage_90",
                        )
                    },
                    **{f"z_{name}": z.get(name, math.nan) for name in JOINT_METRICS},
                    "coverage_floor": (
                        None
                        if self.settings.coverage_floor is None
                        else float(self.settings.coverage_floor)
                    ),
                    "coverage_violation": self._coverage_violation(evaluation),
                    "feasible": (
                        not evaluation.error
                        and self._coverage_violation(evaluation)
                        <= self.settings.exact_tie_tolerance
                    ),
                    "joint_risk": self._risk(evaluation),
                    "evaluation_error": evaluation.error,
                    "selected": False,
                }
            )
        trace = pd.DataFrame(rows)
        if not trace.empty:
            mask = (
                np.isclose(trace["rho"], selected.point.rho, atol=1e-12, rtol=0.0)
                & np.isclose(
                    trace["scale_multiplier"],
                    selected.point.scale_multiplier,
                    atol=1e-12,
                    rtol=0.0,
                )
                & np.isclose(trace["c_u"], selected.point.c_u, atol=1e-12, rtol=0.0)
            )
            if self.family == MOMENT_T:
                mask &= np.isclose(
                    pd.to_numeric(trace["gamma"], errors="coerce"),
                    selected.point.gamma,
                    atol=1e-12,
                    rtol=0.0,
                )
            if bool(mask.any()):
                trace.loc[trace.index[mask][-1], "selected"] = True
        return trace

    def _c_u_sensitivity(
        self, selected: _Evaluation, *, variant: str
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for c_u in self.settings.c_u_values:
            point = CoherentCensoredPoint(
                rho=selected.point.rho,
                scale_multiplier=selected.point.scale_multiplier,
                gamma=selected.point.gamma,
                c_u=float(c_u),
            )
            evaluation = self._request(
                point, variant=variant, stage="final_c_u_sensitivity"
            )
            if evaluation is None:
                continue
            rows.append(
                {
                    "variant": variant,
                    "bridge_family": self.family,
                    **evaluation.point.serializable(family=self.family),
                    **{
                        name: evaluation.metrics.get(name, math.nan)
                        for name in (
                            "overall_rmse",
                            "short_rmse",
                            "long_rmse",
                            "mae",
                            "nll",
                            "wis",
                            "coverage_90",
                        )
                    },
                    "coverage_violation": self._coverage_violation(evaluation),
                    "feasible": (
                        not evaluation.error
                        and self._coverage_violation(evaluation)
                        <= self.settings.exact_tie_tolerance
                    ),
                    "joint_risk": self._risk(evaluation),
                    "evaluation_error": evaluation.error,
                    "selected": math.isclose(
                        evaluation.point.c_u,
                        selected.point.c_u,
                        abs_tol=1e-12,
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _component_report(self, selected: _Evaluation) -> pd.DataFrame:
        state = selected.state
        rows = []
        for key in sorted(self.base_scales):
            rows.append(
                {
                    "bridge_r_key": key,
                    "bridge_family": self.family,
                    "base_validation_residual_rmse_scale": float(
                        self.base_scales[key]
                    ),
                    "shared_scale_multiplier": float(
                        selected.point.scale_multiplier
                    ),
                    "selected_sigma": (
                        float(state.scales[key])
                        if self.family == MOMENT_T
                        else math.nan
                    ),
                    "selected_tau": (
                        float(state.scales[key])
                        if self.family == DRAW_KERNEL_T
                        else math.nan
                    ),
                    "shared_gamma": (
                        float(selected.point.gamma)
                        if self.family == MOMENT_T
                        else math.nan
                    ),
                    "nu": float(self.settings.nu),
                    "base_train_only_upper_raw": float(
                        self.base_upper_raw_by_component[key]
                    ),
                    "c_u": float(selected.point.c_u),
                    "selected_upper_raw": float(
                        state.truncation_upper_raw_by_component[key]
                    ),
                    "rho": float(selected.point.rho),
                    "predictive_contract": self.predictive_contract,
                }
            )
        return pd.DataFrame(rows)

    def select(self, *, variant: str) -> CoherentCensoredSelectionOutcome:
        if variant not in FILTER_VARIANTS:
            raise ValueError(f"unknown filtering variant {variant!r}")
                                                                              
                                                              
        self._events = []
        self._reference_transformed = {}
        self._normalization_scales = {}
        reference, anchors = self._normalization_anchors(variant=variant)
        self._initialize_normalization(reference, anchors, variant=variant)

        c_values = (
            (float(self.settings.fixed_c_u),)
            if self.settings.c_u_mode in {"fixed", "final_sensitivity"}
            else tuple(float(value) for value in self.settings.c_u_values)
        )
        optima: list[_Evaluation] = []
        for index, c_u in enumerate(c_values, start=1):
            start = self._continuous_anchor_start(variant=variant, c_u=c_u)
            optima.append(
                self._pattern_search(
                    start,
                    variant=variant,
                    stage_prefix=f"c_u_{c_u:g}_{index}",
                    max_iterations=int(self.settings.max_pattern_iterations),
                )
            )
        selected = self._best(optima)

        if self.settings.c_u_mode == "final_sensitivity":
            sensitivity_evaluations: list[_Evaluation] = []
            for c_u in self.settings.c_u_values:
                evaluation = self._request(
                    CoherentCensoredPoint(
                        rho=selected.point.rho,
                        scale_multiplier=selected.point.scale_multiplier,
                        gamma=selected.point.gamma,
                        c_u=float(c_u),
                    ),
                    variant=variant,
                    stage="c_u_final_selection",
                )
                if evaluation is not None:
                    sensitivity_evaluations.append(evaluation)
            selected = self._best(sensitivity_evaluations)
            if int(self.settings.final_refinement_iterations) > 0:
                selected = self._pattern_search(
                    selected.point,
                    variant=variant,
                    stage_prefix="selected_c_u_refinement",
                    max_iterations=int(
                        self.settings.final_refinement_iterations
                    ),
                    initial_step_scale=float(self.settings.log_step_shrink),
                )

        if selected.error or selected.artifacts is None:
            raise RuntimeError(
                "coherent-censored search found no valid configuration: "
                + selected.error
            )
        sensitivity = self._c_u_sensitivity(selected, variant=variant)
        trace = self._trace(selected)
        reference_evaluation = self._cache[
            self._point_key(reference, variant)
        ]
        return CoherentCensoredSelectionOutcome(
            variant=variant,
            family=self.family,
            selected_point=selected.point,
            selected_state=selected.state,
            selected_config=selected.state.config(transform="log1p"),
            selected_metrics=dict(selected.metrics),
            selected_objective=float(self._risk(selected)),
            selected_feasible=(
                self._coverage_violation(selected)
                <= self.settings.exact_tie_tolerance
            ),
            reference_metrics=dict(reference_evaluation.metrics),
            reference_transformed=dict(self._reference_transformed),
            normalization_scales=dict(self._normalization_scales),
            trace=trace,
            c_u_sensitivity=sensitivity,
            component_report=self._component_report(selected),
            replay_artifacts=selected.artifacts,
            settings=self.settings,
        )


def selector_identity_payload(
    *,
    family: str,
    base_scales: dict[str, float],
    base_upper_raw_by_component: dict[str, float],
    settings: CoherentCensoredSelectionSettings,
    predictive_contract: str = COHERENT_CENSORED_STUDENT_T,
) -> dict[str, object]:
    ""

    return {
        "schema": "caster_coherent_censored_fullval_selector_v1",
        "predictive_contract": str(predictive_contract),
        "family": str(family),
        "base_scales": {
            key: float(value) for key, value in sorted(base_scales.items())
        },
        "base_upper_raw_by_component": {
            key: float(value)
            for key, value in sorted(base_upper_raw_by_component.items())
        },
        "settings": settings.serializable(),
    }


def selector_identity_json(**kwargs: object) -> str:
    return json.dumps(
        selector_identity_payload(**kwargs),
        sort_keys=True,
        separators=(",", ":"),
    )
