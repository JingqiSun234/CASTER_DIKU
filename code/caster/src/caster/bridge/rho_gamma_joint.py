""

















from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .joint_selection import (
    FILTER_VARIANTS,
    JOINT_METRICS,
    MOMENT_T,
    ExactValidationReplay,
    ParameterState,
    ReplayArtifacts,
    transformed_joint_metrics,
)


JOINT_RHO_GAMMA_OBJECTIVE_WEIGHTS = {
    "nll": 0.15,
    "short_rmse": 0.25,
    "long_rmse": 0.25,
    "mae": 0.10,
    "wis": 0.15,
    "coverage_penalty": 0.10,
}
JOINT_RHO_ANCHOR_BANK = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.35,
    0.50,
    0.75,
    1.0,
)
JOINT_GAMMA_ANCHORS = (
    0.125,
    0.25,
    0.50,
    1.0 / math.sqrt(2.0),
    1.0,
    math.sqrt(2.0),
    2.0,
    4.0,
)

                                                                            
                                                                            
                                                                        
                                       
JOINT_CERTIFICATE_MAX_NEIGHBORS_PER_ATTEMPT = 8


@dataclass(frozen=True)
class JointRhoGammaSelectionSettings:
    ""

    objective_weights: dict[str, float] = field(
        default_factory=lambda: dict(JOINT_RHO_GAMMA_OBJECTIVE_WEIGHTS)
    )
    rho_bounds: tuple[float, float] = (0.001, 0.5)
    gamma_bounds: tuple[float, float] = (0.125, 4.0)
    anchor_rhos: tuple[float, ...] = JOINT_RHO_ANCHOR_BANK[:9]
    anchor_gammas: tuple[float, ...] = JOINT_GAMMA_ANCHORS
    reference_rho: float = 0.50
    reference_gamma: float = 1.0
    multi_starts: int = 4
    coordinate_passes: int = 3
    trust_region_passes: int = 2
    finite_difference_log_step: float = 0.10
    minimum_difference_log_step: float = 0.01
    maximum_newton_log_step: float = 0.50
    maximum_trust_log_step: float = 0.35
                                                                      
                                                                    
                                                                     
    max_evaluations: int = 128
    exploration_evaluation_limit: int = 96
    final_polish_evaluation_limit: int = 32
    final_polish_max_passes: int = 4
    final_polish_max_accepted_steps: int = 8
    certificate_max_restarts: int = 8
                                                                        
                                                                            
    certificate_evaluation_reserve: int = 16
                                                                          
                                                                          
                                                                            
                                                                      
                               
    certificate_recovery_max_passes: int = 0
    certificate_recovery_evaluation_reserve: int = 0
    certificate_recovery_evaluation_limit_per_sweep: int = 24
    certificate_log_step: float = 0.01
    max_backtracks: int = 8
    hessian_floor: float = 1e-6
    trust_condition_limit: float = 1e8
    x_tolerance: float = 1e-3
    objective_tolerance: float = 1e-8
    exact_tie_tolerance: float = 1e-10
    metric_epsilon: float = 1e-8
    coverage_target: float = 0.90
    coverage_tolerance: float = 0.03
    coverage_upper_weight: float = 0.5
    robust_scale_floor: float = 1e-3
    robust_relative_floor: float = 0.05

    def validate(self) -> "JointRhoGammaSelectionSettings":
        if set(self.objective_weights) != set(JOINT_METRICS):
            raise ValueError("joint rho-gamma objective has the wrong metric keys")
        weights = np.asarray(list(self.objective_weights.values()), dtype=float)
        if not np.isfinite(weights).all() or (weights < 0.0).any():
            raise ValueError("joint rho-gamma weights must be finite and nonnegative")
        if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("joint rho-gamma weights must sum to one")

        rho_lower, rho_upper = (float(value) for value in self.rho_bounds)
        if not (
            math.isfinite(rho_lower)
            and math.isfinite(rho_upper)
            and 0.0 < rho_lower < rho_upper
        ):
            raise ValueError("rho bounds must be finite, positive, and increasing")
        gamma_lower, gamma_upper = (float(value) for value in self.gamma_bounds)
        if not (
            math.isfinite(gamma_lower)
            and math.isfinite(gamma_upper)
            and 0.0 < gamma_lower < gamma_upper
        ):
            raise ValueError("gamma bounds must be finite, positive, and increasing")
        if not self.anchor_rhos or any(
            not math.isfinite(float(value))
            or not rho_lower <= float(value) <= rho_upper
            for value in self.anchor_rhos
        ):
            raise ValueError("rho anchors must lie inside the search bounds")
        if not self.anchor_gammas or any(
            not math.isfinite(float(value))
            or not gamma_lower <= float(value) <= gamma_upper
            for value in self.anchor_gammas
        ):
            raise ValueError("gamma anchors must lie inside the search bounds")
        if not rho_lower <= float(self.reference_rho) <= rho_upper:
            raise ValueError("reference rho must lie inside the search bounds")
        if not gamma_lower <= float(self.reference_gamma) <= gamma_upper:
            raise ValueError("reference gamma must lie inside the search bounds")
        if (
            self.multi_starts < 1
            or self.coordinate_passes < 1
            or self.trust_region_passes < 0
            or self.max_evaluations < 9
            or self.exploration_evaluation_limit < 9
            or self.final_polish_evaluation_limit < 1
            or self.final_polish_max_passes < 1
            or self.final_polish_max_accepted_steps < 1
            or self.certificate_max_restarts < 1
            or self.certificate_evaluation_reserve < 1
            or self.certificate_recovery_max_passes < 0
            or self.certificate_recovery_evaluation_reserve < 0
            or self.certificate_recovery_evaluation_limit_per_sweep < 1
        ):
            raise ValueError("invalid joint rho-gamma optimizer limits")
        minimum_points = len(self.search_anchor_points())
        if self.max_evaluations < minimum_points:
            raise ValueError("max_evaluations cannot cover preregistered anchor points")
        if self.exploration_evaluation_limit < minimum_points:
            raise ValueError(
                "exploration_evaluation_limit cannot cover preregistered anchor points"
            )
        if self.exploration_evaluation_limit > self.max_evaluations:
            raise ValueError("exploration_evaluation_limit exceeds max_evaluations")
        if (
            self.exploration_evaluation_limit
            + self.final_polish_evaluation_limit
            > self.max_evaluations
        ):
            raise ValueError(
                "exploration and final-polish limits exceed max_evaluations"
            )
        if self.certificate_evaluation_reserve > self.final_polish_evaluation_limit:
            raise ValueError(
                "certificate_evaluation_reserve exceeds the post-exploration limit"
            )
        recovery_enabled = int(self.certificate_recovery_max_passes) > 0
        if recovery_enabled != (
            int(self.certificate_recovery_evaluation_reserve) > 0
        ):
            raise ValueError(
                "certificate recovery passes and reserve must be enabled together"
            )
        if (
            int(self.certificate_evaluation_reserve)
            + int(self.certificate_recovery_evaluation_reserve)
            > int(self.final_polish_evaluation_limit)
        ):
            raise ValueError(
                "certificate and recovery reserves exceed the post-exploration limit"
            )
        if recovery_enabled and int(
            self.certificate_recovery_evaluation_reserve
        ) < int(self.certificate_max_restarts) * int(
            self.certificate_recovery_evaluation_limit_per_sweep
        ):
            raise ValueError(
                "certificate recovery reserve cannot cover every recovery sweep"
            )
        positive = (
            self.finite_difference_log_step,
            self.minimum_difference_log_step,
            self.maximum_newton_log_step,
            self.maximum_trust_log_step,
            self.certificate_log_step,
            self.hessian_floor,
            self.trust_condition_limit,
            self.x_tolerance,
            self.coverage_tolerance,
            self.robust_scale_floor,
            self.robust_relative_floor,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("joint rho-gamma scale settings must be finite and positive")
        nonnegative = (
            self.objective_tolerance,
            self.exact_tie_tolerance,
            self.metric_epsilon,
            self.coverage_upper_weight,
        )
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in nonnegative
        ):
            raise ValueError("joint rho-gamma tolerances must be nonnegative")
        return self

    def search_anchor_points(self) -> tuple[tuple[float, float], ...]:
        ""

        rho_lower, rho_upper = (float(value) for value in self.rho_bounds)
        gamma_lower, gamma_upper = (float(value) for value in self.gamma_bounds)
        points = {
            *( (float(rho), float(self.reference_gamma)) for rho in self.anchor_rhos ),
            *( (float(self.reference_rho), float(gamma)) for gamma in self.anchor_gammas ),
            (rho_lower, gamma_lower),
            (rho_lower, gamma_upper),
            (rho_upper, gamma_lower),
            (rho_upper, gamma_upper),
        }
        return tuple(sorted(points))


@dataclass
class JointRhoGammaNewtonResult:
    rho: float
    gamma: float
    objective: float
    trace: pd.DataFrame
    evaluation_count: int
    fallback_count: int
    trust_region_attempts: int
    trust_region_accepts: int
    exploration_evaluation_limit: int
    exploration_evaluation_count: int
                                                                         
                                                                
    final_polish_evaluation_count: int
    convergence_status: str
    axis_bracketed: dict[str, bool]
    mixed_checked: bool
    last_accepted_step_capped: bool
    verification_improvement_count: int
    final_polish_accepted_steps: int
    certificate_attempts: int
                                                                         
                                                                     
                                                                  
    best_first_polish_evaluation_limit: int = 16
    best_first_polish_evaluation_count: int = 0
    certificate_evaluation_reserve: int = 16
    certificate_evaluation_count: int = 0
    certificate_max_attempts: int = 9
    certificate_max_neighbors_per_attempt: int = (
        JOINT_CERTIFICATE_MAX_NEIGHBORS_PER_ATTEMPT
    )
    certificate_worst_case_evaluations: int = 72
    certificate_evaluation_counts: dict[str, int] = field(default_factory=dict)
    certified_certificate_index: int = 0
    recovery_sweep_evaluation_limit: int = 0
    recovery_sweep_limit: int = 0
    recovery_evaluation_reserve: int = 0
    recovery_evaluation_count: int = 0
    recovery_evaluation_counts: dict[str, int] = field(default_factory=dict)
    recovery_sweep_count: int = 0
    recovery_pass_count: int = 0
    recovery_pass_counts: dict[str, int] = field(default_factory=dict)
    recovery_coordinate_attempts: int = 0
    recovery_accepted_steps: int = 0
    recovery_last_pass_improved: dict[str, bool] = field(default_factory=dict)
    phase_evaluation_counts: dict[str, int] = field(default_factory=dict)


class JointRhoGammaOptimizationFailure(RuntimeError):
    ""

    def __init__(
        self,
        message: str,
        *,
        failure_code: str,
        trace: pd.DataFrame,
        metadata: Mapping[str, object],
    ) -> None:
        super().__init__(message)
        self.failure_code = str(failure_code)
        self.trace = trace.copy()
        self.metadata = dict(metadata)


@dataclass
class JointRhoGammaSelectionOutcome:
    variant: str
    selected_state: ParameterState
    selected_metrics: dict[str, float]
    selected_objective: float
    reference_transformed: dict[str, float]
    normalization_scales: dict[str, float]
    trace: pd.DataFrame
    replay_artifacts: ReplayArtifacts
    optimizer: JointRhoGammaNewtonResult


def _validated_metric_map(
    values: Mapping[str, float],
    *,
    name: str,
    strictly_positive: bool,
) -> dict[str, float]:
    if set(values) != set(JOINT_METRICS):
        raise ValueError(f"{name} must contain exactly {','.join(JOINT_METRICS)}")
    out = {metric: float(values[metric]) for metric in JOINT_METRICS}
    if not all(math.isfinite(value) for value in out.values()):
        raise ValueError(f"{name} values must be finite")
    if strictly_positive and not all(value > 0.0 for value in out.values()):
        raise ValueError(f"{name} values must be strictly positive")
    return out


def safeguarded_bounded_newton_logrho_gamma(
    objective: Callable[[float, float], float],
    *,
    settings: JointRhoGammaSelectionSettings,
) -> JointRhoGammaNewtonResult:
    ""

    settings = settings.validate()
    lower = np.log(np.asarray([settings.rho_bounds[0], settings.gamma_bounds[0]], dtype=float))
    upper = np.log(np.asarray([settings.rho_bounds[1], settings.gamma_bounds[1]], dtype=float))
    reference = np.log(
        np.asarray([settings.reference_rho, settings.reference_gamma], dtype=float)
    )
    cache: dict[tuple[float, float], float] = {}
    rows: list[dict[str, object]] = []
    fallback_count = 0
    trust_attempts = 0
    trust_accepts = 0
    unique_evaluations_by_phase: dict[str, int] = {}

    def point_key(point: np.ndarray) -> tuple[float, float]:
        clipped = np.minimum(np.maximum(np.asarray(point, dtype=float), lower), upper)
        return (round(float(clipped[0]), 13), round(float(clipped[1]), 13))

    def decoded(key: tuple[float, float]) -> tuple[float, float]:
        return float(math.exp(key[0])), float(math.exp(key[1]))

    def has_budget_for(
        *points: np.ndarray,
        evaluation_limit: int | None = None,
    ) -> bool:
        limit = (
            int(settings.max_evaluations)
            if evaluation_limit is None
            else int(evaluation_limit)
        )
        keys = {point_key(point) for point in points}
        return len(cache) + sum(key not in cache for key in keys) <= limit

    def evaluate(point: np.ndarray, stage: str, **details: object) -> float:
        key = point_key(point)
        hit = key in cache
        if not hit:
            if len(cache) >= settings.max_evaluations:
                return float("inf")
            rho, gamma = decoded(key)
            value = float(objective(rho, gamma))
            cache[key] = value if math.isfinite(value) else float("inf")
            phase = str(details.get("optimizer_phase", stage))
            unique_evaluations_by_phase[phase] = (
                int(unique_evaluations_by_phase.get(phase, 0)) + 1
            )
        rho, gamma = decoded(key)
        rows.append(
            {
                "optimizer_event": "evaluation",
                "stage": stage,
                "log_rho": float(key[0]),
                "log_gamma": float(key[1]),
                "rho": rho,
                "gamma": gamma,
                "joint_risk": float(cache[key]),
                "cache_hit": bool(hit),
                **details,
            }
        )
        return float(cache[key])

    def best_item() -> tuple[tuple[float, float], float]:
        if not cache:
            raise RuntimeError("joint rho-gamma optimizer has no finite candidates")
        minimum = min(cache.values())
        tolerance = float(settings.exact_tie_tolerance)
        tied = [item for item in cache.items() if item[1] <= minimum + tolerance]
        return min(
            tied,
            key=lambda item: (
                abs(float(item[0][1]) - float(reference[1])),
                math.exp(float(item[0][0])),
                math.exp(float(item[0][1])),
            ),
        )

    def conditional_gap_midpoint(point: np.ndarray, coordinate: int) -> np.ndarray | None:
        key = point_key(point)
        other = 1 - coordinate
        candidates = [float(lower[coordinate]), float(upper[coordinate])]
        for cached_key in cache:
            if abs(float(cached_key[other]) - float(key[other])) <= 1e-13:
                candidates.append(float(cached_key[coordinate]))
        candidates = sorted(set(candidates))
        gaps = [(right - left, left, right) for left, right in zip(candidates, candidates[1:])]
        if not gaps:
            return None
        gap, left, right = max(gaps, key=lambda item: item[0])
        if gap <= 2.0 * float(settings.x_tolerance):
            return None
        proposal = np.asarray(point, dtype=float).copy()
        proposal[coordinate] = (left + right) / 2.0
        return proposal

                                                                        
                                                            
    for rho, gamma in settings.search_anchor_points():
        point = np.log(np.asarray([rho, gamma], dtype=float))
        evaluate(
            point,
            "optimizer_anchor",
            optimizer_phase="global_exploration",
        )

    geometric = (lower + upper) / 2.0
    best_key, _ = best_item()
    starts = [
        np.asarray(best_key, dtype=float),
        reference.copy(),
        geometric.copy(),
        np.asarray([geometric[0], reference[1]], dtype=float),
        np.asarray([reference[0], geometric[1]], dtype=float),
    ]
    unique_starts: list[np.ndarray] = []
    seen_starts: set[tuple[float, float]] = set()
    for point in starts:
        key = point_key(point)
        if key not in seen_starts:
            seen_starts.add(key)
            unique_starts.append(np.asarray(key, dtype=float))
        if len(unique_starts) >= int(settings.multi_starts):
            break

    def coordinate_step(
        point: np.ndarray,
        current_f: float,
        *,
        coordinate: int,
        start_index: int,
        pass_index: int,
        evaluation_limit: int,
        optimizer_phase: str,
    ) -> tuple[np.ndarray, float, bool]:
        nonlocal fallback_count
        axis = "rho" if coordinate == 0 else "gamma"
        x = float(point[coordinate])
        left_room = x - float(lower[coordinate])
        right_room = float(upper[coordinate]) - x
        h = min(float(settings.finite_difference_log_step), left_room, right_room)
        gradient = float("nan")
        hessian = float("nan")

        if h >= float(settings.minimum_difference_log_step):
            minus = point.copy()
            plus = point.copy()
            minus[coordinate] -= h
            plus[coordinate] += h
            if has_budget_for(
                minus,
                plus,
                evaluation_limit=evaluation_limit,
            ):
                minus_f = evaluate(
                    minus,
                    "finite_difference_minus",
                    coordinate=axis,
                    start_index=start_index,
                    pass_index=pass_index,
                    optimizer_phase=optimizer_phase,
                )
                plus_f = evaluate(
                    plus,
                    "finite_difference_plus",
                    coordinate=axis,
                    start_index=start_index,
                    pass_index=pass_index,
                    optimizer_phase=optimizer_phase,
                )
                gradient = (plus_f - minus_f) / (2.0 * h)
                hessian = (plus_f - 2.0 * current_f + minus_f) / (h * h)
        else:
                                                                              
                                                                             
            direction = 1.0 if right_room >= left_room else -1.0
            available = right_room if direction > 0.0 else left_room
            h = min(float(settings.finite_difference_log_step), available / 2.0)
            if h >= float(settings.minimum_difference_log_step):
                first = point.copy()
                second = point.copy()
                first[coordinate] += direction * h
                second[coordinate] += direction * 2.0 * h
                if has_budget_for(
                    first,
                    second,
                    evaluation_limit=evaluation_limit,
                ):
                    first_f = evaluate(
                        first,
                        "finite_difference_one_sided_1",
                        coordinate=axis,
                        start_index=start_index,
                        pass_index=pass_index,
                        optimizer_phase=optimizer_phase,
                    )
                    second_f = evaluate(
                        second,
                        "finite_difference_one_sided_2",
                        coordinate=axis,
                        start_index=start_index,
                        pass_index=pass_index,
                        optimizer_phase=optimizer_phase,
                    )
                    if direction > 0.0:
                        gradient = (-3.0 * current_f + 4.0 * first_f - second_f) / (2.0 * h)
                    else:
                        gradient = (3.0 * current_f - 4.0 * first_f + second_f) / (2.0 * h)
                    hessian = (current_f - 2.0 * first_f + second_f) / (h * h)

        use_newton = (
            math.isfinite(gradient)
            and math.isfinite(hessian)
            and hessian > float(settings.hessian_floor)
        )
        if use_newton:
            step = float(
                np.clip(
                    -gradient / hessian,
                    -float(settings.maximum_newton_log_step),
                    float(settings.maximum_newton_log_step),
                )
            )
            use_newton = gradient * step < 0.0
        if not use_newton:
            fallback_count += 1
            if math.isfinite(gradient) and abs(gradient) > 0.0:
                step = -math.copysign(float(settings.maximum_newton_log_step), gradient)
            else:
                midpoint = conditional_gap_midpoint(point, coordinate)
                if midpoint is None or not has_budget_for(
                    midpoint,
                    evaluation_limit=evaluation_limit,
                ):
                    return point, current_f, False
                midpoint_f = evaluate(
                    midpoint,
                    "coordinate_gap_fallback",
                    coordinate=axis,
                    start_index=start_index,
                    pass_index=pass_index,
                    gradient=gradient,
                    hessian=hessian,
                    optimizer_phase=optimizer_phase,
                )
                improved = midpoint_f < current_f - float(settings.objective_tolerance)
                return (midpoint, midpoint_f, True) if improved else (point, current_f, False)

        for backtrack in range(int(settings.max_backtracks) + 1):
            raw = point.copy()
            raw[coordinate] += step * (0.5**backtrack)
            proposed = np.minimum(np.maximum(raw, lower), upper)
            if abs(float(proposed[coordinate]) - x) <= float(settings.x_tolerance):
                continue
            if not has_budget_for(
                proposed,
                evaluation_limit=evaluation_limit,
            ):
                break
            proposed_f = evaluate(
                proposed,
                "coordinate_newton_trial" if use_newton else "coordinate_fallback_trial",
                coordinate=axis,
                start_index=start_index,
                pass_index=pass_index,
                gradient=gradient,
                hessian=hessian,
                backtrack=backtrack,
                projected=bool(np.any(np.abs(raw - proposed) > 0.0)),
                optimizer_phase=optimizer_phase,
            )
            if proposed_f < current_f - float(settings.objective_tolerance):
                return proposed, proposed_f, True

        midpoint = conditional_gap_midpoint(point, coordinate)
        if midpoint is not None and has_budget_for(
            midpoint,
            evaluation_limit=evaluation_limit,
        ):
            fallback_count += 1
            midpoint_f = evaluate(
                midpoint,
                "coordinate_gap_fallback",
                coordinate=axis,
                start_index=start_index,
                pass_index=pass_index,
                gradient=gradient,
                hessian=hessian,
                optimizer_phase=optimizer_phase,
            )
            if midpoint_f < current_f - float(settings.objective_tolerance):
                return midpoint, midpoint_f, True
        return point, current_f, False

    exploration_limit = int(settings.exploration_evaluation_limit)
    for start_index, start in enumerate(unique_starts, start=1):
        if len(cache) >= exploration_limit:
            break
        current = start.copy()
        if not has_budget_for(current, evaluation_limit=exploration_limit):
            break
        current_f = evaluate(
            current,
            f"alternating_start_{start_index}",
            optimizer_phase="global_exploration",
        )
        for pass_index in range(1, int(settings.coordinate_passes) + 1):
            improved_in_pass = False
            order = (0, 1) if pass_index % 2 else (1, 0)
            for coordinate in order:
                if len(cache) >= exploration_limit:
                    break
                current, current_f, improved = coordinate_step(
                    current,
                    current_f,
                    coordinate=coordinate,
                    start_index=start_index,
                    pass_index=pass_index,
                    evaluation_limit=exploration_limit,
                    optimizer_phase="global_exploration",
                )
                improved_in_pass = improved_in_pass or improved
            if not improved_in_pass:
                break

                                                                         
                                                                          
    for trust_index in range(1, int(settings.trust_region_passes) + 1):
        if len(cache) >= exploration_limit:
            break
        best_key, best_f = best_item()
        center = np.asarray(best_key, dtype=float)
        room = np.minimum(center - lower, upper - center)
        h = min(float(settings.finite_difference_log_step), float(room.min()))
        if h < float(settings.minimum_difference_log_step):
            break
        offsets = {
            "rho_minus": np.asarray([-h, 0.0]),
            "rho_plus": np.asarray([h, 0.0]),
            "gamma_minus": np.asarray([0.0, -h]),
            "gamma_plus": np.asarray([0.0, h]),
            "mm": np.asarray([-h, -h]),
            "mp": np.asarray([-h, h]),
            "pm": np.asarray([h, -h]),
            "pp": np.asarray([h, h]),
        }
        points = [center + offset for offset in offsets.values()]
        if not has_budget_for(*points, evaluation_limit=exploration_limit):
            break
        trust_attempts += 1
        values = {
            name: evaluate(
                center + offset,
                "trust_finite_difference",
                trust_index=trust_index,
                stencil=name,
                optimizer_phase="global_exploration",
            )
            for name, offset in offsets.items()
        }
        gradient = np.asarray(
            [
                (values["rho_plus"] - values["rho_minus"]) / (2.0 * h),
                (values["gamma_plus"] - values["gamma_minus"]) / (2.0 * h),
            ],
            dtype=float,
        )
        hessian = np.asarray(
            [
                [
                    (values["rho_plus"] - 2.0 * best_f + values["rho_minus"]) / (h * h),
                    (values["pp"] - values["pm"] - values["mp"] + values["mm"]) / (4.0 * h * h),
                ],
                [
                    (values["pp"] - values["pm"] - values["mp"] + values["mm"]) / (4.0 * h * h),
                    (values["gamma_plus"] - 2.0 * best_f + values["gamma_minus"]) / (h * h),
                ],
            ],
            dtype=float,
        )
        eigenvalues = np.linalg.eigvalsh(hessian) if np.isfinite(hessian).all() else np.asarray([np.nan, np.nan])
        condition = float(np.linalg.cond(hessian)) if np.isfinite(hessian).all() else float("inf")
        usable = bool(
            np.isfinite(gradient).all()
            and np.isfinite(eigenvalues).all()
            and float(eigenvalues.min()) > float(settings.hessian_floor)
            and math.isfinite(condition)
            and condition <= float(settings.trust_condition_limit)
        )
        rows.append(
            {
                "optimizer_event": "trust_diagnostic",
                "stage": "trust_hessian",
                "log_rho": float(center[0]),
                "log_gamma": float(center[1]),
                "rho": float(math.exp(center[0])),
                "gamma": float(math.exp(center[1])),
                "joint_risk": float(best_f),
                "cache_hit": True,
                "trust_index": trust_index,
                "gradient_log_rho": float(gradient[0]),
                "gradient_log_gamma": float(gradient[1]),
                "hessian_log_rho_rho": float(hessian[0, 0]),
                "hessian_log_rho_gamma": float(hessian[0, 1]),
                "hessian_log_gamma_gamma": float(hessian[1, 1]),
                "hessian_min_eigenvalue": float(eigenvalues.min()),
                "hessian_condition_number": condition,
                "trust_hessian_usable": usable,
                "optimizer_phase": "global_exploration",
            }
        )
        if not usable:
            fallback_count += 1
            break
        step = -np.linalg.solve(hessian, gradient)
        infinity_norm = float(np.max(np.abs(step)))
        if infinity_norm > float(settings.maximum_trust_log_step):
            step *= float(settings.maximum_trust_log_step) / infinity_norm
        accepted = False
        for backtrack in range(int(settings.max_backtracks) + 1):
            raw = center + step * (0.5**backtrack)
            proposed = np.minimum(np.maximum(raw, lower), upper)
            if float(np.max(np.abs(proposed - center))) <= float(settings.x_tolerance):
                continue
            if not has_budget_for(
                proposed,
                evaluation_limit=exploration_limit,
            ):
                break
            proposed_f = evaluate(
                proposed,
                "trust_newton_trial",
                trust_index=trust_index,
                backtrack=backtrack,
                hessian_condition_number=condition,
                projected=bool(np.any(np.abs(raw - proposed) > 0.0)),
                optimizer_phase="global_exploration",
            )
            if proposed_f < best_f - float(settings.objective_tolerance):
                accepted = True
                trust_accepts += 1
                break
        if not accepted:
            break

                                                                        
                                                                           
                                                                       
                                           
    exploration_evaluation_count = int(len(cache))
    final_stage_limit = min(
        int(settings.max_evaluations),
        exploration_evaluation_count + int(settings.final_polish_evaluation_limit),
    )
                                                                           
                                                                            
                                                                      
    certificate_reserve = min(
        int(settings.certificate_evaluation_reserve),
        int(settings.final_polish_evaluation_limit),
    )
    recovery_reserve = min(
        int(settings.certificate_recovery_evaluation_reserve),
        int(settings.final_polish_evaluation_limit) - certificate_reserve,
    )
    polish_evaluation_limit = max(
        exploration_evaluation_count,
        final_stage_limit - certificate_reserve - recovery_reserve,
    )
    final_polish_accepted_steps = 0
    last_accepted_step_capped = False

    for polish_pass in range(1, int(settings.final_polish_max_passes) + 1):
        pass_best_before, _ = best_item()
        improved_in_pass = False
        order = (0, 1) if polish_pass % 2 else (1, 0)
        for coordinate in order:
            if len(cache) >= polish_evaluation_limit:
                break
            current_key, current_f = best_item()
            current = np.asarray(current_key, dtype=float)
                                                                            
                                                                            
            coordinate_limit = min(polish_evaluation_limit, len(cache) + 4)
            coordinate_step(
                current,
                current_f,
                coordinate=coordinate,
                start_index=0,
                pass_index=polish_pass,
                evaluation_limit=coordinate_limit,
                optimizer_phase="best_first_polish",
            )
            next_key, _ = best_item()
            if next_key != current_key:
                improved_in_pass = True
                final_polish_accepted_steps += 1
                if (
                    final_polish_accepted_steps
                    >= int(settings.final_polish_max_accepted_steps)
                ):
                    last_accepted_step_capped = True
                    break
        if last_accepted_step_capped:
            break
        pass_best_after, _ = best_item()
        if not improved_in_pass or pass_best_after == pass_best_before:
            break
        if polish_pass == int(settings.final_polish_max_passes):
            last_accepted_step_capped = True

    if last_accepted_step_capped:
        if int(settings.certificate_recovery_max_passes) > 0:
            current_key, current_f = best_item()
            rows.append(
                {
                    "optimizer_event": "failure_diagnostic",
                    "stage": "best_first_polish_failure",
                    "optimizer_phase": "best_first_polish_failure",
                    "log_rho": float(current_key[0]),
                    "log_gamma": float(current_key[1]),
                    "rho": float(math.exp(current_key[0])),
                    "gamma": float(math.exp(current_key[1])),
                    "joint_risk": float(current_f),
                    "cache_hit": True,
                    "failure_code": "initial_polish_accepted_step_cap",
                    "failure_message": (
                        "joint rho-gamma v2.2 initial polish ended on a capped "
                        "accepted step"
                    ),
                }
            )
            failure_trace = pd.DataFrame(rows)
            failure_trace["selected"] = False
            raise JointRhoGammaOptimizationFailure(
                "joint rho-gamma v2.2 initial polish ended on a capped accepted step",
                failure_code="initial_polish_accepted_step_cap",
                trace=failure_trace,
                metadata={
                    "evaluation_count": int(len(cache)),
                    "exploration_evaluation_count": int(
                        exploration_evaluation_count
                    ),
                    "best_first_polish_evaluation_count": int(
                        len(cache) - exploration_evaluation_count
                    ),
                    "phase_evaluation_counts": dict(
                        unique_evaluations_by_phase
                    ),
                    "max_evaluations": int(settings.max_evaluations),
                },
            )
        raise RuntimeError(
            "joint rho-gamma final polish ended on a capped accepted step; "
            "refusing an uncertified selection"
        )

    certificate_start_count = int(len(cache))

    axis_names = ("rho", "gamma")

    def certificate_axis(
        center: np.ndarray,
        coordinate: int,
    ) -> tuple[list[tuple[np.ndarray, str]], list[np.ndarray], bool]:
        ""

        x = float(center[coordinate])
        h = float(settings.certificate_log_step)
        boundary_tolerance = 2e-13
        at_lower = x <= float(lower[coordinate]) + boundary_tolerance
        at_upper = x >= float(upper[coordinate]) - boundary_tolerance
        axial: list[tuple[np.ndarray, str]] = []
        primary: list[np.ndarray] = []

        def add(value: float, label: str, *, mixed_primary: bool) -> None:
            point = center.copy()
            point[coordinate] = float(
                min(max(value, float(lower[coordinate])), float(upper[coordinate]))
            )
            if point_key(point) == point_key(center):
                return
            if point_key(point) not in {point_key(existing) for existing, _ in axial}:
                axial.append((point, label))
            if mixed_primary and point_key(point) not in {
                point_key(existing) for existing in primary
            }:
                primary.append(point)

        if at_lower:
            add(x + h, "one_sided_inward_1", mixed_primary=True)
            add(x + 2.0 * h, "one_sided_inward_2", mixed_primary=False)
            bracketed = len(axial) >= 2 and all(
                float(point[coordinate]) > x + boundary_tolerance for point, _ in axial
            )
        elif at_upper:
            add(x - h, "one_sided_inward_1", mixed_primary=True)
            add(x - 2.0 * h, "one_sided_inward_2", mixed_primary=False)
            bracketed = len(axial) >= 2 and all(
                float(point[coordinate]) < x - boundary_tolerance for point, _ in axial
            )
        else:
            add(x - h, "two_sided_minus", mixed_primary=True)
            add(x + h, "two_sided_plus", mixed_primary=True)
            bracketed = bool(
                any(
                    float(point[coordinate]) < x - boundary_tolerance
                    for point, _ in axial
                )
                and any(
                    float(point[coordinate]) > x + boundary_tolerance
                    for point, _ in axial
                )
            )
        return axial, primary, bracketed

    convergence_status = ""
    axis_bracketed = {"rho": False, "gamma": False}
    mixed_checked = False
    verification_improvement_count = 0
    certificate_attempts = 0
    certificate_seen_centers: set[tuple[float, float]] = set()
    certificate_evaluation_counts: dict[str, int] = {}
    certified_certificate_index = 0
    recovery_evaluation_counts: dict[str, int] = {}
    recovery_pass_counts: dict[str, int] = {}
    recovery_last_pass_improved: dict[str, bool] = {}
    recovery_sweep_count = 0
    recovery_pass_count = 0
    recovery_coordinate_attempts = 0
    recovery_accepted_steps = 0
    recovery_enabled = bool(
        int(settings.certificate_recovery_max_passes) > 0
        and int(recovery_reserve) > 0
    )
    certificate_attempt_evaluation_limit = (
        JOINT_CERTIFICATE_MAX_NEIGHBORS_PER_ATTEMPT
    )
    recovery_sweep_evaluation_limit = int(
        settings.certificate_recovery_evaluation_limit_per_sweep
    )

    def raise_v2_2_failure(
        failure_code: str,
        message: str,
        **details: object,
    ) -> None:
        ""

        current_key, current_f = best_item()
        rows.append(
            {
                "optimizer_event": "failure_diagnostic",
                "stage": "certificate_recovery_failure",
                "optimizer_phase": "certificate_recovery_failure",
                "log_rho": float(current_key[0]),
                "log_gamma": float(current_key[1]),
                "rho": float(math.exp(current_key[0])),
                "gamma": float(math.exp(current_key[1])),
                "joint_risk": float(current_f),
                "cache_hit": True,
                "failure_code": str(failure_code),
                "failure_message": str(message),
                "evaluation_count": int(len(cache)),
                "certificate_attempts": int(certificate_attempts),
                "verification_improvement_count": int(
                    verification_improvement_count
                ),
                "recovery_sweep_count": int(recovery_sweep_count),
                **details,
            }
        )
        failure_trace = pd.DataFrame(rows)
        failure_trace["selected"] = False
        raise JointRhoGammaOptimizationFailure(
            message,
            failure_code=failure_code,
            trace=failure_trace,
            metadata={
                "evaluation_count": int(len(cache)),
                "exploration_evaluation_count": int(
                    exploration_evaluation_count
                ),
                "best_first_polish_evaluation_count": int(
                    certificate_start_count - exploration_evaluation_count
                ),
                "certificate_attempts": int(certificate_attempts),
                "certificate_evaluation_counts": dict(
                    certificate_evaluation_counts
                ),
                "verification_improvement_count": int(
                    verification_improvement_count
                ),
                "recovery_sweep_count": int(recovery_sweep_count),
                "recovery_evaluation_counts": dict(recovery_evaluation_counts),
                "recovery_pass_counts": dict(recovery_pass_counts),
                "recovery_coordinate_attempts": int(
                    recovery_coordinate_attempts
                ),
                "recovery_accepted_steps": int(recovery_accepted_steps),
                "phase_evaluation_counts": dict(unique_evaluations_by_phase),
                "max_evaluations": int(settings.max_evaluations),
                "final_stage_evaluation_limit": int(
                    settings.final_polish_evaluation_limit
                ),
                "certificate_attempt_evaluation_limit": int(
                    certificate_attempt_evaluation_limit
                ),
                "recovery_sweep_evaluation_limit": int(
                    recovery_sweep_evaluation_limit
                ),
            },
        )

    def v2_2_recovery_sweep(
        recovery_index: int,
        *,
        evaluation_limit: int,
    ) -> None:
        ""

        nonlocal recovery_sweep_count
        nonlocal recovery_pass_count
        nonlocal recovery_coordinate_attempts
        nonlocal recovery_accepted_steps
        start_count = int(len(cache))
        start_key, start_f = best_item()
        completed_passes = 0
        last_pass_improved = False
        for local_pass in range(
            1, int(settings.certificate_recovery_max_passes) + 1
        ):
            completed_passes += 1
            recovery_pass_count += 1
            global_pass = (
                (int(recovery_index) - 1)
                * int(settings.certificate_recovery_max_passes)
                + int(local_pass)
            )
            order = (0, 1) if global_pass % 2 else (1, 0)
            improved_in_pass = False
            for coordinate in order:
                recovery_coordinate_attempts += 1
                current_key, current_f = best_item()
                local_limit = min(int(evaluation_limit), int(len(cache) + 4))
                _, _, improved = coordinate_step(
                    np.asarray(current_key, dtype=float),
                    current_f,
                    coordinate=coordinate,
                    start_index=0,
                    pass_index=global_pass,
                    evaluation_limit=local_limit,
                    optimizer_phase="certificate_recovery",
                )
                if improved:
                    recovery_accepted_steps += 1
                    improved_in_pass = True
            last_pass_improved = bool(improved_in_pass)
            if not improved_in_pass:
                break

        end_key, end_f = best_item()
        evaluation_count = int(len(cache) - start_count)
        if evaluation_count > recovery_sweep_evaluation_limit:
            raise_v2_2_failure(
                "recovery_sweep_budget_exceeded",
                "joint rho-gamma v2.2 recovery exceeded its preregistered reserve",
                recovery_index=int(recovery_index),
                recovery_evaluation_count=evaluation_count,
            )
        recovery_sweep_count += 1
        recovery_evaluation_counts[str(recovery_index)] = evaluation_count
        recovery_pass_counts[str(recovery_index)] = int(completed_passes)
        recovery_last_pass_improved[str(recovery_index)] = bool(
            last_pass_improved
        )
        rows.append(
            {
                "optimizer_event": "recovery_diagnostic",
                "stage": "certificate_recovery",
                "optimizer_phase": "certificate_recovery",
                "log_rho": float(end_key[0]),
                "log_gamma": float(end_key[1]),
                "rho": float(math.exp(end_key[0])),
                "gamma": float(math.exp(end_key[1])),
                "joint_risk": float(end_f),
                "cache_hit": True,
                "recovery_index": int(recovery_index),
                "recovery_start_log_rho": float(start_key[0]),
                "recovery_start_log_gamma": float(start_key[1]),
                "recovery_start_risk": float(start_f),
                "recovery_evaluation_count": evaluation_count,
                "recovery_completed_passes": int(completed_passes),
                "recovery_coordinate_attempts": int(2 * completed_passes),
                "recovery_last_pass_improved": bool(last_pass_improved),
                "recovery_complete": True,
            }
        )

    if recovery_enabled:
        initial_polish_limit = int(
            polish_evaluation_limit - exploration_evaluation_count
        )
        expected_final_reserve = int(
            initial_polish_limit
            + (int(settings.certificate_max_restarts) + 1)
            * certificate_attempt_evaluation_limit
            + int(settings.certificate_max_restarts)
            * recovery_sweep_evaluation_limit
        )
        if expected_final_reserve != int(settings.final_polish_evaluation_limit):
            raise_v2_2_failure(
                "invalid_v2_2_budget_decomposition",
                "joint rho-gamma v2.2 final reserve decomposition is inconsistent",
                expected_final_reserve=expected_final_reserve,
            )

        for certificate_index in range(
            1, int(settings.certificate_max_restarts) + 2
        ):
            certificate_stage_limit = int(
                exploration_evaluation_count
                + initial_polish_limit
                + certificate_index * certificate_attempt_evaluation_limit
                + (certificate_index - 1) * recovery_sweep_evaluation_limit
            )
            if certificate_stage_limit > final_stage_limit:
                raise_v2_2_failure(
                    "certificate_allocation_exceeds_final_reserve",
                    "joint rho-gamma v2.2 certificate allocation exceeds final reserve",
                    certificate_index=int(certificate_index),
                )
            certificate_attempt_start = int(len(cache))
            center_key, center_f = best_item()
            if not math.isfinite(float(center_f)):
                raise_v2_2_failure(
                    "nonfinite_certificate_center",
                    "joint rho-gamma v2.2 certificate center has non-finite risk",
                    certificate_index=int(certificate_index),
                )
            if center_key in certificate_seen_centers:
                raise_v2_2_failure(
                    "revisited_certificate_center",
                    "joint rho-gamma v2.2 certificate revisited a center",
                    certificate_index=int(certificate_index),
                )
            certificate_seen_centers.add(center_key)
            center = np.asarray(center_key, dtype=float)

            axial_by_coordinate: list[list[tuple[np.ndarray, str]]] = []
            primary_by_coordinate: list[list[np.ndarray]] = []
            for coordinate, axis in enumerate(axis_names):
                axial, primary, bracketed = certificate_axis(center, coordinate)
                axial_by_coordinate.append(axial)
                primary_by_coordinate.append(primary)
                axis_bracketed[axis] = bool(bracketed)
            if not all(axis_bracketed.values()):
                raise_v2_2_failure(
                    "unbracketed_certificate_axis",
                    "joint rho-gamma v2.2 certificate could not bracket both axes",
                    certificate_index=int(certificate_index),
                )

            mixed_points: list[np.ndarray] = []
            for rho_direction in primary_by_coordinate[0]:
                for gamma_direction in primary_by_coordinate[1]:
                    point = center.copy()
                    point[0] = rho_direction[0]
                    point[1] = gamma_direction[1]
                    if point_key(point) != center_key and point_key(point) not in {
                        point_key(existing) for existing in mixed_points
                    }:
                        mixed_points.append(point)
            mixed_checked = bool(mixed_points)
            if not mixed_checked:
                raise_v2_2_failure(
                    "missing_mixed_certificate_corner",
                    "joint rho-gamma v2.2 certificate has no feasible mixed corner",
                    certificate_index=int(certificate_index),
                )

            required_points = [
                point
                for axial in axial_by_coordinate
                for point, _ in axial
            ] + mixed_points
            unique_required_count = len(
                {point_key(point) for point in required_points}
            )
            if unique_required_count > certificate_attempt_evaluation_limit:
                raise_v2_2_failure(
                    "certificate_neighbor_bound_exceeded",
                    "joint rho-gamma v2.2 certificate exceeded eight neighbors",
                    certificate_index=int(certificate_index),
                    certificate_neighbor_count=unique_required_count,
                )
            if not has_budget_for(
                *required_points,
                evaluation_limit=certificate_stage_limit,
            ):
                raise_v2_2_failure(
                    "incomplete_certificate_budget",
                    "joint rho-gamma v2.2 budget cannot complete its certificate",
                    certificate_index=int(certificate_index),
                )

            certificate_attempts += 1
            certificate_values: list[float] = []
            for coordinate, axis in enumerate(axis_names):
                for point, stencil in axial_by_coordinate[coordinate]:
                    certificate_values.append(
                        evaluate(
                            point,
                            "certificate_axial",
                            optimizer_phase="local_discrete_certificate",
                            certificate_index=certificate_index,
                            certificate_log_step=float(
                                settings.certificate_log_step
                            ),
                            coordinate=axis,
                            stencil=stencil,
                        )
                    )
            for mixed_index, point in enumerate(mixed_points, start=1):
                certificate_values.append(
                    evaluate(
                        point,
                        "certificate_mixed_corner",
                        optimizer_phase="local_discrete_certificate",
                        certificate_index=certificate_index,
                        certificate_log_step=float(
                            settings.certificate_log_step
                        ),
                        stencil=f"mixed_{mixed_index}",
                    )
                )
            if not all(math.isfinite(value) for value in certificate_values):
                raise_v2_2_failure(
                    "nonfinite_certificate_neighbor",
                    "joint rho-gamma v2.2 certificate contains non-finite risk",
                    certificate_index=int(certificate_index),
                )
            certificate_evaluation_count = int(
                len(cache) - certificate_attempt_start
            )
            if certificate_evaluation_count > certificate_attempt_evaluation_limit:
                raise_v2_2_failure(
                    "certificate_attempt_budget_exceeded",
                    "joint rho-gamma v2.2 certificate exceeded its per-attempt reserve",
                    certificate_index=int(certificate_index),
                    certificate_evaluation_count=certificate_evaluation_count,
                )
            certificate_evaluation_counts[str(certificate_index)] = (
                certificate_evaluation_count
            )

            verified_key, verified_f = best_item()
            verified = verified_key == center_key
            rows.append(
                {
                    "optimizer_event": "certificate_diagnostic",
                    "stage": "local_discrete_certificate",
                    "optimizer_phase": "local_discrete_certificate",
                    "log_rho": float(center[0]),
                    "log_gamma": float(center[1]),
                    "rho": float(math.exp(center[0])),
                    "gamma": float(math.exp(center[1])),
                    "joint_risk": float(center_f),
                    "cache_hit": True,
                    "certificate_index": certificate_index,
                    "certificate_log_step": float(settings.certificate_log_step),
                    "axis_rho_bracketed": bool(axis_bracketed["rho"]),
                    "axis_gamma_bracketed": bool(axis_bracketed["gamma"]),
                    "mixed_checked": bool(mixed_checked),
                    "certificate_neighbor_count": unique_required_count,
                    "certificate_evaluation_count": (
                        certificate_evaluation_count
                    ),
                    "certificate_winner_log_rho": float(verified_key[0]),
                    "certificate_winner_log_gamma": float(verified_key[1]),
                    "certificate_winner_risk": float(verified_f),
                    "certificate_verified": bool(verified),
                }
            )
            if verified:
                convergence_status = "verified_local_discrete"
                certified_certificate_index = int(certificate_index)
                break

            verification_improvement_count += 1
            if certificate_index > int(settings.certificate_max_restarts):
                raise_v2_2_failure(
                    "certificate_improved_after_second_recovery",
                    "joint rho-gamma v2.2 third certificate still found an improvement",
                    certificate_index=int(certificate_index),
                )
            recovery_stage_limit = int(
                exploration_evaluation_count
                + initial_polish_limit
                + certificate_index * certificate_attempt_evaluation_limit
                + certificate_index * recovery_sweep_evaluation_limit
            )
            if recovery_stage_limit > final_stage_limit:
                raise_v2_2_failure(
                    "recovery_allocation_exceeds_final_reserve",
                    "joint rho-gamma v2.2 recovery allocation exceeds final reserve",
                    recovery_index=int(certificate_index),
                )
            v2_2_recovery_sweep(
                certificate_index,
                evaluation_limit=recovery_stage_limit,
            )

    alternate_certificate_indices = (
        range(1, int(settings.certificate_max_restarts) + 2)
        if not recovery_enabled
        else ()
    )

    for certificate_index in alternate_certificate_indices:
        center_key, center_f = best_item()
        if not math.isfinite(float(center_f)):
            raise RuntimeError(
                "joint rho-gamma certificate center has non-finite risk"
            )
        if center_key in certificate_seen_centers:
            raise RuntimeError(
                "joint rho-gamma certificate revisited a center; refusing "
                "an uncertified selection"
            )
        certificate_seen_centers.add(center_key)
        center = np.asarray(center_key, dtype=float)

        axial_by_coordinate: list[list[tuple[np.ndarray, str]]] = []
        primary_by_coordinate: list[list[np.ndarray]] = []
        for coordinate, axis in enumerate(axis_names):
            axial, primary, bracketed = certificate_axis(center, coordinate)
            axial_by_coordinate.append(axial)
            primary_by_coordinate.append(primary)
            axis_bracketed[axis] = bool(bracketed)
        if not all(axis_bracketed.values()):
            raise RuntimeError(
                "joint rho-gamma local certificate could not bracket both axes"
            )

        mixed_points: list[np.ndarray] = []
        for rho_direction in primary_by_coordinate[0]:
            for gamma_direction in primary_by_coordinate[1]:
                point = center.copy()
                point[0] = rho_direction[0]
                point[1] = gamma_direction[1]
                if point_key(point) != center_key and point_key(point) not in {
                    point_key(existing) for existing in mixed_points
                }:
                    mixed_points.append(point)
        mixed_checked = bool(mixed_points)
        if not mixed_checked:
            raise RuntimeError(
                "joint rho-gamma local certificate has no feasible mixed corner"
            )

        required_points = [
            point
            for axial in axial_by_coordinate
            for point, _ in axial
        ] + mixed_points
        if not has_budget_for(
            *required_points,
            evaluation_limit=final_stage_limit,
        ):
            raise RuntimeError(
                "joint rho-gamma final budget cannot complete the required "
                "axial/mixed local certificate"
            )

        certificate_attempts += 1
        certificate_values: list[float] = []
        for coordinate, axis in enumerate(axis_names):
            for point, stencil in axial_by_coordinate[coordinate]:
                certificate_values.append(
                    evaluate(
                        point,
                        "certificate_axial",
                        optimizer_phase="local_discrete_certificate",
                        certificate_index=certificate_index,
                        certificate_log_step=float(settings.certificate_log_step),
                        coordinate=axis,
                        stencil=stencil,
                    )
                )
        for mixed_index, point in enumerate(mixed_points, start=1):
            certificate_values.append(
                evaluate(
                    point,
                    "certificate_mixed_corner",
                    optimizer_phase="local_discrete_certificate",
                    certificate_index=certificate_index,
                    certificate_log_step=float(settings.certificate_log_step),
                    stencil=f"mixed_{mixed_index}",
                )
            )
        if not all(math.isfinite(value) for value in certificate_values):
            raise RuntimeError(
                "joint rho-gamma local certificate contains a non-finite risk"
            )

        verified_key, verified_f = best_item()
        verified = verified_key == center_key
        rows.append(
            {
                "optimizer_event": "certificate_diagnostic",
                "stage": "local_discrete_certificate",
                "optimizer_phase": "local_discrete_certificate",
                "log_rho": float(center[0]),
                "log_gamma": float(center[1]),
                "rho": float(math.exp(center[0])),
                "gamma": float(math.exp(center[1])),
                "joint_risk": float(center_f),
                "cache_hit": True,
                "certificate_index": certificate_index,
                "certificate_log_step": float(settings.certificate_log_step),
                "axis_rho_bracketed": bool(axis_bracketed["rho"]),
                "axis_gamma_bracketed": bool(axis_bracketed["gamma"]),
                "mixed_checked": bool(mixed_checked),
                "certificate_neighbor_count": len(required_points),
                "certificate_winner_log_rho": float(verified_key[0]),
                "certificate_winner_log_gamma": float(verified_key[1]),
                "certificate_winner_risk": float(verified_f),
                "certificate_verified": bool(verified),
            }
        )
        if verified:
            convergence_status = "verified_local_discrete"
            certified_certificate_index = int(certificate_index)
            break
        verification_improvement_count += 1
        if certificate_index > int(settings.certificate_max_restarts):
            raise RuntimeError(
                "joint rho-gamma certificate kept finding improvements after "
                "the preregistered restart cap"
            )

    if convergence_status != "verified_local_discrete":
        raise RuntimeError(
            "joint rho-gamma optimizer did not obtain a verified local "
            "discrete certificate"
        )

    best_first_polish_evaluation_count = int(
        certificate_start_count - exploration_evaluation_count
    )
    if recovery_enabled:
        certificate_evaluation_count = int(
            sum(certificate_evaluation_counts.values())
        )
        recovery_evaluation_count = int(sum(recovery_evaluation_counts.values()))
        expected_phase_counts = {
            "global_exploration": int(exploration_evaluation_count),
            "best_first_polish": best_first_polish_evaluation_count,
            "local_discrete_certificate": certificate_evaluation_count,
            "certificate_recovery": recovery_evaluation_count,
        }
        if any(
            int(unique_evaluations_by_phase.get(phase, 0)) != count
            for phase, count in expected_phase_counts.items()
        ) or set(unique_evaluations_by_phase) - set(expected_phase_counts):
            raise_v2_2_failure(
                "phase_evaluation_count_mismatch",
                "joint rho-gamma v2.2 phase counters differ from exact replays",
            )
        if sum(expected_phase_counts.values()) != int(len(cache)):
            raise_v2_2_failure(
                "total_evaluation_count_mismatch",
                "joint rho-gamma v2.2 phase counters do not sum to total evaluations",
            )
        if (
            best_first_polish_evaluation_count
            > int(polish_evaluation_limit - exploration_evaluation_count)
            or certificate_evaluation_count > int(certificate_reserve)
            or recovery_evaluation_count > int(recovery_reserve)
            or certificate_attempts != verification_improvement_count + 1
            or recovery_sweep_count != verification_improvement_count
            or recovery_sweep_count > int(settings.certificate_max_restarts)
            or certified_certificate_index != certificate_attempts
            or recovery_coordinate_attempts != 2 * recovery_pass_count
        ):
            raise_v2_2_failure(
                "recovery_certificate_counter_mismatch",
                "joint rho-gamma v2.2 recovery/certificate counters are inconsistent",
            )
        result_phase_evaluation_counts = dict(expected_phase_counts)
    else:
        certificate_evaluation_count = int(len(cache) - certificate_start_count)
        recovery_evaluation_count = 0
        result_phase_evaluation_counts = dict(unique_evaluations_by_phase)

    best_key, best_f = best_item()
    rho, gamma = decoded(best_key)
    trace = pd.DataFrame(rows)
    trace["selected"] = False
    selected_mask = (
        pd.to_numeric(trace.get("log_rho"), errors="coerce").sub(best_key[0]).abs().le(1e-13)
        & pd.to_numeric(trace.get("log_gamma"), errors="coerce").sub(best_key[1]).abs().le(1e-13)
        & trace["optimizer_event"].eq("evaluation")
    )
    if bool(selected_mask.any()):
        trace.loc[trace.index[selected_mask][-1], "selected"] = True
    return JointRhoGammaNewtonResult(
        rho=rho,
        gamma=gamma,
        objective=float(best_f),
        trace=trace,
        evaluation_count=int(len(cache)),
        fallback_count=int(fallback_count),
        trust_region_attempts=int(trust_attempts),
        trust_region_accepts=int(trust_accepts),
        exploration_evaluation_limit=exploration_limit,
        exploration_evaluation_count=exploration_evaluation_count,
        final_polish_evaluation_count=int(len(cache) - exploration_evaluation_count),
        convergence_status=convergence_status,
        axis_bracketed=dict(axis_bracketed),
        mixed_checked=bool(mixed_checked),
        last_accepted_step_capped=bool(last_accepted_step_capped),
        verification_improvement_count=int(verification_improvement_count),
        final_polish_accepted_steps=int(final_polish_accepted_steps),
        certificate_attempts=int(certificate_attempts),
        best_first_polish_evaluation_limit=int(
            polish_evaluation_limit - exploration_evaluation_count
        ),
        best_first_polish_evaluation_count=int(
            best_first_polish_evaluation_count
        ),
        certificate_evaluation_reserve=int(certificate_reserve),
        certificate_evaluation_count=int(certificate_evaluation_count),
        certificate_max_attempts=int(settings.certificate_max_restarts) + 1,
        certificate_max_neighbors_per_attempt=(
            JOINT_CERTIFICATE_MAX_NEIGHBORS_PER_ATTEMPT
        ),
        certificate_worst_case_evaluations=(
            (int(settings.certificate_max_restarts) + 1)
            * JOINT_CERTIFICATE_MAX_NEIGHBORS_PER_ATTEMPT
        ),
        certificate_evaluation_counts=dict(certificate_evaluation_counts),
        certified_certificate_index=int(certified_certificate_index),
        recovery_sweep_evaluation_limit=(
            int(recovery_sweep_evaluation_limit) if recovery_enabled else 0
        ),
        recovery_sweep_limit=(
            int(settings.certificate_max_restarts) if recovery_enabled else 0
        ),
        recovery_evaluation_reserve=int(recovery_reserve),
        recovery_evaluation_count=int(recovery_evaluation_count),
        recovery_evaluation_counts=dict(recovery_evaluation_counts),
        recovery_sweep_count=int(recovery_sweep_count),
        recovery_pass_count=int(recovery_pass_count),
        recovery_pass_counts=dict(recovery_pass_counts),
        recovery_coordinate_attempts=int(recovery_coordinate_attempts),
        recovery_accepted_steps=int(recovery_accepted_steps),
        recovery_last_pass_improved=dict(recovery_last_pass_improved),
        phase_evaluation_counts=result_phase_evaluation_counts,
    )


class JointRhoGammaSelector:
    ""

    def __init__(
        self,
        replay: ExactValidationReplay,
        *,
        scales: Mapping[str, float],
        reference_transformed: Mapping[str, float] | None = None,
        normalization_scales: Mapping[str, float] | None = None,
        settings: JointRhoGammaSelectionSettings,
        retain_replay_artifacts: bool = True,
    ) -> None:
        if not scales or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in scales.values()
        ):
            raise ValueError("joint rho-gamma fixed scales must be finite and positive")
        self.replay = replay
        self.scales = {str(key): float(value) for key, value in sorted(scales.items())}
        if (reference_transformed is None) != (normalization_scales is None):
            raise ValueError(
                "reference_transformed and normalization_scales must both be "
                "provided or both be computed from validation anchors"
            )
        self.reference_transformed = (
            _validated_metric_map(
                reference_transformed,
                name="reference_transformed",
                strictly_positive=False,
            )
            if reference_transformed is not None
            else None
        )
        self.normalization_scales = (
            _validated_metric_map(
                normalization_scales,
                name="normalization_scales",
                strictly_positive=True,
            )
            if normalization_scales is not None
            else None
        )
        self.settings = settings.validate()
        self.retain_replay_artifacts = bool(retain_replay_artifacts)

    def _validation_normalization(
        self,
        pilot_metrics: list[dict[str, float]],
        reference_metrics: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        ""

        settings = self.settings
        reference = transformed_joint_metrics(
            reference_metrics,
            metric_epsilon=settings.metric_epsilon,
            coverage_target=settings.coverage_target,
            coverage_tolerance=settings.coverage_tolerance,
            coverage_upper_weight=settings.coverage_upper_weight,
        )
        transformed = [
            transformed_joint_metrics(
                metrics,
                metric_epsilon=settings.metric_epsilon,
                coverage_target=settings.coverage_target,
                coverage_tolerance=settings.coverage_tolerance,
                coverage_upper_weight=settings.coverage_upper_weight,
            )
            for metrics in pilot_metrics
        ]
        scales: dict[str, float] = {}
        for name in JOINT_METRICS:
            values = np.asarray(
                [row[name] for row in transformed if math.isfinite(float(row[name]))],
                dtype=float,
            )
            median = float(np.median(values)) if values.size else 0.0
            robust = (
                float(1.4826 * np.median(np.abs(values - median)))
                if values.size
                else 0.0
            )
            relative_floor = settings.robust_relative_floor * max(
                abs(float(reference[name])), settings.robust_scale_floor
            )
            scales[name] = float(
                max(robust, relative_floor, settings.robust_scale_floor)
            )
        return reference, scales

    def state(self, rho: float, gamma: float) -> ParameterState:
        rho_lower, rho_upper = self.settings.rho_bounds
        gamma_lower, gamma_upper = self.settings.gamma_bounds
        selected_rho = float(min(max(float(rho), rho_lower), rho_upper))
        shared_gamma = float(min(max(float(gamma), gamma_lower), gamma_upper))
        return ParameterState(
            family=MOMENT_T,
            scales=dict(self.scales),
            gammas={key: shared_gamma for key in self.scales},
            nus={key: 5.0 for key in self.scales},
            rho=selected_rho,
        )

    def select(self, *, variant: str) -> JointRhoGammaSelectionOutcome:
        if variant not in FILTER_VARIANTS:
            raise ValueError(f"unknown filtering variant {variant!r}")
        artifacts_by_point: dict[tuple[float, float], ReplayArtifacts] = {}
        metrics_by_point: dict[tuple[float, float], dict[str, float]] = {}
        evaluation_rows: list[dict[str, object]] = []

        def point_key(rho: float, gamma: float) -> tuple[float, float]:
            return round(float(rho), 12), round(float(gamma), 12)

        def metrics_at(rho: float, gamma: float) -> dict[str, float]:
            key = point_key(rho, gamma)
            if key not in metrics_by_point:
                artifact = self.replay.evaluate(self.state(*key), variant=variant)
                if self.retain_replay_artifacts:
                    artifacts_by_point[key] = artifact
                metrics_by_point[key] = dict(artifact.metrics)
            return metrics_by_point[key]

        if self.reference_transformed is None:
                                                                             
                                                                                
                                                                           
            pilot_metrics = [
                metrics_at(rho, self.settings.reference_gamma)
                for rho in self.settings.anchor_rhos
            ]
            reference_metrics = metrics_at(
                self.settings.reference_rho,
                self.settings.reference_gamma,
            )
            reference_transformed, normalization_scales = (
                self._validation_normalization(pilot_metrics, reference_metrics)
            )
        else:
            assert self.normalization_scales is not None
            reference_transformed = dict(self.reference_transformed)
            normalization_scales = dict(self.normalization_scales)

        def objective(rho: float, gamma: float) -> float:
            key = point_key(rho, gamma)
            metrics = metrics_at(*key)
            transformed = transformed_joint_metrics(
                metrics,
                metric_epsilon=self.settings.metric_epsilon,
                coverage_target=self.settings.coverage_target,
                coverage_tolerance=self.settings.coverage_tolerance,
                coverage_upper_weight=self.settings.coverage_upper_weight,
            )
            z = {
                name: (
                    float(transformed[name])
                    - float(reference_transformed[name])
                )
                / float(normalization_scales[name])
                for name in JOINT_METRICS
            }
            risk = float(
                sum(
                    float(self.settings.objective_weights[name]) * float(z[name])
                    for name in JOINT_METRICS
                )
            )
            evaluation_rows.append(
                {
                    "variant": variant,
                    "bridge_family": MOMENT_T,
                    "rho": key[0],
                    "gamma": key[1],
                    "joint_risk": risk,
                    **{name: float(value) for name, value in metrics.items()},
                    **{f"z_{name}": float(value) for name, value in z.items()},
                }
            )
            return risk

        optimizer = safeguarded_bounded_newton_logrho_gamma(
            objective,
            settings=self.settings,
        )
        selected_key = point_key(optimizer.rho, optimizer.gamma)
        selected_metrics = metrics_at(*selected_key)
        selected_artifacts = (
            artifacts_by_point[selected_key]
            if self.retain_replay_artifacts
            else self.replay.evaluate(self.state(*selected_key), variant=variant)
        )
        metric_trace = pd.DataFrame(evaluation_rows).drop_duplicates(
            ["variant", "rho", "gamma"], keep="last"
        )
        trace = optimizer.trace.copy()
        trace["rho_join"] = pd.to_numeric(trace["rho"], errors="coerce").round(12)
        trace["gamma_join"] = pd.to_numeric(trace["gamma"], errors="coerce").round(12)
        metric_trace = metric_trace.rename(
            columns={"rho": "rho_join", "gamma": "gamma_join"}
        )
        trace = trace.merge(
            metric_trace.drop(columns=["variant"]),
            on=["rho_join", "gamma_join"],
            how="left",
            suffixes=("_optimizer", ""),
        ).drop(columns=["rho_join", "gamma_join"])
        if "joint_risk" not in trace:
            trace["joint_risk"] = trace["joint_risk_optimizer"]
        return JointRhoGammaSelectionOutcome(
            variant=variant,
            selected_state=self.state(*selected_key),
            selected_metrics=selected_metrics,
            selected_objective=float(optimizer.objective),
            reference_transformed=dict(reference_transformed),
            normalization_scales=dict(normalization_scales),
            trace=trace,
            replay_artifacts=selected_artifacts,
            optimizer=optimizer,
        )
