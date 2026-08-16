""



















from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Callable, Mapping, Sequence

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
from .rho_gamma_joint import (
    JOINT_RHO_ANCHOR_BANK,
    JOINT_RHO_GAMMA_OBJECTIVE_WEIGHTS,
)


VECTOR_GAMMA_ANCHORS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.125,
    0.25,
    0.50,
    1.0 / math.sqrt(2.0),
    1.0,
    math.sqrt(2.0),
    2.0,
    4.0,
)
VECTOR_SUPPORTED_GAMMA_COUNTS = (3, 6)
VECTOR_PROTOCOL_NAME = "rho_component_horizon_gamma_vector_newton_v3"
VECTOR_CONVERGENCE_STATUS = "verified_local_discrete_vector"
VECTOR_PARAMETER_DECIMALS = 12
VECTOR_PARAMETER_CANONICALIZATION = (
    "round_natural_parameter_to_12_decimals_then_clip_closed_bounds"
)


def canonical_parameter_point(
    rho: float,
    gammas: Mapping[str, float],
    *,
    gamma_keys: Sequence[str],
    rho_bounds: tuple[float, float],
    gamma_bounds: tuple[float, float],
) -> tuple[float, dict[str, float]]:
    ""

    keys = tuple(str(key) for key in gamma_keys)
    if len(set(keys)) != len(keys) or set(gammas) != set(keys):
        raise ValueError("canonical vector point must exactly cover gamma_keys")
    rho_lower, rho_upper = (float(value) for value in rho_bounds)
    gamma_lower, gamma_upper = (float(value) for value in gamma_bounds)
    raw_rho = float(rho)
    raw_gammas = {key: float(gammas[key]) for key in keys}
    if not math.isfinite(raw_rho) or any(
        not math.isfinite(value) for value in raw_gammas.values()
    ):
        raise ValueError("canonical vector point must contain finite parameters")

    canonical_rho = min(
        max(round(raw_rho, VECTOR_PARAMETER_DECIMALS), rho_lower),
        rho_upper,
    )
    canonical_gammas = {
        key: min(
            max(
                round(raw_gammas[key], VECTOR_PARAMETER_DECIMALS),
                gamma_lower,
            ),
            gamma_upper,
        )
        for key in keys
    }
    return float(canonical_rho), {
        key: float(value) for key, value in canonical_gammas.items()
    }


@dataclass(frozen=True)
class VectorJointRhoGammaBudget:
    ""

    max_evaluations: int
    exploration_evaluation_limit: int
    final_reserve_evaluation_limit: int

    def serializable(self) -> dict[str, int]:
        return asdict(self)


_DEFAULT_VECTOR_BUDGETS = {
    3: VectorJointRhoGammaBudget(320, 160, 160),
    6: VectorJointRhoGammaBudget(576, 288, 288),
}


@dataclass(frozen=True)
class CertifiedSharedWarmStart:
    ""

    rho: float
    gamma: float
    convergence_status: str
    source_identity: str | None = None
    source_objective: float | None = None

    def validate(
        self,
        *,
        rho_bounds: tuple[float, float],
        gamma_bounds: tuple[float, float],
    ) -> "CertifiedSharedWarmStart":
        if self.convergence_status != "verified_local_discrete":
            raise ValueError(
                "vector-v3 requires a shared-v2 warm start with "
                "convergence_status='verified_local_discrete'"
            )
        rho = float(self.rho)
        gamma = float(self.gamma)
        if not math.isfinite(rho) or not rho_bounds[0] <= rho <= rho_bounds[1]:
            raise ValueError("certified shared warm-start rho is outside bounds")
        if not math.isfinite(gamma) or not gamma_bounds[0] <= gamma <= gamma_bounds[1]:
            raise ValueError("certified shared warm-start gamma is outside bounds")
        if self.source_identity is not None and not str(self.source_identity).strip():
            raise ValueError("shared warm-start source_identity must be nonempty")
        if self.source_objective is not None and not math.isfinite(
            float(self.source_objective)
        ):
            raise ValueError("shared warm-start source_objective must be finite")
        return self

    def serializable(self) -> dict[str, object]:
        return {
            "rho": float(self.rho),
            "gamma": float(self.gamma),
            "convergence_status": str(self.convergence_status),
            "source_identity": self.source_identity,
            "source_objective": (
                None
                if self.source_objective is None
                else float(self.source_objective)
            ),
        }


@dataclass(frozen=True)
class VectorJointRhoGammaSelectionSettings:
    ""






    objective_weights: dict[str, float] = field(
        default_factory=lambda: dict(JOINT_RHO_GAMMA_OBJECTIVE_WEIGHTS)
    )
    rho_bounds: tuple[float, float] = (0.001, 0.5)
    gamma_bounds: tuple[float, float] = (0.005, 4.0)
    anchor_rhos: tuple[float, ...] = JOINT_RHO_ANCHOR_BANK[:9]
    anchor_gammas: tuple[float, ...] = VECTOR_GAMMA_ANCHORS
    reference_rho: float = 0.50
    reference_gamma: float = 1.0
    multi_starts: int = 4
    coordinate_passes: int = 3
    structured_direction_passes: int = 1
    final_polish_max_passes: int = 3
    final_polish_max_accepted_steps: int = 12
    finite_difference_log_step: float = 0.10
    minimum_difference_log_step: float = 0.01
    maximum_newton_log_step: float = 0.50
    maximum_structured_log_step: float = 0.35
    certificate_log_step: float = 0.01
    certificate_max_restarts: int = 2
    max_backtracks: int = 8
    hessian_floor: float = 1e-6
    x_tolerance: float = 1e-3
    objective_tolerance: float = 1e-8
    exact_tie_tolerance: float = 1e-10
    metric_epsilon: float = 1e-8
    coverage_target: float = 0.90
    coverage_tolerance: float = 0.03
    coverage_upper_weight: float = 0.5
    robust_scale_floor: float = 1e-3
    robust_relative_floor: float = 0.05
    max_evaluations: int | None = None
    exploration_evaluation_limit: int | None = None
    final_reserve_evaluation_limit: int | None = None

    def resolved_budget(self, gamma_count: int) -> VectorJointRhoGammaBudget:
        ""

        count = int(gamma_count)
        if count not in VECTOR_SUPPORTED_GAMMA_COUNTS:
            raise ValueError("vector-v3 supports exactly K=3 or K=6 gamma keys")
        default = _DEFAULT_VECTOR_BUDGETS[count]
        budget = VectorJointRhoGammaBudget(
            max_evaluations=(
                default.max_evaluations
                if self.max_evaluations is None
                else int(self.max_evaluations)
            ),
            exploration_evaluation_limit=(
                default.exploration_evaluation_limit
                if self.exploration_evaluation_limit is None
                else int(self.exploration_evaluation_limit)
            ),
            final_reserve_evaluation_limit=(
                default.final_reserve_evaluation_limit
                if self.final_reserve_evaluation_limit is None
                else int(self.final_reserve_evaluation_limit)
            ),
        )
        return budget

    @staticmethod
    def certificate_neighbor_upper_bound(gamma_count: int) -> int:
        ""





        count = int(gamma_count)
        return 2 * (count + 1) + 4 * count + 2 * count

    @staticmethod
    def recovery_sweep_evaluation_upper_bound(gamma_count: int) -> int:
        ""






        count = int(gamma_count)
        return 4 * ((count + 1) + count)

    def final_reserve_decomposition(self, gamma_count: int) -> dict[str, int]:
        ""

        count = int(gamma_count)
        budget = self.resolved_budget(count)
        certificate_attempt_limit = int(self.certificate_max_restarts) + 1
        recovery_sweep_limit = int(self.certificate_max_restarts)
        certificate_per_attempt = self.certificate_neighbor_upper_bound(count)
        recovery_per_sweep = self.recovery_sweep_evaluation_upper_bound(count)
        initial_polish = int(
            budget.final_reserve_evaluation_limit
            - certificate_attempt_limit * certificate_per_attempt
            - recovery_sweep_limit * recovery_per_sweep
        )
        return {
            "initial_final_polish_evaluation_limit": initial_polish,
            "certificate_attempt_evaluation_limit": certificate_per_attempt,
            "certificate_attempt_limit": certificate_attempt_limit,
            "certificate_evaluation_reserve": (
                certificate_attempt_limit * certificate_per_attempt
            ),
            "recovery_sweep_evaluation_limit": recovery_per_sweep,
            "recovery_sweep_limit": recovery_sweep_limit,
            "recovery_evaluation_reserve": recovery_sweep_limit * recovery_per_sweep,
        }

    def validate_for_keys(
        self,
        gamma_keys: Sequence[str],
        shared_warm_start: CertifiedSharedWarmStart | None = None,
    ) -> "VectorJointRhoGammaSelectionSettings":
        keys = tuple(sorted(str(key) for key in gamma_keys))
        if len(keys) not in VECTOR_SUPPORTED_GAMMA_COUNTS:
            raise ValueError("vector-v3 supports exactly K=3 or K=6 gamma keys")
        if len(set(keys)) != len(keys) or any(not key for key in keys):
            raise ValueError("gamma keys must be unique nonempty strings")
        if set(self.objective_weights) != set(JOINT_METRICS):
            raise ValueError("vector-v3 objective has the wrong metric keys")
        weights = np.asarray(list(self.objective_weights.values()), dtype=float)
        if not np.isfinite(weights).all() or (weights < 0.0).any():
            raise ValueError("vector-v3 objective weights must be finite and nonnegative")
        if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("vector-v3 objective weights must sum to one")

        rho_lower, rho_upper = (float(value) for value in self.rho_bounds)
        gamma_lower, gamma_upper = (float(value) for value in self.gamma_bounds)
        if not (
            math.isfinite(rho_lower)
            and math.isfinite(rho_upper)
            and 0.0 < rho_lower < rho_upper
        ):
            raise ValueError("rho bounds must be finite, positive, and increasing")
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
        if shared_warm_start is not None:
            shared_warm_start.validate(
                rho_bounds=(rho_lower, rho_upper),
                gamma_bounds=(gamma_lower, gamma_upper),
            )

        integer_positive = (
            self.multi_starts,
            self.coordinate_passes,
            self.structured_direction_passes,
            self.final_polish_max_passes,
            self.final_polish_max_accepted_steps,
            self.certificate_max_restarts,
            self.max_backtracks,
        )
        if any(int(value) < 1 for value in integer_positive):
            raise ValueError("vector-v3 iteration limits must be positive")
        if int(self.certificate_max_restarts) != 2:
            raise ValueError("vector-v3 preregisters exactly two certificate recoveries")
        positive = (
            self.finite_difference_log_step,
            self.minimum_difference_log_step,
            self.maximum_newton_log_step,
            self.maximum_structured_log_step,
            self.certificate_log_step,
            self.hessian_floor,
            self.x_tolerance,
            self.coverage_tolerance,
            self.robust_scale_floor,
            self.robust_relative_floor,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("vector-v3 scale settings must be finite and positive")
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
            raise ValueError("vector-v3 tolerances must be finite and nonnegative")

        budget = self.resolved_budget(len(keys))
        if (
            budget.max_evaluations < 1
            or budget.exploration_evaluation_limit < 1
            or budget.final_reserve_evaluation_limit < 1
            or budget.exploration_evaluation_limit
            + budget.final_reserve_evaluation_limit
            != budget.max_evaluations
        ):
            raise ValueError(
                "vector-v3 budget must satisfy max=exploration+final_reserve"
            )
        decomposition = self.final_reserve_decomposition(len(keys))
        if decomposition["initial_final_polish_evaluation_limit"] < 1:
            raise ValueError(
                "vector-v3 final reserve cannot cover initial polish, three "
                "certificates, and two complete recovery sweeps"
            )
        return self


@dataclass
class VectorJointRhoGammaNewtonResult:
    rho: float
    gammas: dict[str, float]
    objective: float
    trace: pd.DataFrame
    gamma_keys: tuple[str, ...]
    dimension: int
    max_evaluations: int
    exploration_evaluation_limit: int
    final_reserve_evaluation_limit: int
    initial_final_polish_evaluation_limit: int
    certificate_attempt_evaluation_limit: int
    certificate_attempt_limit: int
    certificate_evaluation_reserve: int
    recovery_sweep_evaluation_limit: int
    recovery_sweep_limit: int
    recovery_evaluation_reserve: int
    evaluation_count: int
    exploration_evaluation_count: int
    final_polish_evaluation_count: int
    recovery_evaluation_count: int
    certificate_evaluation_count: int
    certificate_evaluation_counts: dict[str, int]
    recovery_evaluation_counts: dict[str, int]
    fallback_count: int
    structured_direction_attempts: int
    structured_direction_accepts: int
    convergence_status: str
    axis_bracketed: dict[str, bool]
    rho_gamma_mixed_checked: dict[str, bool]
    rho_gamma_mixed_corner_counts: dict[str, int]
    hadamard_checked: bool
    hadamard_direction_count: int
    hadamard_direction_bank: tuple[tuple[int, ...], ...]
    hadamard_direction_bank_sha256: str
    hadamard_per_direction_checked: dict[str, bool]
    hadamard_distinct_neighbor_counts: dict[str, int]
    certificate_attempts: int
    certified_certificate_index: int
    verification_improvement_count: int
    recovery_sweep_count: int
    recovery_coordinate_attempts: int
    recovery_hadamard_attempts: int
    final_polish_accepted_steps: int
    last_accepted_step_capped: bool
    shared_warm_start: CertifiedSharedWarmStart
    shared_warm_start_objective: float
    selected_minus_warm_delta: float

    def metadata(self) -> dict[str, object]:
        ""

        return {
            "optimizer_protocol": VECTOR_PROTOCOL_NAME,
            "parameter_decimal_places": VECTOR_PARAMETER_DECIMALS,
            "parameter_canonicalization": VECTOR_PARAMETER_CANONICALIZATION,
            "rho": float(self.rho),
            "gammas": {
                key: float(value) for key, value in sorted(self.gammas.items())
            },
            "objective": float(self.objective),
            "gamma_keys": list(self.gamma_keys),
            "dimension": int(self.dimension),
            "max_evaluations": int(self.max_evaluations),
            "exploration_evaluation_limit": int(
                self.exploration_evaluation_limit
            ),
            "final_reserve_evaluation_limit": int(
                self.final_reserve_evaluation_limit
            ),
            "initial_final_polish_evaluation_limit": int(
                self.initial_final_polish_evaluation_limit
            ),
            "certificate_attempt_evaluation_limit": int(
                self.certificate_attempt_evaluation_limit
            ),
            "certificate_attempt_limit": int(self.certificate_attempt_limit),
            "certificate_evaluation_reserve": int(
                self.certificate_evaluation_reserve
            ),
            "recovery_sweep_evaluation_limit": int(
                self.recovery_sweep_evaluation_limit
            ),
            "recovery_sweep_limit": int(self.recovery_sweep_limit),
            "recovery_evaluation_reserve": int(self.recovery_evaluation_reserve),
            "evaluation_count": int(self.evaluation_count),
            "exploration_evaluation_count": int(
                self.exploration_evaluation_count
            ),
            "final_polish_evaluation_count": int(
                self.final_polish_evaluation_count
            ),
            "recovery_evaluation_count": int(self.recovery_evaluation_count),
            "certificate_evaluation_count": int(
                self.certificate_evaluation_count
            ),
            "certificate_evaluation_counts": dict(
                self.certificate_evaluation_counts
            ),
            "recovery_evaluation_counts": dict(self.recovery_evaluation_counts),
            "fallback_count": int(self.fallback_count),
            "structured_direction_attempts": int(
                self.structured_direction_attempts
            ),
            "structured_direction_accepts": int(self.structured_direction_accepts),
            "convergence_status": self.convergence_status,
            "axis_bracketed": dict(self.axis_bracketed),
            "rho_gamma_mixed_checked": dict(self.rho_gamma_mixed_checked),
            "rho_gamma_mixed_corner_counts": dict(
                self.rho_gamma_mixed_corner_counts
            ),
            "hadamard_checked": bool(self.hadamard_checked),
            "hadamard_direction_count": int(self.hadamard_direction_count),
            "hadamard_direction_bank": [
                list(direction) for direction in self.hadamard_direction_bank
            ],
            "hadamard_direction_bank_sha256": str(
                self.hadamard_direction_bank_sha256
            ),
            "hadamard_per_direction_checked": dict(
                self.hadamard_per_direction_checked
            ),
            "hadamard_distinct_neighbor_counts": dict(
                self.hadamard_distinct_neighbor_counts
            ),
            "certificate_attempts": int(self.certificate_attempts),
            "certified_certificate_index": int(self.certified_certificate_index),
            "verification_improvement_count": int(
                self.verification_improvement_count
            ),
            "recovery_sweep_count": int(self.recovery_sweep_count),
            "recovery_coordinate_attempts": int(
                self.recovery_coordinate_attempts
            ),
            "recovery_hadamard_attempts": int(self.recovery_hadamard_attempts),
            "final_polish_accepted_steps": int(
                self.final_polish_accepted_steps
            ),
            "last_accepted_step_capped": bool(self.last_accepted_step_capped),
            "shared_warm_start": self.shared_warm_start.serializable(),
            "shared_warm_start_objective": float(
                self.shared_warm_start_objective
            ),
            "selected_minus_warm_delta": float(self.selected_minus_warm_delta),
        }


@dataclass
class JointRhoGammaVectorSelectionOutcome:
    variant: str
    selected_state: ParameterState
    selected_metrics: dict[str, float]
    selected_objective: float
    reference_transformed: dict[str, float]
    normalization_scales: dict[str, float]
    trace: pd.DataFrame
    replay_artifacts: ReplayArtifacts
    optimizer: VectorJointRhoGammaNewtonResult


def _metric_map(
    values: Mapping[str, float],
    *,
    name: str,
    strictly_positive: bool,
) -> dict[str, float]:
    if set(values) != set(JOINT_METRICS):
        raise ValueError(f"{name} must contain exactly {','.join(JOINT_METRICS)}")
    result = {metric: float(values[metric]) for metric in JOINT_METRICS}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"{name} values must be finite")
    if strictly_positive and not all(value > 0.0 for value in result.values()):
        raise ValueError(f"{name} values must be strictly positive")
    return result


def structured_hadamard_directions(gamma_count: int) -> tuple[tuple[int, ...], ...]:
    ""







    count = int(gamma_count)
    if count not in VECTOR_SUPPORTED_GAMMA_COUNTS:
        raise ValueError("structured Hadamard directions require K=3 or K=6")
    order = 1
    matrix = np.ones((1, 1), dtype=int)
    while order < count:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
        order *= 2
    directions: list[tuple[int, ...]] = []
    for row in matrix[:count, :count]:
        values = tuple(int(value) for value in row)
        if values[0] < 0:
            values = tuple(-value for value in values)
        if values not in directions:
            directions.append(values)
    bank = np.asarray(directions, dtype=float)
    if (
        len(directions) != count
        or bank.shape != (count, count)
        or int(np.linalg.matrix_rank(bank)) != count
    ):
        raise RuntimeError("truncated Hadamard bank is unexpectedly rank-deficient")
    return tuple(directions)


def structured_hadamard_bank_sha256(gamma_count: int) -> str:
    ""

    directions = structured_hadamard_directions(gamma_count)
    payload = {
        "gamma_count": int(gamma_count),
        "directions": [list(direction) for direction in directions],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safeguarded_bounded_newton_logrho_gamma_vector(
    objective: Callable[[float, Mapping[str, float]], float],
    *,
    gamma_keys: Sequence[str],
    shared_warm_start: CertifiedSharedWarmStart,
    settings: VectorJointRhoGammaSelectionSettings,
) -> VectorJointRhoGammaNewtonResult:
    ""

    keys = tuple(sorted(str(key) for key in gamma_keys))
    settings.validate_for_keys(keys, shared_warm_start)
    budget = settings.resolved_budget(len(keys))
    dimension = len(keys) + 1
    lower = np.log(
        np.asarray(
            [settings.rho_bounds[0]]
            + [settings.gamma_bounds[0]] * len(keys),
            dtype=float,
        )
    )
    upper = np.log(
        np.asarray(
            [settings.rho_bounds[1]]
            + [settings.gamma_bounds[1]] * len(keys),
            dtype=float,
        )
    )
    reference = np.log(
        np.asarray(
            [settings.reference_rho]
            + [settings.reference_gamma] * len(keys),
            dtype=float,
        )
    )
    warm = np.log(
        np.asarray(
            [shared_warm_start.rho]
            + [shared_warm_start.gamma] * len(keys),
            dtype=float,
        )
    )
    cache: dict[tuple[float, ...], float] = {}
    rows: list[dict[str, object]] = []
    fallback_count = 0
    structured_attempts = 0
    structured_accepts = 0
    unique_evaluations_by_phase: dict[str, int] = {}

    def point_key(point: np.ndarray) -> tuple[float, ...]:
        clipped = np.minimum(np.maximum(np.asarray(point, dtype=float), lower), upper)
        natural = np.exp(clipped)
        canonical_rho, canonical_gammas = canonical_parameter_point(
            float(natural[0]),
            {
                name: float(natural[index + 1])
                for index, name in enumerate(keys)
            },
            gamma_keys=keys,
            rho_bounds=settings.rho_bounds,
            gamma_bounds=settings.gamma_bounds,
        )
        return tuple(
            math.log(value)
            for value in (
                canonical_rho,
                *[canonical_gammas[name] for name in keys],
            )
        )

    def decoded(key: tuple[float, ...]) -> tuple[float, dict[str, float]]:
        return canonical_parameter_point(
            float(math.exp(key[0])),
            {
                name: float(math.exp(key[index + 1]))
                for index, name in enumerate(keys)
            },
            gamma_keys=keys,
            rho_bounds=settings.rho_bounds,
            gamma_bounds=settings.gamma_bounds,
        )

    def point_columns(key: tuple[float, ...]) -> dict[str, object]:
        rho, gammas = decoded(key)
        result: dict[str, object] = {
            "log_rho": float(key[0]),
            "rho": rho,
            "gammas_json": json.dumps(gammas, sort_keys=True, separators=(",", ":")),
        }
        for index, name in enumerate(keys):
            result[f"log_gamma__{name}"] = float(key[index + 1])
            result[f"gamma__{name}"] = float(gammas[name])
        return result

    def has_budget_for(
        *points: np.ndarray,
        evaluation_limit: int,
    ) -> bool:
        proposed = {point_key(point) for point in points}
        return len(cache) + sum(key not in cache for key in proposed) <= int(
            evaluation_limit
        )

    def evaluate(point: np.ndarray, stage: str, **details: object) -> float:
        key = point_key(point)
        cache_hit = key in cache
        if not cache_hit:
            if len(cache) >= budget.max_evaluations:
                raise RuntimeError("vector-v3 exhausted its exact-replay budget")
            optimizer_phase = details.get("optimizer_phase")
            if not isinstance(optimizer_phase, str) or not optimizer_phase:
                raise RuntimeError("vector-v3 evaluation omitted its optimizer phase")
            rho, gammas = decoded(key)
            value = float(objective(rho, gammas))
            cache[key] = value if math.isfinite(value) else float("inf")
            unique_evaluations_by_phase[optimizer_phase] = (
                unique_evaluations_by_phase.get(optimizer_phase, 0) + 1
            )
        rows.append(
            {
                "optimizer_event": "evaluation",
                "stage": stage,
                "joint_risk": float(cache[key]),
                "cache_hit": bool(cache_hit),
                **point_columns(key),
                **details,
            }
        )
        return float(cache[key])

    def best_item() -> tuple[tuple[float, ...], float]:
        finite = [(key, value) for key, value in cache.items() if math.isfinite(value)]
        if not finite:
            raise RuntimeError("vector-v3 optimizer has no finite candidates")
        minimum = min(value for _, value in finite)
        tolerance = float(settings.exact_tie_tolerance)
        tied = [(key, value) for key, value in finite if value <= minimum + tolerance]
        return min(
            tied,
            key=lambda item: (
                float(np.linalg.norm(np.asarray(item[0], dtype=float) - warm)),
                math.exp(float(item[0][0])),
                tuple(math.exp(float(value)) for value in item[0][1:]),
            ),
        )

    def conditional_gap_midpoint(
        point: np.ndarray,
        coordinate: int,
    ) -> np.ndarray | None:
        key = point_key(point)
        values = [float(lower[coordinate]), float(upper[coordinate])]
        for cached_key in cache:
            if all(
                index == coordinate
                or abs(float(cached_key[index]) - float(key[index])) <= 1e-13
                for index in range(dimension)
            ):
                values.append(float(cached_key[coordinate]))
        values = sorted(set(values))
        gaps = [(right - left, left, right) for left, right in zip(values, values[1:])]
        if not gaps:
            return None
        gap, left, right = max(gaps, key=lambda value: value[0])
        if gap <= 2.0 * float(settings.x_tolerance):
            return None
        proposal = np.asarray(point, dtype=float).copy()
        proposal[coordinate] = (left + right) / 2.0
        return proposal

    def coordinate_step(
        point: np.ndarray,
        current_f: float,
        *,
        coordinate: int,
        evaluation_limit: int,
        optimizer_phase: str,
        pass_index: int,
        start_index: int,
    ) -> tuple[np.ndarray, float, bool]:
        nonlocal fallback_count
        axis = "rho" if coordinate == 0 else keys[coordinate - 1]
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
            if has_budget_for(minus, plus, evaluation_limit=evaluation_limit):
                minus_f = evaluate(
                    minus,
                    "finite_difference_minus",
                    optimizer_phase=optimizer_phase,
                    coordinate=axis,
                    pass_index=pass_index,
                    start_index=start_index,
                )
                plus_f = evaluate(
                    plus,
                    "finite_difference_plus",
                    optimizer_phase=optimizer_phase,
                    coordinate=axis,
                    pass_index=pass_index,
                    start_index=start_index,
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
                if has_budget_for(first, second, evaluation_limit=evaluation_limit):
                    first_f = evaluate(
                        first,
                        "finite_difference_one_sided_1",
                        optimizer_phase=optimizer_phase,
                        coordinate=axis,
                        pass_index=pass_index,
                        start_index=start_index,
                    )
                    second_f = evaluate(
                        second,
                        "finite_difference_one_sided_2",
                        optimizer_phase=optimizer_phase,
                        coordinate=axis,
                        pass_index=pass_index,
                        start_index=start_index,
                    )
                    if direction > 0.0:
                        gradient = (
                            -3.0 * current_f + 4.0 * first_f - second_f
                        ) / (2.0 * h)
                    else:
                        gradient = (
                            3.0 * current_f - 4.0 * first_f + second_f
                        ) / (2.0 * h)
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
                    midpoint, evaluation_limit=evaluation_limit
                ):
                    return point, current_f, False
                midpoint_f = evaluate(
                    midpoint,
                    "coordinate_gap_fallback",
                    optimizer_phase=optimizer_phase,
                    coordinate=axis,
                    pass_index=pass_index,
                    start_index=start_index,
                    gradient=gradient,
                    hessian=hessian,
                )
                if midpoint_f < current_f - float(settings.objective_tolerance):
                    return midpoint, midpoint_f, True
                return point, current_f, False

        for backtrack in range(int(settings.max_backtracks) + 1):
            raw = point.copy()
            raw[coordinate] += step * (0.5**backtrack)
            proposal = np.minimum(np.maximum(raw, lower), upper)
            if abs(float(proposal[coordinate]) - x) <= float(settings.x_tolerance):
                continue
            if not has_budget_for(proposal, evaluation_limit=evaluation_limit):
                break
            proposal_f = evaluate(
                proposal,
                "coordinate_newton_trial" if use_newton else "coordinate_fallback_trial",
                optimizer_phase=optimizer_phase,
                coordinate=axis,
                pass_index=pass_index,
                start_index=start_index,
                gradient=gradient,
                hessian=hessian,
                backtrack=backtrack,
                projected=bool(np.any(np.abs(raw - proposal) > 0.0)),
            )
            if proposal_f < current_f - float(settings.objective_tolerance):
                return proposal, proposal_f, True

        midpoint = conditional_gap_midpoint(point, coordinate)
        if midpoint is not None and has_budget_for(
            midpoint, evaluation_limit=evaluation_limit
        ):
            fallback_count += 1
            midpoint_f = evaluate(
                midpoint,
                "coordinate_gap_fallback",
                optimizer_phase=optimizer_phase,
                coordinate=axis,
                pass_index=pass_index,
                start_index=start_index,
                gradient=gradient,
                hessian=hessian,
            )
            if midpoint_f < current_f - float(settings.objective_tolerance):
                return midpoint, midpoint_f, True
        return point, current_f, False

    def feasible_direction_room(point: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
        positive = float("inf")
        negative = float("inf")
        for index, coefficient in enumerate(direction):
            if coefficient > 0.0:
                positive = min(positive, (upper[index] - point[index]) / coefficient)
                negative = min(negative, (point[index] - lower[index]) / coefficient)
            elif coefficient < 0.0:
                positive = min(positive, (point[index] - lower[index]) / -coefficient)
                negative = min(negative, (upper[index] - point[index]) / -coefficient)
        return max(0.0, float(positive)), max(0.0, float(negative))

    def structured_direction_step(
        point: np.ndarray,
        current_f: float,
        *,
        direction: np.ndarray,
        direction_index: int,
        evaluation_limit: int,
        optimizer_phase: str,
        pass_index: int,
    ) -> tuple[np.ndarray, float, bool]:
        nonlocal fallback_count, structured_attempts, structured_accepts
        plus_room, minus_room = feasible_direction_room(point, direction)
        h = min(
            float(settings.finite_difference_log_step),
            plus_room,
            minus_room,
        )
        gradient = float("nan")
        hessian = float("nan")
        structured_attempts += 1
        if h >= float(settings.minimum_difference_log_step):
            minus = point - h * direction
            plus = point + h * direction
            if not has_budget_for(minus, plus, evaluation_limit=evaluation_limit):
                return point, current_f, False
            minus_f = evaluate(
                minus,
                "structured_finite_difference_minus",
                optimizer_phase=optimizer_phase,
                direction_index=direction_index,
                pass_index=pass_index,
            )
            plus_f = evaluate(
                plus,
                "structured_finite_difference_plus",
                optimizer_phase=optimizer_phase,
                direction_index=direction_index,
                pass_index=pass_index,
            )
            gradient = (plus_f - minus_f) / (2.0 * h)
            hessian = (plus_f - 2.0 * current_f + minus_f) / (h * h)
        else:
            orientation = 1.0 if plus_room >= minus_room else -1.0
            available = max(plus_room, minus_room)
            h = min(float(settings.finite_difference_log_step), available / 2.0)
            if h < float(settings.minimum_difference_log_step):
                return point, current_f, False
            first = point + orientation * h * direction
            second = point + orientation * 2.0 * h * direction
            if not has_budget_for(first, second, evaluation_limit=evaluation_limit):
                return point, current_f, False
            first_f = evaluate(
                first,
                "structured_finite_difference_one_sided_1",
                optimizer_phase=optimizer_phase,
                direction_index=direction_index,
                pass_index=pass_index,
            )
            second_f = evaluate(
                second,
                "structured_finite_difference_one_sided_2",
                optimizer_phase=optimizer_phase,
                direction_index=direction_index,
                pass_index=pass_index,
            )
            directional_gradient = (
                -3.0 * current_f + 4.0 * first_f - second_f
            ) / (2.0 * h)
            gradient = directional_gradient * orientation
            hessian = (current_f - 2.0 * first_f + second_f) / (h * h)

        use_newton = (
            math.isfinite(gradient)
            and math.isfinite(hessian)
            and hessian > float(settings.hessian_floor)
        )
        if use_newton:
            scalar_step = float(
                np.clip(
                    -gradient / hessian,
                    -float(settings.maximum_structured_log_step),
                    float(settings.maximum_structured_log_step),
                )
            )
            use_newton = gradient * scalar_step < 0.0
        if not use_newton:
            fallback_count += 1
            if not math.isfinite(gradient) or gradient == 0.0:
                return point, current_f, False
            scalar_step = -math.copysign(
                float(settings.maximum_structured_log_step), gradient
            )
        for backtrack in range(int(settings.max_backtracks) + 1):
            raw = point + scalar_step * (0.5**backtrack) * direction
            proposal = np.minimum(np.maximum(raw, lower), upper)
            if float(np.max(np.abs(proposal - point))) <= float(settings.x_tolerance):
                continue
            if not has_budget_for(proposal, evaluation_limit=evaluation_limit):
                break
            proposal_f = evaluate(
                proposal,
                "structured_newton_trial" if use_newton else "structured_fallback_trial",
                optimizer_phase=optimizer_phase,
                direction_index=direction_index,
                pass_index=pass_index,
                gradient=gradient,
                hessian=hessian,
                backtrack=backtrack,
                projected=bool(np.any(np.abs(raw - proposal) > 0.0)),
            )
            if proposal_f < current_f - float(settings.objective_tolerance):
                structured_accepts += 1
                return proposal, proposal_f, True
        return point, current_f, False

                                                                             
                                                                        
                                                                            
                                                                          
    anchor_bank: dict[tuple[float, ...], set[str]] = {}

    def add_anchor(rho: float, gamma_values: Sequence[float], kind: str) -> None:
        point = np.log(np.asarray([rho] + list(gamma_values), dtype=float))
        anchor_bank.setdefault(point_key(point), set()).add(kind)

    reference_gammas = [float(settings.reference_gamma)] * len(keys)
    warm_gammas = [float(shared_warm_start.gamma)] * len(keys)
    for rho in settings.anchor_rhos:
        add_anchor(float(rho), reference_gammas, "shared_manifold_reference_gamma")
        add_anchor(float(rho), warm_gammas, "shared_manifold_warm_gamma")
    for gamma in settings.anchor_gammas:
        shared = [float(gamma)] * len(keys)
        add_anchor(float(settings.reference_rho), shared, "shared_manifold_reference_rho")
        add_anchor(float(shared_warm_start.rho), shared, "shared_manifold_warm_rho")
        for index, name in enumerate(keys):
            one_key = list(warm_gammas)
            one_key[index] = float(gamma)
            add_anchor(
                float(shared_warm_start.rho),
                one_key,
                f"per_key_gamma_anchor:{name}",
            )
    for rho in settings.rho_bounds:
        for gamma in settings.gamma_bounds:
            add_anchor(
                float(rho),
                [float(gamma)] * len(keys),
                "shared_manifold_boundary_corner",
            )
    add_anchor(
        float(shared_warm_start.rho),
        warm_gammas,
        "certified_shared_warm_start",
    )
    if len(anchor_bank) > budget.exploration_evaluation_limit:
        raise ValueError("vector-v3 anchor bank exceeds the exploration budget")
    for key, kinds in sorted(anchor_bank.items()):
        evaluate(
            np.asarray(key, dtype=float),
            "optimizer_anchor",
            optimizer_phase="global_exploration",
            anchor_kinds="|".join(sorted(kinds)),
        )

    best_key, _ = best_item()
    geometric = (lower + upper) / 2.0
    starts = [
        np.asarray(best_key, dtype=float),
        warm.copy(),
        reference.copy(),
        geometric.copy(),
    ]
    unique_starts: list[np.ndarray] = []
    seen_starts: set[tuple[float, ...]] = set()
    for point in starts:
        key = point_key(point)
        if key not in seen_starts:
            seen_starts.add(key)
            unique_starts.append(np.asarray(key, dtype=float))
        if len(unique_starts) >= int(settings.multi_starts):
            break

    exploration_limit = int(budget.exploration_evaluation_limit)
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
            start_index=start_index,
        )
        for pass_index in range(1, int(settings.coordinate_passes) + 1):
            improved_in_pass = False
            order = (
                tuple(range(dimension))
                if pass_index % 2
                else tuple(reversed(range(dimension)))
            )
            for coordinate in order:
                if len(cache) >= exploration_limit:
                    break
                current, current_f, improved = coordinate_step(
                    current,
                    current_f,
                    coordinate=coordinate,
                    evaluation_limit=exploration_limit,
                    optimizer_phase="global_exploration",
                    pass_index=pass_index,
                    start_index=start_index,
                )
                improved_in_pass = improved_in_pass or improved
            if not improved_in_pass:
                break

    hadamard = structured_hadamard_directions(len(keys))
    for pass_index in range(1, int(settings.structured_direction_passes) + 1):
        improved_in_pass = False
        for direction_index, gamma_direction in enumerate(hadamard, start=1):
            if len(cache) >= exploration_limit:
                break
            current_key, current_f = best_item()
            current = np.asarray(current_key, dtype=float)
            direction = np.asarray([0.0] + list(gamma_direction), dtype=float)
            local_limit = min(exploration_limit, len(cache) + 4)
            _, _, improved = structured_direction_step(
                current,
                current_f,
                direction=direction,
                direction_index=direction_index,
                evaluation_limit=local_limit,
                optimizer_phase="global_exploration",
                pass_index=pass_index,
            )
            improved_in_pass = improved_in_pass or improved
        if not improved_in_pass:
            break

    exploration_evaluation_count = int(len(cache))
    final_decomposition = settings.final_reserve_decomposition(len(keys))
    initial_polish_limit = int(
        final_decomposition["initial_final_polish_evaluation_limit"]
    )
    certificate_attempt_limit = int(
        final_decomposition["certificate_attempt_evaluation_limit"]
    )
    certificate_attempt_count_limit = int(
        final_decomposition["certificate_attempt_limit"]
    )
    recovery_sweep_limit = int(
        final_decomposition["recovery_sweep_evaluation_limit"]
    )
    recovery_sweep_count_limit = int(final_decomposition["recovery_sweep_limit"])
    final_stage_limit = int(
        exploration_evaluation_count + budget.final_reserve_evaluation_limit
    )
    if final_stage_limit > int(budget.max_evaluations):
        raise RuntimeError("vector-v3 final reserve exceeds the total replay budget")
    polish_limit = int(exploration_evaluation_count + initial_polish_limit)
    final_polish_accepted_steps = 0
    last_accepted_step_capped = False

    for polish_pass in range(1, int(settings.final_polish_max_passes) + 1):
        pass_best_before, _ = best_item()
        improved_in_pass = False
        order = (
            tuple(range(dimension))
            if polish_pass % 2
            else tuple(reversed(range(dimension)))
        )
        for coordinate in order:
            if len(cache) >= polish_limit:
                break
            current_key, current_f = best_item()
            local_limit = min(polish_limit, len(cache) + 4)
            coordinate_step(
                np.asarray(current_key, dtype=float),
                current_f,
                coordinate=coordinate,
                evaluation_limit=local_limit,
                optimizer_phase="best_first_polish",
                pass_index=polish_pass,
                start_index=0,
            )
            next_key, _ = best_item()
            if next_key != current_key:
                improved_in_pass = True
                final_polish_accepted_steps += 1
                if final_polish_accepted_steps >= int(
                    settings.final_polish_max_accepted_steps
                ):
                    last_accepted_step_capped = True
                    break
        if last_accepted_step_capped:
            break
        if len(cache) < polish_limit:
            for direction_index, gamma_direction in enumerate(hadamard, start=1):
                if len(cache) >= polish_limit:
                    break
                current_key, current_f = best_item()
                local_limit = min(polish_limit, len(cache) + 4)
                _, _, improved = structured_direction_step(
                    np.asarray(current_key, dtype=float),
                    current_f,
                    direction=np.asarray([0.0] + list(gamma_direction), dtype=float),
                    direction_index=direction_index,
                    evaluation_limit=local_limit,
                    optimizer_phase="best_first_polish",
                    pass_index=polish_pass,
                )
                if improved:
                    improved_in_pass = True
                    final_polish_accepted_steps += 1
                    if final_polish_accepted_steps >= int(
                        settings.final_polish_max_accepted_steps
                    ):
                        last_accepted_step_capped = True
                        break
        pass_best_after, _ = best_item()
        if last_accepted_step_capped:
            break
        if not improved_in_pass or pass_best_after == pass_best_before:
            break
        if polish_pass == int(settings.final_polish_max_passes):
            last_accepted_step_capped = True

    if last_accepted_step_capped:
        raise RuntimeError(
            "vector-v3 final polish ended on a capped accepted step; refusing "
            "an uncertified selection"
        )
    polish_end_count = int(len(cache))

    axis_names = ("rho",) + keys

    def certificate_axis(
        center: np.ndarray,
        coordinate: int,
    ) -> tuple[list[tuple[np.ndarray, str]], list[np.ndarray], bool]:
        x = float(center[coordinate])
        h = float(settings.certificate_log_step)
        tolerance = 2e-13
        at_lower = x <= float(lower[coordinate]) + tolerance
        at_upper = x >= float(upper[coordinate]) - tolerance
        axial: list[tuple[np.ndarray, str]] = []
        primary: list[np.ndarray] = []

        def add(value: float, label: str, *, mixed_primary: bool) -> None:
            point = center.copy()
            point[coordinate] = min(
                max(float(value), float(lower[coordinate])),
                float(upper[coordinate]),
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
            bracketed = len(axial) == 2 and all(
                point[coordinate] > x + tolerance for point, _ in axial
            )
        elif at_upper:
            add(x - h, "one_sided_inward_1", mixed_primary=True)
            add(x - 2.0 * h, "one_sided_inward_2", mixed_primary=False)
            bracketed = len(axial) == 2 and all(
                point[coordinate] < x - tolerance for point, _ in axial
            )
        else:
            add(x - h, "two_sided_minus", mixed_primary=True)
            add(x + h, "two_sided_plus", mixed_primary=True)
            bracketed = bool(
                any(point[coordinate] < x - tolerance for point, _ in axial)
                and any(point[coordinate] > x + tolerance for point, _ in axial)
            )
        return axial, primary, bracketed

    def projected_hadamard_point(
        center: np.ndarray,
        gamma_direction: tuple[int, ...],
        orientation: int,
        magnitude: float,
    ) -> np.ndarray:
        point = center.copy()
        for gamma_index, sign in enumerate(gamma_direction, start=1):
            desired = float(orientation * sign)
            candidate = float(center[gamma_index]) + desired * magnitude
            point[gamma_index] = min(
                max(candidate, float(lower[gamma_index])),
                float(upper[gamma_index]),
            )
        return point

    recovery_evaluation_counts: dict[str, int] = {}
    recovery_sweep_count = 0
    recovery_coordinate_attempts = 0
    recovery_hadamard_attempts = 0

    def recovery_sweep(
        recovery_index: int,
        *,
        evaluation_limit: int,
    ) -> None:
        ""

        nonlocal recovery_sweep_count
        nonlocal recovery_coordinate_attempts
        nonlocal recovery_hadamard_attempts
        start_count = int(len(cache))
        start_key, start_f = best_item()
        for coordinate in range(dimension):
            recovery_coordinate_attempts += 1
            current_key, current_f = best_item()
            local_limit = min(int(evaluation_limit), int(len(cache) + 4))
            coordinate_step(
                np.asarray(current_key, dtype=float),
                current_f,
                coordinate=coordinate,
                evaluation_limit=local_limit,
                optimizer_phase="certificate_recovery",
                pass_index=recovery_index,
                start_index=0,
            )
        for direction_index, gamma_direction in enumerate(hadamard, start=1):
            recovery_hadamard_attempts += 1
            current_key, current_f = best_item()
            local_limit = min(int(evaluation_limit), int(len(cache) + 4))
            structured_direction_step(
                np.asarray(current_key, dtype=float),
                current_f,
                direction=np.asarray([0.0] + list(gamma_direction), dtype=float),
                direction_index=direction_index,
                evaluation_limit=local_limit,
                optimizer_phase="certificate_recovery",
                pass_index=recovery_index,
            )
        end_key, end_f = best_item()
        evaluation_count = int(len(cache) - start_count)
        if evaluation_count > recovery_sweep_limit:
            raise RuntimeError("vector-v3 recovery exceeded its preregistered reserve")
        recovery_sweep_count += 1
        recovery_evaluation_counts[str(recovery_index)] = evaluation_count
        rows.append(
            {
                "optimizer_event": "recovery_diagnostic",
                "stage": "certificate_recovery",
                "optimizer_phase": "certificate_recovery",
                "joint_risk": float(end_f),
                "cache_hit": True,
                **point_columns(end_key),
                "recovery_index": int(recovery_index),
                "recovery_start_risk": float(start_f),
                "recovery_start_point_json": json.dumps(
                    point_columns(start_key), sort_keys=True, separators=(",", ":")
                ),
                "recovery_evaluation_count": evaluation_count,
                "recovery_coordinate_attempts": dimension,
                "recovery_hadamard_attempts": len(hadamard),
                "recovery_complete": True,
            }
        )

    convergence_status = ""
    axis_bracketed = {name: False for name in axis_names}
    rho_gamma_mixed_checked = {name: False for name in keys}
    rho_gamma_mixed_corner_counts = {name: 0 for name in keys}
    hadamard_checked = False
    hadamard_per_direction_checked = {
        str(index): False for index in range(1, len(hadamard) + 1)
    }
    hadamard_distinct_neighbor_counts = {
        str(index): 0 for index in range(1, len(hadamard) + 1)
    }
    certificate_attempts = 0
    certified_certificate_index = 0
    verification_improvement_count = 0
    certificate_seen_centers: set[tuple[float, ...]] = set()
    certificate_evaluation_counts: dict[str, int] = {}

    for certificate_index in range(1, certificate_attempt_count_limit + 1):
        certificate_stage_limit = int(
            exploration_evaluation_count
            + initial_polish_limit
            + certificate_index * certificate_attempt_limit
            + (certificate_index - 1) * recovery_sweep_limit
        )
        if certificate_stage_limit > final_stage_limit:
            raise RuntimeError("vector-v3 certificate allocation exceeds final reserve")
        certificate_start_count = int(len(cache))
        center_key, center_f = best_item()
        if center_key in certificate_seen_centers:
            raise RuntimeError(
                "vector-v3 certificate revisited a center; refusing an uncertified selection"
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
            raise RuntimeError("vector-v3 certificate could not bracket every axis")

        mixed: list[tuple[np.ndarray, str, str]] = []
        for gamma_index, name in enumerate(keys, start=1):
            per_key: list[np.ndarray] = []
            for rho_direction in primary_by_coordinate[0]:
                for gamma_direction in primary_by_coordinate[gamma_index]:
                    point = center.copy()
                    point[0] = rho_direction[0]
                    point[gamma_index] = gamma_direction[gamma_index]
                    if point_key(point) != center_key and point_key(point) not in {
                        point_key(existing) for existing in per_key
                    }:
                        per_key.append(point)
            rho_gamma_mixed_corner_counts[name] = len(per_key)
            rho_gamma_mixed_checked[name] = bool(per_key)
            mixed.extend(
                (point, name, f"rho_x_gamma_{corner_index}")
                for corner_index, point in enumerate(per_key, start=1)
            )
        if not all(rho_gamma_mixed_checked.values()):
            raise RuntimeError(
                "vector-v3 certificate lacks a feasible rho-by-gamma mixed corner"
            )

        hadamard_points: list[tuple[np.ndarray, int, str]] = []
        hadamard_direction_complete: dict[int, bool] = {}
        hadamard_direction_neighbor_counts: dict[int, int] = {}
        h = float(settings.certificate_log_step)
        for direction_index, gamma_direction in enumerate(hadamard, start=1):
                                                                             
                                                                              
                                                                        
                                                                        
                                                                              
            first = projected_hadamard_point(center, gamma_direction, 1, h)
            second = projected_hadamard_point(center, gamma_direction, -1, h)
            first_label = "plus"
            second_label = "minus"
            if point_key(first) == center_key and point_key(second) != center_key:
                first = projected_hadamard_point(
                    center, gamma_direction, -1, 2.0 * h
                )
                first_label = "minus_one_sided_2"
            if point_key(second) == center_key and point_key(first) != center_key:
                second = projected_hadamard_point(
                    center, gamma_direction, 1, 2.0 * h
                )
                second_label = "plus_one_sided_2"
            if point_key(second) == point_key(first):
                second = projected_hadamard_point(
                    center, gamma_direction, -1, 2.0 * h
                )
                second_label = "minus_one_sided_2"
            points = []
            for point, label in ((first, first_label), (second, second_label)):
                if point_key(point) != center_key and point_key(point) not in {
                    point_key(existing) for existing, _ in points
                }:
                    points.append((point, label))
            hadamard_direction_complete[direction_index] = len(points) == 2
            hadamard_direction_neighbor_counts[direction_index] = len(points)
            hadamard_points.extend(
                (point, direction_index, label) for point, label in points
            )
        hadamard_per_direction_checked = {
            str(index): bool(hadamard_direction_complete.get(index, False))
            for index in range(1, len(hadamard) + 1)
        }
        hadamard_distinct_neighbor_counts = {
            str(index): int(hadamard_direction_neighbor_counts.get(index, 0))
            for index in range(1, len(hadamard) + 1)
        }
        hadamard_checked = (
            set(hadamard_direction_complete) == set(range(1, len(hadamard) + 1))
            and all(hadamard_per_direction_checked.values())
            and all(
                count == 2
                for count in hadamard_distinct_neighbor_counts.values()
            )
        )
        if not hadamard_checked:
            raise RuntimeError(
                "vector-v3 certificate could not check every structured Hadamard direction"
            )

        required_points = [
            point
            for axial in axial_by_coordinate
            for point, _ in axial
        ] + [point for point, _, _ in mixed] + [
            point for point, _, _ in hadamard_points
        ]
        if not has_budget_for(
            *required_points,
            evaluation_limit=certificate_stage_limit,
        ):
            raise RuntimeError(
                "vector-v3 final budget cannot complete all required certificate checks"
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
                        certificate_log_step=h,
                        coordinate=axis,
                        stencil=stencil,
                    )
                )
        for point, name, stencil in mixed:
            certificate_values.append(
                evaluate(
                    point,
                    "certificate_rho_gamma_mixed_corner",
                    optimizer_phase="local_discrete_certificate",
                    certificate_index=certificate_index,
                    certificate_log_step=h,
                    gamma_key=name,
                    stencil=stencil,
                )
            )
        for point, direction_index, orientation in hadamard_points:
            certificate_values.append(
                evaluate(
                    point,
                    "certificate_structured_hadamard",
                    optimizer_phase="local_discrete_certificate",
                    certificate_index=certificate_index,
                    certificate_log_step=h,
                    direction_index=direction_index,
                    stencil=f"hadamard_{direction_index}_{orientation}",
                )
            )
        if not all(math.isfinite(value) for value in certificate_values):
            raise RuntimeError("vector-v3 certificate contains non-finite risk")
        certificate_evaluation_count = int(len(cache) - certificate_start_count)
        if certificate_evaluation_count > certificate_attempt_limit:
            raise RuntimeError("vector-v3 certificate exceeded its per-attempt reserve")
        certificate_evaluation_counts[str(certificate_index)] = (
            certificate_evaluation_count
        )

        verified_key, verified_f = best_item()
        verified = verified_key == center_key
        diagnostic = {
            "optimizer_event": "certificate_diagnostic",
            "stage": "local_discrete_certificate",
            "optimizer_phase": "local_discrete_certificate",
            "joint_risk": float(center_f),
            "cache_hit": True,
            **point_columns(center_key),
            "certificate_index": certificate_index,
            "certificate_log_step": h,
            "all_axes_bracketed": bool(all(axis_bracketed.values())),
            "all_rho_gamma_mixed_checked": bool(
                all(rho_gamma_mixed_checked.values())
            ),
            "hadamard_checked": bool(hadamard_checked),
            "hadamard_direction_count": len(hadamard),
            "hadamard_direction_bank_sha256": structured_hadamard_bank_sha256(
                len(keys)
            ),
            "hadamard_per_direction_checked_json": json.dumps(
                hadamard_per_direction_checked,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "hadamard_distinct_neighbor_counts_json": json.dumps(
                hadamard_distinct_neighbor_counts,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "certificate_neighbor_count": len({point_key(p) for p in required_points}),
            "certificate_evaluation_count": certificate_evaluation_count,
            "certificate_winner_risk": float(verified_f),
            "certificate_verified": bool(verified),
        }
        winner_columns = point_columns(verified_key)
        diagnostic.update(
            {f"certificate_winner_{name}": value for name, value in winner_columns.items()}
        )
        rows.append(diagnostic)
        if verified:
            convergence_status = VECTOR_CONVERGENCE_STATUS
            certified_certificate_index = int(certificate_index)
            break
        verification_improvement_count += 1
        if certificate_index >= certificate_attempt_count_limit:
            raise RuntimeError(
                "vector-v3 certificate kept finding improvements after the restart cap"
            )
        recovery_stage_limit = int(
            exploration_evaluation_count
            + initial_polish_limit
            + certificate_index * certificate_attempt_limit
            + certificate_index * recovery_sweep_limit
        )
        if recovery_stage_limit > final_stage_limit:
            raise RuntimeError("vector-v3 recovery allocation exceeds final reserve")
        recovery_sweep(
            certificate_index,
            evaluation_limit=recovery_stage_limit,
        )

    if convergence_status != VECTOR_CONVERGENCE_STATUS:
        raise RuntimeError("vector-v3 did not obtain its required final certificate")

    best_key, best_f = best_item()
    rho, gammas = decoded(best_key)
    warm_key = point_key(warm)
    if warm_key not in cache or not math.isfinite(float(cache[warm_key])):
        raise RuntimeError("vector-v3 certified shared warm-start risk is unavailable")
    warm_objective = float(cache[warm_key])
    selected_minus_warm_delta = float(best_f - warm_objective)
    if selected_minus_warm_delta > float(settings.exact_tie_tolerance):
        raise RuntimeError(
            "vector-v3 selected risk is worse than its certified shared warm start"
        )
    final_polish_evaluation_count = int(
        polish_end_count - exploration_evaluation_count
    )
    recovery_evaluation_count = int(sum(recovery_evaluation_counts.values()))
    certificate_evaluation_count = int(sum(certificate_evaluation_counts.values()))
    expected_phase_counts = {
        "global_exploration": int(exploration_evaluation_count),
        "best_first_polish": final_polish_evaluation_count,
        "certificate_recovery": recovery_evaluation_count,
        "local_discrete_certificate": certificate_evaluation_count,
    }
    if set(unique_evaluations_by_phase) - set(expected_phase_counts):
        raise RuntimeError("vector-v3 recorded an unknown exact-replay phase")
    if any(
        int(unique_evaluations_by_phase.get(phase, 0)) != count
        for phase, count in expected_phase_counts.items()
    ):
        raise RuntimeError("vector-v3 phase counters differ from exact replay events")
    if sum(expected_phase_counts.values()) != len(cache):
        raise RuntimeError("vector-v3 phase counters do not sum to total evaluations")
    if final_polish_evaluation_count > initial_polish_limit:
        raise RuntimeError("vector-v3 initial polish exceeded its preregistered reserve")
    if recovery_evaluation_count > int(
        final_decomposition["recovery_evaluation_reserve"]
    ) or certificate_evaluation_count > int(
        final_decomposition["certificate_evaluation_reserve"]
    ):
        raise RuntimeError("vector-v3 recovery/certificate reserve was exceeded")
    if (
        certificate_attempts != verification_improvement_count + 1
        or recovery_sweep_count != verification_improvement_count
        or recovery_sweep_count > recovery_sweep_count_limit
        or certified_certificate_index != certificate_attempts
        or recovery_coordinate_attempts != recovery_sweep_count * dimension
        or recovery_hadamard_attempts != recovery_sweep_count * len(hadamard)
    ):
        raise RuntimeError("vector-v3 certificate recovery counters are inconsistent")
    expected_direction_keys = {
        str(index) for index in range(1, len(hadamard) + 1)
    }
    if (
        set(hadamard_per_direction_checked) != expected_direction_keys
        or set(hadamard_distinct_neighbor_counts) != expected_direction_keys
        or not all(hadamard_per_direction_checked.values())
        or any(
            count != 2 for count in hadamard_distinct_neighbor_counts.values()
        )
    ):
        raise RuntimeError("vector-v3 final Hadamard certificate is incomplete")
    trace = pd.DataFrame(rows)
    trace["selected"] = False
    mask = trace["optimizer_event"].eq("evaluation")
    mask &= pd.to_numeric(trace["log_rho"], errors="coerce").sub(best_key[0]).abs().le(1e-13)
    for index, name in enumerate(keys):
        mask &= (
            pd.to_numeric(trace[f"log_gamma__{name}"], errors="coerce")
            .sub(best_key[index + 1])
            .abs()
            .le(1e-13)
        )
    if bool(mask.any()):
        trace.loc[trace.index[mask][-1], "selected"] = True
    return VectorJointRhoGammaNewtonResult(
        rho=float(rho),
        gammas=dict(gammas),
        objective=float(best_f),
        trace=trace,
        gamma_keys=keys,
        dimension=dimension,
        max_evaluations=int(budget.max_evaluations),
        exploration_evaluation_limit=int(budget.exploration_evaluation_limit),
        final_reserve_evaluation_limit=int(budget.final_reserve_evaluation_limit),
        initial_final_polish_evaluation_limit=initial_polish_limit,
        certificate_attempt_evaluation_limit=certificate_attempt_limit,
        certificate_attempt_limit=certificate_attempt_count_limit,
        certificate_evaluation_reserve=int(
            final_decomposition["certificate_evaluation_reserve"]
        ),
        recovery_sweep_evaluation_limit=recovery_sweep_limit,
        recovery_sweep_limit=recovery_sweep_count_limit,
        recovery_evaluation_reserve=int(
            final_decomposition["recovery_evaluation_reserve"]
        ),
        evaluation_count=int(len(cache)),
        exploration_evaluation_count=int(exploration_evaluation_count),
        final_polish_evaluation_count=final_polish_evaluation_count,
        recovery_evaluation_count=recovery_evaluation_count,
        certificate_evaluation_count=certificate_evaluation_count,
        certificate_evaluation_counts=dict(certificate_evaluation_counts),
        recovery_evaluation_counts=dict(recovery_evaluation_counts),
        fallback_count=int(fallback_count),
        structured_direction_attempts=int(structured_attempts),
        structured_direction_accepts=int(structured_accepts),
        convergence_status=convergence_status,
        axis_bracketed=dict(axis_bracketed),
        rho_gamma_mixed_checked=dict(rho_gamma_mixed_checked),
        rho_gamma_mixed_corner_counts=dict(rho_gamma_mixed_corner_counts),
        hadamard_checked=bool(hadamard_checked),
        hadamard_direction_count=len(hadamard),
        hadamard_direction_bank=tuple(hadamard),
        hadamard_direction_bank_sha256=structured_hadamard_bank_sha256(len(keys)),
        hadamard_per_direction_checked=dict(hadamard_per_direction_checked),
        hadamard_distinct_neighbor_counts=dict(
            hadamard_distinct_neighbor_counts
        ),
        certificate_attempts=int(certificate_attempts),
        certified_certificate_index=int(certified_certificate_index),
        verification_improvement_count=int(verification_improvement_count),
        recovery_sweep_count=int(recovery_sweep_count),
        recovery_coordinate_attempts=int(recovery_coordinate_attempts),
        recovery_hadamard_attempts=int(recovery_hadamard_attempts),
        final_polish_accepted_steps=int(final_polish_accepted_steps),
        last_accepted_step_capped=bool(last_accepted_step_capped),
        shared_warm_start=shared_warm_start,
        shared_warm_start_objective=warm_objective,
        selected_minus_warm_delta=selected_minus_warm_delta,
    )


class JointRhoGammaVectorSelector:
    ""

    def __init__(
        self,
        replay: ExactValidationReplay,
        *,
        scales: Mapping[str, float],
        settings: VectorJointRhoGammaSelectionSettings,
        shared_warm_start: CertifiedSharedWarmStart,
        reference_transformed: Mapping[str, float] | None = None,
        normalization_scales: Mapping[str, float] | None = None,
        retain_replay_artifacts: bool = True,
    ) -> None:
        if not scales or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in scales.values()
        ):
            raise ValueError("vector-v3 fixed sigma map must be finite and positive")
        self.replay = replay
        self.scales = {str(key): float(value) for key, value in sorted(scales.items())}
        self.settings = settings.validate_for_keys(
            tuple(self.scales), shared_warm_start
        )
        self.shared_warm_start = shared_warm_start
        if (reference_transformed is None) != (normalization_scales is None):
            raise ValueError(
                "reference_transformed and normalization_scales must both be "
                "provided or both be computed from validation anchors"
            )
        self.reference_transformed = (
            None
            if reference_transformed is None
            else _metric_map(
                reference_transformed,
                name="reference_transformed",
                strictly_positive=False,
            )
        )
        self.normalization_scales = (
            None
            if normalization_scales is None
            else _metric_map(
                normalization_scales,
                name="normalization_scales",
                strictly_positive=True,
            )
        )
        self.retain_replay_artifacts = bool(retain_replay_artifacts)

    def state(self, rho: float, gammas: Mapping[str, float]) -> ParameterState:
        if set(gammas) != set(self.scales):
            raise ValueError("vector-v3 gamma map must exactly match the sigma keys")
        selected_rho, selected_gammas = canonical_parameter_point(
            rho,
            gammas,
            gamma_keys=tuple(self.scales),
            rho_bounds=self.settings.rho_bounds,
            gamma_bounds=self.settings.gamma_bounds,
        )
        return ParameterState(
            family=MOMENT_T,
            scales=dict(self.scales),
            gammas=selected_gammas,
            nus={key: 5.0 for key in self.scales},
            rho=selected_rho,
        )

    def _validation_normalization(
        self,
        pilot_metrics: list[dict[str, float]],
        reference_metrics: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
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

    def select(self, *, variant: str) -> JointRhoGammaVectorSelectionOutcome:
        if variant not in FILTER_VARIANTS:
            raise ValueError(f"unknown filtering variant {variant!r}")
        keys = tuple(self.scales)
        artifacts_by_point: dict[tuple[float, ...], ReplayArtifacts] = {}
        metrics_by_point: dict[tuple[float, ...], dict[str, float]] = {}
        evaluation_rows: list[dict[str, object]] = []

        def key_for(rho: float, gammas: Mapping[str, float]) -> tuple[float, ...]:
            canonical_rho, canonical_gammas = canonical_parameter_point(
                rho,
                gammas,
                gamma_keys=keys,
                rho_bounds=self.settings.rho_bounds,
                gamma_bounds=self.settings.gamma_bounds,
            )
            return (canonical_rho,) + tuple(
                canonical_gammas[name] for name in keys
            )

        def decoded_key(point: tuple[float, ...]) -> tuple[float, dict[str, float]]:
            return float(point[0]), {
                name: float(point[index + 1]) for index, name in enumerate(keys)
            }

        def metrics_at(rho: float, gammas: Mapping[str, float]) -> dict[str, float]:
            key = key_for(rho, gammas)
            if key not in metrics_by_point:
                selected_rho, selected_gammas = decoded_key(key)
                artifact = self.replay.evaluate(
                    self.state(selected_rho, selected_gammas), variant=variant
                )
                if self.retain_replay_artifacts:
                    artifacts_by_point[key] = artifact
                metrics_by_point[key] = dict(artifact.metrics)
            return metrics_by_point[key]

        reference_gammas = {
            name: float(self.settings.reference_gamma) for name in keys
        }
        if self.reference_transformed is None:
            pilot_metrics = [
                metrics_at(float(rho), reference_gammas)
                for rho in self.settings.anchor_rhos
            ]
            reference_metrics = metrics_at(
                float(self.settings.reference_rho), reference_gammas
            )
            reference_transformed, normalization_scales = (
                self._validation_normalization(pilot_metrics, reference_metrics)
            )
        else:
            assert self.normalization_scales is not None
            reference_transformed = dict(self.reference_transformed)
            normalization_scales = dict(self.normalization_scales)

        def objective(rho: float, gammas: Mapping[str, float]) -> float:
            point = key_for(rho, gammas)
            selected_rho, selected_gammas = decoded_key(point)
            metrics = metrics_at(selected_rho, selected_gammas)
            transformed = transformed_joint_metrics(
                metrics,
                metric_epsilon=self.settings.metric_epsilon,
                coverage_target=self.settings.coverage_target,
                coverage_tolerance=self.settings.coverage_tolerance,
                coverage_upper_weight=self.settings.coverage_upper_weight,
            )
            z = {
                name: (
                    float(transformed[name]) - float(reference_transformed[name])
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
            row: dict[str, object] = {
                "variant": variant,
                "bridge_family": MOMENT_T,
                "rho": selected_rho,
                "gammas_json": json.dumps(
                    selected_gammas, sort_keys=True, separators=(",", ":")
                ),
                "joint_risk": risk,
                **{name: float(value) for name, value in metrics.items()},
                **{f"z_{name}": float(value) for name, value in z.items()},
            }
            row.update(
                {f"gamma__{name}": float(value) for name, value in selected_gammas.items()}
            )
            evaluation_rows.append(row)
            return risk

        optimizer = safeguarded_bounded_newton_logrho_gamma_vector(
            objective,
            gamma_keys=keys,
            shared_warm_start=self.shared_warm_start,
            settings=self.settings,
        )
        selected_key = key_for(optimizer.rho, optimizer.gammas)
        selected_rho, selected_gammas = decoded_key(selected_key)
        if optimizer.rho != selected_rho or optimizer.gammas != selected_gammas:
            raise RuntimeError(
                "vector optimizer/result state is not the canonical parameter point"
            )
        selected_metrics = metrics_at(selected_rho, selected_gammas)
        selected_artifacts = (
            artifacts_by_point[selected_key]
            if self.retain_replay_artifacts
            else self.replay.evaluate(
                self.state(selected_rho, selected_gammas), variant=variant
            )
        )

        metric_trace = pd.DataFrame(evaluation_rows).drop_duplicates(
            ["variant", "rho"] + [f"gamma__{name}" for name in keys],
            keep="last",
        )
        trace = optimizer.trace.copy()
        join_columns: list[str] = []
        trace["rho_join"] = pd.to_numeric(trace["rho"], errors="coerce").round(12)
        metric_trace["rho_join"] = pd.to_numeric(
            metric_trace["rho"], errors="coerce"
        ).round(12)
        join_columns.append("rho_join")
        for name in keys:
            join = f"gamma_join__{name}"
            trace[join] = pd.to_numeric(
                trace[f"gamma__{name}"], errors="coerce"
            ).round(12)
            metric_trace[join] = pd.to_numeric(
                metric_trace[f"gamma__{name}"], errors="coerce"
            ).round(12)
            join_columns.append(join)
        metric_payload = metric_trace.drop(
            columns=["variant", "rho", "gammas_json"]
            + [f"gamma__{name}" for name in keys]
        )
        trace = trace.merge(
            metric_payload,
            on=join_columns,
            how="left",
            suffixes=("_optimizer", ""),
        ).drop(columns=join_columns)
        if "joint_risk_optimizer" in trace:
            if "joint_risk" in trace:
                trace["joint_risk"] = pd.to_numeric(
                    trace["joint_risk"], errors="coerce"
                ).fillna(
                    pd.to_numeric(trace["joint_risk_optimizer"], errors="coerce")
                )
            else:
                trace["joint_risk"] = trace["joint_risk_optimizer"]
        return JointRhoGammaVectorSelectionOutcome(
            variant=variant,
            selected_state=self.state(selected_rho, selected_gammas),
            selected_metrics=dict(selected_metrics),
            selected_objective=float(optimizer.objective),
            reference_transformed=dict(reference_transformed),
            normalization_scales=dict(normalization_scales),
            trace=trace,
            replay_artifacts=selected_artifacts,
            optimizer=optimizer,
        )
