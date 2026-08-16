""












from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .joint_selection import (
    DRAW_KERNEL_T,
    FILTER_VARIANTS,
    JOINT_METRICS,
    ExactValidationReplay,
    ParameterState,
    ReplayArtifacts,
    transformed_joint_metrics,
)
from .rho_gamma_joint import (
    JOINT_RHO_ANCHOR_BANK,
    JointRhoGammaOptimizationFailure,
    JointRhoGammaSelectionSettings,
    safeguarded_bounded_newton_logrho_gamma,
)


RHO_TAU_OBJECTIVE_WEIGHTS = {
    "nll": 0.15,
    "short_rmse": 0.25,
    "long_rmse": 0.25,
    "mae": 0.10,
    "wis": 0.15,
    "coverage_penalty": 0.10,
}

KAPPA_TAU_ANCHORS = (
    0.50,
    1.0 / math.sqrt(2.0),
    1.0,
    math.sqrt(2.0),
    2.0,
)


def _replace_coordinate_name(value: str) -> str:
    ""

    return str(value).replace("gamma", "kappa_tau")


def _public_trace(trace: pd.DataFrame) -> pd.DataFrame:
    ""

    renamed = trace.rename(
        columns={column: _replace_coordinate_name(column) for column in trace.columns}
    ).copy()
    for column in renamed.select_dtypes(include=["object", "string"]).columns:
        renamed[column] = renamed[column].map(
            lambda value: (
                _replace_coordinate_name(value) if isinstance(value, str) else value
            )
        )
    return renamed


def _public_value(value):
    if isinstance(value, str):
        return _replace_coordinate_name(value)
    if isinstance(value, Mapping):
        return {
            _replace_coordinate_name(str(key)): _public_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_public_value(item) for item in value)
    return value


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


@dataclass(frozen=True)
class JointRhoTauSelectionSettings:
    ""

    objective_weights: dict[str, float] = field(
        default_factory=lambda: dict(RHO_TAU_OBJECTIVE_WEIGHTS)
    )
    rho_bounds: tuple[float, float] = (0.001, 0.5)
    kappa_tau_bounds: tuple[float, float] = (0.5, 2.0)
    anchor_rhos: tuple[float, ...] = JOINT_RHO_ANCHOR_BANK[:9]
    anchor_kappa_taus: tuple[float, ...] = KAPPA_TAU_ANCHORS
    reference_rho: float = 0.50
    reference_kappa_tau: float = 1.0
    fixed_nu: float = 5.0
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

    def _numerical_settings(self) -> JointRhoGammaSelectionSettings:
        return JointRhoGammaSelectionSettings(
            objective_weights=dict(self.objective_weights),
            rho_bounds=tuple(float(value) for value in self.rho_bounds),
            gamma_bounds=tuple(float(value) for value in self.kappa_tau_bounds),
            anchor_rhos=tuple(float(value) for value in self.anchor_rhos),
            anchor_gammas=tuple(float(value) for value in self.anchor_kappa_taus),
            reference_rho=float(self.reference_rho),
            reference_gamma=float(self.reference_kappa_tau),
            multi_starts=int(self.multi_starts),
            coordinate_passes=int(self.coordinate_passes),
            trust_region_passes=int(self.trust_region_passes),
            finite_difference_log_step=float(self.finite_difference_log_step),
            minimum_difference_log_step=float(self.minimum_difference_log_step),
            maximum_newton_log_step=float(self.maximum_newton_log_step),
            maximum_trust_log_step=float(self.maximum_trust_log_step),
            max_evaluations=int(self.max_evaluations),
            exploration_evaluation_limit=int(self.exploration_evaluation_limit),
            final_polish_evaluation_limit=int(self.final_polish_evaluation_limit),
            final_polish_max_passes=int(self.final_polish_max_passes),
            final_polish_max_accepted_steps=int(
                self.final_polish_max_accepted_steps
            ),
            certificate_max_restarts=int(self.certificate_max_restarts),
            certificate_evaluation_reserve=int(
                self.certificate_evaluation_reserve
            ),
            certificate_recovery_max_passes=int(
                self.certificate_recovery_max_passes
            ),
            certificate_recovery_evaluation_reserve=int(
                self.certificate_recovery_evaluation_reserve
            ),
            certificate_recovery_evaluation_limit_per_sweep=int(
                self.certificate_recovery_evaluation_limit_per_sweep
            ),
            certificate_log_step=float(self.certificate_log_step),
            max_backtracks=int(self.max_backtracks),
            hessian_floor=float(self.hessian_floor),
            trust_condition_limit=float(self.trust_condition_limit),
            x_tolerance=float(self.x_tolerance),
            objective_tolerance=float(self.objective_tolerance),
            exact_tie_tolerance=float(self.exact_tie_tolerance),
            metric_epsilon=float(self.metric_epsilon),
            coverage_target=float(self.coverage_target),
            coverage_tolerance=float(self.coverage_tolerance),
            coverage_upper_weight=float(self.coverage_upper_weight),
            robust_scale_floor=float(self.robust_scale_floor),
            robust_relative_floor=float(self.robust_relative_floor),
        )

    def validate(self) -> "JointRhoTauSelectionSettings":
        fixed_nu = float(self.fixed_nu)
        if math.isnan(fixed_nu) or fixed_nu <= 0.0:
            raise ValueError("fixed_nu must be positive or infinity")
        try:
            self._numerical_settings().validate()
        except ValueError as exc:
            raise ValueError(_replace_coordinate_name(str(exc))) from exc
        return self

    def search_anchor_points(self) -> tuple[tuple[float, float], ...]:
        settings = self.validate()
        rho_lower, rho_upper = (float(value) for value in settings.rho_bounds)
        kappa_lower, kappa_upper = (
            float(value) for value in settings.kappa_tau_bounds
        )
        points = {
            *(
                (float(rho), float(settings.reference_kappa_tau))
                for rho in settings.anchor_rhos
            ),
            *(
                (float(settings.reference_rho), float(kappa))
                for kappa in settings.anchor_kappa_taus
            ),
            (rho_lower, kappa_lower),
            (rho_lower, kappa_upper),
            (rho_upper, kappa_lower),
            (rho_upper, kappa_upper),
        }
        return tuple(sorted(points))


@dataclass
class JointRhoTauNewtonResult:
    rho: float
    kappa_tau: float
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
    best_first_polish_evaluation_limit: int
    best_first_polish_evaluation_count: int
    certificate_evaluation_reserve: int
    certificate_evaluation_count: int
    certificate_max_attempts: int
    certificate_max_neighbors_per_attempt: int
    certificate_worst_case_evaluations: int
    certificate_evaluation_counts: dict[str, int]
    certified_certificate_index: int
    recovery_sweep_evaluation_limit: int
    recovery_sweep_limit: int
    recovery_evaluation_reserve: int
    recovery_evaluation_count: int
    recovery_evaluation_counts: dict[str, int]
    recovery_sweep_count: int
    recovery_pass_count: int
    recovery_pass_counts: dict[str, int]
    recovery_coordinate_attempts: int
    recovery_accepted_steps: int
    recovery_last_pass_improved: dict[str, bool]
    phase_evaluation_counts: dict[str, int]


class JointRhoTauOptimizationFailure(RuntimeError):
    ""

    def __init__(
        self,
        message: str,
        *,
        failure_code: str,
        trace: pd.DataFrame,
        metadata: Mapping[str, object],
    ) -> None:
        super().__init__(_replace_coordinate_name(message))
        self.failure_code = _replace_coordinate_name(failure_code)
        self.trace = _public_trace(trace)
        self.metadata = _public_value(metadata)


def safeguarded_bounded_newton_logrho_kappa_tau(
    objective: Callable[[float, float], float],
    *,
    settings: JointRhoTauSelectionSettings,
) -> JointRhoTauNewtonResult:
    ""

    settings = settings.validate()
    try:
        private = safeguarded_bounded_newton_logrho_gamma(
            objective,
            settings=settings._numerical_settings(),
        )
    except JointRhoGammaOptimizationFailure as exc:
        raise JointRhoTauOptimizationFailure(
            str(exc),
            failure_code=exc.failure_code,
            trace=exc.trace,
            metadata=exc.metadata,
        ) from exc
    return JointRhoTauNewtonResult(
        rho=float(private.rho),
        kappa_tau=float(private.gamma),
        objective=float(private.objective),
        trace=_public_trace(private.trace),
        evaluation_count=int(private.evaluation_count),
        fallback_count=int(private.fallback_count),
        trust_region_attempts=int(private.trust_region_attempts),
        trust_region_accepts=int(private.trust_region_accepts),
        exploration_evaluation_limit=int(private.exploration_evaluation_limit),
        exploration_evaluation_count=int(private.exploration_evaluation_count),
        final_polish_evaluation_count=int(private.final_polish_evaluation_count),
        convergence_status=str(private.convergence_status),
        axis_bracketed={
            _replace_coordinate_name(key): bool(value)
            for key, value in private.axis_bracketed.items()
        },
        mixed_checked=bool(private.mixed_checked),
        last_accepted_step_capped=bool(private.last_accepted_step_capped),
        verification_improvement_count=int(private.verification_improvement_count),
        final_polish_accepted_steps=int(private.final_polish_accepted_steps),
        certificate_attempts=int(private.certificate_attempts),
        best_first_polish_evaluation_limit=int(
            private.best_first_polish_evaluation_limit
        ),
        best_first_polish_evaluation_count=int(
            private.best_first_polish_evaluation_count
        ),
        certificate_evaluation_reserve=int(private.certificate_evaluation_reserve),
        certificate_evaluation_count=int(private.certificate_evaluation_count),
        certificate_max_attempts=int(private.certificate_max_attempts),
        certificate_max_neighbors_per_attempt=int(
            private.certificate_max_neighbors_per_attempt
        ),
        certificate_worst_case_evaluations=int(
            private.certificate_worst_case_evaluations
        ),
        certificate_evaluation_counts=dict(private.certificate_evaluation_counts),
        certified_certificate_index=int(private.certified_certificate_index),
        recovery_sweep_evaluation_limit=int(
            private.recovery_sweep_evaluation_limit
        ),
        recovery_sweep_limit=int(private.recovery_sweep_limit),
        recovery_evaluation_reserve=int(private.recovery_evaluation_reserve),
        recovery_evaluation_count=int(private.recovery_evaluation_count),
        recovery_evaluation_counts=dict(private.recovery_evaluation_counts),
        recovery_sweep_count=int(private.recovery_sweep_count),
        recovery_pass_count=int(private.recovery_pass_count),
        recovery_pass_counts=dict(private.recovery_pass_counts),
        recovery_coordinate_attempts=int(private.recovery_coordinate_attempts),
        recovery_accepted_steps=int(private.recovery_accepted_steps),
        recovery_last_pass_improved={
            _replace_coordinate_name(str(key)): bool(value)
            for key, value in private.recovery_last_pass_improved.items()
        },
        phase_evaluation_counts=dict(private.phase_evaluation_counts),
    )


@dataclass
class JointRhoTauSelectionOutcome:
    variant: str
    selected_state: ParameterState
    selected_kappa_tau: float
    formula_sigma_by_component: dict[str, float]
    selected_tau_by_component: dict[str, float]
    selected_metrics: dict[str, float]
    selected_objective: float
    reference_metrics: dict[str, float]
    reference_transformed: dict[str, float]
    normalization_scales: dict[str, float]
    trace: pd.DataFrame
    replay_artifacts: ReplayArtifacts
    optimizer: JointRhoTauNewtonResult


class JointRhoTauSelector:
    ""

    def __init__(
        self,
        replay: ExactValidationReplay,
        *,
        formula_sigma_by_component: Mapping[str, float],
        reference_transformed: Mapping[str, float] | None = None,
        normalization_scales: Mapping[str, float] | None = None,
        settings: JointRhoTauSelectionSettings,
        retain_replay_artifacts: bool = True,
    ) -> None:
        if not formula_sigma_by_component or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in formula_sigma_by_component.values()
        ):
            raise ValueError("formula sigma values must be finite and positive")
        if (reference_transformed is None) != (normalization_scales is None):
            raise ValueError(
                "reference_transformed and normalization_scales must both be "
                "provided or both be computed from validation anchors"
            )
        self.replay = replay
        self.formula_sigma_by_component = {
            str(key): float(value)
            for key, value in sorted(formula_sigma_by_component.items())
        }
        self.reference_transformed = (
            _metric_map(
                reference_transformed,
                name="reference_transformed",
                strictly_positive=False,
            )
            if reference_transformed is not None
            else None
        )
        self.normalization_scales = (
            _metric_map(
                normalization_scales,
                name="normalization_scales",
                strictly_positive=True,
            )
            if normalization_scales is not None
            else None
        )
        self.settings = settings.validate()
        self.retain_replay_artifacts = bool(retain_replay_artifacts)

    def state(self, rho: float, kappa_tau: float) -> ParameterState:
        rho_lower, rho_upper = self.settings.rho_bounds
        kappa_lower, kappa_upper = self.settings.kappa_tau_bounds
        selected_rho = min(max(float(rho), float(rho_lower)), float(rho_upper))
        selected_kappa = min(
            max(float(kappa_tau), float(kappa_lower)), float(kappa_upper)
        )
        tau = {
            key: float(sigma) * float(selected_kappa)
            for key, sigma in self.formula_sigma_by_component.items()
        }
        return ParameterState(
            family=DRAW_KERNEL_T,
            scales=tau,
            gammas={},
            nus={key: float(self.settings.fixed_nu) for key in tau},
            rho=float(selected_rho),
            distribution="student_t",
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
                [row[name] for row in transformed if math.isfinite(row[name])],
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

    def select(self, *, variant: str) -> JointRhoTauSelectionOutcome:
        if variant not in FILTER_VARIANTS:
            raise ValueError(f"unknown filtering variant {variant!r}")
        artifacts_by_point: dict[tuple[float, float], ReplayArtifacts] = {}
        metrics_by_point: dict[tuple[float, float], dict[str, float]] = {}
        evaluation_rows: list[dict[str, object]] = []

        def point_key(rho: float, kappa_tau: float) -> tuple[float, float]:
            return round(float(rho), 12), round(float(kappa_tau), 12)

        def metrics_at(rho: float, kappa_tau: float) -> dict[str, float]:
            key = point_key(rho, kappa_tau)
            if key not in metrics_by_point:
                artifact = self.replay.evaluate(self.state(*key), variant=variant)
                if self.retain_replay_artifacts:
                    artifacts_by_point[key] = artifact
                metrics_by_point[key] = dict(artifact.metrics)
            return metrics_by_point[key]

        reference_metrics = metrics_at(
            self.settings.reference_rho,
            self.settings.reference_kappa_tau,
        )
        if self.reference_transformed is None:
            pilot_metrics = [
                metrics_at(rho, self.settings.reference_kappa_tau)
                for rho in self.settings.anchor_rhos
            ]
            reference_transformed, normalization_scales = (
                self._validation_normalization(pilot_metrics, reference_metrics)
            )
        else:
            assert self.normalization_scales is not None
            reference_transformed = dict(self.reference_transformed)
            normalization_scales = dict(self.normalization_scales)

        def objective(rho: float, kappa_tau: float) -> float:
            key = point_key(rho, kappa_tau)
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
                    "bridge_family": DRAW_KERNEL_T,
                    "rho": key[0],
                    "kappa_tau": key[1],
                    "joint_risk": risk,
                    **{name: float(value) for name, value in metrics.items()},
                    **{f"z_{name}": float(value) for name, value in z.items()},
                }
            )
            return risk

        optimizer = safeguarded_bounded_newton_logrho_kappa_tau(
            objective,
            settings=self.settings,
        )
        selected_key = point_key(optimizer.rho, optimizer.kappa_tau)
        selected_state = self.state(*selected_key)
        selected_metrics = metrics_at(*selected_key)
        selected_artifacts = (
            artifacts_by_point[selected_key]
            if self.retain_replay_artifacts
            else self.replay.evaluate(selected_state, variant=variant)
        )
        metric_trace = pd.DataFrame(evaluation_rows).drop_duplicates(
            ["variant", "rho", "kappa_tau"], keep="last"
        )
        trace = optimizer.trace.copy()
        trace["rho_join"] = pd.to_numeric(trace["rho"], errors="coerce").round(12)
        trace["kappa_tau_join"] = pd.to_numeric(
            trace["kappa_tau"], errors="coerce"
        ).round(12)
        metric_trace = metric_trace.rename(
            columns={"rho": "rho_join", "kappa_tau": "kappa_tau_join"}
        )
        trace = trace.merge(
            metric_trace,
            on=["rho_join", "kappa_tau_join"],
            how="left",
            suffixes=("_optimizer", ""),
        ).drop(columns=["rho_join", "kappa_tau_join"])
        if "joint_risk" not in trace:
            trace["joint_risk"] = trace["joint_risk_optimizer"]

        return JointRhoTauSelectionOutcome(
            variant=variant,
            selected_state=selected_state,
            selected_kappa_tau=float(optimizer.kappa_tau),
            formula_sigma_by_component=dict(self.formula_sigma_by_component),
            selected_tau_by_component={
                key: float(value) for key, value in selected_state.scales.items()
            },
            selected_metrics=dict(selected_metrics),
            selected_objective=float(optimizer.objective),
            reference_metrics=dict(reference_metrics),
            reference_transformed=dict(reference_transformed),
            normalization_scales=dict(normalization_scales),
            trace=trace,
            replay_artifacts=selected_artifacts,
            optimizer=optimizer,
        )


__all__ = [
    "KAPPA_TAU_ANCHORS",
    "RHO_TAU_OBJECTIVE_WEIGHTS",
    "JointRhoTauNewtonResult",
    "JointRhoTauOptimizationFailure",
    "JointRhoTauSelectionOutcome",
    "JointRhoTauSelectionSettings",
    "JointRhoTauSelector",
    "safeguarded_bounded_newton_logrho_kappa_tau",
]
