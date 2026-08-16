""








from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import pandas as pd


GRID_BRENT = "grid_brent"
GRID_ONLY = "grid_only"
SAFEGUARDED_NEWTON = "safeguarded_newton"
OPTIMIZER_STRATEGIES = frozenset({GRID_BRENT, GRID_ONLY, SAFEGUARDED_NEWTON})

MINIMUM = "minimum"
ONE_SE_SMALLER_RHO = "one_se_smaller_rho"
SELECTION_RULES = frozenset({MINIMUM, ONE_SE_SMALLER_RHO})

Evaluator = Callable[[float], Mapping[str, Any]]


@dataclass(frozen=True)
class RMSEFirstOptimizerSettings:
    ""

    strategy: str = GRID_BRENT
    rho_bounds: tuple[float, float] = (0.05, 1.0)
    coarse_grid_points: int = 17
    additional_grid_rhos: tuple[float, ...] = field(default_factory=tuple)
    selection_rule: str = ONE_SE_SMALLER_RHO
    brent_log_xatol: float = 1e-3
    brent_maxiter: int = 64
    max_evaluations: int = 64
    exact_tie_tolerance: float = 1e-10
    cache_logrho_decimals: int = 13

    def validate(self) -> "RMSEFirstOptimizerSettings":
        if self.strategy not in OPTIMIZER_STRATEGIES:
            raise ValueError(
                f"unknown RMSE-first optimizer strategy {self.strategy!r}; "
                f"expected one of {sorted(OPTIMIZER_STRATEGIES)}"
            )
        if self.selection_rule not in SELECTION_RULES:
            raise ValueError(
                f"unknown RMSE-first selection rule {self.selection_rule!r}; "
                f"expected one of {sorted(SELECTION_RULES)}"
            )
        lower, upper = (float(value) for value in self.rho_bounds)
        if not (
            math.isfinite(lower)
            and math.isfinite(upper)
            and 0.0 < lower < upper
        ):
            raise ValueError("rho_bounds must be finite, positive, and increasing")
        if int(self.coarse_grid_points) < 3:
            raise ValueError("coarse_grid_points must be at least three")
        if int(self.max_evaluations) < int(self.coarse_grid_points):
            raise ValueError("max_evaluations must cover the deterministic coarse grid")
        if int(self.brent_maxiter) < 1:
            raise ValueError("brent_maxiter must be positive")
        if not math.isfinite(float(self.brent_log_xatol)) or float(self.brent_log_xatol) <= 0.0:
            raise ValueError("brent_log_xatol must be finite and positive")
        if (
            not math.isfinite(float(self.exact_tie_tolerance))
            or float(self.exact_tie_tolerance) < 0.0
        ):
            raise ValueError("exact_tie_tolerance must be finite and nonnegative")
        if not 8 <= int(self.cache_logrho_decimals) <= 15:
            raise ValueError("cache_logrho_decimals must lie in [8, 15]")
        extras = tuple(float(value) for value in self.additional_grid_rhos)
        if any(
            not math.isfinite(value) or not lower <= value <= upper
            for value in extras
        ):
            raise ValueError("additional_grid_rhos must be finite and inside rho_bounds")
        if int(self.max_evaluations) < len(self.coarse_grid()):
            raise ValueError(
                "max_evaluations must cover the coarse grid plus additional_grid_rhos"
            )
        return self

    def coarse_grid(self) -> tuple[float, ...]:
        ""

        lower, upper = (float(value) for value in self.rho_bounds)
        log_values = np.linspace(
            math.log(lower),
            math.log(upper),
            int(self.coarse_grid_points),
            dtype=float,
        )
        values = [float(math.exp(value)) for value in log_values]
        values[0] = lower
        values[-1] = upper
        values.extend(float(value) for value in self.additional_grid_rhos)
        return tuple(sorted(set(values)))

    def to_config(self) -> dict[str, object]:
        return {
            "schema": "caster_rmse_first_scalar_optimizer_settings_v1",
            "strategy": self.strategy,
            "domain": "log_rho",
            "rho_bounds": [float(value) for value in self.rho_bounds],
            "coarse_grid_points": int(self.coarse_grid_points),
            "coarse_grid": [float(value) for value in self.coarse_grid()],
            "additional_grid_rhos": [
                float(value) for value in self.additional_grid_rhos
            ],
            "selection_rule": self.selection_rule,
            "brent_log_xatol": float(self.brent_log_xatol),
            "brent_maxiter": int(self.brent_maxiter),
            "max_evaluations": int(self.max_evaluations),
            "exact_tie_tolerance": float(self.exact_tie_tolerance),
            "cache_logrho_decimals": int(self.cache_logrho_decimals),
        }


@dataclass(frozen=True)
class _Evaluation:
    rho: float
    logrho: float
    mean_fold_mse: float
    fold_mse_values: tuple[float, ...]
    protocol_total_rmse: float
    fold_count: int
    se_mse: float
    evaluator_payload: dict[str, Any]


@dataclass
class RMSEFirstOptimizationResult:
    rho: float
    mean_fold_mse: float
    protocol_total_rmse: float
    fold_mse_values: tuple[float, ...]
    selected_evaluator_payload: dict[str, Any]
    trace: pd.DataFrame
    evaluation_count: int
    strategy: str
    selection_rule_requested: str
    selection_rule_applied: str
    empirical_min_rho: float
    empirical_min_mean_fold_mse: float
    one_se_threshold: float | None
    one_se_available: bool
    coarse_grid: tuple[float, ...]
    fallback_count: int = 0
    fallback_reason: str = ""
    scipy_version: str = ""

    def to_config(self) -> dict[str, object]:
        selected_rows = self.trace.loc[self.trace["selected"]]
        selected_stage = (
            str(selected_rows.iloc[-1]["stage"])
            if not selected_rows.empty
            else ""
        )
        selected_flags = (
            str(selected_rows.iloc[-1]["flags"])
            if not selected_rows.empty
            else ""
        )
        return {
            "schema": "caster_rmse_first_scalar_optimizer_result_v1",
            "strategy": self.strategy,
            "selection_rule_requested": self.selection_rule_requested,
            "selection_rule_applied": self.selection_rule_applied,
            "stage": selected_stage,
            "rho": float(self.rho),
            "logrho": float(math.log(self.rho)),
            "mean_fold_mse": float(self.mean_fold_mse),
            "protocol_total_rmse": float(self.protocol_total_rmse),
            "fold_count": int(len(self.fold_mse_values)),
            "se_mse": float(
                np.std(self.fold_mse_values, ddof=1)
                / math.sqrt(len(self.fold_mse_values))
            )
            if len(self.fold_mse_values) >= 2
            else 0.0,
            "empirical_min_rho": float(self.empirical_min_rho),
            "empirical_min_mean_fold_mse": float(
                self.empirical_min_mean_fold_mse
            ),
            "one_se_threshold": (
                float(self.one_se_threshold)
                if self.one_se_threshold is not None
                else None
            ),
            "one_se_available": bool(self.one_se_available),
            "coarse_grid": [float(value) for value in self.coarse_grid],
            "evaluation_count": int(self.evaluation_count),
            "fallback_count": int(self.fallback_count),
            "fallback_reason": self.fallback_reason,
            "flags": selected_flags,
            "scipy_version": self.scipy_version,
        }


class _EvaluationBudgetExceeded(RuntimeError):
    pass


def _coerce_evaluation(rho: float, payload: Mapping[str, Any]) -> _Evaluation:
    if not isinstance(payload, Mapping):
        raise ValueError("RMSE-first evaluator must return a mapping")
    required = {"mean_fold_mse", "fold_mse_values"}
    if missing := sorted(required - set(payload)):
        raise ValueError(f"RMSE-first evaluator payload missing {missing}")
    try:
        declared_mean = float(payload["mean_fold_mse"])
        values = np.asarray(payload["fold_mse_values"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("RMSE-first evaluator returned nonnumeric losses") from exc
    if values.ndim != 1 or values.size == 0:
        raise ValueError("fold_mse_values must be a nonempty one-dimensional sequence")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("fold_mse_values must be finite and nonnegative")
    if not math.isfinite(declared_mean) or declared_mean < 0.0:
        raise ValueError("mean_fold_mse must be finite and nonnegative")
    computed_mean = float(values.mean())
    if not math.isclose(
        declared_mean,
        computed_mean,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "mean_fold_mse must equal the arithmetic mean of fold_mse_values"
        )
    fold_count = int(values.size)
    se_mse = (
        float(values.std(ddof=1) / math.sqrt(fold_count))
        if fold_count >= 2
        else 0.0
    )
    if "protocol_total_rmse" in payload:
        try:
            protocol_total_rmse = float(payload["protocol_total_rmse"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "protocol_total_rmse must be numeric when supplied"
            ) from exc
        if not math.isfinite(protocol_total_rmse) or protocol_total_rmse < 0.0:
            raise ValueError(
                "protocol_total_rmse must be finite and nonnegative when supplied"
            )
    else:
        protocol_total_rmse = float(math.sqrt(declared_mean))
    return _Evaluation(
        rho=float(rho),
        logrho=float(math.log(rho)),
        mean_fold_mse=declared_mean,
        fold_mse_values=tuple(float(value) for value in values),
        protocol_total_rmse=protocol_total_rmse,
        fold_count=fold_count,
        se_mse=se_mse,
                                                                            
                                                                              
                       
        evaluator_payload=dict(payload),
    )


def _empirical_minimum(
    evaluations: Sequence[_Evaluation],
    *,
    tolerance: float,
) -> _Evaluation:
    minimum = min(item.mean_fold_mse for item in evaluations)
    tied = [
        item
        for item in evaluations
        if item.mean_fold_mse <= minimum + float(tolerance)
    ]
    return min(tied, key=lambda item: (item.rho, item.mean_fold_mse))


def optimize_rmse_first_rho(
    evaluator: Evaluator,
    *,
    settings: RMSEFirstOptimizerSettings | None = None,
) -> RMSEFirstOptimizationResult:
    ""







    settings = (settings or RMSEFirstOptimizerSettings()).validate()
    lower, upper = (float(value) for value in settings.rho_bounds)
    lower_x, upper_x = math.log(lower), math.log(upper)
    decimals = int(settings.cache_logrho_decimals)
    lower_key, upper_key = round(lower_x, decimals), round(upper_x, decimals)
    cache: dict[float, _Evaluation] = {}
    trace_rows: list[dict[str, object]] = []
    fallback_reasons: list[str] = []
    fallback_count = 0
    scipy_version = ""

    def key_for_logrho(logrho: float) -> float:
        projected = min(max(float(logrho), lower_x), upper_x)
        return round(projected, decimals)

    def rho_for_key(key: float) -> float:
        if key == lower_key:
            return lower
        if key == upper_key:
            return upper
        return float(math.exp(key))

    def evaluate_logrho(
        logrho: float,
        stage: str,
        *,
        rho_hint: float | None = None,
    ) -> _Evaluation:
        key = key_for_logrho(logrho)
        cache_hit = key in cache
        if not cache_hit:
            if len(cache) >= int(settings.max_evaluations):
                raise _EvaluationBudgetExceeded(
                    "RMSE-first optimizer exhausted max_evaluations"
                )
                                                                            
                                                                             
                                                          
            rho = rho_for_key(key) if rho_hint is None else float(rho_hint)
            rho = min(max(rho, lower), upper)
            if key_for_logrho(math.log(rho)) != key:
                raise RuntimeError("rho_hint does not match the canonical cache key")
            cache[key] = _coerce_evaluation(rho, evaluator(rho))
        item = cache[key]
        trace_rows.append(
            {
                "strategy": settings.strategy,
                "stage": stage,
                "rho": item.rho,
                "logrho": item.logrho,
                "mean_fold_mse": item.mean_fold_mse,
                "protocol_total_rmse": item.protocol_total_rmse,
                "fold_count": item.fold_count,
                "se_mse": item.se_mse,
                "cache_hit": bool(cache_hit),
                "empirical_minimum": False,
                "one_se_eligible": False,
                "selected": False,
                "fallback": False,
                "selection_rule_fallback": False,
                "flags": "",
                "evaluator_payload": dict(item.evaluator_payload),
                "_cache_key": key,
            }
        )
        return item

    coarse_grid = settings.coarse_grid()
    coarse_keys: list[float] = []
    for rho in coarse_grid:
        item = evaluate_logrho(
            math.log(float(rho)),
            "coarse_grid",
            rho_hint=float(rho),
        )
        coarse_keys.append(key_for_logrho(item.logrho))
    coarse_keys = sorted(set(coarse_keys))

    if settings.strategy == GRID_BRENT:
        coarse_evaluations = [cache[key] for key in coarse_keys]
        coarse_best = _empirical_minimum(
            coarse_evaluations,
            tolerance=settings.exact_tie_tolerance,
        )
        best_key = key_for_logrho(coarse_best.logrho)
        best_index = coarse_keys.index(best_key)
        if 0 < best_index < len(coarse_keys) - 1:
            try:
                import scipy
                from scipy.optimize import minimize_scalar

                scipy_version = str(scipy.__version__)

                def bounded_objective(logrho: float) -> float:
                    return evaluate_logrho(
                        logrho, "bounded_brent_refinement"
                    ).mean_fold_mse

                refinement = minimize_scalar(
                    bounded_objective,
                    method="bounded",
                    bounds=(coarse_keys[best_index - 1], coarse_keys[best_index + 1]),
                    options={
                        "xatol": float(settings.brent_log_xatol),
                        "maxiter": int(settings.brent_maxiter),
                    },
                )
                if math.isfinite(float(refinement.x)):
                    evaluate_logrho(float(refinement.x), "bounded_brent_result")
                if not bool(refinement.success):
                    fallback_count += 1
                    fallback_reasons.append("bounded_brent_not_converged")
            except ImportError:
                fallback_count += 1
                fallback_reasons.append("scipy_unavailable_grid_fallback")
            except _EvaluationBudgetExceeded:
                fallback_count += 1
                fallback_reasons.append("bounded_brent_evaluation_budget_exhausted")
        else:
            fallback_count += 1
            fallback_reasons.append("coarse_minimum_on_boundary_no_refinement")
    elif settings.strategy == SAFEGUARDED_NEWTON:
        from .rho_only import (
            RhoOnlySelectionSettings,
            safeguarded_bounded_newton_logrho,
        )

                                                                              
                                                                              
                                                                
        geometric_midpoint = float(math.sqrt(lower * upper))
        reference_rho = min(
            coarse_grid,
            key=lambda value: abs(math.log(float(value) / geometric_midpoint)),
        )
        alternate_settings = RhoOnlySelectionSettings(
            rho_bounds=(lower, upper),
            anchor_rhos=tuple(float(value) for value in coarse_grid),
            reference_rho=reference_rho,
            max_evaluations=int(settings.max_evaluations),
            exact_tie_tolerance=float(settings.exact_tie_tolerance),
        ).validate()

        def newton_objective(rho: float) -> float:
            return evaluate_logrho(
                math.log(float(rho)), "safeguarded_newton_adapter"
            ).mean_fold_mse

        alternate_result = safeguarded_bounded_newton_logrho(
            newton_objective,
            settings=alternate_settings,
        )
        fallback_count += int(alternate_result.fallback_count)
        if int(alternate_result.fallback_count) > 0:
            fallback_reasons.append("alternate_newton_safeguards_used")
        evaluate_logrho(
            math.log(float(alternate_result.rho)), "safeguarded_newton_result"
        )

    evaluations = list(cache.values())
    empirical_min = _empirical_minimum(
        evaluations,
        tolerance=settings.exact_tie_tolerance,
    )
    one_se_available = empirical_min.fold_count >= 2
    threshold: float | None = None
    selection_rule_applied = settings.selection_rule
    if settings.selection_rule == ONE_SE_SMALLER_RHO and one_se_available:
        threshold = empirical_min.mean_fold_mse + empirical_min.se_mse
        eligible = [
            item
            for item in evaluations
            if item.mean_fold_mse
            <= float(threshold) + float(settings.exact_tie_tolerance)
        ]
        selected = min(
            eligible,
            key=lambda item: (item.rho, item.mean_fold_mse),
        )
    else:
        selected = empirical_min
        if settings.selection_rule == ONE_SE_SMALLER_RHO:
            selection_rule_applied = "minimum_fold_count_lt_2"
            fallback_count += 1
            fallback_reasons.append("one_se_requires_at_least_two_folds")

    empirical_key = key_for_logrho(empirical_min.logrho)
    selected_key = key_for_logrho(selected.logrho)
    for row in trace_rows:
        key = float(row["_cache_key"])
        row["empirical_minimum"] = key == empirical_key
        row["one_se_eligible"] = bool(
            threshold is not None
            and cache[key].mean_fold_mse
            <= float(threshold) + float(settings.exact_tie_tolerance)
        )
        row["selection_rule_fallback"] = (
            selection_rule_applied != settings.selection_rule
        )
    selected_rows = [
        index
        for index, row in enumerate(trace_rows)
        if float(row["_cache_key"]) == selected_key
    ]
    if not selected_rows:
        raise RuntimeError("selected rho is absent from the optimizer trace")
    trace_rows[selected_rows[-1]]["selected"] = True
    if fallback_reasons:
        trace_rows[selected_rows[-1]]["fallback"] = True
    flag_columns = (
        "cache_hit",
        "empirical_minimum",
        "one_se_eligible",
        "selected",
        "fallback",
        "selection_rule_fallback",
    )
    for row in trace_rows:
        row["flags"] = "|".join(
            name for name in flag_columns if bool(row[name])
        )

    trace = pd.DataFrame(trace_rows).drop(columns="_cache_key")
    return RMSEFirstOptimizationResult(
        rho=selected.rho,
        mean_fold_mse=selected.mean_fold_mse,
        protocol_total_rmse=selected.protocol_total_rmse,
        fold_mse_values=selected.fold_mse_values,
        selected_evaluator_payload=dict(selected.evaluator_payload),
        trace=trace,
        evaluation_count=int(len(cache)),
        strategy=settings.strategy,
        selection_rule_requested=settings.selection_rule,
        selection_rule_applied=selection_rule_applied,
        empirical_min_rho=empirical_min.rho,
        empirical_min_mean_fold_mse=empirical_min.mean_fold_mse,
        one_se_threshold=threshold,
        one_se_available=one_se_available,
        coarse_grid=tuple(float(value) for value in coarse_grid),
        fallback_count=int(fallback_count),
        fallback_reason=";".join(dict.fromkeys(fallback_reasons)),
        scipy_version=scipy_version,
    )


__all__ = [
    "GRID_BRENT",
    "GRID_ONLY",
    "MINIMUM",
    "ONE_SE_SMALLER_RHO",
    "OPTIMIZER_STRATEGIES",
    "RMSEFirstOptimizationResult",
    "RMSEFirstOptimizerSettings",
    "SAFEGUARDED_NEWTON",
    "SELECTION_RULES",
    "optimize_rmse_first_rho",
]
