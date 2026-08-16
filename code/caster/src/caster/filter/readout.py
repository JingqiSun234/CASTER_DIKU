from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm, t as student_t

from .availability import forecast_unavailable_mask
from .evidence import effective_sample_size

if TYPE_CHECKING:
    from caster.bridge.likelihood import BridgeConfig


alternate_ARCHIVE_MOMENT = "alternate_archive_moment"
COHERENT_MEAN_PRESERVING_TRUNCATED_T = (
    "coherent_mean_preserving_truncated_t"
)
COHERENT_CENSORED_STUDENT_T = "coherent_censored_student_t"
COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T = (
    "coherent_mean_preserving_censored_student_t"
)
ARCHIVE_MEAN_BRIDGE_QUANTILES = "archive_mean_bridge_quantiles"
PREDICTIVE_CONTRACTS = {
    alternate_ARCHIVE_MOMENT,
    COHERENT_MEAN_PRESERVING_TRUNCATED_T,
    COHERENT_CENSORED_STUDENT_T,
    COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T,
    ARCHIVE_MEAN_BRIDGE_QUANTILES,
}
CENSORED_PREDICTIVE_CONTRACTS = {
    COHERENT_CENSORED_STUDENT_T,
    COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T,
}
SCORE_SOURCES = {"archive_moment", "draw_kernel"}
PredictiveComponentCache = dict[tuple[str, str, str, str], object]


def _predictive_contract(bridge_config: "BridgeConfig | None") -> str:
    contract = (
        alternate_ARCHIVE_MOMENT
        if bridge_config is None
        else str(
            getattr(
                bridge_config,
                "predictive_contract",
                alternate_ARCHIVE_MOMENT,
            )
        )
    )
    if contract not in PREDICTIVE_CONTRACTS:
        raise ValueError(f"unknown predictive contract {contract!r}")
    return contract


def _validate_score_source(score_source: str) -> str:
    value = str(score_source)
    if value not in SCORE_SOURCES:
        raise ValueError(
            f"unknown score_source {value!r}; expected one of {sorted(SCORE_SOURCES)}"
        )
    return value


def _bridge_lookup(
    values: dict,
    row: pd.Series,
    default: float,
) -> float:
                                                                             
                                                                           
                                                                             
    from caster.bridge.likelihood import lookup_bridge_float

    return lookup_bridge_float(
        values,
        row.get("mode", ""),
        row.get("component", ""),
        row.get("horizon"),
        default,
    )


def _raw_from_transform(value: float | np.ndarray, transform: str) -> float | np.ndarray:
    array = np.asarray(value, dtype=float)
    if transform == "log1p":
        result = np.maximum(np.expm1(array), 0.0)
    elif transform == "identity":
        result = np.maximum(array, 0.0)
    else:
        raise ValueError(f"unknown transform {transform!r}")
    return float(result) if result.ndim == 0 else result


def _transform_raw(value: np.ndarray, transform: str) -> np.ndarray:
    raw = np.maximum(np.asarray(value, dtype=float), 0.0)
    if transform == "log1p":
        return np.log1p(raw)
    if transform == "identity":
        return raw
    raise ValueError(f"unknown transform {transform!r}")


def _archive_bridge_parameters(
    group: pd.DataFrame,
    bridge_config: "BridgeConfig",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_mean = pd.to_numeric(group["pred_mean"], errors="raise").to_numpy(
        dtype=float
    )
    raw_var = np.maximum(
        pd.to_numeric(group["pred_var"], errors="raise").to_numpy(dtype=float),
        0.0,
    )
    location = _transform_raw(raw_mean, bridge_config.transform)
    if bridge_config.transform == "log1p":
        if (
            bridge_config.predictive_contract
            in CENSORED_PREDICTIVE_CONTRACTS
        ):
            from caster.bridge.likelihood import (
                stable_log1p_transform_var,
            )

            transformed_var = np.asarray(
                stable_log1p_transform_var(raw_mean, raw_var), dtype=float
            )
        else:
            transformed_var = raw_var / np.square(
                1.0 + np.maximum(raw_mean, 0.0)
            )
    elif bridge_config.transform == "identity":
        transformed_var = raw_var
    else:
        raise ValueError(f"unknown transform {bridge_config.transform!r}")

    if bridge_config.predictive_contract in CENSORED_PREDICTIVE_CONTRACTS:
        from caster.bridge.likelihood import _lookup_bridge_series

        mode = (
            group["mode"]
            if "mode" in group.columns
            else pd.Series("", index=group.index, dtype=str)
        )
        component = group["component"]
        horizon = group["horizon"]
        sigma = _lookup_bridge_series(
            bridge_config.sigma_by_component,
            mode,
            component,
            horizon,
            bridge_config.default_sigma,
        )
        gamma = np.maximum(
            _lookup_bridge_series(
                bridge_config.gamma_by_component,
                mode,
                component,
                horizon,
                bridge_config.default_gamma,
            ),
            0.0,
        )
        nu = _lookup_bridge_series(
            bridge_config.nu_by_component,
            mode,
            component,
            horizon,
            bridge_config.nu,
        )
    else:
        sigma = np.asarray(
            [
                _bridge_lookup(
                    bridge_config.sigma_by_component,
                    row,
                    bridge_config.default_sigma,
                )
                for _, row in group.iterrows()
            ],
            dtype=float,
        )
        gamma = np.asarray(
            [
                max(
                    0.0,
                    _bridge_lookup(
                        bridge_config.gamma_by_component,
                        row,
                        bridge_config.default_gamma,
                    ),
                )
                for _, row in group.iterrows()
            ],
            dtype=float,
        )
        nu = np.asarray(
            [
                _bridge_lookup(
                    bridge_config.nu_by_component,
                    row,
                    bridge_config.nu,
                )
                for _, row in group.iterrows()
            ],
            dtype=float,
        )
    scale = np.sqrt(
        np.maximum(gamma * transformed_var, 0.0)
        + np.square(sigma)
        + float(bridge_config.min_scale) ** 2
    )
    scale = np.maximum(scale, 1e-12)
    if np.any((nu <= 0.0) | np.isnan(nu)):
        raise ValueError("Student-t nu must be positive or infinity")
    return location, scale, nu


def _draw_bridge_parameters(
    group: pd.DataFrame,
    draws: pd.DataFrame,
    bridge_config: "BridgeConfig",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    draw_rows = _draw_rows_with_context(group, draws)
    keys = ["forecast_id", "model_id", "particle_id"]
    counts = draw_rows.groupby(keys)["draw"].transform("size").astype(float)
    draw_rows["component_weight"] = (
        pd.to_numeric(draw_rows["mixture_weight"], errors="raise") / counts
    )
    location = _transform_raw(
        draw_rows["draw"].to_numpy(dtype=float), bridge_config.transform
    )
    scale = np.asarray(
        [
            _draw_kernel_bandwidth(row, bridge_config)
            for _, row in draw_rows.iterrows()
        ],
        dtype=float,
    )
    nu = np.asarray(
        [_draw_kernel_nu(row, bridge_config) for _, row in draw_rows.iterrows()],
        dtype=float,
    )
    if np.any((nu <= 0.0) | np.isnan(nu)):
        raise ValueError("Student-t kernel nu must be positive or infinity")
    weights = draw_rows["component_weight"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    return location, scale, nu, weights


def _draw_rows_with_context(
    group: pd.DataFrame,
    forecast_draws: pd.DataFrame,
) -> pd.DataFrame:
    ""







    keys = ["forecast_id", "model_id", "particle_id"]
    required = {*keys, "draw"}
    if missing := sorted(required - set(forecast_draws.columns)):
        raise ValueError(f"draw-kernel readout missing draw columns {missing}")
    group_keys = group[keys].drop_duplicates().copy()
    for key in keys:
        group_keys[key] = group_keys[key].astype(str)
    draw_rows = forecast_draws[[*keys, "draw"]].copy()
    for key in keys:
        draw_rows[key] = draw_rows[key].astype(str)
    draw_rows = draw_rows.merge(group_keys, on=keys, how="inner")
    draw_rows["draw"] = pd.to_numeric(draw_rows["draw"], errors="coerce")
    draw_rows = draw_rows[np.isfinite(draw_rows["draw"])].copy()
    if draw_rows.empty:
        raise ValueError("draw-kernel readout has no finite matching draws")

    context_columns = [*keys, "mode", "component", "horizon", "mixture_weight"]
    context = group[context_columns].drop_duplicates(keys).copy()
    for key in keys:
        context[key] = context[key].astype(str)
    matched = draw_rows[keys].drop_duplicates()
    missing_groups = group_keys.merge(
        matched, on=keys, how="left", indicator=True
    )
    missing_groups = missing_groups[missing_groups["_merge"].eq("left_only")]
    if not missing_groups.empty:
        raise ValueError(
            "draw-kernel readout is missing finite draws for posterior mixture "
            f"components; examples={missing_groups[keys].head(5).to_dict('records')}"
        )
    draw_rows = draw_rows.merge(context, on=keys, how="inner", validate="many_to_one")
    return draw_rows


def _draw_kernel_bandwidth(
    row: pd.Series,
    bridge_config: "BridgeConfig",
) -> float:
    values = (
        bridge_config.tau_by_component
        if bridge_config.tau_by_component
        else bridge_config.sigma_by_component
    )
    default = (
        bridge_config.default_tau
        if bridge_config.tau_by_component
        else bridge_config.default_sigma
    )
    return max(
        float(bridge_config.min_scale),
        _bridge_lookup(values, row, default),
        1e-6,
    )


def _draw_kernel_nu(
    row: pd.Series,
    bridge_config: "BridgeConfig",
) -> float:
    default_nu = (
        bridge_config.nu
        if bridge_config.nu_by_component
        else bridge_config.kernel_nu
    )
    return _bridge_lookup(
        bridge_config.nu_by_component,
        row,
        default_nu,
    )


def _continuous_mixture_quantiles(
    location: np.ndarray,
    scale: np.ndarray,
    nu: np.ndarray,
    weights: np.ndarray,
    probabilities: tuple[float, ...],
    *,
    distribution: str,
) -> np.ndarray:
    location = np.asarray(location, dtype=float)
    scale = np.asarray(scale, dtype=float)
    nu = np.asarray(nu, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    if not (
        len(location) == len(scale) == len(nu) == len(weights)
        and len(location) > 0
    ):
        raise ValueError("bridge mixture parameter arrays must be nonempty and aligned")

    def cdf(value: float) -> float:
        standardized = (float(value) - location) / scale
        if distribution == "gaussian":
            component_cdf = norm.cdf(standardized)
        elif distribution == "student_t":
            component_cdf = np.empty_like(standardized)
            gaussian = np.isinf(nu)
            component_cdf[gaussian] = norm.cdf(standardized[gaussian])
            component_cdf[~gaussian] = student_t.cdf(
                standardized[~gaussian], df=nu[~gaussian]
            )
        else:
            raise ValueError(f"unknown bridge distribution {distribution!r}")
        return float(np.dot(weights, component_cdf))

    span = max(float(np.max(scale)), 1e-6)
    lower = float(np.min(location - 8.0 * scale))
    upper = float(np.max(location + 8.0 * scale))
    minimum_p = min(probabilities)
    maximum_p = max(probabilities)
    for _ in range(64):
        if cdf(lower) <= minimum_p and cdf(upper) >= maximum_p:
            break
        span *= 2.0
        if cdf(lower) > minimum_p:
            lower = float(np.min(location) - span)
        if cdf(upper) < maximum_p:
            upper = float(np.max(location) + span)
    else:
        raise FloatingPointError("failed to bracket bridge mixture quantiles")
    return np.asarray(
        [
            brentq(
                lambda value, probability=probability: cdf(value) - probability,
                lower,
                upper,
                xtol=1e-10,
                rtol=1e-10,
                maxiter=160,
            )
            for probability in probabilities
        ],
        dtype=float,
    )


def _interval_fields(raw_quantiles: np.ndarray) -> dict[str, float]:
    if len(raw_quantiles) != 6:
        raise ValueError("readout requires six fixed interval quantiles")
    return {
        "lower_95": float(raw_quantiles[0]),
        "lower_90": float(raw_quantiles[1]),
        "lower_50": float(raw_quantiles[2]),
        "upper_50": float(raw_quantiles[3]),
        "upper_90": float(raw_quantiles[4]),
        "upper_95": float(raw_quantiles[5]),
    }


INTERVAL_PROBABILITIES = (0.025, 0.05, 0.25, 0.75, 0.95, 0.975)
CENSORED_READOUT_PROBABILITIES = (
    0.025,
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
    0.975,
)
_CENSORED_MEDIAN_QUANTILE_INDEX = 3
_CENSORED_INTERVAL_QUANTILE_INDICES = (0, 1, 2, 4, 5, 6)


def _censored_quantile_fields(
    raw_quantiles: np.ndarray,
) -> dict[str, float]:
    ""

    values = np.asarray(raw_quantiles, dtype=float).reshape(-1)
    if len(values) != len(CENSORED_READOUT_PROBABILITIES):
        raise ValueError(
            "censored readout requires seven fixed quantiles including q0.5"
        )
    interval_quantiles = values[
        np.asarray(_CENSORED_INTERVAL_QUANTILE_INDICES, dtype=int)
    ]
    return {
        "predictive_median": float(
            values[_CENSORED_MEDIAN_QUANTILE_INDEX]
        ),
        **_interval_fields(interval_quantiles),
    }


def _archive_mean_bridge_interval_fields(
    group: pd.DataFrame,
    bridge_config: "BridgeConfig",
    *,
    score_source: str,
    draws: pd.DataFrame | None,
) -> dict[str, float]:
    if score_source == "archive_moment":
        location, scale, nu = _archive_bridge_parameters(group, bridge_config)
        weights = group["mixture_weight"].to_numpy(dtype=float)
        distribution = str(bridge_config.distribution)
    else:
        if draws is None:
            raise ValueError(
                "draw-kernel bridge quantiles require the matching forecast draws"
            )
        location, scale, nu, weights = _draw_bridge_parameters(
            group, draws, bridge_config
        )
        distribution = str(bridge_config.kernel_distribution)
    transformed_quantiles = _continuous_mixture_quantiles(
        location,
        scale,
        nu,
        weights,
        INTERVAL_PROBABILITIES,
        distribution=distribution,
    )
    raw_quantiles = np.asarray(
        _raw_from_transform(transformed_quantiles, bridge_config.transform),
        dtype=float,
    )
    return _interval_fields(raw_quantiles)


def _index_draws_by_forecast(
    draws: pd.DataFrame | None,
) -> dict[str, np.ndarray]:
    ""

    if draws is None:
        return {}
    if "forecast_id" not in draws.columns:
        raise ValueError("draw-kernel readout draws are missing forecast_id")
    forecast_key = draws["forecast_id"].astype(str)
    return {
        str(forecast_id): np.asarray(positions, dtype=np.int64)
        for forecast_id, positions in draws.groupby(
            forecast_key, sort=False
        ).indices.items()
    }


def _weighted_discrete_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: tuple[float, ...],
    *,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if values.size == 0 or values.size != weights.size:
        raise ValueError("coherent quadrature values and weights must be aligned")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0 or not np.isfinite(weights.sum()) or weights.sum() <= 0.0:
        raise ValueError("coherent quadrature mixture has no positive finite mass")
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order] / weights.sum()
                                                                         
                                                                             
                                                                              
                                                          
    midpoint_cdf = np.cumsum(weights) - 0.5 * weights
    probability_grid = np.concatenate(([0.0], midpoint_cdf, [1.0]))
    value_grid = np.concatenate(
        ([float(lower_bound)], values, [float(upper_bound)])
    )
    return np.interp(
        np.asarray(probabilities, dtype=float), probability_grid, value_grid
    )


def _censored_mixture_quantiles(
    predictives: list[object],
    weights: np.ndarray,
    probabilities: tuple[float, ...],
) -> np.ndarray:
    ""






    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    upper_values = np.asarray(
        [float(item.upper_raw) for item in predictives], dtype=float
    )
    if not np.allclose(
        upper_values, upper_values[0], rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "censored posterior mixture components require one common upper bound"
        )
    upper_raw = float(upper_values[0])
    upper_z = float(np.log1p(upper_raw))
    lower_atom = float(
        sum(
            weight * float(item.lower_atom_probability)
            for weight, item in zip(weights, predictives)
        )
    )
    upper_atom = float(
        sum(
            weight * float(item.upper_atom_probability)
            for weight, item in zip(weights, predictives)
        )
    )

    def latent_cdf(z_value: float) -> float:
        total = 0.0
        for weight, item in zip(weights, predictives):
            standardized = (
                float(z_value) - np.asarray(item.centers_z, dtype=float)
            ) / float(item.scale)
            total += float(weight) * float(
                np.mean(student_t.cdf(standardized, df=float(item.nu)))
            )
        return total

    values: list[float] = []
    for probability in probabilities:
        q = float(probability)
        if q <= lower_atom:
            values.append(0.0)
        elif q > 1.0 - upper_atom:
            values.append(upper_raw)
        else:
            z_quantile = brentq(
                lambda value: latent_cdf(value) - q,
                0.0,
                upper_z,
                xtol=1e-11,
                rtol=1e-11,
                maxiter=160,
            )
            values.append(float(np.expm1(z_quantile)))
    return np.asarray(values, dtype=float)


def _censored_predictive_fields(
    group: pd.DataFrame,
    bridge_config: "BridgeConfig",
    *,
    score_source: str,
    forecast_draws: pd.DataFrame | None,
    predictive_component_cache: PredictiveComponentCache | None,
) -> dict[str, object]:
    ""

    from caster.bridge.likelihood import (
        fit_censored_draw_predictive,
        fit_censored_moment_predictive,
        fit_mean_preserving_censored_draw_predictive,
        fit_mean_preserving_censored_moment_predictive,
    )

    if bridge_config.transform != "log1p":
        raise ValueError(
            "coherent censored Student-t requires log1p transform"
        )
    if float(bridge_config.truncation_lower_raw) != 0.0:
        raise ValueError(
            "coherent censored Student-t requires truncation_lower_raw=0"
        )
    quadrature_order = int(bridge_config.truncation_quadrature_order)
    contract = str(bridge_config.predictive_contract)
    mean_preserving = (
        contract == COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T
    )
    components: list[tuple[float, object]] = []

    def cache_key(row: pd.Series) -> tuple[str, str, str, str]:
        return (
            str(row["forecast_id"]),
            str(row["model_id"]),
            str(row["particle_id"]),
            f"{score_source}::{contract}",
        )

    def cached(key: tuple[str, str, str, str]) -> object | None:
        if predictive_component_cache is None:
            return None
        return predictive_component_cache.get(key)

    def remember(key: tuple[str, str, str, str], predictive: object) -> None:
        if predictive_component_cache is not None:
            predictive_component_cache[key] = predictive

    if score_source == "archive_moment":
        distribution = str(bridge_config.distribution).strip().lower()
        if distribution not in {"student_t", "student-t", "t"}:
            raise ValueError(
                "coherent censored moment contract requires "
                "distribution='student_t'"
            )
        location, scale, nu = _archive_bridge_parameters(
            group, bridge_config
        )
        for position, (_, row) in enumerate(group.iterrows()):
            key = cache_key(row)
            predictive = cached(key)
            if predictive is None:
                upper_raw = _bridge_lookup(
                    bridge_config.truncation_upper_raw_by_component,
                    row,
                    bridge_config.default_truncation_upper_raw,
                )
                if mean_preserving:
                    predictive = (
                        fit_mean_preserving_censored_moment_predictive(
                            target_raw_mean=max(
                                float(row["pred_mean"]), 0.0
                            ),
                            scale=float(scale[position]),
                            nu=float(nu[position]),
                            lower_raw=0.0,
                            upper_raw=float(upper_raw),
                            mean_floor=float(
                                bridge_config.truncation_zero_mean_epsilon
                            ),
                            quadrature_order=quadrature_order,
                        )
                    )
                else:
                    predictive = fit_censored_moment_predictive(
                        loc_z=float(location[position]),
                        scale=float(scale[position]),
                        nu=float(nu[position]),
                        lower_raw=0.0,
                        upper_raw=float(upper_raw),
                        quadrature_order=quadrature_order,
                    )
                remember(key, predictive)
            components.append((float(row["mixture_weight"]), predictive))
    else:
        kernel = str(
            bridge_config.kernel_distribution
        ).strip().lower()
        if kernel not in {"student_t", "student-t", "t"}:
            raise ValueError(
                "coherent censored draw contract requires "
                "kernel_distribution='student_t'"
            )
        if forecast_draws is None:
            raise ValueError(
                "coherent censored draw readout requires forecast draws"
            )
        draw_rows = _draw_rows_with_context(group, forecast_draws)
        keys = ["forecast_id", "model_id", "particle_id"]
        for _, particle_draws in draw_rows.groupby(keys, sort=False):
            row = particle_draws.iloc[0]
            key = cache_key(row)
            predictive = cached(key)
            if predictive is None:
                upper_raw = _bridge_lookup(
                    bridge_config.truncation_upper_raw_by_component,
                    row,
                    bridge_config.default_truncation_upper_raw,
                )
                fit_draw = (
                    fit_mean_preserving_censored_draw_predictive
                    if mean_preserving
                    else fit_censored_draw_predictive
                )
                fit_kwargs: dict[str, object] = {
                    "draws_raw": particle_draws["draw"].to_numpy(dtype=float),
                    "bandwidth": _draw_kernel_bandwidth(row, bridge_config),
                    "nu": _draw_kernel_nu(row, bridge_config),
                    "lower_raw": 0.0,
                    "upper_raw": float(upper_raw),
                    "quadrature_order": quadrature_order,
                }
                if mean_preserving:
                    fit_kwargs["mean_floor"] = float(
                        bridge_config.truncation_zero_mean_epsilon
                    )
                predictive = fit_draw(**fit_kwargs)
                remember(key, predictive)
            components.append((float(row["mixture_weight"]), predictive))

    if not components:
        raise ValueError(
            "coherent censored readout has no predictive mixture components"
        )
    outer_weights = np.asarray(
        [item[0] for item in components], dtype=float
    )
    if (
        np.any(~np.isfinite(outer_weights))
        or np.any(outer_weights < 0.0)
        or outer_weights.sum() <= 0.0
    ):
        raise ValueError(
            "coherent censored readout has invalid posterior mixture weights"
        )
    outer_weights /= outer_weights.sum()
    predictives = [item[1] for item in components]
    mean = float(
        sum(
            weight * float(item.mean)
            for weight, item in zip(outer_weights, predictives)
        )
    )
    second = float(
        sum(
            weight * float(item.second_moment)
            for weight, item in zip(outer_weights, predictives)
        )
    )
    quantiles = _censored_mixture_quantiles(
        predictives, outer_weights, CENSORED_READOUT_PROBABILITIES
    )
    lower_atom = float(
        sum(
            weight * float(item.lower_atom_probability)
            for weight, item in zip(outer_weights, predictives)
        )
    )
    upper_atom = float(
        sum(
            weight * float(item.upper_atom_probability)
            for weight, item in zip(outer_weights, predictives)
        )
    )
    upper_raw = float(predictives[0].upper_raw)
    fields: dict[str, object] = {
        "predictive_mean": mean,
        "predictive_var": max(0.0, second - mean * mean),
        **_censored_quantile_fields(quantiles),
        "predictive_contract": contract,
        "predictive_mean_source": (
            "censored_latent_student_t_posterior_mixture_expectation"
        ),
        "predictive_interval_source": (
            f"censored_{score_source}_posterior_mixture_quantiles"
        ),
        "nll_measure_basis": (
            "log1p_transform_mixed_measure_atoms_at_zero_and_upper_bound"
        ),
        "boundary_atom_source": "latent_student_t_censoring",
        "lower_boundary_atom_probability": lower_atom,
        "upper_boundary_atom_probability": upper_atom,
        "truncation_component_count": len(predictives),
        "truncation_lower_raw": 0.0,
        "truncation_upper_raw_frozen_min": upper_raw,
        "truncation_upper_raw_frozen_max": upper_raw,
        "truncation_upper_raw_effective_min": upper_raw,
        "truncation_upper_raw_effective_max": upper_raw,
        "truncation_upper_raw_min": upper_raw,
        "truncation_upper_raw_max": upper_raw,
        "truncation_bound_policy": str(
            bridge_config.truncation_bound_policy
        ),
        "truncation_support_expansion_policy": (
            "strict_frozen_support"
        ),
        "truncation_support_expanded": False,
        "truncation_support_expanded_count": 0,
        "truncation_quadrature_order": quadrature_order,
    }
    if mean_preserving:
        fields.update(
            {
                "truncation_mean_floor": float(
                    bridge_config.truncation_zero_mean_epsilon
                ),
                "truncation_mean_floor_applied_count": int(
                    sum(
                        bool(item.mean_floor_applied)
                        for item in predictives
                    )
                ),
                "censored_location_shift_min": float(
                    min(float(item.location_shift) for item in predictives)
                ),
                "censored_location_shift_max": float(
                    max(float(item.location_shift) for item in predictives)
                ),
                "censored_mean_constraint_max_abs_residual": float(
                    max(
                        abs(float(item.mean_constraint_residual))
                        for item in predictives
                    )
                ),
            }
        )
    return fields


def _coherent_predictive_fields(
    group: pd.DataFrame,
    bridge_config: "BridgeConfig",
    *,
    score_source: str,
    forecast_draws: pd.DataFrame | None,
    predictive_component_cache: PredictiveComponentCache | None,
) -> dict[str, object]:
    ""

    from caster.bridge.likelihood import (
        fit_coherent_draw_predictive,
        fit_coherent_moment_predictive,
    )

    if bridge_config.transform != "log1p":
        raise ValueError(
            "coherent mean-preserving truncated Student-t requires log1p transform"
        )
    lower_raw = float(bridge_config.truncation_lower_raw)
    quadrature_order = int(bridge_config.truncation_quadrature_order)
    mean_floor = float(bridge_config.truncation_zero_mean_epsilon)
    support_expansion_multiplier = (
        bridge_config.truncation_support_expansion_multiplier
    )
    components: list[tuple[float, object]] = []

    def cache_key(row: pd.Series) -> tuple[str, str, str, str]:
        return (
            str(row["forecast_id"]),
            str(row["model_id"]),
            str(row["particle_id"]),
            str(score_source),
        )

    def cached(key: tuple[str, str, str, str]) -> object | None:
        if predictive_component_cache is None:
            return None
        return predictive_component_cache.get(key)

    def remember(key: tuple[str, str, str, str], predictive: object) -> None:
        if predictive_component_cache is not None:
            predictive_component_cache[key] = predictive

    if score_source == "archive_moment":
        distribution = str(bridge_config.distribution).strip().lower()
        if distribution not in {"student_t", "student-t", "t"}:
            raise ValueError(
                "coherent moment contract requires distribution='student_t'"
            )
        location, scale, nu = _archive_bridge_parameters(group, bridge_config)
        for position, (_, row) in enumerate(group.iterrows()):
            key = cache_key(row)
            predictive = cached(key)
            if predictive is None:
                upper_raw = _bridge_lookup(
                    bridge_config.truncation_upper_raw_by_component,
                    row,
                    bridge_config.default_truncation_upper_raw,
                )
                predictive = fit_coherent_moment_predictive(
                    target_raw_mean=max(float(row["pred_mean"]), 0.0),
                    scale=float(scale[position]),
                    nu=float(nu[position]),
                    lower_raw=lower_raw,
                    upper_raw=float(upper_raw),
                    mean_floor=mean_floor,
                    quadrature_order=quadrature_order,
                    loc_z=float(location[position]),
                    support_expansion_multiplier=support_expansion_multiplier,
                )
                remember(key, predictive)
            components.append((float(row["mixture_weight"]), predictive))
    else:
        kernel = str(bridge_config.kernel_distribution).strip().lower()
        if kernel not in {"student_t", "student-t", "t"}:
            raise ValueError(
                "coherent draw contract requires kernel_distribution='student_t'"
            )
        if forecast_draws is None:
            raise ValueError("coherent draw readout requires forecast draws")
        draw_rows = _draw_rows_with_context(group, forecast_draws)
        keys = ["forecast_id", "model_id", "particle_id"]
        for _, particle_draws in draw_rows.groupby(keys, sort=False):
            row = particle_draws.iloc[0]
            key = cache_key(row)
            predictive = cached(key)
            if predictive is None:
                upper_raw = _bridge_lookup(
                    bridge_config.truncation_upper_raw_by_component,
                    row,
                    bridge_config.default_truncation_upper_raw,
                )
                predictive = fit_coherent_draw_predictive(
                    draws_raw=particle_draws["draw"].to_numpy(dtype=float),
                    bandwidth=_draw_kernel_bandwidth(row, bridge_config),
                    nu=_draw_kernel_nu(row, bridge_config),
                    lower_raw=lower_raw,
                    upper_raw=float(upper_raw),
                    mean_floor=mean_floor,
                    quadrature_order=quadrature_order,
                    support_expansion_multiplier=support_expansion_multiplier,
                )
                remember(key, predictive)
            components.append((float(row["mixture_weight"]), predictive))

    if not components:
        raise ValueError("coherent readout has no predictive mixture components")
    outer_weights = np.asarray([item[0] for item in components], dtype=float)
    if (
        np.any(~np.isfinite(outer_weights))
        or np.any(outer_weights < 0.0)
        or outer_weights.sum() <= 0.0
    ):
        raise ValueError("coherent readout has invalid posterior mixture weights")
    outer_weights = outer_weights / outer_weights.sum()
    predictives = [item[1] for item in components]
    mean = float(
        sum(weight * float(predictive.mean) for weight, predictive in zip(outer_weights, predictives))
    )
    second = float(
        sum(
            weight * float(predictive.second_moment)
            for weight, predictive in zip(outer_weights, predictives)
        )
    )
    raw_nodes = np.concatenate(
        [np.asarray(predictive.quadrature_raw, dtype=float) for predictive in predictives]
    )
    raw_probability = np.concatenate(
        [
            weight * np.asarray(predictive.quadrature_probability, dtype=float)
            for weight, predictive in zip(outer_weights, predictives)
        ]
    )
    quantiles = _weighted_discrete_quantiles(
        raw_nodes,
        raw_probability,
        INTERVAL_PROBABILITIES,
        lower_bound=min(float(predictive.lower_raw) for predictive in predictives),
        upper_bound=max(float(predictive.upper_raw) for predictive in predictives),
    )
    fields: dict[str, object] = {
        "predictive_mean": mean,
        "predictive_var": max(0.0, second - mean * mean),
        **_interval_fields(quantiles),
        "predictive_contract": COHERENT_MEAN_PRESERVING_TRUNCATED_T,
        "predictive_mean_source": (
            "coherent_mean_preserving_truncated_posterior_mixture"
        ),
        "predictive_interval_source": (
            f"coherent_{score_source}_truncated_mixture_quantiles"
        ),
        "truncation_component_count": len(predictives),
        "truncation_mean_floor_applied_count": int(
            sum(bool(predictive.mean_floor_applied) for predictive in predictives)
        ),
        "truncation_support_expanded_count": int(
            sum(bool(predictive.support_expanded) for predictive in predictives)
        ),
        "truncation_support_expanded": bool(
            any(bool(predictive.support_expanded) for predictive in predictives)
        ),
        "truncation_lower_raw": lower_raw,
        "truncation_upper_raw_frozen_min": float(
            min(float(predictive.frozen_upper_raw) for predictive in predictives)
        ),
        "truncation_upper_raw_frozen_max": float(
            max(float(predictive.frozen_upper_raw) for predictive in predictives)
        ),
        "truncation_upper_raw_effective_min": float(
            min(float(predictive.upper_raw) for predictive in predictives)
        ),
        "truncation_upper_raw_effective_max": float(
            max(float(predictive.upper_raw) for predictive in predictives)
        ),
        "truncation_upper_raw_min": float(
            min(float(predictive.upper_raw) for predictive in predictives)
        ),
        "truncation_upper_raw_max": float(
            max(float(predictive.upper_raw) for predictive in predictives)
        ),
        "truncation_bound_policy": str(bridge_config.truncation_bound_policy),
        "truncation_support_expansion_policy": str(
            predictives[0].support_expansion_policy
        ),
        "truncation_support_expansion_multiplier": (
            support_expansion_multiplier
        ),
        "truncation_quadrature_order": quadrature_order,
    }
    return fields


def posterior_predictive_readout(
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    model_weights: pd.DataFrame,
    *,
    interval_z: float = 1.96,
    bridge_config: "BridgeConfig | None" = None,
    score_source: str = "archive_moment",
    draws: pd.DataFrame | None = None,
    predictive_component_cache: PredictiveComponentCache | None = None,
) -> pd.DataFrame:
    contract = _predictive_contract(bridge_config)
    source = _validate_score_source(score_source)
    if contract != alternate_ARCHIVE_MOMENT and bridge_config is None:
        raise ValueError("non-alternate predictive contracts require bridge_config")
    if (
        contract != alternate_ARCHIVE_MOMENT
        and source == "draw_kernel"
        and draws is None
    ):
        raise ValueError("draw-kernel predictive contracts require forecast draws")
    draws_by_forecast = (
        _index_draws_by_forecast(draws)
        if (
            contract != alternate_ARCHIVE_MOMENT
            and contract not in CENSORED_PREDICTIVE_CONTRACTS
            and source == "draw_kernel"
        )
        else {}
    )
    total_model_count = int(model_weights["model_id"].astype(str).nunique())
    w = model_weights[["model_id", "weight"]].copy()
    unavailable = forecast_unavailable_mask(
        archive,
        require_provenance="forecast_fallback_used" in archive.columns,
    )
    native_archive = archive.loc[~unavailable].copy()
    context_columns = ["forecast_id", "mode", "component", "horizon"]
    context = ledger[
        [column for column in context_columns if column in ledger.columns]
    ].drop_duplicates("forecast_id")
    for column in context_columns[1:]:
        if column not in native_archive.columns and column in context.columns:
            native_archive = native_archive.merge(
                context[["forecast_id", column]],
                on="forecast_id",
                how="left",
                validate="many_to_one",
            )
    df = native_archive.merge(w, on="model_id", how="inner")
    if df.empty:
        raise ValueError("no native archive forecasts overlap model_weights")
    expected_ids = set(ledger["forecast_id"].astype(str))
    covered_ids = set(df["forecast_id"].astype(str))
    missing_ids = sorted(expected_ids - covered_ids)
    if missing_ids:
        raise ValueError(
            "native posterior readout has no available model forecast for "
            f"{len(missing_ids)} ledger events; examples={missing_ids[:10]}"
        )
    particle_counts = df.groupby(["forecast_id", "model_id"])["particle_id"].transform("nunique").astype(float)
    df["mixture_weight"] = df["weight"].astype(float) / particle_counts
    if (
        contract in CENSORED_PREDICTIVE_CONTRACTS
        and source == "archive_moment"
        and len(df) >= 64
    ):
        assert bridge_config is not None
        out = _censored_posterior_moment_readout_batch(
            df,
            total_model_count=total_model_count,
            bridge_config=bridge_config,
            predictive_component_cache=predictive_component_cache,
        )
        meta_cols = [
            "forecast_id",
            "protocol_version",
            "natural_event_id",
            "dataset",
            "entity_id",
            "mode",
            "mode_kind",
            "forecast_strategy",
            "revision_version",
            "country",
            "country_code",
            "jurisdiction",
            "pathogen",
            "target_variable",
            "target",
            "anchor_week",
            "forecast_origin",
            "target_time",
            "release_time",
            "component",
            "horizon",
            "observed_value",
            "observed_mask",
            "split",
        ]
        meta = ledger[
            [column for column in meta_cols if column in ledger.columns]
        ].drop_duplicates("forecast_id")
        meta["forecast_id"] = meta["forecast_id"].astype(str)
        return meta.merge(
            out, on="forecast_id", how="inner", validate="one_to_one"
        )
    if (
        contract in CENSORED_PREDICTIVE_CONTRACTS
        and source == "draw_kernel"
        and len(df) >= 64
    ):
        assert bridge_config is not None
        assert draws is not None
        out = _censored_posterior_draw_readout_batch(
            df,
            draws,
            total_model_count=total_model_count,
            bridge_config=bridge_config,
            predictive_component_cache=predictive_component_cache,
        )
        meta_cols = [
            "forecast_id",
            "protocol_version",
            "natural_event_id",
            "dataset",
            "entity_id",
            "mode",
            "mode_kind",
            "forecast_strategy",
            "revision_version",
            "country",
            "country_code",
            "jurisdiction",
            "pathogen",
            "target_variable",
            "target",
            "anchor_week",
            "forecast_origin",
            "target_time",
            "release_time",
            "component",
            "horizon",
            "observed_value",
            "observed_mask",
            "split",
        ]
        meta = ledger[
            [column for column in meta_cols if column in ledger.columns]
        ].drop_duplicates("forecast_id")
        meta["forecast_id"] = meta["forecast_id"].astype(str)
        return meta.merge(
            out, on="forecast_id", how="inner", validate="one_to_one"
        )
    if (
        contract in CENSORED_PREDICTIVE_CONTRACTS
        and source == "draw_kernel"
    ):
        draws_by_forecast = _index_draws_by_forecast(draws)
    grouped = []
    for fid, g in df.groupby("forecast_id"):
        draw_positions = draws_by_forecast.get(str(fid))
        forecast_draws = (
            draws.iloc[draw_positions]
            if draws is not None and draw_positions is not None
            else None
        )
        weights = g["mixture_weight"].astype(float).to_numpy()
        available_mass = float(weights.sum())
        if not np.isfinite(available_mass) or available_mass <= 0.0:
            raise ValueError(
                f"forecast_id={fid} has no positive finite posterior mass on native models"
            )
        weights = weights / available_mass
        mu = g["pred_mean"].astype(float).to_numpy()
        var = g["pred_var"].astype(float).to_numpy()
        mean = float(np.sum(weights * mu))
        second = float(np.sum(weights * (var + mu * mu)))
        pred_var = max(0.0, second - mean * mean)
        active_model_count = int(g["model_id"].astype(str).nunique())
        row = {
            "forecast_id": fid,
            "predictive_mean": mean,
            "predictive_var": pred_var,
            "lower_95": max(0.0, mean - interval_z * np.sqrt(pred_var)),
            "upper_95": max(0.0, mean + interval_z * np.sqrt(pred_var)),
            "available_model_count": active_model_count,
            "masked_model_count": total_model_count - active_model_count,
            "availability_mask_applied": active_model_count < total_model_count,
        }
        if contract == ARCHIVE_MEAN_BRIDGE_QUANTILES:
            assert bridge_config is not None
            row.update(
                _archive_mean_bridge_interval_fields(
                    g,
                    bridge_config,
                    score_source=source,
                    draws=forecast_draws,
                )
            )
            row.update(
                {
                    "predictive_contract": contract,
                    "predictive_mean_source": (
                        "posterior_weighted_archived_raw_means"
                    ),
                    "predictive_interval_source": (
                        f"posterior_{source}_bridge_mixture_quantiles"
                    ),
                }
            )
        elif contract == alternate_ARCHIVE_MOMENT:
                                                                           
                                                                    
            pass
        elif contract in CENSORED_PREDICTIVE_CONTRACTS:
            assert bridge_config is not None
            row.update(
                _censored_predictive_fields(
                    g,
                    bridge_config,
                    score_source=source,
                    forecast_draws=forecast_draws,
                    predictive_component_cache=predictive_component_cache,
                )
            )
        else:
            assert bridge_config is not None
            row.update(
                _coherent_predictive_fields(
                    g,
                    bridge_config,
                    score_source=source,
                    forecast_draws=forecast_draws,
                    predictive_component_cache=predictive_component_cache,
                )
            )
        grouped.append(row)
    out = pd.DataFrame(grouped)
    meta_cols = [
        "forecast_id",
        "protocol_version",
        "natural_event_id",
        "dataset",
        "entity_id",
        "mode",
        "mode_kind",
        "forecast_strategy",
        "revision_version",
        "country",
        "country_code",
        "jurisdiction",
        "pathogen",
        "target_variable",
        "target",
        "anchor_week",
        "forecast_origin",
        "target_time",
        "release_time",
        "component",
        "horizon",
        "observed_value",
        "observed_mask",
        "split",
    ]
    meta = ledger[[c for c in meta_cols if c in ledger.columns]].drop_duplicates("forecast_id")
    return meta.merge(out, on="forecast_id", how="inner")


def _censored_single_center_summaries(
    location: np.ndarray,
    scale: np.ndarray,
    nu: np.ndarray,
    upper_raw: np.ndarray,
    *,
    quadrature_order: int,
    include_component_quantiles: bool = True,
) -> dict[str, np.ndarray]:
    ""

    from caster.bridge.likelihood import (
        _coherent_student_t_log_density_array,
        _coherent_transform_quadrature_batch,
    )

    location = np.asarray(location, dtype=float)
    scale = np.asarray(scale, dtype=float)
    nu = np.asarray(nu, dtype=float)
    upper_raw = np.asarray(upper_raw, dtype=float)
    count = len(location)
    means = np.empty(count, dtype=float)
    seconds = np.empty(count, dtype=float)
    lower_atom = np.empty(count, dtype=float)
    upper_atom = np.empty(count, dtype=float)
    order = int(quadrature_order)
    block_size = max(64, 262_144 // order)
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        sl = slice(start, stop)
        lower = np.zeros(stop - start, dtype=float)
        z, raw, log_measure, _, upper_z = (
            _coherent_transform_quadrature_batch(
                lower, upper_raw[sl], order
            )
        )
        log_density = _coherent_student_t_log_density_array(
            z,
            location[sl, None],
            scale[sl, None],
            nu[sl, None],
        )
        interior_weight = np.exp(log_density + log_measure)
        p0 = student_t.cdf(
            -location[sl] / scale[sl], df=nu[sl]
        )
        p_upper = student_t.sf(
            (upper_z - location[sl]) / scale[sl],
            df=nu[sl],
        )
        interior_probability = np.maximum(
            0.0, 1.0 - p0 - p_upper
        )
        numerical_mass = interior_weight.sum(axis=1)
        if np.any(
            (interior_probability > 0.0)
            & (
                ~np.isfinite(numerical_mass)
                | (numerical_mass <= 0.0)
            )
        ):
            raise FloatingPointError(
                "censored batch quadrature has no finite interior mass"
            )
        multiplier = np.divide(
            interior_probability,
            numerical_mass,
            out=np.zeros_like(interior_probability),
            where=numerical_mass > 0.0,
        )
        interior_weight *= multiplier[:, None]
        means[sl] = (
            np.sum(interior_weight * raw, axis=1)
            + p_upper * upper_raw[sl]
        )
        seconds[sl] = (
            np.sum(interior_weight * np.square(raw), axis=1)
            + p_upper * np.square(upper_raw[sl])
        )
        lower_atom[sl] = p0
        upper_atom[sl] = p_upper

    result = {
        "mean": means,
        "second_moment": seconds,
        "variance": np.maximum(seconds - np.square(means), 0.0),
        "lower_atom_probability": lower_atom,
        "upper_atom_probability": upper_atom,
    }
    if include_component_quantiles:
        probability = np.asarray(
            CENSORED_READOUT_PROBABILITIES, dtype=float
        )
        latent_quantiles = (
            location[:, None]
            + scale[:, None]
            * student_t.ppf(probability[None, :], df=nu[:, None])
        )
        raw_quantiles = np.expm1(latent_quantiles)
        all_quantiles = np.minimum(
            np.maximum(raw_quantiles, 0.0), upper_raw[:, None]
        )
        result["quantiles"] = all_quantiles[
            :, _CENSORED_INTERVAL_QUANTILE_INDICES
        ]
        result["median"] = all_quantiles[
            :, _CENSORED_MEDIAN_QUANTILE_INDEX
        ]
    return result


def _censored_group_mixture_quantiles_batch(
    *,
    group_code: np.ndarray,
    group_count: int,
    component_weight: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    nu: np.ndarray,
    upper_raw_by_group: np.ndarray,
    lower_atom_by_group: np.ndarray,
    upper_atom_by_group: np.ndarray,
    probabilities: tuple[float, ...] = INTERVAL_PROBABILITIES,
) -> np.ndarray:
    ""

    from scipy.special import gammaln, stdtr

    probabilities_array = np.asarray(probabilities, dtype=float)
    output = np.empty((group_count, len(probabilities_array)), dtype=float)
    upper_z = np.log1p(upper_raw_by_group)
    log_density_normalizer = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(scale)
    )
    for column, probability in enumerate(probabilities_array):
        at_lower = probability <= lower_atom_by_group
        at_upper = probability > 1.0 - upper_atom_by_group
        interior = ~(at_lower | at_upper)
        output[at_lower, column] = 0.0
        output[at_upper, column] = upper_raw_by_group[at_upper]
        if not np.any(interior):
            continue
        lower = np.zeros(group_count, dtype=float)
        upper = upper_z.copy()
        continuous_fraction = np.divide(
            probability - lower_atom_by_group,
            np.maximum(
                1.0 - lower_atom_by_group - upper_atom_by_group,
                np.finfo(float).tiny,
            ),
        )
        current = upper_z * np.clip(continuous_fraction, 0.0, 1.0)
        active = interior.copy()
                                                                             
                                                                              
                                                                               
                                                                           
                          
        for _ in range(24):
            active_component = active[group_code]
            if not np.any(active_component):
                break
            active_code = group_code[active_component]
            standardized = (
                current[active_code] - location[active_component]
            ) / scale[active_component]
            weighted_cdf = (
                component_weight[active_component]
                * stdtr(nu[active_component], standardized)
            )
            cdf = np.bincount(
                active_code,
                weights=weighted_cdf,
                minlength=group_count,
            )
            log_density = (
                log_density_normalizer[active_component]
                - ((nu[active_component] + 1.0) / 2.0)
                * np.log1p(
                    np.square(standardized) / nu[active_component]
                )
            )
            density = np.bincount(
                active_code,
                weights=(
                    component_weight[active_component]
                    * np.exp(log_density)
                ),
                minlength=group_count,
            )
            below = active & (cdf < probability)
            above = active & ~below
            lower[below] = current[below]
            upper[above] = current[above]
            residual = cdf - probability
            converged = active & (
                (np.abs(residual) <= 1e-12)
                | ((upper - lower) <= 1e-12)
            )
            active[converged] = False
            if not np.any(active):
                break
            midpoint = 0.5 * (lower + upper)
            newton = np.full(group_count, np.nan, dtype=float)
            usable = active & np.isfinite(density) & (density > 0.0)
            newton[usable] = (
                current[usable] - residual[usable] / density[usable]
            )
            inside = (
                usable
                & (newton > lower)
                & (newton < upper)
            )
            current[active] = midpoint[active]
            current[inside] = newton[inside]
        if np.any(active):
                                                                            
                                                                        
                                                                             
            for _ in range(48):
                active_component = active[group_code]
                active_code = group_code[active_component]
                midpoint = 0.5 * (lower + upper)
                standardized = (
                    midpoint[active_code] - location[active_component]
                ) / scale[active_component]
                cdf = np.bincount(
                    active_code,
                    weights=(
                        component_weight[active_component]
                        * stdtr(nu[active_component], standardized)
                    ),
                    minlength=group_count,
                )
                below = active & (cdf < probability)
                above = active & ~below
                lower[below] = midpoint[below]
                upper[above] = midpoint[above]
                current[active] = 0.5 * (
                    lower[active] + upper[active]
                )
                converged = active & ((upper - lower) <= 1e-12)
                active[converged] = False
                if not np.any(active):
                    break
        if np.any(active):
            bad = np.flatnonzero(active)[:5].tolist()
            raise RuntimeError(
                "censored mixture quantile bracket did not contract; "
                f"example group positions={bad}"
            )
        output[interior, column] = np.expm1(
            current[interior]
        )
    return output


def _censored_posterior_moment_readout_batch(
    df: pd.DataFrame,
    *,
    total_model_count: int,
    bridge_config: "BridgeConfig",
    predictive_component_cache: PredictiveComponentCache | None,
) -> pd.DataFrame:
    ""

    import hashlib
    from caster.bridge.likelihood import (
        _lookup_bridge_series,
        solve_censored_mean_preserving_shift_batch,
    )

    contract = str(bridge_config.predictive_contract)
    mean_preserving = (
        contract == COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T
    )

    group_code, forecast_ids = pd.factorize(
        df["forecast_id"].astype(str), sort=False
    )
    group_count = len(forecast_ids)
    raw_weight = pd.to_numeric(
        df["mixture_weight"], errors="raise"
    ).to_numpy(dtype=float)
    available_mass = np.bincount(
        group_code, weights=raw_weight, minlength=group_count
    )
    if np.any(~np.isfinite(available_mass) | (available_mass <= 0.0)):
        raise ValueError(
            "censored batch readout found non-positive available mass"
        )
    component_weight = raw_weight / available_mass[group_code]

    signature_frame = df[
        ["forecast_id", "model_id", "particle_id"]
    ].astype(str)
    row_signature = pd.util.hash_pandas_object(
        signature_frame, index=False
    ).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(row_signature.tobytes())
    digest.update(repr(bridge_config).encode("utf-8"))
    cache_key = (
        "__censored_moment_batch__",
        str(len(df)),
        digest.hexdigest(),
        contract,
    )
    cached = (
        None
        if predictive_component_cache is None
        else predictive_component_cache.get(cache_key)
    )
    if (
        isinstance(cached, dict)
        and np.array_equal(cached.get("row_signature"), row_signature)
    ):
        component = cached
    else:
        location, scale, nu = _archive_bridge_parameters(
            df, bridge_config
        )
        mode = (
            df["mode"]
            if "mode" in df.columns
            else pd.Series("", index=df.index, dtype=str)
        )
        upper_raw = _lookup_bridge_series(
            bridge_config.truncation_upper_raw_by_component,
            mode,
            df["component"],
            df["horizon"],
            bridge_config.default_truncation_upper_raw,
        )
        mean_solution: dict[str, np.ndarray] | None = None
        if mean_preserving:
            requested_mean = np.maximum(
                pd.to_numeric(
                    df["pred_mean"], errors="raise"
                ).to_numpy(dtype=float),
                0.0,
            )
            mean_solution = solve_censored_mean_preserving_shift_batch(
                location[:, None],
                np.ones((len(location), 1), dtype=bool),
                requested_mean,
                scale,
                nu,
                upper_raw,
                mean_floor=float(
                    bridge_config.truncation_zero_mean_epsilon
                ),
                quadrature_order=int(
                    bridge_config.truncation_quadrature_order
                ),
            )
            location = location + mean_solution["shift"]
        if mean_solution is not None:
                                                                               
                                                                           
                                                                             
                                                                          
            upper_z = np.log1p(upper_raw)
            summary = {
                "mean": mean_solution["mean"],
                "second_moment": mean_solution["second_moment"],
                "variance": mean_solution["variance"],
                "lower_atom_probability": student_t.cdf(
                    -location / scale, df=nu
                ),
                "upper_atom_probability": student_t.sf(
                    (upper_z - location) / scale, df=nu
                ),
            }
        else:
            summary = _censored_single_center_summaries(
                location,
                scale,
                nu,
                upper_raw,
                quadrature_order=int(
                    bridge_config.truncation_quadrature_order
                ),
                include_component_quantiles=False,
            )
        component = {
            "row_signature": row_signature.copy(),
            "location": location,
            "scale": scale,
            "nu": nu,
            "upper_raw": upper_raw,
            **summary,
        }
        if mean_solution is not None:
            component.update(
                {
                    "requested_mean": mean_solution["requested_mean"],
                    "effective_target": mean_solution["effective_target"],
                    "mean_floor_applied": mean_solution[
                        "mean_floor_applied"
                    ],
                    "location_shift": mean_solution["shift"],
                    "mean_constraint_residual": mean_solution["residual"],
                    "mean_solver_iterations": mean_solution["iterations"],
                }
            )
        if predictive_component_cache is not None:
            predictive_component_cache[cache_key] = component

    mean = np.bincount(
        group_code,
        weights=component_weight * component["mean"],
        minlength=group_count,
    )
    second = np.bincount(
        group_code,
        weights=component_weight * component["second_moment"],
        minlength=group_count,
    )
    lower_atom = np.bincount(
        group_code,
        weights=(
            component_weight * component["lower_atom_probability"]
        ),
        minlength=group_count,
    )
    upper_atom = np.bincount(
        group_code,
        weights=(
            component_weight * component["upper_atom_probability"]
        ),
        minlength=group_count,
    )
    upper_min = np.full(group_count, np.inf, dtype=float)
    upper_max = np.full(group_count, -np.inf, dtype=float)
    np.minimum.at(upper_min, group_code, component["upper_raw"])
    np.maximum.at(upper_max, group_code, component["upper_raw"])
    if not np.allclose(upper_min, upper_max, rtol=0.0, atol=1e-12):
        raise ValueError(
            "censored posterior mixture components require one common upper bound"
        )
    quantiles = _censored_group_mixture_quantiles_batch(
        group_code=group_code,
        group_count=group_count,
        component_weight=component_weight,
        location=component["location"],
        scale=component["scale"],
        nu=component["nu"],
        upper_raw_by_group=upper_max,
        lower_atom_by_group=lower_atom,
        upper_atom_by_group=upper_atom,
        probabilities=CENSORED_READOUT_PROBABILITIES,
    )
    active_model_count = (
        df.assign(__group_code__=group_code)
        .groupby("__group_code__", sort=True)["model_id"]
        .nunique()
        .reindex(range(group_count), fill_value=0)
        .to_numpy(dtype=int)
    )
    component_count = np.bincount(
        group_code, minlength=group_count
    ).astype(int)
    out = pd.DataFrame(
        {
            "forecast_id": forecast_ids.astype(str),
            "predictive_mean": mean,
            "predictive_median": quantiles[
                :, _CENSORED_MEDIAN_QUANTILE_INDEX
            ],
            "predictive_var": np.maximum(
                second - np.square(mean), 0.0
            ),
            "available_model_count": active_model_count,
            "masked_model_count": total_model_count - active_model_count,
            "availability_mask_applied": (
                active_model_count < total_model_count
            ),
            "predictive_contract": contract,
            "predictive_mean_source": (
                "censored_latent_student_t_posterior_mixture_expectation"
            ),
            "predictive_interval_source": (
                "censored_archive_moment_posterior_mixture_quantiles"
            ),
            "nll_measure_basis": (
                "log1p_transform_mixed_measure_atoms_at_zero_and_upper_bound"
            ),
            "boundary_atom_source": "latent_student_t_censoring",
            "lower_boundary_atom_probability": lower_atom,
            "upper_boundary_atom_probability": upper_atom,
            "truncation_component_count": component_count,
            "truncation_lower_raw": 0.0,
            "truncation_upper_raw_frozen_min": upper_min,
            "truncation_upper_raw_frozen_max": upper_max,
            "truncation_upper_raw_effective_min": upper_min,
            "truncation_upper_raw_effective_max": upper_max,
            "truncation_upper_raw_min": upper_min,
            "truncation_upper_raw_max": upper_max,
            "truncation_bound_policy": str(
                bridge_config.truncation_bound_policy
            ),
            "truncation_support_expansion_policy": (
                "strict_frozen_support"
            ),
            "truncation_support_expanded": False,
            "truncation_support_expanded_count": 0,
            "truncation_quadrature_order": int(
                bridge_config.truncation_quadrature_order
            ),
        }
    )
    for index, column in zip(
        _CENSORED_INTERVAL_QUANTILE_INDICES,
        [
            "lower_95",
            "lower_90",
            "lower_50",
            "upper_50",
            "upper_90",
            "upper_95",
        ],
    ):
        out[column] = quantiles[:, index]
    if mean_preserving:
        out["truncation_mean_floor"] = float(
            bridge_config.truncation_zero_mean_epsilon
        )
        floor_count = np.bincount(
            group_code,
            weights=component["mean_floor_applied"].astype(float),
            minlength=group_count,
        ).astype(int)
        out["truncation_mean_floor_applied_count"] = floor_count
        shift_min = np.full(group_count, np.inf, dtype=float)
        shift_max = np.full(group_count, -np.inf, dtype=float)
        np.minimum.at(
            shift_min, group_code, component["location_shift"]
        )
        np.maximum.at(
            shift_max, group_code, component["location_shift"]
        )
        out["censored_location_shift_min"] = shift_min
        out["censored_location_shift_max"] = shift_max
        residual_max = np.zeros(group_count, dtype=float)
        np.maximum.at(
            residual_max,
            group_code,
            np.abs(component["mean_constraint_residual"]),
        )
        out["censored_mean_constraint_max_abs_residual"] = residual_max
    return out


def _censored_draw_component_summaries_batch(
    *,
    centers: np.ndarray,
    center_mask: np.ndarray,
    finite_count: np.ndarray,
    scale: np.ndarray,
    nu: np.ndarray,
    upper_raw: np.ndarray,
    quadrature_order: int,
) -> dict[str, np.ndarray]:
    ""

    from caster.bridge.likelihood import (
        _coherent_logsumexp,
        _coherent_student_t_log_density_array,
        _coherent_transform_quadrature_batch,
    )

    component_count, max_draw_count = centers.shape
    means = np.empty(component_count, dtype=float)
    seconds = np.empty(component_count, dtype=float)
    lower_atom = np.empty(component_count, dtype=float)
    upper_atom = np.empty(component_count, dtype=float)
    order = int(quadrature_order)
    chunk_size = max(
        8,
        min(2048, 1_048_576 // (order * max_draw_count)),
    )
    for start in range(0, component_count, chunk_size):
        stop = min(start + chunk_size, component_count)
        sl = slice(start, stop)
        lower = np.zeros(stop - start, dtype=float)
        z, raw, log_measure, _, upper_z = (
            _coherent_transform_quadrature_batch(
                lower, upper_raw[sl], order
            )
        )
        kernel_log_density = _coherent_student_t_log_density_array(
            z[:, :, None],
            centers[sl, None, :],
            scale[sl, None, None],
            nu[sl, None, None],
        )
        kernel_log_density = np.where(
            center_mask[sl, None, :],
            kernel_log_density,
            float("-inf"),
        )
        base_log_density = (
            _coherent_logsumexp(kernel_log_density, axis=2)
            - np.log(finite_count[sl])[:, None]
        )
        interior_weight = np.exp(base_log_density + log_measure)
        standardized_lower = (
            -centers[sl] / scale[sl, None]
        )
        standardized_upper = (
            upper_z[:, None] - centers[sl]
        ) / scale[sl, None]
        p0_term = np.where(
            center_mask[sl],
            student_t.cdf(
                standardized_lower, df=nu[sl, None]
            ),
            0.0,
        )
        p_upper_term = np.where(
            center_mask[sl],
            student_t.sf(
                standardized_upper, df=nu[sl, None]
            ),
            0.0,
        )
        p0 = p0_term.sum(axis=1) / finite_count[sl]
        p_upper = p_upper_term.sum(axis=1) / finite_count[sl]
        interior_probability = np.maximum(
            0.0, 1.0 - p0 - p_upper
        )
        numerical_mass = interior_weight.sum(axis=1)
        if np.any(
            (interior_probability > 0.0)
            & (
                ~np.isfinite(numerical_mass)
                | (numerical_mass <= 0.0)
            )
        ):
            raise FloatingPointError(
                "censored draw batch quadrature has no finite interior mass"
            )
        multiplier = np.divide(
            interior_probability,
            numerical_mass,
            out=np.zeros_like(interior_probability),
            where=numerical_mass > 0.0,
        )
        interior_weight *= multiplier[:, None]
        means[sl] = (
            np.sum(interior_weight * raw, axis=1)
            + p_upper * upper_raw[sl]
        )
        seconds[sl] = (
            np.sum(interior_weight * np.square(raw), axis=1)
            + p_upper * np.square(upper_raw[sl])
        )
        lower_atom[sl] = p0
        upper_atom[sl] = p_upper
    return {
        "mean": means,
        "second_moment": seconds,
        "variance": np.maximum(seconds - np.square(means), 0.0),
        "lower_atom_probability": lower_atom,
        "upper_atom_probability": upper_atom,
    }


def _censored_posterior_draw_readout_batch(
    df: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    total_model_count: int,
    bridge_config: "BridgeConfig",
    predictive_component_cache: PredictiveComponentCache | None,
) -> pd.DataFrame:
    ""

    import hashlib
    from caster.bridge.likelihood import (
        _lookup_bridge_series,
        solve_censored_mean_preserving_shift_batch,
    )

    contract = str(bridge_config.predictive_contract)
    mean_preserving = (
        contract == COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T
    )

    keys = ["forecast_id", "model_id", "particle_id"]
    component_rows = df.reset_index(drop=True).copy()
    for key in keys:
        component_rows[key] = component_rows[key].astype(str)
    component_rows["__component_position__"] = np.arange(
        len(component_rows), dtype=int
    )
    draw_rows = draws[[*keys, "draw"]].copy()
    for key in keys:
        draw_rows[key] = draw_rows[key].astype(str)
    draw_rows["draw"] = pd.to_numeric(
        draw_rows["draw"], errors="coerce"
    )
    draw_rows = draw_rows[np.isfinite(draw_rows["draw"])].merge(
        component_rows[[*keys, "__component_position__"]],
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    if draw_rows.empty:
        raise ValueError(
            "censored draw batch has no finite matching forecast draws"
        )
    component_position = draw_rows[
        "__component_position__"
    ].to_numpy(dtype=int)
    finite_count = np.bincount(
        component_position, minlength=len(component_rows)
    ).astype(int)
    if np.any(finite_count == 0):
        bad = np.flatnonzero(finite_count == 0)[:5]
        examples = component_rows.iloc[bad][keys].to_dict("records")
        raise ValueError(
            "censored draw batch is missing finite component draws; "
            f"examples={examples}"
        )

    signature_frame = component_rows[keys].astype(str)
    row_signature = pd.util.hash_pandas_object(
        signature_frame, index=False
    ).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(row_signature.tobytes())
    digest.update(repr(bridge_config).encode("utf-8"))
    cache_key = (
        "__censored_draw_batch__",
        str(len(component_rows)),
        digest.hexdigest(),
        contract,
    )
    cached = (
        None
        if predictive_component_cache is None
        else predictive_component_cache.get(cache_key)
    )
    if (
        isinstance(cached, dict)
        and np.array_equal(cached.get("row_signature"), row_signature)
    ):
        component = cached
    else:
        mode = (
            component_rows["mode"]
            if "mode" in component_rows.columns
            else pd.Series("", index=component_rows.index, dtype=str)
        )
        if bridge_config.tau_by_component:
            scale = _lookup_bridge_series(
                bridge_config.tau_by_component,
                mode,
                component_rows["component"],
                component_rows["horizon"],
                bridge_config.default_tau,
            )
        else:
            scale = _lookup_bridge_series(
                bridge_config.sigma_by_component,
                mode,
                component_rows["component"],
                component_rows["horizon"],
                bridge_config.default_sigma,
            )
        scale = np.maximum(
            np.asarray(scale, dtype=float),
            max(float(bridge_config.min_scale), 1e-6),
        )
        nu = _lookup_bridge_series(
            bridge_config.nu_by_component,
            mode,
            component_rows["component"],
            component_rows["horizon"],
            (
                bridge_config.nu
                if bridge_config.nu_by_component
                else bridge_config.kernel_nu
            ),
        )
        if np.any(~np.isfinite(nu) | (nu <= 0.0)):
            raise ValueError(
                "censored draw batch requires finite positive Student-t nu"
            )
        upper_raw = _lookup_bridge_series(
            bridge_config.truncation_upper_raw_by_component,
            mode,
            component_rows["component"],
            component_rows["horizon"],
            bridge_config.default_truncation_upper_raw,
        )
        max_draw_count = int(finite_count.max())
        centers = np.zeros(
            (len(component_rows), max_draw_count), dtype=float
        )
        center_mask = np.zeros_like(centers, dtype=bool)
        order = np.argsort(component_position, kind="stable")
        sorted_component = component_position[order]
        starts = np.repeat(
            np.cumsum(finite_count) - finite_count, finite_count
        )
        within_component = np.arange(len(draw_rows)) - starts
        centers[sorted_component, within_component] = np.log1p(
            np.maximum(
                draw_rows["draw"].to_numpy(dtype=float)[order],
                0.0,
            )
        )
        center_mask[sorted_component, within_component] = True
        mean_solution: dict[str, np.ndarray] | None = None
        if mean_preserving:
            raw_draw_values = np.maximum(
                draw_rows["draw"].to_numpy(dtype=float), 0.0
            )
            requested_mean = np.bincount(
                component_position,
                weights=raw_draw_values,
                minlength=len(component_rows),
            ) / finite_count
            mean_solution = solve_censored_mean_preserving_shift_batch(
                centers,
                center_mask,
                requested_mean,
                scale,
                nu,
                upper_raw,
                mean_floor=float(
                    bridge_config.truncation_zero_mean_epsilon
                ),
                quadrature_order=int(
                    bridge_config.truncation_quadrature_order
                ),
            )
            centers = np.where(
                center_mask,
                centers + mean_solution["shift"][:, None],
                0.0,
            )
        if mean_solution is not None:
                                                                              
                                                                             
                                                                
            standardized_lower = -centers / scale[:, None]
            standardized_upper = (
                np.log1p(upper_raw)[:, None] - centers
            ) / scale[:, None]
            lower_term = np.where(
                center_mask,
                student_t.cdf(standardized_lower, df=nu[:, None]),
                0.0,
            )
            upper_term = np.where(
                center_mask,
                student_t.sf(standardized_upper, df=nu[:, None]),
                0.0,
            )
            summary = {
                "mean": mean_solution["mean"],
                "second_moment": mean_solution["second_moment"],
                "variance": mean_solution["variance"],
                "lower_atom_probability": (
                    lower_term.sum(axis=1) / finite_count
                ),
                "upper_atom_probability": (
                    upper_term.sum(axis=1) / finite_count
                ),
            }
        else:
            summary = _censored_draw_component_summaries_batch(
                centers=centers,
                center_mask=center_mask,
                finite_count=finite_count,
                scale=scale,
                nu=nu,
                upper_raw=upper_raw,
                quadrature_order=int(
                    bridge_config.truncation_quadrature_order
                ),
            )
        component = {
            "row_signature": row_signature.copy(),
            "centers": centers,
            "center_mask": center_mask,
            "finite_count": finite_count,
            "scale": scale,
            "nu": nu,
            "upper_raw": upper_raw,
            **summary,
        }
        if mean_solution is not None:
            component.update(
                {
                    "requested_mean": mean_solution["requested_mean"],
                    "effective_target": mean_solution["effective_target"],
                    "mean_floor_applied": mean_solution[
                        "mean_floor_applied"
                    ],
                    "location_shift": mean_solution["shift"],
                    "mean_constraint_residual": mean_solution["residual"],
                    "mean_solver_iterations": mean_solution["iterations"],
                }
            )
        if predictive_component_cache is not None:
            predictive_component_cache[cache_key] = component

    forecast_group_code, forecast_ids = pd.factorize(
        component_rows["forecast_id"], sort=False
    )
    group_count = len(forecast_ids)
    raw_weight = pd.to_numeric(
        component_rows["mixture_weight"], errors="raise"
    ).to_numpy(dtype=float)
    available_mass = np.bincount(
        forecast_group_code,
        weights=raw_weight,
        minlength=group_count,
    )
    if np.any(~np.isfinite(available_mass) | (available_mass <= 0.0)):
        raise ValueError(
            "censored draw batch found non-positive available mass"
        )
    component_weight = (
        raw_weight / available_mass[forecast_group_code]
    )
    mean = np.bincount(
        forecast_group_code,
        weights=component_weight * component["mean"],
        minlength=group_count,
    )
    second = np.bincount(
        forecast_group_code,
        weights=component_weight * component["second_moment"],
        minlength=group_count,
    )
    lower_atom = np.bincount(
        forecast_group_code,
        weights=(
            component_weight * component["lower_atom_probability"]
        ),
        minlength=group_count,
    )
    upper_atom = np.bincount(
        forecast_group_code,
        weights=(
            component_weight * component["upper_atom_probability"]
        ),
        minlength=group_count,
    )
    upper_min = np.full(group_count, np.inf, dtype=float)
    upper_max = np.full(group_count, -np.inf, dtype=float)
    np.minimum.at(
        upper_min, forecast_group_code, component["upper_raw"]
    )
    np.maximum.at(
        upper_max, forecast_group_code, component["upper_raw"]
    )
    if not np.allclose(upper_min, upper_max, rtol=0.0, atol=1e-12):
        raise ValueError(
            "censored draw posterior mixture requires one common upper bound"
        )

    center_component = np.repeat(
        np.arange(len(component_rows), dtype=int),
        component["centers"].shape[1],
    )[component["center_mask"].reshape(-1)]
    center_location = component["centers"].reshape(-1)[
        component["center_mask"].reshape(-1)
    ]
    center_forecast_group = forecast_group_code[center_component]
    center_weight = (
        component_weight[center_component]
        / component["finite_count"][center_component]
    )
    quantiles = _censored_group_mixture_quantiles_batch(
        group_code=center_forecast_group,
        group_count=group_count,
        component_weight=center_weight,
        location=center_location,
        scale=component["scale"][center_component],
        nu=component["nu"][center_component],
        upper_raw_by_group=upper_max,
        lower_atom_by_group=lower_atom,
        upper_atom_by_group=upper_atom,
        probabilities=CENSORED_READOUT_PROBABILITIES,
    )
    active_model_count = (
        component_rows.assign(
            __group_code__=forecast_group_code
        )
        .groupby("__group_code__", sort=True)["model_id"]
        .nunique()
        .reindex(range(group_count), fill_value=0)
        .to_numpy(dtype=int)
    )
    predictive_count = np.bincount(
        forecast_group_code, minlength=group_count
    ).astype(int)
    out = pd.DataFrame(
        {
            "forecast_id": forecast_ids.astype(str),
            "predictive_mean": mean,
            "predictive_median": quantiles[
                :, _CENSORED_MEDIAN_QUANTILE_INDEX
            ],
            "predictive_var": np.maximum(
                second - np.square(mean), 0.0
            ),
            "available_model_count": active_model_count,
            "masked_model_count": total_model_count - active_model_count,
            "availability_mask_applied": (
                active_model_count < total_model_count
            ),
            "predictive_contract": contract,
            "predictive_mean_source": (
                "censored_latent_student_t_posterior_mixture_expectation"
            ),
            "predictive_interval_source": (
                "censored_draw_kernel_posterior_mixture_quantiles"
            ),
            "nll_measure_basis": (
                "log1p_transform_mixed_measure_atoms_at_zero_and_upper_bound"
            ),
            "boundary_atom_source": "latent_student_t_censoring",
            "lower_boundary_atom_probability": lower_atom,
            "upper_boundary_atom_probability": upper_atom,
            "truncation_component_count": predictive_count,
            "truncation_lower_raw": 0.0,
            "truncation_upper_raw_frozen_min": upper_min,
            "truncation_upper_raw_frozen_max": upper_max,
            "truncation_upper_raw_effective_min": upper_min,
            "truncation_upper_raw_effective_max": upper_max,
            "truncation_upper_raw_min": upper_min,
            "truncation_upper_raw_max": upper_max,
            "truncation_bound_policy": str(
                bridge_config.truncation_bound_policy
            ),
            "truncation_support_expansion_policy": (
                "strict_frozen_support"
            ),
            "truncation_support_expanded": False,
            "truncation_support_expanded_count": 0,
            "truncation_quadrature_order": int(
                bridge_config.truncation_quadrature_order
            ),
        }
    )
    for index, column in zip(
        _CENSORED_INTERVAL_QUANTILE_INDICES,
        [
            "lower_95",
            "lower_90",
            "lower_50",
            "upper_50",
            "upper_90",
            "upper_95",
        ],
    ):
        out[column] = quantiles[:, index]
    if mean_preserving:
        out["truncation_mean_floor"] = float(
            bridge_config.truncation_zero_mean_epsilon
        )
        floor_count = np.bincount(
            forecast_group_code,
            weights=component["mean_floor_applied"].astype(float),
            minlength=group_count,
        ).astype(int)
        out["truncation_mean_floor_applied_count"] = floor_count
        shift_min = np.full(group_count, np.inf, dtype=float)
        shift_max = np.full(group_count, -np.inf, dtype=float)
        np.minimum.at(
            shift_min, forecast_group_code, component["location_shift"]
        )
        np.maximum.at(
            shift_max, forecast_group_code, component["location_shift"]
        )
        out["censored_location_shift_min"] = shift_min
        out["censored_location_shift_max"] = shift_max
        residual_max = np.zeros(group_count, dtype=float)
        np.maximum.at(
            residual_max,
            forecast_group_code,
            np.abs(component["mean_constraint_residual"]),
        )
        out["censored_mean_constraint_max_abs_residual"] = residual_max
    return out


def _aggregate_equal_particle_log_scores(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    ""

    from scipy.special import logsumexp

    keys = ["forecast_id", "model_id"]
    required = {*keys, "log_score"}
    if missing := sorted(required - set(scored.columns)):
        raise ValueError(
            f"single-model score rows missing columns {missing}"
        )
    if not scored.duplicated(keys, keep=False).any():
        result = scored[[*keys, "log_score"]].copy()
        result["forecast_id"] = result["forecast_id"].astype(str)
        result["model_id"] = result["model_id"].astype(str)
        return result
    rows: list[dict[str, object]] = []
    for key, group in scored.groupby(keys, sort=False):
        values = pd.to_numeric(
            group["log_score"], errors="raise"
        ).to_numpy(dtype=float)
        rows.append(
            {
                "forecast_id": str(key[0]),
                "model_id": str(key[1]),
                "log_score": float(logsumexp(values) - np.log(len(values))),
            }
        )
    return pd.DataFrame(rows)


def single_model_predictive_readout(
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    bridge_config: "BridgeConfig",
    *,
    score_source: str = "archive_moment",
    draws: pd.DataFrame | None = None,
    predictive_component_cache: PredictiveComponentCache | None = None,
) -> pd.DataFrame:
    ""






    from caster.bridge.likelihood import score_archive_rows, score_draw_rows

    contract = _predictive_contract(bridge_config)
    source = _validate_score_source(score_source)
    unavailable = forecast_unavailable_mask(
        archive,
        require_provenance="forecast_fallback_used" in archive.columns,
    )
    native_archive = archive.loc[~unavailable].copy()
    if native_archive.empty:
        raise ValueError("single-model readout has no native archive rows")
    for key in ("forecast_id", "model_id", "particle_id"):
        native_archive[key] = native_archive[key].astype(str)
    context_columns = ["forecast_id", "mode", "component", "horizon"]
    context = ledger[
        [column for column in context_columns if column in ledger.columns]
    ].drop_duplicates("forecast_id")
    context["forecast_id"] = context["forecast_id"].astype(str)
    for column in context_columns[1:]:
        if column not in native_archive.columns and column in context.columns:
            native_archive = native_archive.merge(
                context[["forecast_id", column]],
                on="forecast_id",
                how="left",
                validate="many_to_one",
            )
    group_sizes = native_archive.groupby(
        ["forecast_id", "model_id"], sort=False
    ).size()
    singleton = bool(group_sizes.eq(1).all())

    if (
        contract in CENSORED_PREDICTIVE_CONTRACTS
        and source == "archive_moment"
        and singleton
    ):
        from caster.bridge.likelihood import (
            _lookup_bridge_series,
            solve_censored_mean_preserving_shift_batch,
        )

        location, scale, nu = _archive_bridge_parameters(
            native_archive, bridge_config
        )
        mode = (
            native_archive["mode"]
            if "mode" in native_archive.columns
            else pd.Series("", index=native_archive.index, dtype=str)
        )
        upper_raw = _lookup_bridge_series(
            bridge_config.truncation_upper_raw_by_component,
            mode,
            native_archive["component"],
            native_archive["horizon"],
            bridge_config.default_truncation_upper_raw,
        )
        mean_solution: dict[str, np.ndarray] | None = None
        if (
            contract
            == COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T
        ):
            requested_mean = np.maximum(
                pd.to_numeric(
                    native_archive["pred_mean"], errors="raise"
                ).to_numpy(dtype=float),
                0.0,
            )
            mean_solution = solve_censored_mean_preserving_shift_batch(
                location[:, None],
                np.ones((len(location), 1), dtype=bool),
                requested_mean,
                scale,
                nu,
                upper_raw,
                mean_floor=float(
                    bridge_config.truncation_zero_mean_epsilon
                ),
                quadrature_order=int(
                    bridge_config.truncation_quadrature_order
                ),
            )
            location = location + mean_solution["shift"]
        summary = _censored_single_center_summaries(
            location,
            scale,
            nu,
            upper_raw,
            quadrature_order=int(
                bridge_config.truncation_quadrature_order
            ),
        )
        if mean_solution is not None:
            summary["mean"] = mean_solution["mean"]
            summary["second_moment"] = mean_solution["second_moment"]
            summary["variance"] = mean_solution["variance"]
        quantiles = np.asarray(summary["quantiles"], dtype=float)
        out = native_archive[
            ["forecast_id", "model_id", "particle_id"]
        ].copy()
        out["predictive_mean"] = summary["mean"]
        out["predictive_median"] = summary["median"]
        out["predictive_var"] = summary["variance"]
        for index, column in enumerate(
            [
                "lower_95",
                "lower_90",
                "lower_50",
                "upper_50",
                "upper_90",
                "upper_95",
            ]
        ):
            out[column] = quantiles[:, index]
        out["predictive_contract"] = contract
        out["predictive_mean_source"] = (
            "censored_latent_student_t_single_model_mixture_expectation"
        )
        out["predictive_interval_source"] = (
            "censored_archive_moment_single_model_mixture_quantiles"
        )
        out["nll_measure_basis"] = (
            "log1p_transform_mixed_measure_atoms_at_zero_and_upper_bound"
        )
        out["boundary_atom_source"] = "latent_student_t_censoring"
        out["lower_boundary_atom_probability"] = summary[
            "lower_atom_probability"
        ]
        out["upper_boundary_atom_probability"] = summary[
            "upper_atom_probability"
        ]
        out["truncation_lower_raw"] = 0.0
        out["truncation_upper_raw"] = upper_raw
        out["truncation_quadrature_order"] = int(
            bridge_config.truncation_quadrature_order
        )
        if mean_solution is not None:
            out["truncation_mean_floor"] = float(
                bridge_config.truncation_zero_mean_epsilon
            )
            out["truncation_mean_floor_applied_count"] = (
                mean_solution["mean_floor_applied"].astype(int)
            )
            out["censored_location_shift_min"] = mean_solution["shift"]
            out["censored_location_shift_max"] = mean_solution["shift"]
            out["censored_mean_constraint_max_abs_residual"] = np.abs(
                mean_solution["residual"]
            )
        scored = score_archive_rows(
            ledger, native_archive, bridge_config
        )
        scored["forecast_id"] = scored["forecast_id"].astype(str)
        scored["model_id"] = scored["model_id"].astype(str)
        scored["particle_id"] = scored["particle_id"].astype(str)
        out = out.merge(
            scored[
                ["forecast_id", "model_id", "particle_id", "log_score"]
            ],
            on=["forecast_id", "model_id", "particle_id"],
            how="left",
            validate="one_to_one",
        ).drop(columns=["particle_id"])
    else:
        if source == "draw_kernel" and draws is None:
            raise ValueError(
                "single-model draw-kernel readout requires forecast draws"
            )
        scored = (
            score_archive_rows(ledger, native_archive, bridge_config)
            if source == "archive_moment"
            else score_draw_rows(ledger, draws, bridge_config)
        )
        for key in ("forecast_id", "model_id"):
            scored[key] = scored[key].astype(str)
        aggregate_score = _aggregate_equal_particle_log_scores(scored)
        outputs: list[pd.DataFrame] = []
        draws_by_model: dict[str, np.ndarray] = {}
        if draws is not None:
            draw_model_key = draws["model_id"].astype(str)
            draws_by_model = {
                str(model_id): np.asarray(positions, dtype=np.int64)
                for model_id, positions in draws.groupby(
                    draw_model_key, sort=False
                ).indices.items()
            }
        for model_id, model_archive in native_archive.groupby(
            "model_id", sort=False
        ):
            model_draw_positions = draws_by_model.get(str(model_id))
            model_draws = (
                None
                if draws is None
                else (
                    draws.iloc[model_draw_positions]
                    if model_draw_positions is not None
                    else draws.iloc[0:0]
                )
            )
            weights = pd.DataFrame(
                {"model_id": [str(model_id)], "weight": [1.0]}
            )
            model_ledger = ledger[
                ledger["forecast_id"].astype(str).isin(
                    set(model_archive["forecast_id"].astype(str))
                )
            ]
            readout = posterior_predictive_readout(
                model_ledger,
                model_archive,
                weights,
                bridge_config=bridge_config,
                score_source=source,
                draws=model_draws,
                predictive_component_cache=predictive_component_cache,
            )
            readout["forecast_id"] = readout["forecast_id"].astype(str)
            readout["model_id"] = str(model_id)
            if contract in CENSORED_PREDICTIVE_CONTRACTS:
                readout["predictive_interval_source"] = (
                    f"censored_{source}_single_model_mixture_quantiles"
                )
            outputs.append(readout)
        out = pd.concat(outputs, ignore_index=True)
        out = out.merge(
            aggregate_score,
            on=["forecast_id", "model_id"],
            how="left",
            validate="one_to_one",
        )

    standard_deviation = np.sqrt(
        np.maximum(
            pd.to_numeric(out["predictive_var"], errors="raise").to_numpy(
                dtype=float
            ),
            0.0,
        )
    )
    mean = pd.to_numeric(
        out["predictive_mean"], errors="raise"
    ).to_numpy(dtype=float)
    for lower, upper, z_value in (
        ("lower_50", "upper_50", 0.6744897501960817),
        ("lower_90", "upper_90", 1.6448536269514722),
    ):
        if lower not in out.columns:
            out[lower] = np.maximum(0.0, mean - z_value * standard_deviation)
        if upper not in out.columns:
            out[upper] = np.maximum(0.0, mean + z_value * standard_deviation)

    meta_columns = [
        "forecast_id",
        "protocol_version",
        "natural_event_id",
        "dataset",
        "entity_id",
        "mode",
        "mode_kind",
        "forecast_strategy",
        "revision_version",
        "country",
        "country_code",
        "jurisdiction",
        "pathogen",
        "target_variable",
        "target",
        "anchor_week",
        "forecast_origin",
        "target_time",
        "release_time",
        "component",
        "horizon",
        "observed_value",
        "observed_mask",
        "split",
    ]
    meta = ledger[
        [column for column in meta_columns if column in ledger.columns]
    ].drop_duplicates("forecast_id")
    meta["forecast_id"] = meta["forecast_id"].astype(str)
    duplicate_meta = [
        column
        for column in meta.columns
        if column != "forecast_id" and column in out.columns
    ]
    if duplicate_meta:
        out = out.drop(columns=duplicate_meta)
    return meta.merge(
        out, on="forecast_id", how="inner", validate="one_to_many"
    )


def _coerce_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes", "y"})


def _read_posterior_path(posterior_path: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(posterior_path, pd.DataFrame):
        posterior = posterior_path.copy()
    else:
        posterior = pd.read_csv(posterior_path)
    required = {"model_id", "family", "weight", "release_time"}
    missing = sorted(required - set(posterior.columns))
    if missing:
        raise ValueError(f"posterior_path missing columns {missing}")
    posterior["release_time"] = pd.to_datetime(posterior["release_time"], errors="coerce")
    if posterior["release_time"].isna().any():
        raise ValueError("posterior_path contains invalid release_time values")
    return posterior


def _weights_for_snapshot(posterior: pd.DataFrame, snapshot_time: pd.Timestamp) -> pd.DataFrame:
    snap = posterior[posterior["release_time"] == snapshot_time].copy()
    if snap.empty:
        raise ValueError(f"no posterior snapshot found for {snapshot_time}")
    return snap[["model_id", "family", "weight"]].copy()


def posterior_predictive_readout_asof(
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    posterior_path: str | Path | pd.DataFrame,
    initial_weights: pd.DataFrame,
    *,
    interval_z: float = 1.96,
    posterior_update_policy: str = "holdout_train_val",
    release_availability_rule: str = "date_only_release_time_no_later_than_forecast_origin",
    allow_same_timestamp_release: bool = True,
    bridge_config: "BridgeConfig | None" = None,
    score_source: str = "archive_moment",
    draws: pd.DataFrame | None = None,
    predictive_component_cache: PredictiveComponentCache | None = None,
) -> pd.DataFrame:
    ""








    required_ledger = {"forecast_id", "forecast_origin"}
    required_weights = {"model_id", "family", "weight"}
    missing_ledger = sorted(required_ledger - set(ledger.columns))
    missing_weights = sorted(required_weights - set(initial_weights.columns))
    if missing_ledger:
        raise ValueError(f"ledger missing columns {missing_ledger}")
    if missing_weights:
        raise ValueError(f"initial_weights missing columns {missing_weights}")
    posterior = _read_posterior_path(posterior_path)
    readout_ledger = ledger.copy()
    readout_ledger["forecast_origin"] = pd.to_datetime(readout_ledger["forecast_origin"], errors="coerce")
    if readout_ledger["forecast_origin"].isna().any():
        raise ValueError("ledger contains invalid forecast_origin values")

    snapshot_times = np.array(sorted(posterior["release_time"].dropna().unique()), dtype="datetime64[ns]")
    origins = readout_ledger["forecast_origin"].to_numpy(dtype="datetime64[ns]")
    search_side = "right" if allow_same_timestamp_release else "left"
    positions = np.searchsorted(snapshot_times, origins, side=search_side) - 1
    snapshot_labels: list[str] = []
    snapshot_values: list[pd.Timestamp | pd.NaT] = []
    used_prior: list[bool] = []
    for pos in positions:
        if pos < 0:
            snapshot_labels.append("__prior__")
            snapshot_values.append(pd.NaT)
            used_prior.append(True)
        else:
            ts = pd.Timestamp(snapshot_times[pos])
            snapshot_labels.append(ts.isoformat())
            snapshot_values.append(ts)
            used_prior.append(False)

    readout_ledger["__snapshot_key__"] = snapshot_labels
    readout_ledger["posterior_snapshot_time"] = snapshot_values
    readout_ledger["used_prior_snapshot"] = used_prior
    readout_ledger["posterior_update_policy"] = posterior_update_policy
    readout_ledger["release_availability_rule"] = release_availability_rule
    snapshot_series = pd.to_datetime(readout_ledger["posterior_snapshot_time"], errors="coerce")
    origin_series = pd.to_datetime(readout_ledger["forecast_origin"], errors="coerce")
    unavailable = snapshot_series.notna() & (
        snapshot_series > origin_series if allow_same_timestamp_release else snapshot_series >= origin_series
    )
    readout_ledger["future_snapshot_violation"] = unavailable
    stale_days = (origin_series - snapshot_series).dt.total_seconds() / 86400.0
    readout_ledger["stale_posterior_age_days"] = stale_days.where(snapshot_series.notna())

    draw_index = (
        _index_draws_by_forecast(draws)
        if (
            _predictive_contract(bridge_config) != alternate_ARCHIVE_MOMENT
            and _validate_score_source(score_source) == "draw_kernel"
        )
        else {}
    )

    outputs: list[pd.DataFrame] = []
    base_initial = initial_weights[["model_id", "family", "weight"]].copy()
    for key, group in readout_ledger.groupby("__snapshot_key__", sort=False):
        if key == "__prior__":
            weights = base_initial
        else:
            weights = _weights_for_snapshot(posterior, pd.Timestamp(key))
        group_ids = set(group["forecast_id"].astype(str))
        archive_group = archive[archive["forecast_id"].astype(str).isin(group_ids)].copy()
        if draw_index:
            draw_positions = [
                draw_index[forecast_id]
                for forecast_id in group_ids
                if forecast_id in draw_index
            ]
            snapshot_draws = (
                draws.iloc[np.concatenate(draw_positions)]
                if draw_positions
                else draws.iloc[0:0]
            )
        else:
            snapshot_draws = draws
        out = posterior_predictive_readout(
            group.drop(columns=["__snapshot_key__"]),
            archive_group,
            weights,
            interval_z=interval_z,
            bridge_config=bridge_config,
            score_source=score_source,
            draws=snapshot_draws,
            predictive_component_cache=predictive_component_cache,
        )
        snapshot_meta = group[
            [
                "forecast_id",
                "posterior_snapshot_time",
                "used_prior_snapshot",
                "stale_posterior_age_days",
                "future_snapshot_violation",
                "posterior_update_policy",
                "release_availability_rule",
            ]
        ].drop_duplicates("forecast_id")
        outputs.append(out.merge(snapshot_meta, on="forecast_id", how="left"))

    if not outputs:
        return pd.DataFrame()
    result = pd.concat(outputs, ignore_index=True)
    if "self_target_update_violation" not in result.columns:
        result["self_target_update_violation"] = False
    order = readout_ledger[["forecast_id"]].reset_index().rename(columns={"index": "__order__"})
    result = result.merge(order, on="forecast_id", how="left").sort_values("__order__").drop(columns=["__order__"])
    return result.reset_index(drop=True)


def discovered_model_distribution(model_weights: pd.DataFrame) -> dict[str, object]:
    df = model_weights[["model_id", "family", "weight"]].copy()
    weights = df["weight"].astype(float).to_numpy()
    weights = weights / weights.sum()
    df["weight"] = weights
    family = df.groupby("family")["weight"].sum().sort_index().to_dict()
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-300))))
    return {
        "model_ess": effective_sample_size(df.set_index("model_id")["weight"]),
        "structural_entropy": entropy,
        "family_mass": family,
        "top_model": str(df.sort_values("weight", ascending=False).iloc[0]["model_id"]),
    }


def write_json(obj: dict[str, object], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))
    return path
