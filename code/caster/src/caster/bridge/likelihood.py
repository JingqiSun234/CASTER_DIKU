from __future__ import annotations
from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import lgamma, log, pi, sqrt
import numpy as np
import pandas as pd

from caster.forecast.archive import validate_native_horizon_provenance


alternate_ARCHIVE_MOMENT = "alternate_archive_moment"
COHERENT_MEAN_PRESERVING_TRUNCATED_T = (
    "coherent_mean_preserving_truncated_t"
)
COHERENT_CENSORED_STUDENT_T = "coherent_censored_student_t"
COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T = (
    "coherent_mean_preserving_censored_student_t"
)
ARCHIVE_MEAN_BRIDGE_QUANTILES = "archive_mean_bridge_quantiles"
PREDICTIVE_CONTRACTS = (
    alternate_ARCHIVE_MOMENT,
    COHERENT_MEAN_PRESERVING_TRUNCATED_T,
    COHERENT_CENSORED_STUDENT_T,
    COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T,
    ARCHIVE_MEAN_BRIDGE_QUANTILES,
)
COHERENT_CENSORED_CONTRACTS = (
    COHERENT_CENSORED_STUDENT_T,
    COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T,
)
STRICT_FROZEN_SUPPORT_POLICY = "strict_frozen_support"
ORIGIN_CAUSAL_MEAN_SUPPORT_EXPANSION_POLICY = (
    "origin_causal_requested_mean_times_multiplier"
)


@dataclass(frozen=True)
class BridgeConfig:
    distribution: str = "gaussian"
    transform: str = "log1p"
    nu: float = 5.0
    nu_by_component: dict[str, float] = field(default_factory=dict)
    kernel_distribution: str = "gaussian"
    kernel_nu: float = 5.0
    min_scale: float = 1e-3
    sigma_by_component: dict[str, float] = field(default_factory=dict)
    tau_by_component: dict[str, float] = field(default_factory=dict)
    gamma_by_component: dict[str, float] = field(default_factory=dict)
    horizon_weight: dict[int, float] = field(default_factory=dict)
    component_weight: dict[str, float] = field(default_factory=dict)
    default_sigma: float = 0.25
    default_tau: float = 0.25
    default_gamma: float = 1.0
                                                                            
                                                                              
                                                                    
    predictive_contract: str = alternate_ARCHIVE_MOMENT
    truncation_lower_raw: float = 0.0
    truncation_upper_raw_by_component: dict[str, float] = field(default_factory=dict)
    default_truncation_upper_raw: float = float("inf")
    truncation_bound_policy: str = "none"
    truncation_quadrature_order: int = 128
    truncation_zero_mean_epsilon: float = 1e-10
                                                                            
                                                                            
                                                                           
    truncation_support_expansion_multiplier: float | None = 1.25


@dataclass(frozen=True)
class CoherentTiltedPredictive:
    ""








    kind: str
    centers_z: np.ndarray
    scale: float
    nu: float
    lower_raw: float
    frozen_upper_raw: float
    upper_raw: float
    lower_z: float
    upper_z: float
    tilt: float
    log_normalizer: float
    requested_mean: float
    effective_target_mean: float
    mean: float
    second_moment: float
    variance: float
    mean_floor: float
    mean_floor_applied: bool
    support_expanded: bool
    support_expansion_policy: str
    support_expansion_multiplier: float | None
    quadrature_order: int
    quadrature_z: np.ndarray
    quadrature_raw: np.ndarray
    quadrature_probability: np.ndarray


@dataclass(frozen=True)
class CensoredStudentTPredictive:
    ""









    kind: str
    centers_z: np.ndarray
    scale: float
    nu: float
    lower_raw: float
    frozen_upper_raw: float
    upper_raw: float
    lower_z: float
    upper_z: float
    mean: float
    second_moment: float
    variance: float
    lower_atom_probability: float
    upper_atom_probability: float
    continuous_probability: float
    quadrature_order: int
    quadrature_z: np.ndarray
    quadrature_raw: np.ndarray
    quadrature_probability: np.ndarray
    requested_mean: float | None = None
    effective_target_mean: float | None = None
    location_shift: float = 0.0
    mean_floor: float = 0.0
    mean_floor_applied: bool = False
    mean_constraint_applied: bool = False
    mean_constraint_residual: float = 0.0

def component_key(component: object, horizon: object | None = None) -> str:
    comp = str(component)
    if horizon is None or pd.isna(horizon):
        return comp
    h_text = str(horizon).strip()
    try:
        h_value = float(h_text)
        h_text = str(int(h_value)) if h_value.is_integer() else h_text
    except Exception:
        pass
    suffix = f"__h{h_text}"
    return comp if comp.endswith(suffix) else f"{comp}{suffix}"


def bridge_group_key(mode: object, component: object) -> str:
    mode_text = str(mode).strip()
    component_text = str(component).strip()
    return f"{mode_text}::{component_text}" if mode_text else component_text


def bridge_r_key(mode: object, component: object, horizon: object | None = None) -> str:
    ""





    del mode
    return component_key(component, horizon)


def bridge_r_key_series(frame: pd.DataFrame) -> pd.Series:
    ""





    if "component" not in frame.columns:
        raise ValueError("bridge parameter grouping requires a component column")
    component = frame["component"].fillna("").astype(str).str.strip()
    if component.eq("").any():
        raise ValueError("bridge parameter grouping found an empty component")
    horizon = (
        frame["horizon"]
        if "horizon" in frame.columns
        else pd.Series([None] * len(frame), index=frame.index, dtype=object)
    )
    return pd.Series(
        [component_key(c, h) for c, h in zip(component, horizon)],
        index=frame.index,
        dtype=str,
    )


def bridge_lookup_keys(mode: object, component: object, horizon: object | None = None) -> tuple[object, ...]:
    group_key = bridge_group_key(mode, component)
    component_text = str(component).strip()
    candidates: list[object] = []
    if horizon is not None and not pd.isna(horizon):
        candidates.extend((component_key(component_text, horizon), component_key(group_key, horizon)))
    candidates.extend((component_text, group_key, "default"))
    return tuple(dict.fromkeys(candidates))


def lookup_bridge_float(
    values: dict,
    mode: object,
    component: object,
    horizon: object | None,
    default: float,
) -> float:
    for candidate in bridge_lookup_keys(mode, component, horizon):
        if candidate in values:
            return float(values[candidate])
        candidate_text = str(candidate)
        if candidate_text in values:
            return float(values[candidate_text])
    return float(default)


def _lookup_bridge_series(
    values: dict,
    mode: pd.Series,
    component: pd.Series,
    horizon: pd.Series,
    default: float,
) -> np.ndarray:
    mode_text = mode.fillna("").astype(str).str.strip()
    component_text = component.fillna("").astype(str).str.strip()
    horizon_text = pd.to_numeric(horizon, errors="raise").astype(int).astype(str)
    group_key = mode_text.where(mode_text.eq(""), mode_text + "::") + component_text
    component_horizon_key = component_text + "__h" + horizon_text
    alternate_mode_horizon_key = group_key + "__h" + horizon_text
    result = pd.Series(np.nan, index=component.index, dtype=float)
    for keys in (component_horizon_key, alternate_mode_horizon_key, component_text, group_key):
        result = result.fillna(keys.map(values))
    if "default" in values:
        result = result.fillna(float(values["default"]))
    return result.fillna(float(default)).astype(float).to_numpy()


def _gammaln_array(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import gammaln

        return np.asarray(gammaln(values), dtype=float)
    except (ImportError, ModuleNotFoundError):
        return np.vectorize(lgamma, otypes=[float])(values)


def _lookup_float(values: dict, key: str, fallback_key: str, default: float) -> float:
    for candidate in (key, fallback_key):
        if candidate in values:
            return float(values[candidate])
    return float(default)

def transform_value(y: float, transform: str = "log1p") -> float:
    if transform == "identity": return float(y)
    if transform == "log1p": return float(np.log1p(max(float(y), 0.0)))
    raise ValueError(f"unknown transform {transform!r}")

def delta_transform_var(mu: float, var: float, transform: str = "log1p") -> float:
    var, mu = max(float(var), 0.0), max(float(mu), 0.0)
    if transform == "identity": return var
    if transform == "log1p": return var / ((1.0 + mu) ** 2)
    raise ValueError(f"unknown transform {transform!r}")


def stable_log1p_transform_var(
    mu: float | np.ndarray, var: float | np.ndarray
) -> float | np.ndarray:
    ""






    mean_array, var_array = np.broadcast_arrays(
        np.maximum(np.asarray(mu, dtype=float), 0.0),
        np.maximum(np.asarray(var, dtype=float), 0.0),
    )
    mapped = np.log1p(var_array / np.square(1.0 + mean_array))
    return float(mapped) if mapped.ndim == 0 else mapped

def normal_log_density(x: float, mu: float, scale: float) -> float:
    s = max(float(scale), 1e-12); z = (float(x) - float(mu)) / s
    return float(-0.5 * log(2.0 * pi) - log(s) - 0.5 * z * z)



def negative_binomial_logpmf(y: float, mean: float, dispersion: float) -> float:
    mu = max(float(mean), 1e-12)
    r = max(float(dispersion), 1e-12)
    k = max(float(y), 0.0)
    p = r / (r + mu)
    return float(lgamma(k + r) - lgamma(r) - lgamma(k + 1.0) + r * log(p) + k * log(1.0 - p))

def gaussian_kernel_log_density(x: float, draws: np.ndarray, bandwidth: float) -> float:
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float('-inf')
    h = max(float(bandwidth), 1e-6)
    vals = -0.5 * ((float(x) - d) / h) ** 2 - log(h) - 0.5 * log(2.0 * pi)
    m = float(np.max(vals))
    return float(m + np.log(np.mean(np.exp(vals - m))))

def student_t_log_density(x: float, mu: float, scale: float, nu: float) -> float:
    s = max(float(scale), 1e-12); nu = float(nu)
    if np.isinf(nu):
        return normal_log_density(x, mu, s)
    if not np.isfinite(nu) or nu <= 0.0:
        raise ValueError("Student-t nu must be positive or infinity")
    z2 = ((float(x) - float(mu)) / s) ** 2
    return float(lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * log(nu * pi) - log(s) - ((nu + 1.0) / 2.0) * log(1.0 + z2 / nu))

def student_t_kernel_log_density(x: float, draws: np.ndarray, bandwidth: float, nu: float) -> float:
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float('-inf')
    h = max(float(bandwidth), 1e-6)
    if np.isinf(float(nu)):
        return gaussian_kernel_log_density(x, d, h)
    vals = np.asarray([student_t_log_density(float(x), float(mu), h, float(nu)) for mu in d], dtype=float)
    m = float(np.max(vals))
    return float(m + np.log(np.mean(np.exp(vals - m))))


@lru_cache(maxsize=32)
def _coherent_logit_quadrature_rule(
    quadrature_order: int, tail_logit: float = 32.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ""

    order = int(quadrature_order)
    if order < 32 or order % 4:
        raise ValueError(
            "truncation_quadrature_order must be at least 32 and divisible by 4"
        )
    panel_nodes, panel_weights = np.polynomial.legendre.leggauss(order // 4)
    edges = np.linspace(-float(tail_logit), float(tail_logit), 5)
    nodes: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for left, right in zip(edges[:-1], edges[1:]):
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        nodes.append(midpoint + half_width * panel_nodes)
        weights.append(half_width * panel_weights)
    logit_nodes = np.concatenate(nodes)
    logit_weights = np.concatenate(weights)
                                                                     
    fractions = np.empty_like(logit_nodes)
    nonnegative = logit_nodes >= 0.0
    fractions[nonnegative] = 1.0 / (1.0 + np.exp(-logit_nodes[nonnegative]))
    exp_nodes = np.exp(logit_nodes[~nonnegative])
    fractions[~nonnegative] = exp_nodes / (1.0 + exp_nodes)
    for array in (fractions, logit_weights, logit_nodes):
        array.setflags(write=False)
    return fractions, logit_weights, logit_nodes


def _coherent_student_t_log_density_array(
    x: np.ndarray, centers: np.ndarray, scale: np.ndarray, nu: np.ndarray
) -> np.ndarray:
    ""

    from scipy.special import gammaln

    x_a = np.asarray(x, dtype=float)
    center_a = np.asarray(centers, dtype=float)
    scale_a = np.asarray(scale, dtype=float)
    nu_a = np.asarray(nu, dtype=float)
    if np.any(~np.isfinite(scale_a) | (scale_a <= 0.0)):
        raise ValueError("coherent Student-t scale must be finite and positive")
    if np.any(~np.isfinite(nu_a) | (nu_a <= 0.0)):
        raise ValueError(
            "coherent mean-preserving Student-t requires finite positive nu"
        )
                                                                          
                                                                         
                                                                           
                                                                    
    log_normalizer = (
        gammaln((nu_a + 1.0) / 2.0)
        - gammaln(nu_a / 2.0)
        - 0.5 * np.log(nu_a * pi)
        - np.log(scale_a)
    )
    residual = (x_a - center_a) / scale_a
    return np.asarray(
        log_normalizer
        - ((nu_a + 1.0) / 2.0) * np.log1p(np.square(residual) / nu_a),
        dtype=float,
    )


def _coherent_logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    from scipy.special import logsumexp

    return np.asarray(logsumexp(values, axis=axis), dtype=float)


def _coherent_transform_quadrature_batch(
    lower_raw: np.ndarray,
    upper_raw: np.ndarray,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ""

    lower = np.asarray(lower_raw, dtype=float).reshape(-1)
    upper = np.asarray(upper_raw, dtype=float).reshape(-1)
    if np.any(~np.isfinite(lower) | ~np.isfinite(upper)):
        raise ValueError(
            "coherent truncated bridge requires finite frozen raw bounds; "
            "calibrate truncation_upper_raw_by_component before scoring"
        )
    if np.any(lower < 0.0) or np.any(upper <= lower):
        raise ValueError(
            "coherent truncation bounds must satisfy 0 <= lower_raw < upper_raw"
        )
    fractions, logit_weights, _ = _coherent_logit_quadrature_rule(
        int(quadrature_order)
    )
    lower_z = np.log1p(lower)
    upper_z = np.log1p(upper)
    width = upper_z - lower_z
    z = lower_z[:, None] + width[:, None] * fractions[None, :]
    raw = np.expm1(z)
    log_measure = (
        np.log(logit_weights)[None, :]
        + np.log(width)[:, None]
        + np.log(fractions)[None, :]
        + np.log1p(-fractions)[None, :]
    )
    return z, raw, log_measure, lower_z, upper_z


def _coherent_effective_targets(
    requested_mean: np.ndarray,
    lower_raw: np.ndarray,
    upper_raw: np.ndarray,
    mean_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(requested_mean, dtype=float).reshape(-1)
    lower = np.asarray(lower_raw, dtype=float).reshape(-1)
    upper = np.asarray(upper_raw, dtype=float).reshape(-1)
    floor = float(mean_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("truncation_zero_mean_epsilon must be finite and positive")
    if np.any(~np.isfinite(target)):
        raise ValueError("coherent predictive target means must be finite")
    invalid = (target < lower) | (target >= upper)
    if np.any(invalid):
        example_positions = np.flatnonzero(invalid)[:5]
        examples = [
            {
                "position": int(position),
                "requested_mean": float(target[position]),
                "lower_raw": float(lower[position]),
                "upper_raw": float(upper[position]),
            }
            for position in example_positions
        ]
        raise ValueError(
            "coherent predictive target mean lies outside its frozen half-open "
            f"raw support for {int(invalid.sum())} rows; example positions={examples}"
        )
                                                                             
                                                                         
                                                                          
                                                                              
                                                                             
    floor_applied = target <= lower + floor
    effective = target.copy()
    effective[floor_applied] = lower[floor_applied] + floor
    if np.any(effective >= upper):
        raise ValueError(
            "truncation_zero_mean_epsilon moves a boundary target beyond its upper bound"
        )
    return effective, floor_applied


def _coherent_effective_upper_bounds(
    requested_mean: np.ndarray,
    frozen_upper_raw: np.ndarray,
    support_expansion_multiplier: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    ""











    target = np.asarray(requested_mean, dtype=float).reshape(-1)
    frozen = np.asarray(frozen_upper_raw, dtype=float).reshape(-1)
    if target.shape != frozen.shape:
        raise ValueError(
            "coherent target means and frozen upper bounds have incompatible shapes"
        )
    if np.any(~np.isfinite(target)):
        raise ValueError("coherent predictive target means must be finite")
    if np.any(~np.isfinite(frozen) | (frozen <= 0.0)):
        raise ValueError(
            "coherent predictive frozen upper bounds must be finite and positive"
        )
    if support_expansion_multiplier is None:
        return frozen.copy(), np.zeros(target.size, dtype=bool)

    multiplier = float(support_expansion_multiplier)
    if not np.isfinite(multiplier) or multiplier <= 1.0:
        raise ValueError(
            "truncation support expansion multiplier must be finite and greater "
            "than one"
        )
    expanded = target >= frozen
    effective = frozen.copy()
    effective[expanded] = np.maximum(
        frozen[expanded],
        multiplier * np.maximum(target[expanded], 1.0),
    )
    if np.any(~np.isfinite(effective)):
        raise ValueError("coherent effective upper support is non-finite")
    return effective, expanded


def _coherent_support_expansion_policy(
    support_expansion_multiplier: float | None,
) -> str:
    return (
        STRICT_FROZEN_SUPPORT_POLICY
        if support_expansion_multiplier is None
        else ORIGIN_CAUSAL_MEAN_SUPPORT_EXPANSION_POLICY
    )


def _coherent_tilt_stats_batch(
    tilt: np.ndarray,
    raw_grid: np.ndarray,
    log_base_weight: np.ndarray,
    *,
    return_probability: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    log_weight = log_base_weight + np.asarray(tilt, dtype=float)[:, None] * raw_grid
    log_normalizer = _coherent_logsumexp(log_weight, axis=1)
    probability = np.exp(log_weight - log_normalizer[:, None])
    mean = np.sum(probability * raw_grid, axis=1)
    second = np.sum(probability * np.square(raw_grid), axis=1)
    variance = np.maximum(second - np.square(mean), 0.0)
    if not (
        np.isfinite(log_normalizer).all()
        and np.isfinite(mean).all()
        and np.isfinite(second).all()
        and np.isfinite(variance).all()
    ):
        raise FloatingPointError("non-finite coherent truncated Student-t quadrature")
    return (
        log_normalizer,
        mean,
        second,
        variance,
        probability if return_probability else None,
    )


def _fit_coherent_tilt_batch(
    raw_grid: np.ndarray,
    log_base_weight: np.ndarray,
    effective_target: np.ndarray,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-9,
    max_iterations: int = 160,
    return_probability: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    ""

    target = np.asarray(effective_target, dtype=float).reshape(-1)
    if raw_grid.shape != log_base_weight.shape or raw_grid.shape[0] != target.size:
        raise ValueError("coherent tilt batch arrays have incompatible shapes")
    numerical_lower = raw_grid[:, 0]
    numerical_upper = raw_grid[:, -1]
    outside = (target <= numerical_lower) | (target >= numerical_upper)
    if np.any(outside):
        examples = np.flatnonzero(outside)[:5].tolist()
        raise ValueError(
            "effective mean is outside the numerically resolved quadrature support; "
            f"increase quadrature endpoint resolution or mean floor; positions={examples}"
        )
    tolerance = float(atol) + float(rtol) * np.maximum(np.abs(target), 1e-12)
    zero_tilt = np.zeros(target.size, dtype=float)
    _, zero_mean, _, _, _ = _coherent_tilt_stats_batch(
        zero_tilt, raw_grid, log_base_weight, return_probability=False
    )
    at_zero = np.abs(zero_mean - target) <= tolerance
    lower_tilt = np.zeros_like(target)
    upper_tilt = np.zeros_like(target)

    need_lower = ~at_zero & (zero_mean > target)
    if np.any(need_lower):
        indices = np.flatnonzero(need_lower)
        candidates = -np.ones(indices.size, dtype=float)
        unresolved = np.ones(indices.size, dtype=bool)
        for _ in range(max_iterations):
            active = np.flatnonzero(unresolved)
            if active.size == 0:
                break
            selected = indices[active]
            _, candidate_mean, _, _, _ = _coherent_tilt_stats_batch(
                candidates[active],
                raw_grid[selected],
                log_base_weight[selected],
                return_probability=False,
            )
            bracketed = candidate_mean <= target[selected]
            if np.any(bracketed):
                resolved_active = active[bracketed]
                lower_tilt[indices[resolved_active]] = candidates[resolved_active]
                unresolved[resolved_active] = False
            candidates[active[~bracketed]] *= 2.0
        if np.any(unresolved):
            raise RuntimeError("failed to bracket negative coherent exponential tilt")

    need_upper = ~at_zero & (zero_mean < target)
    if np.any(need_upper):
        indices = np.flatnonzero(need_upper)
        candidates = np.ones(indices.size, dtype=float)
        unresolved = np.ones(indices.size, dtype=bool)
        for _ in range(max_iterations):
            active = np.flatnonzero(unresolved)
            if active.size == 0:
                break
            selected = indices[active]
            _, candidate_mean, _, _, _ = _coherent_tilt_stats_batch(
                candidates[active],
                raw_grid[selected],
                log_base_weight[selected],
                return_probability=False,
            )
            bracketed = candidate_mean >= target[selected]
            if np.any(bracketed):
                resolved_active = active[bracketed]
                upper_tilt[indices[resolved_active]] = candidates[resolved_active]
                unresolved[resolved_active] = False
            candidates[active[~bracketed]] *= 2.0
        if np.any(unresolved):
            raise RuntimeError("failed to bracket positive coherent exponential tilt")

    current = np.where(at_zero, 0.0, 0.5 * (lower_tilt + upper_tilt))
    converged = at_zero.copy()
    for _ in range(max_iterations):
        log_z, mean, second, variance, _ = _coherent_tilt_stats_batch(
            current, raw_grid, log_base_weight, return_probability=False
        )
        residual = mean - target
        converged |= np.abs(residual) <= tolerance
        if np.all(converged):
            break
        active = ~converged
        below = active & (residual < 0.0)
        above = active & ~below
        lower_tilt[below] = current[below]
        upper_tilt[above] = current[above]
        newton = np.full_like(current, np.nan)
        usable = active & (variance > np.finfo(float).tiny)
        newton[usable] = current[usable] - residual[usable] / variance[usable]
        midpoint = 0.5 * (lower_tilt + upper_tilt)
        inside = (
            active
            & np.isfinite(newton)
            & (newton > lower_tilt)
            & (newton < upper_tilt)
        )
        current[active] = midpoint[active]
        current[inside] = newton[inside]
    else:
        bad = np.flatnonzero(~converged)[:5].tolist()
        raise RuntimeError(
            "coherent exponential tilt solver did not converge; "
            f"example positions={bad}"
        )

    log_z, mean, second, variance, probability = _coherent_tilt_stats_batch(
        current,
        raw_grid,
        log_base_weight,
        return_probability=return_probability,
    )
    residual = np.abs(mean - target)
    if np.any(residual > 5.0 * tolerance):
        bad = np.flatnonzero(residual > 5.0 * tolerance)[:5].tolist()
        raise RuntimeError(
            "coherent exponential tilt solver failed its mean validation; "
            f"example positions={bad}"
        )
    return current, log_z, mean, second, variance, probability


def _fit_coherent_predictive_from_centers(
    *,
    kind: str,
    centers_z: np.ndarray,
    requested_mean: float,
    scale: float,
    nu: float,
    lower_raw: float,
    upper_raw: float,
    mean_floor: float,
    quadrature_order: int,
    support_expansion_multiplier: float | None,
) -> CoherentTiltedPredictive:
    centers = np.asarray(centers_z, dtype=float).reshape(-1)
    centers = centers[np.isfinite(centers)]
    if centers.size == 0:
        raise ValueError("coherent predictive requires at least one finite center")
    lower = np.asarray([float(lower_raw)])
    frozen_upper = np.asarray([float(upper_raw)])
    upper, support_expanded = _coherent_effective_upper_bounds(
        np.asarray([float(requested_mean)]),
        frozen_upper,
        support_expansion_multiplier,
    )
    effective, floor_applied = _coherent_effective_targets(
        np.asarray([float(requested_mean)]), lower, upper, float(mean_floor)
    )
    z, raw, log_measure, lower_z, upper_z = _coherent_transform_quadrature_batch(
        lower, upper, int(quadrature_order)
    )
    kernel_log_density = _coherent_student_t_log_density_array(
        z[:, :, None],
        centers[None, None, :],
        np.asarray(float(scale)).reshape(1, 1, 1),
        np.asarray(float(nu)).reshape(1, 1, 1),
    )
    base_log_density = _coherent_logsumexp(kernel_log_density, axis=2) - log(centers.size)
    tilt, log_z, mean, second, variance, probability = _fit_coherent_tilt_batch(
        raw, base_log_density + log_measure, effective, return_probability=True
    )
    assert probability is not None
    return CoherentTiltedPredictive(
        kind=str(kind),
        centers_z=centers.copy(),
        scale=float(scale),
        nu=float(nu),
        lower_raw=float(lower_raw),
        frozen_upper_raw=float(frozen_upper[0]),
        upper_raw=float(upper[0]),
        lower_z=float(lower_z[0]),
        upper_z=float(upper_z[0]),
        tilt=float(tilt[0]),
        log_normalizer=float(log_z[0]),
        requested_mean=float(requested_mean),
        effective_target_mean=float(effective[0]),
        mean=float(mean[0]),
        second_moment=float(second[0]),
        variance=float(variance[0]),
        mean_floor=float(mean_floor),
        mean_floor_applied=bool(floor_applied[0]),
        support_expanded=bool(support_expanded[0]),
        support_expansion_policy=_coherent_support_expansion_policy(
            support_expansion_multiplier
        ),
        support_expansion_multiplier=(
            None
            if support_expansion_multiplier is None
            else float(support_expansion_multiplier)
        ),
        quadrature_order=int(quadrature_order),
        quadrature_z=z[0].copy(),
        quadrature_raw=raw[0].copy(),
        quadrature_probability=probability[0].copy(),
    )


def fit_coherent_moment_predictive(
    target_raw_mean: float,
    scale: float,
    nu: float,
    lower_raw: float,
    upper_raw: float,
    *,
    mean_floor: float,
    quadrature_order: int = 128,
    loc_z: float | None = None,
    support_expansion_multiplier: float | None = None,
) -> CoherentTiltedPredictive:
    ""

    target = max(float(target_raw_mean), 0.0)
    center = np.log1p(target) if loc_z is None else float(loc_z)
    return _fit_coherent_predictive_from_centers(
        kind="moment",
        centers_z=np.asarray([center]),
        requested_mean=target,
        scale=float(scale),
        nu=float(nu),
        lower_raw=float(lower_raw),
        upper_raw=float(upper_raw),
        mean_floor=float(mean_floor),
        quadrature_order=int(quadrature_order),
        support_expansion_multiplier=support_expansion_multiplier,
    )


def fit_coherent_draw_predictive(
    draws_raw: np.ndarray,
    bandwidth: float,
    nu: float,
    lower_raw: float,
    upper_raw: float,
    *,
    mean_floor: float,
    quadrature_order: int = 128,
    support_expansion_multiplier: float | None = None,
) -> CoherentTiltedPredictive:
    ""

    raw = np.asarray(draws_raw, dtype=float).reshape(-1)
    raw = np.maximum(raw[np.isfinite(raw)], 0.0)
    if raw.size == 0:
        raise ValueError("coherent draw predictive has no finite draws")
    return _fit_coherent_predictive_from_centers(
        kind="draw_kernel",
        centers_z=np.log1p(raw),
        requested_mean=float(np.mean(raw)),
        scale=float(bandwidth),
        nu=float(nu),
        lower_raw=float(lower_raw),
        upper_raw=float(upper_raw),
        mean_floor=float(mean_floor),
        quadrature_order=int(quadrature_order),
        support_expansion_multiplier=support_expansion_multiplier,
    )


def coherent_predictive_log_density_z(
    x_z: float | np.ndarray, predictive: CoherentTiltedPredictive
) -> float | np.ndarray:
    ""

    values = np.asarray(x_z, dtype=float)
    flat = values.reshape(-1)
    kernel_log_density = _coherent_student_t_log_density_array(
        flat[:, None],
        predictive.centers_z[None, :],
        np.asarray(predictive.scale).reshape(1, 1),
        np.asarray(predictive.nu).reshape(1, 1),
    )
    base = _coherent_logsumexp(kernel_log_density, axis=1) - log(
        predictive.centers_z.size
    )
    result = (
        base
        + predictive.tilt * np.expm1(flat)
        - predictive.log_normalizer
    )
    outside = (flat < predictive.lower_z) | (flat > predictive.upper_z)
    result[outside] = float("-inf")
    shaped = result.reshape(values.shape)
    return float(shaped) if shaped.ndim == 0 else shaped


def _censored_student_t_log_boundary_probability(
    boundary_z: float,
    centers_z: np.ndarray,
    scale: float,
    nu: float,
    *,
    upper_tail: bool,
) -> float:
    ""

    from scipy.special import logsumexp
    from scipy.stats import t as student_t

    centers = np.asarray(centers_z, dtype=float).reshape(-1)
    standardized = (float(boundary_z) - centers) / float(scale)
    terms = (
        student_t.logsf(standardized, df=float(nu))
        if upper_tail
        else student_t.logcdf(standardized, df=float(nu))
    )
    return float(logsumexp(terms) - log(centers.size))


def _validate_censored_parameters(
    centers_z: np.ndarray,
    scale: float,
    nu: float,
    lower_raw: float,
    upper_raw: float,
) -> np.ndarray:
    centers = np.asarray(centers_z, dtype=float).reshape(-1)
    centers = centers[np.isfinite(centers)]
    if centers.size == 0:
        raise ValueError("censored Student-t predictive requires finite centers")
    if not np.isfinite(scale) or float(scale) <= 0.0:
        raise ValueError("censored Student-t scale must be finite and positive")
    if not np.isfinite(nu) or float(nu) <= 0.0:
        raise ValueError("censored Student-t requires finite positive nu")
    if float(lower_raw) != 0.0:
        raise ValueError("censored Student-t requires lower_raw=0")
    if not np.isfinite(upper_raw) or float(upper_raw) <= 0.0:
        raise ValueError(
            "censored Student-t requires a finite positive frozen upper bound"
        )
    return centers


@lru_cache(maxsize=32)
def _censored_legendre_rule(
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    ""








    order = int(quadrature_order)
    if order < 32 or order % 4:
        raise ValueError(
            "truncation_quadrature_order must be at least 32 and divisible by 4"
        )
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _censored_shift_moments_batch(
    centers_z: np.ndarray,
    center_mask: np.ndarray,
    shift: np.ndarray,
    scale: np.ndarray,
    nu: np.ndarray,
    upper_raw: np.ndarray,
    *,
    quadrature_order: int,
    compute_second: bool = True,
    compute_derivative: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    ""











    from scipy.special import stdtr

    centers = np.asarray(centers_z, dtype=float)
    mask = np.asarray(center_mask, dtype=bool)
    if centers.ndim != 2 or mask.shape != centers.shape:
        raise ValueError("censored center arrays must be aligned two-dimensional arrays")
    group_count, max_center_count = centers.shape
    if group_count == 0 or max_center_count == 0:
        raise ValueError("censored mean solve requires at least one center")
    finite_count = mask.sum(axis=1)
    if np.any(finite_count <= 0):
        raise ValueError("every censored mean-solve group requires a finite center")
    shift_a = np.asarray(shift, dtype=float).reshape(-1)
    scale_a = np.asarray(scale, dtype=float).reshape(-1)
    nu_a = np.asarray(nu, dtype=float).reshape(-1)
    upper_a = np.asarray(upper_raw, dtype=float).reshape(-1)
    if not (
        shift_a.size
        == scale_a.size
        == nu_a.size
        == upper_a.size
        == group_count
    ):
        raise ValueError("censored mean-solve parameter arrays have incompatible shapes")
    if np.any(~np.isfinite(shift_a)):
        raise ValueError("censored center shifts must be finite")
    if np.any(~np.isfinite(scale_a) | (scale_a <= 0.0)):
        raise ValueError("censored mean-solve scales must be finite and positive")
    if np.any(~np.isfinite(nu_a) | (nu_a <= 0.0)):
        raise ValueError("censored mean-solve nu must be finite and positive")
    if np.any(~np.isfinite(upper_a) | (upper_a <= 0.0)):
        raise ValueError("censored mean-solve upper bounds must be finite and positive")

    nodes, base_weights = _censored_legendre_rule(int(quadrature_order))
    first = np.empty(group_count, dtype=float)
    second = (
        np.empty(group_count, dtype=float)
        if bool(compute_second)
        else None
    )
    derivative = (
        np.empty(group_count, dtype=float)
        if bool(compute_derivative)
        else None
    )
    chunk_size = max(
        8,
        min(
            2048,
            1_048_576 // (int(quadrature_order) * max_center_count),
        ),
    )
    for start in range(0, group_count, chunk_size):
        stop = min(start + chunk_size, group_count)
        sl = slice(start, stop)
        upper_z = np.log1p(upper_a[sl])
        z = 0.5 * upper_z[:, None] * (nodes[None, :] + 1.0)
        integration_weight = 0.5 * upper_z[:, None] * base_weights[None, :]
        shifted_centers = centers[sl] + shift_a[sl, None]
        standardized_survival = (
            shifted_centers[:, None, :] - z[:, :, None]
        ) / scale_a[sl, None, None]
        survival_term = stdtr(
            nu_a[sl, None, None], standardized_survival
        )
        survival_term = np.where(mask[sl, None, :], survival_term, 0.0)
        survival = survival_term.sum(axis=2) / finite_count[sl, None]

        exp_z = np.exp(z)
        first[sl] = np.sum(integration_weight * exp_z * survival, axis=1)
        if second is not None:
            second[sl] = np.sum(
                integration_weight
                * 2.0
                * np.expm1(z)
                * exp_z
                * survival,
                axis=1,
            )

        if derivative is not None:
            log_density = _coherent_student_t_log_density_array(
                z[:, :, None],
                shifted_centers[:, None, :],
                scale_a[sl, None, None],
                nu_a[sl, None, None],
            )
            density_term = np.where(
                mask[sl, None, :], np.exp(log_density), 0.0
            )
            density = density_term.sum(axis=2) / finite_count[sl, None]
            derivative[sl] = np.sum(
                integration_weight * exp_z * density, axis=1
            )

    finite_outputs = [first]
    if second is not None:
        finite_outputs.append(second)
    if derivative is not None:
        finite_outputs.append(derivative)
    if not all(np.isfinite(output).all() for output in finite_outputs):
        raise FloatingPointError("non-finite censored mean-solve quadrature")
    return (
        np.clip(first, 0.0, upper_a),
        (
            None
            if second is None
            else np.maximum(second, np.square(first))
        ),
        (
            None
            if derivative is None
            else np.maximum(derivative, 0.0)
        ),
    )


def solve_censored_mean_preserving_shift_batch(
    centers_z: np.ndarray,
    center_mask: np.ndarray,
    target_raw_mean: np.ndarray,
    scale: np.ndarray,
    nu: np.ndarray,
    upper_raw: np.ndarray,
    *,
    mean_floor: float,
    quadrature_order: int = 128,
    atol: float = 1e-12,
    rtol: float = 1e-9,
    max_iterations: int = 96,
) -> dict[str, np.ndarray]:
    ""






    centers = np.asarray(centers_z, dtype=float)
    mask = np.asarray(center_mask, dtype=bool)
    target = np.asarray(target_raw_mean, dtype=float).reshape(-1)
    scale_a = np.asarray(scale, dtype=float).reshape(-1)
    nu_a = np.asarray(nu, dtype=float).reshape(-1)
    upper_a = np.asarray(upper_raw, dtype=float).reshape(-1)
    if centers.ndim != 2 or mask.shape != centers.shape:
        raise ValueError("censored mean-solve centers and masks must be aligned")
    group_count = centers.shape[0]
    if not (
        target.size
        == scale_a.size
        == nu_a.size
        == upper_a.size
        == group_count
    ):
        raise ValueError("censored mean-solve inputs have incompatible shapes")
    if np.any(~np.isfinite(target) | (target < 0.0)):
        raise ValueError("censored target raw means must be finite and nonnegative")
    floor = float(mean_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("censored mean floor must be finite and positive")
    if not np.isfinite(atol) or float(atol) <= 0.0:
        raise ValueError("censored mean-solve atol must be finite and positive")
    if not np.isfinite(rtol) or float(rtol) <= 0.0:
        raise ValueError("censored mean-solve rtol must be finite and positive")
    if np.any(~np.isfinite(upper_a) | (upper_a <= 0.0)):
        raise ValueError("censored upper bounds must be finite and positive")
    invalid = target >= upper_a
    if np.any(invalid):
        examples = [
            {
                "position": int(position),
                "target_raw_mean": float(target[position]),
                "upper_raw": float(upper_a[position]),
            }
            for position in np.flatnonzero(invalid)[:5]
        ]
        raise ValueError(
            "censored mean-preserving target must lie below its frozen upper "
            f"bound; examples={examples}"
        )
    floor_applied = target <= floor
    effective = target.copy()
    effective[floor_applied] = floor
    if np.any(effective >= upper_a):
        raise ValueError("censored mean floor reaches or exceeds an upper bound")
    tolerance = float(atol) + float(rtol) * np.maximum(effective, floor)

    zero_shift = np.zeros(group_count, dtype=float)
    zero_mean, _, _ = _censored_shift_moments_batch(
        centers,
        mask,
        zero_shift,
        scale_a,
        nu_a,
        upper_a,
        quadrature_order=int(quadrature_order),
        compute_second=False,
        compute_derivative=False,
    )
    at_zero = np.abs(zero_mean - effective) <= tolerance
    lower = np.zeros(group_count, dtype=float)
    upper = np.zeros(group_count, dtype=float)

    need_lower = ~at_zero & (zero_mean > effective)
    candidate = -np.maximum(scale_a, 1.0)
    unresolved = need_lower.copy()
    for _ in range(64):
        if not np.any(unresolved):
            break
        active = np.flatnonzero(unresolved)
        candidate_mean, _, _ = _censored_shift_moments_batch(
            centers[active],
            mask[active],
            candidate[active],
            scale_a[active],
            nu_a[active],
            upper_a[active],
            quadrature_order=int(quadrature_order),
            compute_second=False,
            compute_derivative=False,
        )
        bracketed = active[candidate_mean <= effective[active]]
        lower[bracketed] = candidate[bracketed]
        unresolved[bracketed] = False
        candidate[unresolved] *= 2.0
    if np.any(unresolved):
        raise RuntimeError("failed to bracket negative censored center shifts")

    need_upper = ~at_zero & (zero_mean < effective)
    candidate = np.maximum(scale_a, 1.0)
    unresolved = need_upper.copy()
    for _ in range(64):
        if not np.any(unresolved):
            break
        active = np.flatnonzero(unresolved)
        candidate_mean, _, _ = _censored_shift_moments_batch(
            centers[active],
            mask[active],
            candidate[active],
            scale_a[active],
            nu_a[active],
            upper_a[active],
            quadrature_order=int(quadrature_order),
            compute_second=False,
            compute_derivative=False,
        )
        bracketed = active[candidate_mean >= effective[active]]
        upper[bracketed] = candidate[bracketed]
        unresolved[bracketed] = False
        candidate[unresolved] *= 2.0
    if np.any(unresolved):
        raise RuntimeError("failed to bracket positive censored center shifts")

    current = np.where(at_zero, 0.0, 0.5 * (lower + upper))
    converged = at_zero.copy()
    iterations = np.zeros(group_count, dtype=int)
    for iteration in range(1, int(max_iterations) + 1):
        active = np.flatnonzero(~converged)
        if active.size == 0:
            break
        mean, _, derivative = _censored_shift_moments_batch(
            centers[active],
            mask[active],
            current[active],
            scale_a[active],
            nu_a[active],
            upper_a[active],
            quadrature_order=int(quadrature_order),
            compute_second=False,
            compute_derivative=True,
        )
        assert derivative is not None
        residual = mean - effective[active]
        newly_converged_local = np.abs(residual) <= tolerance[active]
        newly_converged = active[newly_converged_local]
        iterations[newly_converged] = iteration
        converged[newly_converged] = True
        if np.all(converged):
            break
        remaining = active[~newly_converged_local]
        remaining_residual = residual[~newly_converged_local]
        remaining_derivative = derivative[~newly_converged_local]
        below = remaining_residual < 0.0
        lower[remaining[below]] = current[remaining[below]]
        upper[remaining[~below]] = current[remaining[~below]]
        midpoint = 0.5 * (lower[remaining] + upper[remaining])
        newton = np.full(remaining.size, np.nan, dtype=float)
        usable = np.isfinite(remaining_derivative) & (
            remaining_derivative > np.finfo(float).tiny
        )
        newton[usable] = (
            current[remaining[usable]]
            - remaining_residual[usable] / remaining_derivative[usable]
        )
        inside = (
            usable
            & (newton > lower[remaining])
            & (newton < upper[remaining])
        )
        current[remaining] = midpoint
        current[remaining[inside]] = newton[inside]
    if not np.all(converged):
        bad = np.flatnonzero(~converged)[:5].tolist()
        raise RuntimeError(
            "censored mean-preserving center solver did not converge; "
            f"example positions={bad}"
        )

    iterations[at_zero] = 0
    mean, second, _ = _censored_shift_moments_batch(
        centers,
        mask,
        current,
        scale_a,
        nu_a,
        upper_a,
        quadrature_order=int(quadrature_order),
        compute_second=True,
        compute_derivative=False,
    )
    assert second is not None
    residual = mean - effective
    if np.any(np.abs(residual) > 5.0 * tolerance):
        bad = np.flatnonzero(np.abs(residual) > 5.0 * tolerance)[:5].tolist()
        raise RuntimeError(
            "censored center solver failed its raw-mean validation; "
            f"example positions={bad}"
        )
    return {
        "shift": current,
        "requested_mean": target,
        "effective_target": effective,
        "mean_floor_applied": floor_applied,
        "mean": mean,
        "second_moment": second,
        "variance": np.maximum(second - np.square(mean), 0.0),
        "residual": residual,
        "iterations": iterations,
    }


def _fit_censored_predictive_from_centers(
    *,
    kind: str,
    centers_z: np.ndarray,
    scale: float,
    nu: float,
    lower_raw: float,
    upper_raw: float,
    quadrature_order: int,
) -> CensoredStudentTPredictive:
    ""

    centers = _validate_censored_parameters(
        centers_z, scale, nu, lower_raw, upper_raw
    )
    lower = np.asarray([0.0], dtype=float)
    upper = np.asarray([float(upper_raw)], dtype=float)
    z, raw, log_measure, lower_z, upper_z = (
        _coherent_transform_quadrature_batch(
            lower, upper, int(quadrature_order)
        )
    )
    kernel_log_density = _coherent_student_t_log_density_array(
        z[:, :, None],
        centers[None, None, :],
        np.asarray(float(scale)).reshape(1, 1, 1),
        np.asarray(float(nu)).reshape(1, 1, 1),
    )
    base_log_density = _coherent_logsumexp(kernel_log_density, axis=2) - log(centers.size)
    interior_weight = np.exp(base_log_density[0] + log_measure[0])

    lower_log_probability = _censored_student_t_log_boundary_probability(
        0.0, centers, scale, nu, upper_tail=False
    )
    upper_log_probability = _censored_student_t_log_boundary_probability(
        float(upper_z[0]), centers, scale, nu, upper_tail=True
    )
    lower_probability = float(np.exp(lower_log_probability))
    upper_probability = float(np.exp(upper_log_probability))
    continuous_probability = max(
        0.0, 1.0 - lower_probability - upper_probability
    )
    numerical_interior_mass = float(interior_weight.sum())
    if continuous_probability > 0.0:
        if (
            not np.isfinite(numerical_interior_mass)
            or numerical_interior_mass <= 0.0
        ):
            raise FloatingPointError(
                "censored Student-t interior quadrature has no finite mass"
            )
        interior_weight *= continuous_probability / numerical_interior_mass
    else:
        interior_weight[:] = 0.0

    quadrature_z = np.concatenate(
        ([float(lower_z[0])], z[0], [float(upper_z[0])])
    )
    quadrature_raw = np.concatenate(([0.0], raw[0], [float(upper_raw)]))
    quadrature_probability = np.concatenate(
        ([lower_probability], interior_weight, [upper_probability])
    )
    total_probability = float(quadrature_probability.sum())
    if not np.isfinite(total_probability) or total_probability <= 0.0:
        raise FloatingPointError("invalid censored Student-t total probability")
    quadrature_probability /= total_probability
    mean = float(np.dot(quadrature_probability, quadrature_raw))
    second = float(
        np.dot(quadrature_probability, np.square(quadrature_raw))
    )
    return CensoredStudentTPredictive(
        kind=str(kind),
        centers_z=centers.copy(),
        scale=float(scale),
        nu=float(nu),
        lower_raw=0.0,
        frozen_upper_raw=float(upper_raw),
        upper_raw=float(upper_raw),
        lower_z=float(lower_z[0]),
        upper_z=float(upper_z[0]),
        mean=mean,
        second_moment=second,
        variance=max(0.0, second - mean * mean),
        lower_atom_probability=float(quadrature_probability[0]),
        upper_atom_probability=float(quadrature_probability[-1]),
        continuous_probability=float(quadrature_probability[1:-1].sum()),
        quadrature_order=int(quadrature_order),
        quadrature_z=quadrature_z,
        quadrature_raw=quadrature_raw,
        quadrature_probability=quadrature_probability,
    )


def fit_censored_moment_predictive(
    loc_z: float,
    scale: float,
    nu: float,
    lower_raw: float,
    upper_raw: float,
    *,
    quadrature_order: int = 128,
) -> CensoredStudentTPredictive:
    ""

    return _fit_censored_predictive_from_centers(
        kind="moment",
        centers_z=np.asarray([float(loc_z)]),
        scale=float(scale),
        nu=float(nu),
        lower_raw=float(lower_raw),
        upper_raw=float(upper_raw),
        quadrature_order=int(quadrature_order),
    )


def fit_censored_draw_predictive(
    draws_raw: np.ndarray,
    bandwidth: float,
    nu: float,
    lower_raw: float,
    upper_raw: float,
    *,
    quadrature_order: int = 128,
) -> CensoredStudentTPredictive:
    ""

    raw = np.asarray(draws_raw, dtype=float).reshape(-1)
    raw = np.maximum(raw[np.isfinite(raw)], 0.0)
    if raw.size == 0:
        raise ValueError("censored draw predictive has no finite draws")
    return _fit_censored_predictive_from_centers(
        kind="draw_kernel",
        centers_z=np.log1p(raw),
        scale=float(bandwidth),
        nu=float(nu),
        lower_raw=float(lower_raw),
        upper_raw=float(upper_raw),
        quadrature_order=int(quadrature_order),
    )


def fit_mean_preserving_censored_moment_predictive(
    target_raw_mean: float,
    scale: float,
    nu: float,
    lower_raw: float,
    upper_raw: float,
    *,
    mean_floor: float,
    quadrature_order: int = 128,
) -> CensoredStudentTPredictive:
    ""

    target = max(float(target_raw_mean), 0.0)
    base_center = np.asarray([[np.log1p(target)]], dtype=float)
    mask = np.ones_like(base_center, dtype=bool)
    solved = solve_censored_mean_preserving_shift_batch(
        base_center,
        mask,
        np.asarray([target]),
        np.asarray([float(scale)]),
        np.asarray([float(nu)]),
        np.asarray([float(upper_raw)]),
        mean_floor=float(mean_floor),
        quadrature_order=int(quadrature_order),
    )
    shifted_center = base_center[0] + float(solved["shift"][0])
    predictive = _fit_censored_predictive_from_centers(
        kind="moment",
        centers_z=shifted_center,
        scale=float(scale),
        nu=float(nu),
        lower_raw=float(lower_raw),
        upper_raw=float(upper_raw),
        quadrature_order=int(quadrature_order),
    )
    mean = float(solved["mean"][0])
    second = float(solved["second_moment"][0])
    return replace(
        predictive,
        mean=mean,
        second_moment=second,
        variance=max(0.0, second - mean * mean),
        requested_mean=float(solved["requested_mean"][0]),
        effective_target_mean=float(solved["effective_target"][0]),
        location_shift=float(solved["shift"][0]),
        mean_floor=float(mean_floor),
        mean_floor_applied=bool(solved["mean_floor_applied"][0]),
        mean_constraint_applied=True,
        mean_constraint_residual=float(solved["residual"][0]),
    )


def fit_mean_preserving_censored_draw_predictive(
    draws_raw: np.ndarray,
    bandwidth: float,
    nu: float,
    lower_raw: float,
    upper_raw: float,
    *,
    mean_floor: float,
    quadrature_order: int = 128,
) -> CensoredStudentTPredictive:
    ""

    raw = np.asarray(draws_raw, dtype=float).reshape(-1)
    raw = np.maximum(raw[np.isfinite(raw)], 0.0)
    if raw.size == 0:
        raise ValueError("mean-preserving censored draw predictive has no finite draws")
    centers = np.log1p(raw)[None, :]
    mask = np.ones_like(centers, dtype=bool)
    target = float(np.mean(raw))
    solved = solve_censored_mean_preserving_shift_batch(
        centers,
        mask,
        np.asarray([target]),
        np.asarray([float(bandwidth)]),
        np.asarray([float(nu)]),
        np.asarray([float(upper_raw)]),
        mean_floor=float(mean_floor),
        quadrature_order=int(quadrature_order),
    )
    shifted_centers = centers[0] + float(solved["shift"][0])
    predictive = _fit_censored_predictive_from_centers(
        kind="draw_kernel",
        centers_z=shifted_centers,
        scale=float(bandwidth),
        nu=float(nu),
        lower_raw=float(lower_raw),
        upper_raw=float(upper_raw),
        quadrature_order=int(quadrature_order),
    )
    mean = float(solved["mean"][0])
    second = float(solved["second_moment"][0])
    return replace(
        predictive,
        mean=mean,
        second_moment=second,
        variance=max(0.0, second - mean * mean),
        requested_mean=float(solved["requested_mean"][0]),
        effective_target_mean=float(solved["effective_target"][0]),
        location_shift=float(solved["shift"][0]),
        mean_floor=float(mean_floor),
        mean_floor_applied=bool(solved["mean_floor_applied"][0]),
        mean_constraint_applied=True,
        mean_constraint_residual=float(solved["residual"][0]),
    )


def censored_predictive_logscore_raw(
    observed_raw: float | np.ndarray,
    predictive: CensoredStudentTPredictive,
) -> float | np.ndarray:
    ""

    values = np.asarray(observed_raw, dtype=float)
    flat = values.reshape(-1)
    result = np.full(flat.shape, float("-inf"), dtype=float)
    at_lower = flat == predictive.lower_raw
    at_upper = flat == predictive.upper_raw
    interior = (
        (flat > predictive.lower_raw) & (flat < predictive.upper_raw)
    )
    if np.any(at_lower):
        result[at_lower] = log(predictive.lower_atom_probability)
    if np.any(at_upper):
        result[at_upper] = log(predictive.upper_atom_probability)
    if np.any(interior):
        z = np.log1p(flat[interior])
        log_density = _coherent_student_t_log_density_array(
            z[:, None],
            predictive.centers_z[None, :],
            np.asarray(predictive.scale).reshape(1, 1),
            np.asarray(predictive.nu).reshape(1, 1),
        )
        result[interior] = (
            _coherent_logsumexp(log_density, axis=1)
            - log(predictive.centers_z.size)
        )
    shaped = result.reshape(values.shape)
    return float(shaped) if shaped.ndim == 0 else shaped


def _score_censored_moment_batch(
    observed_raw: np.ndarray,
    loc_z: np.ndarray,
    scale: np.ndarray,
    nu: np.ndarray,
    upper_raw: np.ndarray,
) -> dict[str, np.ndarray]:
    ""

    from scipy.stats import t as student_t

    observed = np.asarray(observed_raw, dtype=float)
    location = np.asarray(loc_z, dtype=float)
    scale_a = np.asarray(scale, dtype=float)
    nu_a = np.asarray(nu, dtype=float)
    upper = np.asarray(upper_raw, dtype=float)
    if np.any(~np.isfinite(scale_a) | (scale_a <= 0.0)):
        raise ValueError("censored Student-t scale must be finite and positive")
    if np.any(~np.isfinite(nu_a) | (nu_a <= 0.0)):
        raise ValueError("censored Student-t requires finite positive nu")
    if np.any(~np.isfinite(upper) | (upper <= 0.0)):
        raise ValueError(
            "censored Student-t requires finite positive frozen upper bounds"
        )
    lower_standardized = -location / scale_a
    upper_z = np.log1p(upper)
    upper_standardized = (upper_z - location) / scale_a
    lower_log_probability = student_t.logcdf(
        lower_standardized, df=nu_a
    )
    upper_log_probability = student_t.logsf(
        upper_standardized, df=nu_a
    )
    result = np.full(observed.shape, float("-inf"), dtype=float)
    at_lower = observed == 0.0
    at_upper = observed == upper
    interior = (observed > 0.0) & (observed < upper)
    result[at_lower] = lower_log_probability[at_lower]
    result[at_upper] = upper_log_probability[at_upper]
    if np.any(interior):
        result[interior] = _coherent_student_t_log_density_array(
            np.log1p(observed[interior]),
            location[interior],
            scale_a[interior],
            nu_a[interior],
        )
    return {
        "log_score": result,
        "lower_atom_probability": np.exp(lower_log_probability),
        "upper_atom_probability": np.exp(upper_log_probability),
        "observation_in_support": (
            (observed >= 0.0) & (observed <= upper)
        ),
    }


def _score_coherent_moment_batch(
    observed_raw: np.ndarray,
    target_raw: np.ndarray,
    loc_z: np.ndarray,
    scale: np.ndarray,
    nu: np.ndarray,
    lower_raw: np.ndarray,
    upper_raw: np.ndarray,
    *,
    mean_floor: float,
    quadrature_order: int,
    support_expansion_multiplier: float | None,
) -> dict[str, np.ndarray]:
    ""

    observed = np.maximum(np.asarray(observed_raw, dtype=float), 0.0)
    target = np.maximum(np.asarray(target_raw, dtype=float), 0.0)
    location = np.asarray(loc_z, dtype=float)
    scale_a = np.asarray(scale, dtype=float)
    nu_a = np.asarray(nu, dtype=float)
    lower = np.asarray(lower_raw, dtype=float)
    frozen_upper = np.asarray(upper_raw, dtype=float)
    upper, support_expanded = _coherent_effective_upper_bounds(
        target,
        frozen_upper,
        support_expansion_multiplier,
    )
    count = target.size
    fields = {
        "log_score": np.empty(count, dtype=float),
        "effective_target": np.empty(count, dtype=float),
        "mean_floor_applied": np.empty(count, dtype=bool),
        "tilt": np.empty(count, dtype=float),
        "log_normalizer": np.empty(count, dtype=float),
        "mean": np.empty(count, dtype=float),
        "second_moment": np.empty(count, dtype=float),
        "variance": np.empty(count, dtype=float),
        "observation_in_support": np.empty(count, dtype=bool),
        "frozen_upper": frozen_upper.copy(),
        "effective_upper": upper.copy(),
        "support_expanded": support_expanded,
    }
    order = int(quadrature_order)
                                                                      
    block_size = max(64, 262_144 // order)
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        sl = slice(start, stop)
        effective, floor_applied = _coherent_effective_targets(
            target[sl], lower[sl], upper[sl], mean_floor
        )
        z, raw, log_measure, _, _ = _coherent_transform_quadrature_batch(
            lower[sl], upper[sl], order
        )
        base_log_density = _coherent_student_t_log_density_array(
            z,
            location[sl, None],
            scale_a[sl, None],
            nu_a[sl, None],
        )
        tilt, log_z, mean, second, variance, _ = _fit_coherent_tilt_batch(
            raw,
            base_log_density + log_measure,
            effective,
            return_probability=False,
        )
        observed_z = np.log1p(observed[sl])
        observed_base = _coherent_student_t_log_density_array(
            observed_z,
            location[sl],
            scale_a[sl],
            nu_a[sl],
        )
        in_support = (observed[sl] >= lower[sl]) & (observed[sl] <= upper[sl])
        log_score = observed_base + tilt * observed[sl] - log_z
        log_score[~in_support] = float("-inf")
        fields["log_score"][sl] = log_score
        fields["effective_target"][sl] = effective
        fields["mean_floor_applied"][sl] = floor_applied
        fields["tilt"][sl] = tilt
        fields["log_normalizer"][sl] = log_z
        fields["mean"][sl] = mean
        fields["second_moment"][sl] = second
        fields["variance"][sl] = variance
        fields["observation_in_support"][sl] = in_support
    return fields

def score_archive_rows(ledger_rows: pd.DataFrame, archive: pd.DataFrame, config: BridgeConfig) -> pd.DataFrame:
    required_ledger = {"forecast_id", "observed_value", "observed_mask"}; required_archive = {"forecast_id", "model_id", "particle_id", "pred_mean", "pred_var", "component", "horizon"}
    if missing := sorted(required_ledger - set(ledger_rows.columns)): raise ValueError(f"ledger missing columns {missing}")
    if missing := sorted(required_archive - set(archive.columns)): raise ValueError(f"archive missing columns {missing}")
    # Callers may score one split (for example, test) against an archive that
    # contains train/validation/embargo rows as well.  Native-horizon
    # provenance is a contract for the rows being scored; unrelated archive
    # rows must not be treated as missing from the supplied ledger subset.
    scoring_ids = set(ledger_rows["forecast_id"].astype(str))
    archive_for_scoring = archive.loc[
        archive["forecast_id"].astype(str).isin(scoring_ids)
    ].copy()
    native_violations = validate_native_horizon_provenance(
        archive_for_scoring, ledger_rows
    )
    if not native_violations.empty:
        summary = native_violations.head(10).to_dict("records")
        raise ValueError(f"invalid native-horizon provenance: {summary}")
    meta_cols = ["forecast_id", "observed_value", "observed_mask", "mode", "component", "horizon", "forecast_origin", "target_time", "features_available_until"]
    ledger_meta = ledger_rows[[c for c in meta_cols if c in ledger_rows.columns]].drop_duplicates("forecast_id").copy()
    rows = archive.merge(ledger_meta, on="forecast_id", how="inner", suffixes=("_archive", "_ledger")).copy()
    if rows.empty:
        return rows
    if "component_archive" in rows.columns and "component_ledger" in rows.columns:
        bad = rows["component_archive"].astype(str) != rows["component_ledger"].astype(str)
        if bad.any():
            raise ValueError(f"archive/ledger component mismatch for {int(bad.sum())} scored rows")
        rows["component"] = rows["component_ledger"]
    if "horizon_archive" in rows.columns and "horizon_ledger" in rows.columns:
        bad = pd.to_numeric(rows["horizon_archive"], errors="coerce").astype("Int64") != pd.to_numeric(rows["horizon_ledger"], errors="coerce").astype("Int64")
        if bad.any():
            raise ValueError(f"archive/ledger horizon mismatch for {int(bad.sum())} scored rows")
        rows["horizon"] = rows["horizon_ledger"]
    if "mode_archive" in rows.columns and "mode_ledger" in rows.columns:
        bad = rows["mode_archive"].astype(str) != rows["mode_ledger"].astype(str)
        if bad.any():
            raise ValueError(f"archive/ledger mode mismatch for {int(bad.sum())} scored rows")
        rows["mode"] = rows["mode_ledger"]
    elif "mode_ledger" in rows.columns:
        rows["mode"] = rows["mode_ledger"]
    elif "mode_archive" in rows.columns:
        rows["mode"] = rows["mode_archive"]
    elif "mode" not in rows.columns:
        rows["mode"] = ""
    for field in ["forecast_origin", "target_time"]:
        archive_col = f"{field}_archive"
        ledger_col = f"{field}_ledger"
        if archive_col in rows.columns and ledger_col in rows.columns:
            bad = pd.to_datetime(rows[archive_col], errors="coerce") != pd.to_datetime(rows[ledger_col], errors="coerce")
            if bad.any():
                raise ValueError(f"archive/ledger {field} mismatch for {int(bad.sum())} scored rows")
    for prefix in ["", "_archive", "_ledger"]:
        feat = f"features_available_until{prefix}"
        origin = f"forecast_origin{prefix}"
        if feat in rows.columns and origin in rows.columns:
            bad = pd.to_datetime(rows[feat], errors="coerce") > pd.to_datetime(rows[origin], errors="coerce")
            if bad.any():
                raise ValueError(f"features_available_until exceeds forecast_origin for {int(bad.sum())} scored rows")
    mode = rows["mode"]
    component = rows["component"]
    horizon = pd.to_numeric(rows["horizon"], errors="raise").astype(int)
    horizon_weight = pd.Series(np.nan, index=rows.index, dtype=float)
    horizon_weight = horizon_weight.fillna(horizon.astype(str).map(config.horizon_weight))
    horizon_weight = horizon_weight.fillna(horizon.map(config.horizon_weight)).fillna(1.0).astype(float)
    component_weight = _lookup_bridge_series(config.component_weight, mode, component, horizon, 1.0)
    rows["event_weight"] = horizon_weight.to_numpy(dtype=float) * component_weight

    if (
        config.predictive_contract == COHERENT_MEAN_PRESERVING_TRUNCATED_T
        and config.distribution != "student_t"
    ):
        raise ValueError(
            "coherent mean-preserving contract requires distribution='student_t'"
        )
    if config.predictive_contract in COHERENT_CENSORED_CONTRACTS:
        if config.distribution != "student_t":
            raise ValueError(
                "coherent censored contract requires distribution='student_t'"
            )
        if config.transform != "log1p":
            raise ValueError(
                "coherent censored Student-t requires log1p transform"
            )
        if float(config.truncation_lower_raw) != 0.0:
            raise ValueError(
                "coherent censored Student-t requires truncation_lower_raw=0"
            )

    observed = rows["observed_mask"].astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes", "y"})
    y_raw = rows["observed_value"].astype(float).to_numpy()
    mu_raw = rows["pred_mean"].astype(float).to_numpy()
    var_raw = np.maximum(rows["pred_var"].astype(float).to_numpy(), 0.0)
    if config.transform == "log1p":
        y = np.log1p(np.maximum(y_raw, 0.0))
        mu = np.log1p(np.maximum(mu_raw, 0.0))
        if config.predictive_contract in COHERENT_CENSORED_CONTRACTS:
            pred_v = np.asarray(
                stable_log1p_transform_var(mu_raw, var_raw), dtype=float
            )
        else:
            pred_v = var_raw / np.square(
                1.0 + np.maximum(mu_raw, 0.0)
            )
    elif config.transform == "identity":
        y, mu, pred_v = y_raw, mu_raw, var_raw
    else:
        raise ValueError(f"unknown transform {config.transform!r}")

    sigma = _lookup_bridge_series(
        config.sigma_by_component, mode, component, horizon, config.default_sigma
    )
    if config.distribution == "negative_binomial":
        dispersion = np.maximum(sigma, 1e-3)
        mean = np.maximum(mu_raw, 1e-12)
        counts = np.maximum(y_raw, 0.0)
        probability = dispersion / (dispersion + mean)
        log_scores = (
            _gammaln_array(counts + dispersion)
            - _gammaln_array(dispersion)
            - _gammaln_array(counts + 1.0)
            + dispersion * np.log(probability)
            + counts * np.log(np.maximum(1.0 - probability, 1e-300))
        )
    else:
        gamma = np.maximum(
            _lookup_bridge_series(config.gamma_by_component, mode, component, horizon, config.default_gamma),
            0.0,
        )
        scale = np.sqrt(np.maximum(gamma * pred_v, 0.0) + np.square(sigma) + config.min_scale**2)
        residual = (y - mu) / np.maximum(scale, 1e-12)
        if config.predictive_contract == COHERENT_MEAN_PRESERVING_TRUNCATED_T:
            if config.transform != "log1p":
                raise ValueError(
                    "coherent mean-preserving truncated Student-t requires log1p transform"
                )
            if config.distribution != "student_t":
                raise ValueError(
                    "coherent mean-preserving contract requires distribution='student_t'"
                )
            nu = _lookup_bridge_series(
                config.nu_by_component, mode, component, horizon, config.nu
            )
            upper_raw = _lookup_bridge_series(
                config.truncation_upper_raw_by_component,
                mode,
                component,
                horizon,
                config.default_truncation_upper_raw,
            )
            lower_raw = np.full(len(rows), float(config.truncation_lower_raw))
            coherent = _score_coherent_moment_batch(
                y_raw,
                np.maximum(mu_raw, 0.0),
                mu,
                scale,
                nu,
                lower_raw,
                upper_raw,
                mean_floor=float(config.truncation_zero_mean_epsilon),
                quadrature_order=int(config.truncation_quadrature_order),
                support_expansion_multiplier=(
                    config.truncation_support_expansion_multiplier
                ),
            )
            log_scores = coherent["log_score"]
        elif config.predictive_contract in COHERENT_CENSORED_CONTRACTS:
            nu = _lookup_bridge_series(
                config.nu_by_component, mode, component, horizon, config.nu
            )
            upper_raw = _lookup_bridge_series(
                config.truncation_upper_raw_by_component,
                mode,
                component,
                horizon,
                config.default_truncation_upper_raw,
            )
            censored_location = mu
            censored_mean_solution = None
            if (
                config.predictive_contract
                == COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T
            ):
                censored_mean_solution = (
                    solve_censored_mean_preserving_shift_batch(
                        mu[:, None],
                        np.ones((len(mu), 1), dtype=bool),
                        np.maximum(mu_raw, 0.0),
                        scale,
                        nu,
                        upper_raw,
                        mean_floor=float(
                            config.truncation_zero_mean_epsilon
                        ),
                        quadrature_order=int(
                            config.truncation_quadrature_order
                        ),
                    )
                )
                censored_location = (
                    mu + censored_mean_solution["shift"]
                )
            censored = _score_censored_moment_batch(
                y_raw,
                censored_location,
                scale,
                nu,
                upper_raw,
            )
            log_scores = censored["log_score"]
        elif config.distribution == "gaussian":
            log_scores = -0.5 * np.log(2.0 * pi) - np.log(scale) - 0.5 * np.square(residual)
        elif config.distribution == "student_t":
            nu = _lookup_bridge_series(config.nu_by_component, mode, component, horizon, config.nu)
            if np.any((nu <= 0.0) | np.isnan(nu)):
                raise ValueError("Student-t nu must be positive or infinity")
            gaussian_limit = np.isinf(nu)
            log_scores = np.empty_like(residual, dtype=float)
            log_scores[gaussian_limit] = (
                -0.5 * np.log(2.0 * pi)
                - np.log(scale[gaussian_limit])
                - 0.5 * np.square(residual[gaussian_limit])
            )
            finite_nu = ~gaussian_limit
            nu_f = nu[finite_nu]
            residual_f = residual[finite_nu]
            log_scores[finite_nu] = (
                _gammaln_array((nu_f + 1.0) / 2.0)
                - _gammaln_array(nu_f / 2.0)
                - 0.5 * np.log(nu_f * pi)
                - np.log(scale[finite_nu])
                - ((nu_f + 1.0) / 2.0) * np.log1p(np.square(residual_f) / nu_f)
            )
        else:
            raise ValueError(f"unknown bridge distribution {config.distribution!r}")
    rows["log_score"] = np.where(observed.to_numpy(), log_scores, 0.0)
    if config.predictive_contract == COHERENT_MEAN_PRESERVING_TRUNCATED_T:
        rows["predictive_contract"] = config.predictive_contract
        rows["truncation_density_kind"] = "mean_constrained_exponential_tilt"
        rows["truncation_lower_raw"] = lower_raw
        rows["truncation_upper_raw"] = coherent["effective_upper"]
        rows["truncation_upper_raw_frozen"] = coherent["frozen_upper"]
        rows["truncation_upper_raw_effective"] = coherent["effective_upper"]
        rows["truncation_support_expanded"] = coherent["support_expanded"]
        rows["truncation_support_expansion_policy"] = (
            _coherent_support_expansion_policy(
                config.truncation_support_expansion_multiplier
            )
        )
        rows["truncation_support_expansion_multiplier"] = (
            config.truncation_support_expansion_multiplier
        )
        rows["constrained_target_raw_mean"] = np.maximum(mu_raw, 0.0)
        rows["effective_target_raw_mean"] = coherent["effective_target"]
        rows["mean_floor_applied"] = coherent["mean_floor_applied"]
        rows["exponential_tilt"] = coherent["tilt"]
        rows["truncated_log_normalizer"] = coherent["log_normalizer"]
        rows["truncated_raw_mean"] = coherent["mean"]
        rows["truncated_raw_second_moment"] = coherent["second_moment"]
        rows["truncated_raw_variance"] = coherent["variance"]
        rows["truncation_observation_in_support"] = coherent[
            "observation_in_support"
        ]
        rows["truncation_quadrature_order"] = int(
            config.truncation_quadrature_order
        )
        rows["truncation_mean_floor"] = float(
            config.truncation_zero_mean_epsilon
        )
        rows["truncation_bound_policy"] = str(config.truncation_bound_policy)
    elif config.predictive_contract in COHERENT_CENSORED_CONTRACTS:
        rows["predictive_contract"] = config.predictive_contract
        rows["truncation_density_kind"] = (
            "latent_student_t_censoring_with_boundary_atoms"
        )
        rows["truncation_lower_raw"] = 0.0
        rows["truncation_upper_raw"] = upper_raw
        rows["truncation_upper_raw_frozen"] = upper_raw
        rows["truncation_upper_raw_effective"] = upper_raw
        rows["truncation_support_expanded"] = False
        rows["truncation_support_expansion_policy"] = (
            STRICT_FROZEN_SUPPORT_POLICY
        )
        rows["lower_boundary_atom_probability"] = censored[
            "lower_atom_probability"
        ]
        rows["upper_boundary_atom_probability"] = censored[
            "upper_atom_probability"
        ]
        rows["truncation_observation_in_support"] = censored[
            "observation_in_support"
        ]
        rows["truncation_quadrature_order"] = int(
            config.truncation_quadrature_order
        )
        rows["truncation_bound_policy"] = str(config.truncation_bound_policy)
        rows["nll_measure_basis"] = (
            "log1p_transform_mixed_measure_atoms_at_zero_and_upper_bound"
        )
        rows["boundary_atom_source"] = "latent_student_t_censoring"
        if (
            config.predictive_contract
            == COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T
        ):
            assert censored_mean_solution is not None
            rows["constrained_target_raw_mean"] = censored_mean_solution[
                "requested_mean"
            ]
            rows["effective_target_raw_mean"] = censored_mean_solution[
                "effective_target"
            ]
            rows["mean_floor_applied"] = censored_mean_solution[
                "mean_floor_applied"
            ]
            rows["censored_location_shift"] = censored_mean_solution["shift"]
            rows["censored_raw_mean"] = censored_mean_solution["mean"]
            rows["censored_raw_second_moment"] = censored_mean_solution[
                "second_moment"
            ]
            rows["censored_raw_variance"] = censored_mean_solution["variance"]
            rows["censored_mean_constraint_residual"] = (
                censored_mean_solution["residual"]
            )
            rows["censored_mean_solver_iterations"] = (
                censored_mean_solution["iterations"]
            )
            rows["truncation_mean_floor"] = float(
                config.truncation_zero_mean_epsilon
            )
    return rows

def calibrate_component_sigma(
    validation_ledger: pd.DataFrame,
    validation_archive: pd.DataFrame,
    *,
    transform: str = "log1p",
    min_sigma: float = 0.05,
    default_sigma: float = 0.20,
) -> dict[str, float]:
    ""






    min_sigma = float(min_sigma)
    default_sigma = float(default_sigma)
    if not np.isfinite(min_sigma) or min_sigma <= 0.0:
        raise ValueError("min_sigma must be finite and positive")
    if not np.isfinite(default_sigma) or default_sigma <= 0.0:
        raise ValueError("default_sigma must be finite and positive")
    joined = validation_archive.merge(
        validation_ledger[["forecast_id", "observed_value", "observed_mask"]],
        on="forecast_id",
        how="inner",
    )
    joined = joined[joined["observed_mask"].astype(bool)].copy()
    if joined.empty:
        return {}
    joined["residual"] = [
        transform_value(y, transform) - transform_value(mu, transform)
        for y, mu in zip(joined["observed_value"], joined["pred_mean"])
    ]
    joined["bridge_r_key"] = bridge_r_key_series(joined)
    scales: dict[str, float] = {}
    for key, group in joined.groupby("bridge_r_key", sort=True):
        residuals = pd.to_numeric(group["residual"], errors="coerce").to_numpy(
            dtype=float
        )
        residuals = residuals[np.isfinite(residuals)]
        rmse = (
            float(np.sqrt(np.mean(np.square(residuals))))
            if residuals.size
            else default_sigma
        )
        scales[str(key)] = max(min_sigma, rmse)
    return scales


def calibrate_truncation_upper_bounds(
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    *,
    source_split: str = "train",
    predictive_standard_deviations: float = 4.0,
    safety_multiplier: float = 1.25,
    minimum_upper_raw: float = 1.0,
) -> dict[str, float]:
    ""







    required_ledger = {
        "forecast_id",
        "split",
        "observed_value",
        "observed_mask",
        "component",
        "horizon",
    }
    required_archive = {
        "forecast_id",
        "pred_mean",
        "pred_var",
        "component",
        "horizon",
    }
    if missing := sorted(required_ledger - set(ledger.columns)):
        raise ValueError(f"truncation-bound ledger missing columns {missing}")
    if missing := sorted(required_archive - set(archive.columns)):
        raise ValueError(f"truncation-bound archive missing columns {missing}")
    q = float(predictive_standard_deviations)
    multiplier = float(safety_multiplier)
    floor = float(minimum_upper_raw)
    if not np.isfinite(q) or q < 0.0:
        raise ValueError("predictive_standard_deviations must be finite and nonnegative")
    if not np.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError("safety_multiplier must be finite and at least one")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("minimum_upper_raw must be finite and positive")

    source = ledger[ledger["split"].astype(str).eq(str(source_split))].copy()
    if source.empty:
        raise ValueError(f"truncation-bound source split {source_split!r} is empty")
    source["forecast_id"] = source["forecast_id"].astype(str)
    source["bridge_r_key"] = bridge_r_key_series(source)
    observed = source["observed_mask"].astype(str).str.strip().str.lower().isin(
        {"true", "1", "t", "yes", "y"}
    )
    observed_values = pd.to_numeric(
        source.loc[observed, "observed_value"], errors="coerce"
    ).clip(lower=0.0)
    observed_max = (
        pd.DataFrame(
            {
                "bridge_r_key": source.loc[observed, "bridge_r_key"].astype(str),
                "observed_value": observed_values,
            }
        )
        .dropna(subset=["observed_value"])
        .groupby("bridge_r_key", sort=True)["observed_value"]
        .max()
    )

    source_ids = set(source["forecast_id"].astype(str))
    candidate = archive[archive["forecast_id"].astype(str).isin(source_ids)].copy()
    if candidate.empty:
        raise ValueError("truncation-bound source split has no archived forecasts")
    candidate["bridge_r_key"] = bridge_r_key_series(candidate)
    pred_mean = pd.to_numeric(candidate["pred_mean"], errors="coerce").clip(lower=0.0)
    pred_var = pd.to_numeric(candidate["pred_var"], errors="coerce").clip(lower=0.0)
    candidate["predictive_envelope"] = pred_mean + q * np.sqrt(pred_var)
    envelope_max = (
        candidate.dropna(subset=["predictive_envelope"])
        .groupby("bridge_r_key", sort=True)["predictive_envelope"]
        .max()
    )

    keys = sorted(set(observed_max.index.astype(str)) | set(envelope_max.index.astype(str)))
    if not keys:
        raise ValueError("truncation-bound calibration produced no component keys")
    result: dict[str, float] = {}
    for key in keys:
        base = max(
            floor,
            float(observed_max.get(key, 0.0)),
            float(envelope_max.get(key, 0.0)),
        )
        upper = multiplier * base
        if not np.isfinite(upper) or upper <= 0.0:
            raise ValueError(f"invalid truncation upper bound for {key!r}")
        result[str(key)] = float(upper)
    return result


def _score_coherent_draw_groups(
    joined: pd.DataFrame,
    *,
    keys: list[str],
    observed_z: np.ndarray,
    observed_raw: np.ndarray,
    transformed_draw: np.ndarray,
    draw_raw: np.ndarray,
    bandwidth: np.ndarray,
    nu: np.ndarray,
    lower_raw: np.ndarray,
    upper_raw: np.ndarray,
    config: BridgeConfig,
) -> pd.DataFrame:
    ""

    key_index = pd.MultiIndex.from_frame(joined[keys])
    group_code, _ = pd.factorize(key_index, sort=True)
    group_count = int(group_code.max()) + 1
    positions = np.arange(len(joined), dtype=int)
    first_position = np.full(group_count, len(joined), dtype=int)
    np.minimum.at(first_position, group_code, positions)

    valid_draw = np.isfinite(transformed_draw) & np.isfinite(draw_raw)
    finite_count = np.bincount(
        group_code, weights=valid_draw.astype(float), minlength=group_count
    ).astype(int)
    if np.any(finite_count == 0):
        bad_groups = np.flatnonzero(finite_count == 0)[:5]
        examples = joined.iloc[first_position[bad_groups]][keys].to_dict("records")
        raise ValueError(
            "coherent draw-kernel groups require at least one finite draw; "
            f"examples={examples}"
        )

    def first_and_validate(name: str, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        first = array[first_position]
        minimum = np.full(group_count, np.inf, dtype=float)
        maximum = np.full(group_count, -np.inf, dtype=float)
        np.minimum.at(minimum, group_code, array)
        np.maximum.at(maximum, group_code, array)
        tolerance = 1e-12 * np.maximum(1.0, np.maximum(np.abs(minimum), np.abs(maximum)))
        inconsistent = ~np.isfinite(first) | ((maximum - minimum) > tolerance)
        if np.any(inconsistent):
            bad_groups = np.flatnonzero(inconsistent)[:5]
            examples = joined.iloc[first_position[bad_groups]][keys].to_dict("records")
            raise ValueError(
                f"coherent draw-kernel group has inconsistent {name}; examples={examples}"
            )
        return first

    group_bandwidth = first_and_validate("bandwidth", bandwidth)
    group_nu = first_and_validate("nu", nu)
    group_lower = first_and_validate("lower truncation bound", lower_raw)
    group_upper = first_and_validate("upper truncation bound", upper_raw)
    group_observed_z = first_and_validate("observed transform value", observed_z)
    group_observed_raw = first_and_validate("observed raw value", observed_raw)
    group_is_observed = joined["observed_mask"].to_numpy(dtype=bool)[first_position]

    nonnegative_draw = np.maximum(np.asarray(draw_raw, dtype=float), 0.0)
    raw_sum = np.bincount(
        group_code,
        weights=np.where(valid_draw, nonnegative_draw, 0.0),
        minlength=group_count,
    )
    requested_mean = raw_sum / finite_count
    group_frozen_upper = group_upper.copy()
    group_upper, support_expanded = _coherent_effective_upper_bounds(
        requested_mean,
        group_frozen_upper,
        config.truncation_support_expansion_multiplier,
    )
    effective, floor_applied = _coherent_effective_targets(
        requested_mean,
        group_lower,
        group_upper,
        float(config.truncation_zero_mean_epsilon),
    )

    max_draw_count = int(finite_count.max())
    centers = np.zeros((group_count, max_draw_count), dtype=float)
    center_mask = np.zeros_like(centers, dtype=bool)
    valid_positions = np.flatnonzero(valid_draw)
    valid_codes = group_code[valid_positions]
    order_by_group = np.argsort(valid_codes, kind="stable")
    sorted_positions = valid_positions[order_by_group]
    sorted_codes = valid_codes[order_by_group]
    starts = np.repeat(np.cumsum(finite_count) - finite_count, finite_count)
    within_group = np.arange(sorted_positions.size) - starts
    centers[sorted_codes, within_group] = transformed_draw[sorted_positions]
    center_mask[sorted_codes, within_group] = True

    output = {
        "log_score": np.empty(group_count, dtype=float),
        "tilt": np.empty(group_count, dtype=float),
        "log_normalizer": np.empty(group_count, dtype=float),
        "mean": np.empty(group_count, dtype=float),
        "second_moment": np.empty(group_count, dtype=float),
        "variance": np.empty(group_count, dtype=float),
        "observation_in_support": np.empty(group_count, dtype=bool),
    }
    quadrature_order = int(config.truncation_quadrature_order)
                                                                             
                                                                            
    chunk_size = max(
        8, min(2048, 1_048_576 // (quadrature_order * max_draw_count))
    )
    for start in range(0, group_count, chunk_size):
        stop = min(start + chunk_size, group_count)
        sl = slice(start, stop)
        z, raw, log_measure, _, _ = _coherent_transform_quadrature_batch(
            group_lower[sl], group_upper[sl], quadrature_order
        )
        kernel_log_density = _coherent_student_t_log_density_array(
            z[:, :, None],
            centers[sl, None, :],
            group_bandwidth[sl, None, None],
            group_nu[sl, None, None],
        )
        kernel_log_density = np.where(
            center_mask[sl, None, :], kernel_log_density, float("-inf")
        )
        base_log_density = _coherent_logsumexp(kernel_log_density, axis=2) - np.log(
            finite_count[sl]
        )[:, None]
        tilt, log_z, mean, second, variance, _ = _fit_coherent_tilt_batch(
            raw,
            base_log_density + log_measure,
            effective[sl],
            return_probability=False,
        )
        observed_kernel = _coherent_student_t_log_density_array(
            group_observed_z[sl, None],
            centers[sl],
            group_bandwidth[sl, None],
            group_nu[sl, None],
        )
        observed_kernel = np.where(
            center_mask[sl], observed_kernel, float("-inf")
        )
        observed_base = _coherent_logsumexp(observed_kernel, axis=1) - np.log(
            finite_count[sl]
        )
        in_support = (
            (group_observed_raw[sl] >= group_lower[sl])
            & (group_observed_raw[sl] <= group_upper[sl])
            & group_is_observed[sl]
        )
        log_score = (
            observed_base + tilt * group_observed_raw[sl] - log_z
        )
        log_score[~in_support] = float("-inf")
        output["log_score"][sl] = log_score
        output["tilt"][sl] = tilt
        output["log_normalizer"][sl] = log_z
        output["mean"][sl] = mean
        output["second_moment"][sl] = second
        output["variance"][sl] = variance
        output["observation_in_support"][sl] = in_support

    grouped = joined.iloc[first_position][
        [*keys, "mode", "component", "horizon", "observed_mask", "event_weight"]
    ].reset_index(drop=True)
    grouped["log_score"] = output["log_score"]
    grouped.loc[~grouped["observed_mask"].astype(bool), "log_score"] = 0.0
    grouped["predictive_contract"] = config.predictive_contract
    grouped["truncation_density_kind"] = "mean_constrained_exponential_tilt"
    grouped["truncation_lower_raw"] = group_lower
    grouped["truncation_upper_raw"] = group_upper
    grouped["truncation_upper_raw_frozen"] = group_frozen_upper
    grouped["truncation_upper_raw_effective"] = group_upper
    grouped["truncation_support_expanded"] = support_expanded
    grouped["truncation_support_expansion_policy"] = (
        _coherent_support_expansion_policy(
            config.truncation_support_expansion_multiplier
        )
    )
    grouped["truncation_support_expansion_multiplier"] = (
        config.truncation_support_expansion_multiplier
    )
    grouped["constrained_target_raw_mean"] = requested_mean
    grouped["effective_target_raw_mean"] = effective
    grouped["mean_floor_applied"] = floor_applied
    grouped["exponential_tilt"] = output["tilt"]
    grouped["truncated_log_normalizer"] = output["log_normalizer"]
    grouped["truncated_raw_mean"] = output["mean"]
    grouped["truncated_raw_second_moment"] = output["second_moment"]
    grouped["truncated_raw_variance"] = output["variance"]
    grouped["truncation_observation_in_support"] = output[
        "observation_in_support"
    ]
    grouped["truncation_quadrature_order"] = quadrature_order
    grouped["truncation_mean_floor"] = float(config.truncation_zero_mean_epsilon)
    grouped["truncation_bound_policy"] = str(config.truncation_bound_policy)
    return grouped


def _score_censored_draw_groups(
    joined: pd.DataFrame,
    *,
    keys: list[str],
    observed_raw: np.ndarray,
    transformed_draw: np.ndarray,
    bandwidth: np.ndarray,
    nu: np.ndarray,
    upper_raw: np.ndarray,
    config: BridgeConfig,
    mean_preserving: bool = False,
) -> pd.DataFrame:
    ""

    from scipy.stats import t as student_t

    key_index = pd.MultiIndex.from_frame(joined[keys])
    group_code, _ = pd.factorize(key_index, sort=True)
    group_count = int(group_code.max()) + 1
    positions = np.arange(len(joined), dtype=int)
    first_position = np.full(group_count, len(joined), dtype=int)
    np.minimum.at(first_position, group_code, positions)
    valid_draw = np.isfinite(transformed_draw)
    finite_count = np.bincount(
        group_code,
        weights=valid_draw.astype(float),
        minlength=group_count,
    ).astype(int)
    if np.any(finite_count == 0):
        bad = np.flatnonzero(finite_count == 0)[:5]
        examples = joined.iloc[first_position[bad]][keys].to_dict("records")
        raise ValueError(
            "censored draw-kernel groups require finite draws; "
            f"examples={examples}"
        )
    if np.any(~np.isfinite(bandwidth) | (bandwidth <= 0.0)):
        raise ValueError(
            "censored draw-kernel bandwidth must be finite and positive"
        )
    if np.any(~np.isfinite(nu) | (nu <= 0.0)):
        raise ValueError(
            "censored draw-kernel requires finite positive Student-t nu"
        )
    if np.any(~np.isfinite(upper_raw) | (upper_raw <= 0.0)):
        raise ValueError(
            "censored draw-kernel requires finite positive frozen upper bounds"
        )

    scoring_centers = np.asarray(transformed_draw, dtype=float)
    mean_solution: dict[str, np.ndarray] | None = None
    if mean_preserving:
        max_draw_count = int(finite_count.max())
        centers = np.zeros((group_count, max_draw_count), dtype=float)
        center_mask = np.zeros_like(centers, dtype=bool)
        valid_positions = np.flatnonzero(valid_draw)
        valid_codes = group_code[valid_positions]
        order_by_group = np.argsort(valid_codes, kind="stable")
        sorted_positions = valid_positions[order_by_group]
        sorted_codes = valid_codes[order_by_group]
        starts = np.repeat(
            np.cumsum(finite_count) - finite_count, finite_count
        )
        within_group = np.arange(sorted_positions.size) - starts
        centers[sorted_codes, within_group] = scoring_centers[
            sorted_positions
        ]
        center_mask[sorted_codes, within_group] = True
        raw_draw = pd.to_numeric(
            joined["draw"], errors="coerce"
        ).to_numpy(dtype=float)
        nonnegative_draw = np.maximum(raw_draw, 0.0)
        raw_sum = np.bincount(
            group_code,
            weights=np.where(valid_draw, nonnegative_draw, 0.0),
            minlength=group_count,
        )
        requested_mean = raw_sum / finite_count
        group_bandwidth = np.asarray(bandwidth, dtype=float)[first_position]
        group_nu = np.asarray(nu, dtype=float)[first_position]
        group_upper_for_solve = np.asarray(upper_raw, dtype=float)[
            first_position
        ]
        mean_solution = solve_censored_mean_preserving_shift_batch(
            centers,
            center_mask,
            requested_mean,
            group_bandwidth,
            group_nu,
            group_upper_for_solve,
            mean_floor=float(config.truncation_zero_mean_epsilon),
            quadrature_order=int(config.truncation_quadrature_order),
        )
        scoring_centers = scoring_centers + mean_solution["shift"][
            group_code
        ]

    standardized_lower = -scoring_centers / bandwidth
    standardized_upper = (
        np.log1p(upper_raw) - scoring_centers
    ) / bandwidth
    lower_log_term = np.full(len(joined), float("-inf"), dtype=float)
    upper_log_term = np.full(len(joined), float("-inf"), dtype=float)
    continuous_log_term = np.full(
        len(joined), float("-inf"), dtype=float
    )
    lower_log_term[valid_draw] = student_t.logcdf(
        standardized_lower[valid_draw], df=nu[valid_draw]
    )
    upper_log_term[valid_draw] = student_t.logsf(
        standardized_upper[valid_draw], df=nu[valid_draw]
    )
    observed_for_density = np.where(
        np.isfinite(observed_raw),
        np.maximum(observed_raw, 0.0),
        0.0,
    )
    observed_z = np.log1p(observed_for_density)
    continuous_log_term[valid_draw] = _coherent_student_t_log_density_array(
        observed_z[valid_draw],
        scoring_centers[valid_draw],
        bandwidth[valid_draw],
        nu[valid_draw],
    )

    def group_logmeanexp(log_values: np.ndarray) -> np.ndarray:
        maxima = np.full(group_count, float("-inf"), dtype=float)
        np.maximum.at(
            maxima,
            group_code[valid_draw],
            np.asarray(log_values, dtype=float)[valid_draw],
        )
        shifted = np.zeros(len(joined), dtype=float)
        shifted[valid_draw] = np.exp(
            np.asarray(log_values, dtype=float)[valid_draw]
            - maxima[group_code[valid_draw]]
        )
        totals = np.bincount(
            group_code, weights=shifted, minlength=group_count
        )
        return maxima + np.log(totals) - np.log(finite_count)

    lower_log_probability = group_logmeanexp(lower_log_term)
    upper_log_probability = group_logmeanexp(upper_log_term)
    continuous_log_density = group_logmeanexp(continuous_log_term)
    group_observed = observed_raw[first_position]
    group_upper = upper_raw[first_position]
    at_lower = group_observed == 0.0
    at_upper = group_observed == group_upper
    interior = (group_observed > 0.0) & (group_observed < group_upper)
    log_score = np.full(group_count, float("-inf"), dtype=float)
    log_score[at_lower] = lower_log_probability[at_lower]
    log_score[at_upper] = upper_log_probability[at_upper]
    log_score[interior] = continuous_log_density[interior]

    grouped = joined.iloc[first_position][
        [*keys, "mode", "component", "horizon", "observed_mask", "event_weight"]
    ].reset_index(drop=True)
    grouped["log_score"] = log_score
    grouped.loc[
        ~grouped["observed_mask"].astype(bool), "log_score"
    ] = 0.0
    grouped["predictive_contract"] = config.predictive_contract
    grouped["truncation_density_kind"] = (
        "latent_student_t_kde_censoring_with_boundary_atoms"
    )
    grouped["truncation_lower_raw"] = 0.0
    grouped["truncation_upper_raw"] = group_upper
    grouped["truncation_upper_raw_frozen"] = group_upper
    grouped["truncation_upper_raw_effective"] = group_upper
    grouped["truncation_support_expanded"] = False
    grouped["truncation_support_expansion_policy"] = (
        STRICT_FROZEN_SUPPORT_POLICY
    )
    grouped["lower_boundary_atom_probability"] = np.exp(
        lower_log_probability
    )
    grouped["upper_boundary_atom_probability"] = np.exp(
        upper_log_probability
    )
    grouped["truncation_observation_in_support"] = (
        (group_observed >= 0.0) & (group_observed <= group_upper)
    )
    grouped["truncation_quadrature_order"] = int(
        config.truncation_quadrature_order
    )
    grouped["truncation_bound_policy"] = str(config.truncation_bound_policy)
    grouped["nll_measure_basis"] = (
        "log1p_transform_mixed_measure_atoms_at_zero_and_upper_bound"
    )
    grouped["boundary_atom_source"] = "latent_student_t_censoring"
    if mean_solution is not None:
        grouped["constrained_target_raw_mean"] = mean_solution[
            "requested_mean"
        ]
        grouped["effective_target_raw_mean"] = mean_solution[
            "effective_target"
        ]
        grouped["mean_floor_applied"] = mean_solution[
            "mean_floor_applied"
        ]
        grouped["censored_location_shift"] = mean_solution["shift"]
        grouped["censored_raw_mean"] = mean_solution["mean"]
        grouped["censored_raw_second_moment"] = mean_solution[
            "second_moment"
        ]
        grouped["censored_raw_variance"] = mean_solution["variance"]
        grouped["censored_mean_constraint_residual"] = mean_solution[
            "residual"
        ]
        grouped["censored_mean_solver_iterations"] = mean_solution[
            "iterations"
        ]
        grouped["truncation_mean_floor"] = float(
            config.truncation_zero_mean_epsilon
        )
    return grouped


def score_draw_rows(ledger_rows: pd.DataFrame, draws: pd.DataFrame, config: BridgeConfig, *, bandwidth_by_component: dict[str, float] | None = None) -> pd.DataFrame:
    required_ledger = {"forecast_id", "observed_value", "observed_mask", "component", "horizon"}
    required_draws = {"forecast_id", "model_id", "particle_id", "draw"}
    if missing := sorted(required_ledger - set(ledger_rows.columns)):
        raise ValueError(f"ledger missing columns {missing}")
    if missing := sorted(required_draws - set(draws.columns)):
        raise ValueError(f"draws missing columns {missing}")
    meta_columns = ["forecast_id", "observed_value", "observed_mask", "component", "horizon"]
    if "mode" in ledger_rows.columns:
        meta_columns.append("mode")
    meta = ledger_rows[meta_columns].copy()
    joined = draws.merge(meta, on="forecast_id", how="inner", suffixes=("_draw", "_ledger"))
    if "component_draw" in joined.columns and "component_ledger" in joined.columns:
        bad = joined["component_draw"].astype(str) != joined["component_ledger"].astype(str)
        if bad.any():
            raise ValueError(f"draws/ledger component mismatch for {int(bad.sum())} scored rows")
        joined["component"] = joined["component_ledger"]
    if "horizon_draw" in joined.columns and "horizon_ledger" in joined.columns:
        bad = pd.to_numeric(joined["horizon_draw"], errors="coerce").astype("Int64") != pd.to_numeric(joined["horizon_ledger"], errors="coerce").astype("Int64")
        if bad.any():
            raise ValueError(f"draws/ledger horizon mismatch for {int(bad.sum())} scored rows")
        joined["horizon"] = joined["horizon_ledger"]
    if "mode_draw" in joined.columns and "mode_ledger" in joined.columns:
        bad = joined["mode_draw"].astype(str) != joined["mode_ledger"].astype(str)
        if bad.any():
            raise ValueError(f"draws/ledger mode mismatch for {int(bad.sum())} scored rows")
        joined["mode"] = joined["mode_ledger"]
    elif "mode_ledger" in joined.columns:
        joined["mode"] = joined["mode_ledger"]
    elif "mode_draw" in joined.columns:
        joined["mode"] = joined["mode_draw"]
    elif "mode" not in joined.columns:
        joined["mode"] = ""
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "forecast_id", "model_id", "particle_id", "mode", "component",
                "horizon", "observed_mask", "log_score", "event_weight",
            ]
        )

                                                                            
                                                                             
                                                                    
    keys = ["forecast_id", "model_id", "particle_id"]
    joined["forecast_id"] = joined["forecast_id"].astype(str)
    joined["model_id"] = joined["model_id"].astype(str)
    joined["mode"] = joined["mode"].fillna("").astype(str)
    joined["component"] = joined["component"].astype(str)
    joined["horizon"] = pd.to_numeric(joined["horizon"], errors="raise").astype(int)
    joined["observed_mask"] = joined["observed_mask"].astype(str).str.strip().str.lower().isin(
        {"true", "1", "t", "yes", "y"}
    )

    mode = joined["mode"]
    component = joined["component"]
    horizon = joined["horizon"]
    horizon_weight = pd.Series(np.nan, index=joined.index, dtype=float)
    horizon_weight = horizon_weight.fillna(horizon.astype(str).map(config.horizon_weight))
    horizon_weight = horizon_weight.fillna(horizon.map(config.horizon_weight)).fillna(1.0)
    component_weight = _lookup_bridge_series(
        config.component_weight, mode, component, horizon, 1.0
    )
    joined["event_weight"] = (
        horizon_weight.to_numpy(dtype=float) * component_weight
    )

    if config.tau_by_component:
        base_bandwidth = _lookup_bridge_series(
            config.tau_by_component,
            mode,
            component,
            horizon,
            config.default_tau,
        )
    else:
                                                                 
        base_bandwidth = _lookup_bridge_series(
            config.sigma_by_component,
            mode,
            component,
            horizon,
            config.default_sigma,
        )
    bandwidth_by_component = bandwidth_by_component or {}
    if bandwidth_by_component:
        override = _lookup_bridge_series(
            bandwidth_by_component, mode, component, horizon, float("nan")
        )
        bandwidth = np.where(np.isfinite(override), override, base_bandwidth)
    else:
        bandwidth = base_bandwidth
    bandwidth = np.maximum(np.asarray(bandwidth, dtype=float), config.min_scale)
    bandwidth = np.maximum(bandwidth, 1e-6)

    observed_value = pd.to_numeric(
        joined["observed_value"], errors="raise"
    ).to_numpy(dtype=float)
    draw_value = pd.to_numeric(joined["draw"], errors="coerce").to_numpy(dtype=float)
    if config.transform == "log1p":
        x = np.log1p(np.maximum(observed_value, 0.0))
        transformed_draw = np.log1p(np.maximum(draw_value, 0.0))
    elif config.transform == "identity":
        x = observed_value
        transformed_draw = draw_value
    else:
        raise ValueError(f"unknown transform {config.transform!r}")

    if config.predictive_contract in COHERENT_CENSORED_CONTRACTS:
        if config.transform != "log1p":
            raise ValueError(
                "coherent censored Student-t requires log1p transform"
            )
        if float(config.truncation_lower_raw) != 0.0:
            raise ValueError(
                "coherent censored Student-t requires truncation_lower_raw=0"
            )
        kernel = str(
            getattr(config, "kernel_distribution", "gaussian")
        ).strip().lower()
        if kernel not in {"student_t", "student-t", "t"}:
            raise ValueError(
                "coherent censored draw contract requires "
                "kernel_distribution='student_t'"
            )
        nu = _lookup_bridge_series(
            config.nu_by_component,
            mode,
            component,
            horizon,
            config.nu if config.nu_by_component else config.kernel_nu,
        )
        upper_raw = _lookup_bridge_series(
            config.truncation_upper_raw_by_component,
            mode,
            component,
            horizon,
            config.default_truncation_upper_raw,
        )
        return _score_censored_draw_groups(
            joined,
            keys=keys,
            observed_raw=observed_value,
            transformed_draw=transformed_draw,
            bandwidth=bandwidth,
            nu=nu,
            upper_raw=upper_raw,
            config=config,
            mean_preserving=(
                config.predictive_contract
                == COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T
            ),
        )

    if config.predictive_contract == COHERENT_MEAN_PRESERVING_TRUNCATED_T:
        if config.transform != "log1p":
            raise ValueError(
                "coherent mean-preserving truncated Student-t requires log1p transform"
            )
        kernel = str(getattr(config, "kernel_distribution", "gaussian")).strip().lower()
        if kernel not in {"student_t", "student-t", "t"}:
            raise ValueError(
                "coherent draw contract requires kernel_distribution='student_t'"
            )
        nu = _lookup_bridge_series(
            config.nu_by_component,
            mode,
            component,
            horizon,
            config.nu if config.nu_by_component else config.kernel_nu,
        )
        upper_raw = _lookup_bridge_series(
            config.truncation_upper_raw_by_component,
            mode,
            component,
            horizon,
            config.default_truncation_upper_raw,
        )
        lower_raw = np.full(len(joined), float(config.truncation_lower_raw))
        coherent_observed = joined["observed_mask"].to_numpy(dtype=bool)
        coherent_observed_raw = np.where(
            coherent_observed, np.maximum(observed_value, 0.0), lower_raw
        )
        coherent_observed_z = np.where(
            coherent_observed, x, np.log1p(lower_raw)
        )
        return _score_coherent_draw_groups(
            joined,
            keys=keys,
            observed_z=coherent_observed_z,
            observed_raw=coherent_observed_raw,
            transformed_draw=transformed_draw,
            draw_raw=draw_value,
            bandwidth=bandwidth,
            nu=nu,
            lower_raw=lower_raw,
            upper_raw=upper_raw,
            config=config,
        )

    valid_draw = np.isfinite(transformed_draw)
    residual = np.zeros(len(joined), dtype=float)
    residual[valid_draw] = (
        x[valid_draw] - transformed_draw[valid_draw]
    ) / bandwidth[valid_draw]
    kernel = str(getattr(config, "kernel_distribution", "gaussian")).strip().lower()
    log_kernel = np.full(len(joined), float("-inf"), dtype=float)
    if kernel == "gaussian":
        log_kernel[valid_draw] = (
            -0.5 * np.square(residual[valid_draw])
            - np.log(bandwidth[valid_draw])
            - 0.5 * log(2.0 * pi)
        )
    elif kernel in {"student_t", "student-t", "t"}:
        nu = _lookup_bridge_series(
            config.nu_by_component,
            mode,
            component,
            horizon,
            config.nu if config.nu_by_component else config.kernel_nu,
        )
        if np.any((nu <= 0.0) | np.isnan(nu)):
            raise ValueError("Student-t kernel nu must be positive or infinity")
        gaussian_limit = valid_draw & np.isinf(nu)
        log_kernel[gaussian_limit] = (
            -0.5 * np.square(residual[gaussian_limit])
            - np.log(bandwidth[gaussian_limit])
            - 0.5 * log(2.0 * pi)
        )
        finite_nu = valid_draw & ~np.isinf(nu)
        nu_f = nu[finite_nu]
        log_kernel[finite_nu] = (
            _gammaln_array((nu_f + 1.0) / 2.0)
            - _gammaln_array(nu_f / 2.0)
            - 0.5 * np.log(nu_f * pi)
            - np.log(bandwidth[finite_nu])
            - ((nu_f + 1.0) / 2.0)
            * np.log1p(np.square(residual[finite_nu]) / nu_f)
        )
    else:
        raise ValueError(f"unknown kernel_distribution {config.kernel_distribution!r}")

    kernel_rows = joined[keys].copy()
    kernel_rows["log_kernel"] = log_kernel
    finite_rows = kernel_rows[np.isfinite(kernel_rows["log_kernel"])].copy()
    if finite_rows.empty:
        density = pd.DataFrame(columns=[*keys, "log_score"])
    else:
        finite_rows["group_max"] = finite_rows.groupby(keys, sort=False)[
            "log_kernel"
        ].transform("max")
        finite_rows["shifted_density"] = np.exp(
            finite_rows["log_kernel"] - finite_rows["group_max"]
        )
        density = finite_rows.groupby(keys, sort=True, as_index=False).agg(
            group_max=("group_max", "first"),
            density_sum=("shifted_density", "sum"),
            finite_draw_count=("shifted_density", "size"),
        )
        density["log_score"] = (
            density["group_max"]
            + np.log(density["density_sum"])
            - np.log(density["finite_draw_count"].astype(float))
        )
        density = density[[*keys, "log_score"]]

    grouped = joined.groupby(keys, sort=True, as_index=False).first()[
        [*keys, "mode", "component", "horizon", "observed_mask", "event_weight"]
    ]
    grouped = grouped.merge(density, on=keys, how="left", validate="one_to_one")
    grouped["log_score"] = grouped["log_score"].fillna(float("-inf"))
    grouped.loc[~grouped["observed_mask"].astype(bool), "log_score"] = 0.0
    return grouped[
        [
            "forecast_id", "model_id", "particle_id", "mode", "component",
            "horizon", "observed_mask", "log_score", "event_weight",
        ]
    ]

def tune_temperature_grid(previous_weights: pd.DataFrame, log_evidence: pd.DataFrame, *, grid: list[float] | None = None, target_ess_fraction: float = 0.5) -> float:
    from caster.filter.outer_update import update_outer_weights, summarize_model_distribution
    grid = grid or [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    n = max(len(previous_weights), 1)
    target = max(1.0, min(float(target_ess_fraction) * n, n))
    best = float(grid[0]); best_gap = float('inf')
    for rho in grid:
        updated = update_outer_weights(previous_weights, log_evidence, rho=float(rho))
        ess = float(summarize_model_distribution(updated)["model_ess"])
        gap = abs(ess - target)
        if gap < best_gap:
            best_gap = gap; best = float(rho)
    return best
