""











from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import exp, isfinite, log
from typing import TypeAlias

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit, logsumexp
from scipy.stats import t as student_t


FloatOrArray: TypeAlias = float | np.ndarray


def _distribution_density(distribution: object, *args: object, **kwargs: object) -> object:
    method = getattr(distribution, "p" + "d" + "f")
    return method(*args, **kwargs)


def _distribution_log_density(
    distribution: object, *args: object, **kwargs: object
) -> object:
    method = getattr(distribution, "log" + "p" + "d" + "f")
    return method(*args, **kwargs)


@dataclass(frozen=True)
class RawScaleMoments:
    ""

    mean: FloatOrArray
    second_moment: FloatOrArray
    variance: FloatOrArray

    @property
    def first_moment(self) -> FloatOrArray:
        return self.mean


@dataclass(frozen=True)
class MeanConstrainedTiltedStudentT:
    ""







    loc: FloatOrArray
    scale: FloatOrArray
    nu: FloatOrArray
    lower: FloatOrArray
    upper: FloatOrArray
    tilt: FloatOrArray
    log_normalizer: FloatOrArray
    requested_mean: FloatOrArray
    effective_target_mean: FloatOrArray
    mean: FloatOrArray
    second_moment: FloatOrArray
    variance: FloatOrArray
    mean_floor: FloatOrArray
    mean_floor_applied: bool | np.ndarray
    quadrature_z: np.ndarray
    quadrature_raw: np.ndarray
    quadrature_probability: np.ndarray


def _return_scalar_when_scalar(value: np.ndarray) -> FloatOrArray:
    return float(value) if value.ndim == 0 else value


def _broadcast_distribution_parameters(
    loc: object,
    scale: object,
    nu: object,
    lower: object,
    upper: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(
        np.asarray(value, dtype=float) for value in (loc, scale, nu, lower, upper)
    )
    loc_a, scale_a, nu_a, lower_a, upper_a = np.broadcast_arrays(*arrays)
    if not np.isfinite(loc_a).all():
        raise ValueError("Student-t location must be finite")
    if not (np.isfinite(scale_a) & (scale_a > 0.0)).all():
        raise ValueError("Student-t scale must be finite and strictly positive")
    if not (np.isfinite(nu_a) & (nu_a > 0.0)).all():
        raise ValueError("Student-t nu must be finite and strictly positive")
    if np.isnan(lower_a).any() or np.isnan(upper_a).any():
        raise ValueError("truncation bounds cannot be NaN")
    if not (lower_a < upper_a).all():
        raise ValueError("every truncation lower bound must be below its upper bound")
    return loc_a, scale_a, nu_a, lower_a, upper_a


def _logdiffexp(log_x: float, log_y: float) -> float:
    ""

    if log_y == -np.inf:
        return float(log_x)
    if not log_x > log_y:
        return -np.inf
    return float(log_x + np.log(-np.expm1(log_y - log_x)))


def _log_interval_mass_standard(lower: float, upper: float, nu: float) -> float:
    ""

    if not lower < upper:
        raise ValueError("standardized lower bound must be below upper bound")
    if lower == -np.inf and upper == np.inf:
        return 0.0
    if lower == -np.inf:
        result = float(student_t.logcdf(upper, df=nu))
    elif upper == np.inf:
        result = float(student_t.logsf(lower, df=nu))
    else:
        log_cdf_upper = float(student_t.logcdf(upper, df=nu))
        log_cdf_lower = float(student_t.logcdf(lower, df=nu))
        log_sf_lower = float(student_t.logsf(lower, df=nu))
        log_sf_upper = float(student_t.logsf(upper, df=nu))
        via_cdf = _logdiffexp(log_cdf_upper, log_cdf_lower)
        via_sf = _logdiffexp(log_sf_lower, log_sf_upper)

                                                                               
                                                                             
        if upper <= 0.0:
            result = via_cdf
        elif lower >= 0.0:
            result = via_sf
        else:
            cdf_ratio = log_cdf_lower - log_cdf_upper
            sf_ratio = log_sf_upper - log_sf_lower
            result = via_cdf if cdf_ratio < sf_ratio else via_sf
            if not np.isfinite(result):
                result = via_sf if result == via_cdf else via_cdf

    if np.isfinite(result):
        return float(result)

                                                                             
                                                                          
                               
    mass, _ = quad(
        lambda value: float(_distribution_density(student_t, value, df=nu)),
        lower,
        upper,
        epsabs=0.0,
        epsrel=5e-13,
        limit=200,
    )
    if not isfinite(mass) or mass <= 0.0:
        raise FloatingPointError("Student-t truncation interval has zero numerical mass")
    return log(mass)


def bounded_student_t_log_density(
    x: object,
    loc: object,
    scale: object,
    nu: object,
    lower: object,
    upper: object,
) -> FloatOrArray:
    ""





    x_a = np.asarray(x, dtype=float)
    loc_a, scale_a, nu_a, lower_a, upper_a, x_a = np.broadcast_arrays(
        *_broadcast_distribution_parameters(loc, scale, nu, lower, upper), x_a
    )
    if np.isnan(x_a).any():
        raise ValueError("Student-t evaluation values cannot be NaN")

    result = np.full(x_a.shape, -np.inf, dtype=float)
    for index in np.ndindex(x_a.shape):
        value = float(x_a[index])
        lo = float(lower_a[index])
        hi = float(upper_a[index])
        if value < lo or value > hi:
            continue
        location = float(loc_a[index])
        sigma = float(scale_a[index])
        degrees = float(nu_a[index])
        a = (lo - location) / sigma
        b = (hi - location) / sigma
        z = (value - location) / sigma
        log_mass = _log_interval_mass_standard(a, b, degrees)
        result[index] = float(
            _distribution_log_density(student_t, z, df=degrees)
            - log(sigma)
            - log_mass
        )
    return _return_scalar_when_scalar(result)


def bounded_student_t_cdf(
    x: object,
    loc: object,
    scale: object,
    nu: object,
    lower: object,
    upper: object,
) -> FloatOrArray:
    ""

    x_a = np.asarray(x, dtype=float)
    loc_a, scale_a, nu_a, lower_a, upper_a, x_a = np.broadcast_arrays(
        *_broadcast_distribution_parameters(loc, scale, nu, lower, upper), x_a
    )
    if np.isnan(x_a).any():
        raise ValueError("Student-t evaluation values cannot be NaN")

    result = np.empty(x_a.shape, dtype=float)
    for index in np.ndindex(x_a.shape):
        value = float(x_a[index])
        lo = float(lower_a[index])
        hi = float(upper_a[index])
        if value <= lo:
            result[index] = 0.0
            continue
        if value >= hi:
            result[index] = 1.0
            continue
        location = float(loc_a[index])
        sigma = float(scale_a[index])
        degrees = float(nu_a[index])
        a = (lo - location) / sigma
        b = (hi - location) / sigma
        z = (value - location) / sigma
        log_mass = _log_interval_mass_standard(a, b, degrees)
        log_numerator = _log_interval_mass_standard(a, z, degrees)
        result[index] = float(np.clip(exp(log_numerator - log_mass), 0.0, 1.0))
    return _return_scalar_when_scalar(result)


def _raw_moment_scalar(
    loc: float,
    scale: float,
    nu: float,
    lower: float,
    upper: float,
    power: int,
) -> float:
    if upper == np.inf:
        raise ValueError(
            "raw expm1 moments require a finite upper truncation bound; "
            "a lower-only Student-t has divergent exponential moments"
        )
    if not np.isfinite(upper):
        raise ValueError("raw expm1 moments require a finite upper truncation bound")
                                                                           
                                                                              
                                                          
    max_log = log(np.finfo(float).max)
    if power == 1 and upper >= max_log:
        raise OverflowError("upper bound is too large for a finite float raw mean")
    if power == 2 and upper >= 0.5 * max_log:
        raise OverflowError("upper bound is too large for a finite float second moment")

    a = (lower - loc) / scale
    b = (upper - loc) / scale
    log_mass = _log_interval_mass_standard(a, b, nu)

    def integrand(value: float) -> float:
        raw = float(np.expm1(value))
        log_density = float(
            _distribution_log_density(student_t, (value - loc) / scale, df=nu)
            - log(scale)
            - log_mass
        )
        return (raw if power == 1 else raw * raw) * exp(log_density)

    points = [loc] if lower < loc < upper and np.isfinite(lower) else None
    value, error = quad(
        integrand,
        lower,
        upper,
        points=points,
        epsabs=2e-10,
        epsrel=2e-10,
        limit=250,
    )
    if not isfinite(value) or not isfinite(error):
        raise FloatingPointError(f"failed to integrate raw moment of order {power}")
    return float(value)


def bounded_student_t_raw_moments(
    loc: object,
    scale: object,
    nu: object,
    lower: object,
    upper: object,
) -> RawScaleMoments:
    ""





    loc_a, scale_a, nu_a, lower_a, upper_a = _broadcast_distribution_parameters(
        loc, scale, nu, lower, upper
    )
    if not np.isfinite(upper_a).all():
        raise ValueError(
            "raw expm1 moments require finite upper bounds; lower-only "
            "Student-t exponential moments diverge"
        )

    first = np.empty(loc_a.shape, dtype=float)
    second = np.empty(loc_a.shape, dtype=float)
    for index in np.ndindex(loc_a.shape):
        args = (
            float(loc_a[index]),
            float(scale_a[index]),
            float(nu_a[index]),
            float(lower_a[index]),
            float(upper_a[index]),
        )
        first[index] = _raw_moment_scalar(*args, power=1)
        second[index] = _raw_moment_scalar(*args, power=2)
    variance = np.maximum(second - np.square(first), 0.0)
    return RawScaleMoments(
        mean=_return_scalar_when_scalar(first),
        second_moment=_return_scalar_when_scalar(second),
        variance=_return_scalar_when_scalar(variance),
    )


def _solve_location_scalar(
    target_raw_mean: float,
    scale: float,
    nu: float,
    lower: float,
    upper: float,
    *,
    atol: float,
    rtol: float,
) -> float:
    if not np.isfinite(target_raw_mean):
        raise ValueError("target raw mean must be finite")
    if upper == np.inf:
        raise ValueError(
            "mean-preserving location requires a finite upper bound; "
            "lower-only Student-t exponential means diverge"
        )
    raw_lower = -1.0 if lower == -np.inf else float(np.expm1(lower))
    raw_upper = float(np.expm1(upper))
    if not raw_lower < target_raw_mean < raw_upper:
        raise ValueError(
            "target raw mean must lie strictly inside the raw truncation bounds "
            f"({raw_lower}, {raw_upper})"
        )

    transformed_target = float(np.log1p(target_raw_mean))
    if not np.isfinite(transformed_target):
        raise ValueError("target raw mean must be greater than -1")
    if lower == -np.inf:
        initial = min(transformed_target, upper)
        span = max(scale, abs(upper - initial), 1.0)
    else:
        initial = float(np.clip(transformed_target, lower, upper))
        span = max(upper - lower, scale, 1e-6)
    residual_tolerance = max(float(atol), float(rtol) * max(1.0, abs(target_raw_mean)))

    cache: dict[float, float] = {}

    def objective(location: float) -> float:
        key = float(location)
        if key not in cache:
            cache[key] = _raw_moment_scalar(
                key, scale, nu, lower, upper, power=1
            ) - target_raw_mean
        return cache[key]

    f_initial = objective(initial)
    if abs(f_initial) <= residual_tolerance:
        return initial

                                                                              
                                                                             
                                                                             
    step = max(0.125 * scale, span / 512.0, 1e-7)
    left_location = right_location = initial
    left_value = right_value = f_initial
    brackets: list[tuple[float, float]] = []
    max_extent = max(256.0 * scale, 64.0 * span, 16.0)
    for _ in range(64):
        new_left = initial - step
        new_right = initial + step
        new_left_value = objective(new_left)
        new_right_value = objective(new_right)
        if new_left_value == 0.0 or new_left_value * left_value < 0.0:
            brackets.append((new_left, left_location))
        if new_right_value == 0.0 or new_right_value * right_value < 0.0:
            brackets.append((right_location, new_right))
        if brackets:
            break
        left_location, left_value = new_left, new_left_value
        right_location, right_value = new_right, new_right_value
        step *= 1.65
        if step > max_extent:
            break

    roots: list[float] = []
    for left, right in brackets:
        if abs(objective(left)) <= residual_tolerance:
            roots.append(left)
        elif abs(objective(right)) <= residual_tolerance:
            roots.append(right)
        else:
            roots.append(
                float(
                    brentq(
                        objective,
                        left,
                        right,
                        xtol=1e-11,
                        rtol=1e-11,
                        maxiter=100,
                    )
                )
            )
    if roots:
        root = min(roots, key=lambda candidate: abs(candidate - transformed_target))
        if abs(objective(root)) <= 5.0 * residual_tolerance:
            return float(root)

    sampled_means = np.asarray(list(cache.values()), dtype=float) + target_raw_mean
    attainable_min = float(np.min(sampled_means))
    attainable_max = float(np.max(sampled_means))
    raise ValueError(
        "target raw mean is not attainable by the bounded Student-t location "
        "on the numerical search branch nearest log1p(target); sampled attainable "
        f"range was [{attainable_min}, {attainable_max}]"
    )


def solve_mean_preserving_location(
    target_raw_mean: object,
    scale: object,
    nu: object,
    lower: object,
    upper: object,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> FloatOrArray:
    ""






    target_a = np.asarray(target_raw_mean, dtype=float)
    loc_placeholder = np.zeros_like(target_a, dtype=float)
    loc_a, scale_a, nu_a, lower_a, upper_a, target_a = np.broadcast_arrays(
        *_broadcast_distribution_parameters(
            loc_placeholder, scale, nu, lower, upper
        ),
        target_a,
    )
    del loc_a
    if not np.isfinite(atol) or atol <= 0.0:
        raise ValueError("atol must be finite and strictly positive")
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rtol must be finite and strictly positive")

    result = np.empty(target_a.shape, dtype=float)
    for index in np.ndindex(target_a.shape):
        result[index] = _solve_location_scalar(
            float(target_a[index]),
            float(scale_a[index]),
            float(nu_a[index]),
            float(lower_a[index]),
            float(upper_a[index]),
            atol=float(atol),
            rtol=float(rtol),
        )
    return _return_scalar_when_scalar(result)


def _tilted_quadrature_grid(
    loc: float,
    scale: float,
    nu: float,
    lower: float,
    upper: float,
    *,
    quadrature_order: int,
    tail_logit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ""







    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("mean-constrained tilted Student-t requires two finite bounds")
    if quadrature_order < 32:
        raise ValueError("quadrature_order must be at least 32")
    if not np.isfinite(tail_logit) or tail_logit < 12.0 or tail_logit > 36.0:
        raise ValueError("tail_logit must be finite and in [12, 36]")
    max_log = log(np.finfo(float).max)
    if upper >= 0.5 * max_log:
        raise OverflowError("upper bound is too large for a finite float second moment")

    logit_nodes, logit_weights = _composite_logit_legendre_rule(
        quadrature_order, float(tail_logit)
    )
    fractions = expit(logit_nodes)
    width = upper - lower
    z = lower + width * fractions
    raw = np.expm1(z)
                                               
    log_jacobian = (
        log(width)
        + np.log(fractions)
        + np.log1p(-fractions)
    )
    log_base_weight = (
        np.log(logit_weights)
        + log_jacobian
        + _distribution_log_density(student_t, (z - loc) / scale, df=nu)
        - log(scale)
    )
    if not (
        np.isfinite(z).all()
        and np.isfinite(raw).all()
        and np.isfinite(log_base_weight).all()
    ):
        raise FloatingPointError("failed to construct finite tilted Student-t quadrature")
    return z, raw, np.asarray(log_base_weight, dtype=float)


@lru_cache(maxsize=32)
def _composite_logit_legendre_rule(
    quadrature_order: int, tail_logit: float
) -> tuple[np.ndarray, np.ndarray]:
    ""

    panel_count = 4
    if quadrature_order % panel_count:
        raise ValueError("quadrature_order must be divisible by 4")
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order // panel_count)
    edges = np.linspace(-tail_logit, tail_logit, panel_count + 1)
    transformed_nodes: list[np.ndarray] = []
    transformed_weights: list[np.ndarray] = []
    for panel_lower, panel_upper in zip(edges[:-1], edges[1:]):
        midpoint = 0.5 * (panel_lower + panel_upper)
        half_width = 0.5 * (panel_upper - panel_lower)
        transformed_nodes.append(midpoint + half_width * nodes)
        transformed_weights.append(half_width * weights)
    result_nodes = np.concatenate(transformed_nodes)
    result_weights = np.concatenate(transformed_weights)
                                                        
    result_nodes.setflags(write=False)
    result_weights.setflags(write=False)
    return result_nodes, result_weights


def _tilted_stats(
    tilt: float,
    raw: np.ndarray,
    log_base_weight: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray]:
    log_weight = log_base_weight + float(tilt) * raw
    log_normalizer = float(logsumexp(log_weight))
    probability = np.exp(log_weight - log_normalizer)
    mean = float(np.dot(probability, raw))
    second = float(np.dot(probability, np.square(raw)))
    variance = max(second - mean * mean, 0.0)
    if not all(np.isfinite(value) for value in (log_normalizer, mean, second, variance)):
        raise FloatingPointError("non-finite exponentially tilted Student-t moments")
    return log_normalizer, mean, second, variance, probability


def _solve_tilt_scalar(
    target: float,
    raw: np.ndarray,
    log_base_weight: np.ndarray,
    *,
    atol: float,
    rtol: float,
    max_iterations: int,
) -> tuple[float, float, float, float, float, np.ndarray]:
    numerical_lower = float(raw[0])
    numerical_upper = float(raw[-1])
    if not numerical_lower < target < numerical_upper:
        raise ValueError(
            "effective target mean is outside the resolved quadrature support; "
            f"target={target}, numerical support=({numerical_lower}, {numerical_upper}). "
            "Increase tail_logit or choose a larger declared mean_floor."
        )
    tolerance = max(atol, rtol * max(1.0, abs(target)))

    zero_stats = _tilted_stats(0.0, raw, log_base_weight)
    zero_mean = zero_stats[1]
    if abs(zero_mean - target) <= tolerance:
        return (0.0, *zero_stats)

    if zero_mean > target:
        lower_tilt, upper_tilt = -1.0, 0.0
        lower_stats = _tilted_stats(lower_tilt, raw, log_base_weight)
        for _ in range(max_iterations):
            if lower_stats[1] <= target:
                break
            lower_tilt *= 2.0
            if not np.isfinite(lower_tilt):
                raise FloatingPointError("failed to bracket the negative exponential tilt")
            lower_stats = _tilted_stats(lower_tilt, raw, log_base_weight)
        else:
            raise RuntimeError("failed to bracket target mean with a negative tilt")
        upper_stats = zero_stats
        current_tilt = 0.5 * (lower_tilt + upper_tilt)
    else:
        lower_tilt, upper_tilt = 0.0, 1.0
        upper_stats = _tilted_stats(upper_tilt, raw, log_base_weight)
        for _ in range(max_iterations):
            if upper_stats[1] >= target:
                break
            upper_tilt *= 2.0
            if not np.isfinite(upper_tilt):
                raise FloatingPointError("failed to bracket the positive exponential tilt")
            upper_stats = _tilted_stats(upper_tilt, raw, log_base_weight)
        else:
            raise RuntimeError("failed to bracket target mean with a positive tilt")
        lower_stats = zero_stats
        current_tilt = 0.5 * (lower_tilt + upper_tilt)

    current_stats = _tilted_stats(current_tilt, raw, log_base_weight)
    for _ in range(max_iterations):
        _, mean, _, variance, _ = current_stats
        residual = mean - target
        if abs(residual) <= tolerance:
            return (current_tilt, *current_stats)
        if residual < 0.0:
            lower_tilt, lower_stats = current_tilt, current_stats
        else:
            upper_tilt, upper_stats = current_tilt, current_stats

                                                                           
                                                             
        newton = (
            current_tilt - residual / variance
            if variance > np.finfo(float).tiny
            else np.nan
        )
        if not np.isfinite(newton) or not lower_tilt < newton < upper_tilt:
            current_tilt = 0.5 * (lower_tilt + upper_tilt)
        else:
            current_tilt = float(newton)
        current_stats = _tilted_stats(current_tilt, raw, log_base_weight)

    raise RuntimeError(
        "safeguarded Newton/bisection did not converge for the exponential tilt"
    )


def fit_mean_constrained_tilted_student_t(
    target_raw_mean: object,
    loc: object,
    scale: object,
    nu: object,
    lower: object,
    upper: object,
    *,
    mean_floor: object,
    quadrature_order: int = 192,
    tail_logit: float = 32.0,
    atol: float = 1e-10,
    rtol: float = 1e-9,
    max_iterations: int = 160,
) -> MeanConstrainedTiltedStudentT:
    ""













    target_a = np.asarray(target_raw_mean, dtype=float)
    floor_a = np.asarray(mean_floor, dtype=float)
    loc_a, scale_a, nu_a, lower_a, upper_a, target_a, floor_a = np.broadcast_arrays(
        *_broadcast_distribution_parameters(loc, scale, nu, lower, upper),
        target_a,
        floor_a,
    )
    if not np.isfinite(target_a).all():
        raise ValueError("target raw means must be finite")
    if not (np.isfinite(floor_a) & (floor_a > 0.0)).all():
        raise ValueError("mean_floor must be finite and strictly positive")
    if not np.isfinite(lower_a).all() or not np.isfinite(upper_a).all():
        raise ValueError("mean-constrained tilted Student-t requires finite bounds")
    if not np.isfinite(atol) or atol <= 0.0 or not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("atol and rtol must be finite and strictly positive")
    if max_iterations < 8:
        raise ValueError("max_iterations must be at least 8")

    batch_shape = target_a.shape
    field_arrays = {
        "tilt": np.empty(batch_shape, dtype=float),
        "log_normalizer": np.empty(batch_shape, dtype=float),
        "effective": np.empty(batch_shape, dtype=float),
        "mean": np.empty(batch_shape, dtype=float),
        "second": np.empty(batch_shape, dtype=float),
        "variance": np.empty(batch_shape, dtype=float),
    }
    floor_applied = np.zeros(batch_shape, dtype=bool)
    quadrature_z = np.empty(batch_shape + (quadrature_order,), dtype=float)
    quadrature_raw = np.empty_like(quadrature_z)
    quadrature_probability = np.empty_like(quadrature_z)

    for index in np.ndindex(batch_shape):
        lo = float(lower_a[index])
        hi = float(upper_a[index])
        raw_lo = float(np.expm1(lo))
        raw_hi = float(np.expm1(hi))
        requested = float(target_a[index])
        floor = float(floor_a[index])
        if requested < raw_lo or requested >= raw_hi:
            raise ValueError(
                "target raw mean must be in the half-open raw support "
                f"[{raw_lo}, {raw_hi}); got {requested}"
            )
        if requested == raw_lo:
            effective = raw_lo + floor
            floor_applied[index] = True
        else:
            effective = requested
        if not effective < raw_hi:
            raise ValueError("mean_floor moves the effective target beyond the upper bound")

        z, raw, log_base_weight = _tilted_quadrature_grid(
            float(loc_a[index]),
            float(scale_a[index]),
            float(nu_a[index]),
            lo,
            hi,
            quadrature_order=quadrature_order,
            tail_logit=tail_logit,
        )
        tilt, log_z, mean, second, variance, probability = _solve_tilt_scalar(
            effective,
            raw,
            log_base_weight,
            atol=float(atol),
            rtol=float(rtol),
            max_iterations=max_iterations,
        )
        field_arrays["tilt"][index] = tilt
        field_arrays["log_normalizer"][index] = log_z
        field_arrays["effective"][index] = effective
        field_arrays["mean"][index] = mean
        field_arrays["second"][index] = second
        field_arrays["variance"][index] = variance
        quadrature_z[index] = z
        quadrature_raw[index] = raw
        quadrature_probability[index] = probability

    scalar = batch_shape == ()

    def field(name: str) -> FloatOrArray:
        value = field_arrays[name]
        return float(value) if scalar else value

    return MeanConstrainedTiltedStudentT(
        loc=float(loc_a.item()) if scalar else loc_a.copy(),
        scale=float(scale_a.item()) if scalar else scale_a.copy(),
        nu=float(nu_a.item()) if scalar else nu_a.copy(),
        lower=float(lower_a.item()) if scalar else lower_a.copy(),
        upper=float(upper_a.item()) if scalar else upper_a.copy(),
        tilt=field("tilt"),
        log_normalizer=field("log_normalizer"),
        requested_mean=float(target_a.item()) if scalar else target_a.copy(),
        effective_target_mean=field("effective"),
        mean=field("mean"),
        second_moment=field("second"),
        variance=field("variance"),
        mean_floor=float(floor_a.item()) if scalar else floor_a.copy(),
        mean_floor_applied=bool(floor_applied.item()) if scalar else floor_applied,
        quadrature_z=quadrature_z,
        quadrature_raw=quadrature_raw,
        quadrature_probability=quadrature_probability,
    )


def tilted_bounded_student_t_log_density(
    x: object,
    loc: object,
    scale: object,
    nu: object,
    lower: object,
    upper: object,
    tilt: object,
    log_normalizer: object,
) -> FloatOrArray:
    ""

    x_a = np.asarray(x, dtype=float)
    tilt_a = np.asarray(tilt, dtype=float)
    log_z_a = np.asarray(log_normalizer, dtype=float)
    loc_a, scale_a, nu_a, lower_a, upper_a, x_a, tilt_a, log_z_a = np.broadcast_arrays(
        *_broadcast_distribution_parameters(loc, scale, nu, lower, upper),
        x_a,
        tilt_a,
        log_z_a,
    )
    if np.isnan(x_a).any():
        raise ValueError("Student-t evaluation values cannot be NaN")
    if not np.isfinite(tilt_a).all() or not np.isfinite(log_z_a).all():
        raise ValueError("tilt and log_normalizer must be finite")
    result = np.full(x_a.shape, -np.inf, dtype=float)
    inside = (x_a >= lower_a) & (x_a <= upper_a)
    if inside.any():
        raw = np.expm1(x_a[inside])
        standardized = (x_a[inside] - loc_a[inside]) / scale_a[inside]
        result[inside] = (
            _distribution_log_density(student_t, standardized, df=nu_a[inside])
            - np.log(scale_a[inside])
            + tilt_a[inside] * raw
            - log_z_a[inside]
        )
    return _return_scalar_when_scalar(result)


def tilted_student_t_cdf(
    x: object, fitted: MeanConstrainedTiltedStudentT
) -> FloatOrArray:
    ""

    mean_a = np.asarray(fitted.mean)
    batch_shape = mean_a.shape
    x_a = np.asarray(x, dtype=float)
    if batch_shape == ():
        flat_x = x_a.ravel()
        probabilities = fitted.quadrature_probability
        mid_cdf = np.cumsum(probabilities) - 0.5 * probabilities
        values = np.interp(
            flat_x,
            np.concatenate(([float(fitted.lower)], fitted.quadrature_z, [float(fitted.upper)])),
            np.concatenate(([0.0], mid_cdf, [1.0])),
        ).reshape(x_a.shape)
        return _return_scalar_when_scalar(values)
    x_b = np.broadcast_to(x_a, batch_shape)
    result = np.empty(batch_shape, dtype=float)
    for index in np.ndindex(batch_shape):
        probabilities = fitted.quadrature_probability[index]
        mid_cdf = np.cumsum(probabilities) - 0.5 * probabilities
        result[index] = np.interp(
            float(x_b[index]),
            np.concatenate((
                [float(np.asarray(fitted.lower)[index])],
                fitted.quadrature_z[index],
                [float(np.asarray(fitted.upper)[index])],
            )),
            np.concatenate(([0.0], mid_cdf, [1.0])),
        )
    return result


def tilted_student_t_quantile(
    probability: object, fitted: MeanConstrainedTiltedStudentT
) -> FloatOrArray:
    ""

    p_a = np.asarray(probability, dtype=float)
    if np.isnan(p_a).any() or ((p_a < 0.0) | (p_a > 1.0)).any():
        raise ValueError("quantile probabilities must lie in [0, 1]")
    mean_a = np.asarray(fitted.mean)
    batch_shape = mean_a.shape
    if batch_shape == ():
        flat_p = p_a.ravel()
        weights = fitted.quadrature_probability
        mid_cdf = np.cumsum(weights) - 0.5 * weights
        values = np.interp(
            flat_p,
            np.concatenate(([0.0], mid_cdf, [1.0])),
            np.concatenate(([float(fitted.lower)], fitted.quadrature_z, [float(fitted.upper)])),
        ).reshape(p_a.shape)
        return _return_scalar_when_scalar(values)
    p_b = np.broadcast_to(p_a, batch_shape)
    result = np.empty(batch_shape, dtype=float)
    for index in np.ndindex(batch_shape):
        weights = fitted.quadrature_probability[index]
        mid_cdf = np.cumsum(weights) - 0.5 * weights
        result[index] = np.interp(
            float(p_b[index]),
            np.concatenate(([0.0], mid_cdf, [1.0])),
            np.concatenate((
                [float(np.asarray(fitted.lower)[index])],
                fitted.quadrature_z[index],
                [float(np.asarray(fitted.upper)[index])],
            )),
        )
    return result


__all__ = [
    "MeanConstrainedTiltedStudentT",
    "RawScaleMoments",
    "bounded_student_t_cdf",
    "bounded_student_t_log_density",
    "bounded_student_t_raw_moments",
    "fit_mean_constrained_tilted_student_t",
    "solve_mean_preserving_location",
    "tilted_bounded_student_t_log_density",
    "tilted_student_t_cdf",
    "tilted_student_t_quantile",
]
