from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .likelihood import (
    BridgeConfig,
    bridge_r_key_series,
    calibrate_component_sigma,
    delta_transform_var,
    normal_log_density,
    score_archive_rows,
    student_t_log_density,
    transform_value,
)
from caster.filter.evidence import compute_log_evidence, logsumexp
from caster.filter.hierarchical import (
    hierarchical_update_from_log_evidence,
    induce_model_weights,
    initialize_hierarchical_weights,
    summarize_hierarchical_posterior,
)
from caster.filter.outer_update import update_outer_weights, summarize_model_distribution
from caster.filter.prior import compute_family_balanced_prior, compute_model_uniform_prior
from caster.filter.availability import evidence_availability_by_model

DEFAULT_GAMMA_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
DEFAULT_NU_GRID = [5.0, 10.0, float("inf")]
FORMAL_FIXED_GAMMA = 1.0
DEFAULT_SIGMA_MULTIPLIERS = [0.5, 1.0, 2.0, 4.0]
FORMAL_SIGMA_SELECTION_POLICY = "residual_rmse_frozen"
RESULT_RHO_MIN = 0.05
RESULT_RHO_MAX = 1.0
DEFAULT_RHO_GRID = [0.05, 0.5, 1.0]
DEFAULT_GAMMA_PRIOR_STRENGTH = 0.01
DEFAULT_GAMMA_SCALE_RATIO_FLOOR = 0.75
DEFAULT_GAMMA_SCALE_RATIO_PENALTY_STRENGTH = 1.0
DEFAULT_RHO_MODEL_ESS_PENALTY = 0.5
DEFAULT_RHO_FAMILY_ESS_PENALTY = 0.5
DEFAULT_RHO_TOP1_PENALTY = 0.25
DEFAULT_RHO_TOP1_TARGET = 0.95


def config_to_dict(
    config: BridgeConfig,
    *,
    rho: float | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    data = asdict(config)
    if rho is not None:
        data["rho"] = float(rho)
    if metadata:
        data["calibration_metadata"] = metadata
    return data


def write_bridge_config(
    config: BridgeConfig,
    output_path: str | Path,
    *,
    rho: float | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config_to_dict(config, rho=rho, metadata=metadata)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
            path.write_text(yaml.safe_dump(data, sort_keys=True))
            return path
        except Exception:
            path = path.with_suffix(".json")
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return path


def read_bridge_config(path: str | Path) -> tuple[BridgeConfig, float | None]:
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml
        data = yaml.safe_load(path.read_text()) or {}
    else:
        data = json.loads(path.read_text())
    rho = data.pop("rho", None)
    data.pop("calibration_metadata", None)
    allowed = {f.name for f in fields(BridgeConfig)}
    config_data = {k: v for k, v in data.items() if k in allowed}
    return BridgeConfig(**config_data), (None if rho is None else float(rho))


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "t", "yes", "y"})


def _clean_gamma_grid(values: list[float] | None, *, allow_zero_gamma: bool = False) -> list[float]:
    raw = DEFAULT_GAMMA_GRID if values is None else list(values)
    out: set[float] = set()
    for value in raw:
        x = float(value)
        if not np.isfinite(x) or x < 0.0:
            raise ValueError("gamma_grid must contain only finite nonnegative values")
        if x == 0.0 and not allow_zero_gamma:
            raise ValueError("gamma_grid contains 0; pass allow_zero_gamma=True only for diagnostic/sensitivity runs")
        out.add(x)
    if not out:
        raise ValueError("gamma_grid must contain at least one value")
    return sorted(out)


def _clean_nu_grid(values: list[float] | None) -> list[float]:
    raw = DEFAULT_NU_GRID if values is None else list(values)
    out: set[float] = set()
    for value in raw:
        x = float(value)
        if np.isnan(x) or x <= 0.0:
            raise ValueError("nu_grid must contain only positive values or infinity")
        out.add(x)
    if not out:
        raise ValueError("nu_grid must contain at least one value")
    return sorted(out, key=lambda value: (np.isinf(value), value))


def _clean_sigma_multipliers(values: list[float] | None) -> list[float]:
    raw = DEFAULT_SIGMA_MULTIPLIERS if values is None else list(values)
    out: set[float] = set()
    for value in raw:
        x = float(value)
        if not np.isfinite(x) or x <= 0.0:
            raise ValueError("sigma_multipliers must contain only finite positive values")
        out.add(x)
    if not out:
        raise ValueError("sigma_multipliers must contain at least one value")
    return sorted(out)


def _float_list_text(values: list[float]) -> str:
    return ",".join(f"{float(v):g}" for v in values)


def clean_validation_rho_grid(
    values: list[float] | None,
    *,
    enforce_result_range: bool = True,
) -> list[float]:
    ""





    raw = DEFAULT_RHO_GRID if values is None else list(values)
    out: set[float] = set()
    for value in raw:
        x = float(value)
        if not np.isfinite(x) or x < 0.0:
            raise ValueError("rho grid values must be finite and nonnegative")
        if enforce_result_range and (x < RESULT_RHO_MIN or x > RESULT_RHO_MAX):
            raise ValueError(f"validation rho grid values must be in [{RESULT_RHO_MIN:g}, {RESULT_RHO_MAX:g}]")
        out.add(x)
    if not out:
        raise ValueError("rho grid must contain at least one value")
    return sorted(out)


def validate_fixed_rho(value: float | str) -> tuple[float, bool]:
    ""
    rho = float(value)
    if not np.isfinite(rho) or rho < 0.0:
        raise ValueError("fixed rho must be finite and nonnegative")
    outside = bool(rho < RESULT_RHO_MIN or rho > RESULT_RHO_MAX)
    return rho, outside


def _sigma_candidates(base_sigma: float, min_sigma: float, multipliers: list[float], default_sigma: float) -> list[float]:
    raw = [float(min_sigma)] + [float(base_sigma) * float(mult) for mult in multipliers]
    candidates = sorted({float(x) for x in raw if np.isfinite(x) and float(x) > 0.0})
    if not candidates:
        candidates = [max(float(default_sigma), float(min_sigma), 1e-12)]
    return candidates


def _mean_bridge_nll(
    y_t: np.ndarray,
    mu_t: np.ndarray,
    pred_v_t: np.ndarray,
    *,
    distribution: str,
    sigma: float,
    gamma: float,
    nu: float,
    min_scale: float = 1e-3,
) -> float:
    scale = np.sqrt(np.maximum(float(gamma) * pred_v_t, 0.0) + float(sigma) ** 2 + float(min_scale) ** 2)
    if distribution == "gaussian" or np.isinf(float(nu)):
        log_scores = np.asarray([normal_log_density(y, mu, s) for y, mu, s in zip(y_t, mu_t, scale)], dtype=float)
    else:
        log_scores = np.asarray([student_t_log_density(y, mu, s, nu) for y, mu, s in zip(y_t, mu_t, scale)], dtype=float)
    finite = np.isfinite(log_scores)
    if not finite.any():
        return float("nan")
    return float(-np.mean(log_scores[finite]))


def _validate_nonnegative_finite(value: float, name: str) -> float:
    x = float(value)
    if not np.isfinite(x) or x < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return x


def _validate_positive_finite(value: float, name: str) -> float:
    x = float(value)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return x


def _gamma_scale_ratio(
    pred_v_t: np.ndarray,
    *,
    base_sigma: float,
    sigma: float,
    gamma: float,
    min_scale: float = 1e-3,
) -> float:
    ref = np.sqrt(np.maximum(pred_v_t, 0.0) + float(base_sigma) ** 2 + float(min_scale) ** 2)
    cand = np.sqrt(np.maximum(float(gamma) * pred_v_t, 0.0) + float(sigma) ** 2 + float(min_scale) ** 2)
    ratio = cand / np.maximum(ref, 1e-12)
    finite = np.isfinite(ratio)
    if not finite.any():
        return float("nan")
    return float(np.median(ratio[finite]))


def _gamma_regularized_objective(
    raw_validation_nll: float,
    *,
    gamma: float,
    scale_ratio: float,
    gamma_prior_strength: float,
    scale_ratio_floor: float,
    scale_ratio_penalty_strength: float,
) -> tuple[float, float, float]:
    if not np.isfinite(raw_validation_nll):
        return float("nan"), float("nan"), float("nan")
    safe_gamma = max(float(gamma), 1e-8)
    gamma_prior_penalty = float(gamma_prior_strength) * float(np.log(safe_gamma) ** 2)
    if np.isfinite(scale_ratio) and scale_ratio > 0.0:
        deficit = max(0.0, float(np.log(float(scale_ratio_floor) / float(scale_ratio))))
        scale_penalty = float(scale_ratio_penalty_strength) * float(deficit ** 2)
    else:
        scale_penalty = float("inf")
    return float(raw_validation_nll + gamma_prior_penalty + scale_penalty), gamma_prior_penalty, scale_penalty


def calibrate_component_sigma_gamma(
    validation_ledger: pd.DataFrame,
    validation_archive: pd.DataFrame,
    *,
    distribution: str = "gaussian",
    transform: str = "log1p",
    nu: float = 5.0,
    min_sigma: float = 0.04,
    default_sigma: float = 0.20,
    gamma_grid: list[float] | None = None,
    fixed_gamma: float | None = None,
    sigma_multipliers: list[float] | None = None,
    min_rows_per_component: int = 5,
    allow_zero_gamma: bool = False,
    gamma_nll_tie_tol: float = 1e-12,
    gamma_prior_strength: float = DEFAULT_GAMMA_PRIOR_STRENGTH,
    gamma_scale_ratio_floor: float = DEFAULT_GAMMA_SCALE_RATIO_FLOOR,
    gamma_scale_ratio_penalty_strength: float = DEFAULT_GAMMA_SCALE_RATIO_PENALTY_STRENGTH,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
    ""
    distribution = str(distribution)
    tie_tol = float(gamma_nll_tie_tol)
    if not np.isfinite(tie_tol) or tie_tol < 0.0:
        raise ValueError("gamma_nll_tie_tol must be finite and nonnegative")
    gamma_prior_strength = _validate_nonnegative_finite(gamma_prior_strength, "gamma_prior_strength")
    gamma_scale_ratio_floor = _validate_positive_finite(gamma_scale_ratio_floor, "gamma_scale_ratio_floor")
    gamma_scale_ratio_penalty_strength = _validate_nonnegative_finite(
        gamma_scale_ratio_penalty_strength,
        "gamma_scale_ratio_penalty_strength",
    )
    if distribution == "negative_binomial":
        sigma = calibrate_component_sigma(validation_ledger, validation_archive, transform=transform, min_sigma=min_sigma)
        rows = [
            {
                "component": comp,
                "n_rows": 0,
                "selected_sigma": float(value),
                "selected_gamma": 1.0,
                "validation_nll": np.nan,
                "base_sigma": float(value),
                "status": "negative_binomial_gamma_unused",
                "raw_validation_nll": np.nan,
                "objective_nll": np.nan,
                "gamma_prior_penalty": 0.0,
                "scale_ratio": np.nan,
                "scale_ratio_penalty": 0.0,
                "gamma_grid": "",
                "sigma_candidates": "",
                "sigma_selection_policy": "negative_binomial_gamma_unused",
                "candidate_count": 0,
                "allow_zero_gamma": bool(allow_zero_gamma),
                "gamma_nll_tie_tol": float(tie_tol),
                "gamma_prior_strength": float(gamma_prior_strength),
                "gamma_scale_ratio_floor": float(gamma_scale_ratio_floor),
                "gamma_scale_ratio_penalty_strength": float(gamma_scale_ratio_penalty_strength),
            }
            for comp, value in sorted(sigma.items())
        ]
        return sigma, {}, pd.DataFrame(rows)
    if distribution not in {"student_t", "gaussian"}:
        raise ValueError(f"unsupported bridge distribution {distribution!r}")

    fixed_gamma_value: float | None = None
    if fixed_gamma is not None:
        fixed_gamma_value = _validate_nonnegative_finite(fixed_gamma, "fixed_gamma")
        if fixed_gamma_value == 0.0 and not allow_zero_gamma:
            raise ValueError("fixed_gamma is 0; pass allow_zero_gamma=True only for diagnostic/sensitivity runs")
        gamma_values = [fixed_gamma_value]
    else:
        gamma_values = _clean_gamma_grid(gamma_grid, allow_zero_gamma=allow_zero_gamma)
                                                                           
                                                                             
                                                                                  
                 
    _clean_sigma_multipliers(sigma_multipliers)
    min_rows = max(int(min_rows_per_component), 1)

    joined = validation_archive.merge(
        validation_ledger[["forecast_id", "observed_value", "observed_mask"]],
        on="forecast_id",
        how="inner",
    )
    if joined.empty:
        report = pd.DataFrame(
            columns=[
                "component",
                "n_rows",
                "selected_sigma",
                "selected_gamma",
                "validation_nll",
                "base_sigma",
                "status",
                "raw_validation_nll",
                "objective_nll",
                "gamma_prior_penalty",
                "scale_ratio",
                "scale_ratio_penalty",
                "gamma_grid",
                "sigma_candidates",
                "sigma_selection_policy",
                "candidate_count",
                "allow_zero_gamma",
                "gamma_nll_tie_tol",
                "gamma_prior_strength",
                "gamma_scale_ratio_floor",
                "gamma_scale_ratio_penalty_strength",
            ]
        )
        return {}, {}, report
    joined = joined[_bool_series(joined["observed_mask"])].copy()
    if joined.empty:
        report = pd.DataFrame(
            columns=[
                "component",
                "n_rows",
                "selected_sigma",
                "selected_gamma",
                "validation_nll",
                "base_sigma",
                "status",
                "raw_validation_nll",
                "objective_nll",
                "gamma_prior_penalty",
                "scale_ratio",
                "scale_ratio_penalty",
                "gamma_grid",
                "sigma_candidates",
                "sigma_selection_policy",
                "candidate_count",
                "allow_zero_gamma",
                "gamma_nll_tie_tol",
                "gamma_prior_strength",
                "gamma_scale_ratio_floor",
                "gamma_scale_ratio_penalty_strength",
            ]
        )
        return {}, {}, report

    joined["bridge_r_key"] = bridge_r_key_series(joined)
    sigma_by_component: dict[str, float] = {}
    gamma_by_component: dict[str, float] = {}
    report_rows: list[dict[str, object]] = []
    for parameter_key, group in joined.groupby("bridge_r_key", sort=True):
        comp_key = str(parameter_key)
        y_vals: list[float] = []
        mu_vals: list[float] = []
        pred_v_vals: list[float] = []
        for _, row in group.iterrows():
            y_t = transform_value(float(row["observed_value"]), transform)
            mu_t = transform_value(float(row["pred_mean"]), transform)
            pred_v_t = delta_transform_var(float(row["pred_mean"]), float(row["pred_var"]), transform)
            if np.isfinite(y_t) and np.isfinite(mu_t) and np.isfinite(pred_v_t):
                y_vals.append(float(y_t))
                mu_vals.append(float(mu_t))
                pred_v_vals.append(max(float(pred_v_t), 0.0))

        y_arr = np.asarray(y_vals, dtype=float)
        mu_arr = np.asarray(mu_vals, dtype=float)
        pred_v_arr = np.asarray(pred_v_vals, dtype=float)
        n_rows = int(y_arr.size)
        if n_rows:
            base_sigma = max(float(min_sigma), float(np.sqrt(np.mean(np.square(y_arr - mu_arr)))))
        else:
            base_sigma = max(float(min_sigma), float(default_sigma))
        candidates = [float(base_sigma)]

        fallback = n_rows < min_rows or pred_v_arr.size == 0 or float(np.max(pred_v_arr)) <= 1e-12
        if fixed_gamma_value is not None:
            selected_sigma = float(base_sigma)
            selected_gamma = float(fixed_gamma_value)
            validation_nll = (
                _mean_bridge_nll(
                    y_arr,
                    mu_arr,
                    pred_v_arr,
                    distribution=distribution,
                    sigma=selected_sigma,
                    gamma=selected_gamma,
                    nu=nu,
                )
                if n_rows
                else float("nan")
            )
            scale_ratio = (
                _gamma_scale_ratio(
                    pred_v_arr,
                    base_sigma=base_sigma,
                    sigma=selected_sigma,
                    gamma=selected_gamma,
                )
                if n_rows
                else float("nan")
            )
            objective_nll = validation_nll
            gamma_prior_penalty = 0.0
            scale_ratio_penalty = 0.0
            status = "fixed_gamma_sigma_frozen"
            candidate_count = 0
        elif fallback:
            selected_sigma = float(base_sigma)
            selected_gamma = 1.0
            validation_nll = (
                _mean_bridge_nll(
                    y_arr,
                    mu_arr,
                    pred_v_arr,
                    distribution=distribution,
                    sigma=selected_sigma,
                    gamma=selected_gamma,
                    nu=nu,
                )
                if n_rows
                else float("nan")
            )
            scale_ratio = (
                _gamma_scale_ratio(
                    pred_v_arr,
                    base_sigma=base_sigma,
                    sigma=selected_sigma,
                    gamma=selected_gamma,
                )
                if n_rows
                else float("nan")
            )
            objective_nll, gamma_prior_penalty, scale_ratio_penalty = _gamma_regularized_objective(
                validation_nll,
                gamma=selected_gamma,
                scale_ratio=scale_ratio,
                gamma_prior_strength=gamma_prior_strength,
                scale_ratio_floor=gamma_scale_ratio_floor,
                scale_ratio_penalty_strength=gamma_scale_ratio_penalty_strength,
            )
            status = "fallback_too_few_rows" if n_rows < min_rows else "fallback_zero_pred_var"
            candidate_count = 0
        else:
            selected_sigma = float(base_sigma)
            selected_gamma = gamma_values[0]
            validation_nll = float("nan")
            objective_nll = float("nan")
            gamma_prior_penalty = float("nan")
            scale_ratio = float("nan")
            scale_ratio_penalty = float("nan")
            candidate_count = 0
            candidate_records: list[dict[str, float]] = []
            for gamma in gamma_values:
                candidate_count += 1
                nll = _mean_bridge_nll(
                    y_arr,
                    mu_arr,
                    pred_v_arr,
                    distribution=distribution,
                    sigma=selected_sigma,
                    gamma=float(gamma),
                    nu=nu,
                )
                if not np.isfinite(nll):
                    continue
                candidate_scale_ratio = _gamma_scale_ratio(
                    pred_v_arr,
                    base_sigma=base_sigma,
                    sigma=selected_sigma,
                    gamma=float(gamma),
                )
                obj, gamma_penalty, scale_penalty = _gamma_regularized_objective(
                    float(nll),
                    gamma=float(gamma),
                    scale_ratio=candidate_scale_ratio,
                    gamma_prior_strength=gamma_prior_strength,
                    scale_ratio_floor=gamma_scale_ratio_floor,
                    scale_ratio_penalty_strength=gamma_scale_ratio_penalty_strength,
                )
                if not np.isfinite(obj):
                    continue
                candidate_records.append(
                    {
                        "raw_validation_nll": float(nll),
                        "objective_nll": float(obj),
                        "sigma": selected_sigma,
                        "gamma": float(gamma),
                        "gamma_prior_penalty": float(gamma_penalty),
                        "scale_ratio": float(candidate_scale_ratio),
                        "scale_ratio_penalty": float(scale_penalty),
                    }
                )
            if not candidate_records:
                selected_sigma = float(base_sigma)
                selected_gamma = 1.0
                validation_nll = _mean_bridge_nll(
                    y_arr,
                    mu_arr,
                    pred_v_arr,
                    distribution=distribution,
                    sigma=selected_sigma,
                    gamma=selected_gamma,
                    nu=nu,
                )
                scale_ratio = _gamma_scale_ratio(
                    pred_v_arr,
                    base_sigma=base_sigma,
                    sigma=selected_sigma,
                    gamma=selected_gamma,
                )
                objective_nll, gamma_prior_penalty, scale_ratio_penalty = _gamma_regularized_objective(
                    validation_nll,
                    gamma=selected_gamma,
                    scale_ratio=scale_ratio,
                    gamma_prior_strength=gamma_prior_strength,
                    scale_ratio_floor=gamma_scale_ratio_floor,
                    scale_ratio_penalty_strength=gamma_scale_ratio_penalty_strength,
                )
                status = "fallback_no_finite_candidate"
            else:
                min_obj = min(record["objective_nll"] for record in candidate_records)
                eligible = [record for record in candidate_records if record["objective_nll"] <= min_obj + tie_tol]
                best_record = min(
                    eligible,
                    key=lambda record: (
                        abs(float(np.log(max(record["gamma"], 1e-8)))),
                        record["gamma"],
                        record["objective_nll"],
                    ),
                )
                validation_nll = best_record["raw_validation_nll"]
                objective_nll = best_record["objective_nll"]
                selected_sigma = best_record["sigma"]
                selected_gamma = best_record["gamma"]
                gamma_prior_penalty = best_record["gamma_prior_penalty"]
                scale_ratio = best_record["scale_ratio"]
                scale_ratio_penalty = best_record["scale_ratio_penalty"]
                status = "selected_gamma_grid_sigma_frozen"

        sigma_by_component[comp_key] = selected_sigma
        gamma_by_component[comp_key] = selected_gamma
        report_rows.append(
            {
                "component": comp_key,
                "bridge_parameter_key": comp_key,
                "bridge_parameter_scope": "component_horizon",
                "n_rows": n_rows,
                "selected_sigma": selected_sigma,
                "selected_gamma": selected_gamma,
                "validation_nll": validation_nll,
                "base_sigma": float(base_sigma),
                "status": status,
                "raw_validation_nll": validation_nll,
                "objective_nll": objective_nll,
                "gamma_prior_penalty": gamma_prior_penalty,
                "scale_ratio": scale_ratio,
                "scale_ratio_penalty": scale_ratio_penalty,
                "gamma_grid": "" if fixed_gamma_value is not None else _float_list_text(gamma_values),
                "gamma_selection_policy": "fixed_input" if fixed_gamma_value is not None else "validation_grid",
                "fixed_gamma": "" if fixed_gamma_value is None else float(fixed_gamma_value),
                "sigma_candidates": _float_list_text(candidates),
                "sigma_selection_policy": FORMAL_SIGMA_SELECTION_POLICY,
                "candidate_count": int(candidate_count),
                "allow_zero_gamma": bool(allow_zero_gamma),
                "gamma_nll_tie_tol": float(tie_tol),
                "gamma_prior_strength": float(gamma_prior_strength),
                "gamma_scale_ratio_floor": float(gamma_scale_ratio_floor),
                "gamma_scale_ratio_penalty_strength": float(gamma_scale_ratio_penalty_strength),
            }
        )

    return sigma_by_component, gamma_by_component, pd.DataFrame(report_rows)


def calibrate_component_eta(
    validation_ledger: pd.DataFrame,
    validation_archive: pd.DataFrame,
    *,
    distribution: str = "student_t",
    transform: str = "log1p",
    min_sigma: float = 0.04,
    default_sigma: float = 0.20,
    gamma_grid: list[float] | None = None,
    nu_grid: list[float] | None = None,
    fixed_gamma: float | None = None,
    fixed_nu: float | None = None,
    allow_zero_gamma: bool = False,
    eta_regularization: float = DEFAULT_GAMMA_PRIOR_STRENGTH,
    objective_tie_tol: float = 1e-12,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], pd.DataFrame]:
    ""








    distribution = str(distribution).strip().lower()
    if distribution not in {"student_t", "gaussian"}:
        raise NotImplementedError(
            "component eta selection implements the Student-t bridge only; "
            "the optional Negative-Binomial phi_r branch is not implemented"
        )
    eta_regularization = _validate_nonnegative_finite(eta_regularization, "eta_regularization")
    tie_tol = _validate_nonnegative_finite(objective_tie_tol, "objective_tie_tol")
    if fixed_gamma is None:
        gamma_values = _clean_gamma_grid(gamma_grid, allow_zero_gamma=allow_zero_gamma)
    else:
        gamma_value = _validate_nonnegative_finite(fixed_gamma, "fixed_gamma")
        if gamma_value == 0.0 and not allow_zero_gamma:
            raise ValueError("fixed_gamma is 0; allow_zero_gamma is required for a diagnostic run")
        gamma_values = [gamma_value]
    if distribution == "gaussian":
        nu_values = [float("inf")]
    elif fixed_nu is not None:
        nu_values = _clean_nu_grid([fixed_nu])
    else:
        nu_values = _clean_nu_grid(nu_grid)

    ledger_columns = ["forecast_id", "observed_value", "observed_mask"]
    if "release_time" in validation_ledger.columns:
        ledger_columns.append("release_time")
    ledger_meta = validation_ledger[ledger_columns].drop_duplicates("forecast_id")
    joined = validation_archive.merge(ledger_meta, on="forecast_id", how="inner")
    if joined.empty:
        return {}, {}, {}, pd.DataFrame()
    joined = joined[_bool_series(joined["observed_mask"])].copy()
    if joined.empty:
        return {}, {}, {}, pd.DataFrame()
    joined["bridge_r_key"] = bridge_r_key_series(joined)
    if "release_time" not in joined.columns:
        joined["release_time"] = "validation"
    if "model_id" not in joined.columns:
        joined["model_id"] = "candidate"

    sigma_by_component: dict[str, float] = {}
    gamma_by_component: dict[str, float] = {}
    nu_by_component: dict[str, float] = {}
    report_rows: list[dict[str, object]] = []
    for parameter_key, group in joined.groupby("bridge_r_key", sort=True):
        work = group.copy()
        work["y_transformed"] = [transform_value(value, transform) for value in work["observed_value"]]
        work["mu_transformed"] = [transform_value(value, transform) for value in work["pred_mean"]]
        work["pred_var_transformed"] = [
            delta_transform_var(mu, var, transform)
            for mu, var in zip(work["pred_mean"], work["pred_var"])
        ]
        finite = np.isfinite(
            work[["y_transformed", "mu_transformed", "pred_var_transformed"]].astype(float).to_numpy()
        ).all(axis=1)
        work = work.loc[finite].copy()
        if work.empty:
            sigma = max(float(min_sigma), float(default_sigma))
            selected_gamma = 1.0 if 1.0 in gamma_values else float(gamma_values[0])
            selected_nu = float(nu_values[0])
            raw_mean_nll = float("nan")
            data_objective = float("nan")
            objective = float("nan")
            regularization = float("nan")
            status = "fallback_no_finite_validation_rows"
            candidate_count = 0
            n_release_candidate_scores = 0
            n_release_scores = 0
        else:
            residual = work["y_transformed"].astype(float) - work["mu_transformed"].astype(float)
            sigma = max(float(min_sigma), float(np.sqrt(np.mean(np.square(residual.to_numpy())))))
            candidates: list[dict[str, float]] = []
            scale_base = np.square(sigma) + float(1e-3) ** 2
            for gamma in gamma_values:
                scale = np.sqrt(
                    scale_base
                    + np.maximum(float(gamma) * work["pred_var_transformed"].astype(float).to_numpy(), 0.0)
                )
                y = work["y_transformed"].astype(float).to_numpy()
                mu = work["mu_transformed"].astype(float).to_numpy()
                for nu_value in nu_values:
                    if distribution == "gaussian" or np.isinf(nu_value):
                        scores = np.asarray(
                            [normal_log_density(y_i, mu_i, scale_i) for y_i, mu_i, scale_i in zip(y, mu, scale)],
                            dtype=float,
                        )
                    else:
                        scores = np.asarray(
                            [
                                student_t_log_density(y_i, mu_i, scale_i, nu_value)
                                for y_i, mu_i, scale_i in zip(y, mu, scale)
                            ],
                            dtype=float,
                        )
                    scored = work[["release_time", "model_id"]].copy()
                    scored["log_score"] = scores
                                                                                    
                    release_candidate = scored.groupby(
                        ["release_time", "model_id"], sort=True, dropna=False
                    )["log_score"].mean()
                                                                             
                                                                           
                                                                            
                                                                        
                                        
                    release_scores = release_candidate.groupby(level="release_time").mean()
                    raw_mean = float(-release_candidate.mean())
                    data_term = float(-release_scores.sum())
                    penalty = float(eta_regularization) * float(np.log(max(float(gamma), 1e-12)) ** 2)
                    candidates.append(
                        {
                            "gamma": float(gamma),
                            "nu": float(nu_value),
                            "raw_mean_nll": raw_mean,
                            "data_objective": data_term,
                            "regularization": penalty,
                            "objective": data_term + penalty,
                            "n_release_candidate_scores": float(len(release_candidate)),
                            "n_release_scores": float(len(release_scores)),
                        }
                    )
            minimum = min(item["objective"] for item in candidates)
            eligible = [item for item in candidates if item["objective"] <= minimum + tie_tol]
            best = min(
                eligible,
                key=lambda item: (
                    abs(float(np.log(max(item["gamma"], 1e-12)))),
                    item["gamma"],
                    -item["nu"],
                ),
            )
            selected_gamma = float(best["gamma"])
            selected_nu = float(best["nu"])
            raw_mean_nll = float(best["raw_mean_nll"])
            data_objective = float(best["data_objective"])
            regularization = float(best["regularization"])
            objective = float(best["objective"])
            status = (
                "fixed_gamma_sigma_frozen"
                if fixed_gamma is not None and (distribution == "gaussian" or fixed_nu is not None)
                else "selected_joint_validation_grid"
            )
            candidate_count = len(candidates)
            n_release_candidate_scores = int(best["n_release_candidate_scores"])
            n_release_scores = int(best["n_release_scores"])

        key = str(parameter_key)
        sigma_by_component[key] = float(sigma)
        gamma_by_component[key] = float(selected_gamma)
        nu_by_component[key] = float(selected_nu)
        report_rows.append(
            {
                "component": key,
                "bridge_r_key": key,
                "bridge_parameter_key": key,
                "bridge_parameter_scope": "component_horizon",
                "n_rows": int(len(work)),
                "selected_sigma": float(sigma),
                "base_sigma": float(sigma),
                "selected_gamma": float(selected_gamma),
                "selected_nu": float(selected_nu),
                "validation_nll": raw_mean_nll,
                "raw_validation_nll": raw_mean_nll,
                "bridge_data_objective": data_objective,
                "eta_regularization": regularization,
                "objective_nll": objective,
                "status": status,
                "sigma_selection_policy": FORMAL_SIGMA_SELECTION_POLICY,
                "sigma_candidates": f"{float(sigma):g}",
                "eta_selection_policy": "component_horizon_joint_validation_grid",
                "gamma_grid": _float_list_text(gamma_values),
                "nu_grid": _float_list_text(nu_values),
                "gamma_selection_policy": "fixed_input" if fixed_gamma is not None else "validation_grid",
                "fixed_gamma": "" if fixed_gamma is None else float(fixed_gamma),
                "gamma_nll_tie_tol": float(tie_tol),
                "candidate_count": int(candidate_count),
                "n_release_candidate_scores": int(n_release_candidate_scores),
                "n_release_scores": int(n_release_scores),
                "eta_regularization_strength": float(eta_regularization),
                "negative_binomial_phi_implemented": False,
            }
        )
    return sigma_by_component, gamma_by_component, nu_by_component, pd.DataFrame(report_rows)


def fit_bridge_config(
    validation_ledger: pd.DataFrame,
    archive: pd.DataFrame,
    *,
    distribution: str = "gaussian",
    transform: str = "log1p",
    nu: float = 5.0,
    min_sigma: float = 0.04,
    default_sigma: float = 0.20,
    gamma_grid: list[float] | None = None,
    nu_grid: list[float] | None = None,
    fixed_gamma: float | None = None,
    fixed_nu: float | None = None,
    sigma_multipliers: list[float] | None = None,
    min_rows_per_component: int = 5,
    allow_zero_gamma: bool = False,
    gamma_nll_tie_tol: float = 1e-12,
    gamma_prior_strength: float = DEFAULT_GAMMA_PRIOR_STRENGTH,
    gamma_scale_ratio_floor: float = DEFAULT_GAMMA_SCALE_RATIO_FLOOR,
    gamma_scale_ratio_penalty_strength: float = DEFAULT_GAMMA_SCALE_RATIO_PENALTY_STRENGTH,
    return_report: bool = False,
) -> BridgeConfig | tuple[BridgeConfig, pd.DataFrame]:
    tie_tol = float(gamma_nll_tie_tol)
    if not np.isfinite(tie_tol) or tie_tol < 0.0:
        raise ValueError("gamma_nll_tie_tol must be finite and nonnegative")
    gamma_prior_strength = _validate_nonnegative_finite(gamma_prior_strength, "gamma_prior_strength")
    gamma_scale_ratio_floor = _validate_positive_finite(gamma_scale_ratio_floor, "gamma_scale_ratio_floor")
    gamma_scale_ratio_penalty_strength = _validate_nonnegative_finite(
        gamma_scale_ratio_penalty_strength,
        "gamma_scale_ratio_penalty_strength",
    )
    if distribution == "negative_binomial":
        sigma = calibrate_component_sigma(validation_ledger, archive, transform=transform, min_sigma=min_sigma)
        report = pd.DataFrame(
            [
                {
                    "component": comp,
                    "n_rows": 0,
                    "selected_sigma": float(value),
                    "selected_gamma": 1.0,
                    "validation_nll": np.nan,
                    "base_sigma": float(value),
                    "status": "negative_binomial_gamma_unused",
                    "raw_validation_nll": np.nan,
                    "objective_nll": np.nan,
                    "gamma_prior_penalty": 0.0,
                    "scale_ratio": np.nan,
                    "scale_ratio_penalty": 0.0,
                    "gamma_grid": "",
                    "sigma_candidates": "",
                    "sigma_selection_policy": "negative_binomial_gamma_unused",
                    "candidate_count": 0,
                    "allow_zero_gamma": bool(allow_zero_gamma),
                    "gamma_nll_tie_tol": float(tie_tol),
                    "gamma_prior_strength": float(gamma_prior_strength),
                    "gamma_scale_ratio_floor": float(gamma_scale_ratio_floor),
                    "gamma_scale_ratio_penalty_strength": float(gamma_scale_ratio_penalty_strength),
                }
                for comp, value in sorted(sigma.items())
            ]
        )
        config = BridgeConfig(
            distribution=distribution,
            transform=transform,
            nu=nu,
            min_scale=1e-3,
            sigma_by_component=sigma,
            gamma_by_component={},
            default_sigma=default_sigma,
            default_gamma=1.0,
        )
        return (config, report) if return_report else config

                                                                             
                                                                         
    active_fixed_nu = fixed_nu
    if distribution == "student_t" and nu_grid is None and active_fixed_nu is None:
        active_fixed_nu = float(nu)
    sigma, gamma, nu_by_component, report = calibrate_component_eta(
        validation_ledger,
        archive,
        distribution=distribution,
        transform=transform,
        min_sigma=min_sigma,
        default_sigma=default_sigma,
        gamma_grid=gamma_grid,
        nu_grid=nu_grid,
        fixed_gamma=fixed_gamma,
        fixed_nu=active_fixed_nu,
        allow_zero_gamma=allow_zero_gamma,
        eta_regularization=gamma_prior_strength,
        objective_tie_tol=tie_tol,
    )
    config = BridgeConfig(
        distribution=distribution,
        transform=transform,
        nu=nu,
        min_scale=1e-3,
        sigma_by_component=sigma,
        gamma_by_component=gamma,
        nu_by_component=nu_by_component,
        kernel_distribution="gaussian" if distribution == "gaussian" else "student_t",
        default_sigma=default_sigma,
        default_gamma=1.0,
    )
    return (config, report) if return_report else config


def release_log_evidence(
    ledger_rows: pd.DataFrame,
    archive: pd.DataFrame,
    registry: pd.DataFrame,
    config: BridgeConfig,
) -> pd.DataFrame:
    scored = score_archive_rows(ledger_rows, archive, config)
    return pd.DataFrame([
        {
            "release_time": pd.to_datetime(ledger_rows["release_time"]).iloc[0],
            "model_id": str(model_id),
            "log_evidence": compute_log_evidence(scored, model_id=str(model_id)),
        }
        for model_id in registry["model_id"].astype(str)
    ])


def _mean_squared_ess_penalty(values: list[float], target: float) -> float:
    if not values:
        return 0.0
    target = max(float(target), 1e-12)
    penalties = [max(0.0, float(np.log(target / max(float(value), 1e-12)))) ** 2 for value in values]
    return float(np.mean(penalties))


def _mean_squared_top1_penalty(values: list[float], target: float) -> float:
    if not values:
        return 0.0
    target = min(max(float(target), 0.0), 1.0 - 1e-12)
    denom = max(1.0 - target, 1e-12)
    penalties = [max(0.0, (float(value) - target) / denom) ** 2 for value in values]
    return float(np.mean(penalties))


def _first_top1_time(release_times: list[object], values: list[float], target: float) -> str:
    for rt, value in zip(release_times, values):
        if float(value) >= float(target):
            return str(pd.Timestamp(rt))
    return ""


def _validate_rho_regularization(
    *,
    target_ess_fraction: float,
    ess_penalty: float,
    family_ess_penalty: float,
    top1_penalty: float,
    top1_target: float,
) -> tuple[float, float, float, float, float]:
    target_ess_fraction = _validate_positive_finite(target_ess_fraction, "target_ess_fraction")
    if target_ess_fraction > 1.0:
        raise ValueError("target_ess_fraction must be <= 1")
    ess_penalty = _validate_nonnegative_finite(ess_penalty, "ess_penalty")
    family_ess_penalty = _validate_nonnegative_finite(family_ess_penalty, "family_ess_penalty")
    top1_penalty = _validate_nonnegative_finite(top1_penalty, "top1_penalty")
    top1_target = float(top1_target)
    if not np.isfinite(top1_target) or not (0.0 <= top1_target < 1.0):
        raise ValueError("top1_target must be finite and in [0, 1)")
    return target_ess_fraction, ess_penalty, family_ess_penalty, top1_penalty, top1_target


def _prequential_mixture_log_score(scored_rows: pd.DataFrame, model_weights: pd.DataFrame) -> float:
    ""





    rows = scored_rows[_bool_series(scored_rows["observed_mask"])].copy()
    if rows.empty:
        raise ValueError("validation release has no observed native forecast scores")
    weights = model_weights[["model_id", "weight"]].copy()
    weights["model_id"] = weights["model_id"].astype(str)
    weights["weight"] = weights["weight"].astype(float)
    rows["model_id"] = rows["model_id"].astype(str)
    event_model_rows: list[dict[str, object]] = []
    for (forecast_id, model_id), group in rows.groupby(["forecast_id", "model_id"], sort=False):
        values = group["log_score"].astype(float).to_numpy()
        log_density = logsumexp(values) - float(np.log(max(len(values), 1)))
        event_model_rows.append(
            {
                "forecast_id": str(forecast_id),
                "model_id": str(model_id),
                "log_density": float(log_density),
                "event_weight": float(group["event_weight"].astype(float).iloc[0]),
            }
        )
    event_model = pd.DataFrame(event_model_rows).merge(weights, on="model_id", how="inner")
    event_scores: list[tuple[float, float]] = []
    for _, group in event_model.groupby("forecast_id", sort=False):
        positive = group[group["weight"].astype(float) > 0.0]
        if positive.empty:
            raise ValueError(
                f"forecast_id={group['forecast_id'].iloc[0]} has no positive-weight native model"
            )
        available_weight = float(positive["weight"].astype(float).sum())
        if available_weight <= 0.0:
            raise ValueError(
                f"forecast_id={group['forecast_id'].iloc[0]} has nonpositive available posterior mass"
            )
        mixture_log_density = logsumexp(
            np.log(positive["weight"].astype(float).to_numpy() / available_weight)
            + positive["log_density"].astype(float).to_numpy()
        )
        event_scores.append((float(positive["event_weight"].iloc[0]), float(mixture_log_density)))
    if not event_scores:
        raise ValueError("validation release produced no native predictive-mixture scores")
    event_weights = np.asarray([item[0] for item in event_scores], dtype=float)
    if float(event_weights.sum()) <= 0.0:
        event_weights = np.ones_like(event_weights)
    event_weights = event_weights / event_weights.sum()
    return float(np.sum(event_weights * np.asarray([item[1] for item in event_scores], dtype=float)))


def _mark_selected_rho(report: pd.DataFrame, *, tie_tol: float = 1e-12) -> pd.DataFrame:
    ""
    out = report.copy()
    if "validation_mixture_nll" not in out:
        out["validation_mixture_nll"] = -out["objective"].astype(float)
    if "validation_mixture_nll_se" not in out:
        out["validation_mixture_nll_se"] = 0.0
    out = out.sort_values(
        ["validation_mixture_nll", "rho"],
        ascending=[True, True],
    ).reset_index(drop=True)
    out["selected"] = False
    out["one_se_eligible"] = False
    if not out.empty:
        best = float(out["validation_mixture_nll"].astype(float).min())
        cutoff = best + float(tie_tol)
        eligible = out[
            out["validation_mixture_nll"].astype(float) <= cutoff + float(tie_tol)
        ]
        out.loc[eligible.index, "one_se_eligible"] = True
        selected_index = eligible.sort_values("rho", ascending=True).index[0]
        out.loc[selected_index, "selected"] = True
        out["one_se_best_nll"] = best
        out["one_se_cutoff"] = cutoff
        out["selection_preference"] = "smaller_rho"
    return out


def evaluate_temperature_grid(
    validation_ledger: pd.DataFrame,
    archive: pd.DataFrame,
    registry: pd.DataFrame,
    config: BridgeConfig,
    *,
    grid: list[float] | None = None,
    target_ess_fraction: float = 0.5,
    ess_penalty: float = DEFAULT_RHO_MODEL_ESS_PENALTY,
    family_ess_penalty: float = 0.0,
    top1_penalty: float = DEFAULT_RHO_TOP1_PENALTY,
    top1_target: float = DEFAULT_RHO_TOP1_TARGET,
    prior_policy: str = "uniform_model",
) -> pd.DataFrame:
    ""





    target_ess_fraction, ess_penalty, family_ess_penalty, top1_penalty, top1_target = _validate_rho_regularization(
        target_ess_fraction=target_ess_fraction,
        ess_penalty=ess_penalty,
        family_ess_penalty=family_ess_penalty,
        top1_penalty=top1_penalty,
        top1_target=top1_target,
    )
    grid = clean_validation_rho_grid(grid, enforce_result_range=False)
    val = validation_ledger[validation_ledger["observed_mask"].astype(bool)].copy()
    if val.empty:
        raise ValueError("validation ledger has no observed rows")
    release_times = sorted(pd.to_datetime(val["release_time"]).unique())
    release_meta = val[["forecast_id", "release_time"]].drop_duplicates("forecast_id").copy()
    scored_validation = score_archive_rows(val, archive, config).merge(release_meta, on="forecast_id", how="left")
    scored_validation["release_time"] = pd.to_datetime(scored_validation["release_time"])
    n_models = max(len(registry), 1)
    target_ess = max(1.0, min(float(target_ess_fraction) * n_models, n_models))
    if prior_policy in {"uniform_model", "uniform_model_prior", "model_level_uniform"}:
        canonical_prior_policy = "uniform_model_prior"
        initial_prior = compute_model_uniform_prior(registry).rename(columns={"prior_weight": "weight"})[["model_id", "family", "weight"]]
    elif prior_policy == "family_balanced":
        canonical_prior_policy = "family_balanced"
        initial_prior = compute_family_balanced_prior(registry).rename(columns={"prior_weight": "weight"})[["model_id", "family", "weight"]]
    else:
        raise ValueError(f"unknown prior_policy {prior_policy!r}")
    reports = []
    for rho in grid:
        weights = initial_prior.copy()
        cumulative_log_score = 0.0
        release_nll_values: list[float] = []
        min_ess = float("inf")
        model_ess_values: list[float] = []
        top1_values: list[float] = []
        update_times: list[object] = []
        n_releases = 0
        for rt in release_times:
            current = scored_validation[scored_validation["release_time"] == pd.Timestamp(rt)]
            batch_ledger = val[
                pd.to_datetime(val["release_time"]).eq(pd.Timestamp(rt))
            ].copy()
            availability = evidence_availability_by_model(
                current, batch_ledger, registry["model_id"].astype(str)
            )
            log_ev = pd.DataFrame([
                {
                    "release_time": pd.Timestamp(rt),
                    "model_id": str(model_id),
                    "log_evidence": compute_log_evidence(current, model_id=str(model_id))
                    if availability[str(model_id)] else 0.0,
                    "evidence_available": availability[str(model_id)],
                }
                for model_id in registry["model_id"].astype(str)
            ])
            release_log_score = _prequential_mixture_log_score(current, weights)
            cumulative_log_score += release_log_score
            release_nll_values.append(-float(release_log_score))
            weights = update_outer_weights(weights, log_ev, rho=float(rho))
            ess = float(summarize_model_distribution(weights)["model_ess"])
            min_ess = min(min_ess, ess)
            model_ess_values.append(ess)
            top1_values.append(float(weights["weight"].astype(float).max()))
            update_times.append(rt)
            weights = weights[["model_id", "family", "weight"]]
            n_releases += 1
        mean_validation_log_score = cumulative_log_score / max(n_releases, 1)
        validation_mixture_nll = -float(cumulative_log_score)
        validation_mixture_nll_se = (
            float(np.std(release_nll_values, ddof=1) * np.sqrt(len(release_nll_values)))
            if len(release_nll_values) > 1
            else 0.0
        )
        model_penalty_value = _mean_squared_ess_penalty(model_ess_values, target_ess)
        family_penalty_value = 0.0
        top1_penalty_value = _mean_squared_top1_penalty(top1_values, top1_target)
        regularization_penalty = (
            float(ess_penalty) * model_penalty_value
            + float(family_ess_penalty) * family_penalty_value
            + float(top1_penalty) * top1_penalty_value
        )
        objective = cumulative_log_score
        reports.append({
            "rho": float(rho),
            "validation_mixture_nll": validation_mixture_nll,
            "validation_mixture_nll_se": validation_mixture_nll_se,
            "validation_log_score": float(cumulative_log_score),
            "mean_validation_log_score": float(mean_validation_log_score),
            "ess_penalty": float(regularization_penalty),
            "model_ess_penalty": float(model_penalty_value),
            "family_ess_penalty": float(family_penalty_value),
            "top1_penalty": float(top1_penalty_value),
            "regularization_penalty": float(regularization_penalty),
            "objective": float(objective),
            "regularized_objective": float(mean_validation_log_score - regularization_penalty),
            "min_model_ess": float(min_ess if np.isfinite(min_ess) else n_models),
            "target_model_ess": float(target_ess),
            "rho_model_ess_penalty_weight": float(ess_penalty),
            "rho_family_ess_penalty_weight": float(family_ess_penalty),
            "rho_top1_penalty_weight": float(top1_penalty),
            "top1_target": float(top1_target),
            "top1_model_weight_max": float(max(top1_values) if top1_values else 0.0),
            "top1_family_weight_max": np.nan,
            "first_top1_ge_target_release_time": _first_top1_time(update_times, top1_values, top1_target),
            "n_releases": int(n_releases),
            "prior_policy": canonical_prior_policy,
            "rho_selection_variant": "one_layer",
            "rho_selection_mode": "validation_grid",
            "rho_selection_objective": "prequential_untempered_validation_mixture_nll",
            "filter_dynamics": "bayesian_evidence_update",
            "rho_regularization_used_for_selection": False,
            "predictive_density_tempered": False,
            "mixture_scored_before_update": True,
        })
    return _mark_selected_rho(pd.DataFrame(reports))


def _hierarchical_family_evidence(inner_weights: pd.DataFrame, log_evidence: pd.DataFrame) -> pd.DataFrame:
    inner = inner_weights[["family", "model_id", "inner_weight"]].copy()
    ev = log_evidence[["model_id", "log_evidence"]].copy()
    inner["model_id"] = inner["model_id"].astype(str)
    ev["model_id"] = ev["model_id"].astype(str)
    df = inner.merge(ev, on="model_id", how="left")
    df["log_evidence"] = df["log_evidence"].fillna(0.0).astype(float)
    df["inner_weight"] = df["inner_weight"].fillna(0.0).astype(float)
    rows = []
    for family, group in df.groupby("family", sort=True):
        log_terms = np.log(np.maximum(group["inner_weight"].to_numpy(dtype=float), 1e-300)) + group["log_evidence"].to_numpy(dtype=float)
        rows.append({"family": str(family), "family_log_evidence": logsumexp(log_terms)})
    return pd.DataFrame(rows)


def evaluate_hierarchical_temperature_grid(
    validation_ledger: pd.DataFrame,
    archive: pd.DataFrame,
    registry: pd.DataFrame,
    config: BridgeConfig,
    *,
    grid: list[float] | None = None,
    target_ess_fraction: float = 0.5,
    ess_penalty: float = DEFAULT_RHO_MODEL_ESS_PENALTY,
    family_ess_penalty: float = DEFAULT_RHO_FAMILY_ESS_PENALTY,
    top1_penalty: float = DEFAULT_RHO_TOP1_PENALTY,
    top1_target: float = DEFAULT_RHO_TOP1_TARGET,
) -> pd.DataFrame:
    ""





    target_ess_fraction, ess_penalty, family_ess_penalty, top1_penalty, top1_target = _validate_rho_regularization(
        target_ess_fraction=target_ess_fraction,
        ess_penalty=ess_penalty,
        family_ess_penalty=family_ess_penalty,
        top1_penalty=top1_penalty,
        top1_target=top1_target,
    )
    grid = clean_validation_rho_grid(grid, enforce_result_range=False)
    val = validation_ledger[validation_ledger["observed_mask"].astype(bool)].copy()
    if val.empty:
        raise ValueError("validation ledger has no observed rows")
    release_times = sorted(pd.to_datetime(val["release_time"]).unique())
    release_meta = val[["forecast_id", "release_time"]].drop_duplicates("forecast_id").copy()
    scored_validation = score_archive_rows(val, archive, config).merge(release_meta, on="forecast_id", how="left")
    scored_validation["release_time"] = pd.to_datetime(scored_validation["release_time"])
    n_models = max(len(registry), 1)
    n_families = max(registry["family"].astype(str).nunique(), 1)
    target_model_ess = max(1.0, min(float(target_ess_fraction) * n_models, n_models))
    target_family_ess = max(1.0, min(float(target_ess_fraction) * n_families, n_families))
    reports = []
    for rho in grid:
        hp = initialize_hierarchical_weights(registry)
        cumulative_log_score = 0.0
        release_nll_values: list[float] = []
        min_family_ess = float("inf")
        min_model_ess = float("inf")
        family_ess_values: list[float] = []
        model_ess_values: list[float] = []
        top1_model_values: list[float] = []
        top1_family_values: list[float] = []
        update_times: list[object] = []
        n_releases = 0
        for rt in release_times:
            current = scored_validation[scored_validation["release_time"] == pd.Timestamp(rt)]
            batch_ledger = val[
                pd.to_datetime(val["release_time"]).eq(pd.Timestamp(rt))
            ].copy()
            availability = evidence_availability_by_model(
                current, batch_ledger, registry["model_id"].astype(str)
            )
            log_ev = pd.DataFrame([
                {
                    "release_time": pd.Timestamp(rt),
                    "model_id": str(model_id),
                    "log_evidence": compute_log_evidence(current, model_id=str(model_id))
                    if availability[str(model_id)] else 0.0,
                    "evidence_available": availability[str(model_id)],
                }
                for model_id in registry["model_id"].astype(str)
            ])
            predicted_model = induce_model_weights(hp.family_weights, hp.inner_weights)
            release_log_score = _prequential_mixture_log_score(current, predicted_model)
            cumulative_log_score += release_log_score
            release_nll_values.append(-float(release_log_score))
            hp = hierarchical_update_from_log_evidence(
                hp.family_weights,
                hp.inner_weights,
                log_ev,
                rho=float(rho),
            )
            summary = summarize_hierarchical_posterior(hp.model_weights, hp.family_weights)
            family_ess = float(summary["family_ess"])
            model_ess = float(summary["model_ess"])
            min_family_ess = min(min_family_ess, family_ess)
            min_model_ess = min(min_model_ess, model_ess)
            family_ess_values.append(family_ess)
            model_ess_values.append(model_ess)
            top1_model_values.append(float(hp.model_weights["weight"].astype(float).max()))
            top1_family_values.append(float(hp.family_weights["family_weight"].astype(float).max()))
            update_times.append(rt)
            n_releases += 1
        mean_validation_log_score = cumulative_log_score / max(n_releases, 1)
        validation_mixture_nll = -float(cumulative_log_score)
        validation_mixture_nll_se = (
            float(np.std(release_nll_values, ddof=1) * np.sqrt(len(release_nll_values)))
            if len(release_nll_values) > 1
            else 0.0
        )
        model_penalty_value = _mean_squared_ess_penalty(model_ess_values, target_model_ess)
        family_penalty_value = _mean_squared_ess_penalty(family_ess_values, target_family_ess)
        top1_model_penalty_value = _mean_squared_top1_penalty(top1_model_values, top1_target)
        top1_family_penalty_value = _mean_squared_top1_penalty(top1_family_values, top1_target)
        top1_penalty_value = float(top1_model_penalty_value + top1_family_penalty_value)
        regularization_penalty = (
            float(ess_penalty) * model_penalty_value
            + float(family_ess_penalty) * family_penalty_value
            + float(top1_penalty) * top1_penalty_value
        )
        objective = cumulative_log_score
        reports.append({
            "rho": float(rho),
            "validation_mixture_nll": validation_mixture_nll,
            "validation_mixture_nll_se": validation_mixture_nll_se,
            "validation_log_score": float(cumulative_log_score),
            "mean_validation_log_score": float(mean_validation_log_score),
            "ess_penalty": float(regularization_penalty),
            "model_ess_penalty": float(model_penalty_value),
            "family_ess_penalty": float(family_penalty_value),
            "top1_penalty": float(top1_penalty_value),
            "top1_model_penalty": float(top1_model_penalty_value),
            "top1_family_penalty": float(top1_family_penalty_value),
            "regularization_penalty": float(regularization_penalty),
            "objective": float(objective),
            "regularized_objective": float(mean_validation_log_score - regularization_penalty),
            "min_family_ess": float(min_family_ess if np.isfinite(min_family_ess) else n_families),
            "min_model_ess": float(min_model_ess if np.isfinite(min_model_ess) else n_models),
            "target_family_ess": float(target_family_ess),
            "target_model_ess": float(target_model_ess),
            "rho_model_ess_penalty_weight": float(ess_penalty),
            "rho_family_ess_penalty_weight": float(family_ess_penalty),
            "rho_top1_penalty_weight": float(top1_penalty),
            "top1_target": float(top1_target),
            "top1_model_weight_max": float(max(top1_model_values) if top1_model_values else 0.0),
            "top1_family_weight_max": float(max(top1_family_values) if top1_family_values else 0.0),
            "first_top1_ge_target_release_time": _first_top1_time(update_times, top1_model_values, top1_target),
            "first_family_top1_ge_target_release_time": _first_top1_time(update_times, top1_family_values, top1_target),
            "n_releases": int(n_releases),
            "prior_policy": "uniform_family_uniform_inner",
            "rho_selection_variant": "hierarchical",
            "rho_selection_mode": "validation_grid",
            "rho_selection_objective": "prequential_untempered_validation_mixture_nll",
            "filter_dynamics": "hierarchical_bayesian_evidence_update",
            "rho_regularization_used_for_selection": False,
            "predictive_density_tempered": False,
            "mixture_scored_before_update": True,
        })
    return _mark_selected_rho(pd.DataFrame(reports))


def selected_rho(report: pd.DataFrame) -> float:
    if report.empty:
        raise ValueError("temperature report is empty")
    selected = report[report.get("selected", False).astype(bool)] if "selected" in report else report.head(1)
    if selected.empty:
        selected = report.head(1)
    return float(selected.iloc[0]["rho"])


def fit_negative_binomial_bridge_config(
    validation_ledger: pd.DataFrame,
    archive: pd.DataFrame,
    *,
    min_dispersion: float = 1.0,
    default_dispersion: float = 20.0,
) -> BridgeConfig:
    ""





    joined = archive.merge(
        validation_ledger[["forecast_id", "observed_value", "observed_mask"]],
        on="forecast_id",
        how="inner",
    )
    joined = joined[joined["observed_mask"].astype(bool)].copy()
    dispersion: dict[str, float] = {}
    for comp, g in joined.groupby("component"):
        y = np.maximum(g["observed_value"].astype(float).to_numpy(), 0.0)
        mu = np.maximum(g["pred_mean"].astype(float).to_numpy(), 1e-6)
                                                                
        residual_var = np.square(y - mu)
        denom = np.maximum(residual_var - mu, 1e-6)
        r = np.median(np.square(mu) / denom)
        if not np.isfinite(r):
            r = default_dispersion
        dispersion[str(comp)] = max(float(r), float(min_dispersion))
    return BridgeConfig(
        distribution="negative_binomial",
        transform="identity",
        sigma_by_component=dispersion,
        default_sigma=float(default_dispersion),
        min_scale=1e-3,
    )
