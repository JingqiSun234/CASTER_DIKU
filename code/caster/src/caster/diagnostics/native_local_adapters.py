from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import math

import numpy as np
import pandas as pd

from caster.diagnostics.native_likelihood_contract import (
    NativeLikelihoodContractError,
    NativeLikelihoodDiagnosticMixin,
)
from caster.diagnostics.native_sidecar import (
    RETENTION_PERSIST_MINIMAL,
    RETENTION_UNAVAILABLE,
    SCHEMA_VERSION,
    STATUS_BLOCKED,
    STATUS_BLOCKED_NO_PANEL_LEDGER,
    STATUS_BLOCKED_SCALE_MISMATCH,
    STATUS_DETERMINISTIC_NO_NATIVE,
    STATUS_TRUE_NATIVE,
    STATUS_UNAVAILABLE,
    NativeAvailabilityRecord,
    NativeSidecarRecord,
    NativeSidecarStorageValidationRow,
    apply_retention_policy,
    write_minimal_sidecar,
)


LOCAL_NATIVE_MODEL_IDS = [
    "sir_tau",
    "seir_tau",
    "seirs_tau",
    "tv_seir_rt",
    "renewal_rt",
    "local_level",
    "covariate_dynamic_linear_trend",
    "particle_local_level",
    "drift",
    "covariate_drift",
    "rnn_simple",
    "gru_style",
    "lstm_style",
]

NATIVE_CAPABLE_MODEL_IDS = {
    "sir_tau",
    "seir_tau",
    "seirs_tau",
    "tv_seir_rt",
    "renewal_rt",
    "local_level",
    "particle_local_level",
}

STOCHASTIC_COMPARTMENTAL_MODEL_IDS = {"sir_tau", "seir_tau", "seirs_tau", "tv_seir_rt"}
STOCHASTIC_TRANSITION_MODES = {"stochastic", "stochastic_binomial", "binomial", "stochastic_tau_leap"}
DETERMINISTIC_TRANSITION_MODES = {"", "deterministic", "deterministic_mean_field", "mean_field"}

DETERMINISTIC_NO_NATIVE_MODEL_IDS = {
    "drift",
    "covariate_drift",
    "covariate_dynamic_linear_trend",
    "rnn_simple",
    "gru_style",
}
UNAVAILABLE_MODEL_IDS = {"lstm_style"}

LOCAL_NATIVE_SIDECAR_SCHEMA = "local_adapter_native_distribution.v1"


class NativeLikelihoodUnavailable(RuntimeError):
    ""

    def __init__(self, model_id: str, status: str, blocker: str) -> None:
        super().__init__(blocker)
        self.model_id = str(model_id)
        self.status = str(status)
        self.blocker = str(blocker)


@dataclass
class NativeSidecarBuildResult:
    ""

    payload: dict[str, Any]
    storage_row: NativeSidecarStorageValidationRow
    record: NativeSidecarRecord


class LocalNativeLikelihoodAdapter(NativeLikelihoodDiagnosticMixin):
    ""

    supports_native_log_likelihood = True
    native_sidecar_required = True
    native_sidecar_schema = {
        "schema": LOCAL_NATIVE_SIDECAR_SCHEMA,
        "distribution_params": "Origin-time h-step predictive distribution parameters derived from pre-origin panel history.",
        "target_scale": "count or adapter-declared continuous scale; count likelihoods reject per100k/rate targets without exposure conversion.",
    }

    def __init__(self, adapter: object) -> None:
        model_id = str(getattr(adapter, "model_id", ""))
        if model_id not in NATIVE_CAPABLE_MODEL_IDS:
            raise NativeLikelihoodUnavailable(
                model_id,
                _unsupported_status(model_id),
                _unsupported_blocker(model_id),
            )
        self.adapter = adapter
        self.model_id = model_id
        self.family = str(getattr(adapter, "family", "unknown"))
        self.native_likelihood_type = _native_likelihood_type(model_id)

    def initialize(self, panel: pd.DataFrame, seed: int = 0) -> Any:
        return self.adapter.initialize(panel, seed=seed)

    def transition(self, state: Any, forecast_origin: pd.Timestamp) -> Any:
        return self.adapter.transition(state, forecast_origin)

    def forecast_ledger(self, state: Any, ledger: pd.DataFrame) -> pd.DataFrame:
        return self.adapter.forecast_ledger(state, ledger)

    def forecast_draws(self, state: Any, ledger: pd.DataFrame, n_draws: int, seed: int = 0) -> pd.DataFrame:
        return self.adapter.forecast_draws(state, ledger, n_draws, seed=seed)

    def serialize_state(self, state: Any) -> dict[str, Any]:
        return self.adapter.serialize_state(state)

    def save_native_state_sidecar(
        self,
        *,
        panel: pd.DataFrame,
        ledger_row: pd.Series | Mapping[str, Any],
        out_root: str | Path,
    ) -> NativeSidecarBuildResult:
        ""

        payload = build_origin_time_native_payload(self.adapter, panel=panel, ledger_row=ledger_row)
        origin = _origin_label(payload)
        record = write_minimal_sidecar(
            out_root=out_root,
            model_id=self.model_id,
            origin=origin,
            payload=payload,
            sidecar_type="native_distribution_params",
            retention_policy=RETENTION_PERSIST_MINIMAL,
            score_reproducibility_level="origin_time_distribution_params",
        )
        payload = dict(payload)
        payload["artifact_path"] = record.manifest_path
        payload["sidecar_hash"] = record.sidecar_hash
        return NativeSidecarBuildResult(payload=payload, storage_row=record.to_storage_validation_row(), record=record)

    def load_native_state_sidecar(self, path: str | Path) -> dict[str, Any]:
        ""

        p = Path(path)
        doc = json.loads(p.read_text(encoding="utf-8"))
        if "artifact_path" in doc and "payload" not in doc:
            payload_path = Path(doc["artifact_path"])
            payload_doc = json.loads(payload_path.read_text(encoding="utf-8"))
            payload = dict(payload_doc["payload"])
            payload["artifact_path"] = str(p)
            payload["sidecar_hash"] = str(doc.get("sidecar_hash", ""))
            return payload
        if "payload" in doc:
            return dict(doc["payload"])
        return doc

    def log_likelihood(self, y: Any, context: Any, sidecar: Mapping[str, Any]) -> float:
        ""

        self._require_native_likelihood_diagnostic_enabled()
        if str(sidecar.get("model_id", "")) != self.model_id:
            raise NativeLikelihoodContractError("native sidecar model_id does not match adapter")
        ctx = context if isinstance(context, Mapping) else {}
        for key in ("forecast_id", "component", "horizon"):
            if key in ctx and str(ctx.get(key, "")) != str(sidecar.get(key, "")):
                if key == "horizon":
                    try:
                        if int(ctx.get(key)) == int(sidecar.get(key)):
                            continue
                    except Exception:
                        pass
                raise NativeLikelihoodContractError(f"native sidecar {key} does not match scoring context")
        params = sidecar.get("distribution_params")
        if not isinstance(params, Mapping):
            raise NativeLikelihoodContractError("native sidecar missing distribution_params")
        likelihood_type = str(sidecar.get("native_likelihood_type", ""))
        if likelihood_type in {"poisson_count", "negative_binomial_count"}:
            if str(sidecar.get("score_scale", sidecar.get("target_scale", ""))) != "count":
                raise NativeLikelihoodContractError("count likelihood requires score_scale=count")
            if str(sidecar.get("target_scale", "")) != "count" and not sidecar.get("observed_value_count_source"):
                raise NativeLikelihoodContractError("rate target count likelihood requires explicit count conversion provenance")
            score = _score_count_distribution(float(y), likelihood_type, params)
        elif likelihood_type in {"gaussian_dlm_count_scale", "particle_gaussian_mixture_count_scale"}:
            if str(sidecar.get("target_scale", "")) in {"per100k_or_rate", "rate_per100k", "continuous_rate"}:
                raise NativeLikelihoodContractError("count-scale gaussian likelihood cannot score rate target")
            score = _score_continuous_distribution(float(y), likelihood_type, params)
        elif likelihood_type in {"gaussian_target_scale_adapter_native", "log1p_gaussian_target_scale_adapter_native"}:
            if str(sidecar.get("score_scale", "")) != "target":
                raise NativeLikelihoodContractError("target-scale native likelihood requires score_scale=target")
            score = _score_target_scale_distribution(float(y), likelihood_type, params)
        else:
            raise NativeLikelihoodContractError(f"unsupported native_likelihood_type={likelihood_type!r}")
        if not math.isfinite(score):
            raise NativeLikelihoodContractError("adapter-native log_likelihood must be finite")
        return float(score)


def make_native_likelihood_adapter(adapter: object) -> LocalNativeLikelihoodAdapter:
    ""

    return LocalNativeLikelihoodAdapter(adapter)


def build_origin_time_native_payload(
    adapter: object,
    *,
    panel: pd.DataFrame,
    ledger_row: pd.Series | Mapping[str, Any],
) -> dict[str, Any]:
    ""

    row = ledger_row.to_dict() if isinstance(ledger_row, pd.Series) else dict(ledger_row)
    model_id = str(getattr(adapter, "model_id", row.get("model_id", "")))
    if model_id not in NATIVE_CAPABLE_MODEL_IDS:
        raise NativeLikelihoodUnavailable(model_id, _unsupported_status(model_id), _unsupported_blocker(model_id))
    target_scale = detect_target_scale(row)
    likelihood_type = _native_likelihood_type(model_id, target_scale=target_scale)
    normalised = _normalise_panel(panel)
    origin = pd.Timestamp(row["forecast_origin"])
    entity_id = _row_entity(row)
    component = str(row["component"])
    times, values = _series_from_panel(normalised, entity_id, component, origin)
    if len(times) and pd.Timestamp(times[-1]) > origin:
        raise NativeLikelihoodUnavailable(model_id, STATUS_BLOCKED, "pre-origin history filter failed")
    horizon = max(1, int(pd.to_numeric(row.get("horizon", 1), errors="raise")))
    params = _distribution_params_for_model(adapter, values, horizon, ledger_row=row)
    if likelihood_type in {"gaussian_target_scale_adapter_native", "log1p_gaussian_target_scale_adapter_native"}:
        params = _target_scale_params_from_native_params(params)
    return {
        "schema_version": SCHEMA_VERSION,
        "native_sidecar_schema": LOCAL_NATIVE_SIDECAR_SCHEMA,
        "model_id": model_id,
        "family": str(getattr(adapter, "family", "")),
        "origin": _origin_label_from_row(row),
        "forecast_id": str(row.get("forecast_id", "")),
        "entity_id": entity_id,
        "component": component,
        "forecast_origin": str(row["forecast_origin"]),
        "target_time": str(row.get("target_time", "")),
        "horizon": horizon,
        "features_available_until": str(row["forecast_origin"]),
        "target_scale": target_scale,
        "score_scale": "count" if likelihood_type in {"poisson_count", "negative_binomial_count"} else "target",
        "native_likelihood_type": likelihood_type,
        "transition_mode": str(params.get("transition_mode", "")),
        "n_simulations": int(params.get("n_simulations", 0) or 0),
        "rate_process": str(params.get("rate_process", "")),
        "log_beta_rw_scale": float(params.get("log_beta_rw_scale", 0.0) or 0.0),
        "log_gamma_rw_scale": float(params.get("log_gamma_rw_scale", 0.0) or 0.0),
        "distribution_params": params,
        "history_length": int(len(values)),
        "history_last_time": str(pd.Timestamp(times[-1])) if len(times) else "",
        "history_source": "pre_origin_panel_history",
        "uses_archive_predictions": False,
        "uses_bridge_scale": False,
        "uses_forecast_ledger_output": False,
        "no_bridge_sigma": True,
        "archive_prediction_variance_used": False,
        "variance_source": str(params.get("variance_source", "adapter_internal_native")),
        "simulation_basis": str(params.get("simulation_basis", "")),
        "uses_mean_field_fallback": bool(params.get("uses_mean_field_fallback", False)),
                                                                             
                                                                        
                                                                         
                                                                      
                                                                             
                                          
        "posterior_predictive_log_density_available": False,
        "posterior_predictive_schema": "",
        "posterior_predictive_kind": "origin_conditioned_predictive_approximation",
        "posterior_predictive_asof_origin": True,
        "posterior_predictive_is_model_native": False,
        "posterior_predictive_integrates_parameter_or_state_uncertainty": False,
        "posterior_predictive_fallback_reason": (
            "adapter-defined moment/simulation approximation does not satisfy "
            "strict_posterior_predictive.v1"
        ),
    }


def save_origin_time_native_sidecar(
    adapter: object,
    *,
    panel: pd.DataFrame,
    ledger_row: pd.Series | Mapping[str, Any],
    out_root: str | Path,
) -> NativeSidecarBuildResult:
    ""

    return make_native_likelihood_adapter(adapter).save_native_state_sidecar(
        panel=panel,
        ledger_row=ledger_row,
        out_root=out_root,
    )


def build_local_native_availability(
    registry: pd.DataFrame,
    *,
    panel: pd.DataFrame | None = None,
    ledger: pd.DataFrame | None = None,
    out_root: str | Path | None = None,
    registry_source: str = "",
    registry_blocker: str = "",
) -> tuple[list[NativeAvailabilityRecord], list[NativeSidecarStorageValidationRow]]:
    ""

    from caster.models import instantiate_adapter_from_row

    rows: list[NativeAvailabilityRecord] = []
    storage_rows: list[NativeSidecarStorageValidationRow] = []
    registry = registry.copy()
    by_id = {str(r["model_id"]): r for _, r in registry.iterrows() if str(r.get("model_id", "")) in LOCAL_NATIVE_MODEL_IDS}
    ledger_row = ledger.iloc[0].to_dict() if ledger is not None and not ledger.empty else None
    for model_id in LOCAL_NATIVE_MODEL_IDS:
        origin = "registry"
        extra_blocker = "; ".join(x for x in [registry_blocker] if x)
        if model_id not in by_id:
            rows.append(_availability_record(model_id, origin, STATUS_BLOCKED, "missing local model_id in registry", registry_source, extra_blocker))
            continue
        if model_id in DETERMINISTIC_NO_NATIVE_MODEL_IDS or model_id in UNAVAILABLE_MODEL_IDS:
            status = _unsupported_status(model_id)
            rows.append(_availability_record(model_id, origin, status, _unsupported_blocker(model_id), registry_source, extra_blocker))
            continue
        if panel is None or ledger_row is None or out_root is None:
            rows.append(
                _availability_record(
                    model_id,
                    origin,
                    STATUS_BLOCKED_NO_PANEL_LEDGER,
                    "panel+ledger are required to prove an origin-time h-step native sidecar",
                    registry_source,
                    extra_blocker,
                )
            )
            continue
        try:
            adapter = instantiate_adapter_from_row(by_id[model_id])
            result = save_origin_time_native_sidecar(adapter, panel=panel, ledger_row=ledger_row, out_root=out_root)
            storage_rows.append(result.storage_row)
            rows.append(
                NativeAvailabilityRecord(
                    model_id=model_id,
                    origin=origin,
                    supports_native_log_likelihood=True,
                    native_likelihood_type=str(result.payload["native_likelihood_type"]),
                    native_likelihood_status=STATUS_TRUE_NATIVE,
                    native_sidecar_required=True,
                    native_sidecar_schema=LOCAL_NATIVE_SIDECAR_SCHEMA,
                    retention_policy=RETENTION_PERSIST_MINIMAL,
                    artifact_path=result.record.manifest_path,
                    blocker=extra_blocker,
                )
            )
        except NativeLikelihoodUnavailable as exc:
            rows.append(_availability_record(model_id, origin, exc.status, exc.blocker, registry_source, extra_blocker))
        except Exception as exc:
            rows.append(_availability_record(model_id, origin, STATUS_BLOCKED, f"{type(exc).__name__}: {exc}", registry_source, extra_blocker))
    return rows, storage_rows


def detect_target_scale(row: Mapping[str, Any]) -> str:
    ""

    for key in ("target_scale", "score_scale", "scale", "unit", "target_unit"):
        text = str(row.get(key, "")).strip().lower()
        if not text:
            continue
        if "count" in text:
            return "count"
        if any(token in text for token in ("per100k", "per_100k", "per-100k", "rate", "percent", "proportion")):
            return "per100k_or_rate"
    component = str(row.get("component", "")).strip().lower()
    if any(token in component for token in ("per100k", "per_100k", "per-100k", "rate", "percent", "proportion")):
        return "per100k_or_rate"
    return "count"


def _nb_overdispersion_from_mean_var(mean: float, variance: float, *, floor: float = 1e-6) -> float:
    mu = max(float(mean), 0.0)
    var = max(float(variance), 0.0)
    if mu <= 1e-12:
        return float(floor)
    return float(max((var - mu) / max(mu * mu, 1e-12), floor))


def _stochastic_compartmental_params(adapter: object, values: np.ndarray, horizon: int) -> dict[str, Any]:
    model_id = str(getattr(adapter, "model_id", ""))
    mode = str(getattr(adapter, "transition_mode", "")).strip().lower()
    if mode != "stochastic_binomial":
        raise NativeLikelihoodUnavailable(
            model_id,
            STATUS_BLOCKED,
            f"{model_id} native sidecar requires transition_mode=stochastic_binomial, got {mode or 'missing'}",
        )
    if not hasattr(adapter, "forecast_one"):
        raise NativeLikelihoodUnavailable(
            model_id,
            STATUS_BLOCKED,
            f"{model_id} native sidecar requires stochastic forecast_one; mean-field fallback is disabled",
        )
    try:
        mean, variance = adapter.forecast_one(np.asarray(values, dtype=float), int(horizon))
    except Exception as exc:
        raise NativeLikelihoodUnavailable(
            model_id,
            STATUS_BLOCKED,
            f"{model_id} stochastic forecast_one failed; mean-field fallback is disabled: {type(exc).__name__}: {exc}",
        ) from exc
    mean = max(float(mean), 0.0)
    variance = max(float(variance), max(mean, 1.0))
    log_beta_rw_scale = max(float(getattr(adapter, "log_beta_rw_scale", 0.0)), 0.0)
    log_gamma_rw_scale = max(float(getattr(adapter, "log_gamma_rw_scale", 0.0)), 0.0)
    rate_process = (
        "independent_gaussian_log_random_walk"
        if log_beta_rw_scale > 0.0 or log_gamma_rw_scale > 0.0
        else "fixed_from_forecast_origin"
    )
    return {
        "mean": mean,
        "variance": variance,
        "overdispersion": _nb_overdispersion_from_mean_var(mean, variance),
        "overdispersion_source": "moment_matched_from_stochastic_binomial_summary",
        "history_formula": f"pre_origin_{model_id}",
        "transition_mode": str(getattr(adapter, "transition_mode", "")),
        "n_simulations": int(getattr(adapter, "n_simulations", 0) or 0),
        "rate_process": rate_process,
        "log_beta_rw_scale": log_beta_rw_scale,
        "log_gamma_rw_scale": log_gamma_rw_scale,
        "rate_process_parameter_selection": False,
        "simulation_basis": "stochastic_binomial_forecast_one",
        "variance_source": "stochastic_binomial_rate_process_simulator",
        "uses_mean_field_fallback": False,
    }


def _deterministic_compartmental_params(
    adapter: object,
    values: np.ndarray,
    horizon: int,
    *,
    exposed: bool,
    seirs: bool,
    tv: bool,
    overdispersion: float,
    formula: str,
) -> dict[str, Any]:
    model_id = str(getattr(adapter, "model_id", ""))
    mean = _sir_mean(adapter, values, horizon, exposed=exposed, seirs=seirs, tv=tv)
    mean = max(float(mean), 0.0)
    overdispersion = max(float(overdispersion), 1e-9)
    variance = max(mean + overdispersion * mean * mean, 1.0)
    return {
        "mean": mean,
        "variance": variance,
        "overdispersion": overdispersion,
        "overdispersion_source": "fixed_deterministic_mean_field",
        "history_formula": formula,
        "transition_mode": "deterministic_mean_field",
        "n_simulations": 0,
        "simulation_basis": "deterministic_mean_field",
        "variance_source": "deterministic_mean_field_fixed_negative_binomial_overdispersion",
        "uses_mean_field_fallback": False,
    }


def _compartmental_params(
    adapter: object,
    values: np.ndarray,
    horizon: int,
    *,
    exposed: bool,
    seirs: bool,
    tv: bool,
    overdispersion: float,
    formula: str,
) -> dict[str, Any]:
    model_id = str(getattr(adapter, "model_id", ""))
    mode = str(getattr(adapter, "transition_mode", "stochastic_binomial")).strip().lower()
    if mode in STOCHASTIC_TRANSITION_MODES:
        return _stochastic_compartmental_params(adapter, values, horizon)
    if mode not in DETERMINISTIC_TRANSITION_MODES:
        raise NativeLikelihoodUnavailable(
            model_id,
            STATUS_BLOCKED,
            f"{model_id} native sidecar has unsupported transition_mode={mode!r}; "
            "use deterministic_mean_field or explicit stochastic_binomial",
        )
    return _deterministic_compartmental_params(
        adapter,
        values,
        horizon,
        exposed=exposed,
        seirs=seirs,
        tv=tv,
        overdispersion=overdispersion,
        formula=formula,
    )


def _distribution_params_for_model(
    adapter: object,
    values: np.ndarray,
    horizon: int,
    *,
    ledger_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_id = str(getattr(adapter, "model_id", ""))
    if model_id == "sir_tau":
        return _compartmental_params(
            adapter,
            values,
            horizon,
            exposed=False,
            seirs=False,
            tv=False,
            overdispersion=0.10,
            formula="pre_origin_sir_mean_field",
        )
    if model_id == "seir_tau":
        return _compartmental_params(
            adapter,
            values,
            horizon,
            exposed=True,
            seirs=False,
            tv=False,
            overdispersion=0.12,
            formula="pre_origin_seir_mean_field",
        )
    if model_id == "seirs_tau":
        return _compartmental_params(
            adapter,
            values,
            horizon,
            exposed=True,
            seirs=True,
            tv=False,
            overdispersion=0.12,
            formula="pre_origin_seirs_mean_field",
        )
    if model_id == "tv_seir_rt":
        return _compartmental_params(
            adapter,
            values,
            horizon,
            exposed=True,
            seirs=False,
            tv=True,
            overdispersion=0.15,
            formula="pre_origin_tv_seir_mean_field",
        )
    if model_id == "renewal_rt":
        mean = _renewal_mean(adapter, values, horizon)
        return {"mean": mean, "overdispersion": 0.08, "history_formula": "pre_origin_renewal_rt"}
    if model_id == "local_level":
        mean, variance = _local_level_params(adapter, values, horizon, dynamic_trend=False)
        return {"mean": mean, "variance": variance, "history_formula": "pre_origin_local_level_dlm"}
    if model_id == "particle_local_level":
        return _particle_local_level_params(adapter, values, horizon)
    raise NativeLikelihoodUnavailable(model_id, _unsupported_status(model_id), _unsupported_blocker(model_id))


def _forecast_one_mean_var(adapter: object, values: np.ndarray, horizon: int, *, fallback: Any) -> tuple[float, float | None]:
    if hasattr(adapter, "forecast_one"):
        try:
            mean, variance = adapter.forecast_one(np.asarray(values, dtype=float), int(horizon))
            return max(float(mean), 0.0), max(float(variance), 0.0)
        except Exception:
            pass
    mean, variance = fallback()
    return max(float(mean), 0.0), None if variance is None else max(float(variance), 0.0)


def _native_likelihood_type(model_id: str, *, target_scale: str = "count") -> str:
    if target_scale != "count":
        return "gaussian_target_scale_adapter_native"
    if model_id in {"sir_tau", "seir_tau", "seirs_tau", "tv_seir_rt", "renewal_rt"}:
        return "negative_binomial_count"
    if model_id in {"local_level"}:
        return "gaussian_dlm_count_scale"
    if model_id == "particle_local_level":
        return "particle_gaussian_mixture_count_scale"
    return "none"


def _score_count_distribution(y: float, likelihood_type: str, params: Mapping[str, Any]) -> float:
    if not math.isfinite(y) or y < 0.0:
        raise NativeLikelihoodContractError("count likelihood requires nonnegative finite y")
    if abs(y - round(y)) > 1e-8:
        raise NativeLikelihoodContractError("count likelihood requires integer count y")
    count = float(round(y))
    mean = max(float(params.get("mean", 0.0)), 1e-12)
    if likelihood_type == "poisson_count":
        return count * math.log(mean) - mean - math.lgamma(count + 1.0)
    overdispersion = max(float(params.get("overdispersion", 0.0)), 1e-9)
    r = 1.0 / overdispersion
    p = r / (r + mean)
    return (
        math.lgamma(count + r)
        - math.lgamma(r)
        - math.lgamma(count + 1.0)
        + r * math.log(max(p, 1e-15))
        + count * math.log(max(1.0 - p, 1e-15))
    )


def _score_continuous_distribution(y: float, likelihood_type: str, params: Mapping[str, Any]) -> float:
    if likelihood_type == "gaussian_dlm_count_scale":
        mean = float(params.get("mean", 0.0))
        variance = max(float(params.get("variance", 1.0)), 1e-9)
        return -0.5 * (math.log(2.0 * math.pi * variance) + ((float(y) - mean) ** 2) / variance)
    particles = np.asarray(params.get("particles", []), dtype=float)
    weights = np.asarray(params.get("weights", []), dtype=float)
    kernel_variance = max(float(params.get("kernel_variance", 1.0)), 1e-9)
    if particles.size == 0 or weights.size != particles.size or not np.isfinite(particles).all() or not np.isfinite(weights).all():
        raise NativeLikelihoodContractError("particle sidecar requires finite particles and weights")
    weights = weights / max(float(weights.sum()), 1e-12)
    log_terms = np.log(np.maximum(weights, 1e-300)) - 0.5 * (
        math.log(2.0 * math.pi * kernel_variance) + ((float(y) - particles) ** 2) / kernel_variance
    )
    return float(_logsumexp(log_terms))


def _target_scale_params_from_native_params(params: Mapping[str, Any]) -> dict[str, Any]:
    if "particles" in params:
        particles = np.asarray(params.get("particles", []), dtype=float)
        weights = np.asarray(params.get("weights", []), dtype=float)
        if particles.size and weights.size == particles.size and np.isfinite(particles).all() and np.isfinite(weights).all():
            weights = weights / max(float(weights.sum()), 1e-12)
            mean = float(np.sum(weights * particles))
            variance = float(np.sum(weights * (particles - mean) ** 2) + float(params.get("kernel_variance", 1.0)))
        else:
            mean, variance = 0.0, 1.0
    else:
        mean = float(params.get("mean", 0.0))
        if "variance" in params:
            variance = float(params.get("variance", 1.0))
        else:
            overdispersion = max(float(params.get("overdispersion", 0.1)), 1e-9)
            variance = max(mean + overdispersion * mean * mean, 1.0)
    floor = max(1e-6, 1e-4 * max(abs(mean), 1.0))
    out = dict(params)
    out["mean"] = float(mean)
    out["variance"] = float(max(variance, floor))
    out["variance_floor"] = float(floor)
    out["variance_source"] = "adapter_internal_native"
    return out


def _score_target_scale_distribution(y: float, likelihood_type: str, params: Mapping[str, Any]) -> float:
    if not math.isfinite(float(y)):
        raise NativeLikelihoodContractError("target-scale likelihood requires finite y")
    if likelihood_type == "log1p_gaussian_target_scale_adapter_native":
        if float(y) < -1.0:
            raise NativeLikelihoodContractError("log1p target-scale likelihood requires y >= -1")
        mean = math.log1p(max(float(params.get("mean", 0.0)), -0.999999))
        variance = max(float(params.get("log1p_variance", params.get("variance", 1.0))), 1e-9)
        yy = math.log1p(float(y))
    else:
        mean = float(params.get("mean", 0.0))
        variance = max(float(params.get("variance", 1.0)), 1e-9)
        yy = float(y)
    return -0.5 * (math.log(2.0 * math.pi * variance) + ((yy - mean) ** 2) / variance)


def _sir_mean(adapter: object, values: np.ndarray, horizon: int, *, exposed: bool, seirs: bool, tv: bool) -> float:
    y = np.maximum(np.asarray(values, dtype=float), 0.0)
    if len(y) == 0:
        return 0.0
    population = max(float(getattr(adapter, "population", 1_000_000.0)), 1.0)
    gamma = max(float(getattr(adapter, "gamma", 0.2)), 0.0)
    sigma = max(float(getattr(adapter, "sigma", 1.0 / 3.0)), 0.0)
    reporting_rate = max(float(getattr(adapter, "reporting_rate", 0.04)), 1e-6)
    last_obs = float(y[-1])
    i = min(max(last_obs / reporting_rate, 1.0), 0.20 * population if exposed else 0.25 * population)
    e = min(0.7 * i, 0.20 * population) if exposed else 0.0
    r = min(0.05 * population + float(np.nansum(y[-min(len(y), 21):])) / reporting_rate, 0.80 * population)
    s = max(population - e - i - r, 0.0)
    rt0 = _estimate_rt(y, gamma)
    recent_growth = float(np.nanmean(np.diff(np.log(np.maximum(y[-min(len(y), 8):], 1.0))))) if len(y) >= 3 else 0.0
    incidence = 0.0
    for step in range(max(1, int(horizon))):
        rt = float(np.clip(1.0 + (rt0 - 1.0) * math.exp(-0.20 * step) + 0.15 * recent_growth, 0.2, 3.5)) if tv else rt0
        beta = rt * gamma
        new_exp = min(s, s * (1.0 - math.exp(-beta * i / population)))
        if exposed:
            new_inf = min(e, e * (1.0 - math.exp(-sigma)))
            waned = min(r, r * (1.0 - math.exp(-float(getattr(adapter, "waning_rate", 1.0 / 180.0))))) if seirs else 0.0
            new_rec = min(i, i * (1.0 - math.exp(-gamma)))
            s = max(s + waned - new_exp, 0.0)
            e = max(e + new_exp - new_inf, 0.0)
            i = max(i + new_inf - new_rec, 0.0)
            r = max(r + new_rec - waned, 0.0)
            incidence = new_inf
        else:
            new_inf = new_exp
            new_rec = min(i, i * (1.0 - math.exp(-gamma)))
            s = max(s - new_inf, 0.0)
            i = max(i + new_inf - new_rec, 0.0)
            r = max(r + new_rec, 0.0)
            incidence = new_inf
    return max(0.0, reporting_rate * incidence)


def _renewal_mean(adapter: object, values: np.ndarray, horizon: int) -> float:
    hist = list(np.maximum(np.asarray(values, dtype=float), 0.0))
    if not hist:
        return 0.0
    w = np.asarray(getattr(adapter, "serial_interval", (0.05, 0.15, 0.30, 0.25, 0.15, 0.07, 0.03)), dtype=float)
    w = w / max(float(w.sum()), 1e-12)
    rt0 = _renewal_rt(np.asarray(hist), w)
    rt_shrink = float(getattr(adapter, "rt_shrink", 0.85))
    mean = hist[-1]
    for step in range(max(1, int(horizon))):
        tail = np.asarray(hist[-len(w):][::-1], dtype=float)
        ww = w[: len(tail)]
        ww = ww / max(float(ww.sum()), 1e-12)
        rt = 1.0 + (rt0 - 1.0) * (rt_shrink ** step)
        mean = max(0.0, rt * float(np.dot(ww, tail)))
        hist.append(mean)
    return max(0.0, float(mean))


def _local_level_params(adapter: object, values: np.ndarray, horizon: int, *, dynamic_trend: bool) -> tuple[float, float]:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return 0.0, 1.0
    alpha = float(getattr(adapter, "alpha", 0.35))
    level = float(y[0])
    residuals: list[float] = []
    if dynamic_trend:
        beta = float(getattr(adapter, "beta", 0.10))
        damping = float(getattr(adapter, "damping", 0.90))
        trend = 0.0
        for val in y[1:]:
            pred = level + trend
            residuals.append(float(val) - pred)
            new_level = alpha * float(val) + (1.0 - alpha) * pred
            trend = beta * (new_level - level) + (1.0 - beta) * trend
            level = new_level
        mean = level + sum((damping ** i) * trend for i in range(1, int(horizon) + 1))
    else:
        for val in y[1:]:
            residuals.append(float(val) - level)
            level = alpha * float(val) + (1.0 - alpha) * level
        mean = level
    variance = max(float(np.var(residuals)) if residuals else 1.0, 1.0) * max(1, int(horizon))
    return max(0.0, float(mean)), float(variance)


def _particle_local_level_params(adapter: object, values: np.ndarray, horizon: int) -> dict[str, Any]:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return {"particles": [0.0], "weights": [1.0], "kernel_variance": 1.0, "history_formula": "pre_origin_particle_local_level"}
    n_particles = int(getattr(adapter, "n_particles", 128))
    process_scale = float(getattr(adapter, "process_scale", 0.20))
    obs_scale = float(getattr(adapter, "obs_scale", 1.0))
    rng = np.random.default_rng(17 + len(y) + int(horizon))
    base_sigma = math.sqrt(max(_residual_var(y, float(y[-1])), 1e-6))
    particles = rng.normal(float(y[0]), base_sigma, size=n_particles)
    weights = np.ones(n_particles) / n_particles
    for obs in y[1:]:
        particles = particles + rng.normal(0.0, process_scale * base_sigma, size=n_particles)
        ll = -0.5 * ((float(obs) - particles) / max(obs_scale * base_sigma, 1e-6)) ** 2
        ll -= float(np.max(ll))
        weights = np.exp(ll)
        weights = weights / max(float(weights.sum()), 1e-12)
        ess = 1.0 / max(float(np.sum(weights * weights)), 1e-12)
        if ess < n_particles / 2:
            idx = rng.choice(n_particles, size=n_particles, replace=True, p=weights)
            particles = particles[idx]
            weights = np.ones(n_particles) / n_particles
    for _ in range(max(1, int(horizon))):
        particles = particles + rng.normal(0.0, process_scale * base_sigma, size=n_particles)
    return {
        "particles": [max(0.0, float(x)) for x in particles],
        "weights": [float(x) for x in weights],
        "kernel_variance": float(max((obs_scale * base_sigma) ** 2, 1e-6)),
        "history_formula": "pre_origin_particle_local_level",
    }


def _availability_record(
    model_id: str,
    origin: str,
    status: str,
    blocker: str,
    registry_source: str = "",
    extra_blocker: str = "",
) -> NativeAvailabilityRecord:
    blocker_text = "; ".join(x for x in [blocker, extra_blocker] if x)
    return NativeAvailabilityRecord(
        model_id=model_id,
        origin=origin,
        supports_native_log_likelihood=False,
        native_likelihood_type=_native_likelihood_type(model_id),
        native_likelihood_status=status,
        native_sidecar_required=model_id in NATIVE_CAPABLE_MODEL_IDS,
        native_sidecar_schema=LOCAL_NATIVE_SIDECAR_SCHEMA if model_id in NATIVE_CAPABLE_MODEL_IDS else "",
        retention_policy=RETENTION_UNAVAILABLE,
        artifact_path="",
        blocker=blocker_text if not registry_source else f"{blocker_text}; registry_source={registry_source}",
    )


def _unsupported_status(model_id: str) -> str:
    if model_id in DETERMINISTIC_NO_NATIVE_MODEL_IDS:
        return STATUS_DETERMINISTIC_NO_NATIVE
    if model_id in UNAVAILABLE_MODEL_IDS:
        return STATUS_UNAVAILABLE
    return STATUS_BLOCKED


def _unsupported_blocker(model_id: str) -> str:
    if model_id in DETERMINISTIC_NO_NATIVE_MODEL_IDS:
        return f"{model_id} currently has deterministic forecast dynamics but no explicit native observation likelihood sidecar"
    if model_id in UNAVAILABLE_MODEL_IDS:
        return f"{model_id} requires a fitted neural checkpoint/native sidecar that is not available in Phase 4"
    return f"{model_id} does not have a Phase 4 native likelihood wrapper"


def _normalise_panel(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    entity_col = next((c for c in ("entity_id", "jurisdiction", "region", "unit", "unique_id") if c in p.columns), None)
    p["__entity_id__"] = p[entity_col].astype(str) if entity_col else "global"
    time_col = next((c for c in ("week_end", "date", "ds", "time", "target_time") if c in p.columns), None)
    if time_col is None:
        raise ValueError("panel must contain one of week_end/date/ds/time/target_time")
    p["__time__"] = pd.to_datetime(p[time_col], errors="coerce")
    return p.dropna(subset=["__time__"]).sort_values(["__entity_id__", "__time__"]).reset_index(drop=True)


def _series_from_panel(panel: pd.DataFrame, entity_id: str, component: str, origin: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    subset = panel[(panel["__entity_id__"].astype(str) == str(entity_id)) & (panel["__time__"] <= pd.Timestamp(origin))].copy()
    if component in subset.columns:
        values = pd.to_numeric(subset[component], errors="coerce")
        times = subset["__time__"]
    elif "component" in subset.columns and ({"observed_value", "value"} & set(subset.columns)):
        value_col = "observed_value" if "observed_value" in subset.columns else "value"
        subset = subset[subset["component"].astype(str) == str(component)].copy()
        values = pd.to_numeric(subset[value_col], errors="coerce")
        times = subset["__time__"]
    else:
        return np.asarray([], dtype="datetime64[ns]"), np.asarray([], dtype=float)
    mask = np.isfinite(values.to_numpy(dtype=float))
    return times.to_numpy(dtype="datetime64[ns]")[mask], values.to_numpy(dtype=float)[mask]


def _row_entity(row: Mapping[str, Any]) -> str:
    for key in ("entity_id", "jurisdiction", "region", "unit", "unique_id"):
        if key in row and str(row[key]).strip():
            return str(row[key])
    return "global"


def _origin_label(payload: Mapping[str, Any]) -> str:
    return str(payload.get("origin") or f"{payload.get('forecast_origin', 'origin')}_{payload.get('forecast_id', 'forecast')}")


def _origin_label_from_row(row: Mapping[str, Any]) -> str:
    forecast_origin = str(row.get("forecast_origin", "origin")).replace(" ", "T")
    forecast_id = str(row.get("forecast_id", "forecast"))
    return f"{forecast_origin}__{forecast_id}"


def _estimate_rt(values: np.ndarray, gamma: float) -> float:
    y = np.maximum(np.asarray(values, dtype=float), 1.0)
    if len(y) < 3:
        return 1.0
    growth = np.diff(np.log(y[-min(len(y), 8):]))
    return float(np.clip(np.exp(float(np.nanmean(growth))) / max(1.0 - gamma, 1e-6), 0.2, 3.5))


def _renewal_rt(values: np.ndarray, weights: np.ndarray) -> float:
    hist = np.maximum(np.asarray(values, dtype=float), 0.0)
    ratios: list[float] = []
    for t in range(1, len(hist)):
        tail = hist[max(0, t - len(weights)):t][::-1]
        ww = weights[: len(tail)]
        denom = float(np.dot(tail, ww / max(float(ww.sum()), 1e-12)))
        if denom > 0:
            ratios.append(float(hist[t] / denom))
    if not ratios:
        return 1.0
    return float(np.clip(np.nanmedian(ratios[-min(6, len(ratios)):]), 0.2, 3.5))


def _residual_var(values: np.ndarray, pred: float, floor: float = 1.0) -> float:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) >= 3:
        diffs = np.diff(y)
        if np.isfinite(diffs).any():
            return float(max(np.nanvar(diffs), floor))
    return float(max(abs(pred), floor))


def _logsumexp(values: np.ndarray) -> float:
    m = float(np.max(values))
    return m + math.log(float(np.sum(np.exp(values - m))))
