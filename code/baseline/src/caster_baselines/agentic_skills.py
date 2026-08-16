from __future__ import annotations

import json
import math
import time
import hashlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data_validation import baseline_root, caster_root_from_baseline, resolve_manifest_path, sha256_file
from .ledger_runner import (
    Z50,
    Z90,
    build_history_index,
    choose_ledger_entity_col,
    context_columns,
    format_date,
    infer_naive_season_length,
    infer_season_length,
    parse_bool,
    residual_sigma,
    seasonal_naive_for_target,
    values_until_origin,
    write_blocker_report,
    forecast_strategy_manifest_fields,
)
from .forecast_strategy import (
    RECURSIVE_ROLLOUT,
    recursive_mean_path,
    strategy_from_event,
    strategy_group_columns,
)
from .metrics import summarize_forecasts
from .causal_covariates import (
    CausalCovariateIndex,
    CausalCovariateSignal,
    adjust_forecast,
    materialize_benchmark_a_panel,
)


REGISTRY_COLUMNS = [
    "model_id",
    "family",
    "candidate_type",
    "seed",
    "adapter_path",
    "hyperparams_hash",
    "hyperparams_json",
    "description",
    "repo_url",
    "enabled",
    "priority",
    "skill_embedding_text",
    "validation_score",
    "checkpoint_status",
]
RECIPE_ALIASES = {
    "last_value": "last_value",
    "lastvalue": "last_value",
    "seasonal_naive": "seasonal_naive",
    "local_drift": "drift",
    "drift": "drift",
    "causal_covariate_drift": "covariate_drift",
    "covariate_drift": "covariate_drift",
    "local_level": "local_level",
    "local_level_dlm": "local_level",
    "causal_covariate_dynamic_linear_model": "covariate_dynamic_linear_trend",
    "covariate_dynamic_linear_trend": "covariate_dynamic_linear_trend",
    "particle_filtered_state_space": "particle_local_level",
    "particle_local_level": "particle_local_level",
    "renewal": "renewal_rt",
    "renewal_rt": "renewal_rt",
    "sir": "sir_tau",
    "sir_tau": "sir_tau",
    "seir": "seir_tau",
    "seir_tau": "seir_tau",
    "seirs": "seirs_tau",
    "seirs_tau": "seirs_tau",
    "time_varying_seir": "tv_seir_rt",
    "time_varying_seir_rt": "tv_seir_rt",
    "tv_seir_rt": "tv_seir_rt",
               
    "statsforecast_autoarima": "autoarima", "autoarima": "autoarima",
    "statsforecast_autoets": "autoets",     "autoets": "autoets",
    "statsforecast_autotheta": "autotheta", "autotheta": "autotheta",
    "statsforecast_autoces": "autoces",     "autoces": "autoces",
    "prophet": "prophet",
                             
    "rnn_simple": "fixed_rnn",       "rnn": "fixed_rnn",
    "lstm_style": "neural_lstm",     "lstm": "neural_lstm",
    "gru_style": "fixed_gru",        "gru": "fixed_gru",
    "deepar_style": "neural_deepar", "deepar": "neural_deepar",
    "nbeats_basis": "neural_nbeats", "nbeats": "neural_nbeats",
    "nhits_hinterp": "neural_nhits", "nhits": "neural_nhits",
    "patchtst_patched": "neural_patchtst", "patchtst": "neural_patchtst",
    "tft_gated": "neural_tft",       "tft": "neural_tft",
                
    "chronos_external": "foundation_chronos", "chronos": "foundation_chronos",
    "timesfm_external": "foundation_timesfm", "timesfm": "foundation_timesfm",
}
STATEFUL_RECIPES = {
    "autoarima", "autoets", "autotheta", "autoces", "prophet",
    "neural_lstm", "neural_deepar", "neural_nbeats", "neural_nhits",
    "neural_patchtst", "neural_tft",
    "foundation_chronos", "foundation_timesfm",
}
NF_RECIPE_TO_MODEL = {
    "neural_deepar": "deepar",
    "neural_nbeats": "nbeats",
    "neural_nhits": "nhits",
    "neural_patchtst": "patchtst",
    "neural_tft": "tft",
}
NON_NUMERIC_METRIC_COLUMNS = {"dataset_key", "dataset", "method", "mode", "component", "split"}
VALIDATION_SCORE_POLICIES = ("plain_rmse", "recent_scale_growth_rmse")
VALIDATION_LEDGER_POLICIES = ("target_tail", "recent_origin_stratified")
QWEN25_MULTISCALE_CONTEXT_PROFILE = "qwen25_multiscale_released_sequence_v1"
QWEN25_MULTISCALE_CONTEXT_SCHEMA = (
    "caster_qwen25_multiscale_released_sequence_context_v1"
)


def is_stateful(row: pd.Series) -> bool:
    return recipe_name(row) in STATEFUL_RECIPES


def pre_fit_predictions(
    selected: pd.Series,
    ctx: "DatasetContext",
    ledger_subset: pd.DataFrame,
    *,
    device: str = "cpu",
    foundation_checkpoint_id: str = "",
    foundation_checkpoint_path: str = "",
    nf_max_steps: int = 100,
) -> dict[tuple[str, str, str, str, int], float]:
    ""




    recipe = recipe_name(selected)
    cache: dict[tuple[str, str, str, str, int], float] = {}
    group_failures: list[dict[str, object]] = []
    entity_col = ctx.ledger_entity_col

    if recipe == "neural_lstm":
        raise RuntimeError(
            "lstm_style has no supported non-archive executor; use archive_mode=required"
        )

    try:
        if recipe in {"autoarima", "autoets", "autotheta", "autoces", "prophet"}:
            from . import external_forecasting as _ext
            predictor = _ext.make_prophet_predictor() if recipe == "prophet" else _ext.make_statsforecast_predictor()
            external_season_length = infer_season_length(
                int(ctx.manifest_row["cadence_days"])
            )
            group_cols = strategy_group_columns(ledger_subset, [entity_col, "component", "forecast_origin"])
            for _, grp in ledger_subset.groupby(group_cols, dropna=False):
                first = grp.iloc[0]
                entity = str(first[entity_col])
                component = str(first["component"])
                origin_text = str(first["forecast_origin"])
                mode = str(first.get("mode", ""))
                strategy = strategy_from_event(first)
                origin = pd.to_datetime(origin_text, errors="coerce")
                if pd.isna(origin):
                    continue
                series = ctx.history_index.get((str(entity), str(component)))
                if series is None:
                    continue
                hist = _ext.history_until_origin(series, origin)
                if len(hist.values) == 0:
                    continue
                target_dates = {
                    int(pd.to_numeric(r["horizon"], errors="coerce")): pd.to_datetime(r["target_time"], errors="coerce")
                    for _, r in grp.iterrows()
                }
                if not target_dates:
                    continue
                max_h = max(target_dates.keys())
                try:
                    if strategy == RECURSIVE_ROLLOUT:
                        def one_step(step_times: np.ndarray, step_values: np.ndarray, step: int):
                            result = predictor(
                                recipe,
                                step_times,
                                step_values,
                                1,
                                {1: pd.Timestamp(origin) + pd.Timedelta(days=int(ctx.manifest_row["cadence_days"]) * step)},
                                external_season_length,
                            )
                            return float(result[1]), result

                        preds, _ = recursive_mean_path(
                            times=hist.times,
                            values=hist.values,
                            max_horizon=max_h,
                            cadence_days=int(ctx.manifest_row["cadence_days"]),
                            one_step=one_step,
                        )
                    else:
                        preds = predictor(
                            recipe,
                            hist.times,
                            hist.values,
                            max_h,
                            target_dates,
                            external_season_length,
                        )
                    for h, val in preds.items():
                        cache[(str(entity), str(component), str(origin_text), mode, int(h))] = float(val)
                except Exception as exc:
                    group_failures.append(
                        {
                            "backend": "statsforecast_or_prophet",
                            "entity_id": entity,
                            "component": component,
                            "forecast_origin": origin_text,
                            "mode": mode,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )

        elif recipe in NF_RECIPE_TO_MODEL:
            from .neuralforecast_runner import (
                RealNeuralForecastBackend,
                NeuralScope,
                cadence_to_freq,
                choose_input_size,
                panel_component_to_nf_df,
            )
            nf_model = NF_RECIPE_TO_MODEL[recipe]
            backend = RealNeuralForecastBackend(max_steps=nf_max_steps, device=device)
            nf_group_cols = ["component"]
            nf_group_cols.extend(col for col in ("mode", "forecast_strategy") if col in ledger_subset.columns)
            for _, comp_ledger in ledger_subset.groupby(nf_group_cols, dropna=False):
                first = comp_ledger.iloc[0]
                component = str(first["component"])
                mode = str(first.get("mode", ""))
                strategy = strategy_from_event(first)
                try:
                    full_df = panel_component_to_nf_df(ctx.panel, ctx.manifest_row, comp_ledger, component)
                    origins = pd.to_datetime(comp_ledger["forecast_origin"], errors="coerce").dropna()
                    if origins.empty or full_df.empty:
                        continue
                    train_cutoff = pd.Timestamp(origins.min())
                    train_df = full_df[full_df["ds"] <= train_cutoff].copy()
                    if train_df.empty:
                        continue
                    requested_h = int(pd.to_numeric(comp_ledger["horizon"], errors="coerce").max())
                    h = 1 if strategy == RECURSIVE_ROLLOUT else requested_h
                    scope = NeuralScope(
                        dataset_key=ctx.dataset_key,
                        dataset=ctx.dataset,
                        component=component,
                        h=h,
                        input_size=choose_input_size(train_df, h),
                        freq=cadence_to_freq(int(ctx.manifest_row["cadence_days"])),
                        cadence_days=int(ctx.manifest_row["cadence_days"]),
                        train_cutoff=train_cutoff,
                        model_name=nf_model,
                        seed=1,
                        max_steps=nf_max_steps,
                        device=device,
                        forecast_strategy=strategy,
                        mode=mode,
                    )
                    preds, _ = backend.fit_predict(nf_model, train_df, full_df, comp_ledger, entity_col, scope)
                    for (entity, origin_text, horizon), val in preds.items():
                        cache[(str(entity), component, str(origin_text), mode, int(horizon))] = float(val)
                except Exception as exc:
                    group_failures.append(
                        {
                            "backend": "neuralforecast",
                            "component": component,
                            "mode": mode,
                            "forecast_strategy": strategy,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )

        elif recipe in {"foundation_chronos", "foundation_timesfm"}:
            from . import foundation_forecasting as _fnd
            model_name_str = "chronos" if recipe == "foundation_chronos" else "timesfm"
            max_h_global = int(pd.to_numeric(ledger_subset["horizon"], errors="coerce").max())
            backend = _fnd.real_backend(
                model_name_str,
                foundation_checkpoint_id,
                foundation_checkpoint_path or None,
                device,
                max_horizon=max_h_global,
            )
            group_cols = strategy_group_columns(ledger_subset, [entity_col, "component", "forecast_origin"])
            for _, grp in ledger_subset.groupby(group_cols, dropna=False):
                first = grp.iloc[0]
                entity = str(first[entity_col])
                component = str(first["component"])
                origin_text = str(first["forecast_origin"])
                mode = str(first.get("mode", ""))
                strategy = strategy_from_event(first)
                origin = pd.to_datetime(origin_text, errors="coerce")
                if pd.isna(origin):
                    continue
                series = ctx.history_index.get((str(entity), str(component)))
                if series is None:
                    continue
                hist = _fnd.history_until_origin(series, origin)
                if len(hist.values) == 0:
                    continue
                horizons = sorted(
                    pd.to_numeric(grp["horizon"], errors="coerce").dropna().astype(int).tolist()
                )
                if not horizons:
                    continue
                try:
                    if strategy == RECURSIVE_ROLLOUT:
                        def one_step(step_times: np.ndarray, step_values: np.ndarray, _step: int):
                            step_prediction = backend.predict(step_values, 1, [1], int(ctx.manifest_row["cadence_days"]))
                            return float(step_prediction.mean[1]), step_prediction

                        means, _ = recursive_mean_path(
                            times=hist.times,
                            values=hist.values,
                            max_horizon=max(horizons),
                            cadence_days=int(ctx.manifest_row["cadence_days"]),
                            one_step=one_step,
                        )
                    else:
                        means = backend.predict(
                            hist.values,
                            max(horizons),
                            horizons,
                            int(ctx.manifest_row["cadence_days"]),
                        ).mean
                    for h, val in means.items():
                        cache[(str(entity), str(component), str(origin_text), mode, int(h))] = float(val)
                except Exception as exc:
                    group_failures.append(
                        {
                            "backend": "foundation",
                            "entity_id": entity,
                            "component": component,
                            "forecast_origin": origin_text,
                            "mode": mode,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )

    except Exception as exc:
        raise RuntimeError(
            f"pre-fit execution failed for model_id={selected.get('model_id', '')} "
            f"recipe={recipe}: {type(exc).__name__}: {exc}"
        ) from exc

    if group_failures:
        examples = json.dumps(
            group_failures[:10], sort_keys=True, separators=(",", ":")
        )
        raise RuntimeError(
            f"pre-fit group execution failed for model_id={selected.get('model_id', '')} "
            f"recipe={recipe}; group_failure_count={len(group_failures)}; "
            f"examples={examples}; last-value substitution is forbidden"
        )

    if not ledger_subset.empty and not cache:
        raise RuntimeError(
            f"pre-fit execution produced no forecasts for model_id={selected.get('model_id', '')} "
            f"recipe={recipe}; last-value substitution is forbidden"
        )

    return cache


@dataclass
class DatasetContext:
    dataset_key: str
    dataset: str
    manifest_row: pd.Series
    panel: pd.DataFrame
    ledger: pd.DataFrame
    ledger_entity_col: str
    history_index: dict[tuple[str, str], Any]
    season_length: int
    context_cols: list[str]
    covariate_index: CausalCovariateIndex | None = None


def derive_agent_selection_scope(selected_dataset_keys: list[str], excluded_dataset_keys: list[str]) -> str:
    ""
    keys = [str(key) for key in selected_dataset_keys if str(key)]
    if len(keys) == 1:
        return keys[0]
    if not excluded_dataset_keys:
        return "all_available"
    return "custom_multi_dataset"


def _json_dumps(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _hash_hyperparams(params: dict[str, object]) -> str:
    import hashlib

    return hashlib.sha1(_json_dumps(params).encode()).hexdigest()[:12]


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text not in {"0", "false", "f", "no", "n", "disabled"}


def read_candidate_registry(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            data = data.get("candidates", data.get("models", []))
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("candidates", data.get("models", []))
    elif path.suffix.lower() == ".csv":
        data = pd.read_csv(path).to_dict(orient="records")
    else:
        raise ValueError(f"unsupported registry suffix: {path.suffix}")
    rows: list[dict[str, object]] = []
    for raw in data:
        row = dict(raw)
        params = row.get("hyperparams") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                params = {}
        row.setdefault("hyperparams_hash", _hash_hyperparams(params))
        row.setdefault("hyperparams_json", _json_dumps(params))
        row.setdefault("description", " ".join(str(row.get(c, "")) for c in ("model_id", "family", "candidate_type")))
        row.setdefault("repo_url", "")
        row.setdefault("enabled", True)
        row.setdefault("priority", 100)
        row.setdefault("validation_score", 0.0)
        row.setdefault("checkpoint_status", "unknown")
        if not row.get("skill_embedding_text"):
            row["skill_embedding_text"] = str(row.get("description", "")).strip()
        row["enabled"] = _coerce_bool(row.get("enabled", True))
        rows.append({col: row.get(col, "") for col in REGISTRY_COLUMNS})
    registry = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    if registry.empty:
        raise ValueError(f"empty candidate registry: {path}")
    return registry


def recipe_name(row: pd.Series) -> str:
    keys = [
        str(row.get("model_id", "")).strip().lower(),
        str(row.get("candidate_type", "")).strip().lower(),
    ]
    for key in keys:
        if key in RECIPE_ALIASES:
            return RECIPE_ALIASES[key]
    return ""


def eligible_registry(registry: pd.DataFrame) -> pd.DataFrame:
    enabled = registry[registry["enabled"].map(_coerce_bool)].copy()
    enabled["recipe"] = enabled.apply(recipe_name, axis=1)
    return enabled[enabled["recipe"].astype(str) != ""].reset_index(drop=True)


def _fair_order_key(dataset_key: str, forecast_origin: str, model_id: str) -> str:
    seed = f"{dataset_key}|{forecast_origin}|{model_id}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()


def fair_registry_prompt_rows(
    registry: pd.DataFrame,
    *,
    dataset_key: str,
    forecast_origin: str,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Build the anonymous scientific candidate surface shown to an LLM.

    The real model identifier is used only to compute the deterministic display
    order and is returned in a separate trace mapping. It is never placed in a
    prompt candidate row.
    """
    ordered = sorted(
        [
            (
                str(row["model_id"]),
                str(row.get("description", "")),
            )
            for _, row in registry.iterrows()
        ],
        key=lambda item: (
            _fair_order_key(dataset_key, forecast_origin, item[0]),
            item[0],
        ),
    )
    width = max(2, len(str(len(ordered))))
    prompt_rows: list[dict[str, object]] = []
    choice_to_model: dict[str, str] = {}
    for index, (model_id, description) in enumerate(ordered, start=1):
        choice_id = f"C{index:0{width}d}"
        prompt_rows.append(
            {
                "choice_id": choice_id,
                "scientific_description": description,
            }
        )
        choice_to_model[choice_id] = model_id
    return prompt_rows, choice_to_model


def load_dataset_contexts(
    manifest_path: str | Path,
    *,
    root: Path | None = None,
    caster_root: Path | None = None,
) -> tuple[list[DatasetContext], int]:
    root = root or baseline_root()
    caster_root = caster_root or caster_root_from_baseline(root)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    expected_rows = int(pd.to_numeric(manifest["ledger_rows"], errors="raise").sum())
    contexts: list[DatasetContext] = []
    for _, manifest_row in manifest.iterrows():
        panel_path = resolve_manifest_path(manifest_row["panel_path"], caster_root, root)
        ledger_path = resolve_manifest_path(manifest_row["ledger_path"], caster_root, root)
        panel = pd.read_csv(panel_path)
        ledger = pd.read_csv(ledger_path, keep_default_na=False, low_memory=False)
        declared_rows = int(manifest_row["ledger_rows"])
        if len(ledger) != declared_rows:
            raise ValueError(f"{manifest_row['dataset_key']} ledger row mismatch declared={declared_rows} actual={len(ledger)}")
        entity_col = choose_ledger_entity_col(ledger, str(manifest_row["panel_entity_col"]))
        dataset = str(manifest_row["dataset"])
        mobility_join_keys = {"country", "country_code", "entity_id", "date"}
        if dataset == "benchmark_a" and mobility_join_keys <= set(panel.columns):
            panel = materialize_benchmark_a_panel(panel).panel
            if "__release_time__" not in panel.columns:
                panel["__release_time__"] = pd.to_datetime(panel["date"], errors="raise")
        covariate_index = CausalCovariateIndex(panel)
        contexts.append(DatasetContext(
            dataset_key=str(manifest_row["dataset_key"]),
            dataset=str(manifest_row["dataset"]),
            manifest_row=manifest_row,
            panel=panel,
            ledger=ledger,
            ledger_entity_col=entity_col,
            history_index=build_history_index(panel, manifest_row, ledger),
            season_length=infer_naive_season_length(
                int(manifest_row["cadence_days"])
            ),
            context_cols=context_columns(ledger),
            covariate_index=covariate_index,
        ))
    return contexts, expected_rows


def _params(row: pd.Series) -> dict[str, object]:
    text = str(row.get("hyperparams_json", "") or "{}")
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _resid_var(values: np.ndarray, pred: float) -> float:
    sigma = residual_sigma(values, pred)
    return max(float(sigma) ** 2, 1e-6)


def _linear_slope(values: np.ndarray, default: float = 1.0) -> float:
    y = np.asarray(values[-8:], dtype=float)
    if len(y) < 2:
        return float(default)
    x = np.arange(len(y), dtype=float)
    slope, _intercept = np.polyfit(x, y, deg=1)
    return float(slope)


def _stable_uint32(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little", signed=False)


def _binomial_draw(rng: np.random.Generator, n: float, p: float) -> float:
    trials = int(round(max(float(n), 0.0)))
    prob = min(max(float(p), 0.0), 1.0)
    if trials <= 0 or prob <= 0.0:
        return 0.0
    return float(rng.binomial(trials, prob))


def _compartmental_mode_is_stochastic(params: dict[str, object]) -> bool:
    mode = str(params.get("transition_mode", "stochastic_binomial")).strip().lower()
    return mode not in {"deterministic", "deterministic_mean_field", "mean_field"}


def _forecast_compartmental_binomial(
    recipe: str,
    params: dict[str, object],
    values: np.ndarray,
    horizon: int,
    row_key: tuple[object, ...],
    *,
    seed: int = 0,
) -> tuple[float, float]:
    ""






    values = np.asarray(values, dtype=float)
    last = float(max(values[-1], 0.0))
    population = max(float(params.get("population", 1_000_000.0)), 1.0)
    gamma = max(float(params.get("gamma", 0.2)), 0.0)
    sigma = max(float(params.get("sigma", 1.0 / 3.0)), 0.0)
    waning_rate = max(float(params.get("waning_rate", 1.0 / 180.0)), 0.0)
    reporting_rate = max(float(params.get("reporting_rate", 0.04)), 1e-6)
    n_sim = max(1, int(params.get("n_simulations", 64)))
    log_beta_rw_scale = max(float(params.get("log_beta_rw_scale", 0.0)), 0.0)
    log_gamma_rw_scale = max(float(params.get("log_gamma_rw_scale", 0.0)), 0.0)

    y = np.maximum(values, 1.0)
    if len(y) < 3:
        rt0 = 1.0
    else:
        recent_log = np.log(y[-min(len(y), 8):])
        mean_log_growth = float(np.nanmean(np.diff(recent_log)))
        rt0 = float(np.clip(np.exp(mean_log_growth) / max(1.0 - gamma, 1e-6), 0.2, 3.5))

    if recipe == "sir_tau":
        i0 = min(max(last / reporting_rate, 1.0), 0.25 * population)
        e0 = 0.0
        r0 = min(
            0.05 * population
            + max(float(np.nansum(values[-min(len(values), 21):])), 0.0) / reporting_rate,
            0.80 * population,
        )
    else:
        i0 = min(max(last / reporting_rate, 1.0), 0.20 * population)
        e0 = min(0.7 * i0, 0.20 * population)
        if recipe == "seirs_tau":
            r0 = min(0.08 * population, 0.80 * population)
        elif recipe == "tv_seir_rt":
            r0 = min(0.05 * population, 0.80 * population)
        else:
            r0 = min(
                0.05 * population
                + max(float(np.nansum(values[-min(len(values), 21):])), 0.0) / reporting_rate,
                0.80 * population,
            )
    s0 = max(population - e0 - i0 - r0, 0.0)

    rng = np.random.default_rng(_stable_uint32(int(seed), recipe, *row_key))
    s = np.full(n_sim, s0, dtype=float)
    e = np.full(n_sim, e0, dtype=float)
    i = np.full(n_sim, i0, dtype=float)
    r = np.full(n_sim, r0, dtype=float)
    incidence = np.zeros(n_sim, dtype=float)
    beta = np.full(n_sim, rt0 * gamma, dtype=float)
    gamma_path = np.full(n_sim, gamma, dtype=float)
    p_inc = float(np.clip(1.0 - np.exp(-sigma), 0.0, 1.0))
    p_wane = float(np.clip(1.0 - np.exp(-waning_rate), 0.0, 1.0))
    n_steps = max(1, int(horizon))
    for step in range(n_steps):
        p_inf = np.clip(1.0 - np.exp(-beta * np.maximum(i, 0.0) / population), 0.0, 1.0)
        p_rec = np.clip(1.0 - np.exp(-gamma_path), 0.0, 1.0)
        if recipe == "sir_tau":
            new_inf = rng.binomial(np.maximum(np.rint(s).astype(np.int64), 0), p_inf).astype(float)
            new_rec = rng.binomial(np.maximum(np.rint(i).astype(np.int64), 0), p_rec).astype(float)
            new_inf = np.minimum(s, new_inf)
            new_rec = np.minimum(i, new_rec)
            s = np.maximum(s - new_inf, 0.0)
            i = np.maximum(i + new_inf - new_rec, 0.0)
            r = np.maximum(r + new_rec, 0.0)
            incidence = new_inf
        else:
            waned = (
                rng.binomial(np.maximum(np.rint(r).astype(np.int64), 0), p_wane).astype(float)
                if recipe == "seirs_tau"
                else np.zeros(n_sim, dtype=float)
            )
            susceptible = s + waned
            new_exp = rng.binomial(np.maximum(np.rint(susceptible).astype(np.int64), 0), p_inf).astype(float)
            new_inf = rng.binomial(np.maximum(np.rint(e).astype(np.int64), 0), p_inc).astype(float)
            new_rec = rng.binomial(np.maximum(np.rint(i).astype(np.int64), 0), p_rec).astype(float)
            new_exp = np.minimum(susceptible, new_exp)
            new_inf = np.minimum(e, new_inf)
            new_rec = np.minimum(i, new_rec)
            s = np.maximum(susceptible - new_exp, 0.0)
            e = np.maximum(e + new_exp - new_inf, 0.0)
            i = np.maximum(i + new_inf - new_rec, 0.0)
            r = np.maximum(r + new_rec - waned, 0.0)
            incidence = new_inf
        if step + 1 < n_steps:
            if log_beta_rw_scale > 0.0:
                beta = beta * np.exp(rng.normal(0.0, log_beta_rw_scale, size=beta.shape))
            if log_gamma_rw_scale > 0.0:
                gamma_path = gamma_path * np.exp(
                    rng.normal(0.0, log_gamma_rw_scale, size=gamma_path.shape)
                )

    draws = np.maximum(0.0, reporting_rate * incidence)
    pred = float(np.mean(draws)) if draws.size else 0.0
    latent_var = float(np.var(draws, ddof=1)) if draws.size > 1 else 0.0
    return max(0.0, pred), max(latent_var + max(pred, 1.0), 1.0)


def forecast_recipe(
    row: pd.Series,
    values: np.ndarray,
    horizon: int,
    row_key: tuple[object, ...] = (),
    covariate_signal: CausalCovariateSignal | None = None,
) -> tuple[float, float]:
    recipe = recipe_name(row)
    params = _params(row)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("empty history")
    last = float(values[-1])
    if recipe == "last_value":
        pred = last
    elif recipe == "seasonal_naive":
        if "season_length" not in params:
            raise ValueError(
                "seasonal_naive requires event-aware cadence-specific season_length"
            )
        season_length = int(params["season_length"])
        if len(values) < season_length:
            raise ValueError(
                f"seasonal_naive lacks {season_length} native history values"
            )
        pred = float(values[-season_length])
        pred = max(0.0, pred)
    elif recipe == "drift":
        pred = max(0.0, last + _linear_slope(values, float(params.get("slope", 1.0))) * int(horizon))
    elif recipe == "fixed_rnn":
        alpha = float(params.get("alpha", 0.55))
        gain = float(params.get("gain", 1.0))
        hidden = math.log1p(max(float(values[0]), 0.0))
        residuals: list[float] = []
        for value in np.maximum(values[1:], 0.0):
            residuals.append(float(value) - math.expm1(hidden))
            hidden = alpha * math.log1p(float(value)) + (1.0 - alpha) * hidden
        for _ in range(max(1, int(horizon))):
            hidden = gain * hidden
        pred = max(0.0, math.expm1(hidden))
        return pred, max(float(np.var(residuals)) if residuals else 1.0, 1.0)
    elif recipe == "fixed_gru":
        hidden = math.log1p(max(float(values[0]), 0.0))
        residuals = []
        for value in np.maximum(values[1:], 0.0):
            x = math.log1p(float(value))
            residuals.append(float(value) - math.expm1(hidden))
            update = 1.0 / (1.0 + math.exp(-0.6 * (x - hidden)))
            reset = 1.0 / (1.0 + math.exp(-0.5 * hidden))
            candidate = math.tanh(x + reset * hidden)
            hidden = (1.0 - update) * hidden + update * candidate
        for _ in range(max(1, int(horizon))):
            hidden = 0.95 * hidden
        pred = max(0.0, math.expm1(hidden))
        return pred, max(float(np.var(residuals)) if residuals else 1.0, 1.0)
    elif recipe == "covariate_drift":
        base = max(0.0, last + _linear_slope(values, float(params.get("slope", 1.0))) * int(horizon))
        return adjust_forecast(
            base,
            _resid_var(values, base),
            values,
            horizon,
            covariate_signal or CausalCovariateSignal(0.0, 0, ()),
            gain=float(params.get("covariate_gain", 0.15)),
            damping=float(params.get("covariate_damping", 0.90)),
        )
    elif recipe == "local_level":
        alpha = float(params.get("alpha", 0.65))
        level = float(values[0])
        residuals: list[float] = []
        for val in values[1:]:
            residuals.append(float(val) - level)
            level = alpha * float(val) + (1.0 - alpha) * level
        trend = float(values[-1] - values[-2]) if len(values) >= 2 else 0.0
        damping = float(params.get("trend_damping", 0.75))
        pred = max(0.0, level + sum((damping ** h) * trend for h in range(1, int(horizon) + 1)))
        return pred, max(float(np.var(residuals)) if residuals else 1.0, 1.0) * max(1, int(horizon))
    elif recipe == "renewal_rt":
        reproduction = float(params.get("reproduction", 1.02))
        kernel = np.asarray(params.get("kernel", [0.50, 0.30, 0.15, 0.05]), dtype=float)
        kernel = kernel / max(float(kernel.sum()), 1e-12)
        hist = list(np.maximum(values, 0.0))
        for _ in range(int(horizon)):
            tail = np.asarray(hist[-len(kernel):][::-1], dtype=float)
            k = kernel[: len(tail)]
            k = k / max(float(k.sum()), 1e-12)
            hist.append(max(0.0, reproduction * float(np.dot(k, tail))))
        pred = float(hist[-1])
    elif recipe in {"sir_tau", "seir_tau"}:
        if _compartmental_mode_is_stochastic(params):
            try:
                compartmental_seed = int(row.get("seed", 0) or 0)
            except Exception:
                compartmental_seed = 0
            return _forecast_compartmental_binomial(
                recipe,
                params,
                values,
                horizon,
                row_key,
                seed=compartmental_seed,
            )
        population = max(float(params.get("population", 1_000_000.0)), 1.0)
        beta = max(float(params.get("beta", 0.55)), 0.0)
        gamma = max(float(params.get("gamma", 0.33)), 0.0)
        reporting_rate = max(float(params.get("reporting_rate", 0.04)), 1e-6)
        i = min(max(last / reporting_rate, 1.0), 0.25 * population)
        e = 0.6 * i
        r = min(0.05 * population, 0.8 * population)
        s = max(population - e - i - r, 0.0)
        incidence = 0.0
        for _ in range(int(horizon)):
            force = 1.0 - math.exp(-beta * max(i, 0.0) / population)
            rec_prob = 1.0 - math.exp(-gamma)
            if recipe == "seir_tau":
                sigma = max(float(params.get("sigma", 0.45)), 0.0)
                inc_prob = 1.0 - math.exp(-sigma)
                new_exp = min(s, max(0.0, s * force))
                new_inf = min(e, max(0.0, e * inc_prob))
                new_rec = min(i, max(0.0, i * rec_prob))
                s = max(s - new_exp, 0.0)
                e = max(e + new_exp - new_inf, 0.0)
                i = max(i + new_inf - new_rec, 0.0)
                r = max(r + new_rec, 0.0)
                incidence = new_inf
            else:
                new_inf = min(s, max(0.0, s * force))
                new_rec = min(i, max(0.0, i * rec_prob))
                s = max(s - new_inf, 0.0)
                i = max(i + new_inf - new_rec, 0.0)
                r = max(r + new_rec, 0.0)
                incidence = new_inf
        pred = max(0.0, reporting_rate * incidence)
        return pred, max(pred, 1.0) + 0.05 * pred * pred
    elif recipe == "seirs_tau":
        if _compartmental_mode_is_stochastic(params):
            try:
                compartmental_seed = int(row.get("seed", 0) or 0)
            except Exception:
                compartmental_seed = 0
            return _forecast_compartmental_binomial(
                recipe,
                params,
                values,
                horizon,
                row_key,
                seed=compartmental_seed,
            )
        population = max(float(params.get("population", 1_000_000.0)), 1.0)
        gamma = max(float(params.get("gamma", 0.2)), 0.0)
        sigma = max(float(params.get("sigma", 0.3333333333)), 0.0)
        waning_rate = max(float(params.get("waning_rate", 0.0055555556)), 0.0)
        reporting_rate = max(float(params.get("reporting_rate", 0.04)), 1e-6)
        beta = max(float(params.get("beta", 0.55)), 0.0)
        i = min(max(last / reporting_rate, 1.0), 0.25 * population)
        e = 0.6 * i
        r = min(0.05 * population, 0.8 * population)
        s = max(population - e - i - r, 0.0)
        incidence = 0.0
        for _ in range(int(horizon)):
            force = 1.0 - math.exp(-beta * max(i, 0.0) / population)
            rec_prob = 1.0 - math.exp(-gamma)
            inc_prob = 1.0 - math.exp(-sigma)
            wane_prob = 1.0 - math.exp(-waning_rate)
            new_wane = min(r, max(0.0, r * wane_prob))
            new_exp = min(s, max(0.0, s * force))
            new_inf = min(e, max(0.0, e * inc_prob))
            new_rec = min(i, max(0.0, i * rec_prob))
            s = max(s + new_wane - new_exp, 0.0)
            e = max(e + new_exp - new_inf, 0.0)
            i = max(i + new_inf - new_rec, 0.0)
            r = max(r + new_rec - new_wane, 0.0)
            incidence = new_inf
        pred = max(0.0, reporting_rate * incidence)
        return pred, max(pred, 1.0) + 0.05 * pred * pred
    elif recipe == "tv_seir_rt":
        if _compartmental_mode_is_stochastic(params):
            try:
                compartmental_seed = int(row.get("seed", 0) or 0)
            except Exception:
                compartmental_seed = 0
            return _forecast_compartmental_binomial(
                recipe,
                params,
                values,
                horizon,
                row_key,
                seed=compartmental_seed,
            )
        population = max(float(params.get("population", 1_000_000.0)), 1.0)
        gamma = max(float(params.get("gamma", 0.2)), 0.0)
        sigma = max(float(params.get("sigma", 0.3333333333)), 0.0)
        reporting_rate = max(float(params.get("reporting_rate", 0.04)), 1e-6)
        rt_shrink = float(params.get("rt_shrink", 0.85))
        recent = values[-4:] if len(values) >= 4 else values
        if len(recent) >= 2 and recent[-1] > 0 and recent[-2] > 0:
            log_growth = math.log(max(recent[-1], 1.0) / max(recent[-2], 1.0))
            rt = min(max(math.exp(log_growth / max(gamma, 1e-6)), 0.1), 5.0)
        else:
            rt = 1.0
        i = min(max(last / reporting_rate, 1.0), 0.25 * population)
        e = 0.6 * i
        r = min(0.05 * population, 0.8 * population)
        s = max(population - e - i - r, 0.0)
        incidence = 0.0
        for _ in range(int(horizon)):
            force = 1.0 - math.exp(-rt * gamma * max(i, 0.0) / population)
            rec_prob = 1.0 - math.exp(-gamma)
            inc_prob = 1.0 - math.exp(-sigma)
            new_exp = min(s, max(0.0, s * force))
            new_inf = min(e, max(0.0, e * inc_prob))
            new_rec = min(i, max(0.0, i * rec_prob))
            s = max(s - new_exp, 0.0)
            e = max(e + new_exp - new_inf, 0.0)
            i = max(i + new_inf - new_rec, 0.0)
            r = max(r + new_rec, 0.0)
            incidence = new_inf
            rt = 1.0 + (rt - 1.0) * rt_shrink
        pred = max(0.0, reporting_rate * incidence)
        return pred, max(pred, 1.0) + 0.05 * pred * pred
    elif recipe == "covariate_dynamic_linear_trend":
        alpha = float(params.get("alpha", 0.35))
        beta_s = float(params.get("beta", 0.10))
        damping = float(params.get("damping", 0.90))
        level = float(values[0])
        trend = 0.0
        residuals: list[float] = []
        for val in values[1:]:
            v = float(val)
            residuals.append(v - (level + trend))
            new_level = alpha * v + (1.0 - alpha) * (level + trend)
            trend = beta_s * (new_level - level) + (1.0 - beta_s) * damping * trend
            level = new_level
        base = max(0.0, level + sum(damping ** h * trend for h in range(1, int(horizon) + 1)))
        variance = max(float(np.var(residuals)) if residuals else 1.0, 1.0) * max(1, int(horizon))
        return adjust_forecast(
            base,
            variance,
            values,
            horizon,
            covariate_signal or CausalCovariateSignal(0.0, 0, ()),
            gain=float(params.get("covariate_gain", 0.15)),
            damping=float(params.get("covariate_damping", 0.90)),
        )
    elif recipe == "particle_local_level":
        n_particles = int(params.get("n_particles", 128))
        process_scale = float(params.get("process_scale", 0.2))
        obs_scale = float(params.get("obs_scale", 1.0))
        rng = np.random.default_rng(0)
        particles = np.full(n_particles, float(values[0]))
        for val in values[1:]:
            v = float(val)
            scale = max(abs(v), 1.0)
            particles = particles + rng.normal(0.0, process_scale * scale, n_particles)
            weights = np.exp(-0.5 * ((particles - v) / max(obs_scale * scale, 1e-6)) ** 2)
            w_sum = weights.sum()
            weights = weights / w_sum if w_sum > 0 else np.ones(n_particles) / n_particles
            particles = particles[rng.choice(n_particles, size=n_particles, replace=True, p=weights)]
        last_scale = max(abs(float(values[-1])), 1.0)
        for _ in range(int(horizon)):
            particles = particles + rng.normal(0.0, process_scale * last_scale, n_particles)
        pred = max(0.0, float(np.mean(particles)))
        return pred, max(float(np.var(particles)), 1.0)
    else:
        raise ValueError(f"candidate has no executable restart recipe: {row.get('model_id')}")
    return float(pred), _resid_var(values, float(pred))


def group_history_values(ctx: DatasetContext, entity_id: str, component: str, origin: pd.Timestamp) -> np.ndarray:
    series = ctx.history_index.get((str(entity_id), str(component)))
    return values_until_origin(series, origin) if series is not None else np.asarray([], dtype=float)


def group_covariate_signal(
    ctx: DatasetContext, entity_id: str, component: str, origin: pd.Timestamp
) -> CausalCovariateSignal:
    if ctx.covariate_index is None:
        return CausalCovariateSignal(0.0, 0, ())
    return ctx.covariate_index.signal(entity_id, component, origin)


def forecast_recipe_for_event(
    ctx: DatasetContext,
    candidate: pd.Series,
    event: pd.Series,
    values: np.ndarray,
    *,
    covariate_signal: CausalCovariateSignal | None = None,
) -> tuple[float, float]:
    ""
    horizon = int(pd.to_numeric(event.get("horizon"), errors="raise"))
    if recipe_name(candidate) != "seasonal_naive":
        return forecast_recipe(
            candidate,
            values,
            horizon,
            covariate_signal=covariate_signal,
        )
    entity_id = str(event[ctx.ledger_entity_col])
    component = str(event["component"])
    series = ctx.history_index.get((entity_id, component))
    if series is None:
        raise ValueError("seasonal_naive has no timestamped series")
    pred, sigma, fallback, reason = seasonal_naive_for_target(
        series,
        pd.to_datetime(event["forecast_origin"], errors="raise"),
        pd.to_datetime(event["target_time"], errors="raise"),
        ctx.season_length,
        int(ctx.manifest_row["cadence_days"]),
    )
    if fallback:
        raise ValueError(
            "seasonal_naive structural history unavailable; "
            f"agent may not substitute last value: {reason}"
        )
    return pred, max(float(sigma) ** 2, 1e-6)


def _compact_selection_context_text(
    ctx: DatasetContext,
    *,
    release: pd.DataFrame | None = None,
    origin_text: str = "",
) -> str:
    ""
    ledger = release if release is not None else ctx.ledger
    components = sorted(str(x) for x in ledger["component"].dropna().unique()) if "component" in ledger.columns else []
    horizons = pd.to_numeric(ledger.get("horizon", pd.Series(dtype=float)), errors="coerce").dropna()
    entities = int(ledger[ctx.ledger_entity_col].astype(str).nunique()) if ctx.ledger_entity_col in ledger.columns else 0
    origins = sorted(str(x) for x in ledger.get("forecast_origin", pd.Series(dtype=str)).dropna().unique())
    origin = pd.to_datetime(origin_text, errors="coerce") if origin_text else pd.NaT
    history_lengths: list[int] = []
    recent_slopes: list[float] = []
    if not pd.isna(origin):
        group_cols = [ctx.ledger_entity_col, "component"]
        for (entity, component), _group in ledger.groupby(group_cols, dropna=False):
            values = group_history_values(ctx, str(entity), str(component), origin)
            history_lengths.append(int(len(values)))
            if len(values) >= 2:
                recent_slopes.append(float(values[-1] - values[max(0, len(values) - min(4, len(values)))]))
    parts = [
        f"dataset_key={ctx.dataset_key}",
        f"dataset={ctx.dataset}",
        f"cadence_days={ctx.manifest_row.get('cadence_days', '')}",
        f"season_length={ctx.season_length}",
        f"release_rows={len(ledger)}",
        f"entities={entities}",
        f"components={','.join(components[:8])}",
    ]
    if not horizons.empty:
        parts.append(f"horizon_min={int(horizons.min())}")
        parts.append(f"horizon_max={int(horizons.max())}")
    if origin_text:
        parts.append(f"forecast_origin={origin_text}")
    elif origins:
        parts.append(f"forecast_origin_count={len(origins)}")
    if history_lengths:
        parts.append(f"history_rows_min={min(history_lengths)}")
        parts.append(f"history_rows_median={float(np.median(history_lengths)):.1f}")
        parts.append(f"history_rows_max={max(history_lengths)}")
    if recent_slopes:
        parts.append(f"recent_change_median={float(np.median(recent_slopes)):.4g}")
    return "; ".join(parts)


@lru_cache(maxsize=1)
def _canonical_context_api() -> tuple[dict[str, Any], Any, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    newmethod_src = repo_root / "code" / "caster" / "src"
    if str(newmethod_src) not in sys.path:
        sys.path.insert(0, str(newmethod_src))
    from caster.tasks import build_selection_context, load_task_specs, selection_context_sha256

    specs = load_task_specs(repo_root / "configs" / "caster_task_specs_v20.yaml")
    return specs, build_selection_context, selection_context_sha256


@lru_cache(maxsize=1)
def _qwen25_multiscale_context_api() -> Any:
    ""






    repo_root = Path(__file__).resolve().parents[4]
    newmethod_src = repo_root / "code" / "caster" / "src"
    if str(newmethod_src) not in sys.path:
        sys.path.insert(0, str(newmethod_src))
    from caster.tasks import build_qwen25_multiscale_context

    return build_qwen25_multiscale_context


def selection_context_profile(ctx: DatasetContext) -> str:
    row = getattr(ctx, "manifest_row", {})
    if not hasattr(row, "get"):
        return ""
    return str(row.get("selection_context_profile", "")).strip()


def uses_qwen25_multiscale_context(ctx: DatasetContext) -> bool:
    return selection_context_profile(ctx) == QWEN25_MULTISCALE_CONTEXT_PROFILE


def _qwen25_multiscale_context_metadata(
    payload: dict[str, Any],
    text: str,
    digest: str,
    validation: pd.DataFrame,
) -> dict[str, object]:
    ""

    required_validation_columns = {"validation_stage", "check", "status", "value"}
    if validation.empty or not required_validation_columns.issubset(validation.columns):
        raise ValueError("Qwen multiscale context validation is empty or malformed")
    if not bool(validation["status"].astype(str).eq("PASS").all()):
        raise ValueError("Qwen multiscale context validation did not pass")
    if str(payload.get("schema", "")) != QWEN25_MULTISCALE_CONTEXT_SCHEMA:
        raise ValueError("Qwen multiscale context schema mismatch")
    if str(payload.get("profile", "")) != QWEN25_MULTISCALE_CONTEXT_PROFILE:
        raise ValueError("Qwen multiscale context profile mismatch")

    alignment = payload.get("information_alignment", {})
    if not isinstance(alignment, dict):
        raise ValueError("Qwen multiscale context has no information-alignment guard")
    for field in (
        "validation_metrics_visible",
        "test_metrics_visible",
        "future_or_unreleased_target_values_visible",
    ):
        if alignment.get(field) is not False:
            raise ValueError(f"Qwen multiscale context guard must set {field}=false")

    sketch = payload.get("causal_multiscale_sequence_sketch", {})
    if not isinstance(sketch, dict):
        raise ValueError("Qwen multiscale context has no causal sequence sketch")
    guards = sketch.get("causal_guards", {})
    if not isinstance(guards, dict):
        raise ValueError("Qwen multiscale sequence sketch has no causal guards")
    for field in ("target_time_lte_cutoff", "release_time_lte_cutoff"):
        if guards.get(field) is not True:
            raise ValueError(f"Qwen multiscale sequence guard must set {field}=true")
    for field in (
        "split_membership_consulted_for_sequence_values",
        "validation_metrics_included",
        "test_metrics_included",
        "unreleased_labels_included",
    ):
        if guards.get(field) is not False:
            raise ValueError(f"Qwen multiscale sequence guard must set {field}=false")

    validation_records = []
    for row in validation[["validation_stage", "check", "status", "value"]].itertuples(
        index=False, name=None
    ):
        validation_records.append(
            {
                "validation_stage": str(row[0]),
                "check": str(row[1]),
                "status": str(row[2]),
                "value": "" if pd.isna(row[3]) else str(row[3]),
            }
        )
    sequence_validation_records = [
        row
        for row in validation_records
        if row["validation_stage"] == "causal_sequence_sketch"
    ]
    validation_json = json.dumps(
        sequence_validation_records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "selection_context_profile": QWEN25_MULTISCALE_CONTEXT_PROFILE,
        "selection_context_schema": QWEN25_MULTISCALE_CONTEXT_SCHEMA,
        "selection_context_sha256": str(digest),
        "selection_context_text_sha256": text_sha,
        "combined_context_sha256": str(digest),
        "combined_context_text_sha256": text_sha,
        "base_selection_context_schema": str(payload.get("base_context_schema", "")),
        "base_selection_context_sha256": str(payload.get("base_context_sha256", "")),
        "sequence_sketch_schema": str(payload.get("sequence_sketch_schema", "")),
        "sequence_sketch_sha256": str(payload.get("sequence_sketch_sha256", "")),
        "sequence_sketch_validation_status": "PASS",
        "sequence_sketch_validation_rows": int(
            validation["validation_stage"].astype(str).eq("causal_sequence_sketch").sum()
        ),
        "sequence_sketch_validation_sha256": hashlib.sha256(
            validation_json.encode("utf-8")
        ).hexdigest(),
        "sequence_sketch_validation_checks": [
            str(value)
            for value in validation.loc[
                validation["validation_stage"].astype(str).eq("causal_sequence_sketch"),
                "check",
            ].tolist()
        ],
        "selection_context_cutoff": str(payload.get("cutoff_time", "")),
        "selection_context_role": str(payload.get("context_role", "")),
        "selection_context_history_scope": str(payload.get("history_scope", "")),
        "selection_context_history_max": str(payload.get("history_max", "")),
        "selection_context_builder": (
            "caster.tasks.build_selection_context+build_causal_sequence_sketch"
        ),
        "selection_context_validation_visible": False,
        "selection_context_future_or_unreleased_target_values_visible": False,
    }


def selection_context_with_metadata(
    ctx: DatasetContext,
    *,
    release: pd.DataFrame | None = None,
    origin_text: str = "",
) -> tuple[str, dict[str, object]]:
    ""






    specs, build_context, context_sha256 = _canonical_context_api()
    formal_ledger_columns = {"release_time", "revision_version", "target_time", "component", "entity_id"}
    uses_formal_schema = ctx.dataset_key in specs and formal_ledger_columns.issubset(ctx.ledger.columns)
    if not uses_formal_schema:
        text = _compact_selection_context_text(ctx, release=release, origin_text=origin_text)
        fallback_reason = "task_not_in_formal_specs" if ctx.dataset_key not in specs else "nonformal_ledger_schema"
        return text, {
            "selection_context_schema": "agent_compact_context_fallback_v1",
            "selection_context_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "selection_context_cutoff": str(origin_text or "DATASET_LEVEL"),
            "selection_context_role": "nonformal_fallback",
            "selection_context_history_max": str(origin_text or ""),
            "selection_context_builder": "caster_baselines.agentic_skills._compact_selection_context_text",
            "selection_context_fallback_reason": fallback_reason,
            "selection_context_validation_visible": False,
        }

    spec = specs[ctx.dataset_key]
    cutoff = origin_text or spec.t_sel
    role = "agent_origin_selection" if origin_text else "formal_selection"
    if uses_qwen25_multiscale_context(ctx):
        build_augmented_context = _qwen25_multiscale_context_api()
        payload, text, digest, validation = build_augmented_context(
            ctx.panel,
            ctx.ledger,
            spec,
            cutoff_time=cutoff,
            context_role=role,
        )
        return text, _qwen25_multiscale_context_metadata(
            payload, text, digest, validation
        )

    payload, text, validation = build_context(
        ctx.panel,
        ctx.ledger,
        spec,
        cutoff_time=cutoff,
        context_role=role,
    )
    history_ends = [
        str(summary.get("history_end", ""))
        for summary in payload.get("component_summaries", {}).values()
        if isinstance(summary, dict) and str(summary.get("history_end", ""))
    ]
    if not validation.empty and not bool(validation["status"].astype(str).eq("PASS").all()):
        raise ValueError(f"canonical selection context validation failed for {ctx.dataset_key} cutoff={cutoff}")
    return text, {
        "selection_context_schema": str(payload.get("schema", "")),
        "selection_context_sha256": str(context_sha256(payload)),
        "selection_context_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "selection_context_cutoff": str(payload.get("cutoff_time", cutoff)),
        "selection_context_role": str(payload.get("context_role", role)),
        "selection_context_history_scope": str(payload.get("history_scope", "")),
        "selection_context_history_max": max(history_ends) if history_ends else "",
        "selection_context_builder": "caster.tasks.build_selection_context",
        "selection_context_validation_visible": False,
    }


def selection_context_text(
    ctx: DatasetContext,
    *,
    release: pd.DataFrame | None = None,
    origin_text: str = "",
) -> str:
    text, _metadata = selection_context_with_metadata(ctx, release=release, origin_text=origin_text)
    return text


def make_forecast_row(
    *,
    ctx: DatasetContext,
    event: pd.Series,
    ledger_idx: int,
    selected: pd.Series,
    method: str,
    predictions_cache: dict | None = None,
) -> tuple[dict[str, object], int]:
    entity_id = str(event[ctx.ledger_entity_col])
    component = str(event["component"])
    origin = pd.to_datetime(event["forecast_origin"], errors="coerce")
    target = pd.to_datetime(event["target_time"], errors="coerce")
    horizon = int(pd.to_numeric(event["horizon"], errors="raise"))
    values = group_history_values(ctx, entity_id, component, origin)
    if pd.isna(origin):
        raise ValueError(f"forecast_origin parse failed at row {ledger_idx}")
    if len(values) == 0:
        raise ValueError(f"no finite history at row {ledger_idx} entity={entity_id} component={component}")
    covariate_signal = group_covariate_signal(ctx, entity_id, component, origin)
    y_true = pd.to_numeric(pd.Series([event.get("observed_value")]), errors="coerce").iloc[0]
    if parse_bool(event.get("observed_mask", True)) and not np.isfinite(y_true):
        raise ValueError(f"observed_mask true but observed_value missing at row {ledger_idx}")
    if predictions_cache is not None:
        key = (entity_id, component, format_date(origin), str(event.get("mode", "")), horizon)
        if key in predictions_cache:
            pred = float(predictions_cache[key])
        else:
            raise RuntimeError(
                "pre-fit prediction missing; last-value substitution is forbidden: "
                f"model_id={selected.get('model_id', '')} key={key}"
            )
        var = _resid_var(values, pred)
    else:
        row_key = (ctx.dataset_key, entity_id, component, format_date(origin), format_date(target), horizon, event.get("forecast_id", ledger_idx))
        if recipe_name(selected) == "seasonal_naive":
            pred, var = forecast_recipe_for_event(
                ctx,
                selected,
                event,
                values,
                covariate_signal=covariate_signal,
            )
        elif strategy_from_event(event) == RECURSIVE_ROLLOUT:
            recursive_values = np.asarray(values, dtype=float)
            pred, var = float(recursive_values[-1]), _resid_var(recursive_values, float(recursive_values[-1]))
            for step in range(1, horizon + 1):
                pred, var = forecast_recipe(
                    selected,
                    recursive_values,
                    1,
                    row_key=(*row_key, "recursive_step", step),
                    covariate_signal=covariate_signal,
                )
                recursive_values = np.append(recursive_values, float(pred))
        else:
            pred, var = forecast_recipe_for_event(
                ctx,
                selected,
                event,
                values,
                covariate_signal=covariate_signal,
            )
    sigma = math.sqrt(max(float(var), 1e-6))
    row = {
        "dataset_key": ctx.dataset_key,
        "dataset": str(event["dataset"]) if "dataset" in ctx.ledger.columns else ctx.dataset,
        "method": method,
        "entity_id": entity_id,
        "forecast_origin": format_date(origin),
        "target_time": format_date(target),
        "component": component,
        "horizon": horizon,
        "y_true": float(y_true) if np.isfinite(y_true) else np.nan,
        "pred_mean": float(pred),
        "pred_lower_50": float(pred - Z50 * sigma),
        "pred_upper_50": float(pred + Z50 * sigma),
        "pred_lower_90": float(pred - Z90 * sigma),
        "pred_upper_90": float(pred + Z90 * sigma),
        "split": str(event["split"]) if "split" in ctx.ledger.columns else "NA",
        "selected_model_id": str(selected["model_id"]),
        "selected_family": str(selected.get("family", "")),
        "restart_group_id": f"{ctx.dataset_key}:{format_date(origin)}",
    }
    if recipe_name(selected) in {
        "covariate_drift",
        "covariate_dynamic_linear_trend",
    }:
        row.update(
            {
                "causal_covariate_signal": covariate_signal.value,
                "causal_covariate_feature_count": covariate_signal.feature_count,
                "causal_covariate_groups": "|".join(covariate_signal.groups),
                "causal_covariate_adjustment_applied": covariate_signal.feature_count > 0,
            }
        )
    for col in ctx.context_cols:
        row[col] = event[col]
    return row, int(len(values))


def finite_metric_check(metrics: pd.DataFrame) -> list[str]:
    failures = []
    for col in metrics.columns:
        if col in NON_NUMERIC_METRIC_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(metrics[col]):
            vals = pd.to_numeric(metrics[col], errors="coerce")
            if vals.notna().any() and not np.isfinite(vals.dropna()).all():
                failures.append(col)
    return failures


def write_standard_artifacts(
    *,
    out_dir: str | Path,
    forecast: pd.DataFrame,
    timing: dict[str, object],
    run_manifest: dict[str, object],
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(out / "forecast.csv", index=False)
    metric_forecast = forecast[
        ~forecast["split"].astype(str).str.lower().eq("embargo")
    ].copy()
    if metric_forecast.empty:
        raise RuntimeError("agentic run has no metric-eligible non-embargo forecast rows")
    metrics = summarize_forecasts(metric_forecast)
    failures = finite_metric_check(metrics)
    if failures:
        write_blocker_report(out, [{
            "dataset_key": "ALL",
            "ledger_row_index": "",
            "entity_id": "",
            "component": "",
            "forecast_origin": "",
            "reason": f"metrics contain non-finite values: {failures}",
        }])
        raise RuntimeError(f"agentic run produced non-finite metrics: {failures}")
    metrics.to_csv(out / "metrics.csv", index=False)
    timing = dict(timing)
    timing.setdefault("forecast_rows", int(len(forecast)))
    timing.setdefault("metric_rows", int(len(metrics)))
    (out / "timing.json").write_text(json.dumps(timing, indent=2, sort_keys=True), encoding="utf-8")
    run_manifest = dict(run_manifest)
    run_manifest.setdefault("forecast_rows", int(len(forecast)))
    run_manifest.setdefault("metric_rows", int(len(metrics)))
    run_manifest.setdefault(
        "embargo_forecast_rows",
        int(forecast["split"].astype(str).str.lower().eq("embargo").sum()),
    )
    run_manifest.setdefault("embargo_metric_rows", 0)
    run_manifest.setdefault("embargo_selection_eligible", False)
    run_manifest.setdefault("embargo_context_policy", "causal_history_at_or_before_forecast_origin")
    for key, value in forecast_strategy_manifest_fields(forecast).items():
        run_manifest.setdefault(key, value)
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out


def selection_log_row(
    *,
    stage: str,
    dataset_key: str,
    forecast_origin: str,
    selected: pd.Series,
    llm_payload: dict[str, Any],
) -> dict[str, object]:
    row = {
        "stage": stage,
        "dataset_key": dataset_key,
        "forecast_origin": forecast_origin,
        "selected_model_id": str(selected["model_id"]),
        "family": str(selected.get("family", "")),
        "candidate_type": str(selected.get("candidate_type", "")),
        "recipe": recipe_name(selected),
        "llm_reason": str(llm_payload.get("reason", "")),
    }
    extra_keys = (
        "selection_method",
        "candidate_order_strategy",
        "candidate_order",
        "candidate_choice_map",
        "candidate_choice_order",
        "selected_choice_id",
        "llm_proposed_choice_id",
        "llm_proposed_model_id",
        "position_neutrality_instruction",
        "fairness_instruction",
        "position_neutrality_instruction_present",
        "position_neutrality_control_trace_only",
        "selected_candidate_rank",
        "n_candidates",
        "selected_rank_fraction",
        "repair_compatibility_rechecked",
        "repair_target_risk_flags",
        "repair_remaining_target_risk_flags",
        "repair_target_risk_resolved",
        "repair_resolution_available",
        "repair_resolving_model_ids",
        "repair_skipped_reason",
        "repair_attempt_limit",
        "repair_attempt_count",
        "repair_attempts",
        "repair_attempt_consumption_policy",
        "repair_attempt_history_available",
        "repair_attempt_history_source",
        "fallback_to_initial_after_three_failed_repairs",
        "agent_selection_scope",
        "selection_replay_used",
        "selection_replay_key",
        "selection_replay_model_id_source",
        "selection_score_source",
        "validation_score_used",
        "validation_context_visible",
        "validation_context_kind",
        "validation_context_scoreboard_rows",
        "validation_guard_enabled",
        "validation_score_policy",
        "validation_ledger_policy",
        "llm_selected_model_id",
        "validation_rows",
        "validation_best_model_id",
        "validation_best_rmse",
        "validation_best_mae",
        "validation_best_rank_score",
        "validation_best_weighted_rmse",
        "validation_best_weighted_mae",
        "llm_selected_validation_rmse",
        "llm_selected_validation_mae",
        "llm_selected_validation_rank_score",
        "llm_selected_validation_weighted_rmse",
        "llm_selected_validation_weighted_mae",
        "llm_selected_validation_status",
        "validation_override",
        "validation_override_reason",
        "selection_context_schema",
        "selection_context_sha256",
        "selection_context_text_sha256",
        "selection_context_cutoff",
        "selection_context_role",
        "selection_context_history_scope",
        "selection_context_history_max",
        "selection_context_builder",
        "selection_context_validation_visible",
        "selection_context_profile",
        "combined_context_sha256",
        "combined_context_text_sha256",
        "base_selection_context_schema",
        "base_selection_context_sha256",
        "sequence_sketch_schema",
        "sequence_sketch_sha256",
        "sequence_sketch_validation_status",
        "sequence_sketch_validation_rows",
        "sequence_sketch_validation_sha256",
        "sequence_sketch_validation_checks",
        "selection_context_future_or_unreleased_target_values_visible",
    )
    for key in extra_keys:
        if key in llm_payload:
            row[key] = (
                json.dumps(llm_payload[key], sort_keys=True)
                if isinstance(llm_payload[key], (dict, list))
                else llm_payload[key]
            )
    return row


def write_registry_snapshot(registry: pd.DataFrame, out_dir: Path, registry_path: str | Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = out_dir / "candidate_registry_snapshot.csv"
    registry.to_csv(snapshot, index=False)
    return {
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(Path(registry_path)),
        "candidate_registry_snapshot": snapshot.name,
        "enabled_candidates": int(registry["enabled"].map(_coerce_bool).sum()),
        "restart_eligible_candidates": int(len(eligible_registry(registry))),
    }


def select_candidate_with_llm(
    *,
    engine: Any,
    stage: str,
    dataset_key: str,
    forecast_origin: str,
    candidates: pd.DataFrame,
    task: str,
    trace_rows: list[dict[str, object]],
    context_text: str = "",
) -> pd.Series:
    rows, choice_to_model = fair_registry_prompt_rows(
        candidates,
        dataset_key=dataset_key,
        forecast_origin=forecast_origin,
    )
    valid_choice_ids = [str(row["choice_id"]) for row in rows]
    semantic_error = ""
    last_call = None
    order_strategy = "stable_hash_by_dataset_origin_model_id"
    position_neutrality_instruction = (
        "Candidate order is a stable hash order for display only, not a ranking. "
        "Every candidate position has equal weight. Do not prefer candidates because "
        "they appear earlier or later in the list."
    )
    for _attempt in range(3):
        payload = {
            "task": task,
            "forecast_origin": forecast_origin,
            "dataset_context": context_text,
            "valid_choice_ids": valid_choice_ids,
            "candidates": rows,
            "required_response_schema": {
                "choice_id": "<exactly one valid_choice_ids entry>",
                "reason": "<short scientific reason>",
            },
            "strict_instruction": "Return JSON only and select exactly one supplied choice_id.",
        }
        if semantic_error:
            payload["previous_invalid_response"] = semantic_error
        call = engine.generate_json(
            stage=stage,
            system_prompt=(
                "You select one executable forecasting method. "
                "Choose exactly one opaque choice_id using only the dataset context and "
                "scientific_description. Return compact JSON only, with "
                "choice_id as the first key and a reason under 8 words."
            ),
            user_prompt=json.dumps(payload, sort_keys=True),
            valid_model_ids=valid_choice_ids,
            retries=4,
        )
        last_call = call
        trace_rows.append({
            "stage": call.stage,
            "dataset_key": dataset_key,
            "forecast_origin": forecast_origin,
            "model_path": call.model_path,
            "fallback_used": call.fallback_used,
            "fallback_reason": call.fallback_reason,
            "runtime_seconds": call.runtime_seconds,
            "prompt": call.prompt,
            "response_text": call.response_text,
            "candidate_order_strategy": order_strategy,
            "candidate_choice_map": choice_to_model,
            "position_neutrality_instruction": position_neutrality_instruction,
            "fairness_instruction": position_neutrality_instruction,
        })
        selected_choice_id = str(
            call.response_json.get("selected_choice_id")
            or call.response_json.get("choice_id")
            or call.response_json.get("selected_model_id")
            or call.response_json.get("selected")
            or ""
        ).strip()
        selected_id = choice_to_model.get(selected_choice_id, "")
        matches = candidates[candidates["model_id"].astype(str) == selected_id]
        if not matches.empty:
            selected = matches.iloc[0].copy()
            llm_payload = dict(call.response_json)
            for response_key in ("selected_model_id", "model_id", "selected"):
                llm_payload.pop(response_key, None)
            llm_payload["selected_choice_id"] = selected_choice_id
            rank = valid_choice_ids.index(selected_choice_id) + 1
            llm_payload.setdefault("selection_method", "qwen_json_fair_order_context")
            llm_payload.setdefault("candidate_order_strategy", order_strategy)
            llm_payload.setdefault(
                "candidate_order",
                [choice_to_model[choice_id] for choice_id in valid_choice_ids],
            )
            llm_payload.setdefault("candidate_choice_order", valid_choice_ids)
            llm_payload.setdefault("candidate_choice_map", choice_to_model)
            llm_payload.setdefault(
                "position_neutrality_instruction",
                position_neutrality_instruction,
            )
            llm_payload.setdefault("fairness_instruction", position_neutrality_instruction)
            llm_payload.setdefault("position_neutrality_instruction_present", False)
            llm_payload.setdefault("position_neutrality_control_trace_only", True)
            llm_payload.setdefault("selected_candidate_rank", int(rank))
            llm_payload.setdefault("n_candidates", int(len(valid_choice_ids)))
            llm_payload.setdefault(
                "selected_rank_fraction",
                float(rank / len(valid_choice_ids))
                if valid_choice_ids
                else float("nan"),
            )
            selected.attrs["llm_payload"] = llm_payload
            return selected
        semantic_error = (
            f"invalid choice_id={selected_choice_id!r}; "
            f"valid_choice_ids={valid_choice_ids}; response={call.response_text[:300]}"
        )
    raise ValueError(
        f"LLM selected an invalid choice after retries for dataset={dataset_key}: "
        f"{semantic_error}; last={getattr(last_call, 'response_json', {})}"
    )


def select_candidate_no_validation(
    *,
    engine: Any,
    stage: str,
    dataset_key: str,
    forecast_origin: str,
    candidates: pd.DataFrame,
    task: str,
    trace_rows: list[dict[str, object]],
    context_text: str = "",
    context_metadata: dict[str, object] | None = None,
    selection_replay: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> pd.Series:
    ""
    replay_used = selection_replay is not None
    if replay_used:
        selected = select_candidate_with_replay(
            stage=stage,
            dataset_key=dataset_key,
            forecast_origin=forecast_origin,
            candidates=candidates,
            trace_rows=trace_rows,
            selection_replay=selection_replay,
        )
    else:
        if engine is None:
            raise ValueError("engine is required unless selection_replay is provided")
        selected = select_candidate_with_llm(
            engine=engine,
            stage=stage,
            dataset_key=dataset_key,
            forecast_origin=forecast_origin,
            candidates=candidates,
            task=task,
            trace_rows=trace_rows,
            context_text=context_text,
        )

    payload = dict(selected.attrs.get("llm_payload", {}))
    payload["selection_method"] = "selection_replay_log_no_validation" if replay_used else "qwen_json_no_validation"
    payload["selection_score_source"] = "no_validation"
    payload["validation_score_used"] = False
    payload["validation_context_visible"] = False
    payload["validation_context_kind"] = ""
    payload["validation_context_scoreboard_rows"] = 0
    payload["validation_guard_enabled"] = False
    payload["validation_score_policy"] = "none"
    payload["validation_ledger_policy"] = "none"
    payload["llm_selected_model_id"] = str(selected["model_id"])
    payload["validation_rows"] = 0
    payload["validation_best_model_id"] = ""
    payload["validation_override"] = False
    payload["validation_override_reason"] = ""
    if context_metadata:
        payload.update(context_metadata)
    selected.attrs["llm_payload"] = payload
    return selected


def validation_tool_context(
    *,
    scoreboard: pd.DataFrame,
    score_ledger: pd.DataFrame,
    dataset_key: str,
    forecast_origin: str,
    max_models_per_group: int = 5,
) -> str:
    ""





    if scoreboard.empty:
        return (
            "model_validation_tool_observation: no usable no-test validation rows; "
            "prefer simple robust epidemic forecasters and avoid unscored high-variance choices."
        )
    view = scoreboard.copy()
    if "score_group" not in view.columns:
        view.insert(0, "score_group", "overall")
    if "rank_score" not in view.columns:
        view["rank_score"] = pd.to_numeric(view.get("rmse", pd.Series(dtype=float)), errors="coerce")
    view["rank_score"] = pd.to_numeric(view["rank_score"], errors="coerce")
    view.loc[~np.isfinite(view["rank_score"]), "rank_score"] = np.inf
    groups: list[dict[str, object]] = []
    for group_name in ["overall", "short", "long"]:
        group = view[view["score_group"].astype(str) == group_name].copy()
        if group.empty:
            continue
        group = group.sort_values(["rank_score", "model_id"], kind="mergesort")
        top: list[dict[str, object]] = []
        for _, row in group.head(int(max_models_per_group)).iterrows():
            top.append({
                "model_id": str(row.get("model_id", "")),
                "rank_score": None if not np.isfinite(float(row.get("rank_score", np.inf))) else round(float(row.get("rank_score", np.inf)), 4),
                "rmse": None if not np.isfinite(float(row.get("rmse", np.nan))) else round(float(row.get("rmse", np.nan)), 4),
                "mae": None if not np.isfinite(float(row.get("mae", np.nan))) else round(float(row.get("mae", np.nan)), 4),
                "n_eval": int(pd.to_numeric(pd.Series([row.get("n_eval", 0)]), errors="coerce").fillna(0).iloc[0]),
                "status": str(row.get("score_status", "")),
            })
        groups.append({"score_group": group_name, "top_candidates": top})
    release_mix = {}
    if not score_ledger.empty:
        modes = score_ledger["mode"] if "mode" in score_ledger.columns else pd.Series([""] * len(score_ledger), index=score_ledger.index)
        horizons = score_ledger["horizon"] if "horizon" in score_ledger.columns else pd.Series([0] * len(score_ledger), index=score_ledger.index)
        horizon_groups = [forecast_horizon_group(dataset_key, mode, horizon) for mode, horizon in zip(modes, horizons)]
        release_mix = {str(k): int(v) for k, v in pd.Series(horizon_groups).value_counts().to_dict().items()}
    payload = {
        "dataset_key": dataset_key,
        "forecast_origin": forecast_origin,
        "validation_rows": int(len(score_ledger)),
        "validation_horizon_mix": release_mix,
        "selector_instruction": (
            "Use these no-test validation/backtest tool results as primary evidence. "
            "For releases with long-horizon rows or outbreak growth, avoid candidates with no score, unstable scale, "
            "or much worse long-group rank_score."
        ),
        "groups": groups,
    }
    return "model_validation_tool_observation=" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def load_selection_replay_log(path: str | Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or str(path).strip() == "":
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, keep_default_na=False)
    required = {"dataset_key", "forecast_origin"}
    if not required.issubset(df.columns):
        raise ValueError(f"selection replay log missing required columns {sorted(required)}: {p}")
    replay: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in df.iterrows():
        replay[(str(row["dataset_key"]), str(row["forecast_origin"]))] = row.to_dict()
    return replay


def select_candidate_with_replay(
    *,
    stage: str,
    dataset_key: str,
    forecast_origin: str,
    candidates: pd.DataFrame,
    trace_rows: list[dict[str, object]],
    selection_replay: dict[tuple[str, str], dict[str, Any]],
) -> pd.Series:
    key = (str(dataset_key), str(forecast_origin))
    if key not in selection_replay:
        raise KeyError(f"selection replay log has no row for dataset_key={dataset_key} forecast_origin={forecast_origin}")
    replay_row = dict(selection_replay[key])
    replay_fields = (
        ("initial_selected_model_id", "llm_selected_model_id", "selected_model_id")
        if stage == "react_select"
        else (
            "final_selected_model_id",
            "selected_model_id",
            "llm_selected_model_id",
        )
        if stage == "react_repair_select"
        else ("llm_selected_model_id", "selected_model_id")
    )
    replay_id = ""
    replay_id_source = ""
    for field in replay_fields:
        value = str(replay_row.get(field, "") or "").strip()
        if value:
            replay_id = value
            replay_id_source = field
            break
    if not replay_id:
        raise ValueError(
            f"selection replay row has none of {list(replay_fields)} for {key}"
        )
    matches = candidates[candidates["model_id"].astype(str) == replay_id]
    if matches.empty:
        raise ValueError(f"replayed model_id={replay_id!r} not found in executable candidates for {key}")
    payload = {
        "selection_method": "selection_replay_log",
        "selection_replay_used": True,
        "selection_replay_key": f"{dataset_key}|{forecast_origin}",
        "selection_replay_model_id_source": replay_id_source,
        "selected_model_id": replay_id,
        "llm_selected_model_id": replay_id,
        "reason": str(replay_row.get("llm_reason", "replayed_previous_llm_selection")),
    }
    trace_rows.append({
        "stage": stage,
        "dataset_key": dataset_key,
        "forecast_origin": forecast_origin,
        "model_path": "selection_replay_log",
        "fallback_used": False,
        "fallback_reason": "",
        "runtime_seconds": 0.0,
        "prompt": "",
        "response_text": json.dumps(payload, sort_keys=True),
    })
    selected = matches.iloc[0].copy()
    selected.attrs["llm_payload"] = payload
    return selected



def _observed_score_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    ""
    if ledger.empty:
        return ledger.copy()
    observed = ledger.copy()
    if "observed_mask" in observed.columns:
        mask = observed["observed_mask"].map(parse_bool)
    else:
        mask = pd.Series(True, index=observed.index)
    y = pd.to_numeric(observed.get("observed_value", pd.Series(dtype=float)), errors="coerce")
    return observed[mask & np.isfinite(y)].copy()


def _sort_validation_ledger(ctx: DatasetContext, ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or "target_time" not in ledger.columns:
        return ledger.copy()
    entity_col = ctx.ledger_entity_col if ctx.ledger_entity_col in ledger.columns else "component"
    out = ledger.assign(_target_sort=pd.to_datetime(ledger["target_time"], errors="coerce"))
    out = out.sort_values(["_target_sort", entity_col, "component"], kind="mergesort")
    return out.drop(columns=["_target_sort"])


def _recent_origin_stratified_ledger(
    ctx: DatasetContext,
    ledger: pd.DataFrame,
    *,
    origin_text: str,
    max_rows: int,
    lookback_days: int = 28,
) -> pd.DataFrame:
    ""








    if ledger.empty or not origin_text:
        return _sort_validation_ledger(ctx, ledger).tail(int(max_rows)).copy() if max_rows > 0 else _sort_validation_ledger(ctx, ledger)
    origin = pd.to_datetime(origin_text, errors="coerce")
    if pd.isna(origin):
        return ledger.iloc[0:0].copy()
    if "forecast_origin" not in ledger.columns or "target_time" not in ledger.columns:
        return _sort_validation_ledger(ctx, ledger).tail(int(max_rows)).copy() if max_rows > 0 else _sort_validation_ledger(ctx, ledger)

    out = ledger.copy()
    forecast_origin = pd.to_datetime(out["forecast_origin"], errors="coerce")
    target_time = pd.to_datetime(out["target_time"], errors="coerce")
    out = out[(forecast_origin < origin) & (target_time <= origin)].copy()
    if out.empty:
        return out

    forecast_origin = pd.to_datetime(out["forecast_origin"], errors="coerce")
    recent = out[forecast_origin >= origin - pd.Timedelta(days=int(lookback_days))].copy()
    if not recent.empty:
        out = recent

    out = _ledger_with_horizon_group(ctx, out)
    out["_forecast_origin_sort"] = pd.to_datetime(out["forecast_origin"], errors="coerce")
    out["_target_time_sort"] = pd.to_datetime(out["target_time"], errors="coerce")
    if "horizon" in out.columns:
        out["_horizon_sort"] = pd.to_numeric(out["horizon"], errors="coerce").fillna(-1).astype(int)
    else:
        out["_horizon_sort"] = -1

    if max_rows > 0 and len(out) > int(max_rows):
        group_cols = ["horizon_group", "_horizon_sort"]
        groups = list(out.groupby(group_cols, dropna=False, sort=True))
        per_group = max(1, int(math.ceil(float(max_rows) / max(len(groups), 1))))
        pieces: list[pd.DataFrame] = []
        for _key, group in groups:
            group = group.sort_values(
                ["_forecast_origin_sort", "_target_time_sort", ctx.ledger_entity_col if ctx.ledger_entity_col in group.columns else "component"],
                ascending=[False, False, True],
                kind="mergesort",
            )
            pieces.append(group.head(per_group))
        selected = pd.concat(pieces, ignore_index=False, sort=False) if pieces else out.iloc[0:0].copy()
        if len(selected) < int(max_rows):
            remaining = out.drop(index=selected.index, errors="ignore").sort_values(
                ["_forecast_origin_sort", "_target_time_sort", ctx.ledger_entity_col if ctx.ledger_entity_col in out.columns else "component"],
                ascending=[False, False, True],
                kind="mergesort",
            )
            selected = pd.concat([selected, remaining.head(int(max_rows) - len(selected))], ignore_index=False, sort=False)
        out = selected.head(int(max_rows)).copy()

    out = out.sort_values(["_forecast_origin_sort", "_target_time_sort", "horizon_group", "_horizon_sort"], kind="mergesort")
    return out.drop(columns=["horizon_group", "_forecast_origin_sort", "_target_time_sort", "_horizon_sort"], errors="ignore")


def validation_scoring_ledger(
    ctx: DatasetContext,
    *,
    origin_text: str = "",
    max_rows: int = 512,
    ledger_policy: str = "target_tail",
) -> pd.DataFrame:
    ""








    if ledger_policy not in VALIDATION_LEDGER_POLICIES:
        raise ValueError(f"unknown validation ledger policy: {ledger_policy}")
    ledger = _observed_score_rows(ctx.ledger)
    if ledger.empty:
        return ledger
    if "split" in ledger.columns:
        split = ledger["split"].astype(str).str.lower()
        development = ledger[split.isin(["train", "val"])].copy()
        if not development.empty:
            ledger = development
    if origin_text:
        origin = pd.to_datetime(origin_text, errors="coerce")
        if pd.isna(origin):
            return ledger.iloc[0:0].copy()
        target_time = pd.to_datetime(ledger.get("target_time"), errors="coerce")
        ledger = ledger[target_time <= origin].copy()
        if ledger.empty:
            return ledger
    if ledger_policy == "recent_origin_stratified" and origin_text:
        return _recent_origin_stratified_ledger(ctx, ledger, origin_text=origin_text, max_rows=max_rows)
    if "split" in ledger.columns:
        split = ledger["split"].astype(str).str.lower()
        preferred = ledger[split == "val"].copy()
        if preferred.empty and origin_text:
            preferred = ledger[split.isin(["train", "val"])].copy()
        if preferred.empty:
            preferred = ledger[split == "train"].copy()
        if not preferred.empty:
            ledger = preferred
    if "target_time" in ledger.columns:
        ledger = _sort_validation_ledger(ctx, ledger)
    if max_rows > 0 and len(ledger) > int(max_rows):
        ledger = ledger.tail(int(max_rows)).copy()
    return ledger


def normalize_dataset_key(value: object) -> str:
    text = str(value)
    if text.startswith("benchmark_a"):
        return "benchmark_a"
    if text.startswith("benchmark_b"):
        return "benchmark_b"
    return text


def forecast_horizon_group(dataset_key: object, mode: object, horizon: object) -> str:
    ""
    dataset = normalize_dataset_key(dataset_key)
    try:
        h = int(float(horizon))
    except Exception:
        h = 0
    if dataset == "benchmark_b":
        return "short" if h in {1, 2} else "long"
    return "short" if h in {1, 3} else "long"


def _ledger_with_horizon_group(ctx: DatasetContext, ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        out = ledger.copy()
        out["horizon_group"] = pd.Series(dtype=str)
        return out
    out = ledger.copy()
    modes = out["mode"] if "mode" in out.columns else pd.Series([""] * len(out), index=out.index)
    horizons = out["horizon"] if "horizon" in out.columns else pd.Series([0] * len(out), index=out.index)
    out["horizon_group"] = [
        forecast_horizon_group(ctx.dataset_key, mode, horizon)
        for mode, horizon in zip(modes, horizons)
    ]
    return out


def _candidate_validation_metrics(
    *,
    ctx: DatasetContext,
    candidate: pd.Series,
    score_ledger: pd.DataFrame,
    score_policy: str = "plain_rmse",
    score_origin_text: str = "",
) -> dict[str, object]:
    if score_policy not in VALIDATION_SCORE_POLICIES:
        raise ValueError(f"unknown validation score policy: {score_policy}")
    model_id = str(candidate.get("model_id", ""))
    recipe = recipe_name(candidate)
    base = {
        "model_id": model_id,
        "family": str(candidate.get("family", "")),
        "candidate_type": str(candidate.get("candidate_type", "")),
        "recipe": recipe,
        "validation_score_policy": score_policy,
    }
    if score_ledger.empty:
        return {**base, "n_eval": 0, "mae": math.nan, "rmse": math.nan, "score_status": "no_validation_rows"}
    if is_stateful(candidate):
        return {
            **base,
            "n_eval": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "score_status": "not_scored_stateful_dependency_guard",
        }
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    weighted_abs_errors: list[float] = []
    weighted_sq_errors: list[float] = []
    weights: list[float] = []
    failures = 0
    history_cache: dict[tuple[str, str, str], np.ndarray] = {}
    score_origin = pd.to_datetime(score_origin_text, errors="coerce") if score_origin_text else pd.NaT
    if pd.isna(score_origin) and not score_ledger.empty and "target_time" in score_ledger.columns:
        target_times = pd.to_datetime(score_ledger["target_time"], errors="coerce").dropna()
        if not target_times.empty:
            score_origin = pd.Timestamp(target_times.max())
    for ledger_idx, event in score_ledger.iterrows():
        try:
            entity_id = str(event.get(ctx.ledger_entity_col, ""))
            component = str(event.get("component", ""))
            event_origin = pd.to_datetime(event.get("forecast_origin"), errors="coerce")
            if pd.isna(event_origin):
                failures += 1
                continue
            cache_key = (entity_id, component, format_date(event_origin))
            if cache_key not in history_cache:
                history_cache[cache_key] = group_history_values(ctx, entity_id, component, event_origin)
            values = history_cache[cache_key]
            if len(values) == 0:
                failures += 1
                continue
            horizon = int(pd.to_numeric(pd.Series([event.get("horizon")]), errors="raise").iloc[0])
            pred, _var = forecast_recipe_for_event(
                ctx,
                candidate,
                event,
                values,
                covariate_signal=group_covariate_signal(
                    ctx, entity_id, component, event_origin
                ),
            )
            y_true = float(pd.to_numeric(pd.Series([event.get("observed_value")]), errors="coerce").iloc[0])
            if not (np.isfinite(y_true) and np.isfinite(pred)):
                failures += 1
                continue
            err = pred - y_true
            abs_errors.append(abs(err))
            sq_errors.append(err * err)
            if score_policy == "recent_scale_growth_rmse":
                weight = _validation_event_weight(ctx=ctx, event=event, y_true=y_true, score_origin=score_origin)
                weights.append(weight)
                weighted_abs_errors.append(abs(err) * weight)
                weighted_sq_errors.append(err * err * weight)
        except Exception:
            failures += 1
    if not abs_errors:
        return {
            **base,
            "n_eval": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "score_status": f"no_valid_probe_predictions;failures={failures}",
        }
    out = {
        **base,
        "n_eval": int(len(abs_errors)),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(math.sqrt(float(np.mean(sq_errors)))),
        "score_status": "ok" if failures == 0 else f"ok_with_failures={failures}",
    }
    if score_policy == "recent_scale_growth_rmse":
        weight_sum = float(np.sum(weights))
        if weight_sum > 0 and weighted_sq_errors:
            out["weight_sum"] = weight_sum
            out["weighted_mae"] = float(np.sum(weighted_abs_errors) / weight_sum)
            out["weighted_rmse"] = float(math.sqrt(float(np.sum(weighted_sq_errors) / weight_sum)))
        else:
            out["weight_sum"] = 0.0
            out["weighted_mae"] = math.nan
            out["weighted_rmse"] = math.nan
    return out


def _validation_event_weight(
    *,
    ctx: DatasetContext,
    event: pd.Series,
    y_true: float,
    score_origin: pd.Timestamp,
) -> float:
    target_time = pd.to_datetime(event.get("target_time"), errors="coerce")
    if pd.isna(score_origin) or pd.isna(target_time):
        recent_weight = 1.0
    else:
        age_days = max(float((score_origin - target_time).days), 0.0)
        recent_weight = math.exp(-age_days / 21.0)
    entity_col = ctx.ledger_entity_col
    event_origin = pd.to_datetime(event.get("forecast_origin"), errors="coerce")
    values = np.asarray([], dtype=float)
    if not pd.isna(event_origin):
        values = group_history_values(ctx, str(event.get(entity_col, "")), str(event.get("component", "")), event_origin)
    if len(values):
        last_value = max(float(values[-1]), 0.0)
        try:
            cadence = max(int(ctx.manifest_row.get("cadence_days", 1)), 1)
        except Exception:
            cadence = 1
        lookback_steps = max(1, int(round(7.0 / float(cadence))))
        lookback_idx = max(0, len(values) - 1 - lookback_steps)
        lookback_value = max(float(values[lookback_idx]), 0.0)
    else:
        last_value = max(float(y_true), 0.0) if np.isfinite(y_true) else 0.0
        lookback_value = last_value
    scale_base = max(float(y_true) if np.isfinite(y_true) else 0.0, last_value, 0.0)
    scale_weight = min(math.sqrt(scale_base + 1.0), 50.0)
    growth = math.log1p(last_value) - math.log1p(lookback_value)
    growth_weight = min(1.0 + 2.0 * max(growth, 0.0), 5.0)
    weight = recent_weight * scale_weight * growth_weight
    return float(weight) if np.isfinite(weight) and weight > 0 else 1.0


def score_candidate_registry(
    *,
    ctx: DatasetContext,
    candidates: pd.DataFrame,
    score_ledger: pd.DataFrame,
    score_policy: str = "plain_rmse",
    score_origin_text: str = "",
) -> pd.DataFrame:
    rows = [
        _candidate_validation_metrics(
            ctx=ctx,
            candidate=row,
            score_ledger=score_ledger,
            score_policy=score_policy,
            score_origin_text=score_origin_text,
        )
        for _, row in candidates.iterrows()
    ]
    if not rows:
        return pd.DataFrame(columns=["model_id", "family", "candidate_type", "recipe", "n_eval", "mae", "rmse", "score_status"])
    scores = pd.DataFrame(rows)
    rank_col = "weighted_rmse" if score_policy == "recent_scale_growth_rmse" else "rmse"
    if rank_col not in scores.columns:
        scores[rank_col] = np.nan
    if score_policy == "recent_scale_growth_rmse":
        for col in ("weighted_mae", "weight_sum"):
            if col not in scores.columns:
                scores[col] = np.nan
    scores["rank_score_metric"] = rank_col
    scores["rank_score"] = pd.to_numeric(scores[rank_col], errors="coerce")
    scores.loc[~np.isfinite(scores["rank_score"]), "rank_score"] = np.inf
    scores = scores.sort_values(["rank_score", "mae", "model_id"], kind="mergesort").reset_index(drop=True)
    scores["validation_rank"] = np.arange(1, len(scores) + 1)
    return scores


def score_candidate_registry_by_horizon(
    *,
    ctx: DatasetContext,
    candidates: pd.DataFrame,
    score_ledger: pd.DataFrame,
    score_policy: str = "plain_rmse",
    score_origin_text: str = "",
) -> pd.DataFrame:
    ""





    frames: list[pd.DataFrame] = []

    def add_group(name: str, group_ledger: pd.DataFrame) -> None:
        scored = score_candidate_registry(
            ctx=ctx,
            candidates=candidates,
            score_ledger=group_ledger,
            score_policy=score_policy,
            score_origin_text=score_origin_text,
        )
        if scored.empty:
            return
        scored = scored.copy()
        scored.insert(0, "score_group", name)
        scored.insert(1, "score_group_rows", int(len(group_ledger)))
        frames.append(scored)

    add_group("overall", score_ledger)
    grouped = _ledger_with_horizon_group(ctx, score_ledger)
    for group_name in ("short", "long"):
        subset = grouped[grouped["horizon_group"].astype(str) == group_name].copy()
        if not subset.empty:
            subset = subset.drop(columns=["horizon_group"], errors="ignore")
            add_group(group_name, subset)
    if frames:
        return pd.concat(frames, ignore_index=True, sort=False)
    return pd.DataFrame(columns=[
        "score_group",
        "score_group_rows",
        "model_id",
        "family",
        "candidate_type",
        "recipe",
        "n_eval",
        "mae",
        "rmse",
        "score_status",
        "rank_score",
        "validation_rank",
    ])




def _origin_growth_probe_ledger(
    ctx: DatasetContext,
    release_ledger: pd.DataFrame,
    *,
    origin_text: str,
    max_rows: int = 256,
) -> pd.DataFrame:
    ""
    if release_ledger.empty:
        return release_ledger.copy()
    origin = pd.to_datetime(origin_text, errors="coerce")
    if pd.isna(origin):
        return release_ledger.iloc[0:0].copy()
    rows: list[dict[str, float | int]] = []
    entity_col = ctx.ledger_entity_col
    for idx, event in release_ledger.iterrows():
        try:
            horizon = int(float(event.get("horizon", 0)))
        except Exception:
            continue
        entity = str(event.get(entity_col, event.get("entity_id", "")))
        component = str(event.get("component", ""))
        series = ctx.history_index.get((entity, component))
        if series is None:
            continue
        values = values_until_origin(series, origin)
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 4:
            continue
        if "_caster_recent_slope" in event and "_caster_growth_scale" in event and "_caster_last_value" in event:
            recent_slope = float(event.get("_caster_recent_slope", 0.0))
            scale = max(1.0, float(event.get("_caster_growth_scale", 1.0)))
            last = float(event.get("_caster_last_value", values[-1]))
            growth_signal = float(event.get("_caster_growth_signal", recent_slope / scale))
        else:
            recent = values[-min(14, len(values)):]
            last = float(values[-1])
            scale = max(1.0, float(np.nanmedian(np.abs(recent))), abs(last) * 0.5)
            recent_slope = _linear_slope(values, 0.0)
            growth_signal = float(recent_slope) / scale
        if growth_signal < 0.03 and recent_slope < 1.0:
            continue
        rows.append({
            "_idx": idx,
            "_caster_growth_signal": float(growth_signal),
            "_caster_recent_slope": float(recent_slope),
            "_caster_growth_scale": float(scale),
            "_caster_last_value": float(last),
            "_caster_horizon": int(horizon),
        })
    if not rows:
        return release_ledger.iloc[0:0].copy()
    meta = pd.DataFrame(rows).set_index("_idx")
    chosen = meta.sort_values(["_caster_growth_signal", "_caster_horizon"], ascending=[False, False], kind="mergesort")
    if max_rows > 0 and len(chosen) > int(max_rows):
        chosen = chosen.head(int(max_rows))
    out = release_ledger.loc[chosen.index].copy()
    for col in chosen.columns:
        out[col] = chosen[col]
    return out

def _candidate_origin_growth_metrics(
    *,
    ctx: DatasetContext,
    candidate: pd.Series,
    release_ledger: pd.DataFrame,
    origin_text: str,
) -> dict[str, object]:
    ""





    model_id = str(candidate.get("model_id", ""))
    base = {
        "model_id": model_id,
        "origin_growth_guard_rows": int(len(release_ledger)),
        "origin_growth_guard_n_eval": 0,
        "origin_growth_guard_mean": 0.0,
        "origin_growth_guard_p90": 0.0,
        "origin_growth_guard_p95": 0.0,
        "origin_growth_guard_max": 0.0,
        "origin_growth_guard_alert_rate": 0.0,
        "origin_growth_guard_status": "no_growth_signal",
    }
    if release_ledger.empty:
        return {**base, "origin_growth_guard_status": "no_release_rows"}
    origin = pd.to_datetime(origin_text, errors="coerce")
    if pd.isna(origin):
        return {**base, "origin_growth_guard_status": "bad_origin"}
    penalties: list[float] = []
    evaluated = 0
    entity_col = ctx.ledger_entity_col
    for _, event in release_ledger.iterrows():
        try:
            horizon = int(float(event.get("horizon", 0)))
        except Exception:
            continue
        if horizon <= 0:
            continue
        entity = str(event.get(entity_col, event.get("entity_id", "")))
        component = str(event.get("component", ""))
        series = ctx.history_index.get((entity, component))
        if series is None:
            continue
        values = values_until_origin(series, origin)
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 4:
            continue
        recent = values[-min(14, len(values)):]
        last = float(values[-1])
        scale = max(1.0, float(np.nanmedian(np.abs(recent))), abs(last) * 0.5)
        recent_slope = _linear_slope(values, 0.0)
        growth_signal = float(recent_slope) / scale
        if growth_signal < 0.03 and recent_slope < 1.0:
            continue
        try:
            pred, _var = forecast_recipe_for_event(
                ctx,
                candidate,
                event,
                values,
                covariate_signal=group_covariate_signal(
                    ctx, entity, component, origin
                ),
            )
        except Exception:
            continue
        expected_delta = recent_slope * float(horizon)
        pred_delta = float(pred) - last
        under_tracking = max(0.0, (0.60 * expected_delta - pred_delta) / scale)
        over_tracking = max(0.0, (pred_delta - 3.00 * expected_delta) / scale)
        weight = min(1.0, max(0.2, growth_signal * 8.0)) * min(1.0, float(horizon) / 7.0)
        penalties.append(float(weight * max(under_tracking, over_tracking)))
        evaluated += 1
    if not penalties:
        return base
    arr = np.asarray(penalties, dtype=float)
    return {
        **base,
        "origin_growth_guard_n_eval": int(evaluated),
        "origin_growth_guard_mean": float(np.mean(arr)),
        "origin_growth_guard_p90": float(np.percentile(arr, 90)),
        "origin_growth_guard_p95": float(np.percentile(arr, 95)),
        "origin_growth_guard_max": float(np.max(arr)),
        "origin_growth_guard_alert_rate": float(np.mean(arr > 1.0)),
        "origin_growth_guard_status": "ok",
    }


def score_candidate_origin_growth_guard(
    *,
    ctx: DatasetContext,
    candidates: pd.DataFrame,
    release_ledger: pd.DataFrame,
    origin_text: str,
    max_rows: int = 256,
) -> pd.DataFrame:
    ""
    frames: list[pd.DataFrame] = []

    def add_group(name: str, group_ledger: pd.DataFrame) -> None:
        probe_ledger = _origin_growth_probe_ledger(
            ctx,
            group_ledger,
            origin_text=origin_text,
            max_rows=max_rows,
        )
        rows = [
            _candidate_origin_growth_metrics(
                ctx=ctx,
                candidate=row,
                release_ledger=probe_ledger,
                origin_text=origin_text,
            )
            for _, row in candidates.iterrows()
        ]
        if not rows:
            return
        out = pd.DataFrame(rows)
        out.insert(0, "score_group", name)
        frames.append(out)

    add_group("overall", release_ledger)
    grouped = _ledger_with_horizon_group(ctx, release_ledger)
    for group_name in ("short", "long"):
        subset = grouped[grouped["horizon_group"].astype(str) == group_name].copy()
        if not subset.empty:
            subset = subset.drop(columns=["horizon_group"], errors="ignore")
            add_group(group_name, subset)
    if frames:
        return pd.concat(frames, ignore_index=True, sort=False)
    return pd.DataFrame()


def merge_origin_growth_guard(
    scoreboard: pd.DataFrame,
    guard: pd.DataFrame,
    *,
    penalty_weight: float = 1.5,
) -> pd.DataFrame:
    ""
    if scoreboard.empty:
        return scoreboard
    out = scoreboard.copy()
    if guard.empty:
        out["origin_growth_guard_weight"] = float(penalty_weight)
        out["origin_growth_guard_p95"] = 0.0
        out["guarded_rank_score"] = pd.to_numeric(out.get("rank_score", np.nan), errors="coerce")
        return out
    merge_cols = [
        "score_group",
        "model_id",
        "origin_growth_guard_rows",
        "origin_growth_guard_n_eval",
        "origin_growth_guard_mean",
        "origin_growth_guard_p90",
        "origin_growth_guard_p95",
        "origin_growth_guard_max",
        "origin_growth_guard_alert_rate",
        "origin_growth_guard_status",
    ]
    guard_view = guard[[col for col in merge_cols if col in guard.columns]].copy()
    out = out.merge(guard_view, on=["score_group", "model_id"], how="left")
    for col in (
        "origin_growth_guard_rows",
        "origin_growth_guard_n_eval",
        "origin_growth_guard_mean",
        "origin_growth_guard_p90",
        "origin_growth_guard_p95",
        "origin_growth_guard_max",
        "origin_growth_guard_alert_rate",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["origin_growth_guard_status"] = out.get("origin_growth_guard_status", "not_evaluated")
    if hasattr(out["origin_growth_guard_status"], "fillna"):
        out["origin_growth_guard_status"] = out["origin_growth_guard_status"].fillna("not_evaluated")
    out["origin_growth_guard_weight"] = float(penalty_weight)
    base_rank = pd.to_numeric(out.get("rank_score", np.nan), errors="coerce")
    penalty = pd.to_numeric(out.get("origin_growth_guard_p95", 0.0), errors="coerce").fillna(0.0)
    out["guarded_rank_score"] = base_rank + float(penalty_weight) * penalty
    return out


def _best_validation_model_id(scoreboard: pd.DataFrame, *, rank_column: str = "rank_score") -> str:
    if scoreboard.empty or "model_id" not in scoreboard.columns:
        return ""
    rank_key = rank_column if rank_column in scoreboard.columns else "rank_score"
    if rank_key not in scoreboard.columns:
        return ""
    usable = scoreboard.copy()
    usable["_rank_key"] = pd.to_numeric(usable[rank_key], errors="coerce")
    usable = usable[np.isfinite(usable["_rank_key"])]
    usable = usable[pd.to_numeric(usable.get("n_eval", 0), errors="coerce").fillna(0) > 0]
    if usable.empty:
        return ""
    tie_cols = ["_rank_key"]
    if "rank_score" in usable.columns:
        tie_cols.append("rank_score")
    if "mae" in usable.columns:
        tie_cols.append("mae")
    tie_cols.append("model_id")
    usable = usable.sort_values(tie_cols, kind="mergesort")
    return str(usable.iloc[0]["model_id"])


def best_validation_model_id(
    scoreboard: pd.DataFrame,
    *,
    score_group: str | None = None,
    rank_column: str = "rank_score",
) -> str:
    if score_group is None:
        return _best_validation_model_id(scoreboard, rank_column=rank_column)
    if scoreboard.empty or "score_group" not in scoreboard.columns:
        return ""
    return _best_validation_model_id(
        scoreboard[scoreboard["score_group"].astype(str) == str(score_group)].copy(),
        rank_column=rank_column,
    )


def validation_model_summary(scoreboard: pd.DataFrame, model_id: str, *, score_group: str | None = None) -> dict[str, object]:
    if scoreboard.empty or not model_id:
        return {}
    view = scoreboard
    if score_group is not None and "score_group" in view.columns:
        view = view[view["score_group"].astype(str) == str(score_group)]
    rows = view[view["model_id"].astype(str) == str(model_id)]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {
        "model_id": str(row.get("model_id", "")),
        "n_eval": int(pd.to_numeric(pd.Series([row.get("n_eval", 0)]), errors="coerce").fillna(0).iloc[0]),
        "mae": float(row["mae"]) if np.isfinite(float(row.get("mae", np.nan))) else math.nan,
        "rmse": float(row["rmse"]) if np.isfinite(float(row.get("rmse", np.nan))) else math.nan,
        "weighted_mae": float(row["weighted_mae"]) if "weighted_mae" in row and np.isfinite(float(row.get("weighted_mae", np.nan))) else math.nan,
        "weighted_rmse": float(row["weighted_rmse"]) if "weighted_rmse" in row and np.isfinite(float(row.get("weighted_rmse", np.nan))) else math.nan,
        "rank_score": float(row["rank_score"]) if np.isfinite(float(row.get("rank_score", np.nan))) else math.nan,
        "guarded_rank_score": float(row["guarded_rank_score"]) if "guarded_rank_score" in row and np.isfinite(float(row.get("guarded_rank_score", np.nan))) else math.nan,
        "origin_growth_guard_p95": float(row["origin_growth_guard_p95"]) if "origin_growth_guard_p95" in row and np.isfinite(float(row.get("origin_growth_guard_p95", np.nan))) else math.nan,
        "rank": int(pd.to_numeric(pd.Series([row.get("validation_rank", 0)]), errors="coerce").fillna(0).iloc[0]),
        "status": str(row.get("score_status", "")),
    }


def qwen_config(engine: Any) -> dict[str, object]:
    return {
        "llm_model_path": str(getattr(engine, "active_model_path", "unknown")),
        "llm_primary_required": bool(getattr(engine, "primary_required", False)),
        "llm_required_model_path": str(getattr(engine, "required_model_path", "")),
        "llm_cuda_required": bool(getattr(engine, "require_cuda", False)),
        "llm_cuda_available": bool(getattr(engine, "cuda_available", False)),
        "llm_model_device": str(getattr(engine, "model_device", "")),
        "llm_fallback_used": bool(getattr(engine, "fallback_used", False)),
        "llm_fallback_reason": str(getattr(engine, "fallback_reason", "")),
        "llm_fallback_allowed": bool(getattr(engine, "allow_fallback", False)),
        "llm_model_load_seconds": float(getattr(engine, "load_seconds", 0.0)),
    }


def total_trace_runtime(trace_rows: list[dict[str, object]]) -> float:
    return round(float(sum(float(row.get("runtime_seconds", 0.0)) for row in trace_rows)), 6)
