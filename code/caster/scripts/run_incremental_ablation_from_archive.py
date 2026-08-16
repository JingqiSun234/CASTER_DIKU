from __future__ import annotations

from argparse import ArgumentParser
import hashlib
from math import log, pi
from pathlib import Path
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import numpy as np
import pandas as pd

from caster.utils import RuntimeLogger, write_timing_log
from caster.bridge import (
    COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T,
    alternate_ARCHIVE_MOMENT,
    PREDICTIVE_CONTRACTS,
    read_bridge_config,
    score_archive_rows,
    score_draw_rows,
)
from caster.filter import (
    availability_validation_metadata,
    compute_log_evidence,
    evidence_availability_by_model,
    initialize_hierarchical_weights,
    native_forecast_rows,
    posterior_predictive_readout,
    posterior_predictive_readout_asof,
    summarize_model_distribution,
    update_outer_weights,
)
from run_caster_from_archive import _validate_archive_contract
from result_export_core import (
    ASOF_LTE_POLICY,
    DIAGNOSTIC_MOMENT_MATCHED_BASIS,
    EXACT_DRAW_KERNEL_NLL_SCORE_BASIS,
    EXACT_NLL_SCORE_BASIS,
    _bridge_asof_mixture_scores,
    _bridge_readout_scores,
    _draw_kernel_asof_mixture_scores,
    _with_diagnostic_scores,
)
from result_metric_contract import RESULT_GROUP_COLS, apply_result_metric_contract, metric_slices_from_scored_rows


STAGE_A0 = "A0_top1_naive"
METHOD_A0 = "top1_naive"
STAGE_A1 = "A1_top1_shared_bridge"
METHOD_A1 = "top1_shared_bridge"
STAGE_A2 = "A2_topk_static_bridge"
METHOD_A2 = "topk_static_bridge"
STAGE_A3 = "A3_offline_one_layer_caster"
METHOD_A3 = "offline_one_layer_caster"
STAGE_A3_NATIVE = "A3_native_fallback_frozen_diagnostic"
METHOD_A3_NATIVE = "native_fallback_frozen_diagnostic"
STAGE_A3_STRICT_PPD = "A3_strict_posterior_predictive_fallback_frozen"
METHOD_A3_STRICT_PPD = "strict_posterior_predictive_fallback_frozen"
STAGE_A4 = "A4_one_layer_caster_without_selected_rho"
METHOD_A4 = "one_layer_caster_without_selected_rho"
STAGE_A5 = "A5_causal_selected_rho"
METHOD_A5 = "causal_selected_rho"
Z50 = 0.6744897501960817
Z90 = 1.6448536269514722
Z95 = 1.959963984540054
MIN_VARIANCE = 1e-6
EXACT_MIXTURE_VALIDATION_NAME = "asof_mixture_weight_validation.csv"
STRICT_PPD_SCHEMA = "strict_posterior_predictive.v1"
STRICT_PPD_SCORE_SOURCE = "model_posterior_predictive_log_density"
SHARED_FALLBACK_SCORE_SOURCE = "shared_calibrated_bridge_fallback"


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes", "y"})


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bridge_core_hash(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"bridge config must contain an object: {path}")
    payload = dict(payload)
    payload.pop("rho", None)
    payload.pop("calibration_metadata", None)
    return _canonical_hash(payload)


def _weights_hash(weights: pd.DataFrame) -> str:
    rows = weights[["model_id", "weight"]].copy()
    rows["model_id"] = rows["model_id"].astype(str)
    rows["weight"] = pd.to_numeric(rows["weight"], errors="raise").astype(float)
    rows = rows.sort_values("model_id").reset_index(drop=True)
    return _canonical_hash(
        [{"model_id": row.model_id, "weight": format(float(row.weight), ".17g")} for row in rows.itertuples()]
    )


def _read_bridge_metadata(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = payload.get("calibration_metadata", {})
    return meta if isinstance(meta, dict) else {}


def _bridge_score_source(path: str | Path) -> str:
    metadata = _read_bridge_metadata(path)
    family = str(metadata.get("selected_bridge_family", "moment_t"))
    source = str(metadata.get("score_source", "archive_moment"))
    expected = {
        "moment_t": "archive_moment",
        "draw_kernel_t": "draw_kernel",
    }.get(family)
    if expected is None or source != expected:
        raise SystemExit(
            f"inconsistent bridge family/score source in {path}: "
            f"family={family!r} source={source!r}"
        )
    return source


def _frozen_predictive_contract(
    bridge,
    bridge_metadata: dict[str, object],
) -> str:
    ""







    contract = str(
        getattr(bridge, "predictive_contract", alternate_ARCHIVE_MOMENT)
    )
    metadata_contract = str(
        bridge_metadata.get("predictive_contract", alternate_ARCHIVE_MOMENT)
    )
    if contract not in PREDICTIVE_CONTRACTS:
        raise SystemExit(f"unsupported predictive contract {contract!r}")
    if metadata_contract != contract:
        raise SystemExit(
            "bridge config and calibration metadata mix predictive contracts: "
            f"config={contract!r}, metadata={metadata_contract!r}"
        )
    return contract


def _require_draws(path: str | Path | None, bridge_config: str | Path) -> Path:
    if not path:
        raise SystemExit(f"draw-kernel bridge {bridge_config} requires --draws")
    resolved = Path(path)
    if not resolved.is_file():
        raise SystemExit(f"missing draw archive required by {bridge_config}: {resolved}")
    return resolved


def _formal_score_contract(score_source: str) -> dict[str, str]:
    if score_source == "draw_kernel":
        return {
            "nll_score_basis": EXACT_DRAW_KERNEL_NLL_SCORE_BASIS,
            "formal_nll_status": "formal_exact_asof_posterior_draw_kernel_mixture",
            "nll_source_kind": EXACT_DRAW_KERNEL_NLL_SCORE_BASIS,
            "probability_score_basis": EXACT_DRAW_KERNEL_NLL_SCORE_BASIS,
        }
    return {
        "nll_score_basis": EXACT_NLL_SCORE_BASIS,
        "formal_nll_status": "formal_exact_asof_posterior_mixture_bridge",
        "nll_source_kind": "exact_asof_posterior_mixture_bridge",
        "probability_score_basis": "exact_asof_posterior_mixture_bridge",
    }


def _selected_top1(selection: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "model_id" not in selection.columns:
        raise SystemExit("selection must contain model_id")
    if selection.empty:
        raise SystemExit("selection is empty")
    top1 = selection.head(1).copy()
    return top1, str(top1.iloc[0]["model_id"])


def _selected_registry(registry: pd.DataFrame, model_id: str) -> pd.DataFrame:
    if "model_id" not in registry.columns:
        raise SystemExit("registry must contain model_id")
    reg = registry.copy()
    reg["model_id"] = reg["model_id"].astype(str)
    out = reg[reg["model_id"].eq(str(model_id))].copy()
    if out.empty:
        raise SystemExit(f"top-1 model {model_id!r} is absent from registry")
    return out.reset_index(drop=True)


def _selected_registry_many(registry: pd.DataFrame, selection: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if "model_id" not in selection.columns:
        raise SystemExit("selection must contain model_id")
    model_ids = selection["model_id"].dropna().astype(str).tolist()
    if not model_ids:
        raise SystemExit("selection is empty")
    reg = registry.copy()
    reg["model_id"] = reg["model_id"].astype(str)
    missing = [model_id for model_id in model_ids if model_id not in set(reg["model_id"])]
    if missing:
        raise SystemExit(f"selection model_id not present in registry: {missing}")
    order = {model_id: i for i, model_id in enumerate(model_ids)}
    selected = reg[reg["model_id"].isin(model_ids)].copy()
    selected["__order__"] = selected["model_id"].map(order)
    selected = selected.sort_values("__order__").drop(columns=["__order__"]).reset_index(drop=True)
    return selected, model_ids


def _aggregate_archive_predictions(archive: pd.DataFrame) -> pd.DataFrame:
    rows = archive.copy()
    rows["pred_mean"] = pd.to_numeric(rows["pred_mean"], errors="coerce")
    rows["pred_var"] = pd.to_numeric(rows["pred_var"], errors="coerce").clip(lower=0.0)
    rows["second_moment"] = rows["pred_var"] + rows["pred_mean"] * rows["pred_mean"]
    per_model = (
        rows.groupby(["forecast_id", "model_id"], dropna=False)
        .agg(
            pred_mean=("pred_mean", "mean"),
            second_moment=("second_moment", "mean"),
            particle_count=("particle_id", "nunique"),
        )
        .reset_index()
    )
    per_model["model_pred_var"] = (per_model["second_moment"] - per_model["pred_mean"] ** 2).clip(lower=0.0)
    per_model["model_second_moment"] = per_model["model_pred_var"] + per_model["pred_mean"] * per_model["pred_mean"]
    grouped = (
        per_model.groupby("forecast_id", dropna=False)
        .agg(
            pred_mean=("pred_mean", "mean"),
            second_moment=("model_second_moment", "mean"),
            particle_count=("particle_count", "sum"),
            model_count=("model_id", "nunique"),
        )
        .reset_index()
    )
    grouped["archive_pred_var"] = (grouped["second_moment"] - grouped["pred_mean"] ** 2).clip(lower=0.0)
    return grouped.drop(columns=["second_moment"])


def _estimate_train_global_sigma(ledger: pd.DataFrame, archive: pd.DataFrame) -> tuple[float, int]:
    train = ledger[ledger["split"].astype(str).eq("train")].copy()
    if train.empty:
        return 1.0, 0
    preds = _aggregate_archive_predictions(archive)
    rows = train.merge(preds[["forecast_id", "pred_mean"]], on="forecast_id", how="inner")
    if rows.empty:
        return 1.0, 0
    observed = _bool_series(rows.get("observed_mask", pd.Series(True, index=rows.index)))
    y = pd.to_numeric(rows.loc[observed, "observed_value"], errors="coerce")
    mu = pd.to_numeric(rows.loc[observed, "pred_mean"], errors="coerce")
    resid = (y - mu).replace([np.inf, -np.inf], np.nan).dropna()
    if resid.empty:
        return 1.0, 0
    rmse = float(np.sqrt(np.mean(np.square(resid.to_numpy(dtype=float)))))
    if not np.isfinite(rmse) or rmse <= 0:
        rmse = float(np.std(resid.to_numpy(dtype=float)))
    if not np.isfinite(rmse) or rmse <= 0:
        rmse = 1.0
    return max(rmse, 1e-3), int(len(resid))


def _bridge_sigma_original_scale(pred_mean: pd.Series, component: pd.Series, bridge) -> np.ndarray:
    comp_sigma = component.astype(str).map(lambda c: float(bridge.sigma_by_component.get(c, bridge.default_sigma))).to_numpy(dtype=float)
    if bridge.transform == "identity":
        return comp_sigma
    if bridge.transform == "log1p":
        return comp_sigma * (1.0 + np.maximum(pd.to_numeric(pred_mean, errors="coerce").to_numpy(dtype=float), 0.0))
    raise SystemExit(f"unsupported bridge transform {bridge.transform!r}")


def _build_readout(
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    readout_split: str,
    sigma: float,
    *,
    bridge=None,
) -> pd.DataFrame:
    readout_ledger = ledger[ledger["split"].astype(str).eq(str(readout_split))].copy()
    if readout_ledger.empty:
        raise SystemExit(f"no ledger rows for readout split {readout_split!r}")
    preds = _aggregate_archive_predictions(archive)
    out = readout_ledger.merge(preds, on="forecast_id", how="inner")
    if out.empty:
        raise SystemExit("readout split has no archive predictions for selected top-1 model")
    out["predictive_mean"] = pd.to_numeric(out["pred_mean"], errors="coerce")
    if bridge is None:
        extra_var = float(sigma) ** 2
    else:
                                                                          
                                                                            
                                                                            
                                                              
        extra_var = 0.0
    out["predictive_var"] = (pd.to_numeric(out["archive_pred_var"], errors="coerce").fillna(0.0).clip(lower=0.0) + extra_var).clip(lower=MIN_VARIANCE)
    sd = np.sqrt(out["predictive_var"].to_numpy(dtype=float))
    mean = out["predictive_mean"].to_numpy(dtype=float)
    out["lower_50"] = np.maximum(0.0, mean - Z50 * sd)
    out["upper_50"] = np.maximum(0.0, mean + Z50 * sd)
    out["lower_90"] = np.maximum(0.0, mean - Z90 * sd)
    out["upper_90"] = np.maximum(0.0, mean + Z90 * sd)
    out["lower_95"] = np.maximum(0.0, mean - Z95 * sd)
    out["upper_95"] = np.maximum(0.0, mean + Z95 * sd)
    out["pred_var"] = out["predictive_var"]
    out["y_true"] = out["observed_value"]
    out["y_mean"] = out["predictive_mean"]
    out["posterior_snapshot_time"] = ""
    out["used_prior_snapshot"] = True
    out["future_snapshot_violation"] = False
    out["self_target_update_violation"] = False
    out["stale_posterior_age_days"] = ""
    out["posterior_update_policy"] = "none_static"
    out["release_availability_rule"] = "not_applicable_no_posterior_update"
    return out


def _load_nonalternate_readout_draws(
    *,
    predictive_contract: str,
    score_source: str,
    draws_path: str | Path | None,
    bridge_config: str | Path,
) -> pd.DataFrame | None:
    ""

    if (
        predictive_contract == alternate_ARCHIVE_MOMENT
        or score_source != "draw_kernel"
    ):
        return None
    return pd.read_csv(_require_draws(draws_path, bridge_config))


def _decorate_static_readout(
    readout: pd.DataFrame,
    *,
    snapshot_time,
    used_prior: bool,
    policy: str,
) -> pd.DataFrame:
    ""

    out = readout.copy()
    out["pred_var"] = out["predictive_var"]
    out["y_true"] = out["observed_value"]
    out["y_mean"] = out["predictive_mean"]
    out["posterior_snapshot_time"] = (
        "" if pd.isna(snapshot_time) else pd.Timestamp(snapshot_time).isoformat()
    )
    out["used_prior_snapshot"] = bool(used_prior)
    out["future_snapshot_violation"] = False
    out["self_target_update_violation"] = False
    if pd.isna(snapshot_time):
        out["stale_posterior_age_days"] = ""
    else:
        out["stale_posterior_age_days"] = (
            pd.to_datetime(out["forecast_origin"], errors="coerce")
            - pd.Timestamp(snapshot_time)
        ).dt.total_seconds() / 86400.0
    out["posterior_update_policy"] = policy
    out["release_availability_rule"] = (
        "not_applicable_no_posterior_update"
        if policy == "none_static"
        else "release_time_no_later_than_forecast_origin"
    )
    return out


def _nonalternate_static_readout(
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    weights: pd.DataFrame,
    readout_split: str,
    bridge,
    score_source: str,
    draws: pd.DataFrame | None,
    *,
    snapshot_time=pd.NaT,
    used_prior: bool,
    policy: str,
) -> pd.DataFrame:
    ""

    predictive_contract = str(
        getattr(bridge, "predictive_contract", alternate_ARCHIVE_MOMENT)
    )
    if predictive_contract == alternate_ARCHIVE_MOMENT:
        raise ValueError(
            "_nonalternate_static_readout cannot be used for the alternate contract"
        )
    readout_ledger = ledger[
        ledger["split"].astype(str).eq(str(readout_split))
    ].copy()
    if readout_ledger.empty:
        raise SystemExit(f"no ledger rows for readout split {readout_split!r}")
    forecast_ids = set(readout_ledger["forecast_id"].astype(str))
    readout_archive = archive[
        archive["forecast_id"].astype(str).isin(forecast_ids)
    ].copy()
    readout_draws = None
    if draws is not None:
        model_ids = set(weights["model_id"].astype(str))
        readout_draws = draws[
            draws["forecast_id"].astype(str).isin(forecast_ids)
            & draws["model_id"].astype(str).isin(model_ids)
        ].copy()
    readout = posterior_predictive_readout(
        readout_ledger,
        readout_archive,
        weights[["model_id", "family", "weight"]].copy(),
        bridge_config=bridge,
        score_source=score_source,
        draws=readout_draws,
    )
    return _decorate_static_readout(
        readout,
        snapshot_time=snapshot_time,
        used_prior=used_prior,
        policy=policy,
    )


def _uniform_weights(registry: pd.DataFrame, model_ids: list[str]) -> pd.DataFrame:
    family_map = registry.set_index("model_id")["family"].astype(str).to_dict() if "family" in registry.columns else {}
    return pd.DataFrame(
        {
            "model_id": model_ids,
            "family": [family_map.get(model_id, "") for model_id in model_ids],
            "weight": [1.0 / len(model_ids)] * len(model_ids),
        }
    )


def _distribution_summary(weights: pd.DataFrame) -> dict[str, object]:
    summary = summarize_model_distribution(weights[["model_id", "family", "weight"]].copy())
    ordered = weights.sort_values("weight", ascending=False)
    summary["top_model"] = str(ordered.iloc[0]["model_id"]) if not ordered.empty else ""
    summary["model_count"] = int(len(weights))
    return summary


def _weighted_predictions(archive: pd.DataFrame, weights: pd.DataFrame, bridge) -> pd.DataFrame:
    total_model_count = int(weights["model_id"].astype(str).nunique())
    native_archive = native_forecast_rows(archive, require_provenance=False)
    rows = native_archive.merge(weights[["model_id", "weight"]], on="model_id", how="inner").copy()
    if rows.empty:
        return pd.DataFrame(columns=["forecast_id", "pred_mean", "archive_pred_var", "model_count", "particle_count"])
    rows["pred_mean"] = pd.to_numeric(rows["pred_mean"], errors="coerce")
    rows["pred_var"] = pd.to_numeric(rows["pred_var"], errors="coerce").clip(lower=0.0)
    particle_counts = rows.groupby(["forecast_id", "model_id"])["particle_id"].transform("nunique").astype(float)
    rows["mixture_weight"] = rows["weight"].astype(float) / particle_counts
    rows["row_var"] = rows["pred_var"]
    rows["row_second"] = rows["row_var"] + rows["pred_mean"] * rows["pred_mean"]
    out_rows: list[dict[str, object]] = []
    for forecast_id, group in rows.groupby("forecast_id", dropna=False):
        w = group["mixture_weight"].to_numpy(dtype=float)
        total = w.sum()
        if not np.isfinite(total) or total <= 0:
            continue
        w = w / total
        mean = float(np.sum(w * group["pred_mean"].to_numpy(dtype=float)))
        second = float(np.sum(w * group["row_second"].to_numpy(dtype=float)))
        out_rows.append(
            {
                "forecast_id": forecast_id,
                "pred_mean": mean,
                "archive_pred_var": max(0.0, second - mean * mean),
                "model_count": int(group["model_id"].nunique()),
                "particle_count": int(group["particle_id"].nunique()),
                "masked_model_count": total_model_count - int(group["model_id"].nunique()),
                "availability_mask_applied": int(group["model_id"].nunique()) < total_model_count,
            }
        )
    return pd.DataFrame(out_rows)


def _readout_from_predictions(ledger: pd.DataFrame, preds: pd.DataFrame, *, snapshot_time, used_prior: bool, policy: str) -> pd.DataFrame:
    out = ledger.merge(preds, on="forecast_id", how="inner")
    out["predictive_mean"] = pd.to_numeric(out["pred_mean"], errors="coerce")
    out["predictive_var"] = pd.to_numeric(out["archive_pred_var"], errors="coerce").fillna(0.0).clip(lower=MIN_VARIANCE)
    sd = np.sqrt(out["predictive_var"].to_numpy(dtype=float))
    mean = out["predictive_mean"].to_numpy(dtype=float)
    out["lower_50"] = np.maximum(0.0, mean - Z50 * sd)
    out["upper_50"] = np.maximum(0.0, mean + Z50 * sd)
    out["lower_90"] = np.maximum(0.0, mean - Z90 * sd)
    out["upper_90"] = np.maximum(0.0, mean + Z90 * sd)
    out["lower_95"] = np.maximum(0.0, mean - Z95 * sd)
    out["upper_95"] = np.maximum(0.0, mean + Z95 * sd)
    out["pred_var"] = out["predictive_var"]
    out["y_true"] = out["observed_value"]
    out["y_mean"] = out["predictive_mean"]
    out["posterior_snapshot_time"] = "" if pd.isna(snapshot_time) else pd.Timestamp(snapshot_time).isoformat()
    out["used_prior_snapshot"] = bool(used_prior)
    out["future_snapshot_violation"] = False
    out["self_target_update_violation"] = False
    if pd.isna(snapshot_time):
        out["stale_posterior_age_days"] = ""
    else:
        out["stale_posterior_age_days"] = (
            pd.to_datetime(out["forecast_origin"], errors="coerce") - pd.Timestamp(snapshot_time)
        ).dt.total_seconds() / 86400.0
    out["posterior_update_policy"] = policy
    out["release_availability_rule"] = "release_time_no_later_than_forecast_origin"
    return out


def _build_asof_weighted_readout(
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    posterior: pd.DataFrame,
    initial_weights: pd.DataFrame,
    readout_split: str,
    bridge,
    score_source: str = "archive_moment",
    draws: pd.DataFrame | None = None,
) -> pd.DataFrame:
    readout_ledger = ledger[ledger["split"].astype(str).eq(str(readout_split))].copy()
    if readout_ledger.empty:
        raise SystemExit(f"no ledger rows for readout split {readout_split!r}")
    predictive_contract = str(
        getattr(bridge, "predictive_contract", alternate_ARCHIVE_MOMENT)
    )
    if predictive_contract != alternate_ARCHIVE_MOMENT:
        forecast_ids = set(readout_ledger["forecast_id"].astype(str))
        readout_archive = archive[
            archive["forecast_id"].astype(str).isin(forecast_ids)
        ].copy()
        readout_draws = None
        if draws is not None:
            model_ids = set(initial_weights["model_id"].astype(str))
            readout_draws = draws[
                draws["forecast_id"].astype(str).isin(forecast_ids)
                & draws["model_id"].astype(str).isin(model_ids)
            ].copy()
        out = posterior_predictive_readout_asof(
            readout_ledger,
            readout_archive,
            posterior,
            initial_weights[["model_id", "family", "weight"]].copy(),
            posterior_update_policy="prequential_asof",
            release_availability_rule=(
                "release_time_no_later_than_forecast_origin"
            ),
            bridge_config=bridge,
            score_source=score_source,
            draws=readout_draws,
        )
        out["pred_var"] = out["predictive_var"]
        out["y_true"] = out["observed_value"]
        out["y_mean"] = out["predictive_mean"]
        return out

                                                                    
                                                                           
                                              
    readout_ledger["forecast_origin_dt"] = pd.to_datetime(readout_ledger["forecast_origin"], errors="coerce")
    posterior = posterior.copy()
    posterior["release_time_dt"] = pd.to_datetime(posterior["release_time"], errors="coerce")
    snapshots = np.array(sorted(posterior["release_time_dt"].dropna().unique()), dtype="datetime64[ns]")
    origins = readout_ledger["forecast_origin_dt"].to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(snapshots, origins, side="right") - 1
    snapshot_keys: list[str] = []
    snapshot_times: list[pd.Timestamp | pd.NaT] = []
    used_prior_values: list[bool] = []
    for pos in positions:
        if pos < 0:
            snapshot_keys.append("__prior__")
            snapshot_times.append(pd.NaT)
            used_prior_values.append(True)
        else:
            ts = pd.Timestamp(snapshots[pos])
            snapshot_keys.append(ts.isoformat())
            snapshot_times.append(ts)
            used_prior_values.append(False)
    readout_ledger["__snapshot_key__"] = snapshot_keys
    readout_ledger["__snapshot_time__"] = snapshot_times
    readout_ledger["__used_prior__"] = used_prior_values
    outputs: list[pd.DataFrame] = []
    for key, group in readout_ledger.groupby("__snapshot_key__", sort=False):
        if key == "__prior__":
            weights = initial_weights.copy()
            used_prior = True
            snap = pd.NaT
        else:
            snap = pd.Timestamp(key)
            weights = posterior[posterior["release_time_dt"].eq(snap)][["model_id", "family", "weight"]].copy()
            used_prior = False
        fids = set(group["forecast_id"].astype(str))
        preds = _weighted_predictions(archive[archive["forecast_id"].astype(str).isin(fids)].copy(), weights, bridge)
        clean_group = group.drop(columns=["forecast_origin_dt", "__snapshot_key__", "__snapshot_time__", "__used_prior__"])
        outputs.append(
            _readout_from_predictions(
                clean_group,
                preds,
                snapshot_time=snap,
                used_prior=used_prior,
                policy="prequential_asof",
            )
        )
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def _gaussian_nll(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    safe_var = np.maximum(var, MIN_VARIANCE)
    return 0.5 * (np.log(2.0 * pi * safe_var) + np.square(y - mean) / safe_var)


def _wis90(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    alpha = 0.10
    width = hi - lo
    return width + (2.0 / alpha) * (lo - y) * (y < lo) + (2.0 / alpha) * (y - hi) * (y > hi)


def _with_contract_intervals(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if "predictive_contract" not in out.columns:
        contracts = pd.Series(
            alternate_ARCHIVE_MOMENT, index=out.index, dtype=str
        )
    else:
        contracts = out["predictive_contract"].map(
            lambda value: (
                alternate_ARCHIVE_MOMENT
                if pd.isna(value) or not str(value).strip()
                else str(value).strip()
            )
        )
        unknown = sorted(set(contracts) - set(PREDICTIVE_CONTRACTS))
        if unknown:
            raise SystemExit(
                f"readout contains unsupported predictive_contract values {unknown}"
            )
    alternate = contracts.eq(alternate_ARCHIVE_MOMENT)
    if (~alternate).any():
        required = {
            "lower_50",
            "upper_50",
            "lower_90",
            "upper_90",
            "lower_95",
            "upper_95",
        }
        if contracts.eq(
            COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T
        ).any():
            required.add("predictive_median")
        missing = sorted(required - set(out.columns))
        if missing:
            raise SystemExit(
                "nonalternate readout is missing posterior-predictive summaries: "
                f"{missing}"
            )
    if not alternate.any():
        return out

    sigma = (
        pd.to_numeric(out.loc[alternate, "predictive_var"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .pow(0.5)
    )
    mean = pd.to_numeric(
        out.loc[alternate, "predictive_mean"], errors="coerce"
    )
    out.loc[alternate, "lower_50"] = (mean - Z50 * sigma).clip(lower=0.0)
    out.loc[alternate, "upper_50"] = (mean + Z50 * sigma).clip(lower=0.0)
    out.loc[alternate, "lower_90"] = (mean - Z90 * sigma).clip(lower=0.0)
    out.loc[alternate, "upper_90"] = (mean + Z90 * sigma).clip(lower=0.0)
    out.loc[alternate, "lower_95"] = (mean - Z95 * sigma).clip(lower=0.0)
    out.loc[alternate, "upper_95"] = (mean + Z95 * sigma).clip(lower=0.0)
    return out


def _metric_readout_contract(
    rows: pd.DataFrame,
    bridge_config: str | Path | None,
) -> str:
    ""







    if bridge_config:
        bridge, _rho = read_bridge_config(bridge_config)
        expected = _frozen_predictive_contract(
            bridge,
            _read_bridge_metadata(bridge_config),
        )
    else:
        expected = alternate_ARCHIVE_MOMENT
        if "predictive_contract" in rows.columns:
            declared = rows["predictive_contract"].map(
                lambda value: (
                    alternate_ARCHIVE_MOMENT
                    if pd.isna(value) or not str(value).strip()
                    else str(value).strip()
                )
            )
            values = sorted(set(declared))
            if len(values) != 1:
                raise SystemExit(
                    "metric readout mixes predictive contracts without a "
                    f"frozen bridge config: {values}"
                )
            expected = values[0]

    if expected not in PREDICTIVE_CONTRACTS:
        raise SystemExit(
            f"metric readout uses unsupported predictive contract {expected!r}"
        )
    if "predictive_contract" not in rows.columns:
        if expected != alternate_ARCHIVE_MOMENT:
            raise SystemExit(
                "nonalternate metric readout is missing predictive_contract "
                f"for frozen bridge contract {expected!r}"
            )
    else:
        declared = rows["predictive_contract"].map(
            lambda value: (
                alternate_ARCHIVE_MOMENT
                if pd.isna(value) or not str(value).strip()
                else str(value).strip()
            )
        )
        mismatched = declared.ne(expected)
        if mismatched.any():
            found = sorted(set(declared.loc[mismatched]))
            raise SystemExit(
                "metric readout predictive_contract does not match frozen "
                f"bridge contract {expected!r}: found {found}"
            )

    if expected != alternate_ARCHIVE_MOMENT:
        if "predictive_median" not in rows.columns:
            raise SystemExit(
                "nonalternate metric readout is missing exact predictive_median "
                f"for frozen bridge contract {expected!r}"
            )
        median = pd.to_numeric(rows["predictive_median"], errors="coerce")
        finite = np.isfinite(median.to_numpy(dtype=float))
        if not bool(finite.all()):
            raise SystemExit(
                "nonalternate metric readout contains "
                f"{int((~finite).sum())} non-finite exact predictive_median "
                f"value(s) for frozen bridge contract {expected!r}"
            )
    return expected


def _stage_dataset(rows: pd.DataFrame, ledger_path: str | Path) -> str:
    if "dataset" in rows.columns:
        values = rows["dataset"].dropna().astype(str).str.strip()
        values = values[values.ne("")]
        if not values.empty:
            return str(values.iloc[0])
    return Path(ledger_path).parent.name or "incremental_ablation"


def _exact_stage_scores(
    *,
    dataset: str,
    method_id: str,
    ledger_path: str | Path,
    archive_path: str | Path,
    bridge_config: str | Path,
    stage_root: Path,
    readout_path: Path,
    draws_path: str | Path | None = None,
    hierarchical: bool = False,
) -> tuple[pd.DataFrame, Path]:
    score_source = _bridge_score_source(bridge_config)
    if score_source == "draw_kernel":
        scores, validation = _draw_kernel_asof_mixture_scores(
            dataset=dataset,
            method=method_id,
            ledger_path=Path(ledger_path),
            draws_path=_require_draws(draws_path, bridge_config),
            archive_path=Path(archive_path),
            bridge_path=Path(bridge_config),
            root=stage_root,
            policy=ASOF_LTE_POLICY,
            selection_path=stage_root / "candidate_selection_log.csv",
            weights_path=stage_root / "posterior_path.csv",
            hierarchical=hierarchical,
        )
        diagnostic = pd.DataFrame()
    else:
        scores, validation = _bridge_asof_mixture_scores(
            dataset=dataset,
            method=method_id,
            ledger_path=Path(ledger_path),
            archive_path=Path(archive_path),
            bridge_path=Path(bridge_config),
            root=stage_root,
            policy=ASOF_LTE_POLICY,
            hierarchical=hierarchical,
            selection_path=stage_root / "candidate_selection_log.csv",
            weights_path=stage_root / "posterior_path.csv",
        )
        diagnostic = _bridge_readout_scores(
            dataset=dataset,
            method=method_id,
            ledger_path=Path(ledger_path),
            readout_path=readout_path,
            bridge_path=Path(bridge_config),
        )
    scores = _with_diagnostic_scores(scores, diagnostic)
    validation_path = stage_root / EXACT_MIXTURE_VALIDATION_NAME
    validation.to_csv(validation_path, index=False)
    return scores, validation_path


def _diagnostic_moment_matched_scores(rows: pd.DataFrame, method_id: str, bridge_config: str | Path) -> pd.DataFrame:
    bridge, _ = read_bridge_config(bridge_config)
    pseudo_archive = rows[["forecast_id", "component", "horizon", "predictive_mean", "predictive_var"]].copy()
    pseudo_archive["model_id"] = method_id
    pseudo_archive["particle_id"] = 0
    pseudo_archive = pseudo_archive.rename(columns={"predictive_mean": "pred_mean", "predictive_var": "pred_var"})
    scored = score_archive_rows(rows, pseudo_archive, bridge)
    if scored.empty:
        raise SystemExit(f"no bridge score rows for stage metrics using {bridge_config}")
    scores = scored[["forecast_id", "log_score"]].rename(columns={"log_score": "diagnostic_moment_matched_log_score"})
    scores["diagnostic_moment_matched_nll"] = -pd.to_numeric(scores["diagnostic_moment_matched_log_score"], errors="coerce")
    scores["diagnostic_moment_matched_nll_available"] = True
    scores["diagnostic_score_basis"] = DIAGNOSTIC_MOMENT_MATCHED_BASIS
    return scores


def _add_contract_nll(
    rows: pd.DataFrame,
    method_id: str,
    bridge_config: str | Path | None,
    *,
    formal_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = rows.copy()
    if formal_scores is not None:
        score_cols = [
            "forecast_id",
            "bridge_log_score",
            "bridge_nll",
            "nll_score_basis",
            "formal_nll_status",
            "nll_source_kind",
            "probability_score_basis",
            "diagnostic_moment_matched_nll_available",
            "diagnostic_moment_matched_nll",
        ]
        scores = formal_scores[[c for c in score_cols if c in formal_scores.columns]].copy()
        scores["forecast_id"] = scores["forecast_id"].astype(str)
        out["forecast_id"] = out["forecast_id"].astype(str)
        out = out.merge(scores, on="forecast_id", how="inner")
        if out.empty:
            raise SystemExit(f"no exact mixture scores joined for {method_id}")
        out["metric_nll_basis"] = out.get("nll_score_basis", EXACT_NLL_SCORE_BASIS)
        out["nll_score_basis"] = out.get("nll_score_basis", EXACT_NLL_SCORE_BASIS)
        out["formal_nll_status"] = out.get("formal_nll_status", "formal_exact_asof_posterior_mixture_bridge")
        out["nll_source_kind"] = out.get("nll_source_kind", "exact_asof_posterior_mixture_bridge")
        out["probability_score_basis"] = out.get("probability_score_basis", "exact_asof_posterior_mixture_bridge")
        if "diagnostic_moment_matched_nll_available" not in out.columns:
            out["diagnostic_moment_matched_nll_available"] = False
        if "diagnostic_moment_matched_nll" not in out.columns:
            out["diagnostic_moment_matched_nll"] = np.nan
        return out

    if bridge_config:
        scores = _diagnostic_moment_matched_scores(out, method_id, bridge_config)
        scores = scores.rename(
            columns={
                "diagnostic_moment_matched_log_score": "bridge_log_score",
                "diagnostic_moment_matched_nll": "bridge_nll",
            }
        )
        out = out.merge(scores[["forecast_id", "bridge_log_score", "bridge_nll"]], on="forecast_id", how="inner")
        out["metric_nll_basis"] = DIAGNOSTIC_MOMENT_MATCHED_BASIS
        return out

    y = pd.to_numeric(out["observed_value"], errors="coerce").to_numpy(dtype=float)
    mean = pd.to_numeric(out["predictive_mean"], errors="coerce").to_numpy(dtype=float)
    var = pd.to_numeric(out["predictive_var"], errors="coerce").to_numpy(dtype=float)
    out["bridge_log_score"] = np.nan
    out["bridge_nll"] = _gaussian_nll(y, mean, var)
    out["metric_nll_basis"] = "naive_gaussian_predictive_var"
    out["nll_score_basis"] = "naive_gaussian_predictive_var"
    out["formal_nll_status"] = "formal_naive_gaussian_predictive_var"
    out["nll_source_kind"] = "naive_gaussian_predictive_var"
    out["probability_score_basis"] = "naive_gaussian_predictive_var"
    out["diagnostic_moment_matched_nll_available"] = False
    out["diagnostic_moment_matched_nll"] = np.nan
    return out


def _metric_horizon_group(dataset: object, mode: object, horizon: object) -> str:
    ds = str(dataset).strip().lower()
    try:
        h = int(float(horizon))
    except Exception:
        h = 0
    if ds in {"benchmark_b", "benchmark_b_covid", "benchmark_b_flu"} or ds.startswith("benchmark_b_"):
        return "short" if h in {1, 2} else "long"
    return "short" if h in {1, 3} else "long"


def _macro_entity_id(rows: pd.DataFrame) -> pd.Series:
    def usable(value: object) -> str:
        text = str(value).strip()
        return "" if text.lower() in {"", "all", "nan", "none", "null", "na"} else text

    values: list[str] = []
    for _, row in rows.iterrows():
        dataset = str(row.get("dataset", "")).strip().lower()
        columns = (
            ["country", "country_code", "jurisdiction", "entity_id", "raw_region_id"]
            if dataset.startswith("benchmark_a")
            else ["jurisdiction", "entity_id", "raw_region_id", "country", "country_code"]
        )
        values.append(next((value for col in columns if (value := usable(row.get(col, "")))), ""))
    return pd.Series(values, index=rows.index)


def _stage_metrics(
    readout: pd.DataFrame,
    stage: str,
    method_id: str,
    source_artifact: str,
    bridge_config: str | Path | None = None,
    formal_scores: pd.DataFrame | None = None,
    asof_mixture_weight_validation_path: Path | str | None = None,
) -> pd.DataFrame:
    rows = readout.copy()
    observed = _bool_series(rows.get("observed_mask", pd.Series(True, index=rows.index)))
    rows = rows[observed].copy()
    if rows.empty:
        raise SystemExit("readout has no observed rows for metrics")
    metric_contract = _metric_readout_contract(rows, bridge_config)
    rows = _with_contract_intervals(rows)
    rows = _add_contract_nll(rows, method_id, bridge_config, formal_scores=formal_scores)
    rows["method"] = method_id
    rows["method_group"] = "incremental_ablation"
    slices = metric_slices_from_scored_rows(
        rows,
        source=source_artifact,
        y_col="observed_value",
        pred_col="predictive_mean",
        median_col=(
            "predictive_median"
            if metric_contract != alternate_ARCHIVE_MOMENT
            and "predictive_median" in rows.columns
            else None
        ),
        lower_50_col="lower_50",
        upper_50_col="upper_50",
        lower_90_col="lower_90",
        upper_90_col="upper_90",
        nll_col="bridge_nll",
        method_group="incremental_ablation",
    )
    if slices.empty:
        raise SystemExit("no finite metric groups produced")
    out = slices.copy()
    out.insert(0, "method_id", method_id)
    out.insert(0, "stage", stage)
    out["macro_entity_id"] = _macro_entity_id(out)
    out["target_component"] = out["component"].astype(str)
    out["source_artifact"] = source_artifact
    out["metric_contract"] = "result_metric_contract.metric_slices_from_scored_rows"
    formal_basis = EXACT_NLL_SCORE_BASIS
    formal_status = "formal_exact_asof_posterior_mixture_bridge"
    formal_source_kind = "exact_asof_posterior_mixture_bridge"
    formal_probability_basis = "exact_asof_posterior_mixture_bridge"
    if formal_scores is not None and not formal_scores.empty:
        for column, default_name in [
            ("nll_score_basis", "formal_basis"),
            ("formal_nll_status", "formal_status"),
            ("nll_source_kind", "formal_source_kind"),
            ("probability_score_basis", "formal_probability_basis"),
        ]:
            if column in formal_scores.columns:
                values = formal_scores[column].dropna().astype(str).unique().tolist()
                if len(values) != 1:
                    raise SystemExit(f"{stage} has non-unique formal score field {column}: {values}")
                if default_name == "formal_basis":
                    formal_basis = values[0]
                elif default_name == "formal_status":
                    formal_status = values[0]
                elif default_name == "formal_source_kind":
                    formal_source_kind = values[0]
                else:
                    formal_probability_basis = values[0]
    out["nll_basis"] = formal_basis if formal_scores is not None else ("naive_gaussian_predictive_var" if not bridge_config else DIAGNOSTIC_MOMENT_MATCHED_BASIS)
    out["nll_score_basis"] = out["nll_basis"]
    out["formal_nll_status"] = (
        formal_status
        if formal_scores is not None
        else ("formal_naive_gaussian_predictive_var" if not bridge_config else "diagnostic_moment_matched_bridge_excluded_from_formal_nll")
    )
    out["nll_source_kind"] = (
        formal_source_kind
        if formal_scores is not None
        else ("naive_gaussian_predictive_var" if not bridge_config else "diagnostic_moment_matched_readout")
    )
    out["probability_score_basis"] = (
        formal_probability_basis
        if formal_scores is not None
        else ("naive_gaussian_predictive_var" if not bridge_config else DIAGNOSTIC_MOMENT_MATCHED_BASIS)
    )
    if "diagnostic_moment_matched_nll" in rows.columns:
        diag_source = rows.copy()
        diag_source["method"] = method_id
        diag_source["method_group"] = "incremental_ablation"
        diag_source = apply_result_metric_contract(diag_source, method_group="incremental_ablation")
        diag = diag_source.groupby(RESULT_GROUP_COLS, dropna=False).agg(
            diagnostic_moment_matched_nll=("diagnostic_moment_matched_nll", "mean"),
            diagnostic_moment_matched_nll_available=("diagnostic_moment_matched_nll_available", "any"),
        ).reset_index()
        out = out.merge(diag, on=RESULT_GROUP_COLS, how="left")
    if "diagnostic_moment_matched_nll_available" not in out.columns:
        out["diagnostic_moment_matched_nll_available"] = False
    if "diagnostic_moment_matched_nll" not in out.columns:
        out["diagnostic_moment_matched_nll"] = np.nan
    out["asof_mixture_weight_validation_path"] = "" if asof_mixture_weight_validation_path is None else str(asof_mixture_weight_validation_path)
    keep = [
        "stage",
        "method_id",
        "dataset",
        "macro_entity_id",
        "target_component",
        "mode",
        "horizon_group",
        "horizon",
        "rmse",
        "mae",
        "nll",
        "bridge_nll",
        "wis",
        "coverage_90",
        "width_90",
        "n",
        "source_artifact",
        "metric_contract",
        "nll_basis",
        "nll_score_basis",
        "formal_nll_status",
        "nll_source_kind",
        "probability_score_basis",
        "diagnostic_moment_matched_nll_available",
        "diagnostic_moment_matched_nll",
        "asof_mixture_weight_validation_path",
    ]
    return out[[col for col in keep if col in out.columns]].copy()


def _asof_validation(readout: pd.DataFrame, *, stage: str, method_id: str) -> pd.DataFrame:
    validation = readout[
        [
            "forecast_id",
            "forecast_origin",
            "target_time",
            "release_time",
            "component",
            "horizon",
            "posterior_snapshot_time",
            "used_prior_snapshot",
            "future_snapshot_violation",
            "self_target_update_violation",
        ]
    ].copy()
    validation.insert(0, "stage", stage)
    validation.insert(1, "method_id", method_id)
    validation["validation_status"] = "PASS"
    return validation


def run_a0(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = RuntimeLogger()
    with timer.measure("load_inputs"):
        ledger = pd.read_csv(args.ledger)
        archive_all = pd.read_csv(args.archive)
        registry_all = pd.read_csv(args.registry)
        selection_all = pd.read_csv(args.selection)
        selection, model_id = _selected_top1(selection_all)
        registry = _selected_registry(registry_all, model_id)
        archive = archive_all[archive_all["model_id"].astype(str).eq(str(model_id))].copy()
        if archive.empty:
            raise SystemExit(f"archive has no rows for top-1 model {model_id!r}")
        selection.to_csv(out_dir / "candidate_selection_log.csv", index=False)
        registry.to_csv(out_dir / "model_registry.csv", index=False)

    with timer.measure("archive_contract_validation"):
        _validate_archive_contract(archive, ledger, [model_id], out_dir)

    with timer.measure("naive_scale_calibration"):
        if args.naive_scale_source == "train_residual_global":
            sigma, sigma_rows = _estimate_train_global_sigma(ledger, archive)
        else:
            sigma, sigma_rows = 1.0, 0

    with timer.measure("forecast_readout"):
        readout = _build_readout(ledger, archive, args.readout_split, sigma)
        readout_path = out_dir / "forecast_readout.csv"
        readout.to_csv(readout_path, index=False)

    with timer.measure("metrics"):
        metrics = _stage_metrics(readout, STAGE_A0, METHOD_A0, str(readout_path))
        metrics.to_csv(out_dir / "stage_metrics.csv", index=False)

    validation = _asof_validation(readout, stage=STAGE_A0, method_id=METHOD_A0)
    validation.to_csv(out_dir / "asof_posterior_readout_validation.csv", index=False)
    metadata = {
        "stage": STAGE_A0,
        "method_id": METHOD_A0,
        "added_component": "Starting point",
        "selected_model_count": 1,
        "selected_model_ids": [model_id],
        "active_model_set": [model_id],
        "likelihood_policy": "naive_identity_gaussian",
        "naive_scale_source": args.naive_scale_source,
        "naive_global_sigma": float(sigma),
        "naive_scale_train_rows": int(sigma_rows),
        "posterior_update": False,
        "posterior_update_policy": "none_static",
        "prior_policy": "single_model_uniform",
        "readout_split": str(args.readout_split),
        "seed": int(args.seed),
        "ledger": str(args.ledger),
        "archive": str(args.archive),
        "registry": str(args.registry),
        "selection": str(args.selection),
        "ledger_rows": int(len(ledger)),
        "archive_rows": int(len(archive)),
        "readout_rows": int(len(readout)),
        "test_rows_used_for_bridge_calibration": 0,
        "test_rows_used_for_tuning": 0,
        "test_rows_used_for_posterior_update": 0,
        "native_likelihoods_compared": False,
        "incremental_ablation": True,
    }
    _write_json(metadata, out_dir / "stage_metadata.json")
    write_timing_log(timer.summary(seed=args.seed), out_dir / "timing.json")
    print(
        f"ok stage={STAGE_A0} out={out_dir} selected_model_count=1 "
        f"model_id={model_id} readout_rows={len(readout)} sigma={sigma:.6g}"
    )


def run_a1(args) -> None:
    if not args.bridge_config:
        raise SystemExit("A1 requires --bridge-config")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = RuntimeLogger()
    with timer.measure("load_inputs"):
        ledger = pd.read_csv(args.ledger)
        archive_all = pd.read_csv(args.archive)
        registry_all = pd.read_csv(args.registry)
        selection_all = pd.read_csv(args.selection)
        selection, model_id = _selected_top1(selection_all)
        registry = _selected_registry(registry_all, model_id)
        archive = archive_all[archive_all["model_id"].astype(str).eq(str(model_id))].copy()
        if archive.empty:
            raise SystemExit(f"archive has no rows for top-1 model {model_id!r}")
        bridge, rho = read_bridge_config(args.bridge_config)
        bridge_metadata = _read_bridge_metadata(args.bridge_config)
        score_source = _bridge_score_source(args.bridge_config)
        predictive_contract = _frozen_predictive_contract(
            bridge, bridge_metadata
        )
        readout_draws = _load_nonalternate_readout_draws(
            predictive_contract=predictive_contract,
            score_source=score_source,
            draws_path=args.draws,
            bridge_config=args.bridge_config,
        )
        selection.to_csv(out_dir / "candidate_selection_log.csv", index=False)
        registry.to_csv(out_dir / "model_registry.csv", index=False)

    with timer.measure("archive_contract_validation"):
        _validate_archive_contract(archive, ledger, [model_id], out_dir)

                                                                             
                                                                         
                                                                          
                                                          
    static_weights = pd.DataFrame(
        {
            "model_id": [model_id],
            "family": [
                str(registry["family"].iloc[0]) if "family" in registry.columns else ""
            ],
            "weight": [1.0],
            "weight_policy": ["fixed_top1"],
        }
    )
    static_weights.to_csv(out_dir / "static_weights.csv", index=False)

    with timer.measure("forecast_readout"):
        readout = (
            _build_readout(
                ledger,
                archive,
                args.readout_split,
                0.0,
                bridge=bridge,
            )
            if predictive_contract == alternate_ARCHIVE_MOMENT
            else _nonalternate_static_readout(
                ledger,
                archive,
                static_weights,
                args.readout_split,
                bridge,
                score_source,
                readout_draws,
                used_prior=True,
                policy="none_static",
            )
        )
        readout_path = out_dir / "forecast_readout.csv"
        readout.to_csv(readout_path, index=False)

    with timer.measure("exact_mixture_nll"):
        exact_scores, exact_validation_path = _exact_stage_scores(
            dataset=_stage_dataset(readout, args.ledger),
            method_id=METHOD_A1,
            ledger_path=args.ledger,
            archive_path=args.archive,
            bridge_config=args.bridge_config,
            stage_root=out_dir,
            readout_path=readout_path,
            draws_path=args.draws,
        )

    with timer.measure("metrics"):
        metrics = _stage_metrics(
            readout,
            STAGE_A1,
            METHOD_A1,
            str(readout_path),
            bridge_config=args.bridge_config,
            formal_scores=exact_scores,
            asof_mixture_weight_validation_path=exact_validation_path,
        )
        metrics.to_csv(out_dir / "stage_metrics.csv", index=False)

    validation = _asof_validation(readout, stage=STAGE_A1, method_id=METHOD_A1)
    validation.to_csv(out_dir / "asof_posterior_readout_validation.csv", index=False)
    calibration_model_set = bridge_metadata.get("calibration_model_set", [])
    score_contract = _formal_score_contract(score_source)
    metadata = {
        "stage": STAGE_A1,
        "method_id": METHOD_A1,
        "added_component": "Fixed Top-1 model",
        "selected_model_count": 1,
        "selected_model_ids": [model_id],
        "active_model_set": [model_id],
        "calibration_model_set": calibration_model_set,
        "bridge_config": str(args.bridge_config),
        "bridge_distribution": bridge.distribution,
        "bridge_transform": bridge.transform,
        "predictive_contract": predictive_contract,
        "rho": 1.0 if rho is None else float(rho),
        "likelihood_policy": "shared_calibrated_bridge",
        "score_source": score_source,
        "selected_bridge_family": bridge_metadata.get("selected_bridge_family", "moment_t"),
        "draws": str(args.draws) if score_source == "draw_kernel" else "",
        "draws_hash": _sha256_file(args.draws) if score_source == "draw_kernel" else "",
        "posterior_update": False,
        "posterior_update_policy": "none_static",
        "weight_policy": "fixed_top1",
        "prior_policy": "single_model_uniform",
        "readout_split": str(args.readout_split),
        "seed": int(args.seed),
        "ledger": str(args.ledger),
        "archive": str(args.archive),
        "registry": str(args.registry),
        "selection": str(args.selection),
        "ledger_rows": int(len(ledger)),
        "archive_rows": int(len(archive)),
        "readout_rows": int(len(readout)),
        "test_rows_used_for_bridge_calibration": int(bridge_metadata.get("test_rows_used_for_bridge_calibration", bridge_metadata.get("test_rows_used_for_tuning", 0))),
        "test_rows_used_for_tuning": int(bridge_metadata.get("test_rows_used_for_tuning", 0)),
        "test_rows_used_for_posterior_update": 0,
        "native_likelihoods_compared": False,
        **score_contract,
        "diagnostic_moment_matched_nll_available": score_source == "archive_moment",
        "asof_mixture_weight_validation": str(exact_validation_path),
        "incremental_ablation": True,
    }
    _write_json(metadata, out_dir / "stage_metadata.json")
    write_timing_log(timer.summary(seed=args.seed), out_dir / "timing.json")
    print(
        f"ok stage={STAGE_A1} out={out_dir} selected_model_count=1 "
        f"model_id={model_id} readout_rows={len(readout)} likelihood=shared_calibrated_bridge"
    )


def run_a2(args) -> None:
    if not args.bridge_config:
        raise SystemExit("A2 requires --bridge-config")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = RuntimeLogger()
    with timer.measure("load_inputs"):
        ledger = pd.read_csv(args.ledger)
        archive_all = pd.read_csv(args.archive)
        registry_all = pd.read_csv(args.registry)
        selection = pd.read_csv(args.selection)
        registry, model_ids = _selected_registry_many(registry_all, selection)
        archive = archive_all[archive_all["model_id"].astype(str).isin(model_ids)].copy()
        if archive.empty:
            raise SystemExit("archive has no rows for selected Top-K models")
        bridge, rho = read_bridge_config(args.bridge_config)
        bridge_metadata = _read_bridge_metadata(args.bridge_config)
        score_source = _bridge_score_source(args.bridge_config)
        predictive_contract = _frozen_predictive_contract(
            bridge, bridge_metadata
        )
        readout_draws = _load_nonalternate_readout_draws(
            predictive_contract=predictive_contract,
            score_source=score_source,
            draws_path=args.draws,
            bridge_config=args.bridge_config,
        )
        selection.to_csv(out_dir / "candidate_selection_log.csv", index=False)
        registry.to_csv(out_dir / "model_registry.csv", index=False)

    with timer.measure("archive_contract_validation"):
        _validate_archive_contract(archive, ledger, model_ids, out_dir)

    hierarchical_static = args.prior_policy == "hierarchical_family_balanced"
    if hierarchical_static:
        static_weights = initialize_hierarchical_weights(registry).model_weights[
            ["model_id", "family", "weight"]
        ].copy()
        static_weights["weight_policy"] = "hierarchical_family_balanced_static"
    else:
        static_weights = pd.DataFrame(
            {
                "model_id": model_ids,
                "family": registry["family"].astype(str).tolist() if "family" in registry.columns else [""] * len(model_ids),
                "weight": [1.0 / len(model_ids)] * len(model_ids),
                "weight_policy": ["uniform_static"] * len(model_ids),
            }
        )
    static_weights.to_csv(out_dir / "static_weights.csv", index=False)
    if hierarchical_static:
        static_weights[["model_id", "family", "weight"]].to_csv(
            out_dir / "initial_prior.csv", index=False
        )

    with timer.measure("forecast_readout"):
        readout = (
            _build_readout(
                ledger,
                archive,
                args.readout_split,
                0.0,
                bridge=bridge,
            )
            if predictive_contract == alternate_ARCHIVE_MOMENT
            else _nonalternate_static_readout(
                ledger,
                archive,
                static_weights,
                args.readout_split,
                bridge,
                score_source,
                readout_draws,
                used_prior=True,
                policy="none_static",
            )
        )
        readout_path = out_dir / "forecast_readout.csv"
        readout.to_csv(readout_path, index=False)

    with timer.measure("exact_mixture_nll"):
        exact_scores, exact_validation_path = _exact_stage_scores(
            dataset=_stage_dataset(readout, args.ledger),
            method_id=METHOD_A2,
            ledger_path=args.ledger,
            archive_path=args.archive,
            bridge_config=args.bridge_config,
            stage_root=out_dir,
            readout_path=readout_path,
            draws_path=args.draws,
            hierarchical=hierarchical_static,
        )

    with timer.measure("metrics"):
        metrics = _stage_metrics(
            readout,
            STAGE_A2,
            METHOD_A2,
            str(readout_path),
            bridge_config=args.bridge_config,
            formal_scores=exact_scores,
            asof_mixture_weight_validation_path=exact_validation_path,
        )
        metrics.to_csv(out_dir / "stage_metrics.csv", index=False)

    validation = _asof_validation(readout, stage=STAGE_A2, method_id=METHOD_A2)
    validation.to_csv(out_dir / "asof_posterior_readout_validation.csv", index=False)
    score_contract = _formal_score_contract(score_source)
    metadata = {
        "stage": STAGE_A2,
        "method_id": METHOD_A2,
        "added_component": (
            "Family-balanced static mixture"
            if hierarchical_static
            else "Top-K static mixture"
        ),
        "selected_model_count": int(len(model_ids)),
        "selected_model_ids": model_ids,
        "active_model_set": model_ids,
        "calibration_model_set": bridge_metadata.get("calibration_model_set", []),
        "bridge_config": str(args.bridge_config),
        "bridge_distribution": bridge.distribution,
        "bridge_transform": bridge.transform,
        "predictive_contract": predictive_contract,
        "rho": 1.0 if rho is None else float(rho),
        "likelihood_policy": "shared_calibrated_bridge",
        "score_source": score_source,
        "selected_bridge_family": bridge_metadata.get("selected_bridge_family", "moment_t"),
        "draws": str(args.draws) if score_source == "draw_kernel" else "",
        "draws_hash": _sha256_file(args.draws) if score_source == "draw_kernel" else "",
        "posterior_update": False,
        "posterior_update_policy": "none_static",
        "weight_policy": (
            "hierarchical_family_balanced_static"
            if hierarchical_static
            else "uniform_static"
        ),
        "prior_policy": (
            "hierarchical_family_balanced"
            if hierarchical_static
            else "uniform_model_static"
        ),
        "hierarchy_policy": (
            "outer_family_inner_model" if hierarchical_static else "none"
        ),
        "readout_split": str(args.readout_split),
        "seed": int(args.seed),
        "ledger": str(args.ledger),
        "archive": str(args.archive),
        "registry": str(args.registry),
        "selection": str(args.selection),
        "ledger_rows": int(len(ledger)),
        "archive_rows": int(len(archive)),
        "readout_rows": int(len(readout)),
        "test_rows_used_for_bridge_calibration": int(bridge_metadata.get("test_rows_used_for_bridge_calibration", bridge_metadata.get("test_rows_used_for_tuning", 0))),
        "test_rows_used_for_tuning": int(bridge_metadata.get("test_rows_used_for_tuning", 0)),
        "test_rows_used_for_posterior_update": 0,
        "native_likelihoods_compared": False,
        **score_contract,
        "diagnostic_moment_matched_nll_available": score_source == "archive_moment",
        "asof_mixture_weight_validation": str(exact_validation_path),
        "incremental_ablation": True,
    }
    _write_json(metadata, out_dir / "stage_metadata.json")
    write_timing_log(timer.summary(seed=args.seed), out_dir / "timing.json")
    print(
        f"ok stage={STAGE_A2} out={out_dir} selected_model_count={len(model_ids)} "
        f"readout_rows={len(readout)} weight_policy={metadata['weight_policy']}"
    )


def _evidence_partitions(ledger: pd.DataFrame, readout_split: str) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    rows = ledger.copy()
    rows["release_time_dt"] = pd.to_datetime(rows["release_time"], errors="coerce")
    rows["forecast_origin_dt"] = pd.to_datetime(rows["forecast_origin"], errors="coerce")
    test = rows[rows["split"].astype(str).eq(str(readout_split))]
    if test.empty or test["forecast_origin_dt"].dropna().empty:
        raise SystemExit(f"no forecast origins for readout split {readout_split!r}")
    cutoff = pd.Timestamp(test["forecast_origin_dt"].min())
    eligible = rows[
        rows["split"].astype(str).isin(["train", "val", "embargo", str(readout_split)])
    ].copy()
    pretest = eligible[
        eligible["split"].astype(str).isin(["train", "val", "embargo"])
        & eligible["release_time_dt"].notna()
        & eligible["release_time_dt"].le(cutoff)
    ].copy()
    premature_test = eligible[
        eligible["split"].astype(str).eq(str(readout_split))
        & eligible["release_time_dt"].notna()
        & eligible["release_time_dt"].le(cutoff)
    ]
    if not premature_test.empty:
        raise SystemExit(
            "test evidence is already released at or before the first test origin; "
            "cannot construct a leakage-free Offline/Online paired cutoff"
        )
    online = eligible[eligible["release_time_dt"].notna() & eligible["release_time_dt"].gt(cutoff)].copy()
    drop = ["release_time_dt", "forecast_origin_dt"]
    return cutoff, pretest.drop(columns=drop), online.drop(columns=drop)


def _shared_bridge_score_rows(
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    draws: pd.DataFrame | None,
    bridge,
    score_source: str,
) -> pd.DataFrame:
    native_archive = native_forecast_rows(archive, require_provenance=True)
    if score_source == "archive_moment":
        return score_archive_rows(ledger, native_archive, bridge)
    if score_source != "draw_kernel":
        raise SystemExit(f"unsupported frozen bridge score source {score_source!r}")
    if draws is None:
        raise SystemExit("draw-kernel evidence replay requires loaded forecast draws")
    native_pairs = native_archive[["forecast_id", "model_id"]].drop_duplicates()
    eligible_draws = draws.copy()
    eligible_draws["forecast_id"] = eligible_draws["forecast_id"].astype(str)
    eligible_draws["model_id"] = eligible_draws["model_id"].astype(str)
    native_pairs["forecast_id"] = native_pairs["forecast_id"].astype(str)
    native_pairs["model_id"] = native_pairs["model_id"].astype(str)
    eligible_draws = eligible_draws.merge(
        native_pairs, on=["forecast_id", "model_id"], how="inner"
    )
    if eligible_draws.empty:
        raise SystemExit("draw-kernel evidence replay has no native eligible draws")
    return score_draw_rows(ledger, eligible_draws, bridge)


def _replay_model_evidence(
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    draws: pd.DataFrame | None,
    bridge,
    score_source: str,
    model_ids: list[str],
    initial_weights: pd.DataFrame,
    rho: float,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights = initial_weights[["model_id", "family", "weight"]].copy()
    if ledger.empty:
        return pd.DataFrame(), pd.DataFrame(), weights
    update_archive = archive[archive["forecast_id"].astype(str).isin(set(ledger["forecast_id"].astype(str)))].copy()
    update_draws = None
    if draws is not None:
        update_draws = draws[
            draws["forecast_id"].astype(str).isin(set(ledger["forecast_id"].astype(str)))
        ].copy()
    scored = _shared_bridge_score_rows(
        ledger, update_archive, update_draws, bridge, score_source
    )
    release_meta = ledger[["forecast_id", "release_time"]].drop_duplicates("forecast_id")
    scored = scored.merge(release_meta, on="forecast_id", how="left")
    scored["release_time"] = pd.to_datetime(scored["release_time"], errors="coerce")
    posterior_rows: list[pd.DataFrame] = []
    evidence_rows: list[pd.DataFrame] = []
    for release_time in sorted(pd.to_datetime(ledger["release_time"], errors="coerce").dropna().unique()):
        current = scored[scored["release_time"].eq(pd.Timestamp(release_time))]
        batch_ledger = ledger[
            pd.to_datetime(ledger["release_time"], errors="coerce").eq(pd.Timestamp(release_time))
        ].copy()
        availability = evidence_availability_by_model(
            current, batch_ledger, model_ids
        )
        log_evidence = pd.DataFrame(
            [
                {
                    "release_time": pd.Timestamp(release_time),
                    "model_id": str(model_id),
                    "log_evidence": compute_log_evidence(current, model_id=str(model_id))
                    if availability[str(model_id)] else 0.0,
                    "evidence_available": availability[str(model_id)],
                    "n_scored_rows": int(current[current["model_id"].astype(str).eq(str(model_id))].shape[0]),
                    "score_update_basis": (
                        "shared_calibrated_draw_kernel_bridge"
                        if score_source == "draw_kernel"
                        else "shared_calibrated_archive_moment_bridge"
                    ),
                    "update_phase": phase,
                }
                for model_id in model_ids
            ]
        )
        weights = update_outer_weights(
            weights,
            log_evidence,
            rho=float(rho),
        )
        snapshot = weights.copy()
        snapshot["release_time"] = pd.Timestamp(release_time)
        snapshot["rho"] = float(rho)
        snapshot["update_phase"] = phase
        posterior_rows.append(snapshot)
        evidence_rows.append(log_evidence)
        weights = weights[["model_id", "family", "weight"]]
    posterior = pd.concat(posterior_rows, ignore_index=True) if posterior_rows else pd.DataFrame()
    evidence = pd.concat(evidence_rows, ignore_index=True) if evidence_rows else pd.DataFrame()
    return posterior, evidence, weights


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _native_fallback_sources(
    availability: pd.DataFrame,
    registry: pd.DataFrame,
    model_ids: list[str],
) -> pd.DataFrame:
    ""







    required = {"model_id", "native_likelihood_status"}
    missing = sorted(required - set(availability.columns))
    if missing:
        raise SystemExit(f"native availability missing columns: {missing}")
    rows = availability.copy()
    rows["model_id"] = rows["model_id"].astype(str)
    rows = rows[rows["model_id"].isin(model_ids)].copy()
    duplicates = rows[rows.duplicated("model_id", keep=False)]["model_id"].unique().tolist()
    if duplicates:
        raise SystemExit(f"native availability has duplicate selected model rows: {duplicates}")
    missing_models = sorted(set(model_ids) - set(rows["model_id"]))
    if missing_models:
        raise SystemExit(f"native availability omits selected model(s): {missing_models}")

    family_map = (
        registry.set_index("model_id")["family"].astype(str).to_dict()
        if "family" in registry.columns
        else {}
    )
    validation_rows: list[dict[str, object]] = []
    for model_id in model_ids:
        row = rows[rows["model_id"].eq(model_id)].iloc[0]
        status = str(row["native_likelihood_status"]).strip().lower()
        supports = _bool_value(row.get("supports_native_log_likelihood", status == "true_native"))
        if status == "true_native":
            if not supports:
                raise SystemExit(f"{model_id} is true_native but does not support native log likelihood")
            source = "adapter_native_likelihood"
            fallback_reason = ""
        elif status in {
            "proxy_only",
            "deterministic_no_native",
            "sample_kernel_only",
            "quantile_reconstructed_only",
            "unavailable",
            "not_applicable",
        }:
            source = "shared_calibrated_bridge_fallback"
            fallback_reason = str(row.get("fallback_reason", row.get("blocker", status)) or status)
        else:
            raise SystemExit(
                f"{model_id} native status {status!r} is not eligible for the diagnostic; "
                "true-native missing/blocked artifacts must not silently fall back"
            )
        validation_rows.append(
            {
                "model_id": model_id,
                "family": family_map.get(model_id, ""),
                "native_likelihood_status": status,
                "supports_native_log_likelihood": supports,
                "update_score_source": source,
                "fallback_reason": fallback_reason,
            }
        )
    return pd.DataFrame(validation_rows)


def _replay_native_fallback_evidence(
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    draws: pd.DataFrame | None,
    bridge,
    score_source: str,
    model_ids: list[str],
    initial_weights: pd.DataFrame,
    native_scores: pd.DataFrame,
    availability: pd.DataFrame,
    registry: pd.DataFrame,
    rho: float,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ""






    source_validation = _native_fallback_sources(availability, registry, model_ids)
    weights = initial_weights[["model_id", "family", "weight"]].copy()
    if ledger.empty:
        source_validation["required_pretest_rows"] = 0
        source_validation["covered_native_rows"] = 0
        return pd.DataFrame(), pd.DataFrame(), weights, source_validation

    update_archive = archive[archive["forecast_id"].astype(str).isin(set(ledger["forecast_id"].astype(str)))].copy()
    update_draws = None
    if draws is not None:
        update_draws = draws[
            draws["forecast_id"].astype(str).isin(set(ledger["forecast_id"].astype(str)))
        ].copy()
    bridge_scored = _shared_bridge_score_rows(
        ledger, update_archive, update_draws, bridge, score_source
    )
    release_meta = ledger[["forecast_id", "release_time"]].drop_duplicates("forecast_id")
    bridge_scored = bridge_scored.merge(release_meta, on="forecast_id", how="left")
    bridge_scored["release_time"] = pd.to_datetime(bridge_scored["release_time"], errors="raise")

    scores = native_scores.copy()
    required_score_columns = {"forecast_id", "model_id"}
    missing = sorted(required_score_columns - set(scores.columns))
    if missing:
        raise SystemExit(f"native scores missing columns: {missing}")
    score_col = "log_score" if "log_score" in scores.columns else "native_log_likelihood"
    if score_col not in scores.columns:
        raise SystemExit("native scores require log_score or native_log_likelihood")
    scores["forecast_id"] = scores["forecast_id"].astype(str)
    scores["model_id"] = scores["model_id"].astype(str)
    scores[score_col] = pd.to_numeric(scores[score_col], errors="raise")
    scores = scores[scores["forecast_id"].isin(set(ledger["forecast_id"].astype(str)))].copy()
    duplicates = scores[scores.duplicated(["model_id", "forecast_id"], keep=False)]
    if not duplicates.empty:
        sample = duplicates[["model_id", "forecast_id"]].drop_duplicates().head(5).to_dict("records")
        raise SystemExit(f"native scores duplicate model/forecast rows: {sample}")

    observed = ledger.copy()
    observed_mask = observed.get("observed_mask", pd.Series(True, index=observed.index))
    observed = observed[_bool_series(observed_mask)].copy()
    required_fids = set(observed["forecast_id"].astype(str))
    native_models = set(
        source_validation.loc[source_validation["update_score_source"].eq("adapter_native_likelihood"), "model_id"].astype(str)
    )
    coverage_rows: list[dict[str, object]] = []
    native_scored_by_model: dict[str, pd.DataFrame] = {}
    for model_id in model_ids:
        model_scores = scores[scores["model_id"].eq(model_id)].copy()
        if model_id in native_models:
            covered = set(model_scores["forecast_id"].astype(str))
            missing_fids = sorted(required_fids - covered)
            extra_fids = sorted(covered - required_fids)
            if missing_fids:
                raise SystemExit(
                    f"true-native scores incomplete for {model_id}: missing {len(missing_fids)} "
                    f"pre-test forecast rows; sample={missing_fids[:5]}"
                )
            if extra_fids:
                model_scores = model_scores[model_scores["forecast_id"].isin(required_fids)].copy()
            template = (
                bridge_scored[bridge_scored["model_id"].astype(str).eq(model_id)]
                .sort_values(["forecast_id", "particle_id"])
                .drop_duplicates("forecast_id", keep="first")
                .copy()
            )
            template = template[template["forecast_id"].astype(str).isin(required_fids)].copy()
            native_score_col = "__native_update_log_score"
            template = template.merge(
                model_scores[["forecast_id", score_col]].rename(columns={score_col: native_score_col}),
                on="forecast_id",
                how="left",
                validate="one_to_one",
            )
            if template[native_score_col].isna().any() or len(template) != len(required_fids):
                raise SystemExit(f"native/template coverage mismatch for {model_id}")
            template["particle_id"] = 0
            template["log_score"] = template[native_score_col].astype(float)
            template = template.drop(columns=[native_score_col])
            native_scored_by_model[model_id] = template
            coverage_rows.append(
                {"model_id": model_id, "required_pretest_rows": len(required_fids), "covered_native_rows": len(template)}
            )
        else:
            coverage_rows.append(
                {"model_id": model_id, "required_pretest_rows": len(required_fids), "covered_native_rows": 0}
            )
    source_validation = source_validation.merge(pd.DataFrame(coverage_rows), on="model_id", how="left", validate="one_to_one")

    posterior_rows: list[pd.DataFrame] = []
    evidence_rows: list[pd.DataFrame] = []
    source_map = source_validation.set_index("model_id")["update_score_source"].astype(str).to_dict()
    for release_time in sorted(pd.to_datetime(ledger["release_time"], errors="coerce").dropna().unique()):
        release_ts = pd.Timestamp(release_time)
        release_evidence: list[dict[str, object]] = []
        for model_id in model_ids:
            source = source_map[model_id]
            if source == "adapter_native_likelihood":
                current = native_scored_by_model[model_id]
                current = current[current["release_time"].eq(release_ts)]
            else:
                current = bridge_scored[
                    bridge_scored["release_time"].eq(release_ts)
                    & bridge_scored["model_id"].astype(str).eq(model_id)
                ]
            release_evidence.append(
                {
                    "release_time": release_ts,
                    "model_id": model_id,
                    "log_evidence": compute_log_evidence(current, model_id=model_id),
                    "n_scored_rows": int(current.shape[0]),
                    "score_update_basis": source,
                    "update_phase": phase,
                }
            )
        log_evidence = pd.DataFrame(release_evidence)
        weights = update_outer_weights(weights, log_evidence, rho=float(rho))
        snapshot = weights.copy()
        snapshot["release_time"] = release_ts
        snapshot["rho"] = float(rho)
        snapshot["update_phase"] = phase
        posterior_rows.append(snapshot)
        evidence_rows.append(log_evidence)
        weights = weights[["model_id", "family", "weight"]]
    posterior = pd.concat(posterior_rows, ignore_index=True) if posterior_rows else pd.DataFrame()
    evidence = pd.concat(evidence_rows, ignore_index=True) if evidence_rows else pd.DataFrame()
    return posterior, evidence, weights, source_validation


def _strict_posterior_predictive_sources(
    availability: pd.DataFrame,
    registry: pd.DataFrame,
    model_ids: list[str],
) -> pd.DataFrame:
    ""











    if "model_id" not in availability.columns:
        raise SystemExit("posterior-predictive availability missing model_id")
    rows = availability.copy()
    rows["model_id"] = rows["model_id"].astype(str)
    rows = rows[rows["model_id"].isin(model_ids)].copy()
    duplicates = rows[rows.duplicated("model_id", keep=False)]["model_id"].unique().tolist()
    if duplicates:
        raise SystemExit(
            f"posterior-predictive availability has duplicate selected model rows: {duplicates}"
        )
    missing_models = sorted(set(model_ids) - set(rows["model_id"]))
    if missing_models:
        rows = pd.concat(
            [
                rows,
                pd.DataFrame(
                    [
                        {
                            "model_id": model_id,
                            "posterior_predictive_log_density_available": False,
                            "posterior_predictive_fallback_reason": (
                                "no availability row; strict posterior-predictive "
                                "contract is unproved"
                            ),
                        }
                        for model_id in missing_models
                    ]
                ),
            ],
            ignore_index=True,
        )

    family_map = (
        registry.set_index("model_id")["family"].astype(str).to_dict()
        if "family" in registry.columns
        else {}
    )
    required_positive = {
        "posterior_predictive_schema": STRICT_PPD_SCHEMA,
        "posterior_predictive_target_scale": "shared_bridge_transformed_target",
        "posterior_predictive_base_measure": (
            "lebesgue_on_shared_bridge_transformed_target"
        ),
    }
    required_true = [
        "posterior_predictive_asof_origin",
        "posterior_predictive_is_model_native",
        "posterior_predictive_integrates_parameter_or_state_uncertainty",
    ]
    validation_rows: list[dict[str, object]] = []
    for model_id in model_ids:
        row = rows[rows["model_id"].eq(model_id)].iloc[0]
        available = _bool_value(
            row.get("posterior_predictive_log_density_available", False)
        )
        if available:
            failures = [
                f"{key}={row.get(key, '')!r}"
                for key, expected in required_positive.items()
                if str(row.get(key, "")).strip() != expected
            ]
            failures.extend(
                f"{key}={row.get(key, '')!r}"
                for key in required_true
                if not _bool_value(row.get(key, False))
            )
            density_type = str(
                row.get("posterior_predictive_log_density_type", "")
            ).strip()
            if not density_type:
                failures.append("posterior_predictive_log_density_type is empty")
            if failures:
                raise SystemExit(
                    f"{model_id} declares a strict posterior-predictive density but "
                    f"violates {STRICT_PPD_SCHEMA}: {failures}"
                )
            source = STRICT_PPD_SCORE_SOURCE
            fallback_reason = ""
        else:
            density_type = str(
                row.get("posterior_predictive_log_density_type", "")
            ).strip()
            source = SHARED_FALLBACK_SCORE_SOURCE
            fallback_reason = str(
                row.get(
                    "posterior_predictive_fallback_reason",
                    row.get("fallback_reason", row.get("blocker", "")),
                )
                or ""
            ).strip()
            if not fallback_reason:
                alternate_status = str(row.get("native_likelihood_status", "")).strip().lower()
                if alternate_status == "true_native":
                    fallback_reason = (
                        "alternate adapter-defined native-like score does not declare "
                        f"{STRICT_PPD_SCHEMA}"
                    )
                else:
                    fallback_reason = "no validationed model posterior-predictive log-density"
        validation_rows.append(
            {
                "model_id": model_id,
                "family": family_map.get(model_id, ""),
                "posterior_predictive_schema": str(
                    row.get("posterior_predictive_schema", "")
                ).strip(),
                "posterior_predictive_log_density_available": available,
                "posterior_predictive_log_density_type": density_type,
                "posterior_predictive_target_scale": str(
                    row.get("posterior_predictive_target_scale", "")
                ).strip(),
                "posterior_predictive_base_measure": str(
                    row.get("posterior_predictive_base_measure", "")
                ).strip(),
                "posterior_predictive_asof_origin": _bool_value(
                    row.get("posterior_predictive_asof_origin", False)
                ),
                "posterior_predictive_is_model_native": _bool_value(
                    row.get("posterior_predictive_is_model_native", False)
                ),
                "posterior_predictive_integrates_parameter_or_state_uncertainty": _bool_value(
                    row.get(
                        "posterior_predictive_integrates_parameter_or_state_uncertainty",
                        False,
                    )
                ),
                "alternate_native_likelihood_status": str(
                    row.get("native_likelihood_status", "")
                ).strip(),
                "update_score_source": source,
                "fallback_reason": fallback_reason,
            }
        )
    return pd.DataFrame(validation_rows)


def _strict_posterior_predictive_scores(
    native_scores: pd.DataFrame,
    eligible_model_ids: set[str],
) -> pd.DataFrame:
    ""

    columns = ["forecast_id", "model_id", "log_score"]
    if not eligible_model_ids:
        return pd.DataFrame(columns=columns)
    required = {
        "forecast_id",
        "model_id",
        "posterior_predictive_log_density",
        "posterior_predictive_schema",
        "posterior_predictive_target_scale",
        "posterior_predictive_base_measure",
    }
    missing = sorted(required - set(native_scores.columns))
    if missing:
        raise SystemExit(f"strict posterior-predictive scores missing columns: {missing}")
    rows = native_scores.copy()
    rows["model_id"] = rows["model_id"].astype(str)
    rows = rows[rows["model_id"].isin(eligible_model_ids)].copy()
    checks = {
        "posterior_predictive_schema": STRICT_PPD_SCHEMA,
        "posterior_predictive_target_scale": "shared_bridge_transformed_target",
        "posterior_predictive_base_measure": (
            "lebesgue_on_shared_bridge_transformed_target"
        ),
    }
    for column, expected in checks.items():
        bad = ~rows[column].astype(str).eq(expected)
        if bad.any():
            sample = rows.loc[bad, ["model_id", "forecast_id", column]].head(5).to_dict("records")
            raise SystemExit(
                f"strict posterior-predictive score rows violate {column}={expected!r}: {sample}"
            )
    if "score_source" in rows.columns:
        bad = ~rows["score_source"].astype(str).eq(STRICT_PPD_SCORE_SOURCE)
        if bad.any():
            sample = rows.loc[bad, ["model_id", "forecast_id", "score_source"]].head(5).to_dict("records")
            raise SystemExit(f"strict posterior-predictive score_source mismatch: {sample}")
    rows["log_score"] = pd.to_numeric(
        rows["posterior_predictive_log_density"], errors="raise"
    )
    if not np.isfinite(rows["log_score"].to_numpy(dtype=float)).all():
        raise SystemExit("strict posterior-predictive log-density must be finite")
    return rows[columns]


def _replay_strict_posterior_predictive_fallback_evidence(
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    draws: pd.DataFrame | None,
    bridge,
    score_source: str,
    model_ids: list[str],
    initial_weights: pd.DataFrame,
    native_scores: pd.DataFrame,
    availability: pd.DataFrame,
    registry: pd.DataFrame,
    rho: float,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ""

    strict_validation = _strict_posterior_predictive_sources(
        availability, registry, model_ids
    )
    eligible = set(
        strict_validation.loc[
            strict_validation["update_score_source"].eq(STRICT_PPD_SCORE_SOURCE),
            "model_id",
        ].astype(str)
    )
    normalized_scores = _strict_posterior_predictive_scores(native_scores, eligible)
    compatibility_availability = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "native_likelihood_status": (
                    "true_native" if model_id in eligible else "unavailable"
                ),
                "supports_native_log_likelihood": model_id in eligible,
                "fallback_reason": strict_validation.set_index("model_id").loc[
                    model_id, "fallback_reason"
                ],
            }
            for model_id in model_ids
        ]
    )
    posterior, evidence, weights, compatibility_validation = _replay_native_fallback_evidence(
        ledger,
        archive,
        draws,
        bridge,
        score_source,
        model_ids,
        initial_weights,
        normalized_scores,
        compatibility_availability,
        registry,
        rho,
        phase,
    )
    coverage = compatibility_validation[
        ["model_id", "required_pretest_rows", "covered_native_rows"]
    ].rename(columns={"covered_native_rows": "covered_posterior_predictive_rows"})
    strict_validation = strict_validation.merge(
        coverage, on="model_id", how="left", validate="one_to_one"
    )
    if not evidence.empty:
        evidence["score_update_basis"] = evidence["score_update_basis"].replace(
            {"adapter_native_likelihood": STRICT_PPD_SCORE_SOURCE}
        )
    return posterior, evidence, weights, strict_validation


def _static_test_readout(
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    weights: pd.DataFrame,
    readout_split: str,
    cutoff: pd.Timestamp,
    bridge,
    score_source: str = "archive_moment",
    draws: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if (
        str(getattr(bridge, "predictive_contract", alternate_ARCHIVE_MOMENT))
        != alternate_ARCHIVE_MOMENT
    ):
        return _nonalternate_static_readout(
            ledger,
            archive,
            weights,
            readout_split,
            bridge,
            score_source,
            draws,
            snapshot_time=cutoff,
            used_prior=False,
            policy="offline_frozen_pretest",
        )

                                                                              
    test = ledger[ledger["split"].astype(str).eq(str(readout_split))].copy()
    fids = set(test["forecast_id"].astype(str))
    preds = _weighted_predictions(archive[archive["forecast_id"].astype(str).isin(fids)], weights, bridge)
    return _readout_from_predictions(
        test, preds, snapshot_time=cutoff, used_prior=False, policy="offline_frozen_pretest"
    )


def _run_nested_posterior_stage(
    args,
    *,
    stage: str,
    method_id: str,
    added_component: str,
    parent_stage: str,
    only_added_component: str,
    mode: str,
) -> None:
    if not args.bridge_config:
        raise SystemExit(f"{stage} requires --bridge-config")
    if args.prior_policy != "uniform_model":
        raise SystemExit(f"{stage} requires --prior-policy uniform_model")
    native_diagnostic = mode == "offline_native_diagnostic"
    strict_ppd_diagnostic = mode == "offline_strict_ppd_diagnostic"
    offline_mode = mode in {
        "offline",
        "offline_native_diagnostic",
        "offline_strict_ppd_diagnostic",
    }
    if native_diagnostic and (
        not args.native_scores or not args.native_availability
    ):
        raise SystemExit(f"{stage} requires --native-scores and --native-availability")
    if mode in {"online", "one_layer"} and args.posterior_update_policy != "prequential_asof":
        raise SystemExit(f"{stage} requires --posterior-update-policy prequential_asof")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = RuntimeLogger()
    with timer.measure("load_inputs"):
        ledger = pd.read_csv(args.ledger)
        archive_all = pd.read_csv(args.archive)
        registry_all = pd.read_csv(args.registry)
        selection = pd.read_csv(args.selection)
        registry, model_ids = _selected_registry_many(registry_all, selection)
        archive = archive_all[archive_all["model_id"].astype(str).isin(model_ids)].copy()
        if archive.empty:
            raise SystemExit("archive has no rows for selected Top-K models")
        bridge, rho = read_bridge_config(args.bridge_config)
        score_source = _bridge_score_source(args.bridge_config)
        draws = (
            pd.read_csv(_require_draws(args.draws, args.bridge_config))
            if score_source == "draw_kernel"
            else None
        )
        rho_value = 1.0 if rho is None else float(rho)
        if mode in {
            "offline",
            "offline_native_diagnostic",
            "offline_strict_ppd_diagnostic",
            "online",
        } and abs(rho_value - 1.0) > 1e-12:
            raise SystemExit(f"{stage} requires rho=1 bridge config")
        bridge_metadata = _read_bridge_metadata(args.bridge_config)
        predictive_contract = _frozen_predictive_contract(
            bridge, bridge_metadata
        )
        selection.to_csv(out_dir / "candidate_selection_log.csv", index=False)
        registry.to_csv(out_dir / "model_registry.csv", index=False)

    with timer.measure("archive_contract_validation"):
        _validate_archive_contract(archive, ledger, model_ids, out_dir)

    cutoff, pretest_ledger, online_ledger = _evidence_partitions(ledger, args.readout_split)
    initial_weights = _uniform_weights(registry, model_ids)
    initial_weights.to_csv(out_dir / "initial_prior.csv", index=False)
    archive_hash = _sha256_file(args.archive)
    candidate_set_hash = _canonical_hash(sorted(model_ids))
    bridge_core_hash = _bridge_core_hash(args.bridge_config)
    parent_metadata: dict[str, object] = {}
    inherited_static_weights_path: Path | None = None

    if mode == "online":
        if not args.offline_stage_root:
            raise SystemExit(f"{stage} requires --offline-stage-root")
        offline_root = Path(args.offline_stage_root)
        required = [offline_root / name for name in ["static_weights.csv", "posterior_path.csv", "evidence_log.csv", "stage_metadata.json"]]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SystemExit(f"offline parent artifacts missing: {missing}")
        parent_metadata = json.loads((offline_root / "stage_metadata.json").read_text(encoding="utf-8"))
        expected = {
            "stage": STAGE_A3,
            "candidate_set_hash": candidate_set_hash,
            "archive_hash": archive_hash,
            "bridge_core_hash": bridge_core_hash,
            "prior_policy": "uniform_model_prior",
        }
        if predictive_contract != alternate_ARCHIVE_MOMENT:
            expected["predictive_contract"] = predictive_contract
        for key, value in expected.items():
            if parent_metadata.get(key) != value:
                raise SystemExit(f"offline/online parameter mismatch for {key}: {parent_metadata.get(key)!r} != {value!r}")
        if abs(float(parent_metadata.get("rho", float("nan"))) - 1.0) > 1e-12:
            raise SystemExit("offline/online rho mismatch")
        static_weights = pd.read_csv(offline_root / "static_weights.csv")[["model_id", "family", "weight"]]
        if _weights_hash(static_weights) != str(parent_metadata.get("pretest_weight_hash")):
            raise SystemExit("offline static weight hash does not match parent metadata")
        pretest_posterior = pd.read_csv(offline_root / "posterior_path.csv")
        pretest_evidence = pd.read_csv(offline_root / "evidence_log.csv")
        pretest_weights = static_weights.copy()
        inherited_static_weights_path = offline_root / "static_weights.csv"
    else:
        if mode == "one_layer":
            if not args.online_stage_root:
                raise SystemExit(f"{stage} requires --online-stage-root")
            online_metadata_path = Path(args.online_stage_root) / "stage_metadata.json"
            if not online_metadata_path.exists():
                raise SystemExit(f"online parent metadata missing: {online_metadata_path}")
            parent_metadata = json.loads(online_metadata_path.read_text(encoding="utf-8"))
            expected = {
                "stage": STAGE_A4,
                "candidate_set_hash": candidate_set_hash,
                "archive_hash": archive_hash,
                "bridge_core_hash": bridge_core_hash,
                "prior_policy": "uniform_model_prior",
                "test_start_cutoff": cutoff.isoformat(),
            }
            if predictive_contract != alternate_ARCHIVE_MOMENT:
                expected["predictive_contract"] = predictive_contract
            for key, value in expected.items():
                if parent_metadata.get(key) != value:
                    raise SystemExit(f"online/one-layer parameter mismatch for {key}: {parent_metadata.get(key)!r} != {value!r}")
        with timer.measure("pretest_posterior_replay"):
            if strict_ppd_diagnostic:
                native_scores = (
                    pd.read_csv(args.native_scores)
                    if args.native_scores
                    else pd.DataFrame()
                )
                native_availability = (
                    pd.read_csv(args.native_availability)
                    if args.native_availability
                    else pd.DataFrame(columns=["model_id"])
                )
                pretest_posterior, pretest_evidence, pretest_weights, ppd_source_validation = (
                    _replay_strict_posterior_predictive_fallback_evidence(
                        pretest_ledger,
                        archive,
                        draws,
                        bridge,
                        score_source,
                        model_ids,
                        initial_weights,
                        native_scores,
                        native_availability,
                        registry,
                        rho_value,
                        "pretest_strict_posterior_predictive_fallback_diagnostic",
                    )
                )
                ppd_source_validation.to_csv(
                    out_dir / "posterior_predictive_source_validation.csv", index=False
                )
            elif native_diagnostic:
                native_scores = pd.read_csv(args.native_scores)
                native_availability = pd.read_csv(args.native_availability)
                pretest_posterior, pretest_evidence, pretest_weights, native_source_validation = (
                    _replay_native_fallback_evidence(
                        pretest_ledger,
                        archive,
                        draws,
                        bridge,
                        score_source,
                        model_ids,
                        initial_weights,
                        native_scores,
                        native_availability,
                        registry,
                        rho_value,
                        "pretest_native_fallback_diagnostic",
                    )
                )
                native_source_validation.to_csv(out_dir / "native_fallback_source_validation.csv", index=False)
            else:
                pretest_posterior, pretest_evidence, pretest_weights = _replay_model_evidence(
                    pretest_ledger,
                    archive,
                    draws,
                    bridge,
                    score_source,
                    model_ids,
                    initial_weights,
                    rho_value,
                    "pretest_development",
                )
        if pretest_posterior.empty:
            pretest_posterior = pretest_weights.copy()
            pretest_posterior["release_time"] = cutoff
            pretest_posterior["rho"] = rho_value
            pretest_posterior["update_phase"] = "pretest_prior_freeze"
        if pretest_evidence.empty:
            pretest_evidence = pd.DataFrame(columns=["release_time", "model_id", "log_evidence", "n_scored_rows", "score_update_basis", "update_phase"])

    pretest_weight_hash = _weights_hash(pretest_weights)
    online_posterior = pd.DataFrame()
    online_evidence = pd.DataFrame()
    if mode in {"online", "one_layer"}:
        with timer.measure("online_posterior_replay"):
            online_posterior, online_evidence, final_weights = _replay_model_evidence(
                online_ledger,
                archive,
                draws,
                bridge,
                score_source,
                model_ids,
                pretest_weights,
                rho_value,
                "post_cutoff_prequential",
            )
    else:
        final_weights = pretest_weights.copy()

    posterior = pd.concat([frame for frame in [pretest_posterior, online_posterior] if not frame.empty], ignore_index=True)
    evidence_parts = [frame for frame in [pretest_evidence, online_evidence] if not frame.empty]
    evidence = pd.concat(evidence_parts, ignore_index=True) if evidence_parts else pretest_evidence.copy()
    posterior["release_time"] = pd.to_datetime(posterior["release_time"], errors="raise").dt.strftime("%Y-%m-%dT%H:%M:%S")
    if not evidence.empty:
        evidence["release_time"] = pd.to_datetime(evidence["release_time"], errors="raise").dt.strftime("%Y-%m-%dT%H:%M:%S")
    posterior.to_csv(out_dir / "posterior_path.csv", index=False)
    evidence.to_csv(out_dir / "evidence_log.csv", index=False)
    final_snapshot = final_weights[["model_id", "family", "weight"]].copy()
    final_snapshot["rho"] = rho_value
    final_snapshot.to_csv(out_dir / "posterior_weights.csv", index=False)
    static = pretest_weights[["model_id", "family", "weight"]].copy()
    static["included_for_weight"] = True
    static["weight_policy"] = "uniform_prior_pretest_evidence_frozen"
    static["prior_policy"] = "uniform_model_prior"
    static["rho"] = rho_value
    static_path = out_dir / "static_weights.csv"
    if inherited_static_weights_path is not None:
        shutil.copy2(inherited_static_weights_path, static_path)
    else:
        static.to_csv(static_path, index=False)
                                                                            
                                                                               
                                                                              
    pretest_weight_hash = _weights_hash(pd.read_csv(static_path))
    model_distribution = _distribution_summary(final_weights)
    if mode in {"online", "one_layer"}:
        _write_json(model_distribution, out_dir / "model_distribution.json")

    with timer.measure("forecast_readout"):
        readout = (
            _static_test_readout(
                ledger,
                archive,
                pretest_weights,
                args.readout_split,
                cutoff,
                bridge,
                score_source,
                draws,
            )
            if offline_mode
            else _build_asof_weighted_readout(
                ledger,
                archive,
                posterior,
                initial_weights,
                args.readout_split,
                bridge,
                score_source,
                draws,
            )
        )
        readout_path = out_dir / "forecast_readout.csv"
        readout.to_csv(readout_path, index=False)

    with timer.measure("exact_mixture_nll"):
        exact_scores, exact_validation_path = _exact_stage_scores(
            dataset=_stage_dataset(readout, args.ledger), method_id=method_id,
            ledger_path=args.ledger, archive_path=args.archive, bridge_config=args.bridge_config,
            stage_root=out_dir, readout_path=readout_path, draws_path=args.draws,
        )
    with timer.measure("metrics"):
        metrics = _stage_metrics(
            readout, stage, method_id, str(readout_path), bridge_config=args.bridge_config,
            formal_scores=exact_scores, asof_mixture_weight_validation_path=exact_validation_path,
        )
        metrics.to_csv(out_dir / "stage_metrics.csv", index=False)

    _asof_validation(readout, stage=stage, method_id=method_id).to_csv(out_dir / "asof_posterior_readout_validation.csv", index=False)
    late_development_rows = int(
        online_ledger["split"].astype(str).isin(["train", "val", "embargo"]).sum()
    )
    embargo_pretest_rows = int(pretest_ledger["split"].astype(str).eq("embargo").sum())
    embargo_online_rows = int(online_ledger["split"].astype(str).eq("embargo").sum())
    test_update_rows = int(online_ledger["split"].astype(str).eq(str(args.readout_split)).sum()) if not offline_mode else 0
    likelihood_policy = (
        "model_posterior_predictive_if_strictly_eligible_shared_bridge_fallback"
        if strict_ppd_diagnostic
        else (
            "adapter_native_if_available_shared_bridge_fallback"
            if native_diagnostic
            else "shared_calibrated_bridge"
        )
    )
    update_basis = (
        "validationed_model_posterior_predictive_or_shared_bridge_diagnostic"
        if strict_ppd_diagnostic
        else (
            "heterogeneous_adapter_native_or_shared_bridge_diagnostic"
            if native_diagnostic
            else (
            "shared_calibrated_draw_kernel_bridge"
            if score_source == "draw_kernel"
            else "shared_calibrated_archive_moment_bridge"
            )
        )
    )
    score_contract = _formal_score_contract(score_source)
    metadata = {
        "stage": stage, "method_id": method_id, "added_component": added_component,
        "ablation_parent_stage": parent_stage, "only_added_component": only_added_component,
        "ablation_design_version": (
            "bayesian_update_P1_family_v7_strict_ppd"
            if strict_ppd_diagnostic
            else "bayesian_update_P1_family_v6"
        ),
        "selected_model_count": int(len(model_ids)), "selected_model_ids": model_ids, "active_model_set": model_ids,
        "candidate_set_hash": candidate_set_hash, "archive_hash": archive_hash, "bridge_core_hash": bridge_core_hash,
        "calibration_model_set": bridge_metadata.get("calibration_model_set", []),
        "bridge_config": str(args.bridge_config), "bridge_distribution": bridge.distribution,
        "bridge_transform": bridge.transform,
        "predictive_contract": predictive_contract,
        "rho": rho_value,
        "score_source": score_source,
        "selected_bridge_family": bridge_metadata.get("selected_bridge_family", "moment_t"),
        "draws": str(args.draws) if score_source == "draw_kernel" else "",
        "draws_hash": _sha256_file(args.draws) if score_source == "draw_kernel" else "",
        "filter_dynamics": "bayesian_evidence_update",
        "likelihood_policy": likelihood_policy, "posterior_update": not offline_mode,
        "online_update": not offline_mode,
        "pretest_posterior_update": True,
        "posterior_update_policy": "none_static_frozen_pretest" if offline_mode else "prequential_asof",
        "score_update_basis": update_basis, "prior_policy": "uniform_model_prior",
        "initial_prior_family_mass": initial_weights.groupby("family")["weight"].sum().sort_index().to_dict(),
        "initial_prior_model_ess": float(_distribution_summary(initial_weights)["model_ess"]),
        "test_start_cutoff": cutoff.isoformat(), "pretest_weight_hash": pretest_weight_hash,
        "parent_pretest_weight_hash": parent_metadata.get("pretest_weight_hash", ""),
        "readout_split": str(args.readout_split), "seed": int(args.seed), "ledger": str(args.ledger),
        "archive": str(args.archive), "registry": str(args.registry), "selection": str(args.selection),
        "ledger_rows": int(len(ledger)), "development_evidence_rows": int(len(pretest_ledger)),
        "late_development_evidence_rows": late_development_rows,
        "embargo_rows_in_ledger": int(ledger["split"].astype(str).eq("embargo").sum()),
        "embargo_rows_used_for_selection": 0,
        "embargo_rows_used_for_bridge_calibration": 0,
        "embargo_rows_used_for_reported_metrics": 0,
        "embargo_rows_used_for_pretest_weighting": embargo_pretest_rows,
        "embargo_rows_used_for_online_posterior_update": embargo_online_rows if not offline_mode else 0,
        "embargo_posterior_update_policy": "released_evidence_only_asof",
        "posterior_update_rows": int(len(pretest_ledger) + (0 if offline_mode else len(online_ledger))),
        "archive_rows": int(len(archive)), "readout_rows": int(len(readout)),
        "test_rows_used_for_bridge_calibration": int(bridge_metadata.get("test_rows_used_for_bridge_calibration", bridge_metadata.get("test_rows_used_for_tuning", 0))),
        "test_rows_used_for_tuning": int(bridge_metadata.get("test_rows_used_for_tuning", 0)),
        "test_rows_used_for_weighting": 0, "test_rows_used_for_posterior_update": test_update_rows,
        "test_rows_used_for_posterior_update_policy": "none_static" if offline_mode else "released_evidence_only_asof",
        "readout_rows_future_snapshot_violation": int(readout["future_snapshot_violation"].astype(bool).sum()),
        "readout_rows_self_target_update_violation": int(readout["self_target_update_violation"].astype(bool).sum()),
        "model_distribution": str(out_dir / "model_distribution.json") if mode in {"online", "one_layer"} else "",
        "final_model_distribution": model_distribution,
        "native_likelihoods_compared": native_diagnostic,
        "strict_posterior_predictive_log_densities_compared": strict_ppd_diagnostic,
        **score_contract,
        "diagnostic_moment_matched_nll_available": score_source == "archive_moment",
        "diagnostic_moment_matched_nll_available": True, "asof_mixture_weight_validation": str(exact_validation_path),
        "incremental_ablation": True,
        "parameter_selection_performed": False,
        "eta_selection_performed": False,
        "rho_selection_performed": False,
        "fixed_rho": rho_value,
    }
    if native_diagnostic:
        metadata.update(
            {
                "diagnostic_only": True,
                "unsafe_diagnostic": True,
                "not_for_positive_claim": True,
                "posterior_evidence_incomparable_by_design": True,
                "score_scale_harmonization_performed": False,
                "change_of_variable_correction_applied": False,
                "base_measure_harmonization_performed": False,
                "weight_semantics": "diagnostic_pseudo_weights_not_bayesian_posterior",
                "diagnostic_control_stage": STAGE_A3,
                "native_likelihood_scores": str(args.native_scores),
                "native_likelihood_availability": str(args.native_availability),
                "native_fallback_source_validation": str(out_dir / "native_fallback_source_validation.csv"),
                "reported_test_nll_policy": "exact_common_shared_bridge_mixture",
                "native_or_fallback_evidence_used_for_pretest_weighting": True,
                "native_or_fallback_scores_used_for_reported_test_nll": False,
            }
        )
    if strict_ppd_diagnostic:
        eligible_count = int(
            ppd_source_validation["posterior_predictive_log_density_available"]
            .astype(bool)
            .sum()
        )
        metadata.update(
            {
                "diagnostic_only": True,
                "unsafe_diagnostic": False,
                "not_for_positive_claim": True,
                "posterior_evidence_incomparable_by_design": False,
                "diagnostic_control_stage": STAGE_A3,
                "strict_posterior_predictive_contract": STRICT_PPD_SCHEMA,
                "strict_posterior_predictive_eligible_model_count": eligible_count,
                "strict_posterior_predictive_fallback_model_count": int(
                    len(model_ids) - eligible_count
                ),
                "strict_posterior_predictive_noop_against_a3": eligible_count == 0,
                "posterior_predictive_scores": str(args.native_scores or ""),
                "posterior_predictive_availability": str(
                    args.native_availability or ""
                ),
                "posterior_predictive_source_validation": str(
                    out_dir / "posterior_predictive_source_validation.csv"
                ),
                "reported_test_nll_policy": "exact_common_shared_bridge_mixture",
                "posterior_predictive_or_fallback_evidence_used_for_pretest_weighting": True,
                "posterior_predictive_scores_used_for_reported_test_nll": False,
            }
        )
    if not offline_mode:
        metadata.update(availability_validation_metadata(evidence))
    if bridge_metadata.get("rho_grid"):
        metadata["rho_grid"] = bridge_metadata.get("rho_grid")
    if bridge_metadata.get("target_ess_fraction") is not None:
        metadata["target_ess_fraction"] = bridge_metadata.get("target_ess_fraction")
    if bridge_metadata.get("temperature_report"):
        metadata["temperature_report"] = bridge_metadata.get("temperature_report")
    _write_json(metadata, out_dir / "stage_metadata.json")
    write_timing_log(timer.summary(seed=args.seed), out_dir / "timing.json")
    print(
        f"ok stage={stage} out={out_dir} selected_model_count={len(model_ids)} "
        f"readout_rows={len(readout)} mode={mode} rho={rho_value:.6g}"
    )


def run_a3(args) -> None:
    _run_nested_posterior_stage(
        args, stage=STAGE_A3, method_id=METHOD_A3, added_component="Frozen pre-test posterior (ρ=1)",
        parent_stage=STAGE_A2, only_added_component="pretest_development_evidence_weighting_frozen_for_test", mode="offline",
    )


def run_a3_native(args) -> None:
    _run_nested_posterior_stage(
        args,
        stage=STAGE_A3_NATIVE,
        method_id=METHOD_A3_NATIVE,
        added_component="Naive heterogeneous adapter-native/fallback evidence (stress test)",
        parent_stage=STAGE_A3,
        only_added_component=(
            "replace_shared_bridge_pretest_evidence_with_unharmonized_"
            "adapter_native_or_fallback_sources"
        ),
        mode="offline_native_diagnostic",
    )


def run_a3_strict_ppd(args) -> None:
    _run_nested_posterior_stage(
        args,
        stage=STAGE_A3_STRICT_PPD,
        method_id=METHOD_A3_STRICT_PPD,
        added_component=(
            "Frozen pre-test posterior — strict posterior-predictive/shared-bridge "
            "evidence, diagnostic"
        ),
        parent_stage=STAGE_A3,
        only_added_component=(
            "replace_shared_bridge_pretest_evidence_only_for_strictly_eligible_"
            "model_posterior_predictive_scores"
        ),
        mode="offline_strict_ppd_diagnostic",
    )


def run_a4(args) -> None:
    _run_nested_posterior_stage(
        args, stage=STAGE_A4, method_id=METHOD_A4, added_component="Causal online posterior (ρ=1)",
        parent_stage=STAGE_A3, only_added_component="post_cutoff_prequential_posterior_updates", mode="online",
    )


def run_a5(args) -> None:
    _run_nested_posterior_stage(
        args,
        stage=STAGE_A5,
        method_id=METHOD_A5,
        added_component="Causal online posterior (validation-selected ρ)",
        parent_stage=STAGE_A4,
        only_added_component="validation_selected_rho",
        mode="one_layer",
    )



def main() -> None:
    ap = ArgumentParser(description="Run incremental cumulative ablation stages from a frozen forecast archive.")
    ap.add_argument(
        "--stage",
        required=True,
        choices=[
            STAGE_A0,
            STAGE_A1,
            STAGE_A2,
            STAGE_A3,
            STAGE_A3_NATIVE,
            STAGE_A3_STRICT_PPD,
            STAGE_A4,
            STAGE_A5,
        ],
    )
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--draws", default="")
    ap.add_argument("--registry", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--bridge-config", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--naive-scale-source", default="train_residual_global", choices=["train_residual_global", "fixed_one"])
    ap.add_argument("--posterior-update-policy", default="", choices=["", "prequential_asof"])
    ap.add_argument(
        "--prior-policy",
        default="uniform_model",
        choices=["uniform_model", "hierarchical_family_balanced"],
    )
    ap.add_argument("--offline-stage-root", default="")
    ap.add_argument("--online-stage-root", default="")
    ap.add_argument("--native-scores", default="")
    ap.add_argument("--native-availability", default="")
    ap.add_argument("--readout-split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.stage == STAGE_A0:
        run_a0(args)
    elif args.stage == STAGE_A1:
        run_a1(args)
    elif args.stage == STAGE_A2:
        run_a2(args)
    elif args.stage == STAGE_A3:
        run_a3(args)
    elif args.stage == STAGE_A3_NATIVE:
        run_a3_native(args)
    elif args.stage == STAGE_A3_STRICT_PPD:
        run_a3_strict_ppd(args)
    elif args.stage == STAGE_A4:
        run_a4(args)
    elif args.stage == STAGE_A5:
        run_a5(args)


if __name__ == "__main__":
    main()
