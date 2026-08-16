from __future__ import annotations

from math import log, pi
from typing import Any

import numpy as np
import pandas as pd


class NativeProxyBlocker(ValueError):
    def __init__(self, message: str, *, status: str = "blocked_native_proxy", validation: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.validation = validation or {}


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes", "y"})


def _require_columns(df: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise NativeProxyBlocker(f"{label} missing required columns {missing}", status="blocked_native_proxy_missing_columns")


def _validation_pred_var(values: pd.Series, *, min_variance_floor: float) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    raw = pd.to_numeric(values, errors="coerce")
    missing = raw.isna()
    finite = np.isfinite(raw.astype(float).to_numpy())
    nonfinite = ~missing & ~pd.Series(finite, index=raw.index)
    if nonfinite.any():
        raise NativeProxyBlocker("native-proxy blocker: pred_var contains nonfinite values", status="blocked_native_proxy_invalid_archive")
    if (raw.dropna() < 0).any():
        raise NativeProxyBlocker("native-proxy blocker: pred_var contains negative values", status="blocked_native_proxy_invalid_archive")
    zero = raw.notna() & raw.eq(0.0)
    floor_applied = missing | raw.fillna(0.0).le(float(min_variance_floor))
    used = raw.fillna(float(min_variance_floor)).clip(lower=float(min_variance_floor))
    total = int(len(raw))
    floor_count = int(floor_applied.sum())
    validation = {
        "pred_var_missing_count": int(missing.sum()),
        "pred_var_zero_count": int(zero.sum()),
        "floor_applied_count": floor_count,
        "floor_applied_fraction": float(floor_count / total) if total else 0.0,
        "native_proxy_min_variance_floor": float(min_variance_floor),
    }
    return used.astype(float), floor_applied.astype(bool), validation


def _validate_event_contract(joined: pd.DataFrame) -> None:
    origin = pd.to_datetime(joined["forecast_origin_ledger"], errors="coerce")
    target = pd.to_datetime(joined["target_time_ledger"], errors="coerce")
    release = pd.to_datetime(joined["release_time"], errors="coerce")
    if origin.isna().any() or target.isna().any() or release.isna().any():
        raise NativeProxyBlocker("native-proxy blocker: invalid ledger event times", status="blocked_native_proxy_invalid_ledger")
    violations = ~(origin < target) | ~(target <= release)
    if violations.any():
        raise NativeProxyBlocker(
            f"native-proxy blocker: {int(violations.sum())} event rows violate forecast_origin < target_time <= release_time",
            status="blocked_native_proxy_event_contract",
        )


def _validate_event_metadata(joined: pd.DataFrame) -> None:
    origin_archive = pd.to_datetime(joined["forecast_origin"], errors="coerce")
    origin_ledger = pd.to_datetime(joined["forecast_origin_ledger"], errors="coerce")
    target_archive = pd.to_datetime(joined["target_time"], errors="coerce")
    target_ledger = pd.to_datetime(joined["target_time_ledger"], errors="coerce")
    component_mismatch = joined["component"].astype(str) != joined["component_ledger"].astype(str)
    mismatch = (origin_archive != origin_ledger) | (target_archive != target_ledger) | component_mismatch
    if mismatch.any():
        sample = joined.loc[mismatch, "forecast_id"].astype(str).unique()[:10]
        raise NativeProxyBlocker(
            f"native-proxy blocker: archive/ledger event metadata mismatch for forecast_id {','.join(sample)}",
            status="blocked_native_proxy_event_mismatch",
        )


def _validate_selected_coverage(joined: pd.DataFrame, ledger_ids: set[str], selected_model_ids: list[str]) -> None:
    ledger_ids = {str(x) for x in ledger_ids}
    sorted_ledger_ids = sorted(ledger_ids)
    selected_model_ids = [str(x) for x in selected_model_ids]
    expected_model_count = len(set(selected_model_ids))

    pairs = joined[["forecast_id", "model_id"]].drop_duplicates().copy()
    pairs["forecast_id"] = pairs["forecast_id"].astype(str)
    pairs["model_id"] = pairs["model_id"].astype(str)
    model_counts = pairs.groupby("forecast_id", sort=False)["model_id"].nunique()
    bad_forecast_ids = sorted(
        fid for fid in ledger_ids if int(model_counts.get(fid, 0)) != expected_model_count
    )
    if bad_forecast_ids:
        bad_pairs = pairs[pairs["forecast_id"].isin(bad_forecast_ids[:10])]
        present_by_forecast = bad_pairs.groupby("forecast_id")["model_id"].agg(lambda s: set(s))
        missing = []
        for forecast_id in bad_forecast_ids[:10]:
            present = present_by_forecast.get(forecast_id, set())
            absent = [mid for mid in selected_model_ids if mid not in present]
            missing.append(f"{forecast_id}:{','.join(absent[:5])}")
        raise NativeProxyBlocker(
            f"native-proxy blocker: archive missing selected model predictions for {missing}",
            status="blocked_native_proxy_missing_archive_rows",
        )

    particle_pairs = joined[["model_id", "particle_id", "forecast_id"]].drop_duplicates().copy()
    particle_pairs["model_id"] = particle_pairs["model_id"].astype(str)
    particle_pairs["forecast_id"] = particle_pairs["forecast_id"].astype(str)
    particle_counts = particle_pairs.groupby(["model_id", "particle_id"], sort=False)["forecast_id"].nunique()
    bad_particle_keys = [
        key for key, count in particle_counts.items() if int(count) != len(ledger_ids)
    ]
    if bad_particle_keys:
        bad_key_set = set(bad_particle_keys[:10])
        bad_rows = particle_pairs[
            particle_pairs[["model_id", "particle_id"]].apply(tuple, axis=1).isin(bad_key_set)
        ]
        covered_by_particle = bad_rows.groupby(["model_id", "particle_id"])["forecast_id"].agg(lambda s: set(s))
        missing = []
        for model_id, particle_id in bad_particle_keys[:10]:
            covered = covered_by_particle.get((model_id, particle_id), set())
            missing_ids = sorted_ledger_ids if not covered else sorted(ledger_ids - covered)
            missing.append(f"{model_id}/{particle_id}: {missing_ids[:10]}")
        raise NativeProxyBlocker(
            f"native-proxy blocker: archive missing particle coverage for {missing}",
            status="blocked_native_proxy_missing_archive_rows",
        )


def score_native_proxy_rows(
    update_ledger: pd.DataFrame,
    update_archive: pd.DataFrame,
    selected_model_ids: list[str],
    *,
    min_variance_floor: float = 1e-6,
    max_floor_applied_fraction: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ""





    _require_columns(
        update_ledger,
        {"forecast_id", "forecast_origin", "target_time", "component", "observed_value", "observed_mask", "release_time"},
        label="ledger",
    )
    _require_columns(
        update_archive,
        {"forecast_id", "model_id", "particle_id", "forecast_origin", "target_time", "component", "pred_mean"},
        label="archive",
    )
    selected_model_ids = [str(x) for x in selected_model_ids]
    archive = update_archive.copy()
    archive["model_id"] = archive["model_id"].astype(str)
    unexpected = sorted(set(archive["model_id"]) - set(selected_model_ids))
    if unexpected:
        raise NativeProxyBlocker(
            f"native-proxy blocker: archive contains unselected model_id values {unexpected[:10]}",
            status="blocked_native_proxy_unselected_archive_rows",
        )
    if "pred_var" not in archive.columns:
        archive["pred_var"] = np.nan
    duplicate_archive = archive.duplicated(
        ["forecast_id", "model_id", "particle_id", "forecast_origin", "target_time", "component"],
        keep=False,
    )
    if duplicate_archive.any():
        sample = archive.loc[duplicate_archive, ["forecast_id", "model_id", "particle_id"]].head(10).to_dict("records")
        raise NativeProxyBlocker(
            f"native-proxy blocker: duplicate archive prediction rows for strict event/model/particle keys {sample}",
            status="blocked_native_proxy_duplicate_archive_rows",
        )

    ledger_meta = update_ledger[
        [
            "forecast_id",
            "forecast_origin",
            "target_time",
            "component",
            "observed_value",
            "observed_mask",
            "release_time",
        ]
    ].copy()
    duplicate_ledger = ledger_meta.duplicated(["forecast_id"], keep=False)
    if duplicate_ledger.any():
        sample = ledger_meta.loc[duplicate_ledger, "forecast_id"].astype(str).unique()[:10]
        raise NativeProxyBlocker(
            f"native-proxy blocker: duplicate ledger forecast_id values {','.join(sample)}",
            status="blocked_native_proxy_duplicate_ledger_forecast_id",
        )
    ledger_ids = set(ledger_meta["forecast_id"].astype(str))
    joined = archive.merge(ledger_meta, on="forecast_id", how="inner", suffixes=("", "_ledger"))
    if joined.empty:
        raise NativeProxyBlocker("native-proxy blocker: no archive rows matched update ledger", status="blocked_native_proxy_missing_archive_rows")
    matched_ids = set(joined["forecast_id"].astype(str))
    missing_ledger_ids = sorted(ledger_ids - matched_ids)
    if missing_ledger_ids:
        raise NativeProxyBlocker(
            f"native-proxy blocker: archive missing update ledger forecast_id values {missing_ledger_ids[:10]}",
            status="blocked_native_proxy_missing_archive_rows",
        )

    _validate_selected_coverage(joined, ledger_ids, selected_model_ids)
    _validate_event_metadata(joined)
    _validate_event_contract(joined)

    pred_mean = pd.to_numeric(joined["pred_mean"], errors="coerce")
    if pred_mean.isna().any() or not np.isfinite(pred_mean.to_numpy(dtype=float)).all():
        raise NativeProxyBlocker("native-proxy blocker: pred_mean contains nonfinite values", status="blocked_native_proxy_invalid_archive")
    var_used, floor_applied, validation = _validation_pred_var(joined["pred_var"], min_variance_floor=float(min_variance_floor))
    if validation["floor_applied_fraction"] > float(max_floor_applied_fraction):
        raise NativeProxyBlocker(
            "native-proxy blocker: blocked_native_proxy_variance_unavailable "
            f"floor_applied_fraction={validation['floor_applied_fraction']:.6f} "
            f"threshold={float(max_floor_applied_fraction):.6f}",
            status="blocked_native_proxy_variance_unavailable",
            validation=validation,
        )

    observed_mask = _bool_series(joined["observed_mask"])
    observed = pd.to_numeric(joined["observed_value"], errors="coerce")
    observed_finite = pd.Series(np.isfinite(observed.to_numpy(dtype=float)), index=observed.index)
    observed_nonfinite = observed.isna() | ~observed_finite
    observed_nonfinite_scored = observed_nonfinite & observed_mask
    if observed_nonfinite_scored.any():
        raise NativeProxyBlocker(
            "native-proxy blocker: observed_value contains nonfinite values for observed rows",
            status="blocked_native_proxy_invalid_ledger",
        )
    observed_for_score = observed.fillna(0.0)

    raw_log_score = -0.5 * log(2.0 * pi) - 0.5 * np.log(var_used.to_numpy(dtype=float)) - 0.5 * np.square(
        (observed_for_score.to_numpy(dtype=float) - pred_mean.to_numpy(dtype=float)) / np.sqrt(var_used.to_numpy(dtype=float))
    )
    joined["native_proxy_var_used"] = var_used.to_numpy(dtype=float)
    joined["native_proxy_variance_floor_applied"] = floor_applied.to_numpy(dtype=bool)
    joined["native_proxy_log_score"] = np.where(observed_mask.to_numpy(dtype=bool), raw_log_score, 0.0)
    joined["native_proxy_nll"] = -joined["native_proxy_log_score"].astype(float)
    joined["log_score"] = joined["native_proxy_log_score"].astype(float)
    joined["event_weight"] = 1.0
    joined["observed_mask"] = observed_mask.to_numpy(dtype=bool)
    joined["release_time"] = pd.to_datetime(joined["release_time"], errors="coerce")

    validation.update(
        {
            "scored_rows": int(len(joined)),
            "scored_forecast_ids": int(joined["forecast_id"].nunique()),
            "scored_model_ids": int(joined["model_id"].nunique()),
            "observed_value_nonfinite_count": int(observed_nonfinite.sum()),
            "observed_value_nonfinite_unobserved_count": int((observed_nonfinite & ~observed_mask).sum()),
            "observed_value_nonfinite_observed_count": int(observed_nonfinite_scored.sum()),
            "native_proxy_score_mode": "identity_gaussian_pred_mean_pred_var_floor",
            "native_proxy_not_comparable": True,
        }
    )
    return joined, validation
