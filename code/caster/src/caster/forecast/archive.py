from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FORECAST_ARCHIVE_REQUIRED_COLUMNS = [
    "dataset", "model_id", "family", "particle_id", "entity_id",
    "forecast_origin", "target_time", "component", "horizon", "forecast_id",
    "pred_mean", "pred_var", "generated_at", "features_available_until",
]
FORECAST_ARCHIVE_CONTEXT_COLUMNS = [
    "protocol_version", "natural_event_id", "mode", "mode_kind", "forecast_strategy",
]
NATIVE_HORIZON_PROVENANCE_COLUMNS = [
    "last_released_target_time",
    "native_horizon_steps",
    "forecasted_native_target_time",
]
FORECAST_ARCHIVE_COLUMNS = [*FORECAST_ARCHIVE_REQUIRED_COLUMNS, *FORECAST_ARCHIVE_CONTEXT_COLUMNS]


def validate_native_horizon_provenance(
    archive: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Fail closed on release-lag forecasts whose native target is unauditable."""

    columns = set(archive.columns)
    present = columns.intersection(NATIVE_HORIZON_PROVENANCE_COLUMNS)
    lagged = False
    if "release_lag_steps" in ledger.columns:
        lag = pd.to_numeric(ledger["release_lag_steps"], errors="coerce")
        lagged = bool(lag.fillna(0).gt(0).any())
    if present and len(present) != len(NATIVE_HORIZON_PROVENANCE_COLUMNS):
        missing = sorted(set(NATIVE_HORIZON_PROVENANCE_COLUMNS) - present)
        return pd.DataFrame(
            [{"row": None, "violation": "missing_native_horizon_columns", "details": ",".join(missing)}]
        )
    if lagged and not present:
        return pd.DataFrame(
            [{
                "row": None,
                "violation": "missing_native_horizon_columns",
                "details": ",".join(NATIVE_HORIZON_PROVENANCE_COLUMNS),
            }]
        )
    if not present:
        return pd.DataFrame(columns=["row", "violation", "details"])

    required_ledger = {"forecast_id", "forecast_origin", "target_time", "horizon"}
    missing_ledger = sorted(required_ledger - set(ledger.columns))
    if missing_ledger:
        return pd.DataFrame(
            [{"row": None, "violation": "missing_native_horizon_ledger_columns", "details": ",".join(missing_ledger)}]
        )

    ledger_meta = ledger[list(required_ledger)].copy()
    ledger_meta["forecast_id"] = ledger_meta["forecast_id"].astype(str)
    ledger_meta = ledger_meta.drop_duplicates("forecast_id")
    work = archive[["forecast_id", *NATIVE_HORIZON_PROVENANCE_COLUMNS]].copy()
    work["forecast_id"] = work["forecast_id"].astype(str)
    work = work.merge(
        ledger_meta,
        on="forecast_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    violations: list[dict[str, object]] = []

    def record(name: str, mask: pd.Series) -> None:
        selected = mask.fillna(True).astype(bool)
        for idx in selected.index[selected.to_numpy()]:
            violations.append(
                {
                    "row": int(idx),
                    "violation": name,
                    "details": str(work.loc[idx, "forecast_id"]),
                }
            )

    record("native_horizon_forecast_id_not_in_ledger", work["_merge"].ne("both"))
    known = work[work["_merge"].eq("both")].copy()
    if known.empty:
        return pd.DataFrame(violations, columns=["row", "violation", "details"])

    origin = pd.to_datetime(known["forecast_origin"], errors="coerce", utc=True)
    target = pd.to_datetime(known["target_time"], errors="coerce", utc=True)
    last = pd.to_datetime(known["last_released_target_time"], errors="coerce", utc=True)
    forecasted = pd.to_datetime(known["forecasted_native_target_time"], errors="coerce", utc=True)
    native_raw = pd.to_numeric(known["native_horizon_steps"], errors="coerce")
    nominal_raw = pd.to_numeric(known["horizon"], errors="coerce")
    native_valid = native_raw.notna() & native_raw.gt(0) & np.isclose(native_raw, np.round(native_raw))
    nominal_valid = nominal_raw.notna() & nominal_raw.gt(0) & np.isclose(nominal_raw, np.round(nominal_raw))

    record("invalid_last_released_target_time", known["last_released_target_time"].isna() | last.isna())
    record("invalid_forecasted_native_target_time", known["forecasted_native_target_time"].isna() | forecasted.isna())
    record("invalid_native_horizon_steps", ~native_valid)
    record("last_released_target_after_origin", last.gt(origin))
    record("forecasted_native_target_mismatch", forecasted.ne(target))

    timing_valid = native_valid & nominal_valid & origin.notna() & target.notna() & last.notna()
    delta_nominal = (target - origin).dt.total_seconds()
    cadence_seconds = delta_nominal / nominal_raw
    timing_valid &= cadence_seconds.gt(0) & np.isfinite(cadence_seconds)
    expected = last + pd.to_timedelta(native_raw * cadence_seconds, unit="s")
    record("native_horizon_target_mismatch", timing_valid & expected.ne(target))
    record("invalid_native_horizon_timing", ~timing_valid)
    return pd.DataFrame(violations, columns=["row", "violation", "details"])


def attach_ledger_context(archive: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    ""






    out = archive.copy().reset_index(drop=True)
    out["forecast_id"] = out["forecast_id"].astype(str)
    ledger_norm = ledger.copy()
    ledger_norm["forecast_id"] = ledger_norm["forecast_id"].astype(str)
    available = [col for col in FORECAST_ARCHIVE_CONTEXT_COLUMNS if col in ledger_norm.columns]
    if available:
        context = ledger_norm[["forecast_id", *available]].drop_duplicates("forecast_id")
        if len(context) != ledger_norm["forecast_id"].nunique():
            raise ValueError("ledger context is not unique by forecast_id")
        for col in available:
            expected = out[["forecast_id"]].merge(context[["forecast_id", col]], on="forecast_id", how="left")[col]
            if col in out.columns:
                actual = out[col]
                populated = actual.notna() & actual.astype(str).ne("")
                mismatch = populated & actual.astype(str).ne(expected.astype(str))
                if mismatch.any():
                    ids = out.loc[mismatch, "forecast_id"].head(10).tolist()
                    raise ValueError(f"archive {col} conflicts with ledger for forecast_id={ids}")
            out[col] = expected.to_numpy()
    for col in FORECAST_ARCHIVE_CONTEXT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("")
    extras = [col for col in out.columns if col not in FORECAST_ARCHIVE_COLUMNS]
    return out[[*FORECAST_ARCHIVE_COLUMNS, *extras]]

def validate_forecast_archive(archive: pd.DataFrame, ledger: pd.DataFrame, *, require_all_observed: bool = True) -> pd.DataFrame:
    violations: list[dict[str, object]] = []
    missing = sorted(set(FORECAST_ARCHIVE_REQUIRED_COLUMNS) - set(archive.columns))
    if missing:
        return pd.DataFrame([{"row": None, "violation": "missing_columns", "details": ",".join(missing)}])
    if archive.empty:
        return pd.DataFrame([{"row": None, "violation": "empty_archive", "details": "archive has no rows"}])
    native_violations = validate_native_horizon_provenance(archive, ledger)
    if not native_violations.empty:
        violations.extend(native_violations.to_dict("records"))
    key_cols = ["forecast_id", "model_id", "particle_id"]
    dup = archive.duplicated(key_cols)
    if dup.any():
        ids = archive.loc[dup, key_cols].astype(str).agg("|".join, axis=1).unique()[:10]
        violations.append({"row": None, "violation": "duplicate_prediction", "details": ",".join(ids)})
    ledger_meta_cols = ["forecast_id", "forecast_origin", "target_time", "component", "observed_mask"]
    ledger_meta_cols.extend(col for col in FORECAST_ARCHIVE_CONTEXT_COLUMNS if col in ledger.columns)
    ledger_meta = ledger[ledger_meta_cols].copy()
    merged = archive.merge(ledger_meta, on="forecast_id", how="left", suffixes=("", "_ledger"), indicator=True)
    missing_ledger = merged[merged["_merge"] == "left_only"]
    if not missing_ledger.empty:
        ids = missing_ledger["forecast_id"].astype(str).unique()[:10]
        violations.append({"row": None, "violation": "forecast_id_not_in_ledger", "details": ",".join(ids)})
    known = merged[merged["_merge"] == "both"].copy()
    if not known.empty:
        for col in ["forecast_origin", "generated_at", "features_available_until", "target_time", "target_time_ledger"]:
            known[col] = pd.to_datetime(known[col])
        checks = {
            "generated_after_origin": known["generated_at"] > known["forecast_origin"],
            "features_after_origin": known["features_available_until"] > known["forecast_origin"],
            "target_mismatch": known["target_time"] != known["target_time_ledger"],
            "component_mismatch": known["component"] != known["component_ledger"],
        }
        for col in FORECAST_ARCHIVE_CONTEXT_COLUMNS:
            ledger_col = f"{col}_ledger"
            if col in known.columns and ledger_col in known.columns:
                populated = known[col].notna() & known[col].astype(str).ne("")
                checks[f"{col}_mismatch"] = populated & known[col].astype(str).ne(known[ledger_col].astype(str))
        for violation, mask in checks.items():
            for idx in known.index[mask]:
                violations.append({"row": int(idx), "violation": violation, "details": str(known.loc[idx, "forecast_id"])})
    if not np.isfinite(pd.to_numeric(archive["pred_mean"], errors="coerce")).all():
        violations.append({"row": None, "violation": "nonfinite_pred_mean", "details": "pred_mean must be finite"})
    pred_var = pd.to_numeric(archive["pred_var"], errors="coerce")
    if (pred_var < 0).any() or (not np.isfinite(pred_var).all()):
        violations.append({"row": None, "violation": "invalid_pred_var", "details": "pred_var must be finite and nonnegative"})
    if require_all_observed:
        observed_ids = set(ledger.loc[ledger["observed_mask"].astype(bool), "forecast_id"].astype(str))
        for model_id in sorted(archive["model_id"].astype(str).unique()):
            ids = set(archive.loc[archive["model_id"].astype(str) == model_id, "forecast_id"].astype(str))
            missing_ids = sorted(observed_ids - ids)
            if missing_ids:
                violations.append({"row": None, "violation": "missing_observed_forecasts", "details": f"{model_id}:{','.join(missing_ids[:10])}"})
    return pd.DataFrame(violations, columns=["row", "violation", "details"])

def write_forecast_archive(archive: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            archive.to_parquet(path, index=False)
            return path
        except Exception:
            path = path.with_suffix(".csv")
    archive.to_csv(path, index=False)
    return path
