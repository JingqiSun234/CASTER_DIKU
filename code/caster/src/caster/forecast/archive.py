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
FORECAST_ARCHIVE_COLUMNS = [*FORECAST_ARCHIVE_REQUIRED_COLUMNS, *FORECAST_ARCHIVE_CONTEXT_COLUMNS]


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
