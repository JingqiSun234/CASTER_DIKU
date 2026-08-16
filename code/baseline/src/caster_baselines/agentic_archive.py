from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .agentic_skills import DatasetContext
from .ledger_runner import Z50, Z90, format_date, parse_bool, write_blocker_report


ARCHIVE_FORECAST_SOURCE = "immutable_forecast_archive"
ARCHIVE_TIMING_MODE = "archive_backed"
alternate_TIMING_MODE = "alternate_refit_reforecast"
ARCHIVE_TIMING_SEMANTICS = "agent_selection_control_plus_immutable_forecast_archive_readout"


class ForecastArchiveLookupError(RuntimeError):
    def __init__(self, blockers: list[dict[str, object]]) -> None:
        self.blockers = blockers
        super().__init__(f"immutable forecast archive missing {len(blockers)} required rows")


@dataclass(frozen=True)
class ForecastArchiveIndex:
    path: Path
    rows: pd.DataFrame
    by_key: dict[tuple[str, str], tuple[float, float]]
    rows_by_model: dict[str, pd.DataFrame]
    forecast_ids_by_model: dict[str, set[str]]


def load_forecast_archive(path: str | Path | None) -> ForecastArchiveIndex:
    if path is None or str(path).strip() == "":
        raise ValueError("--forecast-archive is required when --archive-mode required")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, keep_default_na=False)
    required = {"model_id", "forecast_id", "pred_mean", "pred_var"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"forecast archive missing required columns {sorted(missing)}: {p}")
    work = df.copy()
    work["model_id"] = work["model_id"].astype(str)
    work["forecast_id"] = work["forecast_id"].astype(str)
    duplicates = work.duplicated(["model_id", "forecast_id"], keep=False)
    if bool(duplicates.any()):
        dup = work.loc[duplicates, ["model_id", "forecast_id"]].head(10).to_dict(orient="records")
        raise ValueError(f"forecast archive has duplicate (model_id, forecast_id) keys: {dup}")
    for col in ["pred_mean", "pred_var"]:
        vals = pd.to_numeric(work[col], errors="coerce")
        if vals.isna().any() or not np.isfinite(vals.astype(float)).all():
            raise ValueError(f"forecast archive column {col} contains non-finite values: {p}")
        work[col] = vals.astype(float)
    by_key = {
        (str(model_id), str(forecast_id)): (float(pred_mean), float(pred_var))
        for model_id, forecast_id, pred_mean, pred_var in work[
            ["model_id", "forecast_id", "pred_mean", "pred_var"]
        ].itertuples(index=False, name=None)
    }
    rows_by_model: dict[str, pd.DataFrame] = {}
    forecast_ids_by_model: dict[str, set[str]] = {}
    for model_id, group in work.groupby("model_id", sort=False):
        model_key = str(model_id)
        model_rows = group[["forecast_id", "pred_mean", "pred_var"]].copy()
        rows_by_model[model_key] = model_rows.rename(columns={"forecast_id": "_forecast_id"})
        forecast_ids_by_model[model_key] = set(model_rows["forecast_id"].astype(str))
    return ForecastArchiveIndex(
        path=p,
        rows=work,
        by_key=by_key,
        rows_by_model=rows_by_model,
        forecast_ids_by_model=forecast_ids_by_model,
    )


def resolve_timing_mode(*, archive_mode: str, timing_mode: str | None) -> str:
    inferred = ARCHIVE_TIMING_MODE if archive_mode == "required" else alternate_TIMING_MODE
    if timing_mode in (None, "", "auto"):
        return inferred
    if timing_mode == "alternate_refit":
        return alternate_TIMING_MODE
    if timing_mode == ARCHIVE_TIMING_MODE:
        return ARCHIVE_TIMING_MODE
    raise ValueError(f"unsupported timing_mode={timing_mode!r}")


def _forecast_id(event: pd.Series, ledger_idx: int) -> str:
    fid = str(event.get("forecast_id", "")).strip()
    if not fid:
        raise ForecastArchiveLookupError([{
            "dataset_key": str(event.get("dataset_key", "")),
            "ledger_row_index": int(ledger_idx),
            "entity_id": str(event.get("entity_id", event.get("jurisdiction", ""))),
            "component": str(event.get("component", "")),
            "forecast_origin": str(event.get("forecast_origin", "")),
            "reason": "ledger row missing forecast_id required for immutable forecast archive lookup",
        }])
    return fid


def missing_archive_blockers(
    *,
    ctx: DatasetContext,
    ledger_subset: pd.DataFrame,
    selected_model_id: str,
    archive_index: ForecastArchiveIndex,
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if "forecast_id" not in ledger_subset.columns:
        return [
            {
                "dataset_key": ctx.dataset_key,
                "ledger_row_index": int(idx),
                "entity_id": str(event.get(ctx.ledger_entity_col, "")),
                "component": str(event.get("component", "")),
                "forecast_origin": str(event.get("forecast_origin", "")),
                "reason": "ledger row missing forecast_id required for immutable forecast archive lookup",
            }
            for idx, event in ledger_subset.iterrows()
        ]
    allowed = archive_index.forecast_ids_by_model.get(str(selected_model_id), set())
    fids = ledger_subset["forecast_id"].astype(str)
    missing_mask = fids.str.len().eq(0) | ~fids.isin(allowed)
    for ledger_idx, event in ledger_subset.loc[missing_mask].iterrows():
        fid = str(event.get("forecast_id", ""))
        if not fid:
            blockers.append({
                "dataset_key": ctx.dataset_key,
                "ledger_row_index": int(ledger_idx),
                "entity_id": str(event.get(ctx.ledger_entity_col, "")),
                "component": str(event.get("component", "")),
                "forecast_origin": str(event.get("forecast_origin", "")),
                "reason": "ledger row missing forecast_id required for immutable forecast archive lookup",
            })
            continue
        blockers.append({
            "dataset_key": ctx.dataset_key,
            "ledger_row_index": int(ledger_idx),
            "entity_id": str(event.get(ctx.ledger_entity_col, "")),
            "component": str(event.get("component", "")),
            "forecast_origin": str(event.get("forecast_origin", "")),
            "reason": (
                "immutable forecast archive lookup miss for "
                f"model_id={selected_model_id} forecast_id={fid}"
            ),
        })
    return blockers


def validate_archive_coverage(
    *,
    ctx: DatasetContext,
    ledger_subset: pd.DataFrame,
    selected_model_id: str,
    archive_index: ForecastArchiveIndex,
    out_dir: str | Path | None = None,
) -> None:
    blockers = missing_archive_blockers(
        ctx=ctx,
        ledger_subset=ledger_subset,
        selected_model_id=selected_model_id,
        archive_index=archive_index,
    )
    if blockers:
        if out_dir is not None:
            write_blocker_report(Path(out_dir), blockers)
        raise ForecastArchiveLookupError(blockers)


def make_forecast_row_from_archive(
    *,
    ctx: DatasetContext,
    event: pd.Series,
    ledger_idx: int,
    selected: pd.Series,
    method: str,
    archive_index: ForecastArchiveIndex,
) -> dict[str, object]:
    fid = _forecast_id(event, int(ledger_idx))
    selected_model_id = str(selected["model_id"])
    key = (selected_model_id, fid)
    if key not in archive_index.by_key:
        raise ForecastArchiveLookupError([{
            "dataset_key": ctx.dataset_key,
            "ledger_row_index": int(ledger_idx),
            "entity_id": str(event.get(ctx.ledger_entity_col, "")),
            "component": str(event.get("component", "")),
            "forecast_origin": str(event.get("forecast_origin", "")),
            "reason": (
                "immutable forecast archive lookup miss for "
                f"model_id={selected_model_id} forecast_id={fid}"
            ),
        }])
    pred, var_raw = archive_index.by_key[key]
    entity_id = str(event[ctx.ledger_entity_col])
    component = str(event["component"])
    origin = pd.to_datetime(event["forecast_origin"], errors="coerce")
    target = pd.to_datetime(event["target_time"], errors="coerce")
    if pd.isna(origin):
        raise ValueError(f"forecast_origin parse failed at row {ledger_idx}")
    horizon = int(pd.to_numeric(event["horizon"], errors="raise"))
    y_true = pd.to_numeric(pd.Series([event.get("observed_value")]), errors="coerce").iloc[0]
    if parse_bool(event.get("observed_mask", True)) and not np.isfinite(y_true):
        raise ValueError(f"observed_mask true but observed_value missing at row {ledger_idx}")
    var = max(float(var_raw), 1e-6)
    sigma = math.sqrt(var)
    row: dict[str, object] = {
        "dataset_key": ctx.dataset_key,
        "dataset": str(event["dataset"]) if "dataset" in ctx.ledger.columns else ctx.dataset,
        "method": method,
        "entity_id": entity_id,
        "forecast_origin": format_date(origin),
        "target_time": format_date(target),
        "component": component,
        "horizon": horizon,
        "y_true": float(y_true) if np.isfinite(y_true) else np.nan,
        "pred_mean": pred,
        "pred_var": var,
        "pred_lower_50": float(pred - Z50 * sigma),
        "pred_upper_50": float(pred + Z50 * sigma),
        "pred_lower_90": float(pred - Z90 * sigma),
        "pred_upper_90": float(pred + Z90 * sigma),
        "split": str(event["split"]) if "split" in ctx.ledger.columns else "NA",
        "selected_model_id": selected_model_id,
        "selected_family": str(selected.get("family", "")),
        "restart_group_id": f"{ctx.dataset_key}:{format_date(origin)}",
        "forecast_source": ARCHIVE_FORECAST_SOURCE,
        "archive_model_id": selected_model_id,
        "archive_lookup_status": "hit",
    }
    for col in ctx.context_cols:
        row[col] = event[col]
    return row


def _format_date_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").fillna("")


def make_forecast_rows_from_archive(
    *,
    ctx: DatasetContext,
    ledger_subset: pd.DataFrame,
    selected: pd.Series,
    method: str,
    archive_index: ForecastArchiveIndex,
) -> pd.DataFrame:
    ""





    if ledger_subset.empty:
        return pd.DataFrame()
    if "forecast_id" not in ledger_subset.columns:
        blockers = [
            {
                "dataset_key": ctx.dataset_key,
                "ledger_row_index": int(idx),
                "entity_id": str(row.get(ctx.ledger_entity_col, "")),
                "component": str(row.get("component", "")),
                "forecast_origin": str(row.get("forecast_origin", "")),
                "reason": "ledger row missing forecast_id required for immutable forecast archive lookup",
            }
            for idx, row in ledger_subset.iterrows()
        ]
        raise ForecastArchiveLookupError(blockers)

    selected_model_id = str(selected["model_id"])
    work = ledger_subset.copy()
    work["_ledger_idx"] = work.index
    work["_forecast_id"] = work["forecast_id"].astype(str)
    missing_fid = work["_forecast_id"].astype(str).str.len().eq(0)
    if bool(missing_fid.any()):
        blockers = [
            {
                "dataset_key": ctx.dataset_key,
                "ledger_row_index": int(row["_ledger_idx"]),
                "entity_id": str(row.get(ctx.ledger_entity_col, "")),
                "component": str(row.get("component", "")),
                "forecast_origin": str(row.get("forecast_origin", "")),
                "reason": "ledger row missing forecast_id required for immutable forecast archive lookup",
            }
            for _, row in work.loc[missing_fid].iterrows()
        ]
        raise ForecastArchiveLookupError(blockers)

    archive_rows = archive_index.rows_by_model.get(
        selected_model_id,
        pd.DataFrame(columns=["_forecast_id", "pred_mean", "pred_var"]),
    )
    merged = work.merge(archive_rows, on="_forecast_id", how="left", sort=False, validate="many_to_one")
    miss = merged["pred_mean"].isna() | merged["pred_var"].isna()
    if bool(miss.any()):
        blockers = [
            {
                "dataset_key": ctx.dataset_key,
                "ledger_row_index": int(row["_ledger_idx"]),
                "entity_id": str(row.get(ctx.ledger_entity_col, "")),
                "component": str(row.get("component", "")),
                "forecast_origin": str(row.get("forecast_origin", "")),
                "reason": (
                    "immutable forecast archive lookup miss for "
                    f"model_id={selected_model_id} forecast_id={row.get('_forecast_id', '')}"
                ),
            }
            for _, row in merged.loc[miss].head(1000).iterrows()
        ]
        raise ForecastArchiveLookupError(blockers)

    origin_text = _format_date_series(merged["forecast_origin"])
    target_text = _format_date_series(merged["target_time"])
    if origin_text.eq("").any():
        bad_idx = int(merged.loc[origin_text.eq(""), "_ledger_idx"].iloc[0])
        raise ValueError(f"forecast_origin parse failed at row {bad_idx}")

    y_true = pd.to_numeric(merged.get("observed_value"), errors="coerce")
    if "observed_mask" in merged.columns:
        observed_mask = merged["observed_mask"].map(parse_bool)
    else:
        observed_mask = pd.Series(True, index=merged.index)
    finite_y = np.isfinite(y_true.to_numpy(dtype=float, na_value=np.nan))
    bad_observed = observed_mask.to_numpy(dtype=bool) & ~finite_y
    if bool(bad_observed.any()):
        bad_idx = int(merged.loc[bad_observed, "_ledger_idx"].iloc[0])
        raise ValueError(f"observed_mask true but observed_value missing at row {bad_idx}")

    pred = pd.to_numeric(merged["pred_mean"], errors="raise").astype(float)
    var = pd.to_numeric(merged["pred_var"], errors="raise").astype(float).clip(lower=1e-6)
    sigma = np.sqrt(var.to_numpy(dtype=float))
    horizon = pd.to_numeric(merged["horizon"], errors="raise").astype(int)
    out = pd.DataFrame(
        {
            "dataset_key": ctx.dataset_key,
            "dataset": merged["dataset"].astype(str) if "dataset" in ctx.ledger.columns else ctx.dataset,
            "method": method,
            "entity_id": merged[ctx.ledger_entity_col].astype(str),
            "forecast_origin": origin_text,
            "target_time": target_text,
            "component": merged["component"].astype(str),
            "horizon": horizon,
            "y_true": np.where(finite_y, y_true.to_numpy(dtype=float, na_value=np.nan), np.nan),
            "pred_mean": pred,
            "pred_var": var,
            "pred_lower_50": pred.to_numpy(dtype=float) - Z50 * sigma,
            "pred_upper_50": pred.to_numpy(dtype=float) + Z50 * sigma,
            "pred_lower_90": pred.to_numpy(dtype=float) - Z90 * sigma,
            "pred_upper_90": pred.to_numpy(dtype=float) + Z90 * sigma,
            "split": merged["split"].astype(str) if "split" in ctx.ledger.columns else "NA",
            "selected_model_id": selected_model_id,
            "selected_family": str(selected.get("family", "")),
            "restart_group_id": [f"{ctx.dataset_key}:{x}" for x in origin_text],
            "forecast_source": ARCHIVE_FORECAST_SOURCE,
            "archive_model_id": selected_model_id,
            "archive_lookup_status": "hit",
        }
    )
    for col in ctx.context_cols:
        out[col] = merged[col].to_numpy()
    return out


def archive_timing_payload(
    *,
    start_time: float,
    archive_lookup_seconds: float,
    agent_control_seconds: float,
    selection_seconds: float,
    charge_selection: bool,
    forecast_rows: int,
    expected_rows: int,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    algorithm_update_sec = (
        float(archive_lookup_seconds)
        + float(agent_control_seconds)
        + (float(selection_seconds) if charge_selection else 0.0)
    )
    payload: dict[str, object] = {
        "timing_mode": ARCHIVE_TIMING_MODE,
        "timing_semantics": ARCHIVE_TIMING_SEMANTICS,
        "model_compute_sec": 0.0,
        "train_seconds": 0.0,
        "forecast_compute_seconds": 0.0,
        "forecast_seconds": 0.0,
        "archive_lookup_seconds": round(float(archive_lookup_seconds), 6),
        "agent_control_seconds": round(float(agent_control_seconds), 6),
        "selection_seconds": round(float(selection_seconds), 6),
        "selection_charged_to_update": bool(charge_selection),
        "algorithm_update_sec": round(float(algorithm_update_sec), 6),
        "update_sec": round(float(algorithm_update_sec), 6),
        "total_process_sec": round(time.time() - start_time, 6),
        "artifact_reuse": False,
        "artifact_reuse_proxy": False,
        "formal_timing_valid": True,
        "forecast_source": ARCHIVE_FORECAST_SOURCE,
        "forecast_rows": int(forecast_rows),
        "expected_rows": int(expected_rows),
    }
    if extra:
        payload.update(extra)
    payload["total_seconds"] = payload["total_process_sec"]
    return payload


def alternate_timing_payload(base: dict[str, object]) -> dict[str, object]:
    payload = dict(base)
    payload.setdefault("timing_mode", alternate_TIMING_MODE)
    payload.setdefault("formal_timing_valid", False)
    payload.setdefault("excluded_from_formal_runtime_reason", "model_compute_charged")
    return payload
