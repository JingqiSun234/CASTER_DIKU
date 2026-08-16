from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


PROVENANCE_COLUMNS = {
    "forecast_status",
    "forecast_fallback_used",
    "forecast_failure_reason",
    "forecast_fallback_method",
    "proxy_fallback_used",
    "unsafe_native_proxy_executed",
}
STRUCTURAL_UNAVAILABLE_REASON = "structural_history_unavailable"
POSTERIOR_AVAILABILITY_POLICY = (
    "sleeping_model_native_only_preserve_absolute_mass"
)


def boolish(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin(
        {"true", "1", "t", "yes", "y"}
    )


def forecast_unavailable_mask(
    archive: pd.DataFrame, *, require_provenance: bool = False
) -> pd.Series:
    ""
    if "forecast_fallback_used" not in archive.columns:
        if require_provenance:
            raise ValueError(
                "forecast archive lacks row-level forecast_fallback_used provenance"
            )
        return pd.Series(False, index=archive.index, dtype=bool)
    unavailable = boolish(archive["forecast_fallback_used"])
    for column in ("proxy_fallback_used", "unsafe_native_proxy_executed"):
        if column in archive.columns:
            unavailable |= boolish(archive[column])
    if "forecast_status" in archive.columns:
        unavailable |= archive["forecast_status"].fillna("").astype(str).str.lower().str.contains(
            "fallback|last_value|unavailable", regex=True
        )
    if "forecast_fallback_method" in archive.columns:
        unavailable |= archive["forecast_fallback_method"].fillna("").astype(str).str.strip().ne("")
    return unavailable


def native_forecast_rows(
    archive: pd.DataFrame, *, require_provenance: bool = False
) -> pd.DataFrame:
    unavailable = forecast_unavailable_mask(
        archive, require_provenance=require_provenance
    )
    return archive.loc[~unavailable].copy()


def validate_sleeping_model_archive(archive: pd.DataFrame) -> pd.DataFrame:
    ""









    rows: list[dict[str, object]] = []
    missing = sorted(PROVENANCE_COLUMNS - set(archive.columns))
    if missing:
        rows.append(
            {
                "violation": "missing_availability_provenance",
                "details": ",".join(missing),
            }
        )
        return pd.DataFrame(rows)

    for column in ("proxy_fallback_used", "unsafe_native_proxy_executed"):
        mask = boolish(archive[column])
        if mask.any():
            models = sorted(archive.loc[mask, "model_id"].astype(str).unique())
            rows.append(
                {
                    "violation": column,
                    "details": ",".join(models[:20]),
                }
            )

    unavailable = forecast_unavailable_mask(archive, require_provenance=True)
    reasons = archive["forecast_failure_reason"].fillna("").astype(str).str.lower()
    explicitly_structural = reasons.str.startswith(STRUCTURAL_UNAVAILABLE_REASON)
    explicitly_unavailable = boolish(archive["forecast_fallback_used"])
    invalid_reason = unavailable & ~(
        explicitly_structural & explicitly_unavailable
    )
    if invalid_reason.any():
        counts = (
            archive.loc[invalid_reason, "model_id"]
            .astype(str)
            .value_counts()
            .sort_index()
            .to_dict()
        )
        rows.append(
            {
                "violation": "non_structural_fallback_cannot_be_masked",
                "details": str(counts),
            }
        )

    if {"forecast_id", "model_id"}.issubset(archive.columns):
        event_state = pd.DataFrame(
            {
                "forecast_id": archive["forecast_id"].astype(str),
                "model_id": archive["model_id"].astype(str),
                "unavailable": unavailable,
            }
        )
        mixed = (
            event_state.groupby(["forecast_id", "model_id"], sort=False)[
                "unavailable"
            ]
            .nunique()
            .gt(1)
        )
        if mixed.any():
            examples = [str(key) for key in mixed[mixed].index[:20]]
            rows.append(
                {
                    "violation": "mixed_native_and_unavailable_rows_for_event",
                    "details": ",".join(examples),
                }
            )

    origins = pd.to_datetime(archive["forecast_origin"], errors="coerce")
    if origins.isna().any():
        rows.append(
            {
                "violation": "invalid_forecast_origin_for_availability",
                "details": int(origins.isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def evidence_availability_by_model(
    scored_native_rows: pd.DataFrame,
    batch_ledger: pd.DataFrame,
    model_ids: Iterable[str],
    *,
    structural_unavailable_rows: pd.DataFrame | None = None,
) -> dict[str, bool]:
    ""









    observed = boolish(batch_ledger["observed_mask"])
    expected = set(batch_ledger.loc[observed, "forecast_id"].astype(str))
    if not expected:
        return {str(model_id): False for model_id in model_ids}
    scored = scored_native_rows.copy()
    if "observed_mask" in scored.columns:
        scored = scored.loc[boolish(scored["observed_mask"])].copy()
    required = {"model_id", "forecast_id"}
    missing = sorted(required - set(scored.columns))
    if missing and not scored.empty:
        raise ValueError(f"scored native rows missing columns {missing}")
    availability: dict[str, bool] = {}
    for model_id_raw in model_ids:
        model_id = str(model_id_raw)
        if scored.empty:
            covered: set[str] = set()
        else:
            covered = set(
                scored.loc[
                    scored["model_id"].astype(str).eq(model_id), "forecast_id"
                ].astype(str)
            )
        extra = covered - expected
        if extra:
            raise ValueError(
                f"model {model_id} scored unexpected forecast_ids: {sorted(extra)[:10]}"
            )
        if covered and covered != expected:
            missing_ids = expected - covered
            proven_structural: set[str] = set()
            if structural_unavailable_rows is not None:
                required_structural = {
                    "forecast_id",
                    "model_id",
                    *PROVENANCE_COLUMNS,
                }
                missing_columns = sorted(
                    required_structural - set(structural_unavailable_rows.columns)
                )
                if missing_columns:
                    raise ValueError(
                        "structural unavailable rows missing provenance columns "
                        f"{missing_columns}"
                    )
                structural = structural_unavailable_rows.loc[
                    structural_unavailable_rows["model_id"]
                    .astype(str)
                    .eq(model_id)
                ].copy()
                structural_mask = forecast_unavailable_mask(
                    structural, require_provenance=True
                )
                structural_reasons = (
                    structural["forecast_failure_reason"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )
                structural_mask &= boolish(structural["forecast_fallback_used"])
                structural_mask &= structural_reasons.str.startswith(
                    STRUCTURAL_UNAVAILABLE_REASON
                )
                structural_mask &= ~boolish(structural["proxy_fallback_used"])
                structural_mask &= ~boolish(
                    structural["unsafe_native_proxy_executed"]
                )
                proven_structural = set(
                    structural.loc[structural_mask, "forecast_id"].astype(str)
                )
            if missing_ids.issubset(proven_structural):
                                                                             
                                                                            
                                                                      
                availability[model_id] = False
                continue
            missing_sorted = sorted(missing_ids)
            unproved = sorted(missing_ids - proven_structural)
            raise ValueError(
                "sleeping-model evidence must cover all or none of each release batch; "
                f"model={model_id} covered={len(covered)}/{len(expected)} "
                f"missing={missing_sorted[:10]} unproved_structural={unproved[:10]}"
            )
        availability[model_id] = covered == expected
    return availability


def availability_validation_metadata(evidence_log: pd.DataFrame) -> dict[str, object]:
    ""
    metadata: dict[str, object] = {
        "posterior_availability_policy": POSTERIOR_AVAILABILITY_POLICY,
        "availability_provenance_required": True,
        "unavailable_model_mass_policy": "preserve_exact_absolute_posterior_mass",
        "active_model_mass_policy": "redistribute_only_previous_active_mass",
        "readout_availability_policy": "native_rows_only_then_renormalize",
        "runtime_failure_masking_allowed": False,
                                                                            
                                                                        
                                                
        "structural_prefix_sleeping_allowed": True,
        "structural_intermittent_sleeping_allowed": True,
        "structural_sleeping_policy": (
            "explicit_structural_history_unavailable_at_any_release_batch"
        ),
        "structural_placeholder_used_for_evidence": False,
        "structural_placeholder_used_for_readout": False,
        "release_batch_availability_policy": "all_or_none_per_model",
        "partial_batch_evidence_allowed": False,
        "structural_partial_batch_policy": (
            "sleep_entire_model_batch_and_ignore_native_subset"
        ),
    }
    if evidence_log.empty or "evidence_available" not in evidence_log.columns:
        return {
            **metadata,
            "sleeping_model_evidence_rows": 0,
            "release_batches_with_sleeping_models": 0,
            "models_ever_sleeping": [],
            "first_native_evidence_release_by_model": {},
        }
    rows = evidence_log.copy()
    available = boolish(rows["evidence_available"])
    rows["__available__"] = available
    release_times = pd.to_datetime(rows.get("release_time"), errors="coerce")
    rows["__release_time__"] = release_times
    sleeping = rows.loc[~available].copy()
    first_native: dict[str, str] = {}
    for model_id, group in rows.loc[available].groupby("model_id"):
        first = group["__release_time__"].dropna().min()
        first_native[str(model_id)] = (
            "" if pd.isna(first) else pd.Timestamp(first).isoformat()
        )
    sleeping_batches = 0
    if "__release_time__" in sleeping.columns:
        sleeping_batches = int(sleeping["__release_time__"].dropna().nunique())
    return {
        **metadata,
        "sleeping_model_evidence_rows": int((~available).sum()),
        "release_batches_with_sleeping_models": sleeping_batches,
        "models_ever_sleeping": sorted(
            sleeping.get("model_id", pd.Series(dtype=str)).astype(str).unique()
        ),
        "first_native_evidence_release_by_model": first_native,
    }
