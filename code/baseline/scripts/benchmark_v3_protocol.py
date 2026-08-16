from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "benchmark_protocol_v3_direct_rollout"

A_DIRECT_MODE = "direct_1d3d7d"
A_ROLLOUT_MODE = "rollout_7d"
A_DIRECT_HORIZONS = (1, 3, 7)
A_ROLLOUT_HORIZONS = tuple(range(1, 8))
A_MIN_HISTORY_POINTS = 21
A_TRAIN_END = pd.Timestamp("2020-04-09")
A_VAL_START = pd.Timestamp("2020-04-10")
A_VAL_END = pd.Timestamp("2020-04-23")
A_EMBARGO_START = pd.Timestamp("2020-04-24")
A_EMBARGO_END = pd.Timestamp("2020-04-30")
A_TEST_START = pd.Timestamp("2020-05-01")
A_TEST_END = pd.Timestamp("2020-05-05")

B_DIRECT_MODE = "direct_1w2w"
B_ROLLOUT_MODE = "rollout_4w"
B_DIRECT_HORIZONS = (1, 2)
B_ROLLOUT_HORIZONS = tuple(range(1, 5))
B_MIN_HISTORY_POINTS = 9                                              
B_FIRST_ORIGIN = pd.Timestamp("2022-11-26")
B_TRAIN_END = pd.Timestamp("2025-03-08")
B_VAL_START = pd.Timestamp("2025-03-15")
B_VAL_END = pd.Timestamp("2025-08-16")
B_EMBARGO_START = pd.Timestamp("2025-08-23")
B_EMBARGO_END = pd.Timestamp("2025-09-13")
B_TEST_START = pd.Timestamp("2025-09-20")
B_TEST_END = pd.Timestamp("2026-02-28")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: object, prefix: str) -> str:
    payload = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def natural_event_id(
    dataset: str,
    entity_id: str,
    component: str,
    forecast_origin: pd.Timestamp,
    target_time: pd.Timestamp,
    horizon: int,
    revision_version: str = "final",
) -> str:
    return stable_id(
        dataset,
        entity_id,
        component,
        pd.Timestamp(forecast_origin).strftime("%Y-%m-%d"),
        pd.Timestamp(target_time).strftime("%Y-%m-%d"),
        int(horizon),
        revision_version,
        prefix="nat",
    )


def forecast_id(event_id: str, mode: str) -> str:
    return stable_id(PROTOCOL_VERSION, event_id, mode, prefix="fcst")


def _split_a(origin: pd.Timestamp) -> str | None:
    if origin <= A_TRAIN_END:
        return "train"
    if A_VAL_START <= origin <= A_VAL_END:
        return "val"
    if A_EMBARGO_START <= origin <= A_EMBARGO_END:
        return "embargo"
    if A_TEST_START <= origin <= A_TEST_END:
        return "test"
    return None


def _split_b(origin: pd.Timestamp) -> str | None:
    if B_FIRST_ORIGIN <= origin <= B_TRAIN_END:
        return "train"
    if B_VAL_START <= origin <= B_VAL_END:
        return "val"
    if B_EMBARGO_START <= origin <= B_EMBARGO_END:
        return "embargo"
    if B_TEST_START <= origin <= B_TEST_END:
        return "test"
    return None


def _date_text(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _finite(value: object) -> bool:
    return bool(np.isfinite(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]))


def _validate_ledger(ledger: pd.DataFrame, *, expected_horizons: set[int]) -> None:
    required = {
        "protocol_version",
        "natural_event_id",
        "forecast_id",
        "event_id",
        "dataset",
        "entity_id",
        "component",
        "forecast_strategy",
        "mode",
        "split",
        "forecast_origin",
        "target_time",
        "release_time",
        "features_available_until",
        "horizon",
        "observed_value",
        "observed_mask",
    }
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"v3 ledger missing columns: {missing}")
    if ledger.empty:
        raise ValueError("v3 ledger is empty")
    if not ledger["forecast_id"].is_unique:
        raise ValueError("v3 forecast_id must be unique")
    if not ledger["event_id"].is_unique:
        raise ValueError("v3 event_id must be unique")
    times = ledger[["forecast_origin", "target_time", "release_time", "features_available_until"]].apply(
        pd.to_datetime
    )
    valid = (
        (times["features_available_until"] <= times["forecast_origin"])
        & (times["forecast_origin"] < times["target_time"])
        & (times["target_time"] <= times["release_time"])
    )
    if not valid.all():
        raise ValueError(f"v3 ledger date/no-leakage violations: {int((~valid).sum())}")
    split_counts = ledger.groupby("natural_event_id")["split"].nunique()
    if int((split_counts > 1).sum()) != 0:
        raise ValueError("natural_event_id crosses forecast-origin splits")
    natural_meta = [
        "entity_id",
        "component",
        "forecast_origin",
        "target_time",
        "release_time",
        "horizon",
        "observed_mask",
        "observed_value",
    ]
    if any(int(ledger.groupby("natural_event_id")[col].nunique(dropna=False).max()) > 1 for col in natural_meta):
        raise ValueError("natural_event_id metadata is inconsistent across modes")
    actual_horizons = set(pd.to_numeric(ledger["horizon"], errors="raise").astype(int))
    if actual_horizons != expected_horizons:
        raise ValueError(f"unexpected horizon coverage expected={sorted(expected_horizons)} actual={sorted(actual_horizons)}")
    val = ledger[ledger["split"].astype(str) == "val"]
    embargo = ledger[ledger["split"].astype(str) == "embargo"]
    test = ledger[ledger["split"].astype(str) == "test"]
    if val.empty or embargo.empty or test.empty:
        raise ValueError("v3 ledger requires non-empty validation, embargo, and test splits")
    if pd.to_datetime(val["release_time"]).max() >= pd.to_datetime(test["forecast_origin"]).min():
        raise ValueError("validation release embargo is not strict")
    declared_calibration = ledger["calibration_eligible"].astype(bool)
    if not declared_calibration.equals(ledger["split"].astype(str).eq("val")):
        raise ValueError("calibration_eligible must be true exactly for validation rows")


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.astype(str).value_counts().sort_index().items()}


def _manifest(
    *,
    dataset: str,
    source_panel: Path,
    panel_name: str,
    ledger: pd.DataFrame,
    modes: dict[str, Iterable[int]],
    cadence_days: int,
    min_history_points: int,
    split_contract: dict[str, object],
    release_time_policy: str,
) -> dict[str, object]:
    natural = ledger.drop_duplicates("natural_event_id")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "scope": "curated_full_v3_direct_rollout",
        "status": "input_ledger_package_not_model_run",
        "source_panel": str(source_panel),
        "source_panel_sha256": sha256_file(source_panel),
        "panel": panel_name,
        "event_ledger": "event_ledger.csv",
        "ledger_rows": int(len(ledger)),
        "natural_event_rows": int(len(natural)),
        "split_counts": _counts(ledger["split"]),
        "natural_event_split_counts": _counts(natural["split"]),
        "mode_counts": _counts(ledger["mode"]),
        "strategy_counts": _counts(ledger["forecast_strategy"]),
        "horizon_counts": _counts(ledger["horizon"]),
        "component_counts": _counts(ledger["component"]),
        "modes": {mode: [int(value) for value in horizons] for mode, horizons in modes.items()},
        "cadence_days": int(cadence_days),
        "min_history_points": int(min_history_points),
        "split_contract": split_contract,
        "release_time_policy": release_time_policy,
        "embargo_policy": {
            "forecast_origins_generated": True,
            "selection_eligible": False,
            "calibration_eligible": False,
            "metric_eligible": False,
            "posterior_update_eligible_after_release": True,
        },
        "evidence_unit": "ledger_task_row",
        "same_outcome_multi_strategy_scores_accumulated": True,
        "posterior_natural_event_normalization": False,
        "observed_mask_policy": "retain_missing_targets_with_observed_mask_false",
        "notes": [
            "Direct uses a native multi-horizon forecast; recursive_rollout feeds each predicted mean into the next one-step context.",
            "All modes share one global forecast-origin split; embargo origins generate forecasts but are excluded from selection, calibration, and reported metrics.",
            "An embargo target may update the online posterior only after its declared release_time; same-date release is available under the benchmark's date-level abstraction.",
            "natural_event_id excludes mode and split; forecast_id is mode-specific.",
        ],
    }


def build_benchmark_a_v3(panel_path: Path, out_dir: Path) -> Path:
    panel = pd.read_csv(panel_path)
    required = {"dataset", "country", "country_code", "entity_id", "date", "component", "observed_value"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Benchmark A panel missing columns: {missing}")
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    panel["entity_id"] = panel["entity_id"].astype(str)
    panel["component"] = panel["component"].astype(str)
    lookup = {
        (str(row.entity_id), str(row.component), pd.Timestamp(row.date)): row.observed_value
        for row in panel.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    modes = {
        A_DIRECT_MODE: ("direct", A_DIRECT_HORIZONS),
        A_ROLLOUT_MODE: ("recursive_rollout", A_ROLLOUT_HORIZONS),
    }
    for (country, country_code, entity_id, component), group in panel.groupby(
        ["country", "country_code", "entity_id", "component"], sort=True
    ):
        group = group.sort_values("date")
        dates = [pd.Timestamp(value) for value in group["date"].drop_duplicates().tolist()]
        date_set = set(dates)
        for origin_idx, origin in enumerate(dates):
            if origin_idx + 1 < A_MIN_HISTORY_POINTS:
                continue
            split = _split_a(origin)
            if split is None or origin + pd.Timedelta(days=max(A_ROLLOUT_HORIZONS)) not in date_set:
                continue
            for mode, (strategy, horizons) in modes.items():
                for horizon in horizons:
                    target = origin + pd.Timedelta(days=int(horizon))
                    value = lookup.get((str(entity_id), str(component), target), np.nan)
                    observed = _finite(value)
                    natural_id = natural_event_id(
                        "benchmark_a",
                        str(entity_id),
                        str(component),
                        origin,
                        target,
                        int(horizon),
                    )
                    fid = forecast_id(natural_id, mode)
                    rows.append(
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "event_id": stable_id(PROTOCOL_VERSION, fid, prefix="evt"),
                            "natural_event_id": natural_id,
                            "forecast_id": fid,
                            "dataset": "benchmark_a",
                            "split": split,
                            "calibration_eligible": split == "val",
                            "mode": mode,
                            "mode_kind": "direct" if strategy == "direct" else "rollout",
                            "forecast_strategy": strategy,
                            "country": str(country),
                            "country_code": str(country_code),
                            "entity_id": str(entity_id),
                            "component": str(component),
                            "horizon": int(horizon),
                            "forecast_origin": _date_text(origin),
                            "target_time": _date_text(target),
                            "release_time": _date_text(target),
                            "features_available_until": _date_text(origin),
                            "observed_mask": observed,
                            "observed_value": float(value) if observed else np.nan,
                            "revision_version": "final",
                        }
                    )
    ledger = pd.DataFrame(rows).sort_values(
        ["country", "entity_id", "forecast_origin", "mode", "horizon"]
    ).reset_index(drop=True)
    _validate_ledger(ledger, expected_horizons=set(A_ROLLOUT_HORIZONS))
    expected = {"train": 49_590, "val": 48_860, "embargo": 24_430, "test": 17_450}
    actual = {str(k): int(v) for k, v in ledger["split"].value_counts().to_dict().items()}
    if actual != expected:
        raise ValueError(f"Benchmark A v3 row-count contract failed expected={expected} actual={actual}")
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_out = panel.copy()
    panel_out["date"] = panel_out["date"].dt.strftime("%Y-%m-%d")
    panel_out.sort_values(["country", "entity_id", "date", "component"]).to_csv(out_dir / "daily_panel.csv", index=False)
    ledger.to_csv(out_dir / "event_ledger.csv", index=False)
    manifest = _manifest(
        dataset="benchmark_a",
        source_panel=panel_path,
        panel_name="daily_panel.csv",
        ledger=ledger,
        modes={A_DIRECT_MODE: A_DIRECT_HORIZONS, A_ROLLOUT_MODE: A_ROLLOUT_HORIZONS},
        cadence_days=1,
        min_history_points=A_MIN_HISTORY_POINTS,
        split_contract={
            "train_end": _date_text(A_TRAIN_END),
            "val_start": _date_text(A_VAL_START),
            "val_end": _date_text(A_VAL_END),
            "embargo_start": _date_text(A_EMBARGO_START),
            "embargo_end": _date_text(A_EMBARGO_END),
            "test_start": _date_text(A_TEST_START),
            "test_end": _date_text(A_TEST_END),
        },
        release_time_policy="target_time_immediate_daily_benchmark_label",
    )
    manifest.update(
        {
            "panel_rows": int(len(panel_out)),
            "countries": sorted(panel_out["country"].astype(str).unique().tolist()),
            "entity_count": int(panel_out["entity_id"].nunique()),
        }
    )
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_dir


def build_benchmark_b_v3(panel_path: Path, out_dir: Path) -> Path:
    panel = pd.read_csv(panel_path)
    targets = ["covid_adm_per100k", "flu_adm_per100k"]
    required = {"jurisdiction", "week_end", *targets}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Benchmark B panel missing columns: {missing}")
    panel = panel.copy()
    panel["week_end"] = pd.to_datetime(panel["week_end"], errors="raise")
    panel["jurisdiction"] = panel["jurisdiction"].astype(str)
    lookup = {
        (str(row.jurisdiction), pd.Timestamp(row.week_end), target): row._asdict()[target]
        for row in panel.itertuples(index=False)
        for target in targets
    }
    rows: list[dict[str, object]] = []
    modes = {
        B_DIRECT_MODE: ("direct", B_DIRECT_HORIZONS),
        B_ROLLOUT_MODE: ("recursive_rollout", B_ROLLOUT_HORIZONS),
    }
    for jurisdiction, group in panel.groupby("jurisdiction", sort=True):
        group = group.sort_values("week_end")
        dates = [pd.Timestamp(value) for value in group["week_end"].drop_duplicates().tolist()]
        date_set = set(dates)
        for origin_idx, origin in enumerate(dates):
            if origin_idx + 1 < B_MIN_HISTORY_POINTS:
                continue
            split = _split_b(origin)
            if split is None or origin + pd.Timedelta(weeks=max(B_ROLLOUT_HORIZONS)) not in date_set:
                continue
            for component in targets:
                for mode, (strategy, horizons) in modes.items():
                    for horizon in horizons:
                        target = origin + pd.Timedelta(weeks=int(horizon))
                        value = lookup.get((str(jurisdiction), target, component), np.nan)
                        observed = _finite(value)
                        natural_id = natural_event_id(
                            "benchmark_b",
                            str(jurisdiction),
                            component,
                            origin,
                            target,
                            int(horizon),
                        )
                        fid = forecast_id(natural_id, mode)
                        rows.append(
                            {
                                "protocol_version": PROTOCOL_VERSION,
                                "event_id": stable_id(PROTOCOL_VERSION, fid, prefix="evt"),
                                "natural_event_id": natural_id,
                                "forecast_id": fid,
                                "dataset": "benchmark_b",
                                "entity_id": str(jurisdiction),
                                "jurisdiction": str(jurisdiction),
                                "split": split,
                                "calibration_eligible": split == "val",
                                "mode": mode,
                                "mode_kind": "direct" if strategy == "direct" else "rollout",
                                "forecast_strategy": strategy,
                                "forecast_origin": _date_text(origin),
                                "target_time": _date_text(target),
                                "release_time": _date_text(target),
                                "component": component,
                                "horizon": int(horizon),
                                "observed_value": float(value) if observed else np.nan,
                                "observed_mask": observed,
                                "revision_version": "final",
                                "features_available_until": _date_text(origin),
                            }
                        )
    ledger = pd.DataFrame(rows).sort_values(
        ["entity_id", "component", "forecast_origin", "mode", "horizon"]
    ).reset_index(drop=True)
    _validate_ledger(ledger, expected_horizons=set(B_ROLLOUT_HORIZONS))
    expected = {"train": 73_440, "val": 14_076, "embargo": 2_448, "test": 14_688}
    actual = {str(k): int(v) for k, v in ledger["split"].value_counts().to_dict().items()}
    if actual != expected:
        raise ValueError(f"Benchmark B v3 row-count contract failed expected={expected} actual={actual}")
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_out = panel.copy()
    panel_out["week_end"] = panel_out["week_end"].dt.strftime("%Y-%m-%d")
    panel_out.sort_values(["jurisdiction", "week_end"]).to_csv(out_dir / "weekly_panel.csv", index=False)
    ledger.to_csv(out_dir / "event_ledger.csv", index=False)
    manifest = _manifest(
        dataset="benchmark_b",
        source_panel=panel_path,
        panel_name="weekly_panel.csv",
        ledger=ledger,
        modes={B_DIRECT_MODE: B_DIRECT_HORIZONS, B_ROLLOUT_MODE: B_ROLLOUT_HORIZONS},
        cadence_days=7,
        min_history_points=B_MIN_HISTORY_POINTS,
        split_contract={
            "first_origin": _date_text(B_FIRST_ORIGIN),
            "train_end": _date_text(B_TRAIN_END),
            "val_start": _date_text(B_VAL_START),
            "val_end": _date_text(B_VAL_END),
            "embargo_start": _date_text(B_EMBARGO_START),
            "embargo_end": _date_text(B_EMBARGO_END),
            "test_start": _date_text(B_TEST_START),
            "test_end": _date_text(B_TEST_END),
        },
        release_time_policy="target_time_immediate_finalized_benchmark_abstraction_not_fixed_vintage",
    )
    manifest.update(
        {
            "panel_rows": int(len(panel_out)),
            "jurisdiction_count": int(panel_out["jurisdiction"].nunique()),
            "target_cols": targets,
            "finalized_release_limitation": (
                "fixed finalized-vintage release timestamps are unavailable; release_time=target_time is a declared benchmark abstraction."
            ),
        }
    )
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_dir
