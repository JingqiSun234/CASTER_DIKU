""











from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import hashlib

import numpy as np
import pandas as pd


LEDGER_COLUMNS = [
    "dataset",
    "entity_id",
    "forecast_origin",
    "target_time",
    "release_time",
    "component",
    "horizon",
    "observed_value",
    "observed_mask",
    "revision_version",
    "forecast_id",
    "features_available_until",
    "split",
]


@dataclass(frozen=True)
class SplitCutoffs:
    ""

    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp

    @classmethod
    def from_strings(cls, train_end: str, val_end: str, test_start: str) -> "SplitCutoffs":
        return cls(
            train_end=pd.Timestamp(train_end),
            val_end=pd.Timestamp(val_end),
            test_start=pd.Timestamp(test_start),
        )

    def assign(self, origin: pd.Timestamp) -> str:
        if origin <= self.train_end:
            return "train"
        if origin <= self.val_end:
            return "val"
        if origin >= self.test_start:
            return "test"
        return "gap"


def _as_week_offset(freq: str, steps: int) -> pd.DateOffset:
    ""






    if not str(freq).upper().startswith("W"):
        raise ValueError(f"Only weekly frequencies are supported in this slice; got {freq!r}.")
    return pd.DateOffset(days=7 * int(steps))


def stable_forecast_id(
    *,
    dataset: str,
    entity_id: str,
    component: str,
    forecast_origin: pd.Timestamp,
    horizon: int,
    revision_version: str,
) -> str:
    ""

    raw = "|".join(
        [
            dataset,
            entity_id,
            component,
            pd.Timestamp(forecast_origin).strftime("%Y-%m-%d"),
            f"h{int(horizon)}",
            revision_version,
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"fcst_{digest}"


def make_entity_id(row: pd.Series, entity_cols: Sequence[str]) -> str:
    return "|".join(str(row[col]) for col in entity_cols)


def build_event_ledger_from_wide_panel(
    panel: pd.DataFrame,
    *,
    dataset: str,
    entity_cols: Sequence[str],
    time_col: str,
    target_cols: Sequence[str],
    horizons: Sequence[int],
    frequency: str = "W-SAT",
    release_lag_steps: int = 0,
    revision_version: str = "final",
    split_cutoffs: SplitCutoffs | None = None,
    anchors: Iterable[pd.Timestamp] | None = None,
    drop_missing_targets: bool = False,
    allow_partial_last_horizon: bool = True,
) -> pd.DataFrame:
    ""



























    required = set(entity_cols) | {time_col} | set(target_cols)
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")

    df = panel.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    for col in entity_cols:
        df[col] = df[col].astype(str)
    df["entity_id"] = df.apply(lambda r: make_entity_id(r, entity_cols), axis=1)

    value_lookup: dict[tuple[str, pd.Timestamp, str], float | None] = {}
    observed_lookup: dict[tuple[str, pd.Timestamp, str], bool] = {}
    for _, row in df.iterrows():
        entity = row["entity_id"]
        when = pd.Timestamp(row[time_col])
        for comp in target_cols:
            value = row[comp]
            is_obs = bool(pd.notna(value))
            value_lookup[(entity, when, comp)] = None if not is_obs else float(value)
            observed_lookup[(entity, when, comp)] = is_obs

    if anchors is None:
        anchor_times = sorted(df[time_col].dropna().unique())
    else:
        anchor_times = sorted(pd.to_datetime(list(anchors)))
    anchor_times = [pd.Timestamp(x) for x in anchor_times]

    entities = sorted(df["entity_id"].unique())
    horizon_offsets = {int(h): _as_week_offset(frequency, int(h)) for h in horizons}
    release_offset = _as_week_offset(frequency, int(release_lag_steps)) if release_lag_steps else pd.DateOffset(days=0)

    rows = []
    for origin in anchor_times:
        origin = pd.Timestamp(origin)
        split = split_cutoffs.assign(origin) if split_cutoffs is not None else "unsplit"
        for entity in entities:
            for comp in target_cols:
                for horizon in sorted(int(h) for h in horizons):
                    target_time = origin + horizon_offsets[horizon]
                    key = (entity, pd.Timestamp(target_time), comp)
                    if key not in value_lookup:
                                                                                   
                                                                                  
                        continue
                    observed = observed_lookup[key]
                    if drop_missing_targets and not observed:
                        continue
                    release_time = pd.Timestamp(target_time) + release_offset
                    fid = stable_forecast_id(
                        dataset=dataset,
                        entity_id=entity,
                        component=comp,
                        forecast_origin=origin,
                        horizon=horizon,
                        revision_version=revision_version,
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "entity_id": entity,
                            "forecast_origin": origin,
                            "target_time": pd.Timestamp(target_time),
                            "release_time": release_time,
                            "component": comp,
                            "horizon": horizon,
                            "observed_value": value_lookup[key],
                            "observed_mask": observed,
                            "revision_version": revision_version,
                            "forecast_id": fid,
                            "features_available_until": origin,
                            "split": split,
                        }
                    )

    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    if not ledger.empty:
        ledger = ledger.sort_values(
            ["forecast_origin", "entity_id", "component", "horizon"]
        ).reset_index(drop=True)
    return ledger


def validate_event_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    ""

    missing = sorted(set(LEDGER_COLUMNS) - set(ledger.columns))
    violations: list[dict[str, object]] = []
    if missing:
        return pd.DataFrame([{"row": None, "violation": "missing_columns", "details": ",".join(missing)}])

    if ledger["forecast_id"].duplicated().any():
        dupes = ledger.loc[ledger["forecast_id"].duplicated(), "forecast_id"].unique()[:10]
        violations.append({"row": None, "violation": "duplicate_forecast_id", "details": ",".join(dupes)})

    times = ledger[["forecast_origin", "target_time", "release_time", "features_available_until"]].apply(pd.to_datetime)
    bad_order = ~(times["forecast_origin"] < times["target_time"])
    bad_release = ~(times["target_time"] <= times["release_time"])
    bad_features = ~(times["features_available_until"] <= times["forecast_origin"])
    for idx in ledger.index[bad_order]:
        violations.append({"row": int(idx), "violation": "origin_not_before_target", "details": str(ledger.loc[idx, "forecast_id"])})
    for idx in ledger.index[bad_release]:
        violations.append({"row": int(idx), "violation": "target_after_release", "details": str(ledger.loc[idx, "forecast_id"])})
    for idx in ledger.index[bad_features]:
        violations.append({"row": int(idx), "violation": "features_after_origin", "details": str(ledger.loc[idx, "forecast_id"])})

    if not ledger["observed_mask"].isin([True, False, np.bool_(True), np.bool_(False)]).all():
        violations.append({"row": None, "violation": "observed_mask_not_boolean", "details": "observed_mask must be boolean"})

    return pd.DataFrame(violations, columns=["row", "violation", "details"])


def write_ledger(ledger: pd.DataFrame, output_path: str | Path) -> Path:
    ""







    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            ledger.to_parquet(path, index=False)
            return path
        except Exception:
            csv_path = path.with_suffix(".csv")
            ledger.to_csv(csv_path, index=False)
            return csv_path
    ledger.to_csv(path, index=False)
    return path
