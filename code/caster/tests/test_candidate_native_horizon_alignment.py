from __future__ import annotations

import numpy as np
import pandas as pd

from caster.bridge.likelihood import BridgeConfig, score_archive_rows
from caster.models.candidate_adapters import (
    SeasonalNaiveReferenceAdapter,
    SeriesForecastAdapter,
)
from caster.forecast.archive import validate_native_horizon_provenance
from scripts.build_selected_forecast_archive_impl import (
    NATIVE_HORIZON_PROVENANCE_COLUMNS,
    _attach_native_horizon_provenance,
    _baseline_forecast_to_archive,
    _validate_one_model_archive,
)


class RecordingAdapter(SeriesForecastAdapter):
    model_id = "recording"
    family = "test"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def forecast_one_for_event(
        self,
        state,
        event: pd.Series,
        values: np.ndarray,
        horizon: int,
        times: np.ndarray | None = None,
        *,
        recursive_step: int | None = None,
    ) -> tuple[float, float]:
        self.calls.append(
            {
                "horizon": int(horizon),
                "recursive_step": recursive_step,
                "target_time": pd.Timestamp(event["target_time"]),
                "last_time": pd.Timestamp(pd.to_datetime(times[-1])),
                "last_value": float(values[-1]),
            }
        )
        if recursive_step is None:
            return float(values[-1] + horizon), float(horizon)
        return float(values[-1] + 1.0), float(recursive_step)


def _weekly_panel(*, release_lag_steps: int, missing_last_released: bool = False) -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.Timestamp]:
    weeks = pd.date_range("2025-08-16", periods=10, freq="7D")
    origin = weeks[5]
    values = np.asarray([10, 20, 30, 40, 50, 999, 999, 999, 999, 999], dtype=float)
    if missing_last_released:
        values[4] = np.nan
    panel = pd.DataFrame(
        {
            "jurisdiction": "A",
            "week_end": weeks,
            "admissions": values,
            "__release_time__": weeks
            + pd.Timedelta(days=7 * int(release_lag_steps)),
        }
    )
    return panel, weeks, origin


def _ledger(origin: pd.Timestamp, horizons: tuple[int, ...], strategy: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "benchmark_b",
                "entity_id": "A",
                "forecast_origin": origin,
                "target_time": origin + pd.Timedelta(days=7 * horizon),
                "component": "admissions",
                "horizon": horizon,
                "forecast_id": f"{strategy}-{horizon}",
                "forecast_strategy": strategy,
                "observed_mask": True,
            }
            for horizon in horizons
        ]
    )


def test_direct_uses_native_horizon_after_release_gating_without_leakage() -> None:
    panel, weeks, origin = _weekly_panel(release_lag_steps=1)
    ledger = _ledger(origin, (1, 2), "direct")
    adapter = RecordingAdapter()

    archive = adapter.forecast_ledger(adapter.initialize(panel), ledger)
    by_horizon = archive.set_index("horizon")

    assert [call["horizon"] for call in adapter.calls] == [2, 3]
    assert [call["last_value"] for call in adapter.calls] == [50.0, 50.0]
    assert by_horizon["pred_mean"].to_dict() == {1: 52.0, 2: 53.0}
    assert by_horizon["pred_var"].to_dict() == {1: 2.0, 2: 3.0}
    assert by_horizon["native_horizon_steps"].to_dict() == {1: 2, 2: 3}
    assert set(pd.to_datetime(archive["last_released_target_time"])) == {weeks[4]}
    assert (
        pd.to_datetime(archive["forecasted_native_target_time"]).to_numpy()
        == pd.to_datetime(archive["target_time"]).to_numpy()
    ).all()


def test_missing_released_value_extends_native_horizon_but_not_cadence() -> None:
    panel, weeks, origin = _weekly_panel(
        release_lag_steps=1, missing_last_released=True
    )
    ledger = _ledger(origin, (1,), "direct")
    adapter = RecordingAdapter()

    archive = adapter.forecast_ledger(adapter.initialize(panel), ledger)

    assert adapter.calls[0]["last_time"] == weeks[3]
    assert adapter.calls[0]["horizon"] == 3
    assert int(archive.loc[0, "native_horizon_steps"]) == 3
    assert pd.Timestamp(archive.loc[0, "last_released_target_time"]) == weeks[3]
    assert pd.Timestamp(archive.loc[0, "forecasted_native_target_time"]) == weeks[6]


def test_recursive_runs_through_native_steps_and_keeps_only_ledger_targets() -> None:
    panel, weeks, origin = _weekly_panel(release_lag_steps=1)
    ledger = _ledger(origin, (1, 2, 4), "recursive_rollout")
    adapter = RecordingAdapter()

    archive = adapter.forecast_ledger(adapter.initialize(panel), ledger)
    by_horizon = archive.set_index("horizon")

    assert len(adapter.calls) == 5
    assert [call["horizon"] for call in adapter.calls] == [1, 1, 1, 1, 1]
    assert [call["recursive_step"] for call in adapter.calls] == [1, 2, 3, 4, 5]
    assert [call["target_time"] for call in adapter.calls] == list(weeks[5:10])
    assert by_horizon["pred_mean"].to_dict() == {1: 52.0, 2: 53.0, 4: 55.0}
    assert by_horizon["pred_var"].to_dict() == {1: 2.0, 2: 3.0, 4: 5.0}
    assert by_horizon["native_horizon_steps"].to_dict() == {1: 2, 2: 3, 4: 5}


def test_no_lag_direct_native_steps_equal_nominal_horizons() -> None:
    panel, _, origin = _weekly_panel(release_lag_steps=0)
    ledger = _ledger(origin, (1, 2), "direct")
    adapter = RecordingAdapter()

    archive = adapter.forecast_ledger(adapter.initialize(panel), ledger)

    assert [call["horizon"] for call in adapter.calls] == [1, 2]
    assert archive.set_index("horizon")["native_horizon_steps"].to_dict() == {
        1: 1,
        2: 2,
    }


def test_seasonal_naive_still_looks_up_actual_target_date() -> None:
    weeks = pd.date_range("2022-01-01", periods=14, freq="7D")
    panel = pd.DataFrame(
        {
            "jurisdiction": "A",
            "week_end": weeks,
            "admissions": list(range(1, 15)),
            "__release_time__": weeks + pd.Timedelta(days=7),
        }
    )
    origin = weeks[8]
    ledger = _ledger(origin, (1, 2, 4), "direct")
    adapter = SeasonalNaiveReferenceAdapter(
        daily_season_length=7, weekly_season_length=8
    )

    archive = adapter.forecast_ledger(adapter.initialize(panel), ledger)

    assert archive.set_index("horizon")["pred_mean"].to_dict() == {
        1: 2.0,
        2: 3.0,
        4: 5.0,
    }
    assert archive.set_index("horizon")["native_horizon_steps"].to_dict() == {
        1: 2,
        2: 3,
        4: 5,
    }


def test_final_archive_contract_is_complete_across_model_sources() -> None:
    panel, _, origin = _weekly_panel(release_lag_steps=1)
    ledger = _ledger(origin, (1, 2), "direct")
    base_rows: list[dict[str, object]] = []
    for model_id in (
        "seasonal_naive",
        "prophet",
        "lstm_style",
        "patchtst_patched",
        "timesfm_external",
    ):
        for _, event in ledger.iterrows():
            base_rows.append(
                {
                    "dataset": "benchmark_b",
                    "model_id": model_id,
                    "family": "test",
                    "particle_id": 0,
                    "entity_id": "A",
                    "forecast_origin": event["forecast_origin"],
                    "target_time": event["target_time"],
                    "component": "admissions",
                    "horizon": event["horizon"],
                    "forecast_id": event["forecast_id"],
                    "pred_mean": 1.0,
                    "pred_var": 1.0,
                    "generated_at": event["forecast_origin"],
                    "features_available_until": event["forecast_origin"],
                }
            )
    archive = _attach_native_horizon_provenance(
        pd.DataFrame(base_rows), ledger, panel
    )

    assert not archive[list(NATIVE_HORIZON_PROVENANCE_COLUMNS)].isna().any().any()
    for model_id, rows in archive.groupby("model_id"):
        assert _validate_one_model_archive(rows, ledger, str(model_id)).empty
        assert rows.set_index("horizon")["native_horizon_steps"].to_dict() == {
            1: 2,
            2: 3,
        }

    corrupted = archive[archive["model_id"].eq("prophet")].copy()
    corrupted.loc[corrupted.index[0], "forecasted_native_target_time"] = origin
    violations = _validate_one_model_archive(corrupted, ledger, "prophet")
    assert "forecasted_native_target_mismatch" in set(violations["violation"])


def test_reused_baseline_merge_preserves_native_horizon_fields(tmp_path) -> None:
    panel, weeks, origin = _weekly_panel(release_lag_steps=1)
    ledger = _ledger(origin, (1, 2), "direct")
    source = pd.DataFrame(
        {
            "forecast_id": ledger["forecast_id"],
            "pred_mean": [1.0, 2.0],
            "pred_var": [1.0, 1.0],
            "last_released_target_time": [weeks[4], weeks[4]],
            "native_horizon_steps": [2, 3],
            "forecasted_native_target_time": ledger["target_time"],
        }
    )
    source_path = tmp_path / "forecast.csv"
    source.to_csv(source_path, index=False)

    converted = _baseline_forecast_to_archive(
        model_id="patchtst_patched",
        family="neural",
        forecast_path=source_path,
        ledger=ledger,
    )

    assert converted.set_index("horizon")["native_horizon_steps"].to_dict() == {
        1: 2,
        2: 3,
    }
    checked = _attach_native_horizon_provenance(converted, ledger, panel)
    assert not checked[list(NATIVE_HORIZON_PROVENANCE_COLUMNS)].isna().any().any()


def test_release_lag_archive_requires_native_horizon_provenance() -> None:
    _, _, origin = _weekly_panel(release_lag_steps=1)
    ledger = _ledger(origin, (1,), "direct")
    ledger["release_lag_steps"] = 1
    archive = pd.DataFrame({"forecast_id": ledger["forecast_id"]})

    violations = validate_native_horizon_provenance(archive, ledger)

    assert set(violations["violation"]) == {"missing_native_horizon_columns"}


def test_native_horizon_provenance_rejects_relabelled_target() -> None:
    panel, _, origin = _weekly_panel(release_lag_steps=1)
    ledger = _ledger(origin, (1,), "direct")
    ledger["release_lag_steps"] = 1
    adapter = RecordingAdapter()
    archive = adapter.forecast_ledger(adapter.initialize(panel), ledger)
    archive.loc[0, "forecasted_native_target_time"] = origin

    violations = validate_native_horizon_provenance(archive, ledger)

    assert "forecasted_native_target_mismatch" in set(violations["violation"])


def test_score_archive_rows_validates_only_requested_ledger_subset() -> None:
    origin = pd.Timestamp("2025-09-20")
    target = origin + pd.Timedelta(days=7)
    ledger = pd.DataFrame(
        {
            "forecast_id": ["test-h1"],
            "observed_value": [2.0],
            "observed_mask": [True],
            "mode": ["direct"],
            "component": ["admissions"],
            "horizon": [1],
            "forecast_origin": [origin],
            "target_time": [target],
            "features_available_until": [origin],
            "release_lag_steps": [1],
        }
    )
    archive = pd.DataFrame(
        {
            "forecast_id": ["test-h1", "validation-h1"],
            "model_id": ["model", "model"],
            "particle_id": [0, 0],
            "pred_mean": [2.0, 3.0],
            "pred_var": [1.0, 1.0],
            "component": ["admissions", "admissions"],
            "horizon": [1, 1],
            "mode": ["direct", "direct"],
            "forecast_origin": [origin, origin - pd.Timedelta(days=7)],
            "target_time": [target, origin],
            "features_available_until": [origin, origin - pd.Timedelta(days=7)],
            "last_released_target_time": [
                origin - pd.Timedelta(days=7),
                origin - pd.Timedelta(days=14),
            ],
            "native_horizon_steps": [2, 2],
            "forecasted_native_target_time": [target, origin],
        }
    )

    scored = score_archive_rows(ledger, archive, BridgeConfig())

    assert scored["forecast_id"].tolist() == ["test-h1"]
