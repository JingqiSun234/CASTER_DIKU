from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caster_baselines.external_forecasting import (
    run_external_forecaster_from_manifest,
)
from caster_baselines.forecast_strategy import RECURSIVE_ROLLOUT
from caster_baselines.foundation_forecasting import (
    FoundationPredictions,
    max_manifest_native_horizon,
    run_foundation_from_manifest,
)


ORIGIN = pd.Timestamp("2022-11-26")
CADENCE_DAYS = 7


def _write_release_case(
    tmp_path: Path,
    *,
    strategy: str,
    horizon_count: int,
    release_lag_steps: int = 1,
) -> Path:
    weeks = pd.date_range("2022-09-24", periods=15, freq="7D")
    panel = pd.DataFrame(
        {
            "jurisdiction": ["E1"] * len(weeks),
            "week_end": weeks,
            "covid_rate": np.arange(1.0, len(weeks) + 1.0),
        }
    )
    if release_lag_steps:
        panel["__release_time__target"] = weeks + pd.Timedelta(
            days=CADENCE_DAYS * release_lag_steps
        )
    panel_path = tmp_path / "data" / "panel.csv"
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(panel_path, index=False)

    ledger = pd.DataFrame(
        [
            {
                "dataset": "benchmark_b_toy",
                "entity_id": "E1",
                "forecast_id": f"event_{horizon}",
                "forecast_origin": ORIGIN.strftime("%Y-%m-%d"),
                "target_time": (
                    ORIGIN + pd.Timedelta(days=CADENCE_DAYS * horizon)
                ).strftime("%Y-%m-%d"),
                "component": "covid_rate",
                "horizon": horizon,
                "observed_value": 20.0 + horizon,
                "observed_mask": True,
                "split": "test",
                "mode": (
                    "rollout_4w"
                    if strategy == RECURSIVE_ROLLOUT
                    else "direct_1w2w"
                ),
                "forecast_strategy": strategy,
            }
            for horizon in range(1, horizon_count + 1)
        ]
    )
    ledger_path = tmp_path / "data" / "ledger.csv"
    ledger.to_csv(ledger_path, index=False)

    manifest = pd.DataFrame(
        [
            {
                "dataset_key": "benchmark_b_toy",
                "dataset": "benchmark_b_toy",
                "panel_path": "data/panel.csv",
                "ledger_path": "data/ledger.csv",
                "panel_format": "wide",
                "panel_entity_col": "jurisdiction",
                "panel_time_col": "week_end",
                "panel_component_col": "NA",
                "panel_value_col": "NA",
                "panel_target_cols": "covid_rate",
                "ledger_rows": len(ledger),
                "cadence_days": CADENCE_DAYS,
            }
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest_path


class RecordingPointPredictor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        model_name,
        history_times,
        values,
        max_horizon,
        target_dates_by_horizon,
        season_length,
    ):
        self.calls.append(
            {
                "max_horizon": int(max_horizon),
                "horizons": list(target_dates_by_horizon),
                "targets": [pd.Timestamp(x) for x in target_dates_by_horizon.values()],
                "last_history_time": pd.Timestamp(history_times[-1]),
            }
        )
        return {int(h): float(h) for h in target_dates_by_horizon}


class RecordingFoundationBackend:
    interval_source = "recording_native_steps"

    def __init__(self) -> None:
        self.calls: list[tuple[int, list[int]]] = []

    def predict(self, values, max_horizon, horizons, cadence_days):
        horizons = [int(h) for h in horizons]
        self.calls.append((int(max_horizon), horizons))
        means = {h: float(h) for h in horizons}
        return FoundationPredictions(
            mean=means,
            lower_50={h: means[h] - 0.5 for h in horizons},
            upper_50={h: means[h] + 0.5 for h in horizons},
            lower_90={h: means[h] - 1.0 for h in horizons},
            upper_90={h: means[h] + 1.0 for h in horizons},
            interval_source=self.interval_source,
        )


def _assert_release_lag_audit(forecast: pd.DataFrame, expected_steps: list[int]) -> None:
    assert forecast["last_released_target_time"].tolist() == [
        "2022-11-19"
    ] * len(expected_steps)
    assert forecast["native_horizon_steps"].astype(int).tolist() == expected_steps
    assert forecast["forecasted_native_target_time"].tolist() == forecast[
        "target_time"
    ].tolist()


def test_statsforecast_direct_requests_native_steps_two_and_three(tmp_path: Path) -> None:
    manifest = _write_release_case(
        tmp_path, strategy="direct", horizon_count=2, release_lag_steps=1
    )
    predictor = RecordingPointPredictor()

    out = run_external_forecaster_from_manifest(
        manifest_path=manifest,
        out_dir=tmp_path / "stats_direct",
        model_name="autoets",
        backend="statsforecast",
        predictor=predictor,
        min_train_rows=2,
        root=tmp_path / "baseline_root",
        caster_root=tmp_path,
    )

    assert predictor.calls == [
        {
            "max_horizon": 3,
            "horizons": [2, 3],
            "targets": [pd.Timestamp("2022-12-03"), pd.Timestamp("2022-12-10")],
            "last_history_time": pd.Timestamp("2022-11-19"),
        }
    ]
    forecast = pd.read_csv(out / "forecast.csv", keep_default_na=False)
    assert forecast["pred_mean"].tolist() == [2.0, 3.0]
    _assert_release_lag_audit(forecast, [2, 3])


def test_statsforecast_recursive_runs_through_native_step_five(tmp_path: Path) -> None:
    manifest = _write_release_case(
        tmp_path,
        strategy=RECURSIVE_ROLLOUT,
        horizon_count=4,
        release_lag_steps=1,
    )
    predictor = RecordingPointPredictor()

    out = run_external_forecaster_from_manifest(
        manifest_path=manifest,
        out_dir=tmp_path / "stats_recursive",
        model_name="autoarima",
        backend="statsforecast",
        predictor=predictor,
        min_train_rows=2,
        root=tmp_path / "baseline_root",
        caster_root=tmp_path,
    )

    assert len(predictor.calls) == 5
    assert [call["last_history_time"] for call in predictor.calls] == list(
        pd.date_range("2022-11-19", periods=5, freq="7D")
    )
    assert [call["targets"][0] for call in predictor.calls] == list(
        pd.date_range("2022-11-26", periods=5, freq="7D")
    )
    forecast = pd.read_csv(out / "forecast.csv", keep_default_na=False)
    _assert_release_lag_audit(forecast, [2, 3, 4, 5])


def test_prophet_direct_keeps_explicit_date_contract_with_native_audit(tmp_path: Path) -> None:
    manifest = _write_release_case(
        tmp_path, strategy="direct", horizon_count=2, release_lag_steps=1
    )
    predictor = RecordingPointPredictor()

    out = run_external_forecaster_from_manifest(
        manifest_path=manifest,
        out_dir=tmp_path / "prophet_direct",
        model_name="prophet",
        backend="prophet",
        predictor=predictor,
        min_train_rows=2,
        root=tmp_path / "baseline_root",
        caster_root=tmp_path,
    )

    assert predictor.calls[0]["max_horizon"] == 2
    assert predictor.calls[0]["horizons"] == [1, 2]
    forecast = pd.read_csv(out / "forecast.csv", keep_default_na=False)
    assert forecast["pred_mean"].tolist() == [1.0, 2.0]
    _assert_release_lag_audit(forecast, [2, 3])


def test_no_lag_statsforecast_retains_nominal_horizons(tmp_path: Path) -> None:
    manifest = _write_release_case(
        tmp_path, strategy="direct", horizon_count=2, release_lag_steps=0
    )
    predictor = RecordingPointPredictor()

    out = run_external_forecaster_from_manifest(
        manifest_path=manifest,
        out_dir=tmp_path / "stats_no_lag",
        model_name="autotheta",
        backend="statsforecast",
        predictor=predictor,
        min_train_rows=2,
        root=tmp_path / "baseline_root",
        caster_root=tmp_path,
    )

    assert predictor.calls[0]["max_horizon"] == 2
    assert predictor.calls[0]["horizons"] == [1, 2]
    forecast = pd.read_csv(out / "forecast.csv", keep_default_na=False)
    assert forecast["last_released_target_time"].tolist() == [
        "2022-11-26",
        "2022-11-26",
    ]
    assert forecast["native_horizon_steps"].astype(int).tolist() == [1, 2]


def test_native_horizon_uses_last_finite_released_target_not_fixed_lag(
    tmp_path: Path,
) -> None:
    manifest = _write_release_case(
        tmp_path, strategy="direct", horizon_count=2, release_lag_steps=1
    )
    panel_path = tmp_path / "data" / "panel.csv"
    panel = pd.read_csv(panel_path)
    panel.loc[panel["week_end"] == "2022-11-19", "covid_rate"] = np.nan
    panel.to_csv(panel_path, index=False)
    predictor = RecordingPointPredictor()

    out = run_external_forecaster_from_manifest(
        manifest_path=manifest,
        out_dir=tmp_path / "stats_missing_target",
        model_name="autoets",
        backend="statsforecast",
        predictor=predictor,
        min_train_rows=2,
        root=tmp_path / "baseline_root",
        caster_root=tmp_path,
    )

    assert predictor.calls[0]["last_history_time"] == pd.Timestamp("2022-11-12")
    assert predictor.calls[0]["max_horizon"] == 4
    assert predictor.calls[0]["horizons"] == [3, 4]
    forecast = pd.read_csv(out / "forecast.csv", keep_default_na=False)
    assert forecast["last_released_target_time"].tolist() == [
        "2022-11-12",
        "2022-11-12",
    ]
    assert forecast["native_horizon_steps"].astype(int).tolist() == [3, 4]


def test_off_cadence_target_fails_closed_before_forecast_write(tmp_path: Path) -> None:
    manifest = _write_release_case(
        tmp_path, strategy="direct", horizon_count=2, release_lag_steps=1
    )
    ledger_path = tmp_path / "data" / "ledger.csv"
    ledger = pd.read_csv(ledger_path)
    ledger.loc[0, "target_time"] = "2022-12-04"
    ledger.to_csv(ledger_path, index=False)

    with pytest.raises(RuntimeError, match="run blocked"):
        run_external_forecaster_from_manifest(
            manifest_path=manifest,
            out_dir=tmp_path / "stats_off_cadence",
            model_name="autoets",
            backend="statsforecast",
            predictor=RecordingPointPredictor(),
            min_train_rows=2,
            root=tmp_path / "baseline_root",
            caster_root=tmp_path,
        )

    assert not (tmp_path / "stats_off_cadence" / "forecast.csv").exists()
    blocker = pd.read_csv(tmp_path / "stats_off_cadence" / "blocker_report.csv")
    assert blocker["reason"].str.contains("native horizon resolution failed").any()


@pytest.mark.parametrize(
    ("strategy", "horizon_count", "expected_steps"),
    [
        ("direct", 2, [2, 3]),
        (RECURSIVE_ROLLOUT, 4, [2, 3, 4, 5]),
    ],
)
def test_foundation_maps_outputs_and_intervals_by_native_step(
    tmp_path: Path,
    strategy: str,
    horizon_count: int,
    expected_steps: list[int],
) -> None:
    manifest = _write_release_case(
        tmp_path,
        strategy=strategy,
        horizon_count=horizon_count,
        release_lag_steps=1,
    )
    backend = RecordingFoundationBackend()

    out = run_foundation_from_manifest(
        manifest_path=manifest,
        out_dir=tmp_path / f"foundation_{strategy}",
        model="chronos",
        checkpoint_id="toy/chronos",
        backend=backend,
        require_dependencies=False,
        root=tmp_path / "baseline_root",
        caster_root=tmp_path,
    )

    if strategy == "direct":
        assert backend.calls == [(3, [2, 3])]
    else:
        assert backend.calls == [(1, [1])] * 5
    forecast = pd.read_csv(out / "forecast.csv", keep_default_na=False)
    _assert_release_lag_audit(forecast, expected_steps)
    if strategy == "direct":
        assert forecast["pred_mean"].tolist() == [2.0, 3.0]
        assert forecast["pred_lower_90"].tolist() == [1.0, 2.0]


def test_foundation_model_capacity_uses_maximum_native_horizon(tmp_path: Path) -> None:
    manifest_path = _write_release_case(
        tmp_path,
        strategy=RECURSIVE_ROLLOUT,
        horizon_count=4,
        release_lag_steps=1,
    )
    manifest = pd.read_csv(manifest_path, keep_default_na=False)

    assert max_manifest_native_horizon(
        manifest, tmp_path / "baseline_root", tmp_path
    ) == 5
