from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from caster_baselines.foundation_forecasting import (
    CHRONOS_INTERVAL_CONSTRUCTION_RULE,
    CHRONOS_INTERVAL_SOURCE,
    NORMAL_CENTRAL_80_Z,
    NORMAL_CENTRAL_90_Z,
    ChronosBoltBackend,
    TimesFMBackend,
    _prediction_from_quantiles,
    run_foundation_from_manifest,
)


def test_chronos_expands_native_central_80_to_gaussian_proxy_central_90() -> None:
    sigma = 2.0
    q10 = 10.0 - NORMAL_CENTRAL_80_Z * sigma
    q90 = 10.0 + NORMAL_CENTRAL_80_Z * sigma
    quantiles = np.asarray([[[q10, 8.0, 10.0, 12.0, q90]]])

    pred = _prediction_from_quantiles(
        quantiles=quantiles,
        mean=np.asarray([[11.0]]),
        horizons=[1],
        quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        gaussian_proxy_90_from_central_80=True,
        interval_source=CHRONOS_INTERVAL_SOURCE,
    )

    assert pred.mean[1] == 11.0
    assert pred.lower_50[1] == 8.0
    assert pred.upper_50[1] == 12.0
    assert np.isclose(pred.lower_90[1], 11.0 - NORMAL_CENTRAL_90_Z * sigma)
    assert np.isclose(pred.upper_90[1], 11.0 + NORMAL_CENTRAL_90_Z * sigma)
    assert pred.interval_source == CHRONOS_INTERVAL_SOURCE
    assert ChronosBoltBackend.interval_source == CHRONOS_INTERVAL_SOURCE
    assert ChronosBoltBackend.interval_construction_rule == CHRONOS_INTERVAL_CONSTRUCTION_RULE


def test_chronos_uses_q50_when_point_forecast_is_unavailable() -> None:
    sigma = 1.5
    q10 = 20.0 - NORMAL_CENTRAL_80_Z * sigma
    q90 = 20.0 + NORMAL_CENTRAL_80_Z * sigma
    quantiles = np.asarray([[[q10, 19.0, 20.0, 21.0, q90]]])

    pred = _prediction_from_quantiles(
        quantiles=quantiles,
        mean=None,
        horizons=[1],
        quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        gaussian_proxy_90_from_central_80=True,
        interval_source=CHRONOS_INTERVAL_SOURCE,
    )

    assert pred.mean[1] == 20.0
    assert np.isclose(pred.lower_90[1], 20.0 - NORMAL_CENTRAL_90_Z * sigma)
    assert np.isclose(pred.upper_90[1], 20.0 + NORMAL_CENTRAL_90_Z * sigma)


def test_default_quantile_path_remains_native_for_non_chronos_callers() -> None:
    quantiles = np.asarray([[[1.0, 2.0, 3.0, 4.0, 5.0]]])

    pred = _prediction_from_quantiles(
        quantiles=quantiles,
        mean=np.asarray([[3.5]]),
        horizons=[1],
        quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
    )

    assert pred.lower_50[1] == 2.0
    assert pred.upper_50[1] == 4.0
    assert pred.lower_90[1] == 1.0
    assert pred.upper_90[1] == 5.0
    assert pred.interval_source == "model_quantiles"


def test_timesfm_decile_tuple_keeps_existing_residual_sigma_fallback() -> None:
    class FakeTimesFMModel:
        def forecast(self, *args, **kwargs):
            point = np.asarray([[5.0]])
            quantiles = np.zeros((1, 1, 10), dtype=float)
            quantiles[0, 0, 1:10] = np.arange(1.0, 10.0)
            return point, quantiles

    backend = object.__new__(TimesFMBackend)
    backend.model = FakeTimesFMModel()
    backend.context_len = 512

    pred = backend.predict(np.asarray([1.0, 2.0, 3.0]), max_horizon=1, horizons=[1], cadence_days=1)

    assert pred.mean[1] == 5.0
    assert pred.interval_source == "residual_sigma"


def test_chronos_interval_rule_is_written_to_run_manifest(tmp_path: Path) -> None:
    panel_path = tmp_path / "data" / "panel.csv"
    ledger_path = tmp_path / "data" / "ledger.csv"
    panel_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"entity_id": "E1", "date": "2020-01-01", "component": "cases", "observed_value": 1.0},
            {"entity_id": "E1", "date": "2020-01-02", "component": "cases", "observed_value": 2.0},
            {"entity_id": "E1", "date": "2020-01-03", "component": "cases", "observed_value": 3.0},
        ]
    ).to_csv(panel_path, index=False)
    pd.DataFrame(
        [
            {
                "dataset": "toy",
                "entity_id": "E1",
                "forecast_origin": "2020-01-02",
                "target_time": "2020-01-03",
                "component": "cases",
                "horizon": 1,
                "observed_value": 3.0,
                "observed_mask": True,
                "split": "test",
                "mode": "direct_1d",
            }
        ]
    ).to_csv(ledger_path, index=False)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "dataset_key": "toy",
                "dataset": "toy",
                "panel_path": "data/panel.csv",
                "ledger_path": "data/ledger.csv",
                "panel_format": "long",
                "panel_entity_col": "entity_id",
                "panel_time_col": "date",
                "panel_component_col": "component",
                "panel_value_col": "observed_value",
                "panel_target_cols": "cases",
                "ledger_rows": 1,
                "cadence_days": 1,
            }
        ]
    ).to_csv(manifest_path, index=False)

    class FakeChronosBackend:
        interval_source = CHRONOS_INTERVAL_SOURCE
        interval_construction_rule = CHRONOS_INTERVAL_CONSTRUCTION_RULE

        def predict(self, values, max_horizon, horizons, cadence_days):
            quantiles = np.asarray([[[1.0, 1.5, 2.0, 2.5, 3.0]]])
            return _prediction_from_quantiles(
                quantiles,
                np.asarray([[2.0]]),
                horizons,
                [0.1, 0.25, 0.5, 0.75, 0.9],
                gaussian_proxy_90_from_central_80=True,
                interval_source=self.interval_source,
            )

    out_dir = run_foundation_from_manifest(
        manifest_path=manifest_path,
        out_dir=tmp_path / "run",
        model="chronos",
        checkpoint_id="toy/chronos",
        backend=FakeChronosBackend(),
        require_dependencies=False,
        root=tmp_path / "baseline_root",
        caster_root=tmp_path,
    )

    run_manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    forecast = pd.read_csv(out_dir / "forecast.csv")
    assert run_manifest["interval_source"] == CHRONOS_INTERVAL_SOURCE
    assert run_manifest["interval_construction_rule"] == CHRONOS_INTERVAL_CONSTRUCTION_RULE
    assert "asof_input_rule" in run_manifest
    assert forecast.loc[0, "interval_source"] == CHRONOS_INTERVAL_SOURCE
    assert forecast.loc[0, "generated_at"] == "2020-01-02"
    assert forecast.loc[0, "features_available_until"] == "2020-01-02"
