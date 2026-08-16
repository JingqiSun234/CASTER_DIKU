from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from caster_baselines import external_forecasting
from caster_baselines.agentic_skills import (
    forecast_recipe,
    is_stateful,
    make_forecast_row,
    pre_fit_predictions,
)
from caster_baselines.ledger_runner import SeriesHistory


def _candidate(model_id: str, recipe: str, **params: float) -> pd.Series:
    return pd.Series(
        {
            "model_id": model_id,
            "recipe": recipe,
            "hyperparams_json": json.dumps(params, sort_keys=True),
        }
    )


def test_fixed_recurrences_execute_locally_instead_of_neuralforecast() -> None:
    values = np.asarray([2.0, 4.0, 3.0, 8.0])
    rnn = _candidate("rnn_simple", "fixed_rnn", alpha=0.55, gain=1.0)
    gru = _candidate("gru_style", "fixed_gru")

    assert not is_stateful(rnn)
    assert not is_stateful(gru)
    for candidate in (rnn, gru):
        mean, variance = forecast_recipe(candidate, values, horizon=3)
        assert np.isfinite(mean) and mean >= 0.0
        assert np.isfinite(variance) and variance >= 1.0


def test_unsupported_nonarchive_lstm_fails_before_execution() -> None:
    candidate = _candidate("lstm_style", "neural_lstm")
    assert is_stateful(candidate)
    with pytest.raises(RuntimeError, match="no supported non-archive executor"):
        pre_fit_predictions(
            candidate,
            SimpleNamespace(ledger_entity_col="entity_id"),
            pd.DataFrame([{"entity_id": "E1"}]),
        )


def test_missing_prefit_prediction_never_silently_uses_last_value() -> None:
    times = np.asarray(["2023-01-01", "2023-01-08"], dtype="datetime64[ns]")
    history = SeriesHistory(
        times=times,
        releases=times,
        values=np.asarray([1.0, 9.0]),
    )
    ctx = SimpleNamespace(
        dataset_key="unit_test",
        dataset="unit_test",
        ledger_entity_col="entity_id",
        history_index={("E1", "target"): history},
        covariate_index=None,
        season_length=8,
        manifest_row=pd.Series({"cadence_days": 7}),
    )
    event = pd.Series(
        {
            "entity_id": "E1",
            "component": "target",
            "forecast_origin": "2023-01-08",
            "target_time": "2023-01-15",
            "horizon": 1,
            "mode": "direct",
            "observed_mask": False,
        }
    )

    with pytest.raises(RuntimeError, match="last-value substitution is forbidden"):
        make_forecast_row(
            ctx=ctx,
            event=event,
            ledger_idx=0,
            selected=_candidate("statsforecast_autoets", "autoets"),
            method="unit_test",
            predictions_cache={},
        )


def test_prefit_group_failure_reports_backend_context(monkeypatch) -> None:
    def fail_predictor(*_args, **_kwargs):
        raise ValueError("unit backend failure")

    monkeypatch.setattr(
        external_forecasting,
        "make_statsforecast_predictor",
        lambda: fail_predictor,
    )
    times = np.asarray(["2023-01-01", "2023-01-08"], dtype="datetime64[ns]")
    history = SeriesHistory(
        times=times,
        releases=times,
        values=np.asarray([1.0, 2.0]),
    )
    ctx = SimpleNamespace(
        ledger_entity_col="entity_id",
        manifest_row=pd.Series({"cadence_days": 7}),
        history_index={("E1", "target"): history},
    )
    ledger = pd.DataFrame(
        [
            {
                "entity_id": "E1",
                "component": "target",
                "forecast_origin": "2023-01-08",
                "target_time": "2023-01-15",
                "horizon": 1,
                "mode": "direct",
            }
        ]
    )

    with pytest.raises(RuntimeError, match="pre-fit group execution failed") as exc_info:
        pre_fit_predictions(
            _candidate("statsforecast_autoets", "autoets"),
            ctx,
            ledger,
        )
    message = str(exc_info.value)
    assert '"backend":"statsforecast_or_prophet"' in message
    assert '"entity_id":"E1"' in message
    assert '"error":"unit backend failure"' in message
    assert "last-value substitution is forbidden" in message
