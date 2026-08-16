from __future__ import annotations

import pandas as pd
import pytest

from caster.models.foundation_adapters import ExternalForecastAdapter, FoundationState


def _ledger() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "unit_test",
                "entity_id": "E1",
                "forecast_origin": "2024-01-01",
                "target_time": "2024-01-08",
                "component": "target",
                "horizon": 1,
                "forecast_id": "F1",
            }
        ]
    )


def _state(**extra: object) -> FoundationState:
    row = {
        "forecast_id": "F1",
        "pred_mean": 2.0,
        **extra,
    }
    return FoundationState(
        panel=pd.DataFrame(),
        seed=0,
        external_forecasts=pd.DataFrame([row]),
    )


def test_external_forecast_requires_explicit_asof_provenance() -> None:
    adapter = ExternalForecastAdapter(model_id="external")
    with pytest.raises(ValueError, match="required as-of provenance"):
        adapter.forecast_ledger(_state(), _ledger())


def test_external_forecast_without_artifact_never_substitutes_last_value() -> None:
    adapter = ExternalForecastAdapter(model_id="external")
    with pytest.raises(RuntimeError, match="silent last-value substitution is forbidden"):
        adapter.forecast_ledger(
            FoundationState(panel=pd.DataFrame(), seed=0, external_forecasts=None),
            _ledger(),
        )


def test_external_forecast_rejects_post_origin_provenance() -> None:
    adapter = ExternalForecastAdapter(model_id="external")
    with pytest.raises(ValueError, match="after forecast_origin"):
        adapter.forecast_ledger(
            _state(
                generated_at="2024-01-02",
                features_available_until="2024-01-01",
            ),
            _ledger(),
        )


def test_external_forecast_rejects_features_after_generation() -> None:
    adapter = ExternalForecastAdapter(model_id="external")
    with pytest.raises(ValueError, match="features_available_until values after generated_at"):
        adapter.forecast_ledger(
            _state(
                generated_at="2023-12-30",
                features_available_until="2023-12-31",
            ),
            _ledger(),
        )


def test_external_forecast_accepts_explicit_origin_valid_provenance() -> None:
    adapter = ExternalForecastAdapter(model_id="external")
    result = adapter.forecast_ledger(
        _state(
            generated_at="2024-01-01",
            features_available_until="2023-12-31",
        ),
        _ledger(),
    )
    assert result.loc[0, "model_id"] == "external"
    assert str(result.loc[0, "generated_at"]).startswith("2024-01-01")
    assert str(result.loc[0, "features_available_until"]).startswith("2023-12-31")
