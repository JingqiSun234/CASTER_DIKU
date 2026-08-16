from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pandas as pd
import pytest

from caster_baselines.forecast_strategy import RECURSIVE_ROLLOUT
from caster_baselines.neuralforecast_runner import (
    RELEASE_TIME_COL,
    NeuralScope,
    RealNeuralForecastBackend,
    _available_nf_df,
    _native_target_requests,
    panel_component_to_nf_df,
)


ORIGIN = "2022-11-26"


def _manifest_row() -> pd.Series:
    return pd.Series(
        {
            "panel_entity_col": "jurisdiction",
            "panel_time_col": "week_end",
            "panel_format": "wide",
            "panel_target_cols": "covid_rate",
        }
    )


def _release_lag_one_panel() -> pd.DataFrame:
    weeks = pd.date_range("2022-10-01", periods=9, freq="7D")
    return pd.DataFrame(
        {
            "jurisdiction": ["E1"] * len(weeks),
            "week_end": weeks,
            "covid_rate": range(1, len(weeks) + 1),
            "__release_time__target": weeks + pd.Timedelta(days=7),
        }
    )


def _ledger(strategy: str, targets: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entity_id": "E1",
                "forecast_origin": ORIGIN,
                "target_time": target,
                "horizon": horizon,
                "split": "test",
                "forecast_strategy": strategy,
            }
            for horizon, target in enumerate(targets, start=1)
        ]
    )


def _scope(strategy: str, h: int) -> NeuralScope:
    return NeuralScope(
        dataset_key="benchmark_b",
        dataset="benchmark_b",
        component="covid_rate",
        h=h,
        input_size=4,
        freq="7D",
        cadence_days=7,
        train_cutoff=pd.Timestamp(ORIGIN),
        model_name="nbeats",
        seed=1,
        max_steps=1,
        device="cuda",
        forecast_strategy=strategy,
        mode="direct_1w2w" if strategy == "direct" else "rollout_4w",
    )


def _install_fake_neuralforecast(monkeypatch: pytest.MonkeyPatch):
    class FakeNeuralForecast:
        instances: list[FakeNeuralForecast] = []

        def __init__(self, models, freq):
            self.h = int(models[0].h)
            self.freq = freq
            self.fit_df = pd.DataFrame()
            self.predict_calls = 0
            self.__class__.instances.append(self)

        def fit(self, df, val_size, verbose):
            self.fit_df = df.copy()

        def predict(self, df, verbose):
            self.predict_calls += 1
            rows = []
            for entity, last_ds in df.groupby("unique_id")["ds"].max().items():
                rows.extend(
                    {
                        "unique_id": entity,
                        "ds": pd.Timestamp(last_ds) + pd.Timedelta(days=7 * step),
                        "nbeats": float(step),
                    }
                    for step in range(1, self.h + 1)
                )
            return pd.DataFrame(rows)

    fake_module = types.ModuleType("neuralforecast")
    fake_module.NeuralForecast = FakeNeuralForecast
    monkeypatch.setitem(sys.modules, "neuralforecast", fake_module)
    return FakeNeuralForecast


def _fake_backend(monkeypatch: pytest.MonkeyPatch):
    fake_class = _install_fake_neuralforecast(monkeypatch)
    backend = RealNeuralForecastBackend(max_steps=1, seed=1, device="cuda")
    monkeypatch.setattr(
        backend,
        "_model",
        lambda model_name, scope: SimpleNamespace(h=scope.h),
    )
    return backend, fake_class


def test_no_release_panel_preserves_benchmark_a_forward_fill_behavior() -> None:
    panel = pd.DataFrame(
        {
            "jurisdiction": ["E1"] * 4,
            "week_end": pd.date_range("2022-11-05", periods=4, freq="7D"),
            "covid_rate": [1.0, 2.0, float("nan"), 9.0],
        }
    )

    full_df = panel_component_to_nf_df(panel, _manifest_row(), pd.DataFrame(), "covid_rate")

    assert list(full_df.columns) == ["unique_id", "ds", "y"]
    assert full_df["y"].tolist() == [1.0, 2.0, 2.0, 9.0]
    available = _available_nf_df(full_df, pd.Timestamp("2022-11-19"), {"E1"})
    assert len(available) == 3
    assert available.iloc[-1]["y"] == 2.0


def test_release_lag_one_first_origin_exposes_only_eight_rows() -> None:
    full_df = panel_component_to_nf_df(
        _release_lag_one_panel(), _manifest_row(), pd.DataFrame(), "covid_rate"
    )

    assert RELEASE_TIME_COL in full_df.columns
    assert len(full_df) == 9
    available = _available_nf_df(full_df, pd.Timestamp(ORIGIN), {"E1"})
    assert len(available) == 8
    assert available["ds"].max() == pd.Timestamp("2022-11-19")


def test_direct_uses_native_steps_two_and_three_and_builds_h_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_df = panel_component_to_nf_df(
        _release_lag_one_panel(), _manifest_row(), pd.DataFrame(), "covid_rate"
    )
    ledger = _ledger("direct", ["2022-12-03", "2022-12-10"])
    available = _available_nf_df(full_df, pd.Timestamp(ORIGIN), {"E1"})
    requests, max_step = _native_target_requests(ledger, available, "entity_id", 7)
    assert sorted(native_step for _, native_step in requests.values()) == [2, 3]
    assert max_step == 3

    backend, fake_class = _fake_backend(monkeypatch)
    predictions, logs = backend.fit_predict(
        "nbeats",
        full_df[full_df["ds"] <= pd.Timestamp(ORIGIN)].copy(),
        full_df,
        ledger,
        "entity_id",
        _scope("direct", h=2),
    )

    fake = fake_class.instances[-1]
    assert fake.h == 3
    assert len(fake.fit_df) == 8
    assert fake.fit_df["ds"].max() == pd.Timestamp("2022-11-19")
    assert set(predictions) == {("E1", ORIGIN, 1), ("E1", ORIGIN, 2)}
    assert logs[0]["native_max_step"] == 3


def test_recursive_runs_native_steps_two_through_five_and_maps_nominal_horizons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_df = panel_component_to_nf_df(
        _release_lag_one_panel(), _manifest_row(), pd.DataFrame(), "covid_rate"
    )
    ledger = _ledger(
        RECURSIVE_ROLLOUT,
        ["2022-12-03", "2022-12-10", "2022-12-17", "2022-12-24"],
    )
    available = _available_nf_df(full_df, pd.Timestamp(ORIGIN), {"E1"})
    requests, max_step = _native_target_requests(ledger, available, "entity_id", 7)
    assert sorted(native_step for _, native_step in requests.values()) == [2, 3, 4, 5]
    assert max_step == 5

    backend, fake_class = _fake_backend(monkeypatch)
    predictions, logs = backend.fit_predict(
        "nbeats",
        full_df[full_df["ds"] <= pd.Timestamp(ORIGIN)].copy(),
        full_df,
        ledger,
        "entity_id",
        _scope(RECURSIVE_ROLLOUT, h=1),
    )

    fake = fake_class.instances[-1]
    assert fake.h == 1
    assert fake.predict_calls == 5
    assert set(predictions) == {("E1", ORIGIN, horizon) for horizon in range(1, 5)}
    assert logs[0]["native_max_step"] == 5


def test_nat_release_is_never_treated_as_available() -> None:
    full_df = pd.DataFrame(
        {
            "unique_id": ["E1", "E1"],
            "ds": pd.to_datetime(["2022-11-12", "2022-11-19"]),
            "y": [1.0, 2.0],
            RELEASE_TIME_COL: [pd.Timestamp("2022-11-19"), pd.NaT],
        }
    )

    available = _available_nf_df(full_df, pd.Timestamp(ORIGIN), {"E1"})
    assert available["ds"].tolist() == [pd.Timestamp("2022-11-12")]

    all_unreleased = full_df.assign(**{RELEASE_TIME_COL: pd.NaT})
    empty = _available_nf_df(all_unreleased, pd.Timestamp(ORIGIN), {"E1"})
    ledger = _ledger("direct", ["2022-12-03", "2022-12-10"])
    with pytest.raises(ValueError, match="no released NeuralForecast history"):
        _native_target_requests(ledger, empty, "entity_id", 7)
