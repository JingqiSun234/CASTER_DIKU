from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caster_baselines.forecast_strategy import RECURSIVE_ROLLOUT
from caster_baselines.ledger_runner import (
    Z90,
    residual_sigma,
    run_baseline_from_manifest,
)


ORIGIN = pd.Timestamp("2022-11-26")


def _write_case(
    tmp_path: Path,
    *,
    horizon: int,
    strategy: str,
    release_lag_steps: int,
) -> Path:
    weeks = pd.date_range("2022-09-03", periods=18, freq="7D")
    panel = pd.DataFrame(
        {
            "jurisdiction": ["E1"] * len(weeks),
            "week_end": weeks,
            "flu_rate": np.arange(1.0, len(weeks) + 1.0),
        }
    )
    if release_lag_steps:
        panel["__release_time__target"] = weeks + pd.Timedelta(
            days=7 * release_lag_steps
        )
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(data_dir / "panel.csv", index=False)

    target = ORIGIN + pd.Timedelta(days=7 * horizon)
    ledger = pd.DataFrame(
        [
            {
                "dataset": "benchmark_b_flu_toy",
                "entity_id": "E1",
                "forecast_id": f"flu_h{horizon}",
                "forecast_origin": ORIGIN.strftime("%Y-%m-%d"),
                "target_time": target.strftime("%Y-%m-%d"),
                "component": "flu_rate",
                "horizon": horizon,
                "observed_value": 25.0,
                "observed_mask": True,
                "split": "test",
                "mode": (
                    "rollout_4w"
                    if strategy == RECURSIVE_ROLLOUT
                    else "direct_1w2w"
                ),
                "forecast_strategy": strategy,
            }
        ]
    )
    ledger.to_csv(data_dir / "ledger.csv", index=False)
    manifest = pd.DataFrame(
        [
            {
                "dataset_key": "benchmark_b_flu_toy",
                "dataset": "benchmark_b_flu_toy",
                "panel_path": "data/panel.csv",
                "ledger_path": "data/ledger.csv",
                "panel_format": "wide",
                "panel_entity_col": "jurisdiction",
                "panel_time_col": "week_end",
                "panel_component_col": "NA",
                "panel_value_col": "NA",
                "panel_target_cols": "flu_rate",
                "ledger_rows": 1,
                "cadence_days": 7,
            }
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest_path


def _run(tmp_path: Path, manifest: Path, model: str) -> pd.DataFrame:
    out = run_baseline_from_manifest(
        manifest_path=manifest,
        out_dir=tmp_path / f"run_{model}",
        model=model,
        root=tmp_path / "baseline_root",
        caster_root=tmp_path,
    )
    return pd.read_csv(out / "forecast.csv", keep_default_na=False)


def test_last_value_recursive_h4_runs_native_step_five_for_interval(
    tmp_path: Path,
) -> None:
    manifest = _write_case(
        tmp_path,
        horizon=4,
        strategy=RECURSIVE_ROLLOUT,
        release_lag_steps=1,
    )
    forecast = _run(tmp_path, manifest, "last_value")
    row = forecast.iloc[0]

    released_values = np.arange(1.0, 13.0)
    expected_sigma_native_five = residual_sigma(
        np.append(released_values, [released_values[-1]] * 4),
        released_values[-1],
    )
    old_incorrect_sigma_nominal_four = residual_sigma(
        np.append(released_values, [released_values[-1]] * 3),
        released_values[-1],
    )
    archived_sigma = (float(row["pred_upper_90"]) - float(row["pred_lower_90"])) / (
        2.0 * Z90
    )

    assert row["last_released_target_time"] == "2022-11-19"
    assert int(row["native_horizon_steps"]) == 5
    assert row["forecasted_native_target_time"] == row["target_time"] == "2022-12-24"
    assert np.isclose(archived_sigma, expected_sigma_native_five)
    assert not np.isclose(archived_sigma, old_incorrect_sigma_nominal_four)


def test_seasonal_naive_keeps_actual_target_date_lookup(tmp_path: Path) -> None:
    manifest = _write_case(
        tmp_path,
        horizon=4,
        strategy=RECURSIVE_ROLLOUT,
        release_lag_steps=1,
    )
    forecast = _run(tmp_path, manifest, "seasonal_naive")
    row = forecast.iloc[0]

    # Weekly seasonal-naive uses an 8-week season.  2022-12-24 - 8 weeks is
    # 2022-10-29, whose panel value is 9.0.
    assert float(row["pred_mean"]) == 9.0
    assert int(row["native_horizon_steps"]) == 5
    assert row["last_released_target_time"] == "2022-11-19"
    assert row["forecasted_native_target_time"] == "2022-12-24"


def test_no_lag_last_value_keeps_nominal_native_alignment(tmp_path: Path) -> None:
    manifest = _write_case(
        tmp_path,
        horizon=1,
        strategy="direct",
        release_lag_steps=0,
    )
    forecast = _run(tmp_path, manifest, "last_value")
    row = forecast.iloc[0]

    assert row["last_released_target_time"] == "2022-11-26"
    assert int(row["native_horizon_steps"]) == 1
    assert row["forecasted_native_target_time"] == row["target_time"]


def test_naive_runner_rejects_off_cadence_target(tmp_path: Path) -> None:
    manifest = _write_case(
        tmp_path,
        horizon=1,
        strategy="direct",
        release_lag_steps=1,
    )
    ledger_path = tmp_path / "data" / "ledger.csv"
    ledger = pd.read_csv(ledger_path)
    ledger.loc[0, "target_time"] = "2022-12-04"
    ledger.to_csv(ledger_path, index=False)

    with pytest.raises(RuntimeError, match="baseline run blocked"):
        _run(tmp_path, manifest, "last_value")

    blocker = pd.read_csv(tmp_path / "run_last_value" / "blocker_report.csv")
    assert blocker["reason"].str.contains("native horizon resolution failed").any()
