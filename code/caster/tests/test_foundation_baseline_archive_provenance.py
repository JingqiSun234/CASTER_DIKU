from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_selected_forecast_archive_impl import (  # noqa: E402
    _baseline_forecast_to_archive,
)


def _ledger() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "toy",
                "entity_id": "E1",
                "forecast_origin": "2023-01-07",
                "target_time": "2023-01-14",
                "component": "target",
                "horizon": 1,
                "forecast_id": "f1",
                "observed_mask": True,
            }
        ]
    )


def test_foundation_baseline_provenance_is_preserved(tmp_path: Path) -> None:
    forecast_path = tmp_path / "forecast.csv"
    pd.DataFrame(
        [
            {
                "forecast_id": "f1",
                "pred_mean": 2.0,
                "pred_lower_90": 1.0,
                "pred_upper_90": 3.0,
                "generated_at": "2023-01-07",
                "features_available_until": "2023-01-06",
            }
        ]
    ).to_csv(forecast_path, index=False)

    archive = _baseline_forecast_to_archive(
        model_id="chronos_external",
        family="foundation_ts",
        forecast_path=forecast_path,
        ledger=_ledger(),
    )

    assert archive.loc[0, "generated_at"] == "2023-01-07"
    assert archive.loc[0, "features_available_until"] == "2023-01-06"


def test_foundation_baseline_missing_provenance_fails_closed(
    tmp_path: Path,
) -> None:
    forecast_path = tmp_path / "forecast.csv"
    pd.DataFrame([{"forecast_id": "f1", "pred_mean": 2.0}]).to_csv(
        forecast_path, index=False
    )

    with pytest.raises(ValueError, match="provenance"):
        _baseline_forecast_to_archive(
            model_id="timesfm_external",
            family="foundation_ts",
            forecast_path=forecast_path,
            ledger=_ledger(),
        )
