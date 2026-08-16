from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
NEW_METHOD_SRC = ROOT / "code/caster/src"
if str(NEW_METHOD_SRC) not in sys.path:
    sys.path.insert(0, str(NEW_METHOD_SRC))

from caster.data.benchmark_a_mobility import (              
    MobilityMaterialization,
    materialize_mobility_features,
)
from caster.models.causal_covariates import (              
    CausalCovariateIndex,
    CausalCovariateSignal,
    adjust_forecast,
)


def materialize_benchmark_a_panel(panel: pd.DataFrame) -> MobilityMaterialization:
    return materialize_mobility_features(panel, ROOT / "data/benchmark_a/raw_all")


__all__ = [
    "CausalCovariateIndex",
    "CausalCovariateSignal",
    "MobilityMaterialization",
    "adjust_forecast",
    "materialize_benchmark_a_panel",
]
