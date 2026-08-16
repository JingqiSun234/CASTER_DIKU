""
from __future__ import annotations

from dataclasses import dataclass


MODEL_POOL_GROUP = "model_pool"


@dataclass(frozen=True)
class SharedModelObject:
    ""

    canonical_model_id: str
    baseline_method: str
    forecast_path: str
    roles: tuple[str, str] = ("baseline", "candidate")

    @property
    def prediction_object_id(self) -> str:
        return f"shared_model::{self.canonical_model_id}"


def _shared(model_id: str, baseline_method: str, forecast_path: str) -> SharedModelObject:
    return SharedModelObject(
        canonical_model_id=model_id,
        baseline_method=baseline_method,
        forecast_path=forecast_path,
    )


                                                                             
                                                                             
CANONICAL_SHARED_MODEL_OBJECTS: dict[str, SharedModelObject] = {
    "last_value": _shared("last_value", "last_value", "naive/lastvalue/forecast.csv"),
    "seasonal_naive": _shared("seasonal_naive", "seasonal_naive", "naive/seasonalnaive/forecast.csv"),
    "statsforecast_autoarima": _shared("statsforecast_autoarima", "autoarima", "statsforecast/autoarima/forecast.csv"),
    "statsforecast_autoets": _shared("statsforecast_autoets", "autoets", "statsforecast/autoets/forecast.csv"),
    "statsforecast_autotheta": _shared("statsforecast_autotheta", "autotheta", "statsforecast/autotheta/forecast.csv"),
    "statsforecast_autoces": _shared("statsforecast_autoces", "autoces", "statsforecast/autoces/forecast.csv"),
    "prophet": _shared("prophet", "prophet", "prophet/forecast.csv"),
    "deepar_style": _shared("deepar_style", "deepar", "neural/deepar/forecast.csv"),
    "nbeats_basis": _shared("nbeats_basis", "nbeats", "neural/nbeats/forecast.csv"),
    "nhits_hinterp": _shared("nhits_hinterp", "nhits", "neural/nhits/forecast.csv"),
    "patchtst_patched": _shared("patchtst_patched", "patchtst", "neural/patchtst/forecast.csv"),
    "tft_gated": _shared("tft_gated", "tft", "neural/tft/forecast.csv"),
    "chronos_external": _shared("chronos_external", "chronos_bolt_small", "foundation/chronos/forecast.csv"),
    "timesfm_external": _shared("timesfm_external", "timesfm_2_0", "foundation/timesfm/forecast.csv"),
}

SHARED_MODEL_CONTRACT: dict[str, tuple[str, str]] = {
    model_id: (obj.baseline_method, obj.forecast_path)
    for model_id, obj in CANONICAL_SHARED_MODEL_OBJECTS.items()
}

SHARED_BASELINE_METHODS = {
    model_id: obj.baseline_method
    for model_id, obj in CANONICAL_SHARED_MODEL_OBJECTS.items()
}
SHARED_FORECAST_PATHS = {
    model_id: obj.forecast_path
    for model_id, obj in CANONICAL_SHARED_MODEL_OBJECTS.items()
}


def model_role_fields(model_id: str) -> dict[str, object]:
    ""

    model_id = str(model_id)
    baseline_method = SHARED_BASELINE_METHODS.get(model_id, "")
    is_baseline = bool(baseline_method)
    prediction_object_id = (
        CANONICAL_SHARED_MODEL_OBJECTS[model_id].prediction_object_id
        if is_baseline
        else f"candidate_model::{model_id}"
    )
    return {
        "canonical_model_id": model_id,
        "prediction_object_id": prediction_object_id,
        "candidate_model_id": model_id,
        "baseline_method": baseline_method,
        "is_baseline": is_baseline,
        "is_candidate": True,
        "model_roles": "baseline;candidate" if is_baseline else "candidate",
        "prediction_result_count": 1,
        "method_group": MODEL_POOL_GROUP,
    }
