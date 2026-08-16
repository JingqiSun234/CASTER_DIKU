from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from caster.forecast import FORECAST_ARCHIVE_COLUMNS
from .base_adapter import BaseCandidateAdapter
@dataclass
class ToyState:
    panel: pd.DataFrame
    seed: int
class LastValueAdapter(BaseCandidateAdapter):
    def __init__(self, model_id: str = "last_value", family: str = "statistical") -> None:
        self.model_id = model_id; self.family = family
    def initialize(self, panel: pd.DataFrame, seed: int = 0) -> ToyState:
        return ToyState(panel=panel.copy(), seed=seed)
    def transition(self, state: ToyState, forecast_origin: pd.Timestamp) -> ToyState:
        return state
    def forecast_ledger(self, state: ToyState, ledger: pd.DataFrame) -> pd.DataFrame:
        panel = state.panel.copy(); panel["week_end"] = pd.to_datetime(panel["week_end"])
        out = []
        for _, e in ledger.iterrows():
            origin = pd.Timestamp(e["forecast_origin"])
            hist = panel[(panel["jurisdiction"].astype(str) == str(e["entity_id"])) & (panel["week_end"] <= origin)]
            pred = 0.0 if hist.empty else float(hist.iloc[-1][e["component"]])
            out.append({"dataset": e["dataset"], "model_id": self.model_id, "family": self.family, "particle_id": 0, "entity_id": e["entity_id"], "forecast_origin": e["forecast_origin"], "target_time": e["target_time"], "component": e["component"], "horizon": int(e["horizon"]), "forecast_id": e["forecast_id"], "pred_mean": max(pred, 0.0), "pred_var": 0.0, "generated_at": e["forecast_origin"], "features_available_until": e["forecast_origin"]})
        return pd.DataFrame(out, columns=FORECAST_ARCHIVE_COLUMNS)
    def forecast_draws(self, state: ToyState, ledger: pd.DataFrame, n_draws: int, seed: int = 0) -> pd.DataFrame:
        arch = self.forecast_ledger(state, ledger); rng = np.random.default_rng(seed); rows = []
        for _, r in arch.iterrows():
            scale = max(1.0, 0.05 * (1.0 + r["pred_mean"]))
            for b, draw in enumerate(rng.normal(r["pred_mean"], scale, size=n_draws)):
                rows.append({"forecast_id": r["forecast_id"], "model_id": self.model_id, "particle_id": 0, "draw_id": b, "draw": max(float(draw), 0.0)})
        return pd.DataFrame(rows)
    def serialize_state(self, state: ToyState) -> dict[str, object]:
        return {"seed": state.seed, "n_rows": int(len(state.panel))}
class DriftAdapter(LastValueAdapter):
    def __init__(self, model_id: str = "drift", family: str = "state_space", slope: float = 1.0) -> None:
        super().__init__(model_id=model_id, family=family); self.slope = float(slope)
    def forecast_ledger(self, state: ToyState, ledger: pd.DataFrame) -> pd.DataFrame:
        arch = super().forecast_ledger(state, ledger)
        arch["pred_mean"] = np.maximum(0.0, arch["pred_mean"].astype(float) + self.slope * arch["horizon"].astype(float))
        return arch
