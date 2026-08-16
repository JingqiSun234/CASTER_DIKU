from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from caster.forecast import FORECAST_ARCHIVE_COLUMNS
from .base_adapter import BaseCandidateAdapter

@dataclass
class PanelState:
    panel: pd.DataFrame
    seed: int

def _panel(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    if "entity_id" not in p.columns:
        p["entity_id"] = p["jurisdiction"].astype(str) if "jurisdiction" in p.columns else "global"
    p["entity_id"] = p["entity_id"].astype(str)
    p["week_end"] = pd.to_datetime(p["week_end"])
    return p

def _series(panel: pd.DataFrame, entity_id: str, component: str, origin: pd.Timestamp) -> np.ndarray:
    if component not in panel.columns:
        return np.asarray([], dtype=float)
    s = panel[(panel["entity_id"].astype(str) == str(entity_id)) & (panel["week_end"] <= origin)].sort_values("week_end")[component]
    return pd.to_numeric(s, errors="coerce").dropna().astype(float).to_numpy()

class RenewalAdapter(BaseCandidateAdapter):
    def __init__(self, model_id: str = "renewal_kernel", family: str = "renewal", *, reproduction: float = 1.03, kernel: tuple[float, ...] = (0.50, 0.30, 0.15, 0.05), min_var: float = 1.0) -> None:
        self.model_id = model_id
        self.family = family
        self.reproduction = float(reproduction)
        weights = np.asarray(kernel, dtype=float)
        weights = weights / max(weights.sum(), 1e-12)
        self.kernel = tuple(float(x) for x in weights)
        self.min_var = float(min_var)

    def initialize(self, panel: pd.DataFrame, seed: int = 0) -> PanelState:
        return PanelState(_panel(panel), int(seed))

    def transition(self, state: PanelState, forecast_origin: pd.Timestamp) -> PanelState:
        return state

    def _forecast(self, y: np.ndarray, horizon: int) -> tuple[float, float]:
        hist = list(np.maximum(y.astype(float), 0.0))
        if not hist:
            return 0.0, self.min_var
        for _ in range(int(horizon)):
            tail = np.asarray((hist[-len(self.kernel):])[::-1], dtype=float)
            k = np.asarray(self.kernel[: len(tail)], dtype=float)
            k = k / max(k.sum(), 1e-12)
            nxt = max(0.0, self.reproduction * float(np.dot(k, tail)))
            hist.append(nxt)
        mean = hist[-1]
        residual_scale = float(np.std(np.diff(hist[-min(len(hist), 6):]))) if len(hist) > 2 else 0.0
        var = max(self.min_var, mean + residual_scale * residual_scale)
        return mean, var

    def forecast_ledger(self, state: PanelState, ledger: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, e in ledger.iterrows():
            y = _series(state.panel, str(e["entity_id"]), str(e["component"]), pd.Timestamp(e["forecast_origin"]))
            mean, var = self._forecast(y, int(e["horizon"]))
            rows.append({"dataset": e["dataset"], "model_id": self.model_id, "family": self.family, "particle_id": 0, "entity_id": e["entity_id"], "forecast_origin": e["forecast_origin"], "target_time": e["target_time"], "component": e["component"], "horizon": int(e["horizon"]), "forecast_id": e["forecast_id"], "pred_mean": mean, "pred_var": var, "generated_at": e["forecast_origin"], "features_available_until": e["forecast_origin"]})
        return pd.DataFrame(rows, columns=FORECAST_ARCHIVE_COLUMNS)

    def forecast_draws(self, state: PanelState, ledger: pd.DataFrame, n_draws: int, seed: int = 0) -> pd.DataFrame:
        arch = self.forecast_ledger(state, ledger)
        rng = np.random.default_rng(seed)
        rows = []
        for _, r in arch.iterrows():
            for b, draw in enumerate(rng.normal(float(r["pred_mean"]), np.sqrt(max(float(r["pred_var"]), 1e-6)), size=int(n_draws))):
                rows.append({"forecast_id": r["forecast_id"], "model_id": self.model_id, "particle_id": 0, "draw_id": b, "draw": max(float(draw), 0.0)})
        return pd.DataFrame(rows)

    def serialize_state(self, state: PanelState) -> dict[str, object]:
        return {"seed": state.seed, "reproduction": self.reproduction, "kernel": list(self.kernel)}

class LocalLevelStateSpaceAdapter(BaseCandidateAdapter):
    def __init__(self, model_id: str = "local_level", family: str = "state_space", *, alpha: float = 0.65, trend_damping: float = 0.75, min_var: float = 1.0) -> None:
        self.model_id = model_id
        self.family = family
        self.alpha = float(alpha)
        self.trend_damping = float(trend_damping)
        self.min_var = float(min_var)

    def initialize(self, panel: pd.DataFrame, seed: int = 0) -> PanelState:
        return PanelState(_panel(panel), int(seed))

    def transition(self, state: PanelState, forecast_origin: pd.Timestamp) -> PanelState:
        return state

    def _forecast(self, y: np.ndarray, horizon: int) -> tuple[float, float]:
        if len(y) == 0:
            return 0.0, self.min_var
        level = float(y[0])
        residuals: list[float] = []
        for val in y[1:]:
            pred = level
            residuals.append(float(val) - pred)
            level = self.alpha * float(val) + (1.0 - self.alpha) * level
        trend = float(y[-1] - y[-2]) if len(y) >= 2 else 0.0
        mean = max(0.0, level + sum((self.trend_damping ** h) * trend for h in range(1, int(horizon) + 1)))
        var = max(self.min_var, float(np.var(residuals)) if residuals else self.min_var) * max(1, int(horizon))
        return mean, var

    def forecast_ledger(self, state: PanelState, ledger: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, e in ledger.iterrows():
            y = _series(state.panel, str(e["entity_id"]), str(e["component"]), pd.Timestamp(e["forecast_origin"]))
            mean, var = self._forecast(y, int(e["horizon"]))
            rows.append({"dataset": e["dataset"], "model_id": self.model_id, "family": self.family, "particle_id": 0, "entity_id": e["entity_id"], "forecast_origin": e["forecast_origin"], "target_time": e["target_time"], "component": e["component"], "horizon": int(e["horizon"]), "forecast_id": e["forecast_id"], "pred_mean": mean, "pred_var": var, "generated_at": e["forecast_origin"], "features_available_until": e["forecast_origin"]})
        return pd.DataFrame(rows, columns=FORECAST_ARCHIVE_COLUMNS)

    def forecast_draws(self, state: PanelState, ledger: pd.DataFrame, n_draws: int, seed: int = 0) -> pd.DataFrame:
        arch = self.forecast_ledger(state, ledger)
        rng = np.random.default_rng(seed)
        rows = []
        for _, r in arch.iterrows():
            for b, draw in enumerate(rng.normal(float(r["pred_mean"]), np.sqrt(max(float(r["pred_var"]), 1e-6)), size=int(n_draws))):
                rows.append({"forecast_id": r["forecast_id"], "model_id": self.model_id, "particle_id": 0, "draw_id": b, "draw": max(float(draw), 0.0)})
        return pd.DataFrame(rows)

    def serialize_state(self, state: PanelState) -> dict[str, object]:
        return {"seed": state.seed, "alpha": self.alpha, "trend_damping": self.trend_damping}
