from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from caster.forecast import FORECAST_ARCHIVE_COLUMNS
from .base_adapter import BaseCandidateAdapter

@dataclass
class NeuralState:
    panel: pd.DataFrame
    seed: int
    fitted: dict[str, dict[str, np.ndarray | float]]

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

def _fit_component(panel: pd.DataFrame, component: str, lags: int, ridge: float, seed: int) -> dict[str, np.ndarray | float]:
    xs, ys = [], []
    if component not in panel.columns:
        return {"coef": np.zeros(lags + 1), "resid_var": 1.0}
    for _, g in panel.sort_values("week_end").groupby("entity_id"):
        y = pd.to_numeric(g[component], errors="coerce").dropna().astype(float).to_numpy()
        for t in range(lags, len(y)):
            xs.append(np.r_[1.0, np.log1p(y[t-lags:t][::-1])])
            ys.append(np.log1p(max(y[t], 0.0)))
    if len(xs) < max(3, lags + 1):
        return {"coef": np.r_[np.log1p(float(np.nanmedian(panel[component])) if component in panel else 0.0), np.zeros(lags)], "resid_var": 1.0}
    X = np.vstack(xs); Y = np.asarray(ys)
    penalty = float(ridge) * np.eye(X.shape[1]); penalty[0, 0] = 0.0
    coef = np.linalg.solve(X.T @ X + penalty, X.T @ Y)
    resid = Y - X @ coef
    return {"coef": coef, "resid_var": float(max(np.var(resid), 1e-4))}

class MLPAdapter(BaseCandidateAdapter):
    def __init__(self, model_id: str = "mlp_random_feature", family: str = "neural", *, lags: int = 4, hidden: int = 8, ridge: float = 1.0) -> None:
        self.model_id = model_id
        self.family = family
        self.lags = int(lags)
        self.hidden = int(hidden)
        self.ridge = float(ridge)

    def initialize(self, panel: pd.DataFrame, seed: int = 0) -> NeuralState:
        p = _panel(panel)
        rng = np.random.default_rng(seed)
        fitted = {}
        for comp in [c for c in p.columns if c not in {"entity_id", "jurisdiction", "week_end"} and pd.api.types.is_numeric_dtype(p[c])]:
            base = _fit_component(p, comp, self.lags, self.ridge, seed)
            W = rng.normal(0, 1.0 / max(self.lags, 1), size=(self.hidden, self.lags))
            b = rng.normal(0, 0.1, size=self.hidden)
            xs, ys = [], []
            for _, g in p.sort_values("week_end").groupby("entity_id"):
                y = pd.to_numeric(g[comp], errors="coerce").dropna().astype(float).to_numpy()
                for t in range(self.lags, len(y)):
                    x = np.log1p(y[t-self.lags:t][::-1])
                    phi = np.tanh(W @ x + b)
                    xs.append(np.r_[1.0, phi])
                    ys.append(np.log1p(max(y[t], 0.0)))
            if len(xs) >= self.hidden + 1:
                X = np.vstack(xs); Y = np.asarray(ys)
                penalty = self.ridge * np.eye(X.shape[1]); penalty[0, 0] = 0.0
                coef = np.linalg.solve(X.T @ X + penalty, X.T @ Y)
                resid = Y - X @ coef
                fitted[comp] = {"W": W, "b": b, "coef": coef, "resid_var": float(max(np.var(resid), 1e-4))}
            else:
                fitted[comp] = {"W": W, "b": b, "coef": np.r_[base["coef"][0], np.zeros(self.hidden)], "resid_var": float(base["resid_var"])}
        return NeuralState(p, int(seed), fitted)

    def transition(self, state: NeuralState, forecast_origin: pd.Timestamp) -> NeuralState:
        return state

    def _one_step(self, params: dict[str, np.ndarray | float], hist: list[float]) -> tuple[float, float]:
        x = np.log1p(np.asarray((hist[-self.lags:])[::-1] + [hist[-1]] * max(0, self.lags - len(hist)), dtype=float)[:self.lags])
        W = np.asarray(params["W"]); b = np.asarray(params["b"]); coef = np.asarray(params["coef"])
        phi = np.tanh(W @ x + b)
        log_mu = float(np.r_[1.0, phi] @ coef)
        mean = max(0.0, float(np.expm1(log_mu)))
        var = max(mean, 1.0) * float(np.exp(float(params.get("resid_var", 0.05))))
        return mean, var

    def forecast_ledger(self, state: NeuralState, ledger: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, e in ledger.iterrows():
            comp = str(e["component"])
            y = _series(state.panel, str(e["entity_id"]), comp, pd.Timestamp(e["forecast_origin"]))
            hist = list(np.maximum(y, 0.0)) or [0.0]
            if comp not in state.fitted:
                mean, var = hist[-1], max(hist[-1], 1.0)
            else:
                var = 1.0
                for _h in range(int(e["horizon"])):
                    mean, var = self._one_step(state.fitted[comp], hist)
                    hist.append(mean)
            rows.append({"dataset": e["dataset"], "model_id": self.model_id, "family": self.family, "particle_id": 0, "entity_id": e["entity_id"], "forecast_origin": e["forecast_origin"], "target_time": e["target_time"], "component": e["component"], "horizon": int(e["horizon"]), "forecast_id": e["forecast_id"], "pred_mean": mean, "pred_var": 0.0, "generated_at": e["forecast_origin"], "features_available_until": e["forecast_origin"]})
        return pd.DataFrame(rows, columns=FORECAST_ARCHIVE_COLUMNS)

    def forecast_draws(self, state: NeuralState, ledger: pd.DataFrame, n_draws: int, seed: int = 0) -> pd.DataFrame:
        arch = self.forecast_ledger(state, ledger)
        rng = np.random.default_rng(seed)
        rows = []
        for _, r in arch.iterrows():
            scale = max(1.0, 0.10 * (1.0 + float(r["pred_mean"])))
            for b, draw in enumerate(rng.normal(float(r["pred_mean"]), scale, size=int(n_draws))):
                rows.append({"forecast_id": r["forecast_id"], "model_id": self.model_id, "particle_id": 0, "draw_id": b, "draw": max(float(draw), 0.0)})
        return pd.DataFrame(rows)

    def serialize_state(self, state: NeuralState) -> dict[str, object]:
        return {"seed": state.seed, "components": sorted(state.fitted), "lags": self.lags, "hidden": self.hidden}

class RNNAdapter(BaseCandidateAdapter):
    def __init__(self, model_id: str = "rnn_ema", family: str = "neural", *, alpha: float = 0.55, gain: float = 1.0, min_var: float = 1.0) -> None:
        self.model_id = model_id
        self.family = family
        self.alpha = float(alpha)
        self.gain = float(gain)
        self.min_var = float(min_var)

    def initialize(self, panel: pd.DataFrame, seed: int = 0) -> NeuralState:
        return NeuralState(_panel(panel), int(seed), {})

    def transition(self, state: NeuralState, forecast_origin: pd.Timestamp) -> NeuralState:
        return state

    def _forecast(self, y: np.ndarray, horizon: int) -> tuple[float, float]:
        if len(y) == 0:
            return 0.0, self.min_var
        hidden = np.log1p(max(float(y[0]), 0.0))
        residuals = []
        for val in y[1:]:
            pred = float(np.expm1(hidden))
            residuals.append(float(val) - pred)
            hidden = self.alpha * np.log1p(max(float(val), 0.0)) + (1.0 - self.alpha) * hidden
        for _ in range(int(horizon)):
            hidden = self.gain * hidden
        mean = max(0.0, float(np.expm1(hidden)))
        var = max(self.min_var, float(np.var(residuals)) if residuals else self.min_var)
        return mean, var

    def forecast_ledger(self, state: NeuralState, ledger: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, e in ledger.iterrows():
            y = _series(state.panel, str(e["entity_id"]), str(e["component"]), pd.Timestamp(e["forecast_origin"]))
            mean, _var = self._forecast(y, int(e["horizon"]))
            rows.append({"dataset": e["dataset"], "model_id": self.model_id, "family": self.family, "particle_id": 0, "entity_id": e["entity_id"], "forecast_origin": e["forecast_origin"], "target_time": e["target_time"], "component": e["component"], "horizon": int(e["horizon"]), "forecast_id": e["forecast_id"], "pred_mean": mean, "pred_var": 0.0, "generated_at": e["forecast_origin"], "features_available_until": e["forecast_origin"]})
        return pd.DataFrame(rows, columns=FORECAST_ARCHIVE_COLUMNS)

    def forecast_draws(self, state: NeuralState, ledger: pd.DataFrame, n_draws: int, seed: int = 0) -> pd.DataFrame:
        arch = self.forecast_ledger(state, ledger)
        rng = np.random.default_rng(seed)
        rows = []
        for _, r in arch.iterrows():
            scale = max(1.0, 0.10 * (1.0 + float(r["pred_mean"])))
            for b, draw in enumerate(rng.normal(float(r["pred_mean"]), scale, size=int(n_draws))):
                rows.append({"forecast_id": r["forecast_id"], "model_id": self.model_id, "particle_id": 0, "draw_id": b, "draw": max(float(draw), 0.0)})
        return pd.DataFrame(rows)

    def serialize_state(self, state: NeuralState) -> dict[str, object]:
        return {"seed": state.seed, "alpha": self.alpha, "gain": self.gain}
