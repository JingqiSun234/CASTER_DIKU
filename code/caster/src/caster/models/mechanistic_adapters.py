from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from caster.forecast import FORECAST_ARCHIVE_COLUMNS
from .base_adapter import BaseCandidateAdapter

@dataclass
class CompartmentState:
    panel: pd.DataFrame
    population: float
    beta: float
    gamma: float
    sigma: float | None
    reporting_rate: float
    weekly_observation_scale: float
    seed: int

def _ensure_panel(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    if "entity_id" not in p.columns:
        if "jurisdiction" in p.columns:
            p["entity_id"] = p["jurisdiction"].astype(str)
        else:
            p["entity_id"] = "global"
    p["entity_id"] = p["entity_id"].astype(str)
    p["week_end"] = pd.to_datetime(p["week_end"])
    return p

def _history(panel: pd.DataFrame, entity_id: str, component: str, origin: pd.Timestamp) -> pd.DataFrame:
    if component not in panel.columns:
        return panel.iloc[0:0]
    return panel[(panel["entity_id"].astype(str) == str(entity_id)) & (panel["week_end"] <= origin)].sort_values("week_end")

def _last_observed(panel: pd.DataFrame, entity_id: str, component: str, origin: pd.Timestamp) -> float:
    hist = _history(panel, entity_id, component, origin)
    if hist.empty or component not in hist.columns:
        return 0.0
    vals = pd.to_numeric(hist[component], errors="coerce").dropna()
    return float(vals.iloc[-1]) if not vals.empty else 0.0

def _initial_compartments(last_obs: float, population: float, reporting_rate: float, exposed_multiplier: float = 0.6) -> tuple[float, float, float, float]:
    pop = max(float(population), 1.0)
    i0 = min(max(float(last_obs) / max(float(reporting_rate), 1e-6), 1.0), 0.25 * pop)
    e0 = min(max(float(exposed_multiplier) * i0, 0.0), 0.25 * pop)
    r0 = min(max(0.05 * pop, 0.0), 0.8 * pop)
    s0 = max(pop - e0 - i0 - r0, 0.0)
    return s0, e0, i0, r0

def _simulate_mean_sir(s: float, i: float, r: float, *, beta: float, gamma: float, population: float, horizon: int) -> tuple[float, float, float, float]:
    weekly_new = 0.0
    for _ in range(int(horizon)):
        force = 1.0 - np.exp(-max(beta, 0.0) * max(i, 0.0) / max(population, 1.0))
        rec_prob = 1.0 - np.exp(-max(gamma, 0.0))
        new_inf = min(s, max(0.0, s * force))
        new_rec = min(i, max(0.0, i * rec_prob))
        s = max(s - new_inf, 0.0)
        i = max(i + new_inf - new_rec, 0.0)
        r = max(r + new_rec, 0.0)
        weekly_new = new_inf
    return s, i, r, weekly_new

def _simulate_mean_seir(s: float, e: float, i: float, r: float, *, beta: float, sigma: float, gamma: float, population: float, horizon: int) -> tuple[float, float, float, float, float]:
    weekly_new = 0.0
    for _ in range(int(horizon)):
        force = 1.0 - np.exp(-max(beta, 0.0) * max(i, 0.0) / max(population, 1.0))
        inc_prob = 1.0 - np.exp(-max(sigma, 0.0))
        rec_prob = 1.0 - np.exp(-max(gamma, 0.0))
        new_exp = min(s, max(0.0, s * force))
        new_inf = min(e, max(0.0, e * inc_prob))
        new_rec = min(i, max(0.0, i * rec_prob))
        s = max(s - new_exp, 0.0)
        e = max(e + new_exp - new_inf, 0.0)
        i = max(i + new_inf - new_rec, 0.0)
        r = max(r + new_rec, 0.0)
        weekly_new = new_inf
    return s, e, i, r, weekly_new

class SIRAdapter(BaseCandidateAdapter):
    def __init__(self, model_id: str = "sir_tau", family: str = "mechanistic", *, population: float = 1_000_000.0, beta: float = 0.55, gamma: float = 0.35, reporting_rate: float = 0.04, weekly_observation_scale: float = 1.0) -> None:
        self.model_id = model_id
        self.family = family
        self.population = float(population)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.reporting_rate = float(reporting_rate)
        self.weekly_observation_scale = float(weekly_observation_scale)

    def initialize(self, panel: pd.DataFrame, seed: int = 0) -> CompartmentState:
        return CompartmentState(_ensure_panel(panel), self.population, self.beta, self.gamma, None, self.reporting_rate, self.weekly_observation_scale, int(seed))

    def transition(self, state: CompartmentState, forecast_origin: pd.Timestamp) -> CompartmentState:
        return state

    def _forecast_one(self, state: CompartmentState, entity_id: str, component: str, origin: pd.Timestamp, horizon: int) -> tuple[float, float]:
        last_obs = _last_observed(state.panel, entity_id, component, origin)
        s, _e, i, r = _initial_compartments(last_obs, state.population, state.reporting_rate)
        _s, _i, _r, incidence = _simulate_mean_sir(s, i, r, beta=state.beta, gamma=state.gamma, population=state.population, horizon=int(horizon))
        mean = max(0.0, state.weekly_observation_scale * state.reporting_rate * incidence)
        var = max(mean, 1.0) + 0.05 * mean * mean
        return mean, var

    def forecast_ledger(self, state: CompartmentState, ledger: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, e in ledger.iterrows():
            mean, var = self._forecast_one(state, str(e["entity_id"]), str(e["component"]), pd.Timestamp(e["forecast_origin"]), int(e["horizon"]))
            rows.append({"dataset": e["dataset"], "model_id": self.model_id, "family": self.family, "particle_id": 0, "entity_id": e["entity_id"], "forecast_origin": e["forecast_origin"], "target_time": e["target_time"], "component": e["component"], "horizon": int(e["horizon"]), "forecast_id": e["forecast_id"], "pred_mean": mean, "pred_var": var, "generated_at": e["forecast_origin"], "features_available_until": e["forecast_origin"]})
        return pd.DataFrame(rows, columns=FORECAST_ARCHIVE_COLUMNS)

    def forecast_draws(self, state: CompartmentState, ledger: pd.DataFrame, n_draws: int, seed: int = 0) -> pd.DataFrame:
        arch = self.forecast_ledger(state, ledger)
        rng = np.random.default_rng(seed)
        rows = []
        for _, r in arch.iterrows():
            scale = np.sqrt(max(float(r["pred_var"]), 1e-6))
            for b, draw in enumerate(rng.normal(float(r["pred_mean"]), scale, size=int(n_draws))):
                rows.append({"forecast_id": r["forecast_id"], "model_id": self.model_id, "particle_id": int(r["particle_id"]), "draw_id": b, "draw": max(float(draw), 0.0)})
        return pd.DataFrame(rows)

    def serialize_state(self, state: CompartmentState) -> dict[str, object]:
        return {"population": state.population, "beta": state.beta, "gamma": state.gamma, "reporting_rate": state.reporting_rate, "seed": state.seed}

class SEIRAdapter(SIRAdapter):
    def __init__(self, model_id: str = "seir_tau", family: str = "mechanistic", *, population: float = 1_000_000.0, beta: float = 0.60, sigma: float = 0.50, gamma: float = 0.33, reporting_rate: float = 0.04, weekly_observation_scale: float = 1.0) -> None:
        super().__init__(model_id=model_id, family=family, population=population, beta=beta, gamma=gamma, reporting_rate=reporting_rate, weekly_observation_scale=weekly_observation_scale)
        self.sigma = float(sigma)

    def initialize(self, panel: pd.DataFrame, seed: int = 0) -> CompartmentState:
        return CompartmentState(_ensure_panel(panel), self.population, self.beta, self.gamma, self.sigma, self.reporting_rate, self.weekly_observation_scale, int(seed))

    def _forecast_one(self, state: CompartmentState, entity_id: str, component: str, origin: pd.Timestamp, horizon: int) -> tuple[float, float]:
        last_obs = _last_observed(state.panel, entity_id, component, origin)
        s, e, i, r = _initial_compartments(last_obs, state.population, state.reporting_rate)
        _s, _e, _i, _r, incidence = _simulate_mean_seir(s, e, i, r, beta=state.beta, sigma=float(state.sigma or self.sigma), gamma=state.gamma, population=state.population, horizon=int(horizon))
        mean = max(0.0, state.weekly_observation_scale * state.reporting_rate * incidence)
        var = max(mean, 1.0) + 0.07 * mean * mean
        return mean, var

    def serialize_state(self, state: CompartmentState) -> dict[str, object]:
        base = super().serialize_state(state)
        base["sigma"] = float(state.sigma or self.sigma)
        return base
