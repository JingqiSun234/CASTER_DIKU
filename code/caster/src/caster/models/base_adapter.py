from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
BANNED_LIKELIHOOD_NAMES = {"likelihood", "log_likelihood", "native_likelihood", "score_native_likelihood", "model_log_prob"}
class BaseCandidateAdapter(ABC):
    model_id: str
    family: str
    @abstractmethod
    def initialize(self, panel: pd.DataFrame, seed: int = 0) -> Any: ...
    @abstractmethod
    def transition(self, state: Any, forecast_origin: pd.Timestamp) -> Any: ...
    @abstractmethod
    def forecast_ledger(self, state: Any, ledger: pd.DataFrame) -> pd.DataFrame: ...
    @abstractmethod
    def forecast_draws(self, state: Any, ledger: pd.DataFrame, n_draws: int, seed: int = 0) -> pd.DataFrame: ...
    @abstractmethod
    def serialize_state(self, state: Any) -> dict[str, Any]: ...

def validate_adapter_contract(adapter: object, *, allow_diagnostic_native: bool = False) -> pd.DataFrame:
    violations = []
    diagnostic_native_valid = False
    if allow_diagnostic_native and callable(getattr(adapter, "log_likelihood", None)):
        from caster.diagnostics.native_likelihood_contract import validate_native_likelihood_diagnostic_contract

        diagnostic_violations = validate_native_likelihood_diagnostic_contract(adapter)
        if diagnostic_violations.empty:
            diagnostic_native_valid = True
        else:
            violations.extend(diagnostic_violations.to_dict(orient="records"))
    for attr in ["model_id", "family"]:
        if not hasattr(adapter, attr) or str(getattr(adapter, attr)).strip() == "":
            violations.append({"violation": f"missing_{attr}", "details": adapter.__class__.__name__})
    for method in ["initialize", "transition", "forecast_ledger", "forecast_draws", "serialize_state"]:
        if not callable(getattr(adapter, method, None)):
            violations.append({"violation": f"missing_method_{method}", "details": adapter.__class__.__name__})
    for name in BANNED_LIKELIHOOD_NAMES:
        if callable(getattr(adapter, name, None)):
            if name == "log_likelihood" and diagnostic_native_valid:
                continue
            violations.append({"violation": "native_likelihood_exposed", "details": name})
    return pd.DataFrame(violations, columns=["violation", "details"])
