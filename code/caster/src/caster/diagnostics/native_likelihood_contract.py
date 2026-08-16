from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
import math

import pandas as pd


CONTRACT_COLUMNS = ["violation", "details"]

NATIVE_LIKELIHOOD_SCHEMA = {
    "supports_native_log_likelihood": "bool flag declaring whether adapter-native diagnostic log likelihood is supported.",
    "native_likelihood_type": "non-empty string naming the adapter-native observation/scoring distribution.",
    "native_sidecar_required": "bool flag declaring whether log_likelihood requires a sidecar artifact.",
    "native_sidecar_schema": "dict describing the sidecar fields needed to reproduce diagnostic scoring.",
    "save_native_state_sidecar": "callable that writes diagnostic-only native state sidecar artifacts.",
    "load_native_state_sidecar": "callable that loads diagnostic-only native state sidecar artifacts.",
    "log_likelihood": "callable with signature log_likelihood(y, context, sidecar) returning one finite numeric log score.",
}

REQUIRED_FIELDS = [
    "supports_native_log_likelihood",
    "native_likelihood_type",
    "native_sidecar_required",
    "native_sidecar_schema",
]

REQUIRED_METHODS = [
    "save_native_state_sidecar",
    "load_native_state_sidecar",
    "log_likelihood",
]

_DIAGNOSTIC_COUNT_ATTR = "_native_likelihood_diagnostic_context_depth"


class NativeLikelihoodContractError(RuntimeError):
    ""


class NativeLikelihoodDiagnosticMixin:
    ""






    supports_native_log_likelihood: bool = False
    native_likelihood_type: str = ""
    native_sidecar_required: bool = True
    native_sidecar_schema: dict[str, Any] = {}

    def _native_likelihood_diagnostic_enabled(self) -> bool:
        ""
        return int(getattr(self, _DIAGNOSTIC_COUNT_ATTR, 0)) > 0

    def _require_native_likelihood_diagnostic_enabled(self) -> None:
        ""
        if not self._native_likelihood_diagnostic_enabled():
            raise NativeLikelihoodContractError(
                "adapter-native log_likelihood is diagnostic-only; use native_likelihood_diagnostic_context"
            )

    def save_native_state_sidecar(self, *args: Any, **kwargs: Any) -> Any:
        ""
        raise NotImplementedError

    def load_native_state_sidecar(self, *args: Any, **kwargs: Any) -> Any:
        ""
        raise NotImplementedError

    def log_likelihood(self, y: Any, context: Any, sidecar: Any) -> float:
        ""
        self._require_native_likelihood_diagnostic_enabled()
        raise NotImplementedError


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def validate_native_likelihood_diagnostic_contract(adapter: object) -> pd.DataFrame:
    ""
    violations: list[dict[str, str]] = []

    for field in REQUIRED_FIELDS:
        if not hasattr(adapter, field):
            violations.append({"violation": f"missing_{field}", "details": adapter.__class__.__name__})

    supports = getattr(adapter, "supports_native_log_likelihood", False)
    if not _truthy(supports):
        violations.append(
            {
                "violation": "native_log_likelihood_not_supported",
                "details": "supports_native_log_likelihood must be true for diagnostic log_likelihood exposure",
            }
        )

    likelihood_type = str(getattr(adapter, "native_likelihood_type", "")).strip()
    if not likelihood_type:
        violations.append({"violation": "empty_native_likelihood_type", "details": adapter.__class__.__name__})

    if not isinstance(getattr(adapter, "native_sidecar_required", None), bool):
        violations.append({"violation": "invalid_native_sidecar_required", "details": "must be bool"})

    sidecar_schema = getattr(adapter, "native_sidecar_schema", None)
    if not isinstance(sidecar_schema, dict):
        violations.append({"violation": "invalid_native_sidecar_schema", "details": "must be dict"})

    for method in REQUIRED_METHODS:
        if not callable(getattr(adapter, method, None)):
            violations.append({"violation": f"missing_method_{method}", "details": adapter.__class__.__name__})

    return pd.DataFrame(violations, columns=CONTRACT_COLUMNS)


@contextmanager
def native_likelihood_diagnostic_context(adapter: object) -> Iterator[object]:
    ""
    previous = int(getattr(adapter, _DIAGNOSTIC_COUNT_ATTR, 0))
    setattr(adapter, _DIAGNOSTIC_COUNT_ATTR, previous + 1)
    try:
        yield adapter
    finally:
        if previous:
            setattr(adapter, _DIAGNOSTIC_COUNT_ATTR, previous)
        elif hasattr(adapter, _DIAGNOSTIC_COUNT_ATTR):
            delattr(adapter, _DIAGNOSTIC_COUNT_ATTR)


def call_native_log_likelihood(adapter: object, y: Any, context: Any, sidecar: Any) -> float:
    ""
    violations = validate_native_likelihood_diagnostic_contract(adapter)
    if not violations.empty:
        raise NativeLikelihoodContractError(f"invalid diagnostic native likelihood contract: {violations.to_dict(orient='records')}")
    with native_likelihood_diagnostic_context(adapter):
        value = adapter.log_likelihood(y, context, sidecar)
    try:
        score = float(value)
    except Exception as exc:
        raise NativeLikelihoodContractError("adapter.log_likelihood must return a numeric finite log score") from exc
    if not math.isfinite(score):
        raise NativeLikelihoodContractError("adapter.log_likelihood must return a finite log score")
    return score
