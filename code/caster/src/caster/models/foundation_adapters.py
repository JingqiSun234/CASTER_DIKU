""






from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from caster.forecast import FORECAST_ARCHIVE_COLUMNS, attach_ledger_context
from .base_adapter import BaseCandidateAdapter

EXTERNAL_FORECAST_REQUIRED = {"entity_id", "forecast_origin", "target_time", "component", "pred_mean"}


@dataclass
class FoundationState:
    panel: pd.DataFrame
    seed: int
    external_forecasts: pd.DataFrame | None


def _normalize_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["forecast_origin", "target_time", "generated_at", "features_available_until"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
    if "entity_id" in out.columns:
        out["entity_id"] = out["entity_id"].astype(str)
    if "component" in out.columns:
        out["component"] = out["component"].astype(str)
    if "forecast_id" in out.columns:
        out["forecast_id"] = out["forecast_id"].astype(str)
    return out


class ExternalForecastAdapter(BaseCandidateAdapter):
    ""






    def __init__(
        self,
        model_id: str = "external_foundation",
        family: str = "foundation",
        forecast_path: str | None = None,
        fallback: str = "last_value",
        pred_var: float = 0.0,
    ) -> None:
        self.model_id = str(model_id)
        self.family = str(family)
        self.forecast_path = forecast_path or ""
        self.fallback = str(fallback)
        self.pred_var = float(pred_var)

    def initialize(self, panel: pd.DataFrame, seed: int = 0) -> FoundationState:
        external: pd.DataFrame | None = None
        if self.forecast_path:
            path = Path(self.forecast_path)
            if not path.exists():
                raise FileNotFoundError(f"foundation forecast_path not found: {path}")
            external = pd.read_csv(path)
            required = {"pred_mean"}
            if "forecast_id" not in external.columns:
                required |= EXTERNAL_FORECAST_REQUIRED - {"pred_mean"}
            missing = sorted(required - set(external.columns))
            if missing:
                raise ValueError(f"external forecast file missing columns: {missing}")
            external = _normalize_time_columns(external)
        return FoundationState(panel=panel.copy(), seed=int(seed), external_forecasts=external)

    def transition(self, state: FoundationState, forecast_origin: pd.Timestamp) -> FoundationState:
        return state

    def _from_external(self, state: FoundationState, ledger: pd.DataFrame) -> pd.DataFrame:
        assert state.external_forecasts is not None
        ledger_norm = _normalize_time_columns(ledger)
        ext = state.external_forecasts.copy()
        if "forecast_id" in ledger_norm.columns:
            ledger_norm["forecast_id"] = ledger_norm["forecast_id"].astype(str)
        if "forecast_id" in ext.columns:
            ext["forecast_id"] = ext["forecast_id"].astype(str)

        provenance_cols = ["generated_at", "features_available_until"]
        missing_provenance = [col for col in provenance_cols if col not in ext.columns]
        if missing_provenance:
            raise ValueError(
                "external forecast artifact lacks required as-of provenance columns: "
                f"{missing_provenance}"
            )
        optional_cols = [c for c in ["pred_var"] if c in ext.columns]
        payload_cols = [*optional_cols, *provenance_cols]
        rename = {"pred_mean": "__external_pred_mean"}
        rename.update({c: f"__external_{c}" for c in payload_cols})

        if "forecast_id" in ext.columns and "forecast_id" in ledger_norm.columns:
            ledger_ids = set(ledger_norm["forecast_id"].astype(str))
            scoped = ext[ext["forecast_id"].astype(str).isin(ledger_ids)].copy()
            duplicate_ids = scoped.loc[scoped["forecast_id"].duplicated(), "forecast_id"].astype(str).head(10).tolist()
            if duplicate_ids:
                raise ValueError(f"external forecast file has duplicate forecast_id rows: {duplicate_ids}")
            missing_ids = sorted(ledger_ids - set(scoped["forecast_id"].astype(str)))
            if missing_ids:
                raise ValueError(f"external forecasts missing {len(missing_ids)} ledger forecast_id rows; examples={missing_ids[:5]}")
            ext_payload = scoped[["forecast_id", "pred_mean", *payload_cols]].rename(columns=rename)
            merged = ledger_norm.merge(ext_payload, on="forecast_id", how="left")
            merge_cols = ["forecast_id"]
        else:
            merge_cols = ["entity_id", "forecast_origin", "target_time", "component"]
            for col in ["dataset", "horizon", "mode", "mode_kind", "revision_version"]:
                if col in ledger_norm.columns and col in ext.columns:
                    merge_cols.append(col)
            ext_payload = ext[merge_cols + ["pred_mean", *payload_cols]].rename(columns=rename)
            merged = ledger_norm.merge(ext_payload, on=merge_cols, how="left")
            if len(merged) != len(ledger_norm):
                raise ValueError(
                    "external forecast merge produced duplicate ledger rows; "
                    "provide forecast_id in the external forecast CSV"
                )
        missing_pred = merged["__external_pred_mean"].isna()
        if missing_pred.any():
            examples = merged.loc[missing_pred, merge_cols].head(5).to_dict(orient="records")
            raise ValueError(f"external forecasts missing {int(missing_pred.sum())} ledger rows; examples={examples}")
        if "__external_pred_var" not in merged.columns:
            merged["__external_pred_var"] = self.pred_var
        merged["__external_pred_var"] = (
            pd.to_numeric(merged["__external_pred_var"], errors="coerce").fillna(self.pred_var).astype(float).clip(lower=0.0)
        )
        origin_time = pd.to_datetime(merged["forecast_origin"], errors="raise")
        provenance_times: dict[str, pd.Series] = {}
        for external_col in (
            "__external_generated_at",
            "__external_features_available_until",
        ):
            parsed = pd.to_datetime(merged[external_col], errors="coerce")
            provenance_times[external_col] = parsed
            if parsed.isna().any():
                raise ValueError(
                    f"external forecast artifact has missing or invalid {external_col[11:]}"
                )
            after_origin = parsed > origin_time
            if after_origin.any():
                raise ValueError(
                    f"external forecast artifact has {int(after_origin.sum())} "
                    f"{external_col[11:]} values after forecast_origin"
                )
        features_after_generation = (
            provenance_times["__external_features_available_until"]
            > provenance_times["__external_generated_at"]
        )
        if features_after_generation.any():
            raise ValueError(
                "external forecast artifact has "
                f"{int(features_after_generation.sum())} features_available_until "
                "values after generated_at"
            )
        out = pd.DataFrame({
            "dataset": merged["dataset"],
            "model_id": self.model_id,
            "family": self.family,
            "particle_id": 0,
            "entity_id": merged["entity_id"],
            "forecast_origin": merged["forecast_origin"],
            "target_time": merged["target_time"],
            "component": merged["component"],
            "horizon": merged["horizon"].astype(int),
            "forecast_id": merged["forecast_id"],
            "pred_mean": pd.to_numeric(merged["__external_pred_mean"], errors="coerce").astype(float).clip(lower=0.0),
            "pred_var": merged["__external_pred_var"],
            "generated_at": merged["__external_generated_at"],
            "features_available_until": merged["__external_features_available_until"],
        })
        return attach_ledger_context(out, ledger)[FORECAST_ARCHIVE_COLUMNS]

    def forecast_ledger(self, state: FoundationState, ledger: pd.DataFrame) -> pd.DataFrame:
        if state.external_forecasts is not None:
            return self._from_external(state, ledger)
        raise RuntimeError(
            f"{self.model_id} requires an explicit external forecast artifact; "
            "silent last-value substitution is forbidden"
        )

    def forecast_draws(self, state: FoundationState, ledger: pd.DataFrame, n_draws: int, seed: int = 0) -> pd.DataFrame:
        archive = self.forecast_ledger(state, ledger)
        rng = np.random.default_rng(seed)
        rows: list[dict[str, Any]] = []
        for _, r in archive.iterrows():
            scale = float(np.sqrt(max(float(r.get("pred_var", 0.0)), 0.0)))
            if scale <= 0.0:
                scale = max(1.0, 0.05 * (1.0 + float(r["pred_mean"])))
            for draw_id, draw in enumerate(rng.normal(float(r["pred_mean"]), scale, size=n_draws)):
                rows.append({"forecast_id": r["forecast_id"], "model_id": self.model_id, "particle_id": 0, "draw_id": int(draw_id), "draw": max(float(draw), 0.0)})
        return pd.DataFrame(rows)

    def serialize_state(self, state: FoundationState) -> dict[str, Any]:
        return {"seed": state.seed, "uses_external_forecast_file": state.external_forecasts is not None, "forecast_path": self.forecast_path, "fallback": self.fallback}


class ChronosForecastAdapter(ExternalForecastAdapter):
    def __init__(self, forecast_path: str | None = None, model_id: str = "chronos_external", pred_var: float = 0.0) -> None:
        super().__init__(model_id=model_id, family="foundation", forecast_path=forecast_path, pred_var=pred_var)


class TimesFMForecastAdapter(ExternalForecastAdapter):
    def __init__(self, forecast_path: str | None = None, model_id: str = "timesfm_external", pred_var: float = 0.0) -> None:
        super().__init__(model_id=model_id, family="foundation", forecast_path=forecast_path, pred_var=pred_var)


class TimeGPTForecastAdapter(ExternalForecastAdapter):
    def __init__(self, forecast_path: str | None = None, model_id: str = "timegpt_external", pred_var: float = 0.0) -> None:
        super().__init__(model_id=model_id, family="foundation", forecast_path=forecast_path, pred_var=pred_var)
