from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

FORECAST_DRAW_COLUMNS = [
    "dataset", "model_id", "family", "particle_id", "entity_id", "forecast_origin",
    "target_time", "component", "horizon", "forecast_id", "draw_id", "draw", "generated_at",
    "features_available_until",
]


def build_normal_forecast_draws(
    archive: pd.DataFrame,
    *,
    n_draws: int = 50,
    seed: int = 0,
    min_scale: float = 1e-6,
) -> pd.DataFrame:
    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    required = {"pred_mean", "pred_var", "forecast_id", "model_id", "particle_id"}
    if missing := sorted(required - set(archive.columns)):
        raise ValueError(f"forecast archive missing columns {missing}")
    rows: list[dict[str, object]] = []
    meta_cols = [c for c in FORECAST_DRAW_COLUMNS if c not in {"draw_id", "draw"}]
    for _, r in archive.iterrows():
        identity = "\x1f".join(
            [
                str(int(seed)),
                str(r["forecast_id"]),
                str(r["model_id"]),
                str(r["particle_id"]),
            ]
        ).encode("utf-8")
        row_seed = int.from_bytes(
            hashlib.sha256(identity).digest()[:8], byteorder="little", signed=False
        )
        rng = np.random.default_rng(row_seed)
        scale = float(np.sqrt(max(float(r.get("pred_var", 0.0)), 0.0) + min_scale * min_scale))
        draws = rng.normal(float(r["pred_mean"]), scale, size=n_draws)
        draws = np.maximum(draws, 0.0)
        base = {c: r.get(c) for c in meta_cols}
        for draw_id, val in enumerate(draws):
            row = dict(base)
            row["draw_id"] = int(draw_id)
            row["draw"] = float(val)
            rows.append(row)
    return pd.DataFrame(rows, columns=FORECAST_DRAW_COLUMNS)


def validate_forecast_draws(draws: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    violations: list[dict[str, object]] = []
    missing = sorted(set(FORECAST_DRAW_COLUMNS) - set(draws.columns))
    if missing:
        return pd.DataFrame([{"row": None, "violation": "missing_columns", "details": ",".join(missing)}])
    if draws.empty:
        return pd.DataFrame([{"row": None, "violation": "empty_draws", "details": "draw table has no rows"}])
    known = set(archive["forecast_id"].astype(str))
    bad = sorted(set(draws["forecast_id"].astype(str)) - known)
    if bad:
        violations.append({"row": None, "violation": "forecast_id_not_in_archive", "details": ",".join(bad[:10])})
    dup = draws.duplicated(["forecast_id", "model_id", "particle_id", "draw_id"])
    if dup.any():
        ids = draws.loc[dup, ["forecast_id", "model_id", "particle_id", "draw_id"]].astype(str).agg("|".join, axis=1).unique()[:10]
        violations.append({"row": None, "violation": "duplicate_draw", "details": ",".join(ids)})
    vals = pd.to_numeric(draws["draw"], errors="coerce")
    if not np.isfinite(vals).all():
        violations.append({"row": None, "violation": "nonfinite_draw", "details": "draw values must be finite"})
    return pd.DataFrame(violations, columns=["row", "violation", "details"])


def validate_draw_kernel_inputs(
    draws: pd.DataFrame,
    update_ledger: pd.DataFrame,
    archive: pd.DataFrame,
    selected_model_ids: list[str],
) -> pd.DataFrame:
    ""
    violations: list[dict[str, object]] = []
    required_draws = {"forecast_id", "model_id", "particle_id", "draw_id", "draw"}
    required_ledger = {"forecast_id", "forecast_origin"}
    required_archive = {"forecast_id", "model_id", "particle_id"}
    if missing := sorted(required_draws - set(draws.columns)):
        return pd.DataFrame([{"row": None, "violation": "missing_draw_columns", "details": ",".join(missing)}])
    if missing := sorted(required_ledger - set(update_ledger.columns)):
        return pd.DataFrame([{"row": None, "violation": "missing_ledger_columns", "details": ",".join(missing)}])
    if missing := sorted(required_archive - set(archive.columns)):
        return pd.DataFrame([{"row": None, "violation": "missing_archive_columns", "details": ",".join(missing)}])
    if draws.empty:
        return pd.DataFrame([{"row": None, "violation": "empty_draws", "details": "draw table has no rows"}])

    selected = {str(x) for x in selected_model_ids}
    draw_scope = draws[draws["model_id"].astype(str).isin(selected)].copy()
    update_ids = set(update_ledger["forecast_id"].astype(str))
    draw_scope = draw_scope[draw_scope["forecast_id"].astype(str).isin(update_ids)].copy()
    if draw_scope.empty:
        violations.append({"row": None, "violation": "no_selected_draw_rows", "details": "no draw rows after selected model/update forecast filter"})

    dup = draw_scope.duplicated(["forecast_id", "model_id", "particle_id", "draw_id"])
    if dup.any():
        ids = draw_scope.loc[dup, ["forecast_id", "model_id", "particle_id", "draw_id"]].astype(str).agg("|".join, axis=1).unique()[:10]
        violations.append({"row": None, "violation": "duplicate_draw_id", "details": ",".join(ids)})

    vals = pd.to_numeric(draw_scope["draw"], errors="coerce")
    if not np.isfinite(vals).all():
        violations.append({"row": None, "violation": "nonfinite_draw", "details": "draw values must be finite"})

    selected_archive = archive[archive["model_id"].astype(str).isin(selected)].copy()
    selected_archive = selected_archive[selected_archive["forecast_id"].astype(str).isin(update_ids)].copy()
    expected_keys = set(
        selected_archive[["forecast_id", "model_id", "particle_id"]]
        .astype(str)
        .agg("|".join, axis=1)
        .tolist()
    )
    observed_keys = set(
        draw_scope[["forecast_id", "model_id", "particle_id"]]
        .astype(str)
        .drop_duplicates()
        .agg("|".join, axis=1)
        .tolist()
    )
    missing_keys = sorted(expected_keys - observed_keys)
    if missing_keys:
        violations.append({"row": None, "violation": "missing_selected_archive_draw_coverage", "details": ",".join(missing_keys[:10])})

    for model_id in sorted(selected):
        model_forecasts = set(selected_archive.loc[selected_archive["model_id"].astype(str) == model_id, "forecast_id"].astype(str))
        draw_forecasts = set(draw_scope.loc[draw_scope["model_id"].astype(str) == model_id, "forecast_id"].astype(str))
        missing_forecasts = sorted(model_forecasts - draw_forecasts)
        if missing_forecasts:
            violations.append({"row": None, "violation": "missing_selected_model_forecast_coverage", "details": f"{model_id}:{','.join(missing_forecasts[:10])}"})

    if {"generated_at", "features_available_until"} <= set(draw_scope.columns):
        meta = update_ledger[["forecast_id", "forecast_origin"]].drop_duplicates("forecast_id").copy()
        joined = draw_scope.merge(meta, on="forecast_id", how="left", suffixes=("_draw", "_ledger"))
        origin_col = "forecast_origin_ledger" if "forecast_origin_ledger" in joined.columns else "forecast_origin"
        origin = pd.to_datetime(joined[origin_col], errors="coerce")
        generated = pd.to_datetime(joined["generated_at"], errors="coerce")
        features = pd.to_datetime(joined["features_available_until"], errors="coerce")
        if (generated > origin).any():
            violations.append({"row": None, "violation": "generated_at_after_forecast_origin", "details": "generated_at must be <= forecast_origin"})
        if (features > origin).any():
            violations.append({"row": None, "violation": "features_available_until_after_forecast_origin", "details": "features_available_until must be <= forecast_origin"})

    return pd.DataFrame(violations, columns=["row", "violation", "details"])


def write_forecast_draws(draws: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            draws.to_parquet(path, index=False)
            return path
        except Exception:
            path = path.with_suffix(".csv")
    draws.to_csv(path, index=False)
    return path
