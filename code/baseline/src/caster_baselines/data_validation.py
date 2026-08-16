from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


NA = "NA"
DATE_COLUMNS = ("forecast_origin", "target_time", "release_time")
ENTITY_CANDIDATES = ("entity_id", "jurisdiction", "location", "node_id", "node_index")
TIME_CANDIDATES = ("date", "week_end", "time", "target_time", "ds")


class DataValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LedgerValidationResult:
    dataset_key: str
    dataset: str
    row_count: int
    failures: list[str]
    warnings: list[str]
    key_columns: list[str]
    split_counts: dict[str, int]
    horizon_counts: dict[str, int]
    component_counts: dict[str, int]
    cadence_days: int
    join_checked_rows: int

    @property
    def passed(self) -> bool:
        return not self.failures


def baseline_root() -> Path:
    return Path(__file__).resolve().parents[2]


def caster_root_from_baseline(root: Path | None = None) -> Path:
    root = root or baseline_root()
    return root.resolve().parents[1]


def default_data_root(root: Path | None = None) -> Path:
    return caster_root_from_baseline(root) / "data"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_to(path: Path, root: Path) -> str:
    path = path.resolve()
    root = root.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def split_values(values: pd.Series | list[object]) -> str:
    if isinstance(values, pd.Series):
        vals = values.dropna().astype(str)
    else:
        vals = pd.Series(values, dtype="object").dropna().astype(str)
    vals = vals[vals.str.len() > 0]
    if vals.empty:
        return NA
    return ";".join(sorted(vals.unique().tolist()))


def parse_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_curated_subsets(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.glob("*/curated_subset") if p.is_dir())


def find_panel_file(curated_dir: Path) -> Path:
    candidates = [
        p
        for p in curated_dir.glob("*.csv")
        if "panel" in p.name.lower() and "manifest" not in p.name.lower()
    ]
    if not candidates:
        raise DataValidationError(f"no panel CSV found in {curated_dir}")
    return sorted(candidates, key=lambda p: (("daily" not in p.name.lower()) and ("weekly" not in p.name.lower()), p.name))[0]


def find_time_col(df: pd.DataFrame) -> str:
    for col in TIME_CANDIDATES:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                return col
    raise DataValidationError("could not infer panel time column")


def find_panel_entity_col(panel: pd.DataFrame, ledger: pd.DataFrame) -> str:
    ledger_entities = None
    if "entity_id" in ledger.columns:
        ledger_entities = set(ledger["entity_id"].dropna().astype(str).unique())
    for col in ENTITY_CANDIDATES:
        if col not in panel.columns:
            continue
        if ledger_entities is None:
            return col
        panel_values = set(panel[col].dropna().astype(str).unique())
        if panel_values & ledger_entities:
            return col
    for col in ENTITY_CANDIDATES:
        if col in panel.columns:
            return col
    raise DataValidationError("could not infer panel entity column")


def infer_panel_shape(panel: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, str]:
    time_col = find_time_col(panel)
    entity_col = find_panel_entity_col(panel, ledger)
    if {"component", "observed_value"}.issubset(panel.columns):
        return {
            "panel_format": "long",
            "panel_entity_col": entity_col,
            "panel_time_col": time_col,
            "panel_component_col": "component",
            "panel_value_col": "observed_value",
            "panel_target_cols": split_values(panel["component"]),
        }
    components = ledger["component"].dropna().astype(str).unique().tolist() if "component" in ledger.columns else []
    target_cols = [c for c in components if c in panel.columns]
    if not target_cols:
        numeric_cols = [
            c
            for c in panel.columns
            if c not in {entity_col, time_col} and pd.api.types.is_numeric_dtype(pd.to_numeric(panel[c], errors="coerce"))
        ]
        target_cols = numeric_cols
    return {
        "panel_format": "wide",
        "panel_entity_col": entity_col,
        "panel_time_col": time_col,
        "panel_component_col": NA,
        "panel_value_col": NA,
        "panel_target_cols": split_values(target_cols),
    }


def infer_cadence_days(panel: pd.DataFrame, time_col: str, ledger: pd.DataFrame) -> int:
    panel_dates = pd.to_datetime(panel[time_col], errors="coerce").dropna().sort_values().unique()
    if len(panel_dates) > 1:
        diffs = pd.Series(panel_dates).diff().dropna().dt.days
        positive = diffs[diffs > 0]
        if not positive.empty:
            median = int(round(float(positive.median())))
            if 6 <= median <= 8:
                return 7
            if 1 <= median <= 2:
                return 1
            return max(median, 1)
    if {"forecast_origin", "target_time", "horizon"}.issubset(ledger.columns):
        origin = pd.to_datetime(ledger["forecast_origin"], errors="coerce")
        target = pd.to_datetime(ledger["target_time"], errors="coerce")
        horizon = pd.to_numeric(ledger["horizon"], errors="coerce")
        ratio = ((target - origin).dt.days / horizon).replace([np.inf, -np.inf], np.nan).dropna()
        if not ratio.empty:
            median = int(round(float(ratio.median())))
            if 6 <= median <= 8:
                return 7
            return max(median, 1)
    return 1


def infer_scope(panel: pd.DataFrame, ledger: pd.DataFrame, run_manifest: dict) -> str:
    parts: list[str] = []
    if "country" in panel.columns:
        countries = sorted(panel["country"].dropna().astype(str).unique().tolist())
        if len(countries) == 1:
            parts.append(f"{countries[0]}-only curated subset")
        elif countries:
            parts.append(f"{len(countries)} countries: {';'.join(countries)}")
    if "jurisdiction" in panel.columns:
        jurisdictions = sorted(panel["jurisdiction"].dropna().astype(str).unique().tolist())
        if jurisdictions:
            parts.append(f"{len(jurisdictions)} jurisdictions: {';'.join(jurisdictions)}")
    if "scope" in run_manifest and str(run_manifest["scope"]).strip():
        parts.append(f"manifest_scope={run_manifest['scope']}")
    if "mode" in ledger.columns:
        modes = split_values(ledger["mode"])
        if modes != NA:
            parts.append(f"modes={modes}")
    if not parts:
        parts.append("curated subset")
    return " | ".join(parts)


def value_counts_string(series: pd.Series, sort_numeric: bool = False) -> str:
    counts = series.fillna(NA).astype(str).value_counts().to_dict()
    if not counts:
        return NA
    def key_fn(item: tuple[str, int]) -> tuple[int, object]:
        key = item[0]
        if sort_numeric:
            try:
                return (0, int(float(key)))
            except ValueError:
                return (1, key)
        return (0, key)
    return ";".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=key_fn))


def date_min_max(series: pd.Series) -> tuple[str, str]:
    parsed = pd.to_datetime(series, errors="coerce").dropna()
    if parsed.empty:
        return NA, NA
    return parsed.min().strftime("%Y-%m-%d"), parsed.max().strftime("%Y-%m-%d")


def build_subset_manifest(data_root: Path | None = None, caster_root: Path | None = None) -> list[dict[str, object]]:
    root = baseline_root()
    caster_root = caster_root or caster_root_from_baseline(root)
    data_root = data_root or default_data_root(root)
    rows: list[dict[str, object]] = []
    for curated_dir in discover_curated_subsets(data_root):
        ledger_path = curated_dir / "event_ledger.csv"
        if not ledger_path.exists():
            continue
        panel_path = find_panel_file(curated_dir)
        panel = pd.read_csv(panel_path)
        ledger = pd.read_csv(ledger_path)
        run_manifest = parse_json_if_exists(curated_dir / "run_manifest.json")
        shape = infer_panel_shape(panel, ledger)
        cadence = infer_cadence_days(panel, shape["panel_time_col"], ledger)
        panel_date_min, panel_date_max = date_min_max(panel[shape["panel_time_col"]])
        ledger_origin_min, ledger_origin_max = date_min_max(ledger["forecast_origin"]) if "forecast_origin" in ledger.columns else (NA, NA)
        ledger_target_min, ledger_target_max = date_min_max(ledger["target_time"]) if "target_time" in ledger.columns else (NA, NA)
        dataset = split_values(ledger["dataset"]) if "dataset" in ledger.columns else curated_dir.parent.name
        row = {
            "dataset_key": curated_dir.parent.name,
            "dataset": dataset,
            "scope": infer_scope(panel, ledger, run_manifest),
            "curated_subset_dir": rel_to(curated_dir, caster_root),
            "panel_path": rel_to(panel_path, caster_root),
            "ledger_path": rel_to(ledger_path, caster_root),
            "panel_format": shape["panel_format"],
            "panel_entity_col": shape["panel_entity_col"],
            "panel_time_col": shape["panel_time_col"],
            "panel_component_col": shape["panel_component_col"],
            "panel_value_col": shape["panel_value_col"],
            "panel_target_cols": shape["panel_target_cols"],
            "panel_rows": int(len(panel)),
            "ledger_rows": int(len(ledger)),
            "panel_sha256": sha256_file(panel_path),
            "ledger_sha256": sha256_file(ledger_path),
            "entity_count": int(panel[shape["panel_entity_col"]].dropna().astype(str).nunique()),
            "entities": split_values(panel[shape["panel_entity_col"]]),
            "countries": split_values(panel["country"]) if "country" in panel.columns else NA,
            "jurisdictions": split_values(panel["jurisdiction"]) if "jurisdiction" in panel.columns else NA,
            "components": split_values(ledger["component"]) if "component" in ledger.columns else NA,
            "horizons": split_values(ledger["horizon"]) if "horizon" in ledger.columns else NA,
            "splits": split_values(ledger["split"]) if "split" in ledger.columns else NA,
            "modes": split_values(ledger["mode"]) if "mode" in ledger.columns else NA,
            "split_counts": value_counts_string(ledger["split"]) if "split" in ledger.columns else NA,
            "horizon_counts": value_counts_string(ledger["horizon"], sort_numeric=True) if "horizon" in ledger.columns else NA,
            "component_counts": value_counts_string(ledger["component"]) if "component" in ledger.columns else NA,
            "panel_date_min": panel_date_min,
            "panel_date_max": panel_date_max,
            "ledger_origin_min": ledger_origin_min,
            "ledger_origin_max": ledger_origin_max,
            "ledger_target_min": ledger_target_min,
            "ledger_target_max": ledger_target_max,
            "cadence_days": cadence,
        }
        empty = [k for k, v in row.items() if v is None or str(v) == ""]
        if empty:
            raise DataValidationError(f"manifest row for {curated_dir} has empty required fields: {empty}")
        rows.append(row)
    if not rows:
        raise DataValidationError(f"no curated subset ledgers found under {data_root}")
    return rows


def write_subset_manifest(out_path: Path, data_root: Path | None = None, caster_root: Path | None = None) -> list[dict[str, object]]:
    rows = build_subset_manifest(data_root=data_root, caster_root=caster_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return rows


def is_na(value: object) -> bool:
    return value is None or str(value) in {"", NA, "nan", "NaN"}


def resolve_manifest_path(value: object, caster_root: Path, root: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [caster_root / path, root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def bool_series(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def add_failure(failures: list[str], name: str, count: int) -> None:
    if count:
        failures.append(f"{name}: {count}")


def choose_key_columns(ledger: pd.DataFrame) -> list[str]:
    for col in ("event_id", "row_id"):
        if col in ledger.columns and ledger[col].notna().all() and ledger[col].astype(str).str.len().gt(0).all():
            return [col]
    key_cols = [c for c in ("dataset", "split", "mode", "entity_id", "jurisdiction", "node_index", "component", "horizon", "forecast_origin", "target_time", "release_time") if c in ledger.columns]
    missing = [c for c in ("component", "horizon", "forecast_origin", "target_time", "release_time") if c not in key_cols]
    if missing:
        raise DataValidationError(f"ledger missing columns required for composite key: {missing}")
    if not any(c in key_cols for c in ("entity_id", "jurisdiction", "node_index")):
        raise DataValidationError("ledger missing entity column for composite key")
    return key_cols


def validate_date_columns(ledger: pd.DataFrame, failures: list[str]) -> dict[str, pd.Series]:
    parsed: dict[str, pd.Series] = {}
    for col in DATE_COLUMNS:
        if col not in ledger.columns:
            add_failure(failures, f"missing date column {col}", 1)
            parsed[col] = pd.Series(pd.NaT, index=ledger.index)
            continue
        parsed[col] = pd.to_datetime(ledger[col], errors="coerce")
        add_failure(failures, f"{col} parse failures", int(parsed[col].isna().sum()))
    return parsed


def validate_panel_join(
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    manifest_row: pd.Series,
    target_time: pd.Series,
) -> tuple[int, int]:
    panel_format = str(manifest_row["panel_format"])
    entity_col = str(manifest_row["panel_entity_col"])
    time_col = str(manifest_row["panel_time_col"])
    panel_dates = pd.to_datetime(panel[time_col], errors="coerce")
    if panel_format == "long":
        component_col = str(manifest_row["panel_component_col"])
        if component_col not in panel.columns:
            return len(ledger), 0
        panel_keys = set(
            zip(
                panel[entity_col].astype(str),
                panel_dates.dt.strftime("%Y-%m-%d"),
                panel[component_col].astype(str),
            )
        )
        ledger_entity = ledger["entity_id"].astype(str) if "entity_id" in ledger.columns else ledger[entity_col].astype(str)
        ledger_keys = zip(
            ledger_entity,
            target_time.dt.strftime("%Y-%m-%d"),
            ledger["component"].astype(str),
        )
    else:
        panel_target_cols = set(str(manifest_row["panel_target_cols"]).split(";"))
        missing_components = int((~ledger["component"].astype(str).isin(panel_target_cols)).sum())
        panel_keys = set(zip(panel[entity_col].astype(str), panel_dates.dt.strftime("%Y-%m-%d")))
        if "entity_id" in ledger.columns:
            ledger_entity = ledger["entity_id"].astype(str)
        elif entity_col in ledger.columns:
            ledger_entity = ledger[entity_col].astype(str)
        else:
            return len(ledger), 0
        ledger_keys = zip(ledger_entity, target_time.dt.strftime("%Y-%m-%d"))
        if missing_components:
            return missing_components + sum(1 for key in ledger_keys if key not in panel_keys), len(ledger)
    missing = sum(1 for key in ledger_keys if key not in panel_keys)
    return missing, len(ledger)


def validation_one_ledger(manifest_row: pd.Series, caster_root: Path, root: Path) -> LedgerValidationResult:
    failures: list[str] = []
    warnings: list[str] = []
    panel_path = resolve_manifest_path(manifest_row["panel_path"], caster_root, root)
    ledger_path = resolve_manifest_path(manifest_row["ledger_path"], caster_root, root)
    panel = pd.read_csv(panel_path)
    ledger = pd.read_csv(ledger_path)
    expected_ledger_rows = int(manifest_row["ledger_rows"])
    expected_panel_rows = int(manifest_row["panel_rows"])
    add_failure(failures, "ledger row count mismatch", int(len(ledger) != expected_ledger_rows))
    add_failure(failures, "panel row count mismatch", int(len(panel) != expected_panel_rows))
    if sha256_file(ledger_path) != str(manifest_row["ledger_sha256"]):
        failures.append("ledger sha256 mismatch: 1")
    if sha256_file(panel_path) != str(manifest_row["panel_sha256"]):
        failures.append("panel sha256 mismatch: 1")
    for col in ("component", "horizon", "observed_mask", "observed_value"):
        if col not in ledger.columns:
            add_failure(failures, f"missing ledger column {col}", 1)
    key_columns = choose_key_columns(ledger)
    key_frame = ledger[key_columns].fillna(NA).astype(str)
    add_failure(failures, "duplicate event/composite keys", int(key_frame.duplicated().sum()))
    parsed_dates = validate_date_columns(ledger, failures)
    origin = parsed_dates["forecast_origin"]
    target = parsed_dates["target_time"]
    release = parsed_dates["release_time"]
    add_failure(failures, "forecast_origin >= target_time", int((origin >= target).fillna(False).sum()))
    add_failure(failures, "target_time > release_time", int((target > release).fillna(False).sum()))
    if "features_available_until" in ledger.columns:
        features = pd.to_datetime(ledger["features_available_until"], errors="coerce")
        add_failure(failures, "features_available_until parse failures", int(features.isna().sum()))
        add_failure(failures, "features_available_until > forecast_origin", int((features > origin).fillna(False).sum()))
    else:
        warnings.append("features_available_until missing")
    mask = bool_series(ledger["observed_mask"]) if "observed_mask" in ledger.columns else pd.Series(True, index=ledger.index)
    observed = pd.to_numeric(ledger["observed_value"], errors="coerce") if "observed_value" in ledger.columns else pd.Series(np.nan, index=ledger.index)
    add_failure(failures, "observed_mask true but observed_value missing/non-finite", int((mask & ~np.isfinite(observed)).sum()))
    warning_count = int((~mask & ~np.isfinite(observed)).sum())
    if warning_count:
        warnings.append(f"observed_mask false with missing observed_value: {warning_count}")
    horizon = pd.to_numeric(ledger["horizon"], errors="coerce")
    add_failure(failures, "horizon parse failures", int(horizon.isna().sum()))
    cadence = int(manifest_row["cadence_days"])
    delta_days = (target - origin).dt.days
    expected_delta = horizon * cadence
    add_failure(failures, "horizon/date cadence mismatches", int((delta_days != expected_delta).fillna(False).sum()))
    missing_join, checked_rows = validate_panel_join(panel, ledger, manifest_row, target)
    add_failure(failures, "ledger target/component cannot join panel", int(missing_join))
    return LedgerValidationResult(
        dataset_key=str(manifest_row["dataset_key"]),
        dataset=str(manifest_row["dataset"]),
        row_count=int(len(ledger)),
        failures=failures,
        warnings=warnings,
        key_columns=key_columns,
        split_counts=ledger["split"].fillna(NA).astype(str).value_counts().sort_index().to_dict() if "split" in ledger.columns else {},
        horizon_counts=ledger["horizon"].fillna(NA).astype(str).value_counts().sort_index().to_dict() if "horizon" in ledger.columns else {},
        component_counts=ledger["component"].fillna(NA).astype(str).value_counts().sort_index().to_dict() if "component" in ledger.columns else {},
        cadence_days=cadence,
        join_checked_rows=checked_rows,
    )


def render_event_ledger_report(results: list[LedgerValidationResult]) -> str:
    lines = [
        "# Event Ledger Validation",
        "",
        "| Dataset key | Dataset | Rows | Cadence days | Key columns | Status |",
        "|---|---|---:|---:|---|---|",
    ]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"| {result.dataset_key} | {result.dataset} | {result.row_count} | {result.cadence_days} | "
            f"{';'.join(result.key_columns)} | {status} |"
        )
    for result in results:
        lines.extend([
            "",
            f"## {result.dataset_key}",
            "",
            f"- dataset: `{result.dataset}`",
            f"- rows: `{result.row_count}`",
            f"- cadence_days: `{result.cadence_days}`",
            f"- join_checked_rows: `{result.join_checked_rows}`",
            f"- split_counts: `{json.dumps(result.split_counts, sort_keys=True)}`",
            f"- horizon_counts: `{json.dumps(result.horizon_counts, sort_keys=True)}`",
            f"- component_counts: `{json.dumps(result.component_counts, sort_keys=True)}`",
            f"- failures: `{json.dumps(result.failures, sort_keys=True)}`",
            f"- warnings: `{json.dumps(result.warnings, sort_keys=True)}`",
        ])
    lines.append("")
    return "\n".join(lines)


def validation_event_ledgers_from_manifest(
    manifest_path: Path,
    out_path: Path,
    root: Path | None = None,
    caster_root: Path | None = None,
    fail_on_error: bool = True,
) -> list[LedgerValidationResult]:
    root = root or baseline_root()
    caster_root = caster_root or caster_root_from_baseline(root)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    missing_required = [col for col in manifest.columns if manifest[col].isna().any() or manifest[col].astype(str).eq("").any()]
    if missing_required:
        raise DataValidationError(f"manifest has empty required fields: {missing_required}")
    results = [validation_one_ledger(row, caster_root, root) for _, row in manifest.iterrows()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_event_ledger_report(results), encoding="utf-8")
    failures = {r.dataset_key: r.failures for r in results if r.failures}
    if fail_on_error and failures:
        raise DataValidationError(f"event ledger validation failed: {json.dumps(failures, sort_keys=True)}")
    return results
