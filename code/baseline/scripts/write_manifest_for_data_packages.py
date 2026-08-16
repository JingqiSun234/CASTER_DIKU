#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caster_baselines.data_validation import (              
    caster_root_from_baseline,
    date_min_max,
    find_panel_file,
    infer_cadence_days,
    infer_panel_shape,
    infer_scope,
    parse_json_if_exists,
    rel_to,
    sha256_file,
    split_values,
    value_counts_string,
)


BENCHMARK_B_V26_PROTOCOL = "caster_data_protocol_v26_1"
BENCHMARK_B_POOLED_TASK = "benchmark_b_pooled"
BENCHMARK_B_REQUIRED_RELEASE_COLUMNS = {
    "__release_time__autoregressive",
    "__release_time__calendar",
    "__release_time__ed",
    "__release_time__target",
    "__release_time__wastewater",
}


def parse_package_dirs(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def _is_benchmark_b(ledger: pd.DataFrame) -> bool:
    return set(ledger.get("dataset", pd.Series(dtype=str)).astype(str)) == {"benchmark_b"}


def _validate_benchmark_b_v26_package(
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    run_manifest: dict[str, object],
) -> None:
    if not _is_benchmark_b(ledger):
        return
    protocol = set(ledger.get("protocol_version", pd.Series(dtype=str)).astype(str))
    tasks = set(ledger.get("task_id", pd.Series(dtype=str)).astype(str))
    if protocol != {BENCHMARK_B_V26_PROTOCOL} or tasks != {BENCHMARK_B_POOLED_TASK}:
        raise ValueError(
            "Benchmark B manifest input must be the v26.1 pooled package; "
            f"protocol={sorted(protocol)} task_id={sorted(tasks)}"
        )
    missing_release = sorted(BENCHMARK_B_REQUIRED_RELEASE_COLUMNS - set(panel.columns))
    if missing_release or len(panel.columns) != 105:
        raise ValueError(
            "Benchmark B v26.1 pooled panel must contain 105 columns and all stream release columns; "
            f"columns={len(panel.columns)} missing_release={missing_release}"
        )
    if run_manifest.get("task_id") != BENCHMARK_B_POOLED_TASK:
        raise ValueError("Benchmark B package run_manifest task_id mismatch")
    if run_manifest.get("protocol_version") != BENCHMARK_B_V26_PROTOCOL:
        raise ValueError("Benchmark B package run_manifest protocol_version mismatch")


def build_manifest_row(package_dir: Path, caster_root: Path, root: Path) -> dict[str, object]:
    ledger_path = package_dir / "event_ledger.csv"
    if not ledger_path.exists():
        raise FileNotFoundError(f"event ledger missing: {ledger_path}")
    panel_path = find_panel_file(package_dir)
    panel = pd.read_csv(panel_path)
    ledger = pd.read_csv(ledger_path)
    run_manifest = parse_json_if_exists(package_dir / "run_manifest.json")
    _validate_benchmark_b_v26_package(panel, ledger, run_manifest)
    shape = infer_panel_shape(panel, ledger)
    cadence = infer_cadence_days(panel, shape["panel_time_col"], ledger)
    panel_date_min, panel_date_max = date_min_max(panel[shape["panel_time_col"]])
    ledger_origin_min, ledger_origin_max = date_min_max(ledger["forecast_origin"])
    ledger_target_min, ledger_target_max = date_min_max(ledger["target_time"])
    benchmark_b_v26 = _is_benchmark_b(ledger)
    dataset_key = "benchmark_b" if benchmark_b_v26 else package_dir.parent.name
    scope_manifest = dict(run_manifest)
    if benchmark_b_v26:
        scope_manifest["scope"] = "benchmark_b_pooled_v26_1"
    row = {
        "dataset_key": dataset_key,
        "dataset": split_values(ledger["dataset"]) if "dataset" in ledger.columns else dataset_key,
        "scope": infer_scope(panel, ledger, scope_manifest),
        "curated_subset_dir": rel_to(package_dir, caster_root),
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
        "countries": split_values(panel["country"]) if "country" in panel.columns else "NA",
        "jurisdictions": split_values(panel["jurisdiction"]) if "jurisdiction" in panel.columns else "NA",
        "components": split_values(ledger["component"]) if "component" in ledger.columns else "NA",
        "horizons": split_values(ledger["horizon"]) if "horizon" in ledger.columns else "NA",
        "splits": split_values(ledger["split"]) if "split" in ledger.columns else "NA",
        "modes": split_values(ledger["mode"]) if "mode" in ledger.columns else "NA",
        "split_counts": value_counts_string(ledger["split"]) if "split" in ledger.columns else "NA",
        "horizon_counts": value_counts_string(ledger["horizon"], sort_numeric=True) if "horizon" in ledger.columns else "NA",
        "component_counts": value_counts_string(ledger["component"]) if "component" in ledger.columns else "NA",
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
        raise ValueError(f"manifest row for {package_dir} has empty fields: {empty}")
    return row


def write_manifest(package_dirs: list[Path], out_path: Path, root: Path = ROOT) -> pd.DataFrame:
    caster_root = caster_root_from_baseline(root)
    rows = [build_manifest_row(path, caster_root, root) for path in package_dirs]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a manifest for explicit panel/event-ledger data packages.")
    parser.add_argument("--package-dirs", required=True, help="Comma-separated package directories containing panel CSV and event_ledger.csv.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    frame = write_manifest(parse_package_dirs(args.package_dirs), Path(args.out))
    print(f"ok out={args.out} datasets={','.join(frame['dataset_key'].astype(str))} rows={len(frame)}")


if __name__ == "__main__":
    main()
