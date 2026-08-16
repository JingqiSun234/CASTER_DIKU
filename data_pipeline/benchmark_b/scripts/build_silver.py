import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


NHSN_TARGET_COLUMNS = {
    "totalconfc19newadm": "covid_adm_count",
    "totalconfc19newadmper100k": "covid_adm_per100k",
    "totalconfflunewadm": "flu_adm_count",
    "totalconfflunewadmper100k": "flu_adm_per100k",
    "totalconfrsvnewadm": "rsv_adm_count",
    "totalconfrsvnewadmper100k": "rsv_adm_per100k",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build silver source tables from bronze parquet files.")
    parser.add_argument("--benchmark-spec", required=True, help="Benchmark spec YAML")
    parser.add_argument("--bronze-root", default="data_intermediate/bronze", help="Bronze input directory")
    parser.add_argument("--out", default="data_intermediate/silver", help="Silver output directory")
    parser.add_argument("--jurisdiction-map", default="reference/jurisdiction_map.csv", help="Canonical jurisdiction CSV")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_jurisdiction_map(path: Path) -> tuple[dict[str, str], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    abbr_to_name = {row["abbr"]: row["jurisdiction"] for row in rows}
    jurisdictions = [row["jurisdiction"] for row in rows]
    if len(jurisdictions) != 51:
        raise ValueError(f"expected 51 canonical jurisdictions, got {len(jurisdictions)}")
    return abbr_to_name, jurisdictions


def canonical_weeks(start: str, end: str) -> list[str]:
    current = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    weeks = []
    while current <= stop:
        weeks.append(current.isoformat())
        current += timedelta(days=7)
    if len(weeks) != 183:
        raise ValueError(f"expected 183 canonical weeks, got {len(weeks)}")
    return weeks


def parse_date_string(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return value[:10]


def to_float_or_null(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    parsed = float(value)
    if math.isnan(parsed):
        return None
    return parsed


def normalize_feature_component(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("<", "lt_").replace(">", "gt_").replace("+", "_plus")
    value = re.sub(r"[^0-9a-z]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def nssp_feature_name(row: dict) -> str:
    return (
        "nssp_"
        f"{normalize_feature_component(row['pathogen'])}__"
        f"{normalize_feature_component(row['demographics_type'])}__"
        f"{normalize_feature_component(row['demographics_values'])}_pct"
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_table_with_metadata(table: pa.Table, parquet_path: Path, metadata_path: Path, metadata: dict) -> dict:
    pq.write_table(table, parquet_path, compression="zstd")
    payload = {
        **metadata,
        "output_path": str(parquet_path),
        "row_count": table.num_rows,
        "column_count": table.num_columns,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "no_final_panel_join": True,
        "no_feature_engineering": True,
        "no_sample_generation": True,
    }
    write_json(metadata_path, payload)
    print(f"OK: wrote silver {parquet_path} rows={table.num_rows}")
    return payload


def table_from_columns(columns: dict[str, list], schema: pa.schema) -> pa.Table:
    return pa.Table.from_pydict(columns, schema=schema)


def build_nhsn(bronze_root: Path, out_dir: Path, abbr_to_name: dict[str, str], weeks: list[str], spec: dict) -> dict:
    source_path = bronze_root / "nhsn_finalized.parquet"
    read_columns = ["weekendingdate", "jurisdiction", *NHSN_TARGET_COLUMNS]
    rows = pq.read_table(source_path, columns=read_columns).to_pylist()
    week_set = set(weeks)
    records = []
    seen = set()

    for row in rows:
        abbr = row["jurisdiction"]
        week = parse_date_string(row["weekendingdate"])
        if abbr not in abbr_to_name or week not in week_set:
            continue
        jurisdiction = abbr_to_name[abbr]
        key = (jurisdiction, week)
        if key in seen:
            raise ValueError(f"duplicate NHSN key: {key}")
        seen.add(key)
        record = {
            "jurisdiction": jurisdiction,
            "jurisdiction_abbr": abbr,
            "week_end": week,
        }
        for raw_column, silver_column in NHSN_TARGET_COLUMNS.items():
            record[silver_column] = to_float_or_null(row[raw_column])
        records.append(record)

    records.sort(key=lambda item: (item["jurisdiction"], item["week_end"]))
    if len(records) != spec["expected_counts"]["canonical_grid_rows"]:
        raise ValueError(f"expected 9333 NHSN rows, got {len(records)}")

    columns = {name: [record[name] for record in records] for name in records[0]}
    schema = pa.schema(
        [
            ("jurisdiction", pa.string()),
            ("jurisdiction_abbr", pa.string()),
            ("week_end", pa.string()),
            ("covid_adm_count", pa.float64()),
            ("covid_adm_per100k", pa.float64()),
            ("flu_adm_count", pa.float64()),
            ("flu_adm_per100k", pa.float64()),
            ("rsv_adm_count", pa.float64()),
            ("rsv_adm_per100k", pa.float64()),
        ]
    )
    table = table_from_columns(columns, schema)
    metadata = {
        "source_tables": [str(source_path)],
        "key_columns": ["jurisdiction", "week_end"],
        "canonical_window": spec["canonical_window"],
        "canonical_jurisdiction_count": spec["jurisdiction_universe"]["count"],
        "transformations": [
            "filtered_to_50_states_plus_dc",
            "filtered_to_canonical_window",
            "jurisdiction_abbr_mapped_to_full_name",
            "target_columns_selected_and_cast_to_float64",
        ],
    }
    return write_table_with_metadata(
        table,
        out_dir / "nhsn_state_weekly.parquet",
        out_dir / "nhsn_state_weekly.metadata.json",
        metadata,
    )


def build_nssp(bronze_root: Path, out_dir: Path, weeks: list[str], spec: dict) -> dict:
    source_path = bronze_root / "nssp.parquet"
    rows = pq.read_table(source_path).to_pylist()
    week_set = set(weeks)
    geographies = {row["geography"] for row in rows}
    if geographies != {"United States"}:
        raise ValueError(f"NSSP must be national-only, got {sorted(geographies)}")

    values = {}
    features = set()
    for row in rows:
        week = parse_date_string(row["week_end"])
        if week not in week_set:
            continue
        feature = nssp_feature_name(row)
        key = (week, feature)
        if key in values:
            raise ValueError(f"duplicate NSSP key: {key}")
        values[key] = to_float_or_null(row["percent_visits"])
        features.add(feature)

    feature_names = sorted(features)
    if len(feature_names) != spec["expected_counts"]["nssp_feature_count"]:
        raise ValueError(f"expected 64 NSSP features, got {len(feature_names)}")

    columns = {"week_end": weeks}
    for feature in feature_names:
        columns[feature] = [values.get((week, feature)) for week in weeks]

    schema = pa.schema([("week_end", pa.string()), *[(feature, pa.float64()) for feature in feature_names]])
    table = table_from_columns(columns, schema)
    metadata = {
        "source_tables": [str(source_path)],
        "key_columns": ["week_end"],
        "canonical_window": spec["canonical_window"],
        "canonical_jurisdiction_count": spec["jurisdiction_universe"]["count"],
        "transformations": [
            "filtered_to_canonical_window",
            "pivoted_pathogen_demographic_features",
            "percent_visits_cast_to_float64",
        ],
        "geography_scope": "national_only",
        "broadcast_to_states": False,
        "feature_count": len(feature_names),
    }
    return write_table_with_metadata(
        table,
        out_dir / "nssp_national_weekly.parquet",
        out_dir / "nssp_national_weekly.metadata.json",
        metadata,
    )


def build_nwss(
    bronze_root: Path,
    out_dir: Path,
    source_stem: str,
    output_stem: str,
    prefix: str,
    jurisdictions: list[str],
    weeks: list[str],
    spec: dict,
) -> dict:
    source_path = bronze_root / f"{source_stem}.parquet"
    rows = pq.read_table(source_path).to_pylist()
    week_set = set(weeks)
    observations = {}
    raw_geographies = set()
    all_results_count = 0

    for row in rows:
        if row["data_collection_period"] != "All Results":
            continue
        all_results_count += 1
        jurisdiction = row["state_territory"]
        raw_geographies.add(jurisdiction)
        week = parse_date_string(row["week_ending_date"])
        if week not in week_set:
            continue
        key = (jurisdiction, week)
        if key in observations:
            raise ValueError(f"duplicate NWSS key in {source_stem}: {key}")
        observations[key] = row

    value_col = f"{prefix}_ww_state_wval"
    missing_col = f"{value_col}_is_missing"
    category_col = f"{prefix}_ww_wval_category"
    coverage_col = f"{prefix}_ww_coverage"
    national_col = f"{prefix}_ww_national_wval"
    regional_col = f"{prefix}_ww_regional_wval"

    output = {
        "jurisdiction": [],
        "week_end": [],
        value_col: [],
        missing_col: [],
        "source_has_observation": [],
        category_col: [],
        coverage_col: [],
        national_col: [],
        regional_col: [],
    }
    for jurisdiction in sorted(jurisdictions):
        for week in weeks:
            row = observations.get((jurisdiction, week))
            value = to_float_or_null(row["state_territory_wval"]) if row else None
            output["jurisdiction"].append(jurisdiction)
            output["week_end"].append(week)
            output[value_col].append(value)
            output[missing_col].append(value is None)
            output["source_has_observation"].append(row is not None)
            output[category_col].append(row.get("wval_category") if row else None)
            output[coverage_col].append(row.get("coverage") if row else None)
            output[national_col].append(to_float_or_null(row.get("national_wval")) if row else None)
            output[regional_col].append(to_float_or_null(row.get("regional_wval")) if row else None)

    schema = pa.schema(
        [
            ("jurisdiction", pa.string()),
            ("week_end", pa.string()),
            (value_col, pa.float64()),
            (missing_col, pa.bool_()),
            ("source_has_observation", pa.bool_()),
            (category_col, pa.string()),
            (coverage_col, pa.string()),
            (national_col, pa.float64()),
            (regional_col, pa.float64()),
        ]
    )
    table = table_from_columns(output, schema)
    expected_rows = spec["expected_counts"]["canonical_grid_rows"]
    if table.num_rows != expected_rows:
        raise ValueError(f"expected {expected_rows} rows for {output_stem}, got {table.num_rows}")

    metadata = {
        "source_tables": [str(source_path)],
        "key_columns": ["jurisdiction", "week_end"],
        "canonical_window": spec["canonical_window"],
        "canonical_jurisdiction_count": spec["jurisdiction_universe"]["count"],
        "transformations": [
            "filtered_to_all_results",
            "filtered_observations_to_canonical_window",
            "rebuilt_full_canonical_grid",
            "left_joined_observations_to_grid",
            "added_missing_mask",
        ],
        "period_filter": "All Results",
        "raw_all_results_row_count": all_results_count,
        "canonical_grid_rows": expected_rows,
        "north_dakota_preserved_with_missing_mask": True,
        "guam_in_raw_not_in_canonical_grid": "Guam" in raw_geographies and "Guam" not in jurisdictions,
    }
    return write_table_with_metadata(
        table,
        out_dir / f"{output_stem}.parquet",
        out_dir / f"{output_stem}.metadata.json",
        metadata,
    )


def write_manifest(out_dir: Path, entries: dict[str, dict]) -> None:
    manifest = {
        "phase": 4,
        "description": "Silver source-specific tables with canonical keys and missing semantics; no final panel join.",
        "entries": entries,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / "silver_manifest.json", manifest)


def main() -> int:
    args = parse_args()
    spec = load_yaml(Path(args.benchmark_spec))
    bronze_root = Path(args.bronze_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    abbr_to_name, jurisdictions = load_jurisdiction_map(Path(args.jurisdiction_map))
    weeks = canonical_weeks(
        spec["canonical_window"]["start_week_end"],
        spec["canonical_window"]["end_week_end"],
    )

    entries = {
        "nhsn_state_weekly": build_nhsn(bronze_root, out_dir, abbr_to_name, weeks, spec),
        "nssp_national_weekly": build_nssp(bronze_root, out_dir, weeks, spec),
        "nwss_flua_state_weekly": build_nwss(
            bronze_root, out_dir, "nwss_flua", "nwss_flua_state_weekly", "flua", jurisdictions, weeks, spec
        ),
        "nwss_rsv_state_weekly": build_nwss(
            bronze_root, out_dir, "nwss_rsv", "nwss_rsv_state_weekly", "rsv", jurisdictions, weeks, spec
        ),
    }
    write_manifest(out_dir, entries)
    print(f"OK: wrote silver manifest {out_dir / 'silver_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
