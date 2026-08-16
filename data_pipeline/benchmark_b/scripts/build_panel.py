import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


KEY_COLUMNS = ["jurisdiction", "jurisdiction_abbr", "week_end"]
TARGET_COLUMNS = [
    "covid_adm_count",
    "covid_adm_per100k",
    "flu_adm_count",
    "flu_adm_per100k",
]
PRIMARY_TARGETS = ["covid_adm_per100k", "flu_adm_per100k"]
LAG_OFFSETS = [1, 2, 4, 8]
ROLL_WINDOWS = [4, 8]
EXPECTED_ROWS = 9333
EXPECTED_WEEKS = 183
EXPECTED_JURISDICTIONS = 51

FLUA_COLUMNS = [
    "flua_ww_state_wval",
    "flua_ww_state_wval_is_missing",
    "flua_ww_source_has_observation",
    "flua_ww_wval_category",
    "flua_ww_coverage",
    "flua_ww_national_wval",
    "flua_ww_regional_wval",
]
RSV_COLUMNS = [
    "rsv_ww_state_wval",
    "rsv_ww_state_wval_is_missing",
    "rsv_ww_source_has_observation",
    "rsv_ww_wval_category",
    "rsv_ww_coverage",
    "rsv_ww_national_wval",
    "rsv_ww_regional_wval",
]
CALENDAR_COLUMNS = ["weekofyear", "weekofyear_sin", "weekofyear_cos"]
AR_COLUMNS = [
    f"{target}_{feature}"
    for target in PRIMARY_TARGETS
    for feature in ["lag1", "lag2", "lag4", "lag8", "roll4_mean", "roll8_mean"]
]


VALUE_TYPES = {
    "jurisdiction": pa.string(),
    "jurisdiction_abbr": pa.string(),
    "week_end": pa.string(),
    "weekofyear": pa.int64(),
    "flua_ww_state_wval_is_missing": pa.bool_(),
    "flua_ww_source_has_observation": pa.bool_(),
    "rsv_ww_state_wval_is_missing": pa.bool_(),
    "rsv_ww_source_has_observation": pa.bool_(),
    "flua_ww_wval_category": pa.string(),
    "flua_ww_coverage": pa.string(),
    "rsv_ww_wval_category": pa.string(),
    "rsv_ww_coverage": pa.string(),
}


for column in TARGET_COLUMNS:
    VALUE_TYPES[column] = pa.float64()
for column in [
    "flua_ww_state_wval",
    "flua_ww_national_wval",
    "flua_ww_regional_wval",
    "rsv_ww_state_wval",
    "rsv_ww_national_wval",
    "rsv_ww_regional_wval",
    "weekofyear_sin",
    "weekofyear_cos",
    *AR_COLUMNS,
]:
    VALUE_TYPES[column] = pa.float64()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final v0-core weekly panel with no-leakage features.")
    parser.add_argument("--targets", default="benchmark/gold/panel_targets_wide.parquet")
    parser.add_argument("--nssp", default="data_intermediate/silver/nssp_national_weekly.parquet")
    parser.add_argument("--nwss-flua", default="data_intermediate/silver/nwss_flua_state_weekly.parquet")
    parser.add_argument("--nwss-rsv", default="data_intermediate/silver/nwss_rsv_state_weekly.parquet")
    parser.add_argument("--out", default="benchmark/gold")
    parser.add_argument("--benchmark-spec", default="configs/benchmark_spec.yaml")
    return parser.parse_args()


def read_required_table(path: Path) -> pa.Table:
    if not path.is_file():
        raise FileNotFoundError(f"missing required input table: {path}")
    return pq.read_table(path)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def unique_key_count(rows: list[dict]) -> int:
    return len({(row["jurisdiction"], row["week_end"]) for row in rows})


def load_targets(path: Path) -> list[dict]:
    table = read_required_table(path)
    required = set(KEY_COLUMNS + TARGET_COLUMNS)
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"target table missing required columns: {sorted(missing)}")
    rows = table.select(KEY_COLUMNS + TARGET_COLUMNS).sort_by(
        [("jurisdiction", "ascending"), ("week_end", "ascending")]
    ).to_pylist()
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} target rows, got {len(rows)}")
    if unique_key_count(rows) != EXPECTED_ROWS:
        raise ValueError("target table has duplicate (jurisdiction, week_end) keys")
    return rows


def load_nssp(path: Path) -> tuple[dict[str, dict], list[str]]:
    table = read_required_table(path)
    nssp_columns = sorted(column for column in table.column_names if column != "week_end")
    if len(nssp_columns) != 64:
        raise ValueError(f"expected 64 NSSP features, got {len(nssp_columns)}")
    if not all(column.startswith("nssp_") and column.endswith("_pct") for column in nssp_columns):
        raise ValueError("NSSP columns must use nssp_*_pct names")
    by_week = {}
    for row in table.to_pylist():
        week = row["week_end"]
        if week in by_week:
            raise ValueError(f"duplicate NSSP week: {week}")
        by_week[week] = {column: row[column] for column in nssp_columns}
    if len(by_week) != EXPECTED_WEEKS:
        raise ValueError(f"expected {EXPECTED_WEEKS} NSSP weeks, got {len(by_week)}")
    return by_week, nssp_columns


def load_nwss(path: Path, pathogen_prefix: str) -> dict[tuple[str, str], dict]:
    table = read_required_table(path)
    if table.num_rows != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} {path} rows, got {table.num_rows}")
    rows = table.to_pylist()
    by_key = {}
    for row in rows:
        key = (row["jurisdiction"], row["week_end"])
        if key in by_key:
            raise ValueError(f"duplicate NWSS key: {key}")
        source_observation = row.pop("source_has_observation")
        row[f"{pathogen_prefix}_ww_source_has_observation"] = source_observation
        by_key[key] = row
    return by_key


def add_calendar_features(row: dict) -> None:
    parsed = date.fromisoformat(row["week_end"])
    weekofyear = int(parsed.isocalendar().week)
    row["weekofyear"] = weekofyear
    row["weekofyear_sin"] = math.sin(2 * math.pi * weekofyear / 52)
    row["weekofyear_cos"] = math.cos(2 * math.pi * weekofyear / 52)


def add_autoregressive_features(rows: list[dict]) -> None:
    by_jurisdiction = defaultdict(list)
    for row in rows:
        by_jurisdiction[row["jurisdiction"]].append(row)

    for jurisdiction_rows in by_jurisdiction.values():
        jurisdiction_rows.sort(key=lambda row: row["week_end"])
        for index, row in enumerate(jurisdiction_rows):
            for target in PRIMARY_TARGETS:
                for lag in LAG_OFFSETS:
                    row[f"{target}_lag{lag}"] = jurisdiction_rows[index - lag][target] if index >= lag else None
                for window in ROLL_WINDOWS:
                    feature = f"{target}_roll{window}_mean"
                    if index < window:
                        row[feature] = None
                        continue
                    prior_values = [jurisdiction_rows[prior_index][target] for prior_index in range(index - window, index)]
                    if any(value is None for value in prior_values):
                        row[feature] = None
                    else:
                        row[feature] = sum(prior_values) / window


def build_panel_rows(target_rows: list[dict], nssp_by_week: dict, flua_by_key: dict, rsv_by_key: dict) -> list[dict]:
    panel_rows = []
    weeks = {row["week_end"] for row in target_rows}
    missing_nssp_weeks = sorted(weeks - set(nssp_by_week))
    if missing_nssp_weeks:
        raise ValueError(f"NSSP is missing panel weeks: {missing_nssp_weeks[:5]}")

    for target_row in target_rows:
        row = dict(target_row)
        key = (row["jurisdiction"], row["week_end"])
        row.update(nssp_by_week[row["week_end"]])
        row.update(flua_by_key[key])
        row.update(rsv_by_key[key])
        add_calendar_features(row)
        panel_rows.append(row)

    add_autoregressive_features(panel_rows)
    panel_rows.sort(key=lambda item: (item["jurisdiction"], item["week_end"]))
    return panel_rows


def column_schema(columns: list[str]) -> pa.Schema:
    fields = []
    for column in columns:
        if column.startswith("nssp_"):
            fields.append((column, pa.float64()))
        else:
            fields.append((column, VALUE_TYPES[column]))
    return pa.schema(fields)


def table_from_rows(rows: list[dict], columns: list[str]) -> pa.Table:
    data = {column: [row.get(column) for row in rows] for column in columns}
    return pa.Table.from_pydict(data, schema=column_schema(columns))


def validate_panel(table: pa.Table) -> None:
    rows = table.to_pylist()
    if table.num_rows != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} panel rows, got {table.num_rows}")
    if unique_key_count(rows) != EXPECTED_ROWS:
        raise ValueError("panel has duplicate (jurisdiction, week_end) keys")
    jurisdictions = {row["jurisdiction"] for row in rows}
    weeks = {row["week_end"] for row in rows}
    if len(jurisdictions) != EXPECTED_JURISDICTIONS:
        raise ValueError(f"expected {EXPECTED_JURISDICTIONS} jurisdictions, got {len(jurisdictions)}")
    if len(weeks) != EXPECTED_WEEKS:
        raise ValueError(f"expected {EXPECTED_WEEKS} weeks, got {len(weeks)}")
    nd_rows = [row for row in rows if row["jurisdiction"] == "North Dakota"]
    if len(nd_rows) != EXPECTED_WEEKS:
        raise ValueError("North Dakota missing from final panel")
    if not all(row["flua_ww_state_wval"] is None and row["flua_ww_state_wval_is_missing"] is True for row in nd_rows):
        raise ValueError("North Dakota FluA wastewater missing mask is not preserved")
    if not all(row["rsv_ww_state_wval"] is None and row["rsv_ww_state_wval_is_missing"] is True for row in nd_rows):
        raise ValueError("North Dakota RSV wastewater missing mask is not preserved")


def variable_group(column: str) -> str:
    if column in TARGET_COLUMNS:
        return "target"
    if column.startswith("nssp_"):
        return "nssp_covariate"
    if column.startswith("flua_ww_") or column.startswith("rsv_ww_"):
        return "wastewater_covariate"
    if column.startswith("weekofyear"):
        return "calendar"
    if "_lag" in column or "_roll" in column:
        return "autoregressive"
    raise ValueError(f"unknown variable group for column: {column}")


def value_to_string(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_long_table(wide_rows: list[dict], non_key_columns: list[str]) -> pa.Table:
    columns = {
        "jurisdiction": [],
        "jurisdiction_abbr": [],
        "week_end": [],
        "variable_name": [],
        "variable_group": [],
        "value": [],
    }
    for row in wide_rows:
        for column in non_key_columns:
            columns["jurisdiction"].append(row["jurisdiction"])
            columns["jurisdiction_abbr"].append(row["jurisdiction_abbr"])
            columns["week_end"].append(row["week_end"])
            columns["variable_name"].append(column)
            columns["variable_group"].append(variable_group(column))
            columns["value"].append(value_to_string(row.get(column)))

    table = pa.Table.from_pydict(
        columns,
        schema=pa.schema(
            [
                ("jurisdiction", pa.string()),
                ("jurisdiction_abbr", pa.string()),
                ("week_end", pa.string()),
                ("variable_name", pa.string()),
                ("variable_group", pa.string()),
                ("value", pa.string()),
            ]
        ),
    )
    expected_rows = len(wide_rows) * len(non_key_columns)
    if table.num_rows != expected_rows:
        raise ValueError(f"expected {expected_rows} long panel rows, got {table.num_rows}")
    return table


def metadata_payload(
    wide_table: pa.Table,
    long_table: pa.Table,
    wide_path: Path,
    long_path: Path,
    feature_groups: dict[str, list[str]],
    args: argparse.Namespace,
) -> tuple[dict, dict]:
    schema_payload = {
        "tables": {
            "panel_weekly_wide": {
                "path": str(wide_path),
                "row_count": wide_table.num_rows,
                "key_columns": KEY_COLUMNS,
                "columns": wide_table.column_names,
            },
            "panel_weekly_long": {
                "path": str(long_path),
                "row_count": long_table.num_rows,
                "key_columns": KEY_COLUMNS + ["variable_name"],
                "columns": long_table.column_names,
            },
        },
        "feature_groups": feature_groups,
        "primary_targets": PRIMARY_TARGETS,
        "no_leakage_policy": [
            "lag features use prior weeks only",
            "rolling features use full prior windows only",
            "current and future weeks are excluded from lag and rolling features",
            "rolling features are null if history is insufficient or any required prior value is missing",
        ],
    }
    metadata = {
        "benchmark_name": "CDC-RespForecast",
        "version": "v0-core",
        "source_snapshot": "2026-04-08",
        "canonical_window": {
            "start_week_end": "2022-10-01",
            "end_week_end": "2026-03-28",
            "week_count": EXPECTED_WEEKS,
        },
        "jurisdiction_count": EXPECTED_JURISDICTIONS,
        "week_count": EXPECTED_WEEKS,
        "panel_weekly_wide_rows": wide_table.num_rows,
        "panel_weekly_long_rows": long_table.num_rows,
        "input_tables": {
            "targets": args.targets,
            "nssp": args.nssp,
            "nwss_flua": args.nwss_flua,
            "nwss_rsv": args.nwss_rsv,
            "benchmark_spec": args.benchmark_spec,
        },
        "feature_group_counts": {name: len(columns) for name, columns in feature_groups.items()},
        "known_caveats": [
            "NSSP is national-only and broadcast by week to canonical jurisdictions",
            "NWSS FluA/RSV are missing North Dakota raw observations and retain missing masks",
            "No target fill is performed",
            "Lag and rolling autoregressive features exclude current and future weeks",
            "Variants are not in v0-core",
            "SARS-CoV-2 wastewater is not in v0-core",
        ],
        "no_sample_generation": True,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return schema_payload, metadata


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_rows = load_targets(Path(args.targets))
    nssp_by_week, nssp_columns = load_nssp(Path(args.nssp))
    flua_by_key = load_nwss(Path(args.nwss_flua), "flua")
    rsv_by_key = load_nwss(Path(args.nwss_rsv), "rsv")

    panel_rows = build_panel_rows(target_rows, nssp_by_week, flua_by_key, rsv_by_key)
    wide_columns = [
        *KEY_COLUMNS,
        *TARGET_COLUMNS,
        *nssp_columns,
        *FLUA_COLUMNS,
        *RSV_COLUMNS,
        *CALENDAR_COLUMNS,
        *AR_COLUMNS,
    ]
    wide_table = table_from_rows(panel_rows, wide_columns)
    validate_panel(wide_table)

    wide_path = out_dir / "panel_weekly_wide.parquet"
    pq.write_table(wide_table, wide_path, compression="zstd")

    non_key_columns = [column for column in wide_columns if column not in KEY_COLUMNS]
    long_table = build_long_table(panel_rows, non_key_columns)
    long_path = out_dir / "panel_weekly_long.parquet"
    pq.write_table(long_table, long_path, compression="zstd")

    feature_groups = {
        "keys": KEY_COLUMNS,
        "targets": TARGET_COLUMNS,
        "nssp_covariates": nssp_columns,
        "wastewater_covariates": [*FLUA_COLUMNS, *RSV_COLUMNS],
        "calendar_features": CALENDAR_COLUMNS,
        "autoregressive_features": AR_COLUMNS,
    }
    schema_payload, metadata = metadata_payload(wide_table, long_table, wide_path, long_path, feature_groups, args)
    write_json(out_dir / "schema.json", schema_payload)
    write_yaml(out_dir / "benchmark_metadata.yaml", metadata)

    sidecar_base = {
        "source_tables": [args.targets, args.nssp, args.nwss_flua, args.nwss_rsv],
        "key_columns": ["jurisdiction", "week_end"],
        "feature_groups": feature_groups,
        "no_sample_generation": True,
        "no_leakage_policy": schema_payload["no_leakage_policy"],
        "build_timestamp_utc": metadata["build_timestamp_utc"],
    }
    write_json(
        out_dir / "panel_weekly_wide.metadata.json",
        {**sidecar_base, "output_path": str(wide_path), "row_count": wide_table.num_rows, "column_count": wide_table.num_columns},
    )
    write_json(
        out_dir / "panel_weekly_long.metadata.json",
        {**sidecar_base, "output_path": str(long_path), "row_count": long_table.num_rows, "column_count": long_table.num_columns},
    )

    print(f"OK: wrote panel {wide_path} rows={wide_table.num_rows} columns={wide_table.num_columns}")
    print(f"OK: wrote panel {long_path} rows={long_table.num_rows} columns={long_table.num_columns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
