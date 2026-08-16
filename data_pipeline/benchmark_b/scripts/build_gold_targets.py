import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


KEY_COLUMNS = ["jurisdiction", "jurisdiction_abbr", "week_end"]
TARGET_DEFINITIONS = {
    "covid_adm_count": {
        "value_type": "count",
        "pathogen": "covid19",
        "is_primary_benchmark_target": False,
    },
    "covid_adm_per100k": {
        "value_type": "per100k",
        "pathogen": "covid19",
        "is_primary_benchmark_target": True,
    },
    "flu_adm_count": {
        "value_type": "count",
        "pathogen": "influenza",
        "is_primary_benchmark_target": False,
    },
    "flu_adm_per100k": {
        "value_type": "per100k",
        "pathogen": "influenza",
        "is_primary_benchmark_target": True,
    },
}
TARGET_COLUMNS = list(TARGET_DEFINITIONS)
PRIMARY_TARGETS = [
    name for name, definition in TARGET_DEFINITIONS.items() if definition["is_primary_benchmark_target"]
]
AUXILIARY_TARGETS = [
    name for name, definition in TARGET_DEFINITIONS.items() if not definition["is_primary_benchmark_target"]
]
EXTENSION_TARGETS_EXCLUDED = ["rsv_adm_count", "rsv_adm_per100k"]
EXPECTED_ROWS = 9333
EXPECTED_MISSING = {
    "covid_adm_count": 36,
    "covid_adm_per100k": 36,
    "flu_adm_count": 36,
    "flu_adm_per100k": 36,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v0-core gold target tables.")
    parser.add_argument(
        "--silver-nhsn",
        default="data_intermediate/silver/nhsn_state_weekly.parquet",
        help="Silver NHSN state-week table",
    )
    parser.add_argument("--out", default="benchmark/gold", help="Gold output directory")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def unique_key_count(rows: list[dict]) -> int:
    return len({(row["jurisdiction"], row["week_end"]) for row in rows})


def missing_counts(table: pa.Table) -> dict[str, int]:
    return {column: table[column].null_count for column in TARGET_COLUMNS}


def validate_wide(table: pa.Table) -> None:
    if table.num_rows != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} wide rows, got {table.num_rows}")
    rows = table.to_pylist()
    if unique_key_count(rows) != EXPECTED_ROWS:
        raise ValueError("wide target table has duplicate (jurisdiction, week_end) keys")
    if any(column in table.column_names for column in EXTENSION_TARGETS_EXCLUDED):
        raise ValueError("RSV extension targets must not enter v0-core wide target table")
    observed_missing = missing_counts(table)
    if observed_missing != EXPECTED_MISSING:
        raise ValueError(f"unexpected missing counts: {observed_missing}")


def build_long_table(wide: pa.Table) -> pa.Table:
    long_rows = {
        "jurisdiction": [],
        "jurisdiction_abbr": [],
        "week_end": [],
        "target_name": [],
        "value_type": [],
        "pathogen": [],
        "is_primary_benchmark_target": [],
        "value": [],
    }
    for row in wide.to_pylist():
        for target_name, definition in TARGET_DEFINITIONS.items():
            long_rows["jurisdiction"].append(row["jurisdiction"])
            long_rows["jurisdiction_abbr"].append(row["jurisdiction_abbr"])
            long_rows["week_end"].append(row["week_end"])
            long_rows["target_name"].append(target_name)
            long_rows["value_type"].append(definition["value_type"])
            long_rows["pathogen"].append(definition["pathogen"])
            long_rows["is_primary_benchmark_target"].append(definition["is_primary_benchmark_target"])
            long_rows["value"].append(row[target_name])

    table = pa.Table.from_pydict(
        long_rows,
        schema=pa.schema(
            [
                ("jurisdiction", pa.string()),
                ("jurisdiction_abbr", pa.string()),
                ("week_end", pa.string()),
                ("target_name", pa.string()),
                ("value_type", pa.string()),
                ("pathogen", pa.string()),
                ("is_primary_benchmark_target", pa.bool_()),
                ("value", pa.float64()),
            ]
        ),
    )
    expected_long_rows = EXPECTED_ROWS * len(TARGET_COLUMNS)
    if table.num_rows != expected_long_rows:
        raise ValueError(f"expected {expected_long_rows} long rows, got {table.num_rows}")
    if any(target in set(table["target_name"].to_pylist()) for target in EXTENSION_TARGETS_EXCLUDED):
        raise ValueError("RSV extension targets must not enter v0-core long target table")
    return table


def base_metadata() -> dict:
    return {
        "primary_benchmark_targets": PRIMARY_TARGETS,
        "auxiliary_retained_targets": AUXILIARY_TARGETS,
        "extension_targets_excluded": EXTENSION_TARGETS_EXCLUDED,
        "missing_counts": EXPECTED_MISSING,
        "no_target_fill": True,
        "no_covariate_join": True,
        "no_sample_generation": True,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    args = parse_args()
    silver_path = Path(args.silver_nhsn)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not silver_path.is_file():
        raise FileNotFoundError(f"missing silver NHSN table: {silver_path}")

    wide = pq.read_table(silver_path, columns=[*KEY_COLUMNS, *TARGET_COLUMNS])
    wide = wide.sort_by([("jurisdiction", "ascending"), ("week_end", "ascending")])
    validate_wide(wide)

    wide_path = out_dir / "panel_targets_wide.parquet"
    pq.write_table(wide, wide_path, compression="zstd")
    wide_metadata = {
        **base_metadata(),
        "source_table": str(silver_path),
        "output_path": str(wide_path),
        "row_count": wide.num_rows,
        "key_columns": ["jurisdiction", "week_end"],
        "target_columns": TARGET_COLUMNS,
    }
    write_json(out_dir / "panel_targets_wide.metadata.json", wide_metadata)

    long = build_long_table(wide)
    long_path = out_dir / "panel_targets_long.parquet"
    pq.write_table(long, long_path, compression="zstd")
    long_metadata = {
        **base_metadata(),
        "source_table": str(wide_path),
        "output_path": str(long_path),
        "row_count": long.num_rows,
        "key_columns": ["jurisdiction", "week_end", "target_name"],
        "long_target_count_per_key": len(TARGET_COLUMNS),
        "target_columns": TARGET_COLUMNS,
    }
    write_json(out_dir / "panel_targets_long.metadata.json", long_metadata)

    manifest = {
        "phase": 5,
        "description": "v0-core gold target tables only; no covariate join or sample generation.",
        "outputs": {
            "panel_targets_wide": {"path": str(wide_path), "row_count": wide.num_rows},
            "panel_targets_long": {"path": str(long_path), "row_count": long.num_rows},
        },
        "v0_core_targets": TARGET_COLUMNS,
        "primary_benchmark_targets": PRIMARY_TARGETS,
        "auxiliary_retained_targets": AUXILIARY_TARGETS,
        "extension_targets_excluded": EXTENSION_TARGETS_EXCLUDED,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / "gold_targets_manifest.json", manifest)
    print(f"OK: wrote gold targets {wide_path} rows={wide.num_rows}")
    print(f"OK: wrote gold targets {long_path} rows={long.num_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
