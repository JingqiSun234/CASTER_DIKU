import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import yaml


MATERIALIZED_SOURCES = {
    "nhsn_finalized": "nhsn_finalized",
    "nhsn_preliminary": "nhsn_preliminary",
    "nssp_ed_respiratory": "nssp",
    "nwss_flua": "nwss_flua",
    "nwss_rsv": "nwss_rsv",
}

SUMMARY_ONLY_SOURCES = {
    "variants": "auxiliary_only",
    "nwss_sars_cov_2_state_trend": "excluded_from_v0_core",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bronze parquet files from raw CDC CSVs.")
    parser.add_argument("--raw-root", required=True, help="Unpacked raw data root")
    parser.add_argument("--out", required=True, help="Bronze output directory")
    parser.add_argument("--contracts", default="configs/source_contracts.yaml", help="Source contracts YAML")
    return parser.parse_args()


def snake_case(name: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", name.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_").lower()
    if not normalized:
        raise ValueError(f"column name normalizes to empty string: {name!r}")
    return normalized


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def load_contracts(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["contracts"]


def resolve_raw_path(contract_raw_path: str, raw_root: Path) -> Path:
    raw_path = Path(contract_raw_path)
    if raw_path.parts and raw_path.parts[0] == "data_raw":
        raw_path = raw_root.joinpath(*raw_path.parts[1:])
    return raw_path


def read_csv_as_string_table(raw_path: Path, raw_columns: list[str]) -> pa.Table:
    convert_options = pacsv.ConvertOptions(
        column_types={column: pa.string() for column in raw_columns},
        strings_can_be_null=True,
        quoted_strings_can_be_null=True,
    )
    return pacsv.read_csv(raw_path, convert_options=convert_options)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_one(source_key: str, output_stem: str, contract: dict, raw_root: Path, out_dir: Path) -> dict:
    raw_path = resolve_raw_path(contract["raw_path"], raw_root)
    if not raw_path.is_file():
        raise FileNotFoundError(f"missing raw file for {source_key}: {raw_path}")

    raw_columns = read_header(raw_path)
    bronze_columns = [snake_case(column) for column in raw_columns]
    duplicates = sorted({column for column in bronze_columns if bronze_columns.count(column) > 1})
    if duplicates:
        raise ValueError(f"{source_key} column normalization produced duplicates: {duplicates}")

    table = read_csv_as_string_table(raw_path, raw_columns).rename_columns(bronze_columns)
    expected_rows = int(contract["metadata_row_count"])
    if table.num_rows != expected_rows:
        raise ValueError(
            f"{source_key} row count mismatch: expected {expected_rows}, got {table.num_rows}"
        )

    parquet_path = out_dir / f"{output_stem}.parquet"
    metadata_path = out_dir / f"{output_stem}.metadata.json"
    pq.write_table(table, parquet_path, compression="zstd")

    sidecar = {
        "source_key": source_key,
        "source_family": contract["source_family"],
        "snapshot_date": contract["snapshot_date"],
        "role": contract["role"],
        "raw_path": str(raw_path),
        "parquet_path": str(parquet_path),
        "metadata_row_count": expected_rows,
        "actual_row_count": table.num_rows,
        "column_count": len(bronze_columns),
        "raw_columns": raw_columns,
        "bronze_columns": bronze_columns,
        "column_name_map": dict(zip(raw_columns, bronze_columns)),
        "value_storage": "all_columns_as_arrow_string",
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "transformations": [
            "csv_loaded",
            "column_names_normalized_to_snake_case",
            "written_to_parquet",
        ],
        "no_business_filtering": True,
        "no_feature_engineering": True,
        "no_join": True,
    }
    write_json(metadata_path, sidecar)

    print(f"OK: wrote bronze {parquet_path} rows={table.num_rows}")
    return {
        "source_key": source_key,
        "output_stem": output_stem,
        "materialized": True,
        "raw_path": str(raw_path),
        "parquet_path": str(parquet_path),
        "metadata_path": str(metadata_path),
        "metadata_row_count": expected_rows,
        "actual_row_count": table.num_rows,
        "column_count": len(bronze_columns),
    }


def build_manifest(contracts: dict, materialized_entries: list[dict], raw_root: Path, out_dir: Path) -> None:
    entries = {entry["source_key"]: entry for entry in materialized_entries}
    for source_key, reason in SUMMARY_ONLY_SOURCES.items():
        contract = contracts[source_key]
        raw_path = resolve_raw_path(contract["raw_path"], raw_root)
        entries[source_key] = {
            "source_key": source_key,
            "materialized": False,
            "reason": reason,
            "raw_path": str(raw_path),
            "metadata_row_count": int(contract["metadata_row_count"]),
            "snapshot_date": contract["snapshot_date"],
            "role": contract["role"],
        }

    manifest = {
        "phase": 3,
        "source_snapshot": "2026-04-08",
        "description": "Bronze layer preserves raw source structure with snake_case columns; no business filtering, joins, or feature engineering.",
        "entries": [entries[key] for key in sorted(entries)],
    }
    write_json(out_dir / "bronze_manifest.json", manifest)


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    contracts = load_contracts(Path(args.contracts))

    materialized_entries = []
    for source_key, output_stem in MATERIALIZED_SOURCES.items():
        materialized_entries.append(
            build_one(source_key, output_stem, contracts[source_key], raw_root, out_dir)
        )

    build_manifest(contracts, materialized_entries, raw_root, out_dir)
    print(f"OK: wrote bronze manifest {out_dir / 'bronze_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
