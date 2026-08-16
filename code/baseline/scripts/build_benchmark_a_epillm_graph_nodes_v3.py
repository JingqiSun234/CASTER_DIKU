#!/usr/bin/env python3
""






from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Mapping

import pandas as pd
import yaml


BASELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
NEW_METHOD_SRC = REPO_ROOT / "code/caster/src"
if str(NEW_METHOD_SRC) not in sys.path:
    sys.path.insert(0, str(NEW_METHOD_SRC))

from benchmark_v3_protocol import (              
    A_DIRECT_HORIZONS,
    A_DIRECT_MODE,
    A_EMBARGO_END,
    A_EMBARGO_START,
    A_ROLLOUT_HORIZONS,
    A_ROLLOUT_MODE,
    A_TEST_END,
    A_TEST_START,
    A_TRAIN_END,
    A_VAL_END,
    A_VAL_START,
    PROTOCOL_VERSION,
    build_benchmark_a_v3,
    sha256_file,
)
from caster.data.benchmark_a_epillm import (              
    COUNTRY_CODES,
    EPILLM_GRAPH_NODE_SCHEMA,
    EXPECTED_NODE_COUNTS,
    LABEL_FILES,
    materialize_epillm_graph_node_panel,
)
from caster.data.benchmark_a_mobility import materialize_mobility_features              


DEFAULT_CONFIG = REPO_ROOT / "configs/benchmark_a_epillm_graph_nodes_v3.yaml"


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _repo_path(value: object, *, label: str) -> Path:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be a path")
    path = (REPO_ROOT / value.strip()).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the noLearning workspace: {value}") from exc
    return path


def _date(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _load_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "Benchmark A graph-node config must be a mapping")
    _require(payload.get("schema_version") == 1, "Benchmark A graph-node schema drift")
    _require(payload.get("protocol_version") == PROTOCOL_VERSION, "Benchmark A protocol drift")
    config = payload.get("benchmark_a")
    _require(isinstance(config, Mapping), "benchmark_a config must be a mapping")
    return dict(config)


def _periods(config: Mapping[str, object]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    raw = config.get("country_periods")
    _require(isinstance(raw, Mapping) and set(raw) == set(COUNTRY_CODES), "country-period drift")
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for country, value in raw.items():
        _require(isinstance(value, Mapping), f"{country} period must be a mapping")
        start, end = pd.Timestamp(value["start"]), pd.Timestamp(value["end"])
        _require(start <= end, f"invalid period for {country}")
        periods[str(country)] = (start, end)
    return periods


def _validate_preserved_contract(config: Mapping[str, object]) -> None:
    contract = config.get("preserved_contract")
    _require(isinstance(contract, Mapping), "preserved_contract must be a mapping")
    expected = {
        A_DIRECT_MODE: list(A_DIRECT_HORIZONS),
        A_ROLLOUT_MODE: list(A_ROLLOUT_HORIZONS),
        "split_basis": "forecast_origin",
        "train_end": _date(A_TRAIN_END),
        "validation_start": _date(A_VAL_START),
        "validation_end": _date(A_VAL_END),
        "embargo_start": _date(A_EMBARGO_START),
        "embargo_end": _date(A_EMBARGO_END),
        "test_start": _date(A_TEST_START),
        "test_end": _date(A_TEST_END),
        "release_policy": "target_time_immediate_daily_benchmark_label",
        "no_learning_algorithm_changed": False,
    }
    _require(dict(contract) == expected, "noLearning Benchmark A task/split contract changed")
    _require(
        dict(config.get("expected_entity_counts", {})) == dict(EXPECTED_NODE_COUNTS),
        "EpiLLM expected node-count drift",
    )
    _require(
        config.get("entity_policy") == "sorted_union_of_first_epillm_daily_graph_endpoints",
        "EpiLLM entity-policy drift",
    )


def _graph_records(
    root: Path,
    periods: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for country, code in COUNTRY_CODES.items():
        start, end = periods[country]
        for date in pd.date_range(start, end, freq="D"):
            path = root / country / "graphs" / f"{code}_{_date(date)}.csv"
            _require(path.is_file(), f"missing graph: {path}")
            records.append(
                {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
            )
    return records


def verify_workspace_mirror(
    raw_root: Path,
    authority_root: Path,
    periods: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, object]:
    ""

    _require(authority_root.is_dir(), f"EpiLLM authority root does not exist: {authority_root}")
    local_graphs = _graph_records(raw_root, periods)
    authority_graphs = _graph_records(authority_root, periods)
    _require(local_graphs == authority_graphs, "workspace graph mirror differs from EpiLLM")

    label_records: list[dict[str, str]] = []
    for country in COUNTRY_CODES:
        dates = [_date(value) for value in pd.date_range(*periods[country], freq="D")]
        local_path = raw_root / country / LABEL_FILES[country]
        authority_path = authority_root / country / LABEL_FILES[country]
        local = pd.read_csv(local_path).set_index("name")
        authority = pd.read_csv(authority_path).set_index("name")
        nodes = sorted(
            set(
                pd.read_csv(
                    raw_root / country / "graphs" / f"{COUNTRY_CODES[country]}_{dates[0]}.csv",
                    header=None,
                    usecols=[0, 1],
                )
                .astype(str)
                .to_numpy()
                .ravel()
            )
        )
        pd.testing.assert_frame_equal(
            local.loc[nodes, dates].apply(pd.to_numeric),
            authority.loc[nodes, dates].apply(pd.to_numeric),
            check_dtype=False,
            check_names=True,
        )
        label_records.append(
            {
                "country": country,
                "case_values_sha256": _canonical_sha256(
                    local.loc[nodes, dates].astype(float).to_numpy().tolist()
                ),
            }
        )
    try:
        authority_relative = authority_root.relative_to(REPO_ROOT)
    except ValueError:
        authority_identity = "external_epillm_authority"
    else:
        authority_identity = authority_relative.as_posix()
    return {
        "epillm_authority_root": authority_identity,
        "authority_graph_file_count": len(authority_graphs),
        "authority_graph_set_sha256": _canonical_sha256(authority_graphs),
        "authority_case_values_sha256": _canonical_sha256(label_records),
        "workspace_graph_mirror_byte_equal": True,
        "workspace_case_values_equal": True,
    }


def build(config_path: Path = DEFAULT_CONFIG) -> Path:
    config = _load_config(config_path)
    _validate_preserved_contract(config)
    raw_root = _repo_path(config["raw_dataset_root"], label="raw_dataset_root")
    source_path = _repo_path(config["source_panel"], label="source_panel")
    output_dir = _repo_path(config["output_dir"], label="output_dir")
    authority_root = Path(str(config["authority_dataset_root"])).expanduser().resolve()
    periods = _periods(config)
    authority = verify_workspace_mirror(raw_root, authority_root, periods)

    materialized = materialize_epillm_graph_node_panel(raw_root, periods)
    source_set_sha256 = str(materialized.metadata["source_set_sha256"])
    _require(
        source_set_sha256 == str(config["expected_entity_source_set_sha256"]),
        "Benchmark A EpiLLM source-set fingerprint drift",
    )
    panel = materialized.panel.copy()
    panel["date"] = panel["date"].dt.strftime("%Y-%m-%d")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(source_path, index=False)
    source_manifest = {
        **dict(materialized.metadata),
        **authority,
        "status": "derived_scientific_input_not_model_output",
        "raw_root": raw_root.relative_to(REPO_ROOT).as_posix(),
        "panel": source_path.name,
        "panel_rows": int(len(panel)),
        "panel_sha256": sha256_file(source_path),
    }
    source_manifest_path = source_path.parent / "source_daily_panel.manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    build_benchmark_a_v3(source_path, output_dir)
    output_manifest_path = output_dir / "run_manifest.json"
    output_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    mobility_root = _repo_path(config["mobility"]["graph_root"], label="mobility.graph_root")
    mobility = materialize_mobility_features(panel, mobility_root)
    coverage = dict(mobility.metadata["benchmark_a_mobility_country_coverage"])
    _require(
        all(float(value["coverage_fraction"]) == 1.0 for value in coverage.values()),
        "official EpiLLM entities must have 100% mobility coverage",
    )
    panel_path = output_dir / "daily_panel.csv"
    ledger_path = output_dir / "event_ledger.csv"
    panel_sha256 = sha256_file(panel_path)
    ledger_sha256 = sha256_file(ledger_path)
    identity = {
        "protocol_version": PROTOCOL_VERSION,
        "entity_schema": EPILLM_GRAPH_NODE_SCHEMA,
        "entity_source_set_sha256": source_set_sha256,
        "source_panel_sha256": sha256_file(source_path),
        "daily_panel_sha256": panel_sha256,
        "event_ledger_sha256": ledger_sha256,
        "modes": {
            A_DIRECT_MODE: list(A_DIRECT_HORIZONS),
            A_ROLLOUT_MODE: list(A_ROLLOUT_HORIZONS),
        },
        "split_contract": output_manifest["split_contract"],
        "release_time_policy": output_manifest["release_time_policy"],
    }
    input_fingerprint = _canonical_sha256(identity)
    output_manifest.update(
        {
            "source_panel": source_path.relative_to(REPO_ROOT).as_posix(),
            "entity_schema": EPILLM_GRAPH_NODE_SCHEMA,
            "entity_policy": materialized.metadata["entity_policy"],
            "epillm_loader_parity": materialized.metadata["epillm_loader_parity"],
            "entity_source_set_sha256": source_set_sha256,
            "entity_node_counts": materialized.metadata["node_counts"],
            "entity_node_hashes": materialized.metadata["node_hashes"],
            "mobility_country_coverage": coverage,
            "mobility_coverage_fraction": 1.0,
            "daily_panel_sha256": panel_sha256,
            "event_ledger_sha256": ledger_sha256,
            "source_panel_manifest": source_manifest_path.relative_to(REPO_ROOT).as_posix(),
            "source_panel_manifest_sha256": sha256_file(source_manifest_path),
            "benchmark_a_input_fingerprint": input_fingerprint,
            "input_fingerprint_sha256": input_fingerprint,
            "input_fingerprint_fields": identity,
            "input_change_invalidates_forecast_posterior_agent_results": True,
            "invalidates_prior_top_n_entity_inputs": True,
            "stale_forecast_posterior_agent_reuse_allowed": False,
            "input_window_aligned_to_epillm": False,
            "task_split_or_mode_changed": False,
            "no_learning_algorithm_changed": False,
            **authority,
        }
    )
    output_manifest_path.write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build noLearning Benchmark A v3 on official EpiLLM graph nodes."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    output = build(args.config.resolve())
    print(f"ok output={output}")


if __name__ == "__main__":
    main()
