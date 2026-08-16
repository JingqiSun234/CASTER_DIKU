""







from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


EPILLM_GRAPH_NODE_SCHEMA = "caster_benchmark_a_epillm_graph_nodes_v1"
COUNTRY_CODES: Mapping[str, str] = {
    "England": "EN",
    "France": "FR",
    "Italy": "IT",
    "Spain": "ES",
}
LABEL_FILES: Mapping[str, str] = {
    country: f"{country.lower()}_labels.csv" for country in COUNTRY_CODES
}
EXPECTED_NODE_COUNTS: Mapping[str, int] = {
    "England": 129,
    "France": 81,
    "Italy": 105,
    "Spain": 34,
}


@dataclass(frozen=True)
class EpiLLMGraphNodePanel:
    panel: pd.DataFrame
    metadata: Mapping[str, object]


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _graph_nodes(path: Path) -> tuple[str, ...]:
    edges = pd.read_csv(path, header=None, usecols=[0, 1], names=["source", "target"])
    _require(not edges.empty, f"empty EpiLLM graph: {path}")
    nodes = set(edges["source"].astype(str)) | set(edges["target"].astype(str))
    return tuple(sorted(nodes))


def materialize_epillm_graph_node_panel(
    raw_root: str | Path,
    country_periods: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> EpiLLMGraphNodePanel:
    ""

    root = Path(raw_root).resolve()
    _require(root.is_dir(), f"Benchmark A raw root does not exist: {root}")
    _require(set(country_periods) == set(COUNTRY_CODES), "Benchmark A country set drift")

    pieces: list[pd.DataFrame] = []
    source_records: list[dict[str, object]] = []
    node_counts: dict[str, int] = {}
    node_hashes: dict[str, str] = {}
    for country, code in COUNTRY_CODES.items():
        start, end = map(pd.Timestamp, country_periods[country])
        graph_dir = root / country / "graphs"
        first_graph = graph_dir / f"{code}_{start.strftime('%Y-%m-%d')}.csv"
        _require(first_graph.is_file(), f"missing first EpiLLM graph: {first_graph}")
        nodes = _graph_nodes(first_graph)
        _require(
            len(nodes) == EXPECTED_NODE_COUNTS[country],
            f"{country} EpiLLM node-count drift: {len(nodes)}",
        )
        expected_dates = pd.date_range(start, end, freq="D")
        graph_records: list[dict[str, str]] = []
        for date in expected_dates:
            path = graph_dir / f"{code}_{date.strftime('%Y-%m-%d')}.csv"
            _require(path.is_file(), f"missing EpiLLM graph: {path}")
            _require(
                _graph_nodes(path) == nodes,
                f"{country} graph-node set/order drifts on {date.date()}",
            )
            graph_records.append(
                {"path": path.relative_to(root).as_posix(), "sha256": _sha256_file(path)}
            )

        label_path = root / country / LABEL_FILES[country]
        _require(label_path.is_file(), f"missing EpiLLM labels: {label_path}")
        labels = pd.read_csv(label_path)
        _require("name" in labels and not labels["name"].duplicated().any(), f"invalid labels: {label_path}")
        labels["name"] = labels["name"].astype(str)
        labels = labels.set_index("name", drop=True)
        _require(set(nodes) <= set(labels.index), f"{country} graph node absent from labels")
        date_columns = tuple(date.strftime("%Y-%m-%d") for date in expected_dates)
        _require(set(date_columns) <= set(labels), f"{country} label dates do not cover protocol period")

        cases = labels.loc[list(nodes), list(date_columns)].apply(pd.to_numeric, errors="raise")
        _require(
            np.isfinite(cases.to_numpy(dtype=float)).all()
            and cases.ge(0).to_numpy().all(),
            f"{country} case labels must be finite and nonnegative",
        )
        long = cases.rename_axis("raw_region_id").reset_index().melt(
            id_vars="raw_region_id",
            var_name="date",
            value_name="observed_value",
        )
        population = (
            pd.to_numeric(labels.loc[list(nodes), "population"], errors="raise")
            if "population" in labels
            else pd.Series(float("nan"), index=list(nodes))
        )
        _require(
            np.isfinite(population.to_numpy(dtype=float)).all()
            and population.gt(0).all(),
            f"{country} population must be finite and positive",
        )
        node_index = {node: index for index, node in enumerate(nodes)}
        long.insert(0, "dataset", "benchmark_a_epillm")
        long.insert(1, "country", country)
        long.insert(2, "country_code", code)
        long.insert(3, "entity_id", code + ":" + long["raw_region_id"].astype(str))
        long.insert(5, "node_index", long["raw_region_id"].map(node_index).astype(int))
        long.insert(7, "component", "cases")
        long["population"] = long["raw_region_id"].map(population).astype(float)
        long["date"] = pd.to_datetime(long["date"], errors="raise")
        pieces.append(long)

        node_counts[country] = len(nodes)
        node_hashes[country] = _canonical_sha256(list(nodes))
        source_records.append(
            {
                "country": country,
                "label_path": label_path.relative_to(root).as_posix(),
                "label_sha256": _sha256_file(label_path),
                "graph_files": graph_records,
                "node_count": len(nodes),
                "node_sha256": node_hashes[country],
            }
        )

    panel = pd.concat(pieces, ignore_index=True).sort_values(
        ["country", "node_index", "date"]
    ).reset_index(drop=True)
    _require(panel["entity_id"].nunique() == 349, "EpiLLM total graph-node count must be 349")
    _require(len(panel) == sum(len(pd.date_range(*country_periods[country], freq="D")) * count for country, count in node_counts.items()), "EpiLLM graph-node panel row-count drift")
    _require(not panel.duplicated(["entity_id", "date", "component"]).any(), "duplicate graph-node observation")
    for country, count in node_counts.items():
        indices = sorted(panel.loc[panel["country"].eq(country), "node_index"].unique())
        _require(indices == list(range(count)), f"{country} node_index is not contiguous")

    metadata: dict[str, object] = {
        "schema_version": 1,
        "entity_schema": EPILLM_GRAPH_NODE_SCHEMA,
        "entity_policy": "sorted_union_of_first_epillm_daily_graph_endpoints",
        "epillm_loader_parity": "generate_graphs_tmp_sorted_nodes_then_labels_loc_graph_nodes",
        "raw_root": str(root),
        "node_counts": node_counts,
        "total_node_count": int(panel["entity_id"].nunique()),
        "node_hashes": node_hashes,
        "source_set_sha256": _canonical_sha256(source_records),
        "source_records": source_records,
        "input_window_aligned_to_epillm": False,
        "task_split_or_mode_changed": False,
    }
    panel.attrs.update(metadata)
    return EpiLLMGraphNodePanel(panel=panel, metadata=metadata)
