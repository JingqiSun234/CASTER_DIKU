""







from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


MOBILITY_SCHEMA = "caster_benchmark_a_mobility_node_summary_v1"
MOBILITY_RELEASE_COLUMN = "__release_time__mobility"
MOBILITY_FEATURE_COLUMNS = (
    "mobility_log_outflow",
    "mobility_log_inflow",
    "mobility_log_self_flow",
    "mobility_net_flow_share",
    "mobility_external_out_share",
    "mobility_active_out_degree_share",
    "mobility_active_in_degree_share",
)
COUNTRY_GRAPH_LAYOUT: Mapping[str, str] = {
    "England": "EN",
    "France": "FR",
    "Italy": "IT",
    "Spain": "ES",
}


@dataclass(frozen=True)
class MobilityMaterialization:
    panel: pd.DataFrame
    metadata: Mapping[str, object]


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _summarize_graph(
    path: Path,
    *,
    country: str,
    country_code: str,
    date: pd.Timestamp,
    node_to_entity: Mapping[str, str],
) -> pd.DataFrame:
    edges = pd.read_csv(path, header=None, names=["source", "target", "weight"])
    _require(not edges.empty, f"empty Benchmark A mobility graph: {path}")
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    edges["weight"] = pd.to_numeric(edges["weight"], errors="raise")
    _require(
        np.isfinite(edges["weight"]).all() and edges["weight"].ge(0).all(),
        f"invalid mobility weights: {path}",
    )
    _require(not edges.duplicated(["source", "target"]).any(), f"duplicate mobility edge: {path}")
    nodes = set(edges["source"]) | set(edges["target"])
    mapped_nodes = set(node_to_entity)
    _require(
        mapped_nodes <= nodes,
        f"mobility/entity mismatch for {country} {date.date()}: "
        f"mapped_nodes_absent_from_graph={sorted(mapped_nodes - nodes)[:5]}",
    )

    ordered_nodes = sorted(nodes)
    n_nodes = len(ordered_nodes)
    out_sum = edges.groupby("source", sort=False)["weight"].sum().reindex(ordered_nodes, fill_value=0.0)
    in_sum = edges.groupby("target", sort=False)["weight"].sum().reindex(ordered_nodes, fill_value=0.0)
    out_degree = edges.groupby("source", sort=False)["target"].nunique().reindex(ordered_nodes, fill_value=0)
    in_degree = edges.groupby("target", sort=False)["source"].nunique().reindex(ordered_nodes, fill_value=0)
    self_flow = (
        edges.loc[edges["source"].eq(edges["target"])]
        .set_index("source")["weight"]
        .reindex(ordered_nodes, fill_value=0.0)
    )

                                                                            
                                                                              
                                                             
    filled_out = out_sum + (n_nodes - out_degree).astype(float)
    filled_in = in_sum + (n_nodes - in_degree).astype(float)
    out_values = filled_out.to_numpy(dtype=float)
    in_values = filled_in.to_numpy(dtype=float)
    self_values = self_flow.to_numpy(dtype=float)
    denom = np.maximum(in_values + out_values, 1.0)
    degree_denom = float(max(n_nodes - 1, 1))
    frame = pd.DataFrame(
        {
            "country": country,
            "country_code": country_code,
            "entity_id": [node_to_entity.get(node, "") for node in ordered_nodes],
            "date": date,
            "mobility_log_outflow": np.log1p(out_values),
            "mobility_log_inflow": np.log1p(in_values),
            "mobility_log_self_flow": np.log1p(self_values),
            "mobility_net_flow_share": (in_values - out_values) / denom,
            "mobility_external_out_share": np.maximum(out_values - self_values, 0.0) / np.maximum(out_values, 1.0),
            "mobility_active_out_degree_share": np.maximum(out_degree.to_numpy(dtype=float) - 1.0, 0.0) / degree_denom,
            "mobility_active_in_degree_share": np.maximum(in_degree.to_numpy(dtype=float) - 1.0, 0.0) / degree_denom,
            MOBILITY_RELEASE_COLUMN: date,
        }
    )
    return frame.loc[frame["entity_id"].ne("")].reset_index(drop=True)


def materialize_mobility_features(
    panel: pd.DataFrame,
    graph_root: str | Path,
    *,
    require_complete_entity_coverage: bool = True,
) -> MobilityMaterialization:
    ""






    required = {"country", "country_code", "entity_id", "date"}
    missing = required - set(panel.columns)
    _require(not missing, f"Benchmark A panel lacks mobility join keys: {sorted(missing)}")
    root = Path(graph_root).resolve()
    _require(root.is_dir(), f"Benchmark A graph root does not exist: {root}")
    base = panel.copy()
    base["country"] = base["country"].astype(str)
    base["country_code"] = base["country_code"].astype(str)
    base["entity_id"] = base["entity_id"].astype(str)
    base["date"] = pd.to_datetime(base["date"], errors="raise")

    feature_frames: list[pd.DataFrame] = []
    file_records: list[dict[str, str]] = []
    coverage: dict[str, dict[str, object]] = {}
    for country, country_frame in base.groupby("country", sort=True):
        _require(country in COUNTRY_GRAPH_LAYOUT, f"unsupported Benchmark A mobility country: {country}")
        expected_code = COUNTRY_GRAPH_LAYOUT[country]
        codes = set(country_frame["country_code"])
        _require(codes == {expected_code}, f"country-code mismatch for {country}: {sorted(codes)}")
        panel_entities = sorted(set(country_frame["entity_id"]))
        first_date = pd.Timestamp(country_frame["date"].min())
        first_path = root / country / "graphs" / f"{expected_code}_{first_date.strftime('%Y-%m-%d')}.csv"
        first_edges = pd.read_csv(first_path, header=None, usecols=[0, 1])
        graph_nodes = set(first_edges[0].astype(str)) | set(first_edges[1].astype(str))
        raw_to_panel = {
            entity.split(":", 1)[1] if entity.startswith(f"{expected_code}:") else entity: entity
            for entity in panel_entities
        }
        node_to_entity = {
            node: entity for node, entity in raw_to_panel.items() if node in graph_nodes
        }
        _require(
            len(raw_to_panel) == len(panel_entities),
            f"non-unique graph-node mapping for {country}",
        )
        unsupported = sorted(set(panel_entities) - set(node_to_entity.values()))
        coverage[country] = {
            "panel_entity_count": len(panel_entities),
            "matched_entity_count": len(node_to_entity),
            "coverage_fraction": len(node_to_entity) / max(len(panel_entities), 1),
            "unsupported_entity_ids": unsupported,
        }
        graph_dir = root / country / "graphs"
        for date in sorted(pd.DatetimeIndex(country_frame["date"].unique())):
            path = graph_dir / f"{expected_code}_{date.strftime('%Y-%m-%d')}.csv"
            _require(path.is_file(), f"missing Benchmark A mobility graph: {path}")
            feature_frames.append(
                _summarize_graph(
                    path,
                    country=country,
                    country_code=expected_code,
                    date=pd.Timestamp(date),
                    node_to_entity=node_to_entity,
                )
            )
            file_records.append({"path": path.relative_to(root).as_posix(), "sha256": _sha256_file(path)})

    if require_complete_entity_coverage:
        incomplete = {
            country: values["unsupported_entity_ids"]
            for country, values in coverage.items()
            if values["coverage_fraction"] != 1.0
        }
        _require(
            not incomplete,
            "formal Benchmark A panel must use EpiLLM graph nodes; "
            f"unsupported entity IDs by country: {incomplete}",
        )

    features = pd.concat(feature_frames, ignore_index=True)
    join_keys = ["country", "country_code", "entity_id", "date"]
    _require(not features.duplicated(join_keys).any(), "duplicate Benchmark A mobility node/date summary")
    out = base.merge(features, on=join_keys, how="left", validate="many_to_one")
    for column in MOBILITY_FEATURE_COLUMNS:
        out[f"{column}__missing_mask"] = out[column].isna()
    out[MOBILITY_RELEASE_COLUMN] = out[MOBILITY_RELEASE_COLUMN].fillna(out["date"])
    _require(
        out[MOBILITY_RELEASE_COLUMN].eq(out["date"]).all(),
        "Benchmark A mobility release time must equal graph date",
    )
    metadata = {
        "benchmark_a_mobility_schema": MOBILITY_SCHEMA,
        "benchmark_a_mobility_graph_root": str(root),
        "benchmark_a_mobility_graph_file_count": len(file_records),
        "benchmark_a_mobility_graph_set_sha256": _canonical_sha256(file_records),
        "benchmark_a_mobility_release_policy": "graph_date_same_day_benchmark_abstraction",
        "benchmark_a_mobility_representation": "causal_node_summaries_from_epillm_daily_weighted_matrix",
        "benchmark_a_mobility_feature_columns": list(MOBILITY_FEATURE_COLUMNS),
        "benchmark_a_mobility_country_coverage": coverage,
        "benchmark_a_mobility_missing_policy": (
            "fail_closed_require_complete_epillm_graph_node_coverage"
            if require_complete_entity_coverage
            else "alternate_diagnostic_mask_missing_and_use_unadjusted_model"
        ),
        "benchmark_a_mobility_complete_coverage_required": bool(
            require_complete_entity_coverage
        ),
        "benchmark_a_mobility_future_graph_access": False,
    }
    out.attrs.update(metadata)
    return MobilityMaterialization(panel=out, metadata=metadata)
