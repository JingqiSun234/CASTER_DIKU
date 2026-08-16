#!/usr/bin/env python3
""




from __future__ import annotations

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "code/caster/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from caster.data.benchmark_b_context import (              
    annotate_stream_release_times,
    input_fingerprint,
    load_context_contract,
    sha256_file,
)


PROTOCOL_VERSION = "caster_data_protocol_v26_1"
DEFAULT_CONFIG = ROOT / "configs/caster_data_protocol_benchmark_b_v26_1.yaml"
COMPONENTS = ("covid_adm_per100k", "flu_adm_per100k")


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _repo_path(raw: object, label: str) -> Path:
    path = (ROOT / str(raw)).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository: {raw}") from exc
    return path


def _date(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}_{sha256('|'.join(map(str, parts)).encode()).hexdigest()[:20]}"


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _ids(task_id: str, entity: str, component: str, mode: str, origin: pd.Timestamp, target: pd.Timestamp, lead: int) -> dict[str, str]:
    path_id = _stable_id("path", PROTOCOL_VERSION, task_id, entity, component, mode, _date(origin))
    natural_id = _stable_id("nat", task_id, entity, component, _date(origin), _date(target), lead, "final")
    forecast_id = _stable_id("fcst", PROTOCOL_VERSION, natural_id, mode)
    return {
        "forecast_path_id": path_id,
        "natural_event_id": natural_id,
        "forecast_id": forecast_id,
        "event_id": _stable_id("evt", PROTOCOL_VERSION, forecast_id),
    }


def _split(origin: pd.Timestamp, config: Mapping[str, Any]) -> str | None:
    ranges = (
        ("train", "first_origin", "train_end"),
        ("val", "validation_start", "validation_end"),
        ("embargo", "embargo_start", "embargo_end"),
        ("test", "test_start", "test_end"),
    )
    return next(
        (name for name, start, end in ranges if pd.Timestamp(config[start]) <= origin <= pd.Timestamp(config[end])),
        None,
    )


def _row(*, config: Mapping[str, Any], entity: str, component: str, mode: str, mode_spec: Mapping[str, Any], origin: pd.Timestamp, target: pd.Timestamp, lead: int, value: object, split: str) -> dict[str, object]:
    task_id = str(config["task_id"])
    release = target + pd.Timedelta(days=7)
    observed = _finite(value)
    return {
        "protocol_version": PROTOCOL_VERSION,
        **_ids(task_id, entity, component, mode, origin, target, lead),
        "task_id": task_id,
        "dataset": "benchmark_b",
        "split": split,
        "split_basis": "forecast_origin",
        "calibration_eligible": bool(split == "val" and release <= pd.Timestamp(config["selection_freeze_time"])),
        "forecast_issued": True,
        "result_metric_eligible": split == "test",
        "posterior_update_eligible_after_release": True,
        "mode": mode,
        "mode_kind": str(mode_spec["mode_kind"]),
        "forecast_strategy": str(mode_spec["forecast_strategy"]),
        "nominal_horizon_steps": max(map(int, mode_spec["horizons"])),
        "lead_steps": int(lead),
        "horizon": int(lead),
        "entity_id": entity,
        "component": component,
        "forecast_origin": _date(origin),
        "target_time": _date(target),
        "release_time": _date(release),
        "release_lag_steps": 1,
        "features_available_until": _date(origin),
        "observed_mask": observed,
        "observed_value": float(value) if observed else np.nan,
        "revision_version": "final",
        "jurisdiction": entity,
    }


def build(config_path: Path = DEFAULT_CONFIG) -> Path:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(payload.get("protocol_version") == PROTOCOL_VERSION, "protocol version drift")
    config = payload["benchmark_b"]
    source = _repo_path(config["source_panel"], "source_panel")
    source_metadata = _repo_path(config["source_panel_metadata"], "source_panel_metadata")
    contract_path = _repo_path(config["context_contract"], "context_contract")
    output = _repo_path(config["output_dir"], "output_dir")
    panel = pd.read_parquet(source)
    required = {"jurisdiction", "week_end", *COMPONENTS}
    _require(required <= set(panel.columns), f"source panel lacks {sorted(required - set(panel.columns))}")
    panel["week_end"] = pd.to_datetime(panel["week_end"], errors="raise")
    panel = panel.sort_values(["jurisdiction", "week_end"], kind="mergesort").reset_index(drop=True)
    contract = load_context_contract(contract_path, panel_columns=panel.columns)
    panel = annotate_stream_release_times(panel, contract)
    _require(panel["jurisdiction"].nunique() == 51, "Benchmark B must contain 51 jurisdictions")

    lookup = {
        (str(row.jurisdiction), pd.Timestamp(row.week_end), component): row._asdict()[component]
        for row in panel.itertuples(index=False)
        for component in COMPONENTS
    }
    rows: list[dict[str, object]] = []
    for jurisdiction, group in panel.groupby("jurisdiction", sort=True):
        dates = tuple(pd.Timestamp(value) for value in group["week_end"].sort_values().unique())
        date_set = set(dates)
        for origin_index, origin in enumerate(dates):
            if origin_index + 1 < int(config["minimum_origin_history_steps"]):
                continue
            split = _split(origin, config)
            if split is None:
                continue
            for component in COMPONENTS:
                for mode, mode_spec in config["modes"].items():
                    for lead in map(int, mode_spec["horizons"]):
                        target = origin + pd.Timedelta(days=7 * lead)
                        if target in date_set:
                            rows.append(
                                _row(
                                    config=config,
                                    entity=str(jurisdiction),
                                    component=component,
                                    mode=str(mode),
                                    mode_spec=mode_spec,
                                    origin=origin,
                                    target=target,
                                    lead=lead,
                                    value=lookup.get((str(jurisdiction), target, component), np.nan),
                                    split=split,
                                )
                            )
    ledger = pd.DataFrame(rows).sort_values(
        ["entity_id", "component", "forecast_origin", "mode", "lead_steps"], kind="mergesort"
    ).reset_index(drop=True)
    _require(ledger["forecast_id"].is_unique, "forecast_id is not unique")
    _require(set(ledger["task_id"]) == {"benchmark_b_pooled"}, "task split drift")
    _require(set(ledger["component"]) == set(COMPONENTS), "component split drift")
    _require(set(ledger["split"]) == {"train", "val", "embargo", "test"}, "split coverage drift")
    origin = pd.to_datetime(ledger["forecast_origin"], errors="raise")
    target = pd.to_datetime(ledger["target_time"], errors="raise")
    release = pd.to_datetime(ledger["release_time"], errors="raise")
    _require(bool((origin < target).all() and (release == target + pd.Timedelta(days=7)).all()), "causal/release invariant failed")
    _require(bool(pd.to_datetime(ledger["features_available_until"]).le(origin).all()), "future feature cutoff")

    output.mkdir(parents=True, exist_ok=True)
    panel_out = panel.copy()
    for column in panel_out.columns:
        if column == "week_end" or column.startswith("__release_time__"):
            panel_out[column] = pd.to_datetime(panel_out[column], errors="raise").dt.strftime("%Y-%m-%d")
    panel_path, ledger_path = output / "weekly_panel.csv", output / "event_ledger.csv"
    panel_out.to_csv(panel_path, index=False)
    ledger.to_csv(ledger_path, index=False)
    panel_hash, ledger_hash = sha256_file(panel_path), sha256_file(ledger_path)
    manifest = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "status": "input_ledger_package_not_model_run",
        "task_id": "benchmark_b_pooled",
        "result_benchmark": "benchmark_b_pooled",
        "algorithm_learning_policy": "no_learning_repository_behavior_preserved",
        "source_panel": source.relative_to(ROOT).as_posix(),
        "source_panel_sha256": sha256_file(source),
        "source_panel_metadata": source_metadata.relative_to(ROOT).as_posix(),
        "source_panel_metadata_sha256": sha256_file(source_metadata),
        "context_contract": contract_path.relative_to(ROOT).as_posix(),
        "context_contract_sha256": contract.sha256,
        "panel": panel_path.name,
        "panel_sha256": panel_hash,
        "event_ledger": ledger_path.name,
        "event_ledger_sha256": ledger_hash,
        "panel_rows": int(len(panel_out)),
        "ledger_rows": int(len(ledger)),
        "entity_count": 51,
        "components": list(COMPONENTS),
        "modes": {str(mode): list(map(int, spec["horizons"])) for mode, spec in config["modes"].items()},
        "split_basis": "forecast_origin",
        "split_contract": {key: str(config[key]) for key in ("first_origin", "train_end", "validation_start", "validation_end", "embargo_start", "embargo_end", "test_start", "test_end")},
        "selection_freeze_time": str(config["selection_freeze_time"]),
        "same_time_order": "ingest_release_then_freeze_selection_then_issue_forecast",
        "runner_stages": list(config["runner_stages"]),
        "forecast_issuance_splits": list(config["forecast_issuance_splits"]),
        "parameter_selection_splits": list(config["parameter_selection_splits"]),
        "result_metric_splits": list(config["result_metric_splits"]),
        "posterior_update_splits": list(config["posterior_update_splits"]),
        "embargo_policy": str(config["embargo_policy"]),
        "release_lag_steps": 1,
        "release_time_policy": str(config["release_policy"]),
        "fixed_vintage_available": False,
        "minimum_origin_history_steps": int(config["minimum_origin_history_steps"]),
        "minimum_origin_history_role": str(config["minimum_origin_history_role"]),
        "adapter_input_window_changed": False,
        "router_context_policy": "task_query_plus_structured_pre_t_sel_causal_history_summary",
        "adapter_input_policy": "origin_visible_values_timestamps_release_times_and_missing_masks",
        "stream_release_policy": {
            group: {
                "columns": list(contract.stream_columns[group]),
                "release_lag_days": int(spec["release_lag_days"]),
                "release_time_status": str(spec["release_time_status"]),
                "formal_status": str(spec["formal_status"]),
            }
            for group, spec in contract.payload["streams"].items()
        },
        "excluded_streams": contract.payload["excluded_streams"],
        "posterior_update_inputs": "frozen_forecast_archive_and_released_batch_only",
        "posterior_current_x_forbidden": True,
        "benchmark_b_input_fingerprint": input_fingerprint(
            contract_sha256=contract.sha256,
            panel_sha256=panel_hash,
            ledger_sha256=ledger_hash,
            frozen_selection_packet_sha256="",
            candidate_registry_sha256="",
        ),
        "stale_artifact_policy": "fail_closed_do_not_reuse_forecast_posterior_or_agent_output",
        "component_resampling": "coupled",
        "split_row_counts": {str(key): int(value) for key, value in ledger["split"].value_counts().sort_index().items()},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        output = build(args.config.resolve())
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc), "formal_experiment_run": False}, indent=2))
        return 2
    print(json.dumps({"status": "PASS", "formal_experiment_run": False, "output": str(output.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
