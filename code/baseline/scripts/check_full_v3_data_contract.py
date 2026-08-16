#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caster_baselines.data_validation import caster_root_from_baseline, resolve_manifest_path, sha256_file              


PROTOCOL = "benchmark_protocol_v3_direct_rollout"
BENCHMARK_B_PROTOCOL = "caster_data_protocol_v26_1"
BENCHMARK_B_TASK = "benchmark_b_pooled"
EXPECTED = {
    "benchmark_a": {
        "protocol": PROTOCOL,
        "rows": 140_330,
        "natural_rows": 98_231,
        "split_rows": {"train": 49_590, "val": 48_860, "embargo": 24_430, "test": 17_450},
        "natural_split_rows": {"train": 34_713, "val": 34_202, "embargo": 17_101, "test": 12_215},
        "modes": {"direct_1d3d7d": {1, 3, 7}, "rollout_7d": set(range(1, 8))},
        "strategies": {"direct_1d3d7d": "direct", "rollout_7d": "recursive_rollout"},
        "origins": {
            "train": ("2020-03-15", "2020-04-09"),
            "val": ("2020-04-10", "2020-04-23"),
            "embargo": ("2020-04-24", "2020-04-30"),
            "test": ("2020-05-01", "2020-05-05"),
        },
        "max_val_release": "2020-04-30",
        "min_test_origin": "2020-05-01",
    },
    "benchmark_b": {
        "protocol": BENCHMARK_B_PROTOCOL,
        "rows": 104_652,
        "natural_rows": 69_768,
        "split_rows": {"train": 73_440, "val": 14_076, "embargo": 2_448, "test": 14_688},
        "natural_split_rows": {"train": 48_960, "val": 9_384, "embargo": 1_632, "test": 9_792},
        "modes": {"direct_1w2w": {1, 2}, "rollout_4w": {1, 2, 3, 4}},
        "strategies": {"direct_1w2w": "direct", "rollout_4w": "recursive_rollout"},
        "origins": {
            "train": ("2022-11-26", "2025-03-08"),
            "val": ("2025-03-15", "2025-08-16"),
            "embargo": ("2025-08-23", "2025-09-13"),
            "test": ("2025-09-20", "2026-02-28"),
        },
        "selection_freeze_time": "2025-08-23",
        "max_calibration_release": "2025-08-23",
        "min_test_origin": "2025-09-20",
    },
}


class ContractError(RuntimeError):
    pass


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.astype(str).value_counts().to_dict().items()}


def _row(manifest: pd.DataFrame, dataset_key: str) -> pd.Series:
    rows = manifest[manifest["dataset_key"].astype(str) == dataset_key]
    if len(rows) != 1:
        raise ContractError(f"expected exactly one manifest row for {dataset_key}; got {len(rows)}")
    return rows.iloc[0]


def _assert_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ContractError(f"{label}: expected={expected!r} actual={actual!r}")


def check_dataset(row: pd.Series, dataset_key: str, caster_root: Path) -> list[str]:
    expected = EXPECTED[dataset_key]
    panel_path = resolve_manifest_path(row["panel_path"], caster_root, ROOT)
    ledger_path = resolve_manifest_path(row["ledger_path"], caster_root, ROOT)
    panel = pd.read_csv(panel_path, low_memory=False)
    ledger = pd.read_csv(ledger_path, keep_default_na=False, low_memory=False)
    package_manifest = json.loads((ledger_path.parent / "run_manifest.json").read_text(encoding="utf-8"))

    required = {
        "protocol_version", "natural_event_id", "forecast_id", "forecast_strategy", "mode",
        "split", "entity_id", "component", "forecast_origin", "target_time", "release_time",
        "features_available_until", "horizon", "calibration_eligible",
    }
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ContractError(f"{dataset_key} missing columns: {missing}")
    _assert_equal(set(ledger["protocol_version"].astype(str)), {expected["protocol"]}, f"{dataset_key} protocol")
    _assert_equal(int(row["panel_rows"]), len(panel), f"{dataset_key} manifest panel rows")
    _assert_equal(int(row["ledger_rows"]), len(ledger), f"{dataset_key} manifest ledger rows")
    _assert_equal(str(row["panel_sha256"]), sha256_file(panel_path), f"{dataset_key} manifest panel hash")
    _assert_equal(str(row["ledger_sha256"]), sha256_file(ledger_path), f"{dataset_key} manifest ledger hash")
    _assert_equal(len(ledger), expected["rows"], f"{dataset_key} ledger rows")
    if not ledger["forecast_id"].is_unique or not ledger["event_id"].is_unique:
        raise ContractError(f"{dataset_key} forecast_id/event_id is not unique")

    natural = ledger.drop_duplicates("natural_event_id")
    _assert_equal(len(natural), expected["natural_rows"], f"{dataset_key} natural rows")
    _assert_equal(_counts(ledger["split"]), expected["split_rows"], f"{dataset_key} split rows")
    _assert_equal(_counts(natural["split"]), expected["natural_split_rows"], f"{dataset_key} natural split rows")
    if (ledger.groupby("natural_event_id")["split"].nunique() > 1).any():
        raise ContractError(f"{dataset_key} natural event crosses splits")

    origin = pd.to_datetime(ledger["forecast_origin"], errors="raise")
    target = pd.to_datetime(ledger["target_time"], errors="raise")
    release = pd.to_datetime(ledger["release_time"], errors="raise")
    features = pd.to_datetime(ledger["features_available_until"], errors="raise")
    if not ((features <= origin) & (origin < target) & (target <= release)).all():
        raise ContractError(f"{dataset_key} violates no-leakage date ordering")
    for split, (expected_min, expected_max) in expected["origins"].items():
        values = ledger.loc[ledger["split"].astype(str) == split, "forecast_origin"].astype(str)
        _assert_equal((values.min(), values.max()), (expected_min, expected_max), f"{dataset_key} {split} origin range")
    declared_calibration = ledger["calibration_eligible"].astype(str).str.lower().isin({"1", "true", "yes"})
    if dataset_key == "benchmark_b":
        expected_calibration = (
            ledger["split"].astype(str).eq("val")
            & pd.to_datetime(ledger["release_time"], errors="raise").le(
                pd.Timestamp(expected["selection_freeze_time"])
            )
        )
    else:
        expected_calibration = ledger["split"].astype(str).eq("val")
    if not declared_calibration.equals(expected_calibration):
        raise ContractError(f"{dataset_key} calibration_eligible does not match its frozen selection rule")
    _assert_equal(
        ledger.loc[declared_calibration, "release_time"].astype(str).max(),
        expected.get("max_calibration_release", expected.get("max_val_release")),
        f"{dataset_key} max calibration release",
    )
    _assert_equal(
        ledger.loc[ledger["split"].astype(str) == "test", "forecast_origin"].astype(str).min(),
        expected["min_test_origin"],
        f"{dataset_key} min test origin",
    )
    max_calibration_release = expected.get("max_calibration_release", expected.get("max_val_release"))
    if max_calibration_release >= expected["min_test_origin"]:
        raise ContractError(f"{dataset_key} validation/test embargo is not strict")

    mode_horizons = {
        str(mode): set(pd.to_numeric(group["horizon"], errors="raise").astype(int))
        for mode, group in ledger.groupby("mode")
    }
    _assert_equal(mode_horizons, expected["modes"], f"{dataset_key} mode horizons")
    shared_horizons = set.intersection(*mode_horizons.values())
    multiplicity = ledger.groupby("natural_event_id").size()
    natural_horizon = natural.set_index("natural_event_id")["horizon"].astype(int)
    expected_multiplicity = natural_horizon.map(lambda horizon: 2 if horizon in shared_horizons else 1)
    if not multiplicity.sort_index().equals(expected_multiplicity.sort_index()):
        raise ContractError(
            f"{dataset_key} natural-event multiplicity does not match shared direct/rollout horizons"
        )
    strategy_by_mode = ledger.groupby("mode")["forecast_strategy"].agg(lambda x: set(x.astype(str))).to_dict()
    strategy_by_mode = {str(k): next(iter(v)) if len(v) == 1 else sorted(v) for k, v in strategy_by_mode.items()}
    _assert_equal(strategy_by_mode, expected["strategies"], f"{dataset_key} strategies")

    origin_key = ["entity_id", "component", "forecast_origin", "split"]
    grids = {
        str(mode): set(map(tuple, group[origin_key].astype(str).to_numpy()))
        for mode, group in ledger.groupby("mode")
    }
    if len({frozenset(grid) for grid in grids.values()}) != 1:
        raise ContractError(f"{dataset_key} modes do not share the exact origin grid/split")
    embargo = ledger[ledger["split"].astype(str).eq("embargo")]
    if embargo.empty or declared_calibration.loc[embargo.index].any():
        raise ContractError(f"{dataset_key} embargo origins must exist and be calibration-ineligible")

    _assert_equal(package_manifest.get("protocol_version"), expected["protocol"], f"{dataset_key} package protocol")
    if dataset_key == "benchmark_b":
        _assert_equal(set(ledger["task_id"].astype(str)), {BENCHMARK_B_TASK}, "benchmark_b pooled task")
        _assert_equal(package_manifest.get("task_id"), BENCHMARK_B_TASK, "benchmark_b package task")
        _assert_equal(package_manifest.get("release_lag_steps"), 1, "benchmark_b release lag")
        _assert_equal(package_manifest.get("runner_stages"), ["train", "val", "embargo", "test"], "benchmark_b runner stages")
        _assert_equal(package_manifest.get("posterior_current_x_forbidden"), True, "benchmark_b posterior x gate")
        _assert_equal(package_manifest.get("component_resampling"), "coupled", "benchmark_b component resampling")
        _assert_equal(package_manifest.get("selection_freeze_time"), expected["selection_freeze_time"], "benchmark_b selection freeze")
        _assert_equal(package_manifest.get("panel_sha256"), sha256_file(panel_path), "benchmark_b package panel hash")
        _assert_equal(package_manifest.get("event_ledger_sha256"), sha256_file(ledger_path), "benchmark_b package ledger hash")
        _assert_equal(len(panel.columns), 105, "benchmark_b panel column count")
        required_release_columns = {
            "__release_time__autoregressive", "__release_time__calendar", "__release_time__ed",
            "__release_time__target", "__release_time__wastewater",
        }
        if not required_release_columns <= set(panel.columns):
            raise ContractError("benchmark_b pooled panel lacks stream release columns")
        release_delta = (pd.to_datetime(ledger["release_time"]) - pd.to_datetime(ledger["target_time"])).dt.days
        _assert_equal(set(release_delta.astype(int)), {7}, "benchmark_b target release delta")
        _assert_equal(set(ledger["forecast_issued"].astype(str).str.lower()), {"true"}, "benchmark_b forecast issuance")
        result_metric = ledger["result_metric_eligible"].astype(str).str.lower().isin({"1", "true", "yes"})
        if not result_metric.equals(ledger["split"].astype(str).eq("test")):
            raise ContractError("benchmark_b result_metric_eligible must be test-only")
    else:
        _assert_equal(package_manifest.get("evidence_unit"), "ledger_task_row", f"{dataset_key} evidence unit")
        _assert_equal(package_manifest.get("posterior_natural_event_normalization"), False, f"{dataset_key} posterior normalization")
        _assert_equal(package_manifest.get("same_outcome_multi_strategy_scores_accumulated"), True, f"{dataset_key} composite evidence")
        _assert_equal(
            package_manifest.get("embargo_policy"),
            {
                "forecast_origins_generated": True,
                "selection_eligible": False,
                "calibration_eligible": False,
                "metric_eligible": False,
                "posterior_update_eligible_after_release": True,
            },
            f"{dataset_key} embargo policy",
        )
    return [
        f"{dataset_key}: rows={len(ledger)} natural_events={len(natural)}",
        f"{dataset_key}: modes={sorted(mode_horizons)} shared_origin_grid=true",
        f"{dataset_key}: split_rows={_counts(ledger['split'])}",
        f"{dataset_key}: embargo={max_calibration_release} < {expected['min_test_origin']}",
    ]


def check_contract(manifest_path: Path) -> list[str]:
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    caster_root = caster_root_from_baseline(ROOT)
    notes: list[str] = []
    for dataset_key in ("benchmark_a", "benchmark_b"):
        notes.extend(check_dataset(_row(manifest, dataset_key), dataset_key, caster_root))
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Benchmark A/B full v3 direct/recursive-rollout contract.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    notes = check_contract(Path(args.manifest))
    lines = ["PASS full_v3_data_contract", *[f"- {note}" for note in notes]]
    text = "\n".join(lines) + "\n"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Benchmark v3 data contract\n\n" + text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
