#!/usr/bin/env python3
""






from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASTER_SRC = ROOT / "code/caster/src"
if str(CASTER_SRC) not in sys.path:
    sys.path.insert(0, str(CASTER_SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from caster.bridge.likelihood import bridge_r_key_series              
from formal_candidate_bank import (              
    FORMAL_RESULT_ELIGIBLE_CANDIDATE_COUNT,
    FORMAL_RESULT_EXCLUDED_MODEL_IDS,
)


SCHEMA = "caster_censoring_support_manifest_v1"
BOUND_SCOPE = "eligible27_train"
ELIGIBILITY_COLUMN = "eligible_for_caster_top_k"
ARCHIVE_COLUMNS = (
    "forecast_id",
    "model_id",
    "pred_mean",
    "pred_var",
    "mode",
    "component",
    "horizon",
)
LEDGER_COLUMNS = (
    "forecast_id",
    "split",
    "observed_value",
    "observed_mask",
    "mode",
    "component",
    "horizon",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "t", "yes", "y"})
    )


def _available_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _max_by_key(
    frame: pd.DataFrame,
    value: pd.Series,
) -> dict[str, float]:
    keyed = pd.DataFrame(
        {
            "bridge_r_key": bridge_r_key_series(frame).astype(str),
            "value": pd.to_numeric(value, errors="coerce"),
        }
    ).dropna(subset=["value"])
    if keyed.empty:
        return {}
    return {
        str(key): float(item)
        for key, item in keyed.groupby(
            "bridge_r_key", sort=True
        )["value"].max().items()
    }


def build_manifest(
    *,
    ledger_path: Path,
    archive_path: Path,
    eligibility_path: Path,
    task_id: str,
    predictive_standard_deviations: float = 4.0,
    minimum_upper_raw: float = 1.0,
    chunksize: int = 250_000,
    declared_archive_sha256: str = "",
    candidate_manifest_path: Path,
) -> dict[str, object]:
    q = float(predictive_standard_deviations)
    floor = float(minimum_upper_raw)
    if not math.isfinite(q) or q < 0.0:
        raise ValueError(
            "predictive_standard_deviations must be finite and nonnegative"
        )
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("minimum_upper_raw must be finite and positive")
    if int(chunksize) < 1:
        raise ValueError("chunksize must be positive")

    for path in (ledger_path, archive_path, eligibility_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    ledger_columns = _available_columns(ledger_path)
    missing_ledger = sorted(set(LEDGER_COLUMNS) - set(ledger_columns))
    if missing_ledger:
        raise ValueError(
            f"support ledger missing required columns {missing_ledger}"
        )
    ledger = pd.read_csv(
        ledger_path,
        usecols=list(LEDGER_COLUMNS),
        low_memory=False,
    )
    train = ledger[ledger["split"].astype(str).eq("train")].copy()
    if train.empty:
        raise ValueError("eligible27 support ledger has no train rows")
    train["forecast_id"] = train["forecast_id"].astype(str)
    if train["forecast_id"].duplicated().any():
        raise ValueError("eligible27 support ledger has duplicate train forecast_id")
    train_ids = set(train["forecast_id"])
    observed = truthy(train["observed_mask"])
    observed_max = _max_by_key(
        train.loc[observed],
        pd.to_numeric(
            train.loc[observed, "observed_value"], errors="coerce"
        ).clip(lower=0.0),
    )

    eligibility = pd.read_csv(eligibility_path, low_memory=False)
    required_eligibility = {"task_id", "model_id", ELIGIBILITY_COLUMN}
    if missing := sorted(required_eligibility - set(eligibility.columns)):
        raise ValueError(
            f"support eligibility file missing required columns {missing}"
        )
    eligibility = eligibility[
        eligibility["task_id"].astype(str).eq(str(task_id))
        & truthy(eligibility[ELIGIBILITY_COLUMN])
    ].copy()
    model_ids = sorted(
        eligibility["model_id"].dropna().astype(str).drop_duplicates()
    )
    if len(model_ids) != FORMAL_RESULT_ELIGIBLE_CANDIDATE_COUNT:
        raise ValueError(
            "eligible support requires exactly "
            f"{FORMAL_RESULT_ELIGIBLE_CANDIDATE_COUNT} models; "
            f"found {len(model_ids)}"
        )
    excluded_present = set(model_ids) & set(FORMAL_RESULT_EXCLUDED_MODEL_IDS)
    if excluded_present:
        raise ValueError(
            f"eligible support contains policy-excluded models: "
            f"{sorted(excluded_present)}"
        )
    model_set = set(model_ids)

    archive_columns = _available_columns(archive_path)
    required_archive = set(ARCHIVE_COLUMNS) - {"mode"}
    if missing := sorted(required_archive - set(archive_columns)):
        raise ValueError(
            f"support archive missing required columns {missing}"
        )
    usecols = [
        column for column in ARCHIVE_COLUMNS if column in archive_columns
    ]
    envelope_max: dict[str, float] = {}
    rows_used = 0
    rows_by_model = {model_id: 0 for model_id in model_ids}
    forecast_ids_by_model = {
        model_id: set() for model_id in model_ids
    }
    nonfinite_rows = 0
    for chunk in pd.read_csv(
        archive_path,
        usecols=usecols,
        chunksize=int(chunksize),
        low_memory=False,
    ):
        keep = (
            chunk["forecast_id"].astype(str).isin(train_ids)
            & chunk["model_id"].astype(str).isin(model_set)
        )
        selected = chunk.loc[keep].copy()
        if selected.empty:
            continue
        selected["forecast_id"] = selected["forecast_id"].astype(str)
        selected["model_id"] = selected["model_id"].astype(str)
        mean = pd.to_numeric(
            selected["pred_mean"], errors="coerce"
        ).clip(lower=0.0)
        variance = pd.to_numeric(
            selected["pred_var"], errors="coerce"
        ).clip(lower=0.0)
        finite = np.isfinite(mean.to_numpy(dtype=float)) & np.isfinite(
            variance.to_numpy(dtype=float)
        )
        nonfinite_rows += int((~finite).sum())
        if not bool(finite.any()):
            continue
        selected = selected.loc[finite].copy()
        envelope = (
            mean.loc[finite]
            + q * np.sqrt(variance.loc[finite].to_numpy(dtype=float))
        )
        maxima = _max_by_key(selected, pd.Series(envelope, index=selected.index))
        for key, value in maxima.items():
            envelope_max[key] = max(envelope_max.get(key, 0.0), float(value))
        counts = selected["model_id"].value_counts()
        for model_id, count in counts.items():
            rows_by_model[str(model_id)] += int(count)
        for model_id, model_rows in selected.groupby("model_id", sort=False):
            model_id = str(model_id)
            ids = model_rows["forecast_id"].astype(str)
            if ids.duplicated().any():
                raise ValueError(
                    "eligible27 support archive has duplicate "
                    f"(model_id, forecast_id) pairs for {model_id}"
                )
            new_ids = set(ids)
            already_seen = forecast_ids_by_model[model_id]
            if already_seen.intersection(new_ids):
                raise ValueError(
                    "eligible27 support archive has duplicate "
                    f"(model_id, forecast_id) pairs across chunks for {model_id}"
                )
            already_seen.update(new_ids)
        rows_used += int(len(selected))

    expected_rows = int(len(train_ids) * len(model_ids))
    if nonfinite_rows:
        raise ValueError(
            f"eligible27 support archive has {nonfinite_rows} non-finite train rows"
        )
    if rows_used != expected_rows:
        raise ValueError(
            "eligible27 support archive coverage mismatch: "
            f"expected={expected_rows} found={rows_used}"
        )
    bad_counts = {
        model_id: count
        for model_id, count in rows_by_model.items()
        if count != len(train_ids)
    }
    if bad_counts:
        raise ValueError(
            "eligible27 support archive has incomplete per-model train coverage: "
            f"{dict(list(bad_counts.items())[:5])}"
        )
    incomplete_pairs = {
        model_id: {
            "missing": len(train_ids - ids),
            "unexpected": len(ids - train_ids),
        }
        for model_id, ids in forecast_ids_by_model.items()
        if ids != train_ids
    }
    if incomplete_pairs:
        raise ValueError(
            "eligible27 support archive does not contain exactly one row for "
            "every eligible (model_id, train forecast_id) pair: "
            f"{dict(list(incomplete_pairs.items())[:5])}"
        )

    keys = sorted(set(observed_max) | set(envelope_max))
    if not keys:
        raise ValueError("eligible27 support calibration produced no keys")
    base_upper = {
        key: max(
            floor,
            float(observed_max.get(key, 0.0)),
            float(envelope_max.get(key, 0.0)),
        )
        for key in keys
    }
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in base_upper.values()
    ):
        raise ValueError("eligible27 support produced an invalid upper bound")

    archive_sha256 = str(declared_archive_sha256).strip()
    if archive_sha256 and (
        len(archive_sha256) != 64
        or any(character not in "0123456789abcdef" for character in archive_sha256)
    ):
        raise ValueError("declared archive sha256 must be lowercase hexadecimal")
    if not archive_sha256:
        archive_sha256 = sha256_file(archive_path)
    if not candidate_manifest_path.is_file():
        raise FileNotFoundError(candidate_manifest_path)
    candidate_manifest_sha256 = sha256_file(candidate_manifest_path)
    return {
        "schema": SCHEMA,
        "task_id": str(task_id),
        "bound_scope": BOUND_SCOPE,
        "source_split": "train",
        "eligible_model_column": ELIGIBILITY_COLUMN,
        "eligible_model_count": int(len(model_ids)),
        "eligible_model_ids": model_ids,
        "excluded_model_ids": list(FORMAL_RESULT_EXCLUDED_MODEL_IDS),
        "train_forecast_id_count": int(len(train_ids)),
        "archive_train_rows_used": int(rows_used),
        "expected_archive_train_rows": expected_rows,
        "predictive_standard_deviations": q,
        "minimum_upper_raw": floor,
        "base_upper_raw_by_component": {
            key: float(value) for key, value in sorted(base_upper.items())
        },
        "observed_train_max_by_component": {
            key: float(value) for key, value in sorted(observed_max.items())
        },
        "forecast_envelope_train_max_by_component": {
            key: float(value) for key, value in sorted(envelope_max.items())
        },
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": sha256_file(ledger_path),
        "archive_path": str(archive_path.resolve()),
        "archive_sha256": archive_sha256,
        "eligibility_path": str(eligibility_path.resolve()),
        "eligibility_sha256": sha256_file(eligibility_path),
        "candidate_manifest_path": str(candidate_manifest_path.resolve()),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "test_targets_used": 0,
        "selection_outcomes_used": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--eligibility", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--predictive-standard-deviations", type=float, default=4.0)
    parser.add_argument("--minimum-upper-raw", type=float, default=1.0)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--declared-archive-sha256", default="")
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_manifest(
        ledger_path=args.ledger,
        archive_path=args.archive,
        eligibility_path=args.eligibility,
        task_id=str(args.task_id),
        predictive_standard_deviations=float(
            args.predictive_standard_deviations
        ),
        minimum_upper_raw=float(args.minimum_upper_raw),
        chunksize=int(args.chunksize),
        declared_archive_sha256=str(args.declared_archive_sha256),
        candidate_manifest_path=args.candidate_manifest,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"support_manifest={args.out}")
    print(f"task_id={payload['task_id']}")
    print(f"eligible_models={payload['eligible_model_count']}")
    print(f"train_rows={payload['archive_train_rows_used']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
