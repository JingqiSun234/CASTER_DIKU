#!/usr/bin/env python3
""





from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NEWMETHOD_ROOT = ROOT / "code/caster"
sys.path.insert(0, str(NEWMETHOD_ROOT / "src"))

from caster.data import add_task_columns, filter_ledger_archive_for_task, task_from_args, task_metadata              
from caster.forecast import (              
    FORECAST_ARCHIVE_COLUMNS,
    build_normal_forecast_draws,
    validate_forecast_archive,
    validate_forecast_draws,
    write_forecast_archive,
    write_forecast_draws,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_selection(path: Path) -> list[str]:
    selection = pd.read_csv(path)
    if "model_id" not in selection.columns:
        raise SystemExit(f"selection missing model_id column: {path}")
    if "rank" in selection.columns:
        selection = selection.sort_values("rank", kind="mergesort")
    model_ids = selection["model_id"].dropna().astype(str).drop_duplicates().tolist()
    if not model_ids:
        raise SystemExit(f"selection has no model_id rows: {path}")
    return model_ids


def _coverage_violations(archive: pd.DataFrame, ledger: pd.DataFrame, selected_model_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ledger_ids = set(ledger["forecast_id"].astype(str))
    for model_id in selected_model_ids:
        model_rows = archive[archive["model_id"].astype(str) == str(model_id)]
        archive_ids = set(model_rows["forecast_id"].astype(str))
        missing = sorted(ledger_ids - archive_ids)
        extra = sorted(archive_ids - ledger_ids)
        if missing:
            rows.append({"model_id": model_id, "violation": "missing_ledger_predictions", "details": ",".join(missing[:10])})
        if extra:
            rows.append({"model_id": model_id, "violation": "forecast_id_not_in_ledger", "details": ",".join(extra[:10])})
        if len(model_rows) != len(ledger_ids):
            rows.append({"model_id": model_id, "violation": "row_count_mismatch", "details": f"expected={len(ledger_ids)} actual={len(model_rows)}"})
    return pd.DataFrame(rows, columns=["model_id", "violation", "details"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, action="append", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws-out", type=Path, default=None)
    parser.add_argument("--n-draws", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-id", default="")
    parser.add_argument("--target-components", default="")
    parser.add_argument("--posterior-scope", default="", choices=["", "component_stratified", "pooled_sensitivity", "pooled_shared_posterior"])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--violations", type=Path, default=None)
    parser.add_argument("--derived-view", action="store_true")
    args = parser.parse_args(argv)

    missing = [path for path in [args.ledger, args.selection, *args.source_archive] if not path.exists()]
    if missing:
        raise SystemExit("shared archive assembly missing inputs: " + ", ".join(str(path) for path in missing))

    ledger = pd.read_csv(args.ledger)
    if "forecast_id" not in ledger.columns:
        raise SystemExit(f"ledger missing forecast_id column: {args.ledger}")
    task = task_from_args(
        task_id=args.task_id,
        target_components=args.target_components,
        posterior_scope=args.posterior_scope,
        dataset=ledger["dataset"].dropna().astype(str).iloc[0] if "dataset" in ledger.columns and not ledger.empty else "",
    )

    source_records: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for source in args.source_archive:
        frame = pd.read_csv(source)
        missing_cols = sorted(set(FORECAST_ARCHIVE_COLUMNS) - set(frame.columns))
        if missing_cols:
            raise SystemExit(f"source archive missing columns {missing_cols}: {source}")
        frames.append(frame[FORECAST_ARCHIVE_COLUMNS + [c for c in frame.columns if c not in FORECAST_ARCHIVE_COLUMNS]].copy())
        source_records.append({"path": str(source), "sha256": _sha256_file(source), "rows": int(len(frame))})
    archive = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FORECAST_ARCHIVE_COLUMNS)

    ledger, archive = filter_ledger_archive_for_task(ledger, archive, task)
    ledger_ids = set(ledger["forecast_id"].astype(str))
    archive = archive[archive["forecast_id"].astype(str).isin(ledger_ids)].copy()

    selected_model_ids = _read_selection(args.selection)
    available = set(archive["model_id"].astype(str))
    missing_models = [model_id for model_id in selected_model_ids if model_id not in available]
    if missing_models:
        raise SystemExit(f"shared archive missing selected model_id values: {missing_models}")
    archive = archive[archive["model_id"].astype(str).isin(set(selected_model_ids))].copy()
    order = {model_id: i for i, model_id in enumerate(selected_model_ids)}
    archive["_model_order"] = archive["model_id"].astype(str).map(order)
    sort_cols = ["_model_order", "forecast_origin", "target_time", "entity_id", "component", "horizon", "forecast_id"]
    archive = archive.sort_values([c for c in sort_cols if c in archive.columns], kind="mergesort").drop(columns=["_model_order"])
    archive = add_task_columns(archive, task)

    violations = validate_forecast_archive(archive, ledger)
    coverage = _coverage_violations(archive, ledger, selected_model_ids)
    if not coverage.empty:
        violations = pd.concat([violations, coverage], ignore_index=True)
    violations_path = args.violations or args.out.with_name(args.out.stem + "_violations.csv")
    if not violations.empty:
        violations_path.parent.mkdir(parents=True, exist_ok=True)
        violations.to_csv(violations_path, index=False)
        raise SystemExit(f"shared archive assembly validation failed; see {violations_path}")

    archive_path = write_forecast_archive(archive, args.out)

    draws_path = ""
    if int(args.n_draws) > 0:
        draws_out = args.draws_out or args.out.with_name("forecast_draws.csv")
        draws = build_normal_forecast_draws(archive, n_draws=int(args.n_draws), seed=int(args.seed))
        draw_violations = validate_forecast_draws(draws, archive)
        if not draw_violations.empty:
            draw_violations_path = draws_out.with_name(draws_out.stem + "_violations.csv")
            draw_violations_path.parent.mkdir(parents=True, exist_ok=True)
            draw_violations.to_csv(draw_violations_path, index=False)
            raise SystemExit(f"forecast draws validation failed; see {draw_violations_path}")
        draws_path = str(write_forecast_draws(draws, draws_out))

    manifest_path = args.manifest or args.out.with_name(args.out.stem + "_manifest.json")
    manifest = {
        "archive_path": str(archive_path),
        "archive_sha256": _sha256_file(Path(archive_path)),
        "draws_path": draws_path,
        "draws_sha256": _sha256_file(Path(draws_path)) if draws_path else "",
        "draw_generation_distribution": (
            "nonnegative_normal_moment_projection" if draws_path else "not_materialized"
        ),
        "draw_seed_policy": (
            "sha256_global_seed_forecast_id_model_id_particle_id_v1"
            if draws_path
            else "not_materialized"
        ),
        "source_archives": source_records,
        "derived_from": [record["path"] for record in source_records],
        "derived_view": bool(args.derived_view or len(source_records) > 1),
        "candidate_training_performed": False,
        "adapter_training_performed": False,
        "adapter_forecast_generation_performed": False,
        "selection": str(args.selection),
        "selection_sha256": _sha256_file(args.selection),
        "selected_model_ids": selected_model_ids,
        "models": int(archive["model_id"].nunique()),
        "ledger_path": str(args.ledger),
        "ledger_sha256": _sha256_file(args.ledger),
        "ledger_rows": int(len(ledger)),
        "ledger_split_counts": {
            str(key): int(value)
            for key, value in ledger["split"].astype(str).value_counts().sort_index().items()
        },
        "embargo_rows": int(ledger["split"].astype(str).eq("embargo").sum()),
        "embargo_forecast_coverage_required": True,
        "embargo_metric_eligible": False,
        "archive_rows": int(len(archive)),
        "n_draws": int(args.n_draws),
    }
    manifest.update(task_metadata(task, ledger))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"archive={archive_path}")
    if draws_path:
        print(f"draws={draws_path}")
    print(f"manifest={manifest_path}")
    print(f"rows={len(archive)} models={archive['model_id'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
