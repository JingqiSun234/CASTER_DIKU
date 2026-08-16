from __future__ import annotations
import os as _os
import sys as _sys
from pathlib import Path as _Path

if __name__ == "__main__" and not _os.environ.get("CASTER_alternate_PHASE20_BUILDER"):
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from build_selected_forecast_archive_impl import main as _phase20_main

    raise SystemExit(_phase20_main())

from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from caster.forecast import validate_forecast_archive, write_forecast_archive, build_normal_forecast_draws, validate_forecast_draws, write_forecast_draws
from caster.models import read_registry, instantiate_adapters_from_registry
from caster.utils import RuntimeLogger, write_timing_log


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_selected_coverage(
    archive: pd.DataFrame,
    ledger: pd.DataFrame,
    selected_model_ids: list[str],
) -> pd.DataFrame:
    ledger_ids = set(ledger["forecast_id"].astype(str))
    rows: list[dict[str, object]] = []
    for model_id in selected_model_ids:
        model_rows = archive[archive["model_id"].astype(str) == str(model_id)]
        archive_ids = set(model_rows["forecast_id"].astype(str))
        missing = sorted(ledger_ids - archive_ids)
        extra = sorted(archive_ids - ledger_ids)
        if missing:
            rows.append({
                "model_id": model_id,
                "violation": "missing_ledger_predictions",
                "details": ",".join(missing[:10]),
            })
        if extra:
            rows.append({
                "model_id": model_id,
                "violation": "forecast_id_not_in_ledger",
                "details": ",".join(extra[:10]),
            })
        if len(model_rows) != len(ledger_ids):
            rows.append({
                "model_id": model_id,
                "violation": "row_count_mismatch",
                "details": f"expected={len(ledger_ids)} actual={len(model_rows)}",
            })
    return pd.DataFrame(rows, columns=["model_id", "violation", "details"])


def main() -> None:
    ap = ArgumentParser(description="Build immutable forecast archive for selected CASTER candidate particles.")
    ap.add_argument("--panel", required=True, help="Panel CSV used to initialize each candidate adapter.")
    ap.add_argument("--ledger", required=True, help="Event ledger CSV.")
    ap.add_argument("--registry", required=True, help="Model registry YAML/CSV/JSON.")
    ap.add_argument("--selection", required=True, help="Top-K candidate selection CSV with model_id column.")
    ap.add_argument("--out", required=True, help="Output forecast archive CSV path.")
    ap.add_argument("--draws-out", default="", help="Optional output forecast draws CSV path.")
    ap.add_argument("--n-draws", type=int, default=0, help="If >0, create normal draws from archive means/variances.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers for adapter forecasts (0=serial).")
    args = ap.parse_args()

    timer = RuntimeLogger()
    with timer.measure("load_inputs"):
        panel = pd.read_csv(args.panel)
        ledger = pd.read_csv(args.ledger)
        registry = read_registry(args.registry)
        selection = pd.read_csv(args.selection)
        if "model_id" not in selection.columns:
            raise SystemExit("selection must contain model_id column")
        model_ids = selection["model_id"].astype(str).tolist()
        registry_ids = set(registry["model_id"].astype(str))
        missing_registry = [m for m in model_ids if m not in registry_ids]
        if missing_registry:
            raise SystemExit(f"selection contains model_id not in registry: {missing_registry}")

    archives = []
    with timer.measure("adapter_forecasts"):
        adapters = instantiate_adapters_from_registry(registry, model_ids=model_ids)
        adapter_ids = [a.model_id for a in adapters]
        if adapter_ids != model_ids:
            raise SystemExit(f"adapter order/coverage mismatch selected={model_ids} instantiated={adapter_ids}")

        def _run_adapter(adapter):
            state = adapter.initialize(panel, seed=args.seed)
            return adapter.model_id, adapter.forecast_ledger(state, ledger)

        workers = args.workers if args.workers > 0 else len(adapters)
        results: dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_adapter, a): a.model_id for a in adapters}
            for future in as_completed(futures):
                model_id, df = future.result()
                results[model_id] = df
                print(f"adapter_done={model_id}", flush=True)
        archives = [results[mid] for mid in model_ids]
    archive = pd.concat(archives, ignore_index=True) if archives else pd.DataFrame()
    violations = validate_forecast_archive(archive, ledger)
    if not violations.empty:
        viol_path = Path(args.out).with_name(Path(args.out).stem + "_violations.csv")
        violations.to_csv(viol_path, index=False)
        raise SystemExit(f"forecast archive validation failed; see {viol_path}")
    coverage_violations = _validate_selected_coverage(archive, ledger, model_ids)
    if not coverage_violations.empty:
        viol_path = Path(args.out).with_name(Path(args.out).stem + "_coverage_violations.csv")
        coverage_violations.to_csv(viol_path, index=False)
        raise SystemExit(f"forecast archive coverage validation failed; see {viol_path}")
    out_path = write_forecast_archive(archive, args.out)

    if args.n_draws > 0:
        draws_out = args.draws_out or str(Path(args.out).with_name("forecast_draws.csv"))
        with timer.measure("forecast_draws"):
            draws = build_normal_forecast_draws(archive, n_draws=args.n_draws, seed=args.seed)
            draw_violations = validate_forecast_draws(draws, archive)
            if not draw_violations.empty:
                viol_path = Path(draws_out).with_name(Path(draws_out).stem + "_violations.csv")
                draw_violations.to_csv(viol_path, index=False)
                raise SystemExit(f"forecast draws validation failed; see {viol_path}")
            write_forecast_draws(draws, draws_out)
    else:
        draws_out = ""

    timing_path = Path(args.out).with_name("build_forecast_archive_timing.json")
    write_timing_log(timer.summary(seed=args.seed), timing_path)
    manifest_path = Path(args.out).with_name("forecast_archive_manifest.json")
    manifest = {
        "archive_path": str(out_path),
        "archive_sha256": _sha256_file(Path(out_path)),
        "draws_path": str(draws_out) if draws_out else "",
        "draws_sha256": _sha256_file(Path(draws_out)) if draws_out else "",
        "ledger_rows": int(len(ledger)),
        "archive_rows": int(len(archive)),
        "selected_model_ids": model_ids,
        "models": int(archive["model_id"].nunique() if not archive.empty else 0),
        "coverage_rule": "each selected model_id has exactly one prediction for every ledger forecast_id",
        "immutable_rule": "archive and draws are content-addressed by SHA256 in this manifest",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"archive={out_path}")
    print(f"rows={len(archive)} models={archive['model_id'].nunique() if not archive.empty else 0}")
    if draws_out:
        print(f"draws={draws_out}")
    print(f"manifest={manifest_path}")
    print(f"archive_sha256={manifest['archive_sha256']}")

if __name__ == "__main__":
    main()
