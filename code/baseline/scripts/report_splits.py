#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caster_baselines.data_validation import caster_root_from_baseline, resolve_manifest_path


def parse_count_string(text: str) -> dict[str, int]:
    if not text or text == "NA":
        return {}
    out: dict[str, int] = {}
    for part in str(text).split(";"):
        if not part:
            continue
        key, value = part.rsplit(":", 1)
        out[key] = int(value)
    return out


def entity_column(df: pd.DataFrame) -> str | None:
    for col in ("entity_id", "jurisdiction", "location", "node_id", "node_index"):
        if col in df.columns:
            return col
    return None


def date_range(series: pd.Series) -> tuple[str, str]:
    parsed = pd.to_datetime(series, errors="coerce").dropna()
    if parsed.empty:
        return "NA", "NA"
    return parsed.min().strftime("%Y-%m-%d"), parsed.max().strftime("%Y-%m-%d")


def semicolon_counts(series: pd.Series) -> str:
    counts = series.fillna("NA").astype(str).value_counts().to_dict()
    def key_fn(item: tuple[str, int]) -> tuple[int, object]:
        try:
            return (0, int(float(item[0])))
        except ValueError:
            return (1, item[0])
    return ";".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=key_fn))


def split_rows(manifest_path: Path, root: Path = ROOT) -> tuple[pd.DataFrame, list[str]]:
    caster_root = caster_root_from_baseline(root)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for _, mrow in manifest.iterrows():
        ledger_path = resolve_manifest_path(mrow["ledger_path"], caster_root, root)
        ledger = pd.read_csv(ledger_path, keep_default_na=False)
        e_col = entity_column(ledger)
        expected_split_counts = parse_count_string(str(mrow.get("split_counts", "")))
        actual_split_counts = ledger["split"].astype(str).value_counts().to_dict() if "split" in ledger.columns else {}
        if expected_split_counts != actual_split_counts:
            failures.append(f"{mrow['dataset_key']} split_counts mismatch expected={expected_split_counts} actual={actual_split_counts}")
        group_cols = [col for col in ("dataset", "split", "mode", "component", "horizon") if col in ledger.columns]
        for keys, g in ledger.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {
                "dataset_key": mrow["dataset_key"],
                "scope": mrow["scope"],
                "rows": int(len(g)),
                "entity_count": int(g[e_col].astype(str).nunique()) if e_col else 0,
                "entity_col": e_col or "NA",
            }
            row.update(dict(zip(group_cols, keys)))
            for source_col, prefix in (("forecast_origin", "origin"), ("target_time", "target"), ("release_time", "release")):
                if source_col in g.columns:
                    row[f"{prefix}_min"], row[f"{prefix}_max"] = date_range(g[source_col])
            rows.append(row)
    summary = pd.DataFrame(rows).fillna("NA")
    sort_cols = [c for c in ("dataset_key", "dataset", "split", "mode", "component", "horizon") if c in summary.columns]
    return summary.sort_values(sort_cols).reset_index(drop=True), failures


def render_report(summary: pd.DataFrame, failures: list[str]) -> str:
    lines = ["# Split Report", ""]
    status = "PASS" if not failures else "FAIL"
    lines.append(f"- status: `{status}`")
    if failures:
        lines.append(f"- failures: `{failures}`")
    lines.extend(["", "## Split detail", ""])
    detail_cols = [c for c in ("dataset_key", "dataset", "split", "mode", "component", "horizon", "rows", "entity_count", "origin_min", "origin_max", "target_min", "target_max") if c in summary.columns]
    lines.append("| " + " | ".join(detail_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(detail_cols)) + "|")
    for _, row in summary[detail_cols].iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in detail_cols) + " |")
    lines.extend(["", "## Horizons by mode", ""])
    mode_cols = [c for c in ("dataset_key", "dataset", "mode", "component", "split") if c in summary.columns]
    lines.append("| " + " | ".join(mode_cols + ["horizon_counts", "rows"]) + " |")
    lines.append("|" + "|".join(["---"] * (len(mode_cols) + 2)) + "|")
    for keys, g in summary.groupby(mode_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = dict(zip(mode_cols, keys))
        horizon_counts = semicolon_counts(g.loc[g.index.repeat(g["rows"].astype(int)), "horizon"])
        row_values = [str(values[c]) for c in mode_cols] + [horizon_counts, str(int(g["rows"].sum()))]
        lines.append("| " + " | ".join(row_values) + " |")
    lines.append("")
    return "\n".join(lines)


def write_split_report(manifest_path: Path, out_path: Path, root: Path = ROOT) -> pd.DataFrame:
    summary, failures = split_rows(manifest_path, root=root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(summary, failures), encoding="utf-8")
    if failures:
        raise RuntimeError("; ".join(failures))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test split report from curated subset event ledgers.")
    parser.add_argument("--manifest", default="data/subset_manifest.csv")
    parser.add_argument("--out", default="reports/split_report.md")
    args = parser.parse_args()
    summary = write_split_report(Path(args.manifest), Path(args.out), root=ROOT)
    print(f"ok out={args.out} rows={len(summary)} datasets={','.join(sorted(summary['dataset_key'].astype(str).unique()))}")


if __name__ == "__main__":
    main()
