from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _shared import LINEAR_METRICS, macro, prepare, read_csv, unique_text, weighted, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize test forecast metric slices.")
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = prepare(pd.concat([read_csv(path) for path in args.input], ignore_index=True))
    output: list[dict[str, object]] = []
    for (task, method), group in rows.groupby(["task", "method"], sort=True, dropna=False):
        overall = macro(group)
        short = macro(group[group["horizon_group"].eq("short")])
        long = macro(group[group["horizon_group"].eq("long")])
        row: dict[str, object] = {
            "task": task,
            "method_group": unique_text(group["method_group"]),
            "method": method,
            "rmse": overall["rmse"],
            "short_rmse": short["rmse"],
            "long_rmse": long["rmse"],
            "mae": overall["mae"],
            "nll": overall["nll"],
            "wis": overall["wis"],
            "coverage_90": overall["coverage_90"],
            "width_90": overall["width_90"],
            "n_total": float(group["n"].sum()),
            "metric_rows": int(len(group)),
        }
        for name in ("coverage_50", "width_50"):
            row[name] = overall.get(name, float("nan"))
        for name in ("coverage_50", "width_50", "coverage_90", "width_90"):
            row[f"endpoint_{name}"] = weighted(group[name], group["n"]) if name in group.columns else float("nan")
        output.append(row)
    result = pd.DataFrame(output).sort_values(["task", "method"], kind="stable")
    write_csv(result, args.output)
    print(f"rows={len(result)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
