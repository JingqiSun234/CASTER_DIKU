from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _shared import endpoint_macro, prepare, read_csv, unique_text, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize scores at each forecast endpoint.")
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = prepare(pd.concat([read_csv(path) for path in args.input], ignore_index=True))
    output: list[dict[str, object]] = []
    keys = ["task", "method", "horizon"]
    for (task, method, horizon), group in rows.groupby(keys, sort=True, dropna=False):
        values = endpoint_macro(group)
        output.append(
            {
                "task": task,
                "method_group": unique_text(group["method_group"]),
                "method": method,
                "horizon": int(horizon),
                **values,
                "n_total": float(group["n"].sum()),
                "metric_rows": int(len(group)),
            }
        )
    result = pd.DataFrame(output).sort_values(keys, kind="stable")
    write_csv(result, args.output)
    print(f"rows={len(result)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
