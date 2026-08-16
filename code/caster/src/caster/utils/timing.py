from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import json, platform, time

class RuntimeLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, float | str]] = []
    @contextmanager
    def measure(self, name: str):
        start = time.perf_counter()
        yield
        self.records.append({"name": name, "seconds": time.perf_counter() - start})
    def summary(self, seed: int | None = None) -> dict[str, object]:
        total = sum(float(r["seconds"]) for r in self.records)
        return {"records": self.records, "total_sec": total, "seed": seed, "python": platform.python_version(), "platform": platform.platform()}

def write_timing_log(summary: dict[str, object], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return path
