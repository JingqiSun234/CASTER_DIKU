from __future__ import annotations

import importlib
import json
from importlib import metadata
from pathlib import Path


REQUIRED_PACKAGES = {
    "statsforecast": "2.0.3",
    "prophet": "1.3.0",
}

PACKAGE_IMPORTS = {
    "statsforecast": "statsforecast",
    "prophet": "prophet",
}


class DependencyError(RuntimeError):
    pass


def inspect_forecasting_dependencies(packages: list[str] | None = None) -> list[dict[str, object]]:
    packages = packages or list(REQUIRED_PACKAGES)
    rows: list[dict[str, object]] = []
    for package in packages:
        import_name = PACKAGE_IMPORTS.get(package, package)
        expected = REQUIRED_PACKAGES.get(package, "")
        row: dict[str, object] = {
            "package": package,
            "import_name": import_name,
            "expected_version": expected,
            "installed": False,
            "version": "",
            "import_path": "",
            "status": "missing",
            "error": "",
        }
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "")
            if not version:
                try:
                    version = metadata.version(package)
                except metadata.PackageNotFoundError:
                    version = ""
            row.update({
                "installed": True,
                "version": str(version),
                "import_path": str(getattr(module, "__file__", "")),
                "status": "ok" if not expected or str(version) == expected else "version_mismatch",
            })
        except Exception as exc:                                                                  
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def write_dependency_report(out_path: Path, packages: list[str] | None = None) -> list[dict[str, object]]:
    rows = inspect_forecasting_dependencies(packages=packages)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return rows


def require_forecasting_dependencies(out_path: Path, packages: list[str]) -> list[dict[str, object]]:
    rows = write_dependency_report(out_path, packages=packages)
    bad = [row for row in rows if row["status"] != "ok"]
    if bad:
        raise DependencyError(f"forecasting dependency check failed: {json.dumps(bad, sort_keys=True)}")
    return rows

