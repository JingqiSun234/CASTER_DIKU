from __future__ import annotations

import importlib
import json
import os
import platform
from importlib import metadata
from pathlib import Path


NEURAL_REQUIRED = {
    "neuralforecast": "3.1.7",
    "torch": "",
    "pytorch_lightning": "",
    "numpy": "",
    "pandas": "",
}


class NeuralDependencyError(RuntimeError):
    pass


def _version(package: str, module: object) -> str:
    version = getattr(module, "__version__", "")
    if version:
        return str(version)
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return ""


def inspect_neuralforecast_dependencies(seed: int = 1, selected_device: str = "cpu") -> dict[str, object]:
    packages = []
    for package, expected in NEURAL_REQUIRED.items():
        row: dict[str, object] = {
            "package": package,
            "import_name": package,
            "expected_version": expected,
            "installed": False,
            "version": "",
            "import_path": "",
            "status": "missing",
            "error": "",
        }
        try:
            module = importlib.import_module(package)
            version = _version(package, module)
            row.update({
                "installed": True,
                "version": version,
                "import_path": str(getattr(module, "__file__", "")),
                "status": "ok" if not expected or version == expected else "version_mismatch",
            })
        except Exception as exc:                                           
            row["error"] = f"{type(exc).__name__}: {exc}"
        packages.append(row)

    lightning_row: dict[str, object] = {
        "package": "lightning",
        "import_name": "lightning",
        "expected_version": "",
        "installed": False,
        "version": "",
        "import_path": "",
        "status": "missing",
        "error": "",
    }
    try:
        module = importlib.import_module("lightning")
        lightning_row.update({
            "installed": True,
            "version": _version("lightning", module),
            "import_path": str(getattr(module, "__file__", "")),
            "status": "ok",
        })
    except Exception as exc:                                                           
        lightning_row["error"] = f"{type(exc).__name__}: {exc}"
    packages.append(lightning_row)

    cuda: dict[str, object] = {
        "available": False,
        "device_count": 0,
        "torch_cuda_version": "",
        "error": "",
    }
    try:
        torch = importlib.import_module("torch")
        cuda["torch_cuda_version"] = str(getattr(torch.version, "cuda", ""))
        try:
            cuda["available"] = bool(torch.cuda.is_available())
            cuda["device_count"] = int(torch.cuda.device_count())
        except Exception as exc:                                       
            cuda["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:                    
        cuda["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "packages": packages,
        "seed": int(seed),
        "selected_device": selected_device,
        "cuda": cuda,
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
        },
    }


def write_neural_dependency_report(
    out_path: str | Path,
    seed: int = 1,
    selected_device: str = "cpu",
) -> dict[str, object]:
    report = inspect_neuralforecast_dependencies(seed=seed, selected_device=selected_device)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def require_neuralforecast_dependencies(
    out_path: str | Path,
    seed: int = 1,
    selected_device: str = "cpu",
) -> dict[str, object]:
    report = write_neural_dependency_report(out_path, seed=seed, selected_device=selected_device)
    required = {"neuralforecast", "torch", "pytorch_lightning", "numpy", "pandas"}
    bad = [
        row
        for row in report["packages"]
        if row["package"] in required and row["status"] != "ok"
    ]
    if bad:
        raise NeuralDependencyError(f"neuralforecast dependency check failed: {json.dumps(bad, sort_keys=True)}")
    return report
