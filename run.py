#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
STAGES = (
    "data",
    "candidates",
    "ranking",
    "selection",
    "agents",
    "method",
    "ablation",
    "runtime",
    "summaries",
    "visualization",
)
TASKS = ("benchmark_a", "benchmark_b_covid", "benchmark_b_flu")
QWEN_CHECKPOINT_SHA256 = (
    "b4a6bd43ac91ee4c1526814cfd006f40f08c65b30474b85895cc63a639a247d0"
)
CHRONOS_MODEL_SHA256 = (
    "06a6a19bbe74bc10a9cd193bd4bf2bf638ae07f7e0d51653ae7ab8ea968a21dd"
)
TIMESFM_MODEL_SHA256 = (
    "bb9d7022d8027325f4e656b7d5cf36dd919269fd79ad714eab41f5ba49cc1cfe"
)
TIMESFM_CHECKPOINT_SHA256 = (
    "a3a1362cdc26f0dde45c9790c438951ff02af96b2d1d6445fdb58f28c77e18de"
)


class RunError(RuntimeError):
    pass


@dataclass(frozen=True)
class Assets:
    qwen: Path | None = None
    chronos: Path | None = None
    timesfm: Path | None = None
    model_cache: Path | None = None
    gpus: str = "0,1,2,3"


def optional_path(value: object, base: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def gpu_csv(value: object) -> str:
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    selected = [str(item).strip() for item in items if str(item).strip()]
    if not selected or any(not item.isdigit() for item in selected):
        raise RunError("GPU identifiers must be non-negative integers")
    return ",".join(selected)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qwen_checkpoint_sha256(root: Path) -> str:
    suffixes = {".bin", ".json", ".model", ".safetensors", ".txt"}
    assets = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in suffixes
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not assets:
        raise RunError(f"Qwen checkpoint contains no model assets: {root}")
    records = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in assets
    ]
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RunError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RunError(
            f"{label} differs from the required checkpoint: "
            f"expected={expected} observed={observed}"
        )


def load_assets(path: Path | None) -> Assets:
    if path is None:
        return Assets()
    location = path.expanduser().resolve()
    if not location.is_file():
        raise RunError(f"asset configuration does not exist: {location}")
    try:
        import yaml
    except ImportError as error:
        raise RunError("PyYAML is required") from error
    value = yaml.safe_load(location.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "caster_external_assets_v1":
        raise RunError("asset configuration has an invalid schema")
    allowed = {
        "schema",
        "qwen_checkpoint",
        "chronos_checkpoint",
        "timesfm_checkpoint",
        "huggingface_cache",
        "cuda_devices",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RunError("unknown asset fields: " + ", ".join(unknown))
    base = location.parent
    return Assets(
        qwen=optional_path(value.get("qwen_checkpoint"), base),
        chronos=optional_path(value.get("chronos_checkpoint"), base),
        timesfm=optional_path(value.get("timesfm_checkpoint"), base),
        model_cache=optional_path(value.get("huggingface_cache"), base),
        gpus=gpu_csv(value.get("cuda_devices", [0, 1, 2, 3])),
    )


def environment(assets: Assets) -> dict[str, str]:
    values: dict[str, str] = {}
    if assets.qwen is not None:
        values["CASTER_QWEN_CHECKPOINT"] = str(assets.qwen)
    if assets.chronos is not None:
        values["CHRONOS_CHECKPOINT_PATH"] = str(assets.chronos)
    if assets.timesfm is not None:
        values["TIMESFM_CHECKPOINT_PATH"] = str(assets.timesfm / "torch_model.ckpt")
    if assets.model_cache is not None:
        values["HF_HOME"] = str(assets.model_cache)
    if assets.qwen or assets.chronos or assets.timesfm:
        values["HF_HUB_OFFLINE"] = "1"
        values["TRANSFORMERS_OFFLINE"] = "1"
    return values


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise RunError(f"required directory is missing: {source}")
    shutil.copytree(source, destination, dirs_exist_ok=True)


def stage_project(work: Path, resume: bool) -> Path:
    project = work / "project"
    if project.exists():
        if not resume:
            raise RunError(f"work directory exists: {work}")
        return project
    project.mkdir(parents=True)
    for name in (
        "code",
        "configs",
        "scripts",
        "data_pipeline",
        "summaries",
        "visualization",
    ):
        copy_tree(ROOT / name, project / name)
    copy_tree(ROOT / "data/benchmark_a/raw", project / "data/benchmark_a/raw_all")
    source_b = ROOT / "data/benchmark_b"
    raw_b = project / "data/benchmark_b/raw_all"
    copy_tree(source_b / "source", raw_b / "data_raw")
    copy_tree(source_b / "configs", raw_b / "configs")
    copy_tree(ROOT / "data_pipeline/benchmark_b/reference", raw_b / "reference")
    gold = raw_b / "gold"
    gold.mkdir(parents=True)
    for name in (
        "aux_variants_summary.parquet",
        "aux_preliminary_vs_finalized_paired.parquet",
    ):
        shutil.copy2(source_b / "frozen_inputs" / name, gold / name)
    return project


def run_command(
    command: list[str],
    cwd: Path,
    execute: bool,
    extra_environment: dict[str, str],
) -> None:
    print("+", " ".join(command), flush=True)
    if not execute:
        return
    values = dict(os.environ)
    values.setdefault("PYTHONNOUSERSITE", "1")
    values.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    values.update(extra_environment)
    subprocess.run(command, cwd=cwd, env=values, check=True)


def paths(project: Path, name: str) -> dict[str, Path]:
    root = project / "experiments" / name
    return {
        "root": root,
        "shared": root / "shared",
        "inputs": root / "inputs",
        "reference": root / "reference",
        "run": root / "method",
        "results": root / "results",
        "images": root / "images",
    }


def selection_hash(shared: Path) -> str:
    pointer = (
        shared
        / "caster_selections/qwen25_multiscale_released_sequence_v1/active_selection.json"
    )
    if not pointer.is_file():
        return "active"
    value = json.loads(pointer.read_text(encoding="utf-8"))
    return Path(str(value["selection_root"])).name


def data_commands() -> list[list[str]]:
    prefix = "data/benchmark_b/raw_all"
    builder = "data_pipeline/benchmark_b/scripts"
    return [
        [
            sys.executable,
            "code/baseline/scripts/build_benchmark_a_epillm_graph_nodes_v3.py",
            "--config",
            "configs/benchmark_a_epillm_graph_nodes_v3.yaml",
        ],
        [
            sys.executable,
            f"{builder}/build_bronze.py",
            "--raw-root",
            f"{prefix}/data_raw",
            "--out",
            f"{prefix}/data_intermediate/bronze",
            "--contracts",
            f"{prefix}/configs/source_contracts_v17.yaml",
        ],
        [
            sys.executable,
            f"{builder}/build_silver.py",
            "--benchmark-spec",
            f"{prefix}/configs/benchmark_spec.yaml",
            "--bronze-root",
            f"{prefix}/data_intermediate/bronze",
            "--out",
            f"{prefix}/data_intermediate/silver",
            "--jurisdiction-map",
            f"{prefix}/reference/jurisdiction_map.csv",
        ],
        [
            sys.executable,
            f"{builder}/build_gold_targets.py",
            "--silver-nhsn",
            f"{prefix}/data_intermediate/silver/nhsn_state_weekly.parquet",
            "--out",
            f"{prefix}/gold",
        ],
        [
            sys.executable,
            f"{builder}/build_panel.py",
            "--targets",
            f"{prefix}/gold/panel_targets_wide.parquet",
            "--nssp",
            f"{prefix}/data_intermediate/silver/nssp_national_weekly.parquet",
            "--nwss-flua",
            f"{prefix}/data_intermediate/silver/nwss_flua_state_weekly.parquet",
            "--nwss-rsv",
            f"{prefix}/data_intermediate/silver/nwss_rsv_state_weekly.parquet",
            "--out",
            f"{prefix}/gold",
            "--benchmark-spec",
            f"{prefix}/configs/benchmark_spec.yaml",
        ],
        [
            sys.executable,
            "scripts/build_v26_1_benchmark_b.py",
            "--config",
            "configs/caster_data_protocol_benchmark_b_v26_1.yaml",
        ],
    ]


def commands(stage: str, location: dict[str, Path], assets: Assets, resume: bool = False) -> list[list[str]]:
    shared = location["shared"]
    root = location["root"]
    run = location["run"]
    tasks = ",".join(TASKS)
    if stage == "data":
        return data_commands()
    if stage == "candidates":
        return [[
            sys.executable,
            "scripts/_candidate_pipeline.py",
            "--runs-root",
            str(root),
            "--shared-root",
            str(shared),
            "--top-k",
            "27",
            "--n-draws",
            "10",
            "--base-seed",
            "42",
            "--prophet-b-yearly-seasonality-mode",
            "off",
            "--cache-scope",
            "all-result-tasks",
            "--resume-baseline" if resume else "--rerun-baseline",
            "--strict",
            "--skip-agents",
            "--gpu-ids",
            assets.gpus,
        ]]
    if stage == "ranking":
        cache = shared / "caster_candidates/k27__draws10__seed42"
        return [[
            sys.executable,
            "scripts/build_shared_formal_selection.py",
            "--candidate-cache",
            str(cache),
            "--out-root",
            str(shared / "caster_selections"),
            "--panel-a",
            "data/benchmark_a/curated_full_v3_direct_rollout7/daily_panel.csv",
            "--ledger-a",
            "data/benchmark_a/curated_full_v3_direct_rollout7/event_ledger.csv",
            "--panel-b",
            "data/benchmark_b/curated_v26_1_release_lag1/benchmark_b_pooled/weekly_panel.csv",
            "--ledger-b",
            "data/benchmark_b/curated_v26_1_release_lag1/benchmark_b_pooled/event_ledger.csv",
            "--selection-profile",
            "formal_27_country_macro_v1",
            "--top-k",
            "10",
        ]]
    if stage == "selection":
        checkpoint = str(assets.qwen) if assets.qwen else "${QWEN_CHECKPOINT}"
        return [[
            sys.executable,
            "scripts/build_qwen_formal_top10.py",
            "--shared",
            str(shared),
            "--checkpoint",
            checkpoint,
            "--device",
            "cuda:0",
            "--top-k",
            "10",
        ]]
    if stage == "agents":
        cache = shared / "caster_candidates/k27__draws10__seed42"
        return [[
            sys.executable,
            "scripts/run_formal_agents_all_tasks.py",
            "--shared",
            str(shared),
            "--candidate-cache-root",
            str(cache),
            "--tasks",
            tasks,
            "--methods",
            "agentic_top_one,react,agentic_full_recovery",
            "--gpus",
            assets.gpus,
            "--run-profile",
            "qwen25_multiscale_released_sequence_v1",
            "--selection-context-profile",
            "qwen25_multiscale_released_sequence_v1",
            "--no-update-shared-manifest",
        ]]
    if stage == "method":
        selected = selection_hash(shared)
        return [
            [
                sys.executable,
                "scripts/build_predeclared_shift_events.py",
                "--run-root",
                str(location["reference"]),
                "--datasets",
                "benchmark_a,benchmark_b",
                "--out-root",
                str(location["reference"] / "events"),
                "--manifest",
                str(location["reference"] / "events/event_selection_manifest.json"),
                "--gold-root",
                "data/benchmark_b/raw_all/gold",
                "--benchmark-a-panel",
                "data/benchmark_a/curated_full_v3_direct_rollout7/daily_panel.csv",
            ],
            [
                sys.executable,
                "scripts/run_rho_only_smallval_pilot.py",
                "--source-run",
                str(location["reference"]),
                "--out-run",
                str(run),
                "--tasks",
                tasks,
                "--task-workers",
                "3",
                "--export-workers",
                "3",
                "--seed",
                "42",
                "--top-k",
                "10",
                "--benchmark-a-top-k",
                "10",
                "--benchmark-b-top-k",
                "10",
                "--fixed-gamma",
                "1",
                "--fixed-c-u",
                "1.25",
                "--censoring-bound-scope",
                "eligible27_train",
                "--distribution",
                "student_t",
                "--predictive-contract",
                "coherent_mean_preserving_censored_student_t",
                "--fixed-nu",
                "5",
                "--benchmark-a-rho-min",
                "0.4",
                "--benchmark-a-rho-max",
                "0.6",
                "--benchmark-b-rho-min",
                "0.005",
                "--benchmark-b-rho-max",
                "0.5",
                "--weight-nll",
                "0.20",
                "--weight-wis",
                "0.20",
                "--weight-short-rmse",
                "0.20",
                "--weight-long-rmse",
                "0.20",
                "--weight-mae",
                "0.10",
                "--weight-coverage-penalty",
                "0.10",
                "--shared-root",
                str(shared),
                "--shared-input-root",
                str(location["inputs"]),
                "--selection-profile",
                "qwen25_multiscale_released_sequence_v1",
                "--selection-hash",
                selected,
                "--agent-run-profile",
                "qwen25_multiscale_released_sequence_v1",
                "--include-draw-kernel",
                "--resume",
            ],
        ]
    if stage == "ablation":
        return [
            [
                sys.executable,
                "scripts/run_incremental_ablation_all_tasks.py",
                "--spec",
                str(run / "ablation_spec.yaml"),
                "--out-root",
                str(run / "ablation/one_layer_moment"),
                "--repo-root",
                str(location["root"].parents[1]),
                "--python",
                sys.executable,
                "--seed",
                "42",
                "--tasks",
                *TASKS,
                "--skip-existing",
            ],
            [
                sys.executable,
                "scripts/run_additional_ablation_variants.py",
                "--run-root",
                str(run),
                "--out-root",
                str(run / "ablation/additional"),
                "--python",
                sys.executable,
                "--seed",
                "42",
                "--tasks",
                *TASKS,
                "--skip-existing",
            ],
        ]
    if stage == "runtime":
        selected = selection_hash(shared)
        selection_root = (
            shared
            / "caster_selections/qwen25_multiscale_released_sequence_v1"
            / selected
        )
        return [[
            sys.executable,
            "scripts/run_runtime_experiment.py",
            "--run-root",
            str(run),
            "--shared-root",
            str(shared),
            "--selection-root",
            str(selection_root),
            "--output",
            str(run / "runtime/cells.csv"),
        ]]
    if stage == "summaries":
        return [[
            sys.executable,
            "summaries/generate_all.py",
            "--run-root",
            str(run),
            "--shared-root",
            str(shared),
            "--events-root",
            str(location["reference"] / "events"),
            "--output-dir",
            str(location["results"]),
        ]]
    if stage == "visualization":
        return [[
            sys.executable,
            "visualization/generate_all.py",
            "--run-root",
            str(run),
            "--shared-root",
            str(shared),
            "--events-root",
            str(location["reference"] / "events"),
            "--summary-dir",
            str(location["results"]),
            "--output-dir",
            str(location["images"]),
        ]]
    raise RunError(f"unknown stage: {stage}")


def prepare_ablation(project: Path, run: Path) -> None:
    template = (project / "configs/incremental_ablation_spec.yaml").read_text(encoding="utf-8")
    run.mkdir(parents=True, exist_ok=True)
    (run / "ablation_spec.yaml").write_text(
        template.replace("__RUN_ROOT__", str(run)), encoding="utf-8"
    )


def validate(stages: tuple[str, ...], assets: Assets, execute: bool) -> None:
    required = [
        ROOT / "code",
        ROOT / "configs",
        ROOT / "scripts",
        ROOT / "data_pipeline/benchmark_b/scripts",
        ROOT / "data/benchmark_a/raw",
        ROOT / "data/benchmark_b/source",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RunError("package is incomplete: " + ", ".join(missing))
    if not execute:
        return
    if any(stage in stages for stage in ("selection", "agents", "runtime")):
        if assets.qwen is None or not assets.qwen.is_dir():
            raise RunError("a local Qwen checkpoint is required")
        observed_qwen = qwen_checkpoint_sha256(assets.qwen)
        if observed_qwen != QWEN_CHECKPOINT_SHA256:
            raise RunError(
                "Qwen checkpoint differs from the required checkpoint: "
                f"expected={QWEN_CHECKPOINT_SHA256} observed={observed_qwen}"
            )
    if "candidates" in stages:
        missing_models = [
            name
            for name, path in (("Chronos", assets.chronos), ("TimesFM", assets.timesfm))
            if path is None or not path.is_dir()
        ]
        if missing_models:
            raise RunError("missing local checkpoints: " + ", ".join(missing_models))
        assert assets.chronos is not None
        assert assets.timesfm is not None
        require_sha256(
            assets.chronos / "model.safetensors",
            CHRONOS_MODEL_SHA256,
            "Chronos model.safetensors",
        )
        require_sha256(
            assets.timesfm / "model.safetensors",
            TIMESFM_MODEL_SHA256,
            "TimesFM model.safetensors",
        )
        require_sha256(
            assets.timesfm / "torch_model.ckpt",
            TIMESFM_CHECKPOINT_SHA256,
            "TimesFM torch_model.ckpt",
        )


def selected_stages(start: str | None, end: str | None) -> tuple[str, ...]:
    values = list(STAGES)
    if start is not None:
        values = values[values.index(start):]
    if end is not None:
        values = values[: values.index(end) + 1]
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("--work-dir", required=True, type=Path)
        command.add_argument("--run-name", default="seed42_top10")
        command.add_argument("--from-stage", choices=STAGES)
        command.add_argument("--to-stage", choices=STAGES)
        command.add_argument("--external-assets", type=Path)
        command.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "list":
        print("stages=" + ",".join(STAGES))
        print("tasks=" + ",".join(TASKS))
        return 0
    stages = selected_stages(args.from_stage, args.to_stage)
    assets = load_assets(args.external_assets)
    execute = args.command == "run"
    validate(stages, assets, execute)
    work = args.work_dir.expanduser().resolve()
    project = stage_project(work, args.resume) if execute else work / "project"
    location = paths(project, args.run_name)
    values = environment(assets)
    for stage in stages:
        print(f"stage={stage}", flush=True)
        if execute and stage == "ablation":
            prepare_ablation(project, location["run"])
        for command in commands(stage, location, assets, args.resume):
            run_command(command, project, execute, values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
