#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASTER_ROOT="$(cd "${CODE_ROOT}/../.." && pwd)"
cd "${CODE_ROOT}"

PYTHON="${PYTHON:-${CODE_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

TIMESFM_PYTHON="${TIMESFM_PYTHON:-${CODE_ROOT}/.venv_timesfm311/bin/python}"
if [[ ! -x "${TIMESFM_PYTHON}" ]]; then
  TIMESFM_PYTHON="${PYTHON}"
fi

RUN_TAG="${RUN_TAG:-real_full_v3_direct_rollout_baselines_all}"
LOG_DIR="${LOG_DIR:-${CODE_ROOT}/logs/${RUN_TAG}}"
RUN_ROOT="${RUN_ROOT:-runs_v3_full/baselines}"
MANIFEST="${MANIFEST:-data/full_manifest_v3.csv}"
RESULTS_DIR="${RESULTS_DIR:-results/real_full_v3_baselines}"
BASELINE_METRICS="${BASELINE_METRICS:-${RESULTS_DIR}/baseline_metrics.csv}"
BASELINE_METRIC_SLICES="${BASELINE_METRIC_SLICES:-${RESULTS_DIR}/baseline_metric_slices.csv}"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${RESULTS_DIR}/baseline_run_manifest.csv}"
summary_PACKET="${summary_PACKET:-reports/real_full_v3_baseline_summary_packet.md}"
LEDGER_VALIDATION_REPORT="${LEDGER_VALIDATION_REPORT:-reports/full_v3_event_ledger_validation.md}"
SPLIT_REPORT="${SPLIT_REPORT:-reports/full_v3_split_report.md}"
CONTRACT_REPORTS_DIR="${CONTRACT_REPORTS_DIR:-reports}"
FOUNDATION_ACCEPTANCE_REPORT="${FOUNDATION_ACCEPTANCE_REPORT:-reports/foundation_acceptance_real_full_v3.md}"
V3_CONTRACT_REPORT="${V3_CONTRACT_REPORT:-reports/full_v3_data_contract.md}"
BENCHMARK_PROTOCOL="${BENCHMARK_PROTOCOL:-benchmark_a_v3_benchmark_b_v26_1_pooled}"
case "${BENCHMARK_PROTOCOL}" in
  benchmark_a_v3_benchmark_b_v26_1_pooled|v3_direct_rollout) ;;
  *)
    echo "unsupported BENCHMARK_PROTOCOL=${BENCHMARK_PROTOCOL}; expected benchmark_a_v3_benchmark_b_v26_1_pooled or v3_direct_rollout" >&2
    exit 2
    ;;
esac
RUN_FLAT_AGENT_PREFLIGHT="${RUN_FLAT_AGENT_PREFLIGHT:-0}"
BASELINE_BRIDGE_CONFIG_ROOT="${BASELINE_BRIDGE_CONFIG_ROOT:-}"
if [[ -z "${BASELINE_BRIDGE_CONFIG_ROOT}" ]]; then
  BASELINE_BRIDGE_CONFIG_ROOT="${RESULTS_DIR}/provisional_bridge_configs"
fi
REGISTRY="${REGISTRY:-../caster/configs/model_registry.yaml}"

DATA_A="${DATA_A:-${CASTER_ROOT}/data/benchmark_a/curated_full_v3_direct_rollout7}"
DATA_B="${DATA_B:-${CASTER_ROOT}/data/benchmark_b/curated_v26_1_release_lag1/benchmark_b_pooled}"
NEURAL_MAX_STEPS="${NEURAL_MAX_STEPS:-3}"
SEED="${SEED:-1}"
USE_GPU="${USE_GPU:-auto}"
GPU_IDS="${GPU_IDS:-}"
CHRONOS_CHECKPOINT_PATH="${CHRONOS_CHECKPOINT_PATH:-}"
CHRONOS_CHECKPOINT_ID="${CHRONOS_CHECKPOINT_ID:-amazon/chronos-bolt-small}"
TIMESFM_CHECKPOINT_PATH="${TIMESFM_CHECKPOINT_PATH:-}"
TIMESFM_CHECKPOINT_ID="${TIMESFM_CHECKPOINT_ID:-google/timesfm-2.0-500m-pytorch}"
AGENTIC_TOP_ONE_BUDGET_SECONDS="${AGENTIC_TOP_ONE_BUDGET_SECONDS:-7200}"
AGENT_REACT_BUDGET_SECONDS="${AGENT_REACT_BUDGET_SECONDS:-7200}"
AGENTIC_FULL_RECOVERY_BUDGET_SECONDS="${AGENTIC_FULL_RECOVERY_BUDGET_SECONDS:-7200}"
AGENTIC_TOP_ONE_MAX_NEW_TOKENS="${AGENTIC_TOP_ONE_MAX_NEW_TOKENS:-128}"
AGENT_REACT_MAX_NEW_TOKENS="${AGENT_REACT_MAX_NEW_TOKENS:-192}"
AGENTIC_FULL_RECOVERY_MAX_NEW_TOKENS="${AGENTIC_FULL_RECOVERY_MAX_NEW_TOKENS:-256}"
AGENT_REACT_SELECTION_POLICY="${AGENT_REACT_SELECTION_POLICY:-llm_only}"
AGENTIC_FULL_RECOVERY_SELECTION_POLICY="${AGENTIC_FULL_RECOVERY_SELECTION_POLICY:-llm_only}"
AGENTIC_FULL_RECOVERY_METHOD_NAME="${AGENTIC_FULL_RECOVERY_METHOD_NAME:-agentic_full_recovery}"
ENABLE_NATIVE_SIDECARS="${ENABLE_NATIVE_SIDECARS:-0}"
NATIVE_SIDECAR_ROOT="${NATIVE_SIDECAR_ROOT:-${RUN_ROOT}/native_diagnostics}"
REUSE_COMPLETE_BASELINE_MODELS="${REUSE_COMPLETE_BASELINE_MODELS:-0}"
PROPHET_B_YEARLY_SEASONALITY_MODE="${PROPHET_B_YEARLY_SEASONALITY_MODE:-auto}"
case "${PROPHET_B_YEARLY_SEASONALITY_MODE}" in
  auto|off) ;;
  *)
    echo "unsupported PROPHET_B_YEARLY_SEASONALITY_MODE=${PROPHET_B_YEARLY_SEASONALITY_MODE}; expected auto or off" >&2
    exit 2
    ;;
esac

NAIVE_MODELS=(lastvalue seasonalnaive)
STATS_MODELS=(autoarima autoets autotheta autoces)
NEURAL_MODELS=(nbeats nhits deepar patchtst tft)
FOUNDATION_MODELS=(chronos timesfm)

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_real_full_baselines_all.sh [--clean]

Environment overrides:
  PYTHON, TIMESFM_PYTHON, DATA_A, DATA_B, MANIFEST, RUN_ROOT, LOG_DIR,
  RESULTS_DIR, BASELINE_METRICS, BASELINE_METRIC_SLICES, BASELINE_MANIFEST,
  BASELINE_BRIDGE_CONFIG_ROOT, summary_PACKET, REGISTRY,
  LEDGER_VALIDATION_REPORT, SPLIT_REPORT, CONTRACT_REPORTS_DIR,
  FOUNDATION_ACCEPTANCE_REPORT,
  USE_GPU, GPU_IDS, NEURAL_MAX_STEPS, SEED,
  CHRONOS_CHECKPOINT_PATH, CHRONOS_CHECKPOINT_ID,
  TIMESFM_CHECKPOINT_PATH, TIMESFM_CHECKPOINT_ID,
  AGENTIC_TOP_ONE_BUDGET_SECONDS, AGENT_REACT_BUDGET_SECONDS, AGENTIC_FULL_RECOVERY_BUDGET_SECONDS,
  AGENTIC_TOP_ONE_MAX_NEW_TOKENS, AGENT_REACT_MAX_NEW_TOKENS, AGENTIC_FULL_RECOVERY_MAX_NEW_TOKENS,
  AGENT_REACT_SELECTION_POLICY, AGENTIC_FULL_RECOVERY_SELECTION_POLICY,
  AGENTIC_FULL_RECOVERY_METHOD_NAME
  RUN_FLAT_AGENT_PREFLIGHT, REUSE_COMPLETE_BASELINE_MODELS,
  PROPHET_B_YEARLY_SEASONALITY_MODE

Notes:
  - Reuses existing curated full panels and event_ledger.csv files.
  - Launches all baseline model jobs in parallel after manifest/data checks.
  - Uses GPU by default when CUDA is visible; set USE_GPU=0 to force CPU.
  - GPU jobs are assigned round-robin over GPU_IDS. If GPU_IDS is unset,
    nvidia-smi -L is used to discover GPU indices.
USAGE
}

CLEAN=0
for arg in "$@"; do
  case "${arg}" in
    --clean) CLEAN=1 ;;
    --skip-tests) ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: ${arg}" >&2; usage; exit 2 ;;
  esac
done

require_checkpoint_dir() {
  local label="$1"
  local path="$2"
  if [[ -n "${path}" && ! -d "${path}" ]]; then
    echo "${label} checkpoint directory does not exist: ${path}" >&2
    exit 2
  fi
}

require_checkpoint_dir "Chronos" "${CHRONOS_CHECKPOINT_PATH}"
if [[ -n "${TIMESFM_CHECKPOINT_PATH}" && ! -f "${TIMESFM_CHECKPOINT_PATH}" ]]; then
  echo "TimesFM checkpoint file does not exist: ${TIMESFM_CHECKPOINT_PATH}" >&2
  exit 2
fi

NATIVE_SIDECAR_ARGS=()
if [[ "${ENABLE_NATIVE_SIDECARS}" == "1" || "${ENABLE_NATIVE_SIDECARS}" == "true" || "${ENABLE_NATIVE_SIDECARS}" == "yes" ]]; then
  NATIVE_SIDECAR_ARGS=(--enable-native-sidecars --native-sidecar-root "${NATIVE_SIDECAR_ROOT}")
fi
PROPHET_DATASET_EXECUTION_ARGS=()
STATSFORECAST_DATASET_EXECUTION_ARGS=()
if [[ ${#NATIVE_SIDECAR_ARGS[@]} -eq 0 ]]; then
  PROPHET_DATASET_EXECUTION_ARGS=(--parallel-datasets)
  STATSFORECAST_DATASET_EXECUTION_ARGS=(--parallel-datasets)
fi

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "required file missing: ${path}" >&2
    exit 1
  fi
}

run_logged() {
  local name="$1"
  shift
  local log="${LOG_DIR}/${name}.log"
  mkdir -p "$(dirname "${log}")"
  set +e
  {
    echo "cmd: $*"
    echo "started_at=$(date -Is)"
    "$@"
    status=$?
    echo "finished_at=$(date -Is)"
    echo "exit_status=${status}"
    exit "${status}"
  } 2>&1 | tee "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  return "${status}"
}

gpu_available() {
  [[ "${USE_GPU}" != "0" && "${USE_GPU}" != "false" ]] || return 1
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi -L >/dev/null 2>&1 || return 1
}

discover_gpus() {
  if [[ -n "${GPU_IDS}" ]]; then
    echo "${GPU_IDS}" | tr ',' ' '
    return 0
  fi
  if gpu_available; then
    nvidia-smi -L | sed -n 's/^GPU \([0-9][0-9]*\):.*/\1/p' | tr '\n' ' '
  fi
}

mapfile -t GPU_LIST < <(discover_gpus | tr ' ' '\n' | sed '/^$/d')
if gpu_available && [[ "${#GPU_LIST[@]}" -gt 0 ]]; then
  DEVICE="cuda"
else
  DEVICE="cpu"
fi
GPU_COUNTER_FILE=$(mktemp)
echo "0" > "${GPU_COUNTER_FILE}"

next_gpu() {
  if [[ "${DEVICE}" != "cuda" || "${#GPU_LIST[@]}" -eq 0 ]]; then
    echo ""
    return 0
  fi
  local cursor
  cursor=$(cat "${GPU_COUNTER_FILE}")
  local gpu="${GPU_LIST[$((cursor % ${#GPU_LIST[@]}))]}"
  echo $((cursor + 1)) > "${GPU_COUNTER_FILE}"
  echo "${gpu}"
}

JOB_PIDS=()
JOB_NAMES=()
JOB_LOGS=()

run_parallel_job() {
  local name="$1"
  local device_policy="$2"
  shift 2
  local -a command=("$@")
  local out_dir=""
  local i
  for i in "${!command[@]}"; do
    if [[ "${command[$i]}" == "--out" && $((i + 1)) -lt ${#command[@]} ]]; then
      out_dir="${command[$((i + 1))]}"
      break
    fi
  done
  if [[ "${REUSE_COMPLETE_BASELINE_MODELS}" == "1" || "${REUSE_COMPLETE_BASELINE_MODELS}" == "true" || "${REUSE_COMPLETE_BASELINE_MODELS}" == "yes" ]]; then
    if [[ -n "${out_dir}" ]] && "${PYTHON}" scripts/check_baseline_artifacts.py "${out_dir}" --formal-reuse >/dev/null 2>&1; then
      echo "reused complete model_job=${name} out=${out_dir} validation=formal-reuse"
      return 0
    fi
  fi
  local log="${LOG_DIR}/models/${name}.log"
  local gpu=""
  local effective_backend_device="cpu"
  if [[ "${device_policy}" != "cpu" ]]; then
    gpu="$(next_gpu)"
  fi
  if [[ "${device_policy}" == "gpu" && -n "${gpu}" ]]; then
    effective_backend_device="cuda"
  elif [[ "${device_policy}" == "mixed" && -n "${gpu}" ]]; then
    effective_backend_device="mixed(controller=cuda,forecast_backend=runtime_dependent)"
  elif [[ "${device_policy}" == "mixed" ]]; then
    effective_backend_device="mixed(controller=cpu,forecast_backend=runtime_dependent)"
  fi
  mkdir -p "$(dirname "${log}")"
  (
    set -o pipefail
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    if [[ -n "${gpu}" ]]; then
      export CUDA_VISIBLE_DEVICES="${gpu}"
    fi
    {
      echo "cmd: $*"
      echo "started_at=$(date -Is)"
      echo "requested_device=${DEVICE}"
      echo "device_policy=${device_policy}"
      echo "effective_backend_device=${effective_backend_device}"
      echo "device=$([[ "${effective_backend_device}" == cuda ]] && echo cuda || echo cpu)"
      echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
      "$@"
      status=$?
      echo "finished_at=$(date -Is)"
      echo "exit_status=${status}"
      exit "${status}"
    } 2>&1 | tee "${log}"
  ) &
  local pid=$!
  JOB_PIDS+=("${pid}")
  JOB_NAMES+=("${name}")
  JOB_LOGS+=("${log}")
  echo "started model_job=${name} pid=${pid} log=${log}"
}

wait_parallel_jobs() {
  local failures=0
  local i pid name log status
  for i in "${!JOB_PIDS[@]}"; do
    pid="${JOB_PIDS[$i]}"
    name="${JOB_NAMES[$i]}"
    log="${JOB_LOGS[$i]}"
    if wait "${pid}"; then
      echo "finished model_job=${name} status=0 log=${log}"
    else
      status=$?
      echo "failed model_job=${name} status=${status} log=${log}" >&2
      failures=$((failures + 1))
    fi
  done
  if [[ "${failures}" -ne 0 ]]; then
    echo "parallel baseline jobs failed: ${failures}" >&2
    return 1
  fi
}

run_foundation_chronos() {
  local -a cmd=("${PYTHON}" scripts/run_foundation.py
    --model chronos
    --manifest "${MANIFEST}"
    --out "${RUN_ROOT}/foundation/chronos"
    --checkpoint-id "${CHRONOS_CHECKPOINT_ID}"
    --device "${DEVICE}")
  if [[ -n "${CHRONOS_CHECKPOINT_PATH}" ]]; then
    cmd+=(--checkpoint-path "${CHRONOS_CHECKPOINT_PATH}")
  fi
  run_parallel_job foundation_chronos gpu "${cmd[@]}"
}

run_foundation_timesfm() {
  local -a cmd=("${TIMESFM_PYTHON}" scripts/run_foundation.py
    --model timesfm
    --manifest "${MANIFEST}"
    --out "${RUN_ROOT}/foundation/timesfm"
    --checkpoint-id "${TIMESFM_CHECKPOINT_ID}"
    --device "${DEVICE}")
  if [[ -n "${TIMESFM_CHECKPOINT_PATH}" ]]; then
    cmd+=(--checkpoint-path "${TIMESFM_CHECKPOINT_PATH}")
  fi
  run_parallel_job foundation_timesfm gpu "${cmd[@]}"
}

if [[ "${CLEAN}" == "1" ]]; then
  echo "clean=true removing generated baseline outputs"
  rm -rf "${RUN_ROOT}" "${RESULTS_DIR}" "${summary_PACKET}" "${LOG_DIR}"
  mkdir -p "${RUN_ROOT}" "${RESULTS_DIR}" "$(dirname "${summary_PACKET}")" "${LOG_DIR}" \
    "$(dirname "${MANIFEST}")" "$(dirname "${LEDGER_VALIDATION_REPORT}")" \
    "$(dirname "${SPLIT_REPORT}")" "${CONTRACT_REPORTS_DIR}" \
    "$(dirname "${FOUNDATION_ACCEPTANCE_REPORT}")"
fi

mkdir -p "${LOG_DIR}" "${RUN_ROOT}" "${RESULTS_DIR}" reports data \
  "$(dirname "${MANIFEST}")" "$(dirname "${LEDGER_VALIDATION_REPORT}")" \
  "$(dirname "${SPLIT_REPORT}")" "${CONTRACT_REPORTS_DIR}" \
  "$(dirname "${FOUNDATION_ACCEPTANCE_REPORT}")"

require_file "${DATA_A}/daily_panel.csv"
require_file "${DATA_A}/event_ledger.csv"
require_file "${DATA_B}/weekly_panel.csv"
require_file "${DATA_B}/event_ledger.csv"
require_file "${REGISTRY}"

run_logged 00_env "${PYTHON}" - <<PY
from pathlib import Path
import platform
import subprocess
import sys

print("python=" + sys.version.replace("\\n", " "))
print("executable=" + sys.executable)
print("platform=" + platform.platform())
print("code_root=${CODE_ROOT}")
print("caster_root=${CASTER_ROOT}")
print("device=${DEVICE}")
print("gpu_ids=${GPU_LIST[*]-}")
try:
    import pandas as pd
    import numpy as np
    print("pandas=" + pd.__version__)
    print("numpy=" + np.__version__)
except Exception as exc:
    print("env_import_error=" + repr(exc))
    raise
try:
    import torch
    print("torch=" + torch.__version__)
    print("torch_cuda_available=" + str(torch.cuda.is_available()))
    print("torch_cuda_device_count=" + str(torch.cuda.device_count()))
except Exception as exc:
    print("torch_import_error=" + repr(exc))
PY

run_logged 05_write_full_manifest "${PYTHON}" scripts/write_manifest_for_data_packages.py \
  --package-dirs "${DATA_A},${DATA_B}" \
  --out "${MANIFEST}"

run_logged 06_check_event_ledger "${PYTHON}" scripts/check_event_ledger.py \
  --manifest "${MANIFEST}" \
  --out "${LEDGER_VALIDATION_REPORT}"

run_logged 07_report_splits "${PYTHON}" scripts/report_splits.py \
  --manifest "${MANIFEST}" \
  --out "${SPLIT_REPORT}"

run_logged 08_check_full_contract "${PYTHON}" scripts/check_full_v3_data_contract.py \
  --manifest "${MANIFEST}" \
  --out "${V3_CONTRACT_REPORT}"

echo "launching all baseline models in parallel device=${DEVICE} gpu_ids=${GPU_LIST[*]-}"

for model in "${NAIVE_MODELS[@]}"; do
  out_name="${model}"
  if [[ "${model}" == "lastvalue" ]]; then
    out_name="lastvalue"
  fi
  run_parallel_job "naive_${model}" cpu "${PYTHON}" scripts/run_baseline.py \
    --model "${model}" \
    --manifest "${MANIFEST}" \
    --out "${RUN_ROOT}/naive/${out_name}" \
    "${NATIVE_SIDECAR_ARGS[@]}"
done

for model in "${STATS_MODELS[@]}"; do
  statsforecast_dataset_args=()
  if [[ "${model}" == "autoarima" ]]; then
    statsforecast_dataset_args=("${STATSFORECAST_DATASET_EXECUTION_ARGS[@]}")
  fi
  run_parallel_job "statsforecast_${model}" cpu "${PYTHON}" scripts/run_statsforecast.py \
    --model "${model}" \
    --manifest "${MANIFEST}" \
    --out "${RUN_ROOT}/statsforecast/${model}" \
    "${statsforecast_dataset_args[@]}" \
    "${NATIVE_SIDECAR_ARGS[@]}"
done

run_parallel_job prophet cpu "${PYTHON}" scripts/run_prophet.py \
  --manifest "${MANIFEST}" \
  --out "${RUN_ROOT}/prophet" \
  --dataset-key all \
  --benchmark-b-yearly-seasonality-mode "${PROPHET_B_YEARLY_SEASONALITY_MODE}" \
  "${PROPHET_DATASET_EXECUTION_ARGS[@]}" \
  "${NATIVE_SIDECAR_ARGS[@]}"

for model in "${NEURAL_MODELS[@]}"; do
  run_parallel_job "neural_${model}" gpu "${PYTHON}" scripts/run_neuralforecast.py \
    --model "${model}" \
    --manifest "${MANIFEST}" \
    --out "${RUN_ROOT}/neural/${model}" \
    --max-steps "${NEURAL_MAX_STEPS}" \
    --seed "${SEED}" \
    --device "${DEVICE}"
done

run_foundation_chronos
run_foundation_timesfm

if [[ "${RUN_FLAT_AGENT_PREFLIGHT}" == "1" || "${RUN_FLAT_AGENT_PREFLIGHT}" == "true" ]]; then
  run_parallel_job agentic_top_one mixed "${PYTHON}" scripts/run_agentic_top_one.py \
    --manifest "${MANIFEST}" \
    --registry "${REGISTRY}" \
    --out "${RUN_ROOT}/agentic/top_one" \
    --runtime-budget-seconds "${AGENTIC_TOP_ONE_BUDGET_SECONDS}" \
    --max-new-tokens "${AGENTIC_TOP_ONE_MAX_NEW_TOKENS}"

  run_parallel_job agentic_react mixed "${PYTHON}" scripts/run_react_agent.py \
    --manifest "${MANIFEST}" \
    --registry "${REGISTRY}" \
    --out "${RUN_ROOT}/agentic/react" \
    --runtime-budget-seconds "${AGENT_REACT_BUDGET_SECONDS}" \
    --max-new-tokens "${AGENT_REACT_MAX_NEW_TOKENS}" \
    --selection-policy "${AGENT_REACT_SELECTION_POLICY}" \
    --dataset-key all

  run_parallel_job agentic_full_recovery mixed "${PYTHON}" scripts/run_agentic_full_recovery.py \
    --manifest "${MANIFEST}" \
    --registry "${REGISTRY}" \
    --out "${RUN_ROOT}/agentic/full_recovery" \
    --runtime-budget-seconds "${AGENTIC_FULL_RECOVERY_BUDGET_SECONDS}" \
    --max-new-tokens "${AGENTIC_FULL_RECOVERY_MAX_NEW_TOKENS}" \
    --selection-policy "${AGENTIC_FULL_RECOVERY_SELECTION_POLICY}" \
    --method-name "${AGENTIC_FULL_RECOVERY_METHOD_NAME}" \
    --dataset-key all
else
  echo "skip flat agent preflight; formal agents run after immutable forecast archive construction"
fi

wait_parallel_jobs

RUN_DIRS=( \
  "${RUN_ROOT}/naive/lastvalue" \
  "${RUN_ROOT}/naive/seasonalnaive" \
  "${RUN_ROOT}/statsforecast/autoarima" \
  "${RUN_ROOT}/statsforecast/autoets" \
  "${RUN_ROOT}/statsforecast/autotheta" \
  "${RUN_ROOT}/statsforecast/autoces" \
  "${RUN_ROOT}/prophet" \
  "${RUN_ROOT}/neural/nbeats" \
  "${RUN_ROOT}/neural/nhits" \
  "${RUN_ROOT}/neural/deepar" \
  "${RUN_ROOT}/neural/patchtst" \
  "${RUN_ROOT}/neural/tft" \
  "${RUN_ROOT}/foundation/chronos" \
  "${RUN_ROOT}/foundation/timesfm"
)
if [[ "${RUN_FLAT_AGENT_PREFLIGHT}" == "1" || "${RUN_FLAT_AGENT_PREFLIGHT}" == "true" ]]; then
  RUN_DIRS+=(
    "${RUN_ROOT}/agentic/top_one"
    "${RUN_ROOT}/agentic/react"
    "${RUN_ROOT}/agentic/full_recovery"
  )
fi
for run_dir in "${RUN_DIRS[@]}"; do
  run_logged "80_check_$(echo "${run_dir#${RUN_ROOT}/}" | tr '/' '_')" "${PYTHON}" scripts/check_baseline_artifacts.py "${run_dir}"
done

run_logged 81_check_foundation_acceptance "${PYTHON}" scripts/check_foundation_acceptance.py \
  --manifest "${MANIFEST}" \
  --runs-root "${RUN_ROOT}/foundation" \
  --models chronos,timesfm \
  --out "${FOUNDATION_ACCEPTANCE_REPORT}"

run_logged 90_aggregate_baseline_results "${PYTHON}" scripts/aggregate_baseline_results.py \
  --run-root "${RUN_ROOT}" \
  --out-metrics "${BASELINE_METRICS}" \
  --out-metric-slices "${BASELINE_METRIC_SLICES}" \
  --out-manifest "${BASELINE_MANIFEST}" \
  --report "${summary_PACKET}" \
  --bridge-config-root "${BASELINE_BRIDGE_CONFIG_ROOT}" \
  --allow-missing-agents

run_logged 91_artifact_verify "${PYTHON}" - <<PY
from pathlib import Path
import json
import pandas as pd

run_root = Path(${RUN_ROOT@Q})
metrics_path = Path(${BASELINE_METRICS@Q})
manifest_path = Path(${BASELINE_MANIFEST@Q})
summary_path = Path(${summary_PACKET@Q})
device = ${DEVICE@Q}

expected = {
    "last_value",
    "seasonal_naive",
    "autoarima",
    "autoets",
    "autotheta",
    "autoces",
    "prophet",
    "nbeats",
    "nhits",
    "deepar",
    "patchtst",
    "tft",
    "chronos_bolt_small",
    "timesfm_2_0",
}
run_flat_agent_preflight = ${RUN_FLAT_AGENT_PREFLIGHT@Q}.strip().lower() in {"1", "true", "yes"}
if run_flat_agent_preflight:
    expected.update({"agentic_top_one", "agent_react", "agentic_full_recovery"})
for path in [metrics_path, manifest_path, summary_path]:
    assert path.exists(), path
metrics = pd.read_csv(metrics_path)
manifest = pd.read_csv(manifest_path)
assert len(metrics) > 0, metrics_path
assert len(manifest) > 0, manifest_path
present = set(metrics["method"].astype(str))
missing = sorted(expected - present)
assert not missing, missing
numeric_manifest = manifest[manifest["status"].astype(str) == "numeric"]
assert len(numeric_manifest) >= len(expected), len(numeric_manifest)
dataset_col = "dataset_key" if "dataset_key" in metrics.columns else "dataset"
assert dataset_col in metrics.columns, "dataset_key_or_dataset"
assert set(metrics[dataset_col].astype(str)) >= {"benchmark_a", "benchmark_b"}
for col in ["n", "mae", "rmse", "coverage_90", "width_90"]:
    assert col in metrics.columns, col
    values = pd.to_numeric(metrics[col], errors="coerce")
    assert values.notna().all(), col
assert "nll" in metrics.columns, "nll"
nll_values = pd.to_numeric(metrics["nll"], errors="coerce")
if "nll_status" in metrics.columns:
    nll_ok_mask = metrics["nll_status"].astype(str) == "ok"
    if nll_ok_mask.any():
        assert nll_values[nll_ok_mask].notna().all(), "nll"
else:
    assert nll_values.notna().all(), "nll"
crps_col = "crps" if "crps" in metrics.columns else "crps_gaussian"
assert crps_col in metrics.columns, "crps_or_crps_gaussian"
values = pd.to_numeric(metrics[crps_col], errors="coerce")
assert values.notna().all(), crps_col
if device == "cuda":
    for rel in [
        "neural/nbeats",
        "neural/nhits",
        "neural/deepar",
        "neural/patchtst",
        "neural/tft",
        "foundation/chronos",
        "foundation/timesfm",
    ]:
        meta = json.loads((run_root / rel / "run_manifest.json").read_text())
        assert str(meta.get("device")) == "cuda", (rel, meta.get("device"))
print("verify=ok")
print(f"run_root={run_root}")
print(f"baseline_metrics={metrics_path}")
print(f"baseline_manifest={manifest_path}")
print(f"summary_packet={summary_path}")
PY

echo "ok real_full_baselines_all"
echo "logs=${LOG_DIR}"
echo "run_root=${RUN_ROOT}"
echo "baseline_metrics=${BASELINE_METRICS}"
echo "baseline_manifest=${BASELINE_MANIFEST}"
echo "summary_packet=${summary_PACKET}"
