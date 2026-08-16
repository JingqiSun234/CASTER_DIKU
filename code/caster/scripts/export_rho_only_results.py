#!/usr/bin/env python3
""





from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import result_export_core as core
from caster.bridge.likelihood import (
    alternate_ARCHIVE_MOMENT,
    PREDICTIVE_CONTRACTS,
    calibrate_component_sigma,
)


TASKS = (
    "benchmark_a",
    "benchmark_b_covid",
    "benchmark_b_flu",
)

JOINT_V2_WEIGHTS = {
    "nll": 0.15,
    "short_rmse": 0.25,
    "long_rmse": 0.25,
    "mae": 0.10,
    "wis": 0.15,
    "coverage_penalty": 0.10,
}
JOINT_V2_GAMMA_BOUNDS = (0.125, 4.0)
JOINT_FULLVAL_V2_GAMMA_BOUNDS = (0.005, 4.0)
JOINT_V2_OPTIMIZER_SCHEMA = "joint_rho_gamma_newton_v2"
JOINT_V2_TOTAL_EVALUATIONS = 128
JOINT_V2_EXPLORATION_EVALUATIONS = 96
JOINT_V2_FINAL_POLISH_EVALUATIONS = 32
JOINT_VARIANTS = ("one_layer", "hierarchical")
JOINT_FULLVAL_V2_RHO_PROFILE = "fullval_a_high2"
JOINT_FULLVAL_V2_MANIFEST = "full_validation_manifest.csv"
JOINT_FULLVAL_V2_SIGMA_POLICY = (
    "full_validation_formula_recomputed_and_frozen"
)
JOINT_FULLVAL_V2_FIXED_SIGMA = "calculated_full_validation_formula"
JOINT_FULLVAL_V2_SIGMA_FORMULA = (
    "sqrt_mean_squared_log1p_observation_minus_log1p_pred_mean_by_component_horizon"
)
JOINT_FULLVAL_V2_MIN_SIGMA = 0.04
JOINT_FULLVAL_V2_DEFAULT_SIGMA = 0.20
JOINT_FULLVAL_V2_1_RHO_PROFILE = "fullval_a_high2_v2_1"
JOINT_FULLVAL_V2_1_OPTIMIZER_SCHEMA = "joint_rho_gamma_newton_v2_1"
JOINT_FULLVAL_V2_1_TOTAL_EVALUATIONS = 184
JOINT_FULLVAL_V2_1_EXPLORATION_EVALUATIONS = 96
JOINT_FULLVAL_V2_1_FINAL_STAGE_EVALUATIONS = 88
JOINT_FULLVAL_V2_1_BEST_FIRST_POLISH_EVALUATIONS = 16
JOINT_FULLVAL_V2_1_CERTIFICATE_EVALUATIONS = 72
JOINT_FULLVAL_V2_1_CERTIFICATE_MAX_ATTEMPTS = 9
JOINT_FULLVAL_V2_1_CERTIFICATE_MAX_NEIGHBORS = 8
JOINT_FULLVAL_V2_1_AMENDMENT_SCHEMA = (
    "caster_joint_rho_gamma_fullval_certificate_budget_amendment_v2_1"
)
JOINT_FULLVAL_V2_1_AMENDMENT_SHA256 = (
    "51764418ef40c39fd08b8566025fc189e43cbea2d7fe84ea7aaaa205e1d1ad04"
)
JOINT_FULLVAL_V2_1_alternate_RUN_CONFIG_SHA256 = (
    "1bc7bdc9b06634ca38f47b8a09393a677ee6d4d0be52d2ef2caf78e1529d9a35"
)
JOINT_FULLVAL_V2_1_alternate_LOG_SHA256 = {
    "benchmark_a": "bed537c03e477e4d27ed96e0f7214ff67ca168a92b0423e2c49b15b94fcf7a2f",
    "benchmark_b_covid": "388c0f04587d64ec78933d6dea3bea48c385d44b4a79d759cc2050745e0fbf04",
    "benchmark_b_flu": "4404abf59ecdf7559c65df6fc07dbbb9aa941b6f060c4ed2c0662834ee0b9317",
    "benchmark_b_pooled": "e7f10a9a0fea15b75c4a4197d65fdb7d8eecb0350d378ffe77ca60fe549908ac",
}
JOINT_FULLVAL_V2_2_RHO_PROFILE = "fullval_a_high2_v2_2_recovery"
JOINT_FULLVAL_V2_2_OPTIMIZER_SCHEMA = "joint_rho_gamma_newton_recovery_v2_2"
JOINT_FULLVAL_V2_2_TOTAL_EVALUATIONS = 184
JOINT_FULLVAL_V2_2_EXPLORATION_EVALUATIONS = 96
JOINT_FULLVAL_V2_2_FINAL_STAGE_EVALUATIONS = 88
JOINT_FULLVAL_V2_2_BEST_FIRST_POLISH_EVALUATIONS = 16
JOINT_FULLVAL_V2_2_CERTIFICATE_EVALUATIONS = 24
JOINT_FULLVAL_V2_2_CERTIFICATE_MAX_ATTEMPTS = 3
JOINT_FULLVAL_V2_2_CERTIFICATE_MAX_NEIGHBORS = 8
JOINT_FULLVAL_V2_2_RECOVERY_EVALUATIONS = 48
JOINT_FULLVAL_V2_2_RECOVERY_MAX_SWEEPS = 2
JOINT_FULLVAL_V2_2_RECOVERY_MAX_PASSES = 3
JOINT_FULLVAL_V2_2_RECOVERY_SWEEP_EVALUATIONS = 24
JOINT_FULLVAL_V2_2_AMENDMENT_SCHEMA = (
    "caster_joint_rho_gamma_fullval_certificate_recovery_amendment_v2_2"
)
JOINT_FULLVAL_V2_2_AMENDMENT_SHA256 = (
    "99afe959ce4ef857b8654d84e91bae6cdce8ddf321ad92eb0492092bdbed4e16"
)
JOINT_FULLVAL_V2_2_V2_1_RUN_CONFIG_SHA256 = (
    "8b83c12ab689b760523cb39728c7c6fa753aabbf60642a5a9252a367a8ff4aa7"
)
JOINT_FULLVAL_V2_2_V2_1_DRIVER_SHA256 = (
    "bf0b210c5f0c73829d592af83f6fd8c9fbf932a8cf6851058f8e2aff83d4dd77"
)
JOINT_FULLVAL_V2_2_V2_1_LOG_SHA256 = {
    "benchmark_a": "743fae0a9c248cc61580688687f6c821602ac05f6deea28ecf97280428164ee9",
    "benchmark_b_covid": "56370ff6ee6b79c3467e5910f4ff233a52fabc202afc109afac74b9c4fa73fa1",
    "benchmark_b_flu": "89ec0db9f9ad387833a5ceda844d9762653168f921ba923c8c422622abd36d82",
    "benchmark_b_pooled": "51d3f0dbea4dc4eef8f64a372bad0f690cdc4ab746a0e69f33e71d2105803170",
}
JOINT_FULLVAL_V2_SHAPES = {
    "benchmark_a": (14_658, 14, 1_047),
    "benchmark_b_covid": (2_907, 19, 153),
    "benchmark_b_flu": (2_907, 19, 153),
    "benchmark_b_pooled": (5_814, 19, 306),
}
V21_POSTHOC_PROFILE = "archived_v21_fullval_gamma0125_ref025_draw_tau_gamma"
V21_SELECTION_PROFILE = "formal_27_country_macro_v1"
V21_SELECTION_PROFILE_STATUS = "archived_nonrelease_profile"
V21_INPUT_MANIFEST_SCHEMA = "caster_rho_only_shared_selection_input_manifest_v1"
V21_POSTHOC_MOMENT_FREEZE_SCHEMA = (
    "caster_shared_gamma_rho_joint_fullval_v21_archived_freeze_v1"
)
V21_POSTHOC_MOMENT_PROTOCOL = (
    "fixed_formula_shared_gamma_rho_joint_fullval_v21_archived_recovery_v1"
)
V21_POSTHOC_REFERENCE_SCHEMA = (
    "caster_shared_gamma_rho_joint_fullval_v21_archived_reference_v1"
)
V21_POSTHOC_OPTIMIZER_SCHEMA = "joint_rho_gamma_newton_recovery_v21_archived_v1"
V21_POSTHOC_DRAW_FREEZE_SCHEMA = (
    "caster_draw_kernel_tau_from_parent_gamma_v21_archived_freeze_v1"
)
V21_POSTHOC_DRAW_PROTOCOL = (
    "derived_draw_kernel_tau_from_parent_selected_shared_gamma_v21_archived_v1"
)
V21_POSTHOC_DIMENSIONAL_BINDING = (
    "exploratory_tau_numeric_equals_parent_selected_moment_gamma"
)
V21_POSTHOC_DRAW_TAU_SOURCE = (
    "parent_moment_selected_shared_gamma_archived_v21"
)
V21_POSTHOC_GAMMA_BOUNDS = (0.125, 4.0)
V21_POSTHOC_REFERENCE_POINT = {"rho": 0.5, "gamma": 0.25}
COHERENT_CENSORED_FULLVAL_WEIGHTS = {
    "nll": 0.20,
    "short_rmse": 0.20,
    "long_rmse": 0.20,
    "mae": 0.10,
    "wis": 0.20,
    "coverage_penalty": 0.10,
}
COHERENT_CENSORED_FULLVAL_SHAPES = {
    "benchmark_a": (14_658, 14, 0),
    "benchmark_b_covid": (3_519, 23, 612),
    "benchmark_b_flu": (3_519, 23, 612),
    "benchmark_b_pooled": (7_038, 23, 1_224),
}
COHERENT_CENSORED_CONTRACT = "coherent_censored_student_t"
MEAN_PRESERVING_CENSORED_CONTRACT = (
    "coherent_mean_preserving_censored_student_t"
)
COHERENT_CENSORED_CONTRACTS = {
    COHERENT_CENSORED_CONTRACT,
    MEAN_PRESERVING_CENSORED_CONTRACT,
}
COHERENT_CENSORED_NLL_MEASURE_BASIS = (
    "log1p_transform_mixed_measure_atoms_at_zero_and_upper_bound"
)
COHERENT_CENSORED_FREEZE_SCHEMA = (
    "caster_coherent_censored_fullval_freeze_manifest_v1"
)
COHERENT_CENSORED_PROTOCOL = (
    "coherent_censored_shared_coordinates_pattern_v1"
)
MEAN_PRESERVING_CENSORED_PROTOCOL = (
    "coherent_mean_preserving_censored_shared_coordinates_pattern_v1"
)
MEAN_PRESERVING_CENSORED_SMALLVAL360_JOINT_PILOT = (
    "coherent_mean_preserving_censored_smallval360_joint_rho_gamma_scale"
)
MEAN_PRESERVING_CENSORED_SMALLVAL360_JOINT_FREEZE_SCHEMA = (
    "caster_coherent_mean_preserving_censored_smallval360_joint_rho_gamma_"
    "scale_freeze_manifest_v1"
)
MEAN_PRESERVING_CENSORED_SMALLVAL360_JOINT_PROTOCOL = (
    "coherent_mean_preserving_censored_smallval360_joint_rho_gamma_scale_"
    "pattern_v1"
)
MEAN_PRESERVING_CENSORED_SMALLVAL360_RMSE_HEAVY_PILOT = (
    "coherent_mean_preserving_censored_smallval360_joint_rho_gamma_scale_"
    "rmse_heavy_task_bounds"
)
MEAN_PRESERVING_CENSORED_SMALLVAL360_RMSE_HEAVY_FREEZE_SCHEMA = (
    "caster_coherent_mean_preserving_censored_smallval360_joint_rho_gamma_"
    "scale_rmse_heavy_task_bounds_freeze_manifest_v1"
)
MEAN_PRESERVING_CENSORED_SMALLVAL360_RMSE_HEAVY_PROTOCOL = (
    "coherent_mean_preserving_censored_smallval360_joint_rho_gamma_scale_"
    "rmse_heavy_task_bounds_pattern_v1"
)
COHERENT_CENSORED_SMALLVAL360_JOINT_SHAPE = (360, 10, 36)
COHERENT_CENSORED_SMALLVAL360_FIXED_C_U = 1.25
COHERENT_CENSORED_SMALLVAL360_GAMMA_BOUNDS = (0.25, 4.0)
COHERENT_CENSORED_SMALLVAL360_SCALE_BOUNDS = (0.5, 2.5)
COHERENT_CENSORED_SMALLVAL360_RMSE_HEAVY_WEIGHTS = {
    "nll": 0.10,
    "short_rmse": 0.30,
    "long_rmse": 0.30,
    "mae": 0.10,
    "wis": 0.10,
    "coverage_penalty": 0.10,
}
COHERENT_CENSORED_SMALLVAL360_RMSE_HEAVY_BOUNDS = {
    "benchmark_a": {
        "rho": (0.20, 0.60),
        "gamma": (0.50, 1.50),
        "scale_multiplier": (0.50, 1.25),
    },
    "benchmark_b_covid": {
        "rho": (0.25, 1.00),
        "gamma": (0.125, 0.75),
        "scale_multiplier": (0.25, 1.00),
    },
    "benchmark_b_flu": {
        "rho": (0.005, 0.50),
        "gamma": (0.25, 4.00),
        "scale_multiplier": (0.50, 2.50),
    },
    "benchmark_b_pooled": {
        "rho": (0.005, 0.50),
        "gamma": (0.25, 4.00),
        "scale_multiplier": (0.50, 2.50),
    },
}
MEAN_PRESERVING_CENSORED_SMALLVAL360_JOINT_PILOTS = frozenset(
    {
        MEAN_PRESERVING_CENSORED_SMALLVAL360_JOINT_PILOT,
        MEAN_PRESERVING_CENSORED_SMALLVAL360_RMSE_HEAVY_PILOT,
    }
)

PILOT_CONTRACTS = {
    "rho_only": {
        "freeze_schema": "caster_rho_only_smallval_freeze_manifest_v1",
        "parameter_selection_protocol": "fixed_formula_rho_only_smallval_newton_v1",
        "selection_description": (
            "fixed-formula rho selected by small-validation safeguarded Newton"
        ),
    },
    "gamma_only": {
        "freeze_schema": "caster_shared_gamma_only_smallval_freeze_manifest_v1",
        "parameter_selection_protocol": (
            "fixed_formula_shared_gamma_only_smallval_newton_v1"
        ),
        "selection_description": (
            "shared gamma selected by small-validation safeguarded Newton with "
            "rho, sigma, and nu frozen from the source rho-only run"
        ),
    },
    "joint_rho_gamma": {
        "freeze_schema": "caster_shared_gamma_rho_joint_smallval_freeze_manifest_v2",
        "parameter_selection_protocol": (
            "fixed_formula_shared_gamma_rho_joint_smallval_newton_v2"
        ),
        "selection_description": (
            "global rho and shared gamma jointly selected by validation-only "
            "alternating safeguarded Newton with bounded trust refinement"
        ),
    },
    "joint_rho_gamma_fullval": {
        "freeze_schema": "caster_shared_gamma_rho_joint_fullval_freeze_manifest_v2",
        "parameter_selection_protocol": (
            "fixed_formula_shared_gamma_rho_joint_fullval_newton_v2"
        ),
        "selection_description": (
            "global rho and shared gamma jointly selected by full-validation-only "
            "alternating safeguarded Newton with bounded trust refinement"
        ),
    },
    "joint_rho_gamma_fullval_v2_1": {
        "freeze_schema": (
            "caster_shared_gamma_rho_joint_fullval_freeze_manifest_v2_1"
        ),
        "parameter_selection_protocol": (
            "fixed_formula_shared_gamma_rho_joint_fullval_newton_v2_1"
        ),
        "selection_description": (
            "global rho and shared gamma jointly selected by amended "
            "full-validation-only safeguarded Newton with a worst-case "
            "9-by-8 local-certificate reserve"
        ),
    },
    "joint_rho_gamma_fullval_v2_2": {
        "freeze_schema": (
            "caster_shared_gamma_rho_joint_fullval_freeze_manifest_v2_2"
        ),
        "parameter_selection_protocol": (
            "fixed_formula_shared_gamma_rho_joint_fullval_newton_recovery_v2_2"
        ),
        "selection_description": (
            "global rho and shared gamma jointly selected by full-validation-only "
            "safeguarded Newton with two bounded three-pass recoveries and three "
            "complete local certificates"
        ),
    },
    "joint_rho_gamma_v21_archived_dual": {
        "freeze_schema": V21_POSTHOC_MOMENT_FREEZE_SCHEMA,
        "parameter_selection_protocol": V21_POSTHOC_MOMENT_PROTOCOL,
        "draw_freeze_schema": V21_POSTHOC_DRAW_FREEZE_SCHEMA,
        "draw_parameter_selection_protocol": V21_POSTHOC_DRAW_PROTOCOL,
        "selection_description": (
            "archived v21 full-validation joint rho/shared-gamma sensitivity; "
            "draw-kernel tau is a dimensionally exploratory numeric binding to "
            "the corresponding parent moment-t selected gamma"
        ),
    },
    "coherent_censored_fullval": {
        "freeze_schema": COHERENT_CENSORED_FREEZE_SCHEMA,
        "parameter_selection_protocol": COHERENT_CENSORED_PROTOCOL,
        "selection_description": (
            "rho, one shared scale multiplier, optional shared gamma, and a "
            "train-calibrated censoring upper-bound multiplier selected on "
            "complete causal validation folds"
        ),
        "predictive_contract": COHERENT_CENSORED_CONTRACT,
    },
    "coherent_mean_preserving_censored_fullval": {
        "freeze_schema": COHERENT_CENSORED_FREEZE_SCHEMA,
        "parameter_selection_protocol": MEAN_PRESERVING_CENSORED_PROTOCOL,
        "selection_description": (
            "rho, one shared scale multiplier, optional shared gamma, and a "
            "train-calibrated censoring upper-bound multiplier selected on "
            "complete causal validation folds with componentwise "
            "raw-mean-preserving latent locations"
        ),
        "predictive_contract": MEAN_PRESERVING_CENSORED_CONTRACT,
    },
    MEAN_PRESERVING_CENSORED_SMALLVAL360_JOINT_PILOT: {
        "freeze_schema": (
            MEAN_PRESERVING_CENSORED_SMALLVAL360_JOINT_FREEZE_SCHEMA
        ),
        "parameter_selection_protocol": (
            MEAN_PRESERVING_CENSORED_SMALLVAL360_JOINT_PROTOCOL
        ),
        "selection_description": (
            "global rho, shared gamma for moment-t, and one shared scale "
            "multiplier jointly selected on the frozen 360-endpoint, "
            "10-fold validation subset with c_U fixed at 1.25"
        ),
        "predictive_contract": MEAN_PRESERVING_CENSORED_CONTRACT,
    },
    MEAN_PRESERVING_CENSORED_SMALLVAL360_RMSE_HEAVY_PILOT: {
        "freeze_schema": (
            MEAN_PRESERVING_CENSORED_SMALLVAL360_RMSE_HEAVY_FREEZE_SCHEMA
        ),
        "parameter_selection_protocol": (
            MEAN_PRESERVING_CENSORED_SMALLVAL360_RMSE_HEAVY_PROTOCOL
        ),
        "selection_description": (
            "global rho, shared gamma for moment-t, and one shared scale "
            "multiplier jointly selected on the frozen 360-endpoint, "
            "10-fold validation subset with RMSE-heavy objective weights, "
            "task-specific bounds, and c_U fixed at 1.25"
        ),
        "predictive_contract": MEAN_PRESERVING_CENSORED_CONTRACT,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_map_matches(payload: object, expected: dict[str, float]) -> bool:
    if not isinstance(payload, dict) or set(payload) != set(expected):
        return False
    try:
        return all(
            math.isclose(
                float(payload[key]), float(value), rel_tol=0.0, abs_tol=1e-12
            )
            for key, value in expected.items()
        )
    except (TypeError, ValueError):
        return False


def _float_pair(payload: object) -> tuple[float, float] | None:
    if not isinstance(payload, list) or len(payload) != 2:
        return None
    try:
        pair = (float(payload[0]), float(payload[1]))
    except (TypeError, ValueError):
        return None
    return pair if all(math.isfinite(value) for value in pair) else None


def _isclose_number(payload: object, expected: float) -> bool:
    try:
        value = float(payload)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(
        value, float(expected), rel_tol=0.0, abs_tol=1e-12
    )


def _joint_v2_expected_rho_bounds(dataset: str, profile: object) -> tuple[float, float]:
    if profile == "a_high":
        return (0.5, 1.0) if dataset == "benchmark_a" else (0.001, 0.5)
    if profile == "a_low":
        return (0.001, 0.5)
    raise SystemExit(f"joint v2 freeze has unknown rho profile {profile!r}")


def _check_joint_v2_optimizer(
    payload: object,
    *,
    label: str,
    optimizer_schema: str = JOINT_V2_OPTIMIZER_SCHEMA,
    total_evaluations: int = JOINT_V2_TOTAL_EVALUATIONS,
    exploration_evaluations: int = JOINT_V2_EXPLORATION_EVALUATIONS,
    final_stage_evaluations: int = JOINT_V2_FINAL_POLISH_EVALUATIONS,
    strict_v2_1: bool = False,
    strict_v2_2: bool = False,
) -> None:
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} has no joint v2 optimizer payload")
    required_exact = {
        "schema": optimizer_schema,
        "evaluation_budget": total_evaluations,
        "exploration_evaluation_limit": exploration_evaluations,
        "final_polish_evaluation_limit": final_stage_evaluations,
        "convergence_status": "verified_local_discrete",
        "axis_bracketed": {"rho": True, "gamma": True},
        "mixed_checked": True,
        "last_accepted_step_capped": False,
    }
    for key, expected in required_exact.items():
        if payload.get(key) != expected:
            raise SystemExit(f"{label} has invalid joint v2 optimizer field {key}")
    try:
        evaluation_count = int(payload["evaluation_count"])
        exploration_count = int(payload["exploration_evaluation_count"])
        polish_count = int(payload["final_polish_evaluation_count"])
        accepted_steps = int(payload["final_polish_accepted_steps"])
        certificate_attempts = int(payload["certificate_attempts"])
        improvement_count = int(payload["verification_improvement_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"{label} omits joint v2 convergence counts") from error
    if not 0 < evaluation_count <= total_evaluations:
        raise SystemExit(f"{label} has an invalid joint v2 evaluation count")
    if not 0 < exploration_count <= exploration_evaluations:
        raise SystemExit(f"{label} has an invalid exploration evaluation count")
    if not 0 <= polish_count <= final_stage_evaluations:
        raise SystemExit(f"{label} has an invalid final-polish evaluation count")
    if exploration_count + polish_count != evaluation_count:
        raise SystemExit(f"{label} has inconsistent optimizer phase counts")
    if accepted_steps < 0 or certificate_attempts < 1 or improvement_count < 0:
        raise SystemExit(f"{label} has invalid final-polish convergence counts")
    if strict_v2_1 and strict_v2_2:
        raise SystemExit(f"{label} cannot be both v2.1 and v2.2")
    if strict_v2_1:
        required_v2_1 = {
            "final_stage_evaluation_limit": (
                JOINT_FULLVAL_V2_1_FINAL_STAGE_EVALUATIONS
            ),
            "best_first_polish_evaluation_limit": (
                JOINT_FULLVAL_V2_1_BEST_FIRST_POLISH_EVALUATIONS
            ),
            "certificate_evaluation_reserve": (
                JOINT_FULLVAL_V2_1_CERTIFICATE_EVALUATIONS
            ),
            "certificate_max_attempts": (
                JOINT_FULLVAL_V2_1_CERTIFICATE_MAX_ATTEMPTS
            ),
            "certificate_max_neighbors_per_attempt": (
                JOINT_FULLVAL_V2_1_CERTIFICATE_MAX_NEIGHBORS
            ),
            "certificate_worst_case_evaluations": (
                JOINT_FULLVAL_V2_1_CERTIFICATE_EVALUATIONS
            ),
        }
        for key, expected in required_v2_1.items():
            if payload.get(key) != expected:
                raise SystemExit(f"{label} has invalid v2.1 optimizer field {key}")
        try:
            final_stage_count = int(payload["final_stage_evaluation_count"])
            best_count = int(payload["best_first_polish_evaluation_count"])
            certificate_count = int(payload["certificate_evaluation_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"{label} omits v2.1 phase counts") from error
        if (
            final_stage_count != polish_count
            or best_count + certificate_count != final_stage_count
            or not 0
            <= best_count
            <= JOINT_FULLVAL_V2_1_BEST_FIRST_POLISH_EVALUATIONS
            or not 0
            <= certificate_count
            <= JOINT_FULLVAL_V2_1_CERTIFICATE_EVALUATIONS
            or not 1
            <= certificate_attempts
            <= JOINT_FULLVAL_V2_1_CERTIFICATE_MAX_ATTEMPTS
            or improvement_count != certificate_attempts - 1
        ):
            raise SystemExit(f"{label} has inconsistent v2.1 phase counts")
    if strict_v2_2:
        required_v2_2 = {
            "final_stage_evaluation_limit": (
                JOINT_FULLVAL_V2_2_FINAL_STAGE_EVALUATIONS
            ),
            "best_first_polish_evaluation_limit": (
                JOINT_FULLVAL_V2_2_BEST_FIRST_POLISH_EVALUATIONS
            ),
            "certificate_attempt_evaluation_limit": (
                JOINT_FULLVAL_V2_2_CERTIFICATE_MAX_NEIGHBORS
            ),
            "certificate_evaluation_reserve": (
                JOINT_FULLVAL_V2_2_CERTIFICATE_EVALUATIONS
            ),
            "certificate_max_attempts": (
                JOINT_FULLVAL_V2_2_CERTIFICATE_MAX_ATTEMPTS
            ),
            "certificate_worst_case_evaluations": (
                JOINT_FULLVAL_V2_2_CERTIFICATE_EVALUATIONS
            ),
            "recovery_sweep_evaluation_limit": (
                JOINT_FULLVAL_V2_2_RECOVERY_SWEEP_EVALUATIONS
            ),
            "recovery_sweep_limit": JOINT_FULLVAL_V2_2_RECOVERY_MAX_SWEEPS,
            "recovery_evaluation_reserve": (
                JOINT_FULLVAL_V2_2_RECOVERY_EVALUATIONS
            ),
            "recovery_max_passes_per_sweep": (
                JOINT_FULLVAL_V2_2_RECOVERY_MAX_PASSES
            ),
        }
        for key, expected in required_v2_2.items():
            if payload.get(key) != expected:
                raise SystemExit(f"{label} has invalid v2.2 optimizer field {key}")
        try:
            final_stage_count = int(payload["final_stage_evaluation_count"])
            best_count = int(payload["best_first_polish_evaluation_count"])
            certificate_count = int(payload["certificate_evaluation_count"])
            recovery_count = int(payload["recovery_evaluation_count"])
            certified_index = int(payload["certified_certificate_index"])
            recovery_sweeps = int(payload["recovery_sweep_count"])
            recovery_pass_count = int(payload["recovery_pass_count"])
            recovery_coordinate_attempts = int(
                payload["recovery_coordinate_attempts"]
            )
            recovery_accepted_steps = int(payload["recovery_accepted_steps"])
            certificate_counts = {
                str(key): int(value)
                for key, value in dict(
                    payload["certificate_evaluation_counts"]
                ).items()
            }
            recovery_counts = {
                str(key): int(value)
                for key, value in dict(payload["recovery_evaluation_counts"]).items()
            }
            recovery_pass_counts = {
                str(key): int(value)
                for key, value in dict(payload["recovery_pass_counts"]).items()
            }
            phase_counts = {
                str(key): int(value)
                for key, value in dict(payload["phase_evaluation_counts"]).items()
            }
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"{label} omits v2.2 recovery counts") from error
        expected_recovery_keys = {
            str(index) for index in range(1, recovery_sweeps + 1)
        }
        if (
            final_stage_count != polish_count
            or best_count + certificate_count + recovery_count
            != final_stage_count
            or not 0
            <= best_count
            <= JOINT_FULLVAL_V2_2_BEST_FIRST_POLISH_EVALUATIONS
            or not 0
            <= certificate_count
            <= JOINT_FULLVAL_V2_2_CERTIFICATE_EVALUATIONS
            or not 0
            <= recovery_count
            <= JOINT_FULLVAL_V2_2_RECOVERY_EVALUATIONS
            or not 1
            <= certificate_attempts
            <= JOINT_FULLVAL_V2_2_CERTIFICATE_MAX_ATTEMPTS
            or improvement_count != certificate_attempts - 1
            or recovery_sweeps != improvement_count
            or certified_index != certificate_attempts
            or set(certificate_counts)
            != {str(index) for index in range(1, certificate_attempts + 1)}
            or any(
                not 0 <= value <= JOINT_FULLVAL_V2_2_CERTIFICATE_MAX_NEIGHBORS
                for value in certificate_counts.values()
            )
            or sum(certificate_counts.values()) != certificate_count
            or set(recovery_counts) != expected_recovery_keys
            or set(recovery_pass_counts) != expected_recovery_keys
            or any(
                not 0 <= value <= JOINT_FULLVAL_V2_2_RECOVERY_SWEEP_EVALUATIONS
                for value in recovery_counts.values()
            )
            or any(
                not 1 <= value <= JOINT_FULLVAL_V2_2_RECOVERY_MAX_PASSES
                for value in recovery_pass_counts.values()
            )
            or sum(recovery_counts.values()) != recovery_count
            or sum(recovery_pass_counts.values()) != recovery_pass_count
            or recovery_coordinate_attempts != 2 * recovery_pass_count
            or not 0 <= recovery_accepted_steps <= recovery_coordinate_attempts
            or phase_counts
            != {
                "global_exploration": exploration_count,
                "best_first_polish": best_count,
                "local_discrete_certificate": certificate_count,
                "certificate_recovery": recovery_count,
            }
        ):
            raise SystemExit(f"{label} has inconsistent v2.2 recovery counts")


def _check_joint_v2_state(
    payload: object,
    *,
    label: str,
    rho_bounds: tuple[float, float],
) -> tuple[float, float]:
    if not isinstance(payload, dict) or payload.get("family") != "moment_t":
        raise SystemExit(f"{label} does not freeze the moment-t family")
    scales = payload.get("scales")
    gammas = payload.get("gammas")
    nus = payload.get("nus")
    if not isinstance(scales, dict) or not scales:
        raise SystemExit(f"{label} has no frozen sigma map")
    if not isinstance(gammas, dict) or set(gammas) != set(scales):
        raise SystemExit(f"{label} does not define gamma for every sigma key")
    if not isinstance(nus, dict) or set(nus) != set(scales):
        raise SystemExit(f"{label} does not define nu for every sigma key")
    try:
        sigma_values = [float(value) for value in scales.values()]
        gamma_values = [float(value) for value in gammas.values()]
        nu_values = [float(value) for value in nus.values()]
        rho = float(payload["rho"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"{label} has a nonnumeric frozen parameter") from error
    if not all(math.isfinite(value) and value > 0.0 for value in sigma_values):
        raise SystemExit(f"{label} has an invalid frozen sigma")
    if not all(
        math.isfinite(value)
        and JOINT_V2_GAMMA_BOUNDS[0] <= value <= JOINT_V2_GAMMA_BOUNDS[1]
        for value in gamma_values
    ):
        raise SystemExit(f"{label} has a gamma outside the preregistered bounds")
    gamma = gamma_values[0]
    if any(
        not math.isclose(value, gamma, rel_tol=0.0, abs_tol=1e-12)
        for value in gamma_values
    ):
        raise SystemExit(f"{label} does not use one shared gamma")
    if any(not math.isclose(value, 5.0, rel_tol=0.0, abs_tol=1e-12) for value in nu_values):
        raise SystemExit(f"{label} does not freeze nu=5")
    if not rho_bounds[0] <= rho <= rho_bounds[1]:
        raise SystemExit(f"{label} has rho outside the preregistered bounds")
    return rho, gamma


def _check_joint_v2_artifacts(root: Path, freeze: dict[str, object]) -> None:
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SystemExit(f"{root} joint v2 freeze has no artifact hashes")
    required = {
        "small_validation_manifest.csv",
        "bridge_config.one_layer.json",
        "bridge_config.hierarchical.json",
        "joint_rho_gamma_selection_report.one_layer.csv",
        "joint_rho_gamma_selection_report.hierarchical.csv",
        "bridge_component_calibration_report.csv",
        "parameter_selection_reference.json",
    }
    if not required <= set(artifacts):
        missing = ",".join(sorted(required - set(artifacts)))
        raise SystemExit(f"{root} joint v2 freeze omits artifacts: {missing}")
    for name, entry in artifacts.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise SystemExit(f"{root} joint v2 freeze has a malformed artifact entry")
        expected_path = (root / name).resolve()
        try:
            recorded_path = Path(str(entry["path"])).resolve()
            recorded_sha = str(entry["sha256"])
        except KeyError as error:
            raise SystemExit(f"{root} joint v2 artifact {name} has no path/hash") from error
        if recorded_path != expected_path or not expected_path.is_file():
            raise SystemExit(f"{root} joint v2 artifact {name} is missing or misbound")
        if _sha256(expected_path) != recorded_sha:
            raise SystemExit(f"{root} joint v2 artifact {name} fails SHA-256 verification")


def _check_joint_v2_freeze(
    root: Path,
    freeze: dict[str, object],
    *,
    dataset: str,
    contract: dict[str, str],
) -> tuple[
    tuple[float, float],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    if freeze.get("parameter_selection_protocol") != contract["parameter_selection_protocol"]:
        raise SystemExit(f"{root} has the wrong joint v2 parameter-selection protocol")
    if freeze.get("task_id") != dataset:
        raise SystemExit(f"{root} joint v2 freeze has the wrong task id")
    if not _float_map_matches(freeze.get("objective_weights"), JOINT_V2_WEIGHTS):
        raise SystemExit(f"{root} joint v2 freeze has the wrong objective weights")
    fixed = freeze.get("fixed_parameters")
    if not isinstance(fixed, dict) or (
        fixed.get("family") != "moment_t"
        or fixed.get("distribution") != "student_t"
        or fixed.get("sigma") != "frozen_alternate_formula_from_source_rho_run"
        or not _isclose_number(fixed.get("nu"), 5.0)
    ):
        raise SystemExit(f"{root} joint v2 freeze must fix moment-t Student-t nu=5")
    if _float_pair(freeze.get("gamma_bounds")) != JOINT_V2_GAMMA_BOUNDS:
        raise SystemExit(f"{root} joint v2 freeze has the wrong gamma bounds")
    rho_bounds = _joint_v2_expected_rho_bounds(dataset, freeze.get("rho_profile"))
    if _float_pair(freeze.get("rho_bounds")) != rho_bounds:
        raise SystemExit(f"{root} joint v2 freeze has the wrong rho bounds")
    if (
        int(freeze.get("validation_endpoint_rows", -1)) != 360
        or int(freeze.get("validation_fold_count", -1)) != 10
        or int(freeze.get("validation_endpoints_per_fold", -1)) != 36
        or freeze.get("validation_fold_column")
        not in {"smallval_fold_order", "fold_id"}
        or int(freeze.get("test_rows_used_for_tuning", -1)) != 0
        or int(freeze.get("embargo_rows_used_for_tuning", -1)) != 0
        or freeze.get("all_choices_frozen_before_test") is not True
    ):
        raise SystemExit(f"{root} joint v2 freeze violates validation-only selection")

    optimizer = freeze.get("optimizer")
    if not isinstance(optimizer, dict) or (
        optimizer.get("schema") != JOINT_V2_OPTIMIZER_SCHEMA
        or optimizer.get("max_evaluations_per_variant") != JOINT_V2_TOTAL_EVALUATIONS
        or optimizer.get("exploration_evaluation_limit_per_variant")
        != JOINT_V2_EXPLORATION_EVALUATIONS
        or optimizer.get("final_polish_evaluation_limit_per_variant")
        != JOINT_V2_FINAL_POLISH_EVALUATIONS
    ):
        raise SystemExit(f"{root} has the wrong joint v2 optimizer budget contract")
    if (
        freeze.get("max_evaluations") != JOINT_V2_TOTAL_EVALUATIONS
        or freeze.get("exploration_evaluation_limit")
        != JOINT_V2_EXPLORATION_EVALUATIONS
        or freeze.get("final_polish_reserve")
        != JOINT_V2_FINAL_POLISH_EVALUATIONS
        or freeze.get("required_convergence_status")
        != "verified_local_discrete"
    ):
        raise SystemExit(f"{root} has inconsistent top-level joint v2 budget metadata")
    variant_results = optimizer.get("variant_results")
    if not isinstance(variant_results, dict) or set(variant_results) != set(JOINT_VARIANTS):
        raise SystemExit(f"{root} joint v2 freeze omits variant optimizer results")

    selected = freeze.get("selected")
    if not isinstance(selected, dict) or set(selected) != set(JOINT_VARIANTS):
        raise SystemExit(f"{root} joint v2 freeze omits selected variant states")
    selected_states: dict[str, dict[str, object]] = {}
    checked_results: dict[str, dict[str, object]] = {}
    for variant in JOINT_VARIANTS:
        _check_joint_v2_state(
            selected[variant], label=f"{root}:{variant}", rho_bounds=rho_bounds
        )
        selected_states[variant] = selected[variant]
        result = variant_results[variant]
        _check_joint_v2_optimizer(result, label=f"{root}:{variant}")
        checked_results[variant] = result
    _check_joint_v2_artifacts(root, freeze)
    smallval_path = root / "small_validation_manifest.csv"
    if freeze.get("small_validation_manifest_sha256") != _sha256(smallval_path):
        raise SystemExit(f"{root} small-validation hash differs from its freeze")
    return rho_bounds, selected_states, checked_results


def _check_joint_v2_bridge(
    bridge_path: Path,
    payload: dict[str, object],
    *,
    variant: str,
    rho_bounds: tuple[float, float],
    selected_state: dict[str, object],
    optimizer_result: dict[str, object],
) -> None:
    metadata = payload.get("calibration_metadata")
    if not isinstance(metadata, dict):
        raise SystemExit(f"{bridge_path} has no joint v2 calibration metadata")
    obsolete_keys = {
        "allow_zero_gamma",
        "default_gamma",
        "fixed_gamma",
        "min_gamma_rows",
        "model_specific_rho",
        "only_optimized_coordinate",
        "rho_family_ess_penalty",
        "rho_model_ess_penalty",
        "rho_top1_penalty",
        "rho_top1_target",
    }
    inherited = obsolete_keys.intersection(metadata)
    inherited.update(
        str(key)
        for key in metadata
        if (
            str(key).startswith("gamma_")
            or str(key).startswith("rho_selection_")
        )
        and str(key)
        not in {
            "gamma_bounds",
            "gamma_anchor_values",
            "gamma_selection_performed",
            "gamma_selection_policy",
            "gamma_parameter_scope",
            "rho_selection_performed",
            "rho_selection_policy",
        }
    )
    joint_policy = (
        "joint_log_coordinate_safeguarded_newton_with_discrete_certificate"
    )
    if (
        inherited
        or metadata.get("rho_selection_policy") != joint_policy
        or metadata.get("gamma_selection_policy") != joint_policy
    ):
        raise SystemExit(
            f"{bridge_path} retains obsolete scalar-selection metadata: "
            f"{sorted(inherited)}"
        )
    if set(metadata.get("optimized_coordinates", [])) != {"global_rho", "shared_gamma"}:
        raise SystemExit(f"{bridge_path} does not declare both optimized coordinates")
    if (
        metadata.get("selected_bridge_family") != "moment_t"
        or metadata.get("distribution") != "student_t"
        or metadata.get("score_source") != "archive_moment"
        or metadata.get("sigma_selection_policy")
        != "byte_frozen_alternate_formula_from_source_rho_run"
        or metadata.get("sigma_selection_performed") is not False
        or metadata.get("sigma_calculation_performed") is not False
        or metadata.get("gamma_parameter_scope")
        != "shared_across_component_horizon_within_task_variant"
        or not _isclose_number(metadata.get("fixed_nu"), 5.0)
    ):
        raise SystemExit(f"{bridge_path} violates the fixed moment-t/nu=5 contract")
    if not _float_map_matches(metadata.get("objective_weights"), JOINT_V2_WEIGHTS):
        raise SystemExit(f"{bridge_path} has the wrong joint v2 objective weights")
    if (
        int(metadata.get("distinct_validation_endpoint_rows", -1)) != 360
        or int(metadata.get("validation_fold_count", -1)) != 10
        or int(metadata.get("validation_endpoints_per_fold", -1)) != 36
        or metadata.get("validation_fold_column")
        not in {"smallval_fold_order", "fold_id"}
        or int(metadata.get("test_rows_used_for_tuning", -1)) != 0
        or int(metadata.get("embargo_rows_used_for_tuning", -1)) != 0
    ):
        raise SystemExit(f"{bridge_path} violates the frozen fold provenance")
    if _float_pair(metadata.get("rho_bounds")) != rho_bounds or (
        _float_pair(metadata.get("gamma_bounds")) != JOINT_V2_GAMMA_BOUNDS
    ):
        raise SystemExit(f"{bridge_path} has bounds inconsistent with its freeze")
    _check_joint_v2_optimizer(metadata.get("optimizer"), label=str(bridge_path))
    if metadata.get("optimizer") != optimizer_result:
        raise SystemExit(f"{bridge_path} optimizer result differs from its freeze")

    selected_rho, selected_gamma = _check_joint_v2_state(
        selected_state,
        label=f"{bridge_path}:{variant}:freeze",
        rho_bounds=rho_bounds,
    )
    try:
        config_rho = float(payload["rho"])
        config_default_gamma = float(payload["default_gamma"])
        config_nu = float(payload["nu"])
        config_kernel_nu = float(payload["kernel_nu"])
        meta_rho = float(metadata["selected_rho"])
        meta_gamma = float(metadata["selected_gamma"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"{bridge_path} omits selected joint parameters") from error
    values = (config_rho, meta_rho, selected_rho)
    if any(not math.isclose(value, selected_rho, rel_tol=0.0, abs_tol=1e-12) for value in values):
        raise SystemExit(f"{bridge_path} selected rho differs from its freeze")
    values = (config_default_gamma, meta_gamma, selected_gamma)
    if any(not math.isclose(value, selected_gamma, rel_tol=0.0, abs_tol=1e-12) for value in values):
        raise SystemExit(f"{bridge_path} selected gamma differs from its freeze")
    gammas = payload.get("gamma_by_component")
    nus = payload.get("nu_by_component")
    sigmas = payload.get("sigma_by_component")
    frozen_sigmas = selected_state["scales"]
    if (
        not isinstance(gammas, dict)
        or set(gammas) != set(frozen_sigmas)
        or any(
            not math.isclose(
                float(value), selected_gamma, rel_tol=0.0, abs_tol=1e-12
            )
            for value in gammas.values()
        )
    ):
        raise SystemExit(f"{bridge_path} does not apply one shared gamma")
    if (
        payload.get("distribution") != "student_t"
        or payload.get("kernel_distribution") != "student_t"
    ):
        raise SystemExit(f"{bridge_path} does not configure Student-t scoring")
    if not isinstance(nus, dict) or set(nus) != set(gammas) or any(
        not math.isclose(float(value), 5.0, rel_tol=0.0, abs_tol=1e-12)
        for value in nus.values()
    ) or not math.isclose(config_nu, 5.0, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
        config_kernel_nu, 5.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SystemExit(f"{bridge_path} does not apply nu=5 to every component")
    if (
        not isinstance(sigmas, dict)
        or set(sigmas) != set(frozen_sigmas)
        or any(
            not math.isclose(
                float(sigmas[key]),
                float(frozen_sigmas[key]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for key in frozen_sigmas
        )
    ):
        raise SystemExit(f"{bridge_path} sigma map differs from its freeze")
def _joint_fullval_v2_expected_rho_bounds(dataset: str) -> tuple[float, float]:
    if dataset == "benchmark_a":
        return (0.5, 2.0)
    if dataset in JOINT_FULLVAL_V2_SHAPES:
        return (0.001, 0.5)
    raise SystemExit(f"joint full-validation freeze has unknown task {dataset!r}")


def _check_joint_fullval_v2_state(
    payload: object,
    *,
    label: str,
    rho_bounds: tuple[float, float],
) -> tuple[float, float]:
    ""

    if not isinstance(payload, dict) or payload.get("family") != "moment_t":
        raise SystemExit(f"{label} does not freeze the moment-t family")
    scales = payload.get("scales")
    gammas = payload.get("gammas")
    nus = payload.get("nus")
    if not isinstance(scales, dict) or not scales:
        raise SystemExit(f"{label} has no full-validation sigma map")
    if not isinstance(gammas, dict) or set(gammas) != set(scales):
        raise SystemExit(f"{label} does not define gamma for every sigma key")
    if not isinstance(nus, dict) or set(nus) != set(scales):
        raise SystemExit(f"{label} does not define nu for every sigma key")
    try:
        sigma_values = [float(value) for value in scales.values()]
        gamma_values = [float(value) for value in gammas.values()]
        nu_values = [float(value) for value in nus.values()]
        rho = float(payload["rho"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"{label} has a nonnumeric frozen parameter") from error
    if not all(
        math.isfinite(value) and value >= JOINT_FULLVAL_V2_MIN_SIGMA
        for value in sigma_values
    ):
        raise SystemExit(f"{label} has a sigma below the full-validation floor")
    if not all(
        math.isfinite(value)
        and JOINT_FULLVAL_V2_GAMMA_BOUNDS[0]
        <= value
        <= JOINT_FULLVAL_V2_GAMMA_BOUNDS[1]
        for value in gamma_values
    ):
        raise SystemExit(f"{label} has a gamma outside the full-validation bounds")
    gamma = gamma_values[0]
    if any(
        not math.isclose(value, gamma, rel_tol=0.0, abs_tol=1e-12)
        for value in gamma_values
    ):
        raise SystemExit(f"{label} does not use one shared gamma")
    if any(
        not math.isclose(value, 5.0, rel_tol=0.0, abs_tol=1e-12)
        for value in nu_values
    ):
        raise SystemExit(f"{label} does not freeze nu=5")
    if not rho_bounds[0] <= rho <= rho_bounds[1]:
        raise SystemExit(f"{label} has rho outside the full-validation bounds")
    return rho, gamma


def _check_joint_fullval_v2_artifacts(
    root: Path,
    freeze: dict[str, object],
) -> Path:
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SystemExit(f"{root} joint full-validation freeze has no artifact hashes")
    required = {
        JOINT_FULLVAL_V2_MANIFEST,
        "bridge_config.one_layer.json",
        "bridge_config.hierarchical.json",
        "joint_rho_gamma_selection_report.one_layer.csv",
        "joint_rho_gamma_selection_report.hierarchical.csv",
        "bridge_component_calibration_report.csv",
        "parameter_selection_reference.json",
    }
    if not required <= set(artifacts):
        missing = ",".join(sorted(required - set(artifacts)))
        raise SystemExit(
            f"{root} joint full-validation freeze omits artifacts: {missing}"
        )
    for name, entry in artifacts.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise SystemExit(
                f"{root} joint full-validation freeze has a malformed artifact entry"
            )
        expected_path = (root / name).resolve()
        try:
            recorded_path = Path(str(entry["path"])).resolve()
            recorded_sha = str(entry["sha256"])
        except KeyError as error:
            raise SystemExit(
                f"{root} joint full-validation artifact {name} has no path/hash"
            ) from error
        if recorded_path != expected_path or not expected_path.is_file():
            raise SystemExit(
                f"{root} joint full-validation artifact {name} is missing or misbound"
            )
        if _sha256(expected_path) != recorded_sha:
            raise SystemExit(
                f"{root} joint full-validation artifact {name} fails SHA-256 verification"
            )
    return (root / JOINT_FULLVAL_V2_MANIFEST).resolve()


def _check_joint_fullval_v2_1_amendment(payload: object, *, label: str) -> None:
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} omits the v2.1 amendment binding")
    required = {
        "schema": JOINT_FULLVAL_V2_1_AMENDMENT_SCHEMA,
        "sha256": JOINT_FULLVAL_V2_1_AMENDMENT_SHA256,
        "alternate_run_config_sha256": (
            JOINT_FULLVAL_V2_1_alternate_RUN_CONFIG_SHA256
        ),
        "alternate_log_sha256": JOINT_FULLVAL_V2_1_alternate_LOG_SHA256,
        "alternate_run_eligible_for_reuse": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise SystemExit(f"{label} has invalid v2.1 amendment field {key}")
    amendment_path = Path(str(payload.get("path", ""))).resolve()
    alternate_root = Path(str(payload.get("alternate_run_root", ""))).resolve()
    if not amendment_path.is_file() or _sha256(amendment_path) != (
        JOINT_FULLVAL_V2_1_AMENDMENT_SHA256
    ):
        raise SystemExit(f"{label} amendment file is missing or changed")
    run_config = alternate_root / "run_config.json"
    if not run_config.is_file() or _sha256(run_config) != (
        JOINT_FULLVAL_V2_1_alternate_RUN_CONFIG_SHA256
    ):
        raise SystemExit(f"{label} alternate run config is missing or changed")
    for task, expected_sha in JOINT_FULLVAL_V2_1_alternate_LOG_SHA256.items():
        log_path = alternate_root / "logs" / f"{task}_select_joint.log"
        if not log_path.is_file() or _sha256(log_path) != expected_sha:
            raise SystemExit(f"{label} alternate log changed for {task}")


def _check_joint_fullval_v2_2_amendment(payload: object, *, label: str) -> None:
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} omits the v2.2 amendment binding")
    required = {
        "schema": JOINT_FULLVAL_V2_2_AMENDMENT_SCHEMA,
        "sha256": JOINT_FULLVAL_V2_2_AMENDMENT_SHA256,
        "v2_1_run_config_sha256": (
            JOINT_FULLVAL_V2_2_V2_1_RUN_CONFIG_SHA256
        ),
        "v2_1_driver_command_sha256": JOINT_FULLVAL_V2_2_V2_1_DRIVER_SHA256,
        "v2_1_selection_log_sha256": JOINT_FULLVAL_V2_2_V2_1_LOG_SHA256,
        "v2_1_run_eligible_for_reuse": False,
        "fresh_task_variant_runs_required": 8,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise SystemExit(f"{label} has invalid v2.2 amendment field {key}")
    amendment_path = Path(str(payload.get("path", ""))).resolve()
    failed_root = Path(str(payload.get("v2_1_run_root", ""))).resolve()
    if not amendment_path.is_file() or _sha256(amendment_path) != (
        JOINT_FULLVAL_V2_2_AMENDMENT_SHA256
    ):
        raise SystemExit(f"{label} v2.2 amendment file is missing or changed")
    run_config = failed_root / "run_config.json"
    driver = failed_root / "logs/driver_command.txt"
    if (
        not run_config.is_file()
        or _sha256(run_config) != JOINT_FULLVAL_V2_2_V2_1_RUN_CONFIG_SHA256
        or not driver.is_file()
        or _sha256(driver) != JOINT_FULLVAL_V2_2_V2_1_DRIVER_SHA256
    ):
        raise SystemExit(f"{label} v2.1 failure identity changed")
    for task, expected_sha in JOINT_FULLVAL_V2_2_V2_1_LOG_SHA256.items():
        log_path = failed_root / "logs" / f"{task}_select_joint.log"
        if not log_path.is_file() or _sha256(log_path) != expected_sha:
            raise SystemExit(f"{label} v2.1 failure log changed for {task}")


def _check_joint_fullval_v2_freeze(
    root: Path,
    freeze: dict[str, object],
    *,
    dataset: str,
    contract: dict[str, str],
) -> tuple[
    tuple[float, float],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    ""

    strict_v2_1 = contract["freeze_schema"].endswith("_v2_1")
    strict_v2_2 = contract["freeze_schema"].endswith("_v2_2")
    expected_profile = (
        JOINT_FULLVAL_V2_2_RHO_PROFILE
        if strict_v2_2
        else JOINT_FULLVAL_V2_1_RHO_PROFILE
        if strict_v2_1
        else JOINT_FULLVAL_V2_RHO_PROFILE
    )

    if freeze.get("parameter_selection_protocol") != contract["parameter_selection_protocol"]:
        raise SystemExit(
            f"{root} has the wrong joint full-validation parameter-selection protocol"
        )
    if freeze.get("task_id") != dataset:
        raise SystemExit(f"{root} joint full-validation freeze has the wrong task id")
    if freeze.get("rho_profile") != expected_profile:
        raise SystemExit(f"{root} has the wrong joint full-validation rho profile")
    if strict_v2_1:
        _check_joint_fullval_v2_1_amendment(
            freeze.get("protocol_amendment"), label=str(root)
        )
    if strict_v2_2:
        _check_joint_fullval_v2_2_amendment(
            freeze.get("protocol_amendment"), label=str(root)
        )
    if not _float_map_matches(freeze.get("objective_weights"), JOINT_V2_WEIGHTS):
        raise SystemExit(
            f"{root} joint full-validation freeze has the wrong objective weights"
        )
    fixed = freeze.get("fixed_parameters")
    if not isinstance(fixed, dict) or (
        fixed.get("family") != "moment_t"
        or fixed.get("distribution") != "student_t"
        or fixed.get("sigma") != JOINT_FULLVAL_V2_FIXED_SIGMA
        or not _isclose_number(fixed.get("nu"), 5.0)
    ):
        raise SystemExit(
            f"{root} must freeze formula-calculated sigma and Student-t nu=5"
        )
    if (
        freeze.get("validation_scope") != "full_validation"
        or freeze.get("sigma_formula") != JOINT_FULLVAL_V2_SIGMA_FORMULA
        or freeze.get("sigma_grouping") != "component_horizon"
        or not _isclose_number(
            freeze.get("sigma_min"), JOINT_FULLVAL_V2_MIN_SIGMA
        )
        or not _isclose_number(
            freeze.get("sigma_default"), JOINT_FULLVAL_V2_DEFAULT_SIGMA
        )
    ):
        raise SystemExit(f"{root} has the wrong full-validation sigma formula")
    if _float_pair(freeze.get("gamma_bounds")) != JOINT_FULLVAL_V2_GAMMA_BOUNDS:
        raise SystemExit(f"{root} has the wrong joint full-validation gamma bounds")
    rho_bounds = _joint_fullval_v2_expected_rho_bounds(dataset)
    if _float_pair(freeze.get("rho_bounds")) != rho_bounds:
        raise SystemExit(f"{root} has the wrong joint full-validation rho bounds")

    endpoint_rows, fold_count, endpoints_per_fold = JOINT_FULLVAL_V2_SHAPES[dataset]
    if (
        int(freeze.get("validation_endpoint_rows", -1)) != endpoint_rows
        or int(freeze.get("validation_fold_count", -1)) != fold_count
        or int(freeze.get("validation_endpoints_per_fold", -1))
        != endpoints_per_fold
        or freeze.get("validation_fold_column") != "fold_id"
        or int(freeze.get("test_rows_used_for_tuning", -1)) != 0
        or int(freeze.get("embargo_rows_used_for_tuning", -1)) != 0
        or freeze.get("all_choices_frozen_before_test") is not True
    ):
        raise SystemExit(
            f"{root} joint full-validation freeze violates validation-only selection"
        )

    manifest_path = _check_joint_fullval_v2_artifacts(root, freeze)
    try:
        recorded_manifest_path = Path(
            str(freeze["full_validation_manifest_path"])
        ).resolve()
    except KeyError as error:
        raise SystemExit(
            f"{root} joint full-validation freeze omits its manifest path"
        ) from error
    if recorded_manifest_path != manifest_path:
        raise SystemExit(f"{root} full-validation manifest is misbound")
    if freeze.get("full_validation_manifest_sha256") != _sha256(manifest_path):
        raise SystemExit(f"{root} full-validation hash differs from its freeze")
    reference_path = (root / "parameter_selection_reference.json").resolve()
    if freeze.get("normalization_source_sha256") != _sha256(reference_path):
        raise SystemExit(
            f"{root} did not recompute normalization on full validation"
        )
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    expected_reference_schema = (
        "caster_shared_gamma_rho_joint_fullval_reference_v2_2"
        if strict_v2_2
        else "caster_shared_gamma_rho_joint_fullval_reference_v2_1"
        if strict_v2_1
        else "caster_shared_gamma_rho_joint_fullval_reference_v2"
    )
    if not isinstance(reference_payload, dict) or (
        reference_payload.get("schema") != expected_reference_schema
        or reference_payload.get("rho_profile") != expected_profile
    ):
        raise SystemExit(f"{root} has the wrong full-validation reference schema")
    if strict_v2_1:
        _check_joint_fullval_v2_1_amendment(
            reference_payload.get("protocol_amendment"),
            label=str(reference_path),
        )
    if strict_v2_2:
        _check_joint_fullval_v2_2_amendment(
            reference_payload.get("protocol_amendment"),
            label=str(reference_path),
        )

    optimizer_schema = (
        JOINT_FULLVAL_V2_2_OPTIMIZER_SCHEMA
        if strict_v2_2
        else JOINT_FULLVAL_V2_1_OPTIMIZER_SCHEMA
        if strict_v2_1
        else JOINT_V2_OPTIMIZER_SCHEMA
    )
    total_evaluations = (
        JOINT_FULLVAL_V2_2_TOTAL_EVALUATIONS
        if strict_v2_2
        else JOINT_FULLVAL_V2_1_TOTAL_EVALUATIONS
        if strict_v2_1
        else JOINT_V2_TOTAL_EVALUATIONS
    )
    exploration_evaluations = (
        JOINT_FULLVAL_V2_2_EXPLORATION_EVALUATIONS
        if strict_v2_2
        else JOINT_FULLVAL_V2_1_EXPLORATION_EVALUATIONS
        if strict_v2_1
        else JOINT_V2_EXPLORATION_EVALUATIONS
    )
    final_stage_evaluations = (
        JOINT_FULLVAL_V2_2_FINAL_STAGE_EVALUATIONS
        if strict_v2_2
        else JOINT_FULLVAL_V2_1_FINAL_STAGE_EVALUATIONS
        if strict_v2_1
        else JOINT_V2_FINAL_POLISH_EVALUATIONS
    )
    optimizer = freeze.get("optimizer")
    if not isinstance(optimizer, dict) or (
        optimizer.get("schema") != optimizer_schema
        or optimizer.get("max_evaluations_per_variant")
        != total_evaluations
        or optimizer.get("exploration_evaluation_limit_per_variant")
        != exploration_evaluations
        or optimizer.get("final_polish_evaluation_limit_per_variant")
        != final_stage_evaluations
    ):
        raise SystemExit(
            f"{root} has the wrong joint full-validation optimizer budget contract"
        )
    if (
        freeze.get("max_evaluations") != total_evaluations
        or freeze.get("exploration_evaluation_limit")
        != exploration_evaluations
        or freeze.get("final_polish_reserve")
        != final_stage_evaluations
        or freeze.get("required_convergence_status")
        != "verified_local_discrete"
    ):
        raise SystemExit(
            f"{root} has inconsistent full-validation optimizer metadata"
        )
    if strict_v2_1 and (
        optimizer.get("best_first_polish_evaluation_limit_per_variant")
        != JOINT_FULLVAL_V2_1_BEST_FIRST_POLISH_EVALUATIONS
        or optimizer.get("certificate_evaluation_reserve_per_variant")
        != JOINT_FULLVAL_V2_1_CERTIFICATE_EVALUATIONS
        or optimizer.get("certificate_max_attempts")
        != JOINT_FULLVAL_V2_1_CERTIFICATE_MAX_ATTEMPTS
        or optimizer.get("certificate_max_neighbors_per_attempt")
        != JOINT_FULLVAL_V2_1_CERTIFICATE_MAX_NEIGHBORS
        or optimizer.get("certificate_worst_case_evaluations")
        != JOINT_FULLVAL_V2_1_CERTIFICATE_EVALUATIONS
        or freeze.get("final_stage_reserve")
        != JOINT_FULLVAL_V2_1_FINAL_STAGE_EVALUATIONS
        or freeze.get("best_first_polish_reserve")
        != JOINT_FULLVAL_V2_1_BEST_FIRST_POLISH_EVALUATIONS
        or freeze.get("certificate_evaluation_reserve")
        != JOINT_FULLVAL_V2_1_CERTIFICATE_EVALUATIONS
    ):
        raise SystemExit(f"{root} has the wrong v2.1 certificate budget partition")
    if strict_v2_2 and (
        optimizer.get("best_first_polish_evaluation_limit_per_variant")
        != JOINT_FULLVAL_V2_2_BEST_FIRST_POLISH_EVALUATIONS
        or optimizer.get("certificate_evaluation_reserve_per_variant")
        != JOINT_FULLVAL_V2_2_CERTIFICATE_EVALUATIONS
        or optimizer.get("certificate_max_attempts")
        != JOINT_FULLVAL_V2_2_CERTIFICATE_MAX_ATTEMPTS
        or optimizer.get("certificate_max_neighbors_per_attempt")
        != JOINT_FULLVAL_V2_2_CERTIFICATE_MAX_NEIGHBORS
        or optimizer.get("certificate_worst_case_evaluations")
        != JOINT_FULLVAL_V2_2_CERTIFICATE_EVALUATIONS
        or optimizer.get("recovery_evaluation_reserve_per_variant")
        != JOINT_FULLVAL_V2_2_RECOVERY_EVALUATIONS
        or optimizer.get("recovery_sweep_evaluation_limit")
        != JOINT_FULLVAL_V2_2_RECOVERY_SWEEP_EVALUATIONS
        or optimizer.get("recovery_sweep_limit")
        != JOINT_FULLVAL_V2_2_RECOVERY_MAX_SWEEPS
        or optimizer.get("recovery_max_passes_per_sweep")
        != JOINT_FULLVAL_V2_2_RECOVERY_MAX_PASSES
        or optimizer.get("formal_replay_requires_all_eight_fresh_freezes")
        is not True
        or freeze.get("final_stage_reserve")
        != JOINT_FULLVAL_V2_2_FINAL_STAGE_EVALUATIONS
        or freeze.get("best_first_polish_reserve")
        != JOINT_FULLVAL_V2_2_BEST_FIRST_POLISH_EVALUATIONS
        or freeze.get("certificate_evaluation_reserve")
        != JOINT_FULLVAL_V2_2_CERTIFICATE_EVALUATIONS
        or freeze.get("recovery_evaluation_reserve")
        != JOINT_FULLVAL_V2_2_RECOVERY_EVALUATIONS
        or freeze.get("selection_outputs_released_after_both_variants_succeeded")
        is not True
        or freeze.get("active_likelihood_scale_map") != "sigma_by_component"
        or freeze.get("tau_equals_computed_sigma") is not True
        or freeze.get("tau_active_for_selected_family") is not False
        or freeze.get("tau_by_component_materialized") is not False
    ):
        raise SystemExit(f"{root} has the wrong v2.2 recovery budget partition")
    variant_results = optimizer.get("variant_results")
    if not isinstance(variant_results, dict) or set(variant_results) != set(
        JOINT_VARIANTS
    ):
        raise SystemExit(
            f"{root} joint full-validation freeze omits variant optimizer results"
        )
    selected = freeze.get("selected")
    if not isinstance(selected, dict) or set(selected) != set(JOINT_VARIANTS):
        raise SystemExit(
            f"{root} joint full-validation freeze omits selected variant states"
        )

    selected_states: dict[str, dict[str, object]] = {}
    checked_results: dict[str, dict[str, object]] = {}
    for variant in JOINT_VARIANTS:
        _check_joint_fullval_v2_state(
            selected[variant], label=f"{root}:{variant}", rho_bounds=rho_bounds
        )
        selected_states[variant] = selected[variant]
        result = variant_results[variant]
        _check_joint_v2_optimizer(
            result,
            label=f"{root}:{variant}",
            optimizer_schema=optimizer_schema,
            total_evaluations=total_evaluations,
            exploration_evaluations=exploration_evaluations,
            final_stage_evaluations=final_stage_evaluations,
            strict_v2_1=strict_v2_1,
            strict_v2_2=strict_v2_2,
        )
        checked_results[variant] = result

    one_scales = selected_states["one_layer"].get("scales")
    hierarchical_scales = selected_states["hierarchical"].get("scales")
    if not isinstance(one_scales, dict) or not _float_map_matches(
        hierarchical_scales,
        {str(key): float(value) for key, value in one_scales.items()},
    ):
        raise SystemExit(
            f"{root} full-validation variants do not share one calculated sigma map"
        )
    archive_path = (root / "forecast_archive.csv").resolve()
    if not archive_path.is_file() or freeze.get("sigma_input_archive_sha256") != _sha256(
        archive_path
    ):
        raise SystemExit(f"{root} formula-sigma archive is missing or misbound")
    validation = pd.read_csv(manifest_path, low_memory=False)
    archive = pd.read_csv(archive_path, low_memory=False)
    formula_scales = calibrate_component_sigma(
        validation,
        archive,
        transform="log1p",
        min_sigma=JOINT_FULLVAL_V2_MIN_SIGMA,
        default_sigma=JOINT_FULLVAL_V2_DEFAULT_SIGMA,
    )
    expected_formula_scales = {
        str(key): float(value) for key, value in formula_scales.items()
    }
    if not expected_formula_scales or not _float_map_matches(
        one_scales, expected_formula_scales
    ) or not _float_map_matches(
        reference_payload.get("sigma_formula_scales"), expected_formula_scales
    ):
        raise SystemExit(
            f"{root} sigma was not recomputed from the frozen full validation formula"
        )
    return rho_bounds, selected_states, checked_results


def _check_joint_fullval_v2_bridge(
    bridge_path: Path,
    payload: dict[str, object],
    *,
    dataset: str,
    variant: str,
    rho_bounds: tuple[float, float],
    selected_state: dict[str, object],
    optimizer_result: dict[str, object],
    strict_v2_1: bool = False,
    strict_v2_2: bool = False,
) -> None:
    metadata = payload.get("calibration_metadata")
    if not isinstance(metadata, dict):
        raise SystemExit(f"{bridge_path} has no joint full-validation metadata")
    if strict_v2_1:
        _check_joint_fullval_v2_1_amendment(
            metadata.get("protocol_amendment"), label=str(bridge_path)
        )
    if strict_v2_2:
        _check_joint_fullval_v2_2_amendment(
            metadata.get("protocol_amendment"), label=str(bridge_path)
        )
    obsolete_keys = {
        "allow_zero_gamma",
        "default_gamma",
        "fixed_gamma",
        "min_gamma_rows",
        "model_specific_rho",
        "only_optimized_coordinate",
        "rho_family_ess_penalty",
        "rho_model_ess_penalty",
        "rho_top1_penalty",
        "rho_top1_target",
    }
    inherited = obsolete_keys.intersection(metadata)
    inherited.update(
        str(key)
        for key in metadata
        if (
            str(key).startswith("gamma_")
            or str(key).startswith("rho_selection_")
        )
        and str(key)
        not in {
            "gamma_bounds",
            "gamma_anchor_values",
            "gamma_selection_performed",
            "gamma_selection_policy",
            "gamma_parameter_scope",
            "rho_selection_performed",
            "rho_selection_policy",
        }
    )
    joint_policy = (
        "joint_log_coordinate_safeguarded_newton_with_"
        "bounded_recovery_and_discrete_certificate"
        if strict_v2_2
        else "joint_log_coordinate_safeguarded_newton_with_discrete_certificate"
    )
    if (
        inherited
        or metadata.get("rho_selection_policy") != joint_policy
        or metadata.get("gamma_selection_policy") != joint_policy
    ):
        raise SystemExit(
            f"{bridge_path} retains obsolete scalar-selection metadata: "
            f"{sorted(inherited)}"
        )
    if set(metadata.get("optimized_coordinates", [])) != {
        "global_rho",
        "shared_gamma",
    }:
        raise SystemExit(
            f"{bridge_path} does not declare both optimized coordinates"
        )
    if (
        metadata.get("selected_bridge_family") != "moment_t"
        or metadata.get("bridge_family_selection_performed") is not False
        or metadata.get("distribution") != "student_t"
        or metadata.get("score_source") != "archive_moment"
        or metadata.get("rho_selection_performed") is not True
        or metadata.get("gamma_selection_performed") is not True
        or metadata.get("sigma_selection_policy")
        != JOINT_FULLVAL_V2_SIGMA_POLICY
        or metadata.get("sigma_calculation_performed") is not True
        or metadata.get("sigma_selection_performed") is not False
        or metadata.get("tau_equals_computed_sigma") is not True
        or metadata.get("sigma_formula") != JOINT_FULLVAL_V2_SIGMA_FORMULA
        or metadata.get("sigma_grouping") != "component_horizon"
        or not _isclose_number(
            metadata.get("sigma_min"), JOINT_FULLVAL_V2_MIN_SIGMA
        )
        or not _isclose_number(
            metadata.get("sigma_default"), JOINT_FULLVAL_V2_DEFAULT_SIGMA
        )
        or metadata.get("gamma_parameter_scope")
        != "shared_across_component_horizon_within_task_variant"
        or metadata.get("nu_selection_performed") is not False
        or not _isclose_number(metadata.get("fixed_nu"), 5.0)
    ):
        raise SystemExit(
            f"{bridge_path} violates the full-validation sigma/moment-t contract"
        )
    if (
        metadata.get("validation_scope") != "full_validation"
        or metadata.get("sigma_reused_from_source_run") is not False
        or "small_validation_manifest_path" in metadata
        or "small_validation_manifest_sha256" in metadata
    ):
        raise SystemExit(
            f"{bridge_path} retains small-validation sigma/provenance metadata"
        )
    if strict_v2_2 and (
        metadata.get("active_likelihood_scale_map") != "sigma_by_component"
        or metadata.get("tau_calculation_performed") is not True
        or metadata.get("tau_selection_performed") is not False
        or metadata.get("tau_formula") != "tau_equals_computed_sigma"
        or metadata.get("tau_active_for_selected_family") is not False
        or metadata.get("tau_by_component_materialized") is not False
    ):
        raise SystemExit(
            f"{bridge_path} violates the inactive tau=sigma moment-t contract"
        )
    if not _float_map_matches(metadata.get("objective_weights"), JOINT_V2_WEIGHTS):
        raise SystemExit(
            f"{bridge_path} has the wrong joint full-validation objective weights"
        )

    endpoint_rows, fold_count, endpoints_per_fold = JOINT_FULLVAL_V2_SHAPES[dataset]
    if (
        metadata.get("rho_profile")
        != (
            JOINT_FULLVAL_V2_2_RHO_PROFILE
            if strict_v2_2
            else JOINT_FULLVAL_V2_1_RHO_PROFILE
            if strict_v2_1
            else JOINT_FULLVAL_V2_RHO_PROFILE
        )
        or int(metadata.get("distinct_validation_endpoint_rows", -1))
        != endpoint_rows
        or int(metadata.get("validation_fold_count", -1)) != fold_count
        or int(metadata.get("validation_endpoints_per_fold", -1))
        != endpoints_per_fold
        or metadata.get("validation_fold_column") != "fold_id"
        or int(metadata.get("test_rows_used_for_tuning", -1)) != 0
        or int(metadata.get("embargo_rows_used_for_tuning", -1)) != 0
        or metadata.get("all_choices_frozen_before_test") is not True
    ):
        raise SystemExit(
            f"{bridge_path} violates the full-validation fold provenance"
        )
    manifest_path = (bridge_path.parent / JOINT_FULLVAL_V2_MANIFEST).resolve()
    try:
        recorded_manifest_path = Path(
            str(metadata["full_validation_manifest_path"])
        ).resolve()
    except KeyError as error:
        raise SystemExit(
            f"{bridge_path} omits its full-validation manifest path"
        ) from error
    if (
        recorded_manifest_path != manifest_path
        or not manifest_path.is_file()
        or metadata.get("full_validation_manifest_sha256") != _sha256(manifest_path)
    ):
        raise SystemExit(f"{bridge_path} has invalid full-validation provenance")
    if _float_pair(metadata.get("rho_bounds")) != rho_bounds or (
        _float_pair(metadata.get("gamma_bounds"))
        != JOINT_FULLVAL_V2_GAMMA_BOUNDS
    ):
        raise SystemExit(f"{bridge_path} has bounds inconsistent with its freeze")
    _check_joint_v2_optimizer(
        metadata.get("optimizer"),
        label=str(bridge_path),
        optimizer_schema=(
            JOINT_FULLVAL_V2_2_OPTIMIZER_SCHEMA
            if strict_v2_2
            else JOINT_FULLVAL_V2_1_OPTIMIZER_SCHEMA
            if strict_v2_1
            else JOINT_V2_OPTIMIZER_SCHEMA
        ),
        total_evaluations=(
            JOINT_FULLVAL_V2_2_TOTAL_EVALUATIONS
            if strict_v2_2
            else JOINT_FULLVAL_V2_1_TOTAL_EVALUATIONS
            if strict_v2_1
            else JOINT_V2_TOTAL_EVALUATIONS
        ),
        exploration_evaluations=(
            JOINT_FULLVAL_V2_2_EXPLORATION_EVALUATIONS
            if strict_v2_2
            else JOINT_FULLVAL_V2_1_EXPLORATION_EVALUATIONS
            if strict_v2_1
            else JOINT_V2_EXPLORATION_EVALUATIONS
        ),
        final_stage_evaluations=(
            JOINT_FULLVAL_V2_2_FINAL_STAGE_EVALUATIONS
            if strict_v2_2
            else JOINT_FULLVAL_V2_1_FINAL_STAGE_EVALUATIONS
            if strict_v2_1
            else JOINT_V2_FINAL_POLISH_EVALUATIONS
        ),
        strict_v2_1=strict_v2_1,
        strict_v2_2=strict_v2_2,
    )
    if metadata.get("optimizer") != optimizer_result:
        raise SystemExit(f"{bridge_path} optimizer result differs from its freeze")

    selected_rho, selected_gamma = _check_joint_fullval_v2_state(
        selected_state,
        label=f"{bridge_path}:{variant}:freeze",
        rho_bounds=rho_bounds,
    )
    try:
        config_rho = float(payload["rho"])
        config_default_gamma = float(payload["default_gamma"])
        config_default_sigma = float(payload["default_sigma"])
        config_default_tau = float(payload["default_tau"])
        config_nu = float(payload["nu"])
        config_kernel_nu = float(payload["kernel_nu"])
        meta_rho = float(metadata["selected_rho"])
        meta_gamma = float(metadata["selected_gamma"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(
            f"{bridge_path} omits selected full-validation parameters"
        ) from error
    if any(
        not math.isclose(value, selected_rho, rel_tol=0.0, abs_tol=1e-12)
        for value in (config_rho, meta_rho, selected_rho)
    ):
        raise SystemExit(f"{bridge_path} selected rho differs from its freeze")
    if any(
        not math.isclose(value, selected_gamma, rel_tol=0.0, abs_tol=1e-12)
        for value in (config_default_gamma, meta_gamma, selected_gamma)
    ):
        raise SystemExit(f"{bridge_path} selected gamma differs from its freeze")

    gammas = payload.get("gamma_by_component")
    nus = payload.get("nu_by_component")
    sigmas = payload.get("sigma_by_component")
    taus = payload.get("tau_by_component")
    frozen_sigmas = selected_state["scales"]
    if (
        not isinstance(gammas, dict)
        or set(gammas) != set(frozen_sigmas)
        or any(
            not math.isclose(
                float(value), selected_gamma, rel_tol=0.0, abs_tol=1e-12
            )
            for value in gammas.values()
        )
    ):
        raise SystemExit(f"{bridge_path} does not apply one shared gamma")
    if (
        payload.get("distribution") != "student_t"
        or payload.get("kernel_distribution") != "student_t"
    ):
        raise SystemExit(f"{bridge_path} does not configure Student-t scoring")
    if (
        not isinstance(nus, dict)
        or set(nus) != set(gammas)
        or any(
            not math.isclose(float(value), 5.0, rel_tol=0.0, abs_tol=1e-12)
            for value in nus.values()
        )
        or not math.isclose(config_nu, 5.0, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(config_kernel_nu, 5.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise SystemExit(f"{bridge_path} does not apply nu=5 to every component")
    if (
        not isinstance(sigmas, dict)
        or set(sigmas) != set(frozen_sigmas)
        or any(
            not math.isclose(
                float(sigmas[key]),
                float(frozen_sigmas[key]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for key in frozen_sigmas
        )
    ):
        raise SystemExit(f"{bridge_path} sigma map differs from its freeze")
    if strict_v2_2:
        median_sigma = float(
            median([float(value) for value in frozen_sigmas.values()])
        )
        if (
            not isinstance(taus, dict)
            or bool(taus)
            or not math.isclose(
                config_default_sigma,
                median_sigma,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                config_default_tau,
                config_default_sigma,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise SystemExit(
                f"{bridge_path} does not freeze tau=sigma under the inactive "
                "moment-t tau-map contract"
            )


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} is not a JSON object: {path}")
    return payload


def _check_hashed_artifacts(
    root: Path,
    artifacts: object,
    *,
    required: set[str],
    label: str,
) -> None:
    if not isinstance(artifacts, dict):
        raise SystemExit(f"{label} has no artifact inventory")
    missing = required - set(map(str, artifacts))
    if missing:
        raise SystemExit(f"{label} omits artifacts: {','.join(sorted(missing))}")
    for raw_name, raw_entry in artifacts.items():
        name = str(raw_name)
        if not isinstance(raw_entry, dict):
            raise SystemExit(f"{label} has a malformed artifact entry for {name}")
        expected_path = (root / name).resolve()
        try:
            recorded_path = Path(str(raw_entry["path"])).resolve()
            recorded_sha = str(raw_entry["sha256"])
        except KeyError as error:
            raise SystemExit(f"{label} artifact {name} omits path/hash") from error
        if recorded_path != expected_path or not expected_path.is_file():
            raise SystemExit(f"{label} artifact {name} is missing or misbound")
        if _sha256(expected_path) != recorded_sha:
            raise SystemExit(f"{label} artifact {name} fails SHA-256 verification")


def _check_coherent_censored_freeze(
    root: Path,
    freeze: dict[str, object],
    *,
    dataset: str,
    family: str,
    expected_contract: str = COHERENT_CENSORED_CONTRACT,
) -> dict[str, dict[str, object]]:
    label = f"{root} coherent-censored full-validation freeze"
    if dataset not in COHERENT_CENSORED_FULLVAL_SHAPES:
        raise SystemExit(f"{label} has an unknown task")
    rows, folds, extension_rows = COHERENT_CENSORED_FULLVAL_SHAPES[dataset]
    fixed = freeze.get("fixed_parameters")
    selected = freeze.get("selected")
    fold_validation = freeze.get("fold_assignment_validation")
    if not isinstance(fixed, dict) or not isinstance(selected, dict):
        raise SystemExit(f"{label} omits fixed or selected parameters")
    if set(selected) != set(JOINT_VARIANTS):
        raise SystemExit(f"{label} must freeze both filtering variants")
    if (
        freeze.get("bridge_family") != family
        or fixed.get("family") != family
        or fixed.get("distribution") != "student_t"
        or not _isclose_number(fixed.get("nu"), 5.0)
        or fixed.get("predictive_contract") != expected_contract
        or freeze.get("validation_scope") != "full_validation"
        or int(freeze.get("validation_endpoint_rows", -1)) != rows
        or int(freeze.get("formal_validation_rows", -1)) != rows
        or int(freeze.get("validation_fold_count", -1)) != folds
        or int(freeze.get("test_rows_used_for_tuning", -1)) != 0
        or int(freeze.get("embargo_rows_used_for_tuning", -1))
        != extension_rows
        or freeze.get("all_choices_frozen_before_test") is not True
        or freeze.get("optimizer")
        != "deterministic_bounded_log_coordinate_pattern_search"
        or not _isclose_number(freeze.get("coverage_feasibility_floor"), 0.87)
        or not _float_map_matches(
            freeze.get("objective_weights"),
            COHERENT_CENSORED_FULLVAL_WEIGHTS,
        )
    ):
        raise SystemExit(f"{label} violates its registered full-validation contract")
    if not isinstance(fold_validation, dict) or (
        int(fold_validation.get("effective_fold_assignment_rows", -1)) != rows
        or int(fold_validation.get("effective_fold_count", -1)) != folds
        or int(fold_validation.get("full_validation_fold_extension_rows", -1))
        != extension_rows
        or int(fold_validation.get("test_rows_used_for_tuning", -1)) != 0
        or int(fold_validation.get("embargo_rows_used_for_tuning", -1))
        != extension_rows
        or fold_validation.get("asof_release_rule")
        != "date_only_release_time_lte_forecast_origin"
        or fold_validation.get("allow_same_timestamp") is not True
    ):
        raise SystemExit(f"{label} has an invalid effective fold validation")
    _check_hashed_artifacts(
        root,
        freeze.get("artifacts"),
        required={
            "input_validation.json",
            "effective_full_validation_fold_manifest.csv",
            "bridge_config.one_layer.json",
            "bridge_config.hierarchical.json",
            "selection_trace.one_layer.csv",
            "selection_trace.hierarchical.csv",
        },
        label=label,
    )
    for variant, entry in selected.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("state"), dict):
            raise SystemExit(f"{label}:{variant} has a malformed selected state")
        state = entry["state"]
        if (
            state.get("family") != family
            or state.get("predictive_contract") != expected_contract
            or not _isclose_number(state.get("rho"), entry.get("parameters", {}).get("rho"))
        ):
            raise SystemExit(f"{label}:{variant} has an inconsistent selected state")
    return selected


def _smallval360_rho_bounds(dataset: str) -> tuple[float, float]:
    if dataset not in TASKS:
        raise SystemExit(f"unknown smallval360 task {dataset!r}")
    return (0.4, 0.6) if dataset == "benchmark_a" else (0.005, 0.5)


def _smallval360_joint_expected_settings(
    dataset: str,
    *,
    pilot_contract: str,
) -> dict[str, object]:
    ""

    if dataset not in TASKS:
        raise SystemExit(f"unknown smallval360 task {dataset!r}")
    if pilot_contract == MEAN_PRESERVING_CENSORED_SMALLVAL360_JOINT_PILOT:
        return {
            "pilot_contract": (
                MEAN_PRESERVING_CENSORED_SMALLVAL360_JOINT_PILOT
            ),
            "parameter_selection_protocol": (
                MEAN_PRESERVING_CENSORED_SMALLVAL360_JOINT_PROTOCOL
            ),
            "objective_weights": COHERENT_CENSORED_FULLVAL_WEIGHTS,
            "rho_bounds": _smallval360_rho_bounds(dataset),
            "gamma_bounds": COHERENT_CENSORED_SMALLVAL360_GAMMA_BOUNDS,
            "scale_multiplier_bounds": (
                COHERENT_CENSORED_SMALLVAL360_SCALE_BOUNDS
            ),
        }
    if (
        pilot_contract
        == MEAN_PRESERVING_CENSORED_SMALLVAL360_RMSE_HEAVY_PILOT
    ):
        bounds = COHERENT_CENSORED_SMALLVAL360_RMSE_HEAVY_BOUNDS[dataset]
        return {
            "pilot_contract": (
                MEAN_PRESERVING_CENSORED_SMALLVAL360_RMSE_HEAVY_PILOT
            ),
            "parameter_selection_protocol": (
                MEAN_PRESERVING_CENSORED_SMALLVAL360_RMSE_HEAVY_PROTOCOL
            ),
            "objective_weights": (
                COHERENT_CENSORED_SMALLVAL360_RMSE_HEAVY_WEIGHTS
            ),
            "rho_bounds": bounds["rho"],
            "gamma_bounds": bounds["gamma"],
            "scale_multiplier_bounds": bounds["scale_multiplier"],
        }
    raise SystemExit(f"unsupported smallval360 joint pilot {pilot_contract!r}")


def _check_smallval360_manifest(
    root: Path,
    freeze: dict[str, object],
    *,
    label: str,
) -> None:
    path = root / "small_validation_manifest.csv"
    if not path.is_file():
        raise SystemExit(f"{label} omits small_validation_manifest.csv")
    recorded_path = Path(
        str(freeze.get("small_validation_manifest_path", ""))
    ).resolve()
    if (
        recorded_path != path.resolve()
        or freeze.get("small_validation_manifest_sha256") != _sha256(path)
    ):
        raise SystemExit(f"{label} has a changed or misbound smallval manifest")
    manifest = pd.read_csv(path)
    required = {"forecast_id", "split", "smallval_fold_order"}
    if not required <= set(manifest):
        raise SystemExit(f"{label} smallval manifest omits required columns")
    fold_order = pd.to_numeric(
        manifest["smallval_fold_order"], errors="coerce"
    )
    fold_counts = fold_order.value_counts(dropna=False).to_dict()
    if (
        len(manifest) != 360
        or manifest["forecast_id"].astype(str).nunique() != 360
        or not manifest["split"].astype(str).eq("val").all()
        or fold_order.isna().any()
        or set(int(value) for value in fold_order) != set(range(1, 11))
        or any(int(fold_counts.get(value, 0)) != 36 for value in range(1, 11))
    ):
        raise SystemExit(f"{label} is not the exact 360-endpoint/10-fold subset")


def _check_coherent_censored_smallval360_joint_freeze(
    root: Path,
    freeze: dict[str, object],
    *,
    dataset: str,
    family: str,
    pilot_contract: str = (
        MEAN_PRESERVING_CENSORED_SMALLVAL360_JOINT_PILOT
    ),
) -> dict[str, dict[str, object]]:
    label = f"{root} coherent mean-preserving smallval360 joint freeze"
    rows, folds, per_fold = COHERENT_CENSORED_SMALLVAL360_JOINT_SHAPE
    expected = _smallval360_joint_expected_settings(
        dataset,
        pilot_contract=pilot_contract,
    )
    fixed = freeze.get("fixed_parameters")
    settings = freeze.get("settings")
    selected = freeze.get("selected")
    selected_ids = freeze.get("selected_model_ids")
    if (
        not isinstance(fixed, dict)
        or not isinstance(settings, dict)
        or not isinstance(selected, dict)
    ):
        raise SystemExit(f"{label} omits fixed, settings, or selected state")
    if (
        freeze.get("task_id") != dataset
        or freeze.get("pilot_contract")
        != expected["pilot_contract"]
        or freeze.get("parameter_selection_protocol")
        != expected["parameter_selection_protocol"]
        or freeze.get("predictive_contract")
        != MEAN_PRESERVING_CENSORED_CONTRACT
        or freeze.get("bridge_family") != family
        or fixed.get("family") != family
        or fixed.get("distribution") != "student_t"
        or fixed.get("predictive_contract")
        != MEAN_PRESERVING_CENSORED_CONTRACT
        or not _isclose_number(fixed.get("nu"), 5.0)
        or not _isclose_number(
            fixed.get("fixed_c_u"),
            COHERENT_CENSORED_SMALLVAL360_FIXED_C_U,
        )
        or fixed.get("c_u_mode") != "fixed"
        or freeze.get("validation_scope") != "smallval360"
        or int(freeze.get("validation_endpoint_rows", -1)) != rows
        or int(freeze.get("formal_validation_rows", -1)) != rows
        or int(freeze.get("validation_fold_count", -1)) != folds
        or int(freeze.get("validation_endpoints_per_fold", -1)) != per_fold
        or int(freeze.get("test_rows_used_for_tuning", -1)) != 0
        or int(freeze.get("embargo_rows_used_for_tuning", -1)) != 0
        or freeze.get("all_choices_frozen_before_test") is not True
        or freeze.get("optimizer")
        != "deterministic_bounded_log_coordinate_pattern_search"
        or freeze.get("coverage_feasibility_floor") is not None
        or not _float_map_matches(
            freeze.get("objective_weights"),
            expected["objective_weights"],
        )
        or settings.get("c_u_mode") != "fixed"
        or not _isclose_number(
            settings.get("fixed_c_u"),
            COHERENT_CENSORED_SMALLVAL360_FIXED_C_U,
        )
        or settings.get("coverage_floor") is not None
        or _float_pair(freeze.get("rho_bounds"))
        != expected["rho_bounds"]
        or _float_pair(freeze.get("gamma_bounds"))
        != expected["gamma_bounds"]
        or _float_pair(freeze.get("scale_multiplier_bounds"))
        != expected["scale_multiplier_bounds"]
        or set(freeze.get("optimized_coordinates", []))
        != (
            {"global_rho", "shared_gamma", "shared_scale_multiplier"}
            if family == "moment_t"
            else {"global_rho", "shared_scale_multiplier"}
        )
    ):
        raise SystemExit(f"{label} violates its registered experiment contract")
    if (
        not isinstance(selected_ids, list)
        or len(selected_ids) != 10
        or len(set(map(str, selected_ids))) != 10
        or set(selected) != set(JOINT_VARIANTS)
    ):
        raise SystemExit(f"{label} is not bound to both variants of Top-10")
    selection_path = root / "candidate_selection_log.csv"
    if not selection_path.is_file():
        raise SystemExit(f"{label} omits candidate_selection_log.csv")
    selection = pd.read_csv(selection_path)
    if (
        "model_id" not in selection
        or selection["model_id"].astype(str).tolist()
        != [str(value) for value in selected_ids]
    ):
        raise SystemExit(f"{label} Top-10 selection ordering changed")
    _check_smallval360_manifest(root, freeze, label=label)
    _check_hashed_artifacts(
        root,
        freeze.get("artifacts"),
        required={
            "input_validation.json",
            "effective_full_validation_fold_manifest.csv",
            "bridge_config.one_layer.json",
            "bridge_config.hierarchical.json",
            "selection_trace.one_layer.csv",
            "selection_trace.hierarchical.csv",
        },
        label=label,
    )
    for variant, entry in selected.items():
        if not isinstance(entry, dict):
            raise SystemExit(f"{label}:{variant} has no selected record")
        parameters = entry.get("parameters")
        state = entry.get("state")
        if (
            not isinstance(parameters, dict)
            or set(parameters)
            != {"rho", "gamma", "scale_multiplier", "c_u"}
            or not isinstance(state, dict)
            or state.get("family") != family
            or state.get("predictive_contract")
            != MEAN_PRESERVING_CENSORED_CONTRACT
            or not _isclose_number(state.get("rho"), parameters.get("rho"))
            or not _isclose_number(
                parameters.get("c_u"),
                COHERENT_CENSORED_SMALLVAL360_FIXED_C_U,
            )
        ):
            raise SystemExit(f"{label}:{variant} has an inconsistent joint state")
        rho = float(parameters["rho"])
        scale_multiplier = float(parameters["scale_multiplier"])
        if (
            not expected["rho_bounds"][0]
            <= rho
            <= expected["rho_bounds"][1]
            or not expected["scale_multiplier_bounds"][0]
            <= scale_multiplier
            <= expected["scale_multiplier_bounds"][1]
        ):
            raise SystemExit(f"{label}:{variant} selects outside its bounds")
        scales = state.get("scales")
        nus = state.get("nus")
        if (
            not isinstance(scales, dict)
            or not scales
            or not isinstance(nus, dict)
            or set(nus) != set(scales)
            or any(not _isclose_number(value, 5.0) for value in nus.values())
        ):
            raise SystemExit(f"{label}:{variant} has invalid scale/nu maps")
        gammas = state.get("gammas")
        if family == "moment_t":
            gamma = parameters.get("gamma")
            if (
                not _isclose_number(gamma, gamma)
                or not expected["gamma_bounds"][0]
                <= float(gamma)
                <= expected["gamma_bounds"][1]
                or not isinstance(gammas, dict)
                or set(gammas) != set(scales)
                or any(
                    not _isclose_number(value, float(gamma))
                    for value in gammas.values()
                )
            ):
                raise SystemExit(f"{label}:{variant} has invalid shared gamma")
        elif parameters.get("gamma") is not None or gammas != {}:
            raise SystemExit(f"{label}:{variant} activates gamma for draw-kernel")
    return selected


def _check_coherent_censored_bridge(
    bridge_path: Path,
    payload: dict[str, object],
    *,
    variant: str,
    family: str,
    selected: dict[str, dict[str, object]],
    expected_contract: str = COHERENT_CENSORED_CONTRACT,
    expected_protocol: str = COHERENT_CENSORED_PROTOCOL,
) -> None:
    label = f"{bridge_path} coherent-censored bridge"
    metadata = payload.get("calibration_metadata")
    if not isinstance(metadata, dict):
        raise SystemExit(f"{label} omits calibration metadata")
    state = selected[variant]["state"]
    active_scales = (
        payload.get("tau_by_component")
        if family == "draw_kernel_t"
        else payload.get("sigma_by_component")
    )
    if (
        metadata.get("parameter_selection_protocol")
        != expected_protocol
        or metadata.get("rho_selection_variant") != variant
        or metadata.get("selected_bridge_family") != family
        or metadata.get("predictive_contract") != expected_contract
        or payload.get("predictive_contract") != expected_contract
        or payload.get("distribution") != "student_t"
        or payload.get("kernel_distribution") != "student_t"
        or not _isclose_number(payload.get("rho"), state.get("rho"))
        or not _float_map_matches(active_scales, state.get("scales", {}))
        or not _float_map_matches(
            payload.get("nu_by_component"), state.get("nus", {})
        )
        or not _float_map_matches(
            payload.get("truncation_upper_raw_by_component"),
            state.get("truncation_upper_raw_by_component", {}),
        )
    ):
        raise SystemExit(f"{label} differs from its selected freeze state")
    if family == "moment_t":
        if not _float_map_matches(
            payload.get("gamma_by_component"), state.get("gammas", {})
        ) or payload.get("tau_by_component"):
            raise SystemExit(f"{label} has invalid moment-t coordinate maps")
    elif payload.get("sigma_by_component") or payload.get("gamma_by_component"):
        raise SystemExit(f"{label} has invalid draw-kernel coordinate maps")


def _check_coherent_censored_smallval360_joint_bridge(
    bridge_path: Path,
    payload: dict[str, object],
    *,
    variant: str,
    family: str,
    selected: dict[str, dict[str, object]],
    dataset: str = "benchmark_a",
    pilot_contract: str = (
        MEAN_PRESERVING_CENSORED_SMALLVAL360_JOINT_PILOT
    ),
) -> None:
    expected = _smallval360_joint_expected_settings(
        dataset,
        pilot_contract=pilot_contract,
    )
    _check_coherent_censored_bridge(
        bridge_path,
        payload,
        variant=variant,
        family=family,
        selected=selected,
        expected_contract=MEAN_PRESERVING_CENSORED_CONTRACT,
        expected_protocol=str(expected["parameter_selection_protocol"]),
    )
    label = f"{bridge_path} smallval360 joint bridge"
    metadata = payload.get("calibration_metadata")
    entry = selected[variant]
    parameters = entry["parameters"]
    state = entry["state"]
    selected_ids = metadata.get("selected_model_ids") if isinstance(metadata, dict) else None
    optimized = (
        {"global_rho", "shared_gamma", "shared_scale_multiplier"}
        if family == "moment_t"
        else {"global_rho", "shared_scale_multiplier"}
    )
    metadata_settings = (
        metadata.get("settings") if isinstance(metadata, dict) else None
    )
    fold_validation = (
        metadata.get("fold_assignment_validation")
        if isinstance(metadata, dict)
        else None
    )
    if (
        not isinstance(metadata, dict)
        or metadata.get("pilot_contract")
        != expected["pilot_contract"]
        or metadata.get("calibration_split") != "val"
        or int(metadata.get("formal_validation_rows", -1)) != 360
        or int(metadata.get("validation_fold_count", -1)) != 10
        or int(metadata.get("test_rows_used_for_tuning", -1)) != 0
        or int(metadata.get("embargo_rows_used_for_tuning", -1)) != 0
        or metadata.get("all_choices_frozen_before_test") is not True
        or not _float_map_matches(
            metadata.get("objective_weights"),
            expected["objective_weights"],
        )
        or _float_pair(metadata.get("rho_bounds"))
        != expected["rho_bounds"]
        or _float_pair(metadata.get("gamma_bounds"))
        != expected["gamma_bounds"]
        or _float_pair(metadata.get("scale_multiplier_bounds"))
        != expected["scale_multiplier_bounds"]
        or metadata.get("c_u_selection_mode") != "fixed"
        or not isinstance(metadata_settings, dict)
        or not _isclose_number(
            metadata_settings.get("fixed_c_u"),
            COHERENT_CENSORED_SMALLVAL360_FIXED_C_U,
        )
        or metadata_settings.get("coverage_floor") is not None
        or set(metadata.get("optimized_coordinates", [])) != optimized
        or int(metadata.get("selected_model_count", -1)) != 10
        or not isinstance(selected_ids, list)
        or len(selected_ids) != 10
        or len(set(map(str, selected_ids))) != 10
        or not _isclose_number(
            metadata.get("selected_parameters", {}).get("rho"),
            parameters["rho"],
        )
        or not _isclose_number(
            metadata.get("selected_parameters", {}).get("scale_multiplier"),
            parameters["scale_multiplier"],
        )
        or not _isclose_number(
            metadata.get("selected_parameters", {}).get("c_u"),
            COHERENT_CENSORED_SMALLVAL360_FIXED_C_U,
        )
    ):
        raise SystemExit(f"{label} violates the joint smallval360 metadata")
    if (
        not isinstance(fold_validation, dict)
        or fold_validation.get("validation_scope")
        != "frozen_small_validation_360"
        or int(fold_validation.get("effective_fold_assignment_rows", -1)) != 360
        or int(fold_validation.get("effective_fold_count", -1)) != 10
        or int(fold_validation.get("full_validation_fold_extension_rows", -1))
        != 0
        or int(fold_validation.get("test_rows_used_for_tuning", -1)) != 0
        or int(fold_validation.get("embargo_rows_used_for_tuning", -1)) != 0
    ):
        raise SystemExit(f"{label} has invalid frozen smallval fold provenance")
    if family == "moment_t" and not _isclose_number(
        metadata.get("selected_parameters", {}).get("gamma"),
        parameters["gamma"],
    ):
        raise SystemExit(f"{label} selected gamma differs from its freeze")
    if family == "draw_kernel_t" and metadata.get(
        "selected_parameters", {}
    ).get("gamma") is not None:
        raise SystemExit(f"{label} activates gamma for draw-kernel")
    base_scales = metadata.get("base_scales")
    if (
        not isinstance(base_scales, dict)
        or set(base_scales) != set(state["scales"])
        or any(
            not math.isclose(
                float(state["scales"][key]),
                float(base_scales[key]) * float(parameters["scale_multiplier"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for key in base_scales
        )
    ):
        raise SystemExit(f"{label} scale multiplier differs from selected state")
    base_upper = metadata.get("base_upper_raw_by_component")
    selected_upper = state.get("truncation_upper_raw_by_component")
    if (
        not isinstance(base_upper, dict)
        or not isinstance(selected_upper, dict)
        or set(base_upper) != set(selected_upper)
        or any(
            not math.isclose(
                float(selected_upper[key]),
                float(base_upper[key])
                * COHERENT_CENSORED_SMALLVAL360_FIXED_C_U,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for key in base_upper
        )
    ):
        raise SystemExit(f"{label} fixed c_U differs from selected support state")


def _v21_expected_input_task_root(manifest_path: Path, dataset: str) -> Path:
    relative = {
        "benchmark_a": Path("benchmark_a"),
        "benchmark_b_covid": Path("benchmark_b/benchmark_b_covid"),
        "benchmark_b_flu": Path("benchmark_b/benchmark_b_flu"),
        "benchmark_b_pooled": Path("benchmark_b/benchmark_b_pooled"),
    }[dataset]
    return (manifest_path.parent / "new_method/artifacts" / relative).resolve()


def _v21_expected_task_top_k(
    manifest_path: Path,
    dataset: str,
    *,
    label: str,
) -> int:
    ""

    run_config_path = manifest_path.parent / "run_config.json"
    if not run_config_path.is_file():
        return 10
    run_config = _read_json_object(
        run_config_path,
        label=f"{label} input run config",
    )
    top_k_by_task = run_config.get("top_k_by_task")
    if top_k_by_task is None:
        raw_top_k = run_config.get("top_k", 10)
        raw_top_k = 10 if raw_top_k is None else raw_top_k
    else:
        if not isinstance(top_k_by_task, dict):
            raise SystemExit(f"{label} input run config has invalid top_k_by_task")
        if dataset not in top_k_by_task:
            raise SystemExit(
                f"{label} input run config omits top_k_by_task[{dataset!r}]"
            )
        raw_top_k = top_k_by_task[dataset]
    try:
        top_k = int(raw_top_k)
    except (TypeError, ValueError) as error:
        raise SystemExit(
            f"{label} input run config has invalid Top-K for {dataset}"
        ) from error
    if isinstance(raw_top_k, bool) or top_k < 1 or str(raw_top_k).strip() != str(top_k):
        raise SystemExit(
            f"{label} input run config has invalid Top-K for {dataset}"
        )
    return top_k


def _check_v21_input_identity(
    payload: dict[str, object],
    manifest: dict[str, object],
    *,
    label: str,
) -> None:
    if manifest.get("schema") != V21_INPUT_MANIFEST_SCHEMA:
        raise SystemExit(f"{label} has invalid v21 input field schema")
    expected_shared_identity = {
        "selection_profile": V21_SELECTION_PROFILE,
        "selection_profile_status": V21_SELECTION_PROFILE_STATUS,
        "formal_preregistration_claimed": False,
    }
    for key, expected in expected_shared_identity.items():
        if manifest.get(key) != expected or payload.get(key) != expected:
            raise SystemExit(f"{label} has invalid v21 input field {key}")


def _check_v21_input_binding(
    payload: dict[str, object],
    *,
    dataset: str,
    label: str,
) -> dict[str, object]:
    try:
        manifest_path = Path(str(payload["input_manifest_path"])).resolve()
        recorded_sha = str(payload["input_manifest_sha256"])
        input_task_root = Path(str(payload["input_task_root"])).resolve()
    except KeyError as error:
        raise SystemExit(f"{label} omits its v21 input binding") from error
    manifest = _read_json_object(
        manifest_path, label=f"{label} v21 input manifest"
    )
    if _sha256(manifest_path) != recorded_sha:
        raise SystemExit(f"{label} v21 input manifest hash changed")
    _check_v21_input_identity(payload, manifest, label=label)
    selection_hash = str(manifest.get("selection_input_hash", ""))
    if len(selection_hash) != 64 or payload.get("selection_input_hash") != selection_hash:
        raise SystemExit(f"{label} has an invalid v21 selection-input hash")
    if int(manifest.get("n_draws", -1)) != 10:
        raise SystemExit(f"{label} does not bind the immutable 10-draw input view")
    if input_task_root != _v21_expected_input_task_root(manifest_path, dataset):
        raise SystemExit(f"{label} input task root is outside its v21 input bundle")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict) or not isinstance(tasks.get(dataset), dict):
        raise SystemExit(f"{label} input manifest omits task {dataset}")
    task_record = tasks[dataset]
    selected_ids = task_record.get("selected_model_ids")
    expected_top_k = _v21_expected_task_top_k(
        manifest_path,
        dataset,
        label=label,
    )
    manifest_top_k_by_task = manifest.get("top_k_by_task")
    if manifest_top_k_by_task is not None and (
        not isinstance(manifest_top_k_by_task, dict)
        or manifest_top_k_by_task.get(dataset) != expected_top_k
    ):
        raise SystemExit(f"{label} input manifest Top-K differs from its run config")
    if task_record.get("top_k") is not None and task_record.get("top_k") != expected_top_k:
        raise SystemExit(f"{label} task Top-K differs from its run config")
    if (
        not isinstance(selected_ids, list)
        or len(selected_ids) != expected_top_k
        or len(set(map(str, selected_ids))) != expected_top_k
    ):
        raise SystemExit(
            f"{label} is not bound to the configured Top-{expected_top_k}"
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SystemExit(f"{label} input manifest has no artifact inventory")
    required_names = {
        "event_ledger.csv",
        "forecast_archive.csv",
        "forecast_draws.csv",
        "forecast_archive_manifest.json",
        "selection_fold_manifest.csv",
        "model_registry.csv",
        "candidate_selection_log.csv",
    }
    expected_entries = {
        (input_task_root / name).resolve(): name for name in required_names
    }
    seen: set[str] = set()
    for raw_entry in artifacts.values():
        if not isinstance(raw_entry, dict):
            continue
        bound_path = Path(str(raw_entry.get("path", ""))).resolve()
        name = expected_entries.get(bound_path)
        if name is None:
            continue
        if not bound_path.is_file() or _sha256(bound_path) != str(
            raw_entry.get("sha256", "")
        ):
            raise SystemExit(f"{label} v21 input artifact changed: {bound_path}")
        seen.add(name)
    if seen != required_names:
        raise SystemExit(
            f"{label} v21 input manifest omits task artifacts: "
            f"{','.join(sorted(required_names - seen))}"
        )

    selection = pd.read_csv(input_task_root / "candidate_selection_log.csv")
    if "model_id" not in selection or selection["model_id"].astype(str).tolist() != [
        str(value) for value in selected_ids
    ]:
        raise SystemExit(f"{label} v21 selected-model ordering changed")
    if str(task_record.get("archive_sha256", "")) != _sha256(
        input_task_root / "forecast_archive.csv"
    ) or str(task_record.get("draws_sha256", "")) != _sha256(
        input_task_root / "forecast_draws.csv"
    ):
        raise SystemExit(f"{label} v21 task record does not bind archive/draws")
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": recorded_sha,
        "input_task_root": input_task_root,
        "selection_input_hash": selection_hash,
        "selected_model_ids": [str(value) for value in selected_ids],
    }


def _check_v21_archived_state(
    payload: object,
    *,
    label: str,
    rho_bounds: tuple[float, float],
) -> tuple[float, float]:
    if not isinstance(payload, dict) or payload.get("family") != "moment_t":
        raise SystemExit(f"{label} does not freeze moment_t")
    scales = payload.get("scales")
    gammas = payload.get("gammas")
    nus = payload.get("nus")
    if not isinstance(scales, dict) or not scales:
        raise SystemExit(f"{label} has no formula sigma map")
    if not isinstance(gammas, dict) or set(gammas) != set(scales):
        raise SystemExit(f"{label} has no gamma for every sigma key")
    if not isinstance(nus, dict) or set(nus) != set(scales):
        raise SystemExit(f"{label} has no nu for every sigma key")
    try:
        rho = float(payload["rho"])
        sigma_values = [float(value) for value in scales.values()]
        gamma_values = [float(value) for value in gammas.values()]
        nu_values = [float(value) for value in nus.values()]
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"{label} contains a nonnumeric parameter") from error
    if not rho_bounds[0] <= rho <= rho_bounds[1]:
        raise SystemExit(f"{label} rho is outside the archived bounds")
    if not all(
        math.isfinite(value) and value >= JOINT_FULLVAL_V2_MIN_SIGMA
        for value in sigma_values
    ):
        raise SystemExit(f"{label} contains an invalid formula sigma")
    if not all(
        math.isfinite(value)
        and V21_POSTHOC_GAMMA_BOUNDS[0]
        <= value
        <= V21_POSTHOC_GAMMA_BOUNDS[1]
        for value in gamma_values
    ):
        raise SystemExit(f"{label} gamma is outside the archived bounds")
    gamma = gamma_values[0]
    if any(
        not math.isclose(value, gamma, rel_tol=0.0, abs_tol=1e-12)
        for value in gamma_values
    ):
        raise SystemExit(f"{label} does not use one shared gamma")
    if any(
        not math.isclose(value, 5.0, rel_tol=0.0, abs_tol=1e-12)
        for value in nu_values
    ):
        raise SystemExit(f"{label} does not freeze nu=5")
    return rho, gamma


def _check_v21_archived_optimizer_contract(
    root: Path,
    freeze: dict[str, object],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    optimizer = freeze.get("optimizer")
    if not isinstance(optimizer, dict):
        raise SystemExit(f"{root} has no v21 archived optimizer contract")
    required_optimizer = {
        "schema": V21_POSTHOC_OPTIMIZER_SCHEMA,
        "max_evaluations_per_variant": JOINT_FULLVAL_V2_2_TOTAL_EVALUATIONS,
        "exploration_evaluation_limit_per_variant": (
            JOINT_FULLVAL_V2_2_EXPLORATION_EVALUATIONS
        ),
        "final_polish_evaluation_limit_per_variant": (
            JOINT_FULLVAL_V2_2_FINAL_STAGE_EVALUATIONS
        ),
        "best_first_polish_evaluation_limit_per_variant": (
            JOINT_FULLVAL_V2_2_BEST_FIRST_POLISH_EVALUATIONS
        ),
        "certificate_evaluation_reserve_per_variant": (
            JOINT_FULLVAL_V2_2_CERTIFICATE_EVALUATIONS
        ),
        "certificate_max_attempts": JOINT_FULLVAL_V2_2_CERTIFICATE_MAX_ATTEMPTS,
        "certificate_max_neighbors_per_attempt": (
            JOINT_FULLVAL_V2_2_CERTIFICATE_MAX_NEIGHBORS
        ),
        "certificate_worst_case_evaluations": (
            JOINT_FULLVAL_V2_2_CERTIFICATE_EVALUATIONS
        ),
        "certificate_attempt_evaluation_limit": (
            JOINT_FULLVAL_V2_2_CERTIFICATE_MAX_NEIGHBORS
        ),
        "recovery_evaluation_reserve_per_variant": (
            JOINT_FULLVAL_V2_2_RECOVERY_EVALUATIONS
        ),
        "recovery_sweep_evaluation_limit": (
            JOINT_FULLVAL_V2_2_RECOVERY_SWEEP_EVALUATIONS
        ),
        "recovery_sweep_limit": JOINT_FULLVAL_V2_2_RECOVERY_MAX_SWEEPS,
        "recovery_max_passes_per_sweep": JOINT_FULLVAL_V2_2_RECOVERY_MAX_PASSES,
        "formal_replay_requires_all_eight_fresh_freezes": True,
    }
    for key, expected in required_optimizer.items():
        if optimizer.get(key) != expected:
            raise SystemExit(f"{root} has invalid v21 optimizer field {key}")
    required_freeze = {
        "max_evaluations": JOINT_FULLVAL_V2_2_TOTAL_EVALUATIONS,
        "exploration_evaluation_limit": JOINT_FULLVAL_V2_2_EXPLORATION_EVALUATIONS,
        "final_polish_reserve": JOINT_FULLVAL_V2_2_FINAL_STAGE_EVALUATIONS,
        "final_stage_reserve": JOINT_FULLVAL_V2_2_FINAL_STAGE_EVALUATIONS,
        "best_first_polish_reserve": (
            JOINT_FULLVAL_V2_2_BEST_FIRST_POLISH_EVALUATIONS
        ),
        "certificate_evaluation_reserve": (
            JOINT_FULLVAL_V2_2_CERTIFICATE_EVALUATIONS
        ),
        "recovery_evaluation_reserve": JOINT_FULLVAL_V2_2_RECOVERY_EVALUATIONS,
        "recovery_sweep_evaluation_limit": (
            JOINT_FULLVAL_V2_2_RECOVERY_SWEEP_EVALUATIONS
        ),
        "recovery_sweep_limit": JOINT_FULLVAL_V2_2_RECOVERY_MAX_SWEEPS,
        "recovery_max_passes_per_sweep": JOINT_FULLVAL_V2_2_RECOVERY_MAX_PASSES,
        "required_convergence_status": "verified_local_discrete",
        "selection_outputs_released_after_both_variants_succeeded": True,
        "active_likelihood_scale_map": "sigma_by_component",
        "tau_equals_computed_sigma": True,
        "tau_active_for_selected_family": False,
        "tau_by_component_materialized": False,
    }
    for key, expected in required_freeze.items():
        if freeze.get(key) != expected:
            raise SystemExit(f"{root} has invalid v21 freeze field {key}")
    results = optimizer.get("variant_results")
    if not isinstance(results, dict) or set(results) != set(JOINT_VARIANTS):
        raise SystemExit(f"{root} omits v21 variant optimizer results")
    checked: dict[str, dict[str, object]] = {}
    for variant in JOINT_VARIANTS:
        result = results[variant]
        _check_joint_v2_optimizer(
            result,
            label=f"{root}:{variant}",
            optimizer_schema=V21_POSTHOC_OPTIMIZER_SCHEMA,
            total_evaluations=JOINT_FULLVAL_V2_2_TOTAL_EVALUATIONS,
            exploration_evaluations=JOINT_FULLVAL_V2_2_EXPLORATION_EVALUATIONS,
            final_stage_evaluations=JOINT_FULLVAL_V2_2_FINAL_STAGE_EVALUATIONS,
            strict_v2_2=True,
        )
        checked[variant] = result
    return optimizer, checked


def _check_v21_archived_moment_freeze(
    root: Path,
    freeze: dict[str, object],
    *,
    dataset: str,
) -> dict[str, object]:
    label = f"{root} v21 archived moment freeze"
    required_exact = {
        "schema": V21_POSTHOC_MOMENT_FREEZE_SCHEMA,
        "parameter_selection_protocol": V21_POSTHOC_MOMENT_PROTOCOL,
        "task_id": dataset,
        "rho_profile": V21_POSTHOC_PROFILE,
        "validation_scope": "full_validation",
        "selection_profile": V21_SELECTION_PROFILE,
        "selection_profile_status": V21_SELECTION_PROFILE_STATUS,
        "formal_preregistration_claimed": False,
        "test_rows_used_for_tuning": 0,
        "embargo_rows_used_for_tuning": 0,
        "all_choices_frozen_before_test": True,
    }
    for key, expected in required_exact.items():
        if freeze.get(key) != expected:
            raise SystemExit(f"{label} has invalid field {key}")
    forbidden = {
        "protocol_amendment",
        "source_rho_task_root",
        "formal_task_root",
        "source_small_validation_manifest_sha256",
    }
    inherited = forbidden.intersection(freeze)
    if inherited:
        raise SystemExit(f"{label} inherits old-run fields: {sorted(inherited)}")
    if not _float_map_matches(freeze.get("objective_weights"), JOINT_V2_WEIGHTS):
        raise SystemExit(f"{label} has the wrong objective weights")
    if _float_pair(freeze.get("gamma_bounds")) != V21_POSTHOC_GAMMA_BOUNDS:
        raise SystemExit(f"{label} has the wrong gamma bounds")
    rho_bounds = _joint_fullval_v2_expected_rho_bounds(dataset)
    if _float_pair(freeze.get("rho_bounds")) != rho_bounds:
        raise SystemExit(f"{label} has the wrong rho bounds")
    if not _float_map_matches(
        freeze.get("reference_point"), V21_POSTHOC_REFERENCE_POINT
    ):
        raise SystemExit(f"{label} has the wrong reference point")
    fixed = freeze.get("fixed_parameters")
    if not isinstance(fixed, dict) or (
        fixed.get("family") != "moment_t"
        or fixed.get("distribution") != "student_t"
        or fixed.get("sigma") != JOINT_FULLVAL_V2_FIXED_SIGMA
        or fixed.get("active_likelihood_scale_map") != "sigma_by_component"
        or not _isclose_number(fixed.get("nu"), 5.0)
    ):
        raise SystemExit(f"{label} does not freeze formula-sigma moment-t nu=5")
    if (
        freeze.get("sigma_formula") != JOINT_FULLVAL_V2_SIGMA_FORMULA
        or freeze.get("sigma_grouping") != "component_horizon"
        or not _isclose_number(freeze.get("sigma_min"), JOINT_FULLVAL_V2_MIN_SIGMA)
        or not _isclose_number(
            freeze.get("sigma_default"), JOINT_FULLVAL_V2_DEFAULT_SIGMA
        )
    ):
        raise SystemExit(f"{label} has the wrong sigma formula")
    endpoint_rows, fold_count, endpoints_per_fold = JOINT_FULLVAL_V2_SHAPES[dataset]
    if (
        int(freeze.get("validation_endpoint_rows", -1)) != endpoint_rows
        or int(freeze.get("validation_fold_count", -1)) != fold_count
        or int(freeze.get("validation_endpoints_per_fold", -1))
        != endpoints_per_fold
        or freeze.get("validation_fold_column") != "fold_id"
    ):
        raise SystemExit(f"{label} has the wrong full-validation shape")

    input_identity = _check_v21_input_binding(
        freeze, dataset=dataset, label=label
    )
    required_artifacts = {
        JOINT_FULLVAL_V2_MANIFEST,
        "bridge_config.one_layer.json",
        "bridge_config.hierarchical.json",
        "joint_rho_gamma_selection_report.one_layer.csv",
        "joint_rho_gamma_selection_report.hierarchical.csv",
        "bridge_component_calibration_report.csv",
        "parameter_selection_reference.json",
    }
    _check_hashed_artifacts(
        root,
        freeze.get("artifacts"),
        required=required_artifacts,
        label=label,
    )
    manifest_path = (root / JOINT_FULLVAL_V2_MANIFEST).resolve()
    try:
        recorded_manifest = Path(
            str(freeze["full_validation_manifest_path"])
        ).resolve()
    except KeyError as error:
        raise SystemExit(f"{label} omits the full-validation manifest path") from error
    if (
        recorded_manifest != manifest_path
        or freeze.get("full_validation_manifest_sha256") != _sha256(manifest_path)
    ):
        raise SystemExit(f"{label} has invalid full-validation provenance")

    reference_path = (root / "parameter_selection_reference.json").resolve()
    if freeze.get("normalization_source_sha256") != _sha256(reference_path):
        raise SystemExit(f"{label} has invalid normalization provenance")
    reference = _read_json_object(reference_path, label=f"{label} reference")
    if (
        reference.get("schema") != V21_POSTHOC_REFERENCE_SCHEMA
        or reference.get("rho_profile") != V21_POSTHOC_PROFILE
        or reference.get("task_id") != dataset
        or reference.get("validation_scope") != "full_validation"
        or "protocol_amendment" in reference
        or not _float_map_matches(
            reference.get("reference_point"), V21_POSTHOC_REFERENCE_POINT
        )
        or not _float_map_matches(reference.get("objective_weights"), JOINT_V2_WEIGHTS)
        or _float_pair(reference.get("rho_bounds")) != rho_bounds
        or _float_pair(reference.get("gamma_bounds"))
        != V21_POSTHOC_GAMMA_BOUNDS
        or reference.get("input_manifest_sha256")
        != input_identity["manifest_sha256"]
        or reference.get("selection_input_hash")
        != input_identity["selection_input_hash"]
    ):
        raise SystemExit(f"{label} reference does not match its freeze")

    _, optimizer_results = _check_v21_archived_optimizer_contract(root, freeze)
    selected = freeze.get("selected")
    if not isinstance(selected, dict) or set(selected) != set(JOINT_VARIANTS):
        raise SystemExit(f"{label} omits selected variant states")
    selected_states: dict[str, dict[str, object]] = {}
    for variant in JOINT_VARIANTS:
        _check_v21_archived_state(
            selected[variant], label=f"{label}:{variant}", rho_bounds=rho_bounds
        )
        selected_states[variant] = selected[variant]
    one_scales = selected_states["one_layer"].get("scales")
    hierarchical_scales = selected_states["hierarchical"].get("scales")
    if not isinstance(one_scales, dict) or not _float_map_matches(
        hierarchical_scales,
        {str(key): float(value) for key, value in one_scales.items()},
    ):
        raise SystemExit(f"{label} variants do not share the formula sigma map")
    archive_path = (root / "forecast_archive.csv").resolve()
    input_archive = Path(input_identity["input_task_root"]) / "forecast_archive.csv"
    if (
        not archive_path.is_file()
        or _sha256(archive_path) != _sha256(input_archive)
        or freeze.get("sigma_input_archive_sha256") != _sha256(archive_path)
    ):
        raise SystemExit(f"{label} formula-sigma archive is missing or misbound")
    validation = pd.read_csv(manifest_path, low_memory=False)
    archive = pd.read_csv(archive_path, low_memory=False)
    formula_scales = calibrate_component_sigma(
        validation,
        archive,
        transform="log1p",
        min_sigma=JOINT_FULLVAL_V2_MIN_SIGMA,
        default_sigma=JOINT_FULLVAL_V2_DEFAULT_SIGMA,
    )
    expected_scales = {str(key): float(value) for key, value in formula_scales.items()}
    if (
        not expected_scales
        or not _float_map_matches(one_scales, expected_scales)
        or not _float_map_matches(
            reference.get("sigma_formula_scales"), expected_scales
        )
    ):
        raise SystemExit(f"{label} sigma was not recomputed from full validation")
    return {
        "root": root.resolve(),
        "freeze": freeze,
        "freeze_path": (root / "parameter_selection_freeze_manifest.json").resolve(),
        "rho_bounds": rho_bounds,
        "selected_states": selected_states,
        "optimizer_results": optimizer_results,
        "input_identity": input_identity,
    }


def _check_v21_archived_moment_bridge(
    bridge_path: Path,
    payload: dict[str, object],
    *,
    dataset: str,
    variant: str,
    context: dict[str, object],
) -> None:
    label = f"{bridge_path} v21 archived moment bridge"
    metadata = payload.get("calibration_metadata")
    if not isinstance(metadata, dict):
        raise SystemExit(f"{label} has no calibration metadata")
    forbidden = {
        "protocol_amendment",
        "source_rho_task_root",
        "source_bridge_config_path",
        "small_validation_manifest_path",
        "small_validation_manifest_sha256",
    }
    inherited = forbidden.intersection(metadata)
    if inherited:
        raise SystemExit(f"{label} inherits old-run fields: {sorted(inherited)}")
    required_exact = {
        "parameter_selection_protocol": V21_POSTHOC_MOMENT_PROTOCOL,
        "rho_profile": V21_POSTHOC_PROFILE,
        "selection_profile": V21_SELECTION_PROFILE,
        "selection_profile_status": V21_SELECTION_PROFILE_STATUS,
        "formal_preregistration_claimed": False,
        "parameter_selection_split": "val",
        "validation_scope": "full_validation",
        "selected_bridge_family": "moment_t",
        "score_source": "archive_moment",
        "distribution": "student_t",
        "rho_selection_performed": True,
        "gamma_selection_performed": True,
        "sigma_selection_policy": JOINT_FULLVAL_V2_SIGMA_POLICY,
        "sigma_calculation_performed": True,
        "sigma_selection_performed": False,
        "sigma_formula": JOINT_FULLVAL_V2_SIGMA_FORMULA,
        "sigma_grouping": "component_horizon",
        "active_likelihood_scale_map": "sigma_by_component",
        "tau_calculation_performed": True,
        "tau_selection_performed": False,
        "tau_formula": "tau_equals_computed_sigma",
        "tau_active_for_selected_family": False,
        "tau_by_component_materialized": False,
        "nu_selection_performed": False,
        "test_rows_used_for_tuning": 0,
        "embargo_rows_used_for_tuning": 0,
        "all_choices_frozen_before_test": True,
    }
    for key, expected in required_exact.items():
        if metadata.get(key) != expected:
            raise SystemExit(f"{label} has invalid field {key}")
    if set(metadata.get("optimized_coordinates", [])) != {
        "global_rho",
        "shared_gamma",
    }:
        raise SystemExit(f"{label} does not declare both optimized coordinates")
    if not _float_map_matches(metadata.get("objective_weights"), JOINT_V2_WEIGHTS):
        raise SystemExit(f"{label} has the wrong objective weights")
    rho_bounds = context["rho_bounds"]
    if (
        _float_pair(metadata.get("rho_bounds")) != rho_bounds
        or _float_pair(metadata.get("gamma_bounds"))
        != V21_POSTHOC_GAMMA_BOUNDS
        or not _isclose_number(
            metadata.get("reference_rho"), V21_POSTHOC_REFERENCE_POINT["rho"]
        )
        or not _isclose_number(
            metadata.get("reference_gamma"), V21_POSTHOC_REFERENCE_POINT["gamma"]
        )
        or not _isclose_number(metadata.get("fixed_nu"), 5.0)
    ):
        raise SystemExit(f"{label} bounds/reference/nu differ from the freeze")
    input_identity = context["input_identity"]
    if (
        metadata.get("input_manifest_sha256")
        != input_identity["manifest_sha256"]
        or metadata.get("selection_input_hash")
        != input_identity["selection_input_hash"]
    ):
        raise SystemExit(f"{label} input identity differs from the freeze")
    selected_state = context["selected_states"][variant]
    optimizer_result = context["optimizer_results"][variant]
    _check_joint_v2_optimizer(
        metadata.get("optimizer"),
        label=label,
        optimizer_schema=V21_POSTHOC_OPTIMIZER_SCHEMA,
        total_evaluations=JOINT_FULLVAL_V2_2_TOTAL_EVALUATIONS,
        exploration_evaluations=JOINT_FULLVAL_V2_2_EXPLORATION_EVALUATIONS,
        final_stage_evaluations=JOINT_FULLVAL_V2_2_FINAL_STAGE_EVALUATIONS,
        strict_v2_2=True,
    )
    if metadata.get("optimizer") != optimizer_result:
        raise SystemExit(f"{label} optimizer result differs from the freeze")
    selected_rho, selected_gamma = _check_v21_archived_state(
        selected_state, label=f"{label}:freeze", rho_bounds=rho_bounds
    )
    try:
        config_rho = float(payload["rho"])
        config_gamma = float(payload["default_gamma"])
        config_sigma = float(payload["default_sigma"])
        config_tau = float(payload["default_tau"])
        config_nu = float(payload["nu"])
        kernel_nu = float(payload["kernel_nu"])
        meta_rho = float(metadata["selected_rho"])
        meta_gamma = float(metadata["selected_gamma"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"{label} omits selected parameters") from error
    if any(
        not math.isclose(value, selected_rho, rel_tol=0.0, abs_tol=1e-12)
        for value in (config_rho, meta_rho)
    ) or any(
        not math.isclose(value, selected_gamma, rel_tol=0.0, abs_tol=1e-12)
        for value in (config_gamma, meta_gamma)
    ):
        raise SystemExit(f"{label} selected rho/gamma differ from the freeze")
    scales = selected_state["scales"]
    sigmas = payload.get("sigma_by_component")
    gammas = payload.get("gamma_by_component")
    nus = payload.get("nu_by_component")
    taus = payload.get("tau_by_component")
    if not _float_map_matches(
        sigmas, {str(key): float(value) for key, value in scales.items()}
    ) or not _float_map_matches(
        gammas, {str(key): selected_gamma for key in scales}
    ) or not _float_map_matches(
        nus, {str(key): 5.0 for key in scales}
    ):
        raise SystemExit(f"{label} component maps differ from the freeze")
    median_sigma = float(median([float(value) for value in scales.values()]))
    if (
        not isinstance(taus, dict)
        or bool(taus)
        or payload.get("distribution") != "student_t"
        or payload.get("kernel_distribution") != "student_t"
        or not math.isclose(config_sigma, median_sigma, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(config_tau, config_sigma, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(config_nu, 5.0, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(kernel_nu, 5.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise SystemExit(f"{label} has an active or inconsistent moment tau map")


def _check_v21_archived_draw_freeze(
    root: Path,
    freeze: dict[str, object],
    *,
    dataset: str,
    moment_context: dict[str, object],
) -> dict[str, object]:
    label = f"{root} v21 derived draw freeze"
    required_exact = {
        "schema": V21_POSTHOC_DRAW_FREEZE_SCHEMA,
        "parameter_selection_protocol": V21_POSTHOC_DRAW_PROTOCOL,
        "task_id": dataset,
        "bridge_family": "draw_kernel_t",
        "rho_profile": V21_POSTHOC_PROFILE,
        "selection_profile": V21_SELECTION_PROFILE,
        "selection_profile_status": V21_SELECTION_PROFILE_STATUS,
        "formal_preregistration_claimed": False,
        "dimensional_binding": V21_POSTHOC_DIMENSIONAL_BINDING,
        "test_rows_used_for_tuning": 0,
        "embargo_rows_used_for_tuning": 0,
        "draw_test_rows_used_for_config_derivation": 0,
        "draw_embargo_rows_used_for_config_derivation": 0,
        "all_choices_frozen_before_test": True,
    }
    for key, expected in required_exact.items():
        if freeze.get(key) != expected:
            raise SystemExit(f"{label} has invalid field {key}")
    if "protocol_amendment" in freeze or "source_rho_task_root" in freeze:
        raise SystemExit(f"{label} inherits an old-run selection protocol")
    input_identity = moment_context["input_identity"]
    draw_input = _check_v21_input_binding(freeze, dataset=dataset, label=label)
    for key in ("manifest_sha256", "selection_input_hash", "input_task_root"):
        if draw_input[key] != input_identity[key]:
            raise SystemExit(f"{label} input identity differs from its moment parent")
    parent_root = Path(str(freeze.get("parent_moment_task_root", ""))).resolve()
    parent_freeze_path = Path(
        str(freeze.get("parent_moment_freeze_path", ""))
    ).resolve()
    expected_parent_root = Path(moment_context["root"]).resolve()
    expected_parent_freeze = Path(moment_context["freeze_path"]).resolve()
    if (
        parent_root != expected_parent_root
        or parent_freeze_path != expected_parent_freeze
        or not parent_freeze_path.is_file()
        or freeze.get("parent_moment_freeze_sha256")
        != _sha256(parent_freeze_path)
    ):
        raise SystemExit(f"{label} is not bound to its moment freeze")
    parent_configs = freeze.get("parent_variant_bridge_configs")
    selected = freeze.get("selected")
    if (
        not isinstance(parent_configs, dict)
        or set(parent_configs) != set(JOINT_VARIANTS)
        or not isinstance(selected, dict)
        or set(selected) != set(JOINT_VARIANTS)
    ):
        raise SystemExit(f"{label} omits a parent or selected variant")
    checked_parents: dict[str, dict[str, object]] = {}
    for variant in JOINT_VARIANTS:
        parent = parent_configs[variant]
        child = selected[variant]
        if not isinstance(parent, dict) or not isinstance(child, dict):
            raise SystemExit(f"{label}:{variant} has malformed lineage")
        expected_config = (parent_root / f"bridge_config.{variant}.json").resolve()
        parent_path = Path(str(parent.get("path", ""))).resolve()
        parent_state = moment_context["selected_states"][variant]
        parent_rho, parent_gamma = _check_v21_archived_state(
            parent_state,
            label=f"{label}:{variant}:parent",
            rho_bounds=moment_context["rho_bounds"],
        )
        if (
            parent_path != expected_config
            or not parent_path.is_file()
            or parent.get("sha256") != _sha256(parent_path)
            or not _isclose_number(parent.get("selected_rho"), parent_rho)
            or not _isclose_number(parent.get("selected_gamma"), parent_gamma)
        ):
            raise SystemExit(f"{label}:{variant} parent config is misbound")
        scales = parent_state["scales"]
        if (
            child.get("family") != "draw_kernel_t"
            or not _isclose_number(child.get("rho"), parent_rho)
            or not _float_map_matches(
                child.get("taus"),
                {str(key): parent_gamma for key in scales},
            )
            or not _float_map_matches(
                child.get("nus"), {str(key): 5.0 for key in scales}
            )
        ):
            raise SystemExit(f"{label}:{variant} is not derived from parent rho/gamma")
        checked_parents[variant] = {
            "path": parent_path,
            "sha256": str(parent["sha256"]),
            "rho": parent_rho,
            "gamma": parent_gamma,
            "keys": set(map(str, scales)),
        }
    _check_hashed_artifacts(
        root,
        freeze.get("artifacts"),
        required={
            "bridge_config.one_layer.json",
            "bridge_config.hierarchical.json",
        },
        label=label,
    )
    draws_path = (root / "forecast_draws.csv").resolve()
    input_draws = Path(draw_input["input_task_root"]) / "forecast_draws.csv"
    if not draws_path.is_file() or _sha256(draws_path) != _sha256(input_draws):
        raise SystemExit(f"{label} does not replay the immutable v21 draws")
    return {
        "root": root.resolve(),
        "freeze": freeze,
        "parents": checked_parents,
        "input_identity": draw_input,
    }


def _check_v21_archived_draw_bridge(
    bridge_path: Path,
    payload: dict[str, object],
    *,
    variant: str,
    context: dict[str, object],
) -> None:
    label = f"{bridge_path} v21 derived draw bridge"
    metadata = payload.get("calibration_metadata")
    if not isinstance(metadata, dict):
        raise SystemExit(f"{label} has no calibration metadata")
    required_exact = {
        "parameter_selection_protocol": V21_POSTHOC_DRAW_PROTOCOL,
        "rho_profile": V21_POSTHOC_PROFILE,
        "selection_profile": V21_SELECTION_PROFILE,
        "selection_profile_status": V21_SELECTION_PROFILE_STATUS,
        "formal_preregistration_claimed": False,
        "parameter_selection_split": "val",
        "selected_bridge_family": "draw_kernel_t",
        "score_source": "draw_kernel",
        "distribution": "student_t",
        "draw_kernel_tau_source": V21_POSTHOC_DRAW_TAU_SOURCE,
        "dimensional_binding": V21_POSTHOC_DIMENSIONAL_BINDING,
        "tau_parameter_scope": "shared_across_component_horizon_within_task_variant",
        "active_likelihood_scale_map": "tau_by_component",
        "tau_calculation_performed": True,
        "tau_selection_performed": False,
        "sigma_active_for_selected_family": False,
        "sigma_by_component_materialized": False,
        "sigma_selection_performed": False,
        "gamma_active_for_selected_family": False,
        "gamma_by_component_materialized": False,
        "gamma_selection_performed": False,
        "draw_test_rows_used_for_config_derivation": 0,
        "draw_embargo_rows_used_for_config_derivation": 0,
        "test_rows_used_for_tuning": 0,
        "embargo_rows_used_for_tuning": 0,
        "all_choices_frozen_before_test": True,
    }
    for key, expected in required_exact.items():
        if metadata.get(key) != expected:
            raise SystemExit(f"{label} has invalid field {key}")
    if "protocol_amendment" in metadata or "source_rho_task_root" in metadata:
        raise SystemExit(f"{label} inherits an old-run selection protocol")
    parent = context["parents"][variant]
    parent_freeze_path = Path(
        str(context["freeze"].get("parent_moment_freeze_path", ""))
    ).resolve()
    if (
        Path(str(metadata.get("parent_moment_freeze_path", ""))).resolve()
        != parent_freeze_path
        or metadata.get("parent_moment_freeze_sha256")
        != _sha256(parent_freeze_path)
        or Path(
            str(metadata.get("parent_moment_bridge_config_path", ""))
        ).resolve()
        != parent["path"]
        or metadata.get("parent_moment_bridge_config_sha256")
        != parent["sha256"]
        or not _isclose_number(metadata.get("parent_selected_rho"), parent["rho"])
        or not _isclose_number(
            metadata.get("parent_selected_gamma"), parent["gamma"]
        )
    ):
        raise SystemExit(f"{label} parent lineage differs from the draw freeze")
    input_identity = context["input_identity"]
    if (
        metadata.get("input_manifest_sha256")
        != input_identity["manifest_sha256"]
        or metadata.get("selection_input_hash")
        != input_identity["selection_input_hash"]
    ):
        raise SystemExit(f"{label} input identity differs from the draw freeze")
    try:
        rho = float(payload["rho"])
        default_tau = float(payload["default_tau"])
        nu = float(payload["nu"])
        kernel_nu = float(payload["kernel_nu"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"{label} omits draw parameters") from error
    expected_tau = {str(key): float(parent["gamma"]) for key in parent["keys"]}
    expected_nu = {str(key): 5.0 for key in parent["keys"]}
    if (
        not math.isclose(rho, float(parent["rho"]), rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(
            default_tau, float(parent["gamma"]), rel_tol=0.0, abs_tol=1e-12
        )
        or not _float_map_matches(payload.get("tau_by_component"), expected_tau)
        or not _float_map_matches(payload.get("nu_by_component"), expected_nu)
        or not isinstance(payload.get("sigma_by_component"), dict)
        or bool(payload.get("sigma_by_component"))
        or not isinstance(payload.get("gamma_by_component"), dict)
        or bool(payload.get("gamma_by_component"))
        or payload.get("kernel_distribution") != "student_t"
        or not math.isclose(nu, 5.0, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(kernel_nu, 5.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise SystemExit(f"{label} does not implement tau=parent-gamma exactly")


def _check_v21_w0_test_replay(
    root: Path,
    *,
    metadata_name: str,
    bridge_path: Path,
    family: str,
    variant: str,
) -> None:
    metadata_path = root / metadata_name
    metadata = _read_json_object(metadata_path, label="v21 W0 replay metadata")
    expected_score_source = "draw_kernel" if family == "draw_kernel_t" else "archive_moment"
    if (
        metadata.get("bridge_calibration_split") != "val"
        or int(metadata.get("test_rows_used_for_bridge_calibration", -1)) != 0
        or int(metadata.get("embargo_rows_used_for_bridge_calibration", -1)) != 0
        or metadata.get("readout_split") != "test"
        or metadata.get("score_source") != expected_score_source
        or str(metadata.get("bridge_config_sha256", "")) != _sha256(bridge_path)
        or Path(str(metadata.get("bridge_config", ""))).resolve()
        != bridge_path.resolve()
    ):
        raise SystemExit(f"{metadata_path} is not an independent validation-frozen test replay")
    if "parameter_selection_split" in metadata and metadata.get(
        "parameter_selection_split"
    ) != "val":
        raise SystemExit(f"{metadata_path} has a non-validation parameter split")
    warm_start_keys = {
        "warm_start",
        "warm_start_path",
        "initial_posterior_path",
        "source_posterior_path",
    }
    if any(metadata.get(key) not in {None, "", False} for key in warm_start_keys):
        raise SystemExit(f"{metadata_path} is not a fresh W0 replay")
    if variant == "one_layer":
        prior_path = root / "initial_prior.csv"
        if not prior_path.is_file():
            raise SystemExit(f"{metadata_path} omits persisted W0")
        prior = pd.read_csv(prior_path)
        if "model_id" not in prior or "weight" not in prior or prior.empty:
            raise SystemExit(f"{prior_path} is not a valid W0")
        weights = pd.to_numeric(prior["weight"], errors="coerce")
        expected = 1.0 / float(len(prior))
        if (
            prior["model_id"].astype(str).nunique() != len(prior)
            or weights.isna().any()
            or any(
                not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
                for value in weights
            )
        ):
            raise SystemExit(f"{prior_path} is not uniform W0")


def _task_root(args, task: str, *, draw_kernel: bool = False) -> Path:
    prefix = "draw_kernel_" if draw_kernel else ""
    return Path(getattr(args, f"{prefix}{task}_root"))


def _draw_roots_enabled(args) -> bool:
    values = [getattr(args, f"draw_kernel_{task}_root") for task in TASKS]
    if any(values) and not all(values):
        raise SystemExit("all four --draw-kernel-*-root options must be supplied together")
    return bool(values[0])


def _collect_export_frames(
    args,
    *,
    lanes_override: tuple[tuple[str, bool], ...] | None = None,
    tasks_override: tuple[str, ...] = TASKS,
    initial_v21_moment_contexts: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    ""

    out_dir = Path(args.out_dir)
    validation_path = out_dir / "asof_mixture_weight_validation.csv"
    metric_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []
    interval_frames: list[pd.DataFrame] = []
    macro_rows: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    pilot_contract = PILOT_CONTRACTS[str(args.pilot_contract)]
    draw_roots_enabled = _draw_roots_enabled(args)
    if str(args.pilot_contract) in {
        "gamma_only",
        "joint_rho_gamma",
        "joint_rho_gamma_fullval",
        "joint_rho_gamma_fullval_v2_1",
        "joint_rho_gamma_fullval_v2_2",
    } and draw_roots_enabled:
        raise SystemExit(
            f"{args.pilot_contract} export does not support draw-kernel roots"
        )

    lanes = [("moment_t", False)]
    if draw_roots_enabled:
        lanes.append(("draw_kernel_t", True))
    if lanes_override is not None:
        lanes = list(lanes_override)

    v21_moment_contexts: dict[str, dict[str, object]] = dict(
        initial_v21_moment_contexts or {}
    )
    for family, is_draw in lanes:
      for dataset in tasks_override:
        calibration_root = _task_root(args, dataset, draw_kernel=is_draw)
        ledger_path = calibration_root / "event_ledger.csv"
        archive_path = calibration_root / "forecast_archive.csv"
        freeze_path = calibration_root / "parameter_selection_freeze_manifest.json"
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        expected_freeze_schema = (
            pilot_contract["draw_freeze_schema"]
            if is_draw
            and str(args.pilot_contract) == "joint_rho_gamma_v21_archived_dual"
            else pilot_contract["freeze_schema"]
        )
        if freeze.get("schema") != expected_freeze_schema:
            raise SystemExit(
                f"{freeze_path} does not match the {args.pilot_contract} freeze contract"
            )
        if int(freeze.get("test_rows_used_for_tuning", -1)) != 0:
            raise SystemExit(f"{freeze_path} records test rows used for tuning")
        frozen_family = (
            freeze.get("bridge_family")
            if is_draw
            and str(args.pilot_contract) == "joint_rho_gamma_v21_archived_dual"
            else freeze.get("fixed_parameters", {}).get("family")
        )
        if frozen_family != family:
            raise SystemExit(f"{freeze_path} does not freeze {family}")
        fixed_parameters = freeze.get("fixed_parameters", {})
        if not isinstance(fixed_parameters, dict):
            raise SystemExit(f"{freeze_path} has invalid fixed_parameters")
        frozen_predictive_contract = str(
            fixed_parameters.get(
                "predictive_contract",
                alternate_ARCHIVE_MOMENT,
            )
        )
        if frozen_predictive_contract not in PREDICTIVE_CONTRACTS:
            raise SystemExit(
                f"{freeze_path} freezes an unsupported predictive contract"
            )
        frozen_distribution = ""
        if str(args.pilot_contract) == "rho_only":
            frozen_distribution = str(
                freeze.get("fixed_parameters", {}).get("distribution", "")
            )
            if frozen_distribution not in {"student_t", "gaussian"}:
                raise SystemExit(
                    f"{freeze_path} does not freeze a supported bridge distribution"
                )
        joint_v2_contract = None
        joint_fullval_v2_contract = None
        v21_contract = None
        coherent_censored_selected = None
        coherent_smallval360_joint_selected = None
        if str(args.pilot_contract) == "joint_rho_gamma":
            joint_v2_contract = _check_joint_v2_freeze(
                calibration_root,
                freeze,
                dataset=dataset,
                contract=pilot_contract,
            )
        elif str(args.pilot_contract) == "joint_rho_gamma_v21_archived_dual":
            if is_draw:
                if dataset not in v21_moment_contexts:
                    raise SystemExit(
                        f"{freeze_path} has no verified moment-t parent context"
                    )
                v21_contract = _check_v21_archived_draw_freeze(
                    calibration_root,
                    freeze,
                    dataset=dataset,
                    moment_context=v21_moment_contexts[dataset],
                )
            else:
                v21_contract = _check_v21_archived_moment_freeze(
                    calibration_root,
                    freeze,
                    dataset=dataset,
                )
                v21_moment_contexts[dataset] = v21_contract
        elif str(args.pilot_contract) in {
            "joint_rho_gamma_fullval",
            "joint_rho_gamma_fullval_v2_1",
            "joint_rho_gamma_fullval_v2_2",
        }:
            joint_fullval_v2_contract = _check_joint_fullval_v2_freeze(
                calibration_root,
                freeze,
                dataset=dataset,
                contract=pilot_contract,
            )
        elif str(args.pilot_contract) in {
            "coherent_censored_fullval",
            "coherent_mean_preserving_censored_fullval",
        }:
            coherent_censored_selected = _check_coherent_censored_freeze(
                calibration_root,
                freeze,
                dataset=dataset,
                family=family,
                expected_contract=str(
                    pilot_contract["predictive_contract"]
                ),
            )
        elif (
            str(args.pilot_contract)
            in MEAN_PRESERVING_CENSORED_SMALLVAL360_JOINT_PILOTS
        ):
            coherent_smallval360_joint_selected = (
                _check_coherent_censored_smallval360_joint_freeze(
                    calibration_root,
                    freeze,
                    dataset=dataset,
                    family=family,
                    pilot_contract=str(args.pilot_contract),
                )
            )

        method_prefix = "caster_"
        method_suffix = "_draw_kernel" if is_draw else ""
        for method, variant, metadata_name, readout_name, hierarchical in (
            (
                f"{method_prefix}one_layer{method_suffix}",
                "one_layer",
                "caster_run_metadata.json",
                "forecast_readout.csv",
                False,
            ),
            (
                f"{method_prefix}hierarchical{method_suffix}",
                "hierarchical",
                "hierarchical_run_metadata.json",
                "hierarchical_forecast_readout.csv",
                True,
            ),
        ):
            root = (
                calibration_root / ("hierarchical" if hierarchical else "one_layer")
                if is_draw
                else calibration_root
            )
            selection_path = root / "candidate_selection_log.csv"
            bridge_path = calibration_root / f"bridge_config.{variant}.json"
            payload = json.loads(bridge_path.read_text(encoding="utf-8"))
            bridge_meta = payload.get("calibration_metadata", {})
            distribution = str(payload.get("distribution", ""))
            kernel_distribution = str(payload.get("kernel_distribution", ""))
            config_predictive_contract = str(
                payload.get("predictive_contract", alternate_ARCHIVE_MOMENT)
            )
            metadata_predictive_contract = str(
                bridge_meta.get("predictive_contract", alternate_ARCHIVE_MOMENT)
            )
            if (
                config_predictive_contract != frozen_predictive_contract
                or metadata_predictive_contract != frozen_predictive_contract
            ):
                raise SystemExit(
                    f"{bridge_path} predictive contract differs from its freeze"
                )
            if str(args.pilot_contract) == "rho_only" and (
                distribution != frozen_distribution
                or kernel_distribution != frozen_distribution
                or str(bridge_meta.get("distribution", "")) != frozen_distribution
                or str(bridge_meta.get("kernel_distribution", "")) != frozen_distribution
            ):
                raise SystemExit(
                    f"{bridge_path} distribution differs from its rho-only freeze"
                )
            expected_protocol = (
                pilot_contract["draw_parameter_selection_protocol"]
                if is_draw
                and str(args.pilot_contract)
                == "joint_rho_gamma_v21_archived_dual"
                else pilot_contract["parameter_selection_protocol"]
            )
            if (
                bridge_meta.get("parameter_selection_protocol")
                != expected_protocol
            ):
                raise SystemExit(
                    f"{bridge_path} has the wrong {args.pilot_contract} protocol"
                )
            expected_score_source = "draw_kernel" if is_draw else "archive_moment"
            if bridge_meta.get("selected_bridge_family") != family:
                raise SystemExit(f"{bridge_path} must fix {family} in this pilot")
            if bridge_meta.get("score_source") != expected_score_source:
                raise SystemExit(f"{bridge_path} must use {expected_score_source} scoring")
            if joint_v2_contract is not None:
                rho_bounds, selected_states, optimizer_results = joint_v2_contract
                _check_joint_v2_bridge(
                    bridge_path,
                    payload,
                    variant=variant,
                    rho_bounds=rho_bounds,
                    selected_state=selected_states[variant],
                    optimizer_result=optimizer_results[variant],
                )
            if joint_fullval_v2_contract is not None:
                rho_bounds, selected_states, optimizer_results = (
                    joint_fullval_v2_contract
                )
                _check_joint_fullval_v2_bridge(
                    bridge_path,
                    payload,
                    dataset=dataset,
                    variant=variant,
                    rho_bounds=rho_bounds,
                    selected_state=selected_states[variant],
                    optimizer_result=optimizer_results[variant],
                    strict_v2_1=(
                        str(args.pilot_contract)
                        == "joint_rho_gamma_fullval_v2_1"
                    ),
                    strict_v2_2=(
                        str(args.pilot_contract)
                        == "joint_rho_gamma_fullval_v2_2"
                    ),
                )
            if v21_contract is not None:
                if is_draw:
                    _check_v21_archived_draw_bridge(
                        bridge_path,
                        payload,
                        variant=variant,
                        context=v21_contract,
                    )
                else:
                    _check_v21_archived_moment_bridge(
                        bridge_path,
                        payload,
                        dataset=dataset,
                        variant=variant,
                        context=v21_contract,
                    )
                _check_v21_w0_test_replay(
                    root,
                    metadata_name=metadata_name,
                    bridge_path=bridge_path,
                    family=family,
                    variant=variant,
                )
            if coherent_censored_selected is not None:
                _check_coherent_censored_bridge(
                    bridge_path,
                    payload,
                    variant=variant,
                    family=family,
                    selected=coherent_censored_selected,
                    expected_contract=str(
                        pilot_contract["predictive_contract"]
                    ),
                    expected_protocol=str(
                        pilot_contract["parameter_selection_protocol"]
                    ),
                )
            if coherent_smallval360_joint_selected is not None:
                _check_coherent_censored_smallval360_joint_bridge(
                    bridge_path,
                    payload,
                    variant=variant,
                    family=family,
                    selected=coherent_smallval360_joint_selected,
                    dataset=dataset,
                    pilot_contract=str(args.pilot_contract),
                )
                _check_v21_w0_test_replay(
                    root,
                    metadata_name=metadata_name,
                    bridge_path=bridge_path,
                    family=family,
                    variant=variant,
                )
            contract = core._contract(root, metadata_name, payload)
            core._check_contract(contract, f"{dataset}:{method}")
            contract = {
                **contract,
                "posterior_readout_policy": core.ASOF_LTE_POLICY,
                "release_availability_rule": "release_time_no_later_than_forecast_origin",
            }
            score_kwargs = dict(
                dataset=dataset,
                method=method,
                ledger_path=ledger_path,
                archive_path=archive_path,
                bridge_path=bridge_path,
                root=root,
                policy=core.ASOF_LTE_POLICY,
                hierarchical=hierarchical,
                selection_path=selection_path,
                weights_path=(root / "posterior_weights.csv") if not hierarchical else None,
            )
            if is_draw:
                score_kwargs["draws_path"] = calibration_root / "forecast_draws.csv"
                scores, validation = core._draw_kernel_asof_mixture_scores(**score_kwargs)
            else:
                scores, validation = core._bridge_asof_mixture_scores(**score_kwargs)
            score_label = (
                "draw-kernel" if is_draw else "moment-t"
            ) if distribution == "student_t" else (
                "Gaussian draw-kernel" if is_draw else "Gaussian moment"
            )
            macro, slices = core._metric_frame(
                dataset,
                method,
                "caster",
                root / readout_name,
                scores,
                contract,
                nll_reason=(
                    (
                        "exact as-of posterior mixed-measure mixture with "
                        "continuous interior density and censoring atoms at "
                        "zero and the frozen upper bound; "
                        if frozen_predictive_contract
                        in COHERENT_CENSORED_CONTRACTS
                        else f"exact as-of posterior {score_label} mixture; "
                    )
                    +
                    f"{pilot_contract['selection_description']}"
                ),
                asof_weight_validation_path=validation_path,
            )
                                                                             
                                                                            
                                                                 
            macro["bridge_config_path"] = str(bridge_path)
            slices["bridge_config_path"] = str(bridge_path)
            macro["bridge_family"] = family
            slices["bridge_family"] = family
            if frozen_predictive_contract in COHERENT_CENSORED_CONTRACTS:
                macro["nll_measure_basis"] = (
                    COHERENT_CENSORED_NLL_MEASURE_BASIS
                )
                slices["nll_measure_basis"] = (
                    COHERENT_CENSORED_NLL_MEASURE_BASIS
                )
                macro["posterior_predictive_measure"] = (
                    "continuous interior plus censoring atoms at zero and "
                    "the frozen upper bound"
                )
                slices["posterior_predictive_measure"] = (
                    "continuous interior plus censoring atoms at zero and "
                    "the frozen upper bound"
                )
            comparison_role = (
                ("draw_kernel_branch" if is_draw else "moment_t_branch")
                if distribution == "student_t"
                else (
                    "draw_kernel_gaussian_branch"
                    if is_draw
                    else "moment_gaussian_branch"
                )
            )
            macro["comparison_role"] = comparison_role
            slices["comparison_role"] = comparison_role
            macro_rows.append(macro)
            metric_frames.append(slices)
            alias = core._pooled_result_alias_slices(dataset, method, slices)
            if not alias.empty:
                metric_frames.append(alias)
            score_frames.append(scores)
            validation_frames.append(validation)
            interval_frames.append(
                core._forecast_intervals(
                    dataset,
                    method,
                    root / readout_name,
                )
            )
            manifest_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "bridge_family": family,
                    "distribution": distribution,
                    "kernel_distribution": kernel_distribution,
                    "predictive_contract": contract["predictive_contract"],
                    "nll_measure_basis": (
                        COHERENT_CENSORED_NLL_MEASURE_BASIS
                        if frozen_predictive_contract
                        in COHERENT_CENSORED_CONTRACTS
                        else ""
                    ),
                    "bridge_config": str(bridge_path),
                    "readout": str(root / readout_name),
                    "pilot_contract": str(args.pilot_contract),
                    "status": "ok",
                }
            )

    return {
        "macro_rows": macro_rows,
        "metric_frames": metric_frames,
        "score_frames": score_frames,
        "validation_frames": validation_frames,
        "interval_frames": interval_frames,
        "manifest_rows": manifest_rows,
        "v21_moment_contexts": v21_moment_contexts,
    }


def _export_worker_count(args) -> int:
    try:
        count = int(getattr(args, "workers", 1))
    except (TypeError, ValueError) as error:
        raise SystemExit("--workers must be an integer of at least 1") from error
    if count < 1:
        raise SystemExit("--workers must be at least 1")
    return count


def _compute_family_task_unit(
    args,
    family: str,
    is_draw: bool,
    dataset: str,
    v21_moment_context: dict[str, object] | None = None,
) -> dict[str, object]:
    ""

    initial_contexts = (
        {dataset: v21_moment_context}
        if v21_moment_context is not None
        else None
    )
    return _collect_export_frames(
        args,
        lanes_override=((family, is_draw),),
        tasks_override=(dataset,),
        initial_v21_moment_contexts=initial_contexts,
    )


def _empty_export_frames() -> dict[str, object]:
    return {
        "macro_rows": [],
        "metric_frames": [],
        "score_frames": [],
        "validation_frames": [],
        "interval_frames": [],
        "manifest_rows": [],
        "v21_moment_contexts": {},
    }


def _merge_export_frames(
    units: list[dict[str, object]],
) -> dict[str, object]:
    merged = _empty_export_frames()
    for unit in units:
        for key in (
            "macro_rows",
            "metric_frames",
            "score_frames",
            "validation_frames",
            "interval_frames",
            "manifest_rows",
        ):
            merged[key].extend(unit[key])
        merged["v21_moment_contexts"].update(unit["v21_moment_contexts"])
    return merged


def _parallel_export_frames(
    args,
    *,
    lanes: list[tuple[str, bool]],
    workers: int,
) -> dict[str, object]:
    ""

    ordered_specs = [
        (family, is_draw, dataset)
        for family, is_draw in lanes
        for dataset in TASKS
    ]
    results: dict[tuple[str, str], dict[str, object]] = {}
    pilot_contract = str(args.pilot_contract)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        if pilot_contract == "joint_rho_gamma_v21_archived_dual":
            moment_specs = [spec for spec in ordered_specs if not spec[1]]
            moment_futures = {
                (family, dataset): executor.submit(
                    _compute_family_task_unit,
                    args,
                    family,
                    is_draw,
                    dataset,
                )
                for family, is_draw, dataset in moment_specs
            }
            moment_contexts: dict[str, dict[str, object]] = {}
            for family, _is_draw, dataset in moment_specs:
                result = moment_futures[(family, dataset)].result()
                results[(family, dataset)] = result
                moment_contexts[dataset] = result["v21_moment_contexts"][dataset]
            draw_specs = [spec for spec in ordered_specs if spec[1]]
            draw_futures = {
                (family, dataset): executor.submit(
                    _compute_family_task_unit,
                    args,
                    family,
                    is_draw,
                    dataset,
                    moment_contexts[dataset],
                )
                for family, is_draw, dataset in draw_specs
            }
            for family, _is_draw, dataset in draw_specs:
                results[(family, dataset)] = draw_futures[
                    (family, dataset)
                ].result()
        else:
            futures = {
                (family, dataset): executor.submit(
                    _compute_family_task_unit,
                    args,
                    family,
                    is_draw,
                    dataset,
                )
                for family, is_draw, dataset in ordered_specs
            }
            for family, _is_draw, dataset in ordered_specs:
                results[(family, dataset)] = futures[(family, dataset)].result()

    return _merge_export_frames(
        [
            results[(family, dataset)]
            for family, _is_draw, dataset in ordered_specs
        ]
    )


def _write_export_frames(args, frames: dict[str, object]) -> dict[str, Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "caster_metrics": out_dir / "caster_metrics.csv",
        "metric_slices": out_dir / "metric_slices.csv",
        "test_bridge_scores": out_dir / "test_bridge_scores.csv",
        "asof_mixture_weight_validation": out_dir / "asof_mixture_weight_validation.csv",
        "forecast_intervals": out_dir / "forecast_intervals.csv",
        "result_export_manifest": out_dir / "result_export_manifest.csv",
    }
    pd.concat(frames["macro_rows"], ignore_index=True).to_csv(
        outputs["caster_metrics"], index=False
    )
    pd.concat(frames["metric_frames"], ignore_index=True).to_csv(
        outputs["metric_slices"], index=False
    )
    pd.concat(frames["score_frames"], ignore_index=True).to_csv(
        outputs["test_bridge_scores"], index=False
    )
    pd.concat(frames["validation_frames"], ignore_index=True).to_csv(
        outputs["asof_mixture_weight_validation"], index=False
    )
    pd.concat(frames["interval_frames"], ignore_index=True)[
        core.FORECAST_INTERVAL_COLUMNS
    ].to_csv(outputs["forecast_intervals"], index=False)
    pd.DataFrame(frames["manifest_rows"]).to_csv(
        outputs["result_export_manifest"], index=False
    )
    return outputs


def export(args) -> dict[str, Path]:
    ""

    workers = _export_worker_count(args)
    draw_roots_enabled = _draw_roots_enabled(args)
    if str(args.pilot_contract) in {
        "gamma_only",
        "joint_rho_gamma",
        "joint_rho_gamma_fullval",
        "joint_rho_gamma_fullval_v2_1",
        "joint_rho_gamma_fullval_v2_2",
    } and draw_roots_enabled:
        raise SystemExit(
            f"{args.pilot_contract} export does not support draw-kernel roots"
        )
    lanes = [("moment_t", False)]
    if draw_roots_enabled:
        lanes.append(("draw_kernel_t", True))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    frames = (
        _collect_export_frames(args)
        if workers == 1
        else _parallel_export_frames(args, lanes=lanes, workers=workers)
    )
    return _write_export_frames(args, frames)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--pilot-contract",
        choices=tuple(PILOT_CONTRACTS),
        default="rho_only",
        help="Freeze/protocol contract expected from every supplied task root.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Family/task export worker processes. The default 1 preserves "
            "the fixed serial execution and output ordering."
        ),
    )
    for task in TASKS:
        parser.add_argument(f"--{task.replace('_', '-')}-root", dest=f"{task}_root", required=True)
        parser.add_argument(
            f"--draw-kernel-{task.replace('_', '-')}-root",
            dest=f"draw_kernel_{task}_root",
            default="",
        )
    args = parser.parse_args()
    for name, path in export(args).items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
