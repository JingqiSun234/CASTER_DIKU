""





from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from collections import OrderedDict
import json
import math
from typing import Callable

import numpy as np
import pandas as pd

from .likelihood import (
    COHERENT_CENSORED_STUDENT_T,
    COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T,
    COHERENT_MEAN_PRESERVING_TRUNCATED_T,
    alternate_ARCHIVE_MOMENT,
    PREDICTIVE_CONTRACTS,
    BridgeConfig,
    bridge_r_key_series,
    calibrate_component_sigma,
    score_archive_rows,
    score_draw_rows,
)
from caster.filter import (
    compute_log_evidence,
    compute_model_uniform_prior,
    evidence_availability_by_model,
    hierarchical_update_from_log_evidence,
    initialize_hierarchical_weights,
    native_forecast_rows,
    posterior_predictive_readout_asof,
    update_outer_weights,
    validate_sleeping_model_archive,
)
from caster.filter.evidence import logsumexp


MOMENT_T = "moment_t"
DRAW_KERNEL_T = "draw_kernel_t"
BRIDGE_FAMILIES = (MOMENT_T, DRAW_KERNEL_T)
FILTER_VARIANTS = ("one_layer", "hierarchical")
JOINT_METRICS = (
    "nll",
    "short_rmse",
    "long_rmse",
    "mae",
    "wis",
    "coverage_penalty",
)
DEFAULT_OBJECTIVE_WEIGHTS = {
    "nll": 0.30,
    "short_rmse": 0.10,
    "long_rmse": 0.10,
    "mae": 0.10,
    "wis": 0.30,
    "coverage_penalty": 0.10,
}
Z50 = 0.6744897501960817
Z90 = 1.6448536269514722


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin(
        {"true", "1", "t", "yes", "y"}
    )


def _exact_log_mixture_weights(values: pd.Series) -> np.ndarray:
    ""

    weights = values.astype(float).to_numpy()
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("mixture weights must be finite and nonnegative")
    result = np.full(weights.shape, -np.inf, dtype=float)
    positive = weights > 0.0
    result[positive] = np.log(weights[positive])
    return result


def coverage_penalty(
    coverage: float,
    *,
    target: float = 0.90,
    tolerance: float = 0.03,
    upper_weight: float = 0.5,
) -> float:
    if not math.isfinite(float(coverage)):
        return float("nan")
    delta = float(tolerance)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("coverage tolerance must be finite and positive")
    value = float(coverage)
    lower_edge = float(target) - delta
    upper_edge = float(target) + delta
                                                                         
                                                                            
                                                
    edge_tolerance = 8.0 * np.finfo(float).eps * max(
        1.0, abs(value), abs(lower_edge), abs(upper_edge)
    )
    lower_gap = lower_edge - value
    upper_gap = value - upper_edge
    lower = (max(0.0, lower_gap) / delta) ** 2 if lower_gap > edge_tolerance else 0.0
    upper = (max(0.0, upper_gap) / delta) ** 2 if upper_gap > edge_tolerance else 0.0
    return float(lower + float(upper_weight) * upper)


def transformed_joint_metrics(
    metrics: dict[str, float],
    *,
    metric_epsilon: float,
    coverage_target: float,
    coverage_tolerance: float,
    coverage_upper_weight: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ("short_rmse", "long_rmse", "mae", "wis"):
        value = float(metrics[name])
        out[name] = (
            float(math.log(value + float(metric_epsilon)))
            if math.isfinite(value) and value >= 0.0
            else float("nan")
        )
    out["nll"] = float(metrics["nll"])
    out["coverage_penalty"] = coverage_penalty(
        float(metrics["coverage_90"]),
        target=coverage_target,
        tolerance=coverage_tolerance,
        upper_weight=coverage_upper_weight,
    )
    return out


@dataclass(frozen=True)
class JointSelectionSettings:
    objective_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_OBJECTIVE_WEIGHTS)
    )
    nu_values: tuple[float, ...] = (5.0, 10.0, float("inf"))
    rho_bounds: tuple[float, float] = (0.05, 1.0)
    gamma_bounds: tuple[float, float] = (0.125, 8.0)
    scale_bound_multiplier: float = 4.0
    min_scale: float = 0.04
    coordinate_passes: int = 2
    refinement_passes: int = 1
    multi_starts: int = 2
    initial_log_step: float = math.log(2.0)
    exact_tie_tolerance: float = 1e-10
    metric_epsilon: float = 1e-8
    robust_scale_floor: float = 1e-3
    robust_relative_floor: float = 0.05
    coverage_target: float = 0.90
    coverage_tolerance: float = 0.03
    coverage_upper_weight: float = 0.5
    seed: int = 0

    def validate(self) -> "JointSelectionSettings":
        if set(self.objective_weights) != set(JOINT_METRICS):
            raise ValueError(
                "objective_weights must contain exactly " + ",".join(JOINT_METRICS)
            )
        weights = np.asarray(list(self.objective_weights.values()), dtype=float)
        if not np.isfinite(weights).all() or (weights < 0.0).any():
            raise ValueError("objective weights must be finite and nonnegative")
        if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("objective weights must sum to one")
        rho_lower, rho_upper = self.rho_bounds
        if not (
            math.isfinite(rho_lower)
            and math.isfinite(rho_upper)
            and 0.0 < rho_lower <= rho_upper
        ):
            raise ValueError("rho_bounds must be finite, positive, and nondecreasing")
        gamma_lower, gamma_upper = self.gamma_bounds
        if not (
            math.isfinite(gamma_lower)
            and math.isfinite(gamma_upper)
            and 0.0 < gamma_lower < gamma_upper
        ):
            raise ValueError("gamma_bounds must be finite, positive, and increasing")
        if self.scale_bound_multiplier <= 1.0 or not math.isfinite(self.scale_bound_multiplier):
            raise ValueError("scale_bound_multiplier must be finite and > 1")
        if self.coordinate_passes < 1 or self.refinement_passes < 0:
            raise ValueError("coordinate passes must be nonnegative and primary passes >= 1")
        if self.multi_starts < 1:
            raise ValueError("multi_starts must be >= 1")
        if not self.nu_values:
            raise ValueError("nu_values must be nonempty")
        if any(math.isnan(float(value)) or float(value) <= 0.0 for value in self.nu_values):
            raise ValueError("nu_values must be positive or infinity")
        return self


@dataclass(frozen=True)
class ParameterState:
    family: str
    scales: dict[str, float]
    gammas: dict[str, float]
    nus: dict[str, float]
    rho: float
    distribution: str = "student_t"
    predictive_contract: str = alternate_ARCHIVE_MOMENT
    truncation_upper_raw_by_component: dict[str, float] = field(default_factory=dict)
    default_truncation_upper_raw: float = float("inf")
    truncation_bound_policy: str = "none"
    truncation_quadrature_order: int = 128
    truncation_zero_mean_epsilon: float = 1e-10
    truncation_support_expansion_multiplier: float | None = 1.25

    def config(self, *, transform: str = "log1p") -> BridgeConfig:
        if self.family not in BRIDGE_FAMILIES:
            raise ValueError(f"unknown bridge family {self.family!r}")
        if self.distribution not in {"student_t", "gaussian"}:
            raise ValueError(
                f"unknown bridge distribution {self.distribution!r}"
            )
        if self.predictive_contract not in PREDICTIVE_CONTRACTS:
            raise ValueError(
                f"unknown predictive contract {self.predictive_contract!r}"
            )
        if (
            self.predictive_contract
            in {
                COHERENT_MEAN_PRESERVING_TRUNCATED_T,
                COHERENT_CENSORED_STUDENT_T,
                COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T,
            }
            and self.distribution != "student_t"
        ):
            raise ValueError(
                "the coherent mean-preserving truncated contract requires Student-t"
            )
        if (
            self.predictive_contract
            in {
                COHERENT_MEAN_PRESERVING_TRUNCATED_T,
                COHERENT_CENSORED_STUDENT_T,
                COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T,
            }
            and (
                self.truncation_quadrature_order < 32
                or self.truncation_quadrature_order % 4
            )
        ):
            raise ValueError(
                "truncation quadrature order must be at least 32 and "
                "divisible by 4"
            )
                                                                             
                                                                              
                                                                            
                                                                             
                                          
        default_nu = 5.0
        nu_values = {float(value) for value in self.nus.values()}
        if len(nu_values) == 1:
            default_nu = next(iter(nu_values))
        return BridgeConfig(
            distribution=self.distribution,
            transform=transform,
            nu=default_nu,
            nu_by_component=dict(self.nus),
            kernel_distribution=self.distribution,
            kernel_nu=default_nu,
            min_scale=1e-3,
            sigma_by_component=(
                dict(self.scales) if self.family == MOMENT_T else {}
            ),
            tau_by_component=(
                dict(self.scales) if self.family == DRAW_KERNEL_T else {}
            ),
            gamma_by_component=(
                dict(self.gammas) if self.family == MOMENT_T else {}
            ),
            default_sigma=float(np.median(list(self.scales.values()))),
            default_tau=float(np.median(list(self.scales.values()))),
            default_gamma=1.0,
            predictive_contract=self.predictive_contract,
            truncation_upper_raw_by_component=dict(
                self.truncation_upper_raw_by_component
            ),
            default_truncation_upper_raw=float(
                self.default_truncation_upper_raw
            ),
            truncation_bound_policy=self.truncation_bound_policy,
            truncation_quadrature_order=int(self.truncation_quadrature_order),
            truncation_zero_mean_epsilon=float(
                self.truncation_zero_mean_epsilon
            ),
            truncation_support_expansion_multiplier=(
                None
                if self.truncation_support_expansion_multiplier is None
                else float(self.truncation_support_expansion_multiplier)
            ),
        )

    def serializable(self) -> dict[str, object]:
        payload = {
            "family": self.family,
            "scales": {key: float(value) for key, value in sorted(self.scales.items())},
            "gammas": {key: float(value) for key, value in sorted(self.gammas.items())},
            "nus": {
                key: ("inf" if math.isinf(float(value)) else float(value))
                for key, value in sorted(self.nus.items())
            },
            "rho": float(self.rho),
        }
        if self.distribution != "student_t":
            payload["distribution"] = self.distribution
        if self.predictive_contract != alternate_ARCHIVE_MOMENT:
            payload["predictive_contract"] = self.predictive_contract
        if self.truncation_upper_raw_by_component:
            payload["truncation_upper_raw_by_component"] = {
                key: float(value)
                for key, value in sorted(
                    self.truncation_upper_raw_by_component.items()
                )
            }
            payload["default_truncation_upper_raw"] = float(
                self.default_truncation_upper_raw
            )
            payload["truncation_bound_policy"] = self.truncation_bound_policy
            payload["truncation_quadrature_order"] = int(
                self.truncation_quadrature_order
            )
            payload["truncation_zero_mean_epsilon"] = float(
                self.truncation_zero_mean_epsilon
            )
            payload["truncation_support_expansion_multiplier"] = (
                self.truncation_support_expansion_multiplier
            )
        return payload


@dataclass
class ReplayArtifacts:
    metrics: dict[str, float]
    metric_validation: dict[str, object]
    replay_validation: dict[str, object]
    posterior_path: pd.DataFrame
    scored_readout: pd.DataFrame
    metric_slices: pd.DataFrame


@dataclass
class JointSelectionOutcome:
    variant: str
    selected_state: ParameterState
    selected_config: BridgeConfig
    selected_metrics: dict[str, float]
    selected_objective: float
    reference_metrics: dict[str, float]
    reference_transformed: dict[str, float]
    normalization_scales: dict[str, float]
    family_best_states: dict[str, ParameterState]
    family_report: pd.DataFrame
    trace: pd.DataFrame
    component_report: pd.DataFrame
    replay_artifacts: ReplayArtifacts


MetricEvaluator = Callable[
    [pd.DataFrame],
    tuple[dict[str, float], dict[str, object], pd.DataFrame],
]


class ExactValidationReplay:
    ""

    def __init__(
        self,
        *,
        validation_ledger: pd.DataFrame,
        metric_ledger: pd.DataFrame,
        archive: pd.DataFrame,
        draws: pd.DataFrame,
        registry: pd.DataFrame,
        dataset_key: str,
        metric_evaluator: MetricEvaluator,
    ) -> None:
        self.validation_ledger = validation_ledger.copy()
        self.metric_ledger = metric_ledger.copy()
        availability_violations = validate_sleeping_model_archive(archive)
        if not availability_violations.empty:
            sample = availability_violations.head(5).to_dict("records")
            raise ValueError(
                "exact validation replay requires a valid native/sleeping-model archive; "
                f"violations={sample}"
            )
        self.archive = native_forecast_rows(
            archive, require_provenance=True
        ).copy()
        if self.archive.empty:
            raise ValueError("exact validation replay has no native forecast rows")
        self.draws = draws.copy()
        self.registry = registry.copy()
        self.dataset_key = str(dataset_key)
        self.metric_evaluator = metric_evaluator
        for frame in (self.validation_ledger, self.metric_ledger):
            frame["forecast_id"] = frame["forecast_id"].astype(str)
            frame["observed_mask"] = _bool_series(frame["observed_mask"])
            frame["release_time"] = pd.to_datetime(
                frame["release_time"], errors="raise"
            )
            frame["forecast_origin"] = pd.to_datetime(
                frame["forecast_origin"], errors="raise"
            )
        validation_fold_column = next(
            (
                column
                for column in ("validation_fold", "fold_id", "fold")
                if column in self.validation_ledger.columns
            ),
            "",
        )
        metric_fold_column = next(
            (
                column
                for column in ("validation_fold", "fold_id", "fold")
                if column in self.metric_ledger.columns
            ),
            "",
        )
        if bool(validation_fold_column) != bool(metric_fold_column):
            raise ValueError(
                "validation replay and metric ledger must declare folds consistently"
            )
        self.fold_column = validation_fold_column
        if self.fold_column:
            self.validation_ledger["__selection_fold"] = self.validation_ledger[
                validation_fold_column
            ].fillna("fold_0").astype(str)
            self.metric_ledger["__selection_fold"] = self.metric_ledger[
                metric_fold_column
            ].fillna("fold_0").astype(str)
            metric_ids_by_fold = self.metric_ledger[
                ["forecast_id", "__selection_fold"]
            ].drop_duplicates()
            if metric_ids_by_fold["forecast_id"].duplicated().any():
                raise ValueError("a metric forecast_id may belong to only one fold")
            validation_folds = set(
                self.validation_ledger["__selection_fold"].astype(str)
            )
            metric_folds = set(self.metric_ledger["__selection_fold"].astype(str))
            if not validation_folds.issubset(metric_folds):
                raise ValueError(
                    "validation replay contains a fold absent from the metric ledger"
                )
        self.archive["forecast_id"] = self.archive["forecast_id"].astype(str)
        self.archive["model_id"] = self.archive["model_id"].astype(str)
        self.registry["model_id"] = self.registry["model_id"].astype(str)
        self.model_ids = self.registry["model_id"].astype(str).tolist()
        if not self.model_ids:
            raise ValueError("exact validation replay requires at least one model")
        metric_ids = set(self.metric_ledger["forecast_id"].astype(str))
        self.metric_archive = self.archive[
            self.archive["forecast_id"].astype(str).isin(metric_ids)
        ].copy()
        if self.metric_archive.empty:
            raise ValueError("metric archive has no formal validation rows")
        if not self.draws.empty:
            self.draws["forecast_id"] = self.draws["forecast_id"].astype(str)
            self.draws["model_id"] = self.draws["model_id"].astype(str)
            native_pairs = self.archive[["forecast_id", "model_id"]].drop_duplicates()
            self.draws = self.draws.merge(
                native_pairs, on=["forecast_id", "model_id"], how="inner"
            )
        self._fold_replays: list[tuple[str, "ExactValidationReplay"]] | None = None
        self._score_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._predictive_component_caches: OrderedDict[
            str, dict[tuple[str, str, str, str], object]
        ] = OrderedDict()

    def _predictive_component_cache_for(
        self,
        config: BridgeConfig,
        family: str,
    ) -> dict[tuple[str, str, str, str], object]:
        ""







        cache_key = json.dumps(
            {"family": family, "config": asdict(config)},
            sort_keys=True,
            separators=(",", ":"),
        )
        if cache_key in self._predictive_component_caches:
            cache = self._predictive_component_caches.pop(cache_key)
            self._predictive_component_caches[cache_key] = cache
            return cache
        cache: dict[tuple[str, str, str, str], object] = {}
        self._predictive_component_caches[cache_key] = cache
                                                                           
                                                                            
                                                                 
        while len(self._predictive_component_caches) > 2:
            self._predictive_component_caches.popitem(last=False)
        return cache

    def _scored_rows(self, config: BridgeConfig, family: str) -> pd.DataFrame:
        score_key = json.dumps(
            {"family": family, "config": asdict(config)},
            sort_keys=True,
            separators=(",", ":"),
        )
        if score_key in self._score_cache:
            scored = self._score_cache.pop(score_key)
            self._score_cache[score_key] = scored
            return scored
        score_ledger = pd.concat(
            [self.validation_ledger, self.metric_ledger], ignore_index=True
        ).drop_duplicates("forecast_id")
        if family == MOMENT_T:
            scored = score_archive_rows(
                score_ledger, self.archive, config
            )
        elif family == DRAW_KERNEL_T:
            if self.draws.empty:
                raise ValueError("draw-kernel selection requires forecast draws")
            scored = score_draw_rows(
                score_ledger, self.draws, config
            )
        else:
            raise ValueError(f"unknown bridge family {family!r}")
        release_meta = score_ledger[
            ["forecast_id", "release_time"]
        ].drop_duplicates("forecast_id")
        scored = scored.merge(release_meta, on="forecast_id", how="left")
        scored["release_time"] = pd.to_datetime(
            scored["release_time"], errors="raise"
        )
        self._score_cache[score_key] = scored
        while len(self._score_cache) > 2:
            self._score_cache.popitem(last=False)
        return scored

    def _replay_posterior(
        self,
        scored: pd.DataFrame,
        *,
        rho: float,
        variant: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        observed = self.validation_ledger["observed_mask"].astype(bool)
        releases = sorted(
            self.validation_ledger.loc[observed, "release_time"].dropna().unique()
        )
        posterior_rows: list[pd.DataFrame] = []
        available_updates = 0
        if variant == "one_layer":
            initial = compute_model_uniform_prior(self.registry).rename(
                columns={"prior_weight": "weight"}
            )[["model_id", "family", "weight"]]
            weights = initial.copy()
            hp = None
        elif variant == "hierarchical":
            hp = initialize_hierarchical_weights(self.registry)
            initial = hp.model_weights[["model_id", "family", "weight"]].copy()
            weights = initial.copy()
        else:
            raise ValueError(f"unknown filtering variant {variant!r}")

        for release_time in releases:
            timestamp = pd.Timestamp(release_time)
            batch = self.validation_ledger[
                self.validation_ledger["release_time"].eq(timestamp)
            ].copy()
            batch_ids = set(batch["forecast_id"].astype(str))
            current = scored[
                scored["release_time"].eq(timestamp)
                & scored["forecast_id"].astype(str).isin(batch_ids)
            ].copy()
            availability = evidence_availability_by_model(
                current, batch, self.model_ids
            )
            available_updates += int(sum(bool(value) for value in availability.values()))
            log_evidence = pd.DataFrame(
                [
                    {
                        "release_time": timestamp,
                        "model_id": str(model_id),
                        "log_evidence": (
                            compute_log_evidence(current, model_id=str(model_id))
                            if availability[str(model_id)]
                            else 0.0
                        ),
                        "evidence_available": availability[str(model_id)],
                    }
                    for model_id in self.model_ids
                ]
            )
            if variant == "one_layer":
                weights = update_outer_weights(weights, log_evidence, rho=float(rho))
                snapshot = weights[["model_id", "family", "weight"]].copy()
                weights = snapshot.copy()
            else:
                assert hp is not None
                hp = hierarchical_update_from_log_evidence(
                    hp.family_weights,
                    hp.inner_weights,
                    log_evidence,
                    rho=float(rho),
                )
                snapshot = hp.model_weights[
                    ["model_id", "family", "weight"]
                ].copy()
            snapshot["release_time"] = timestamp
            snapshot["rho"] = float(rho)
            posterior_rows.append(snapshot)
        posterior = (
            pd.concat(posterior_rows, ignore_index=True)
            if posterior_rows
            else pd.DataFrame(
                columns=["model_id", "family", "weight", "release_time", "rho"]
            )
        )
        replay_validation = {
            "validation_release_count": int(len(releases)),
            "available_model_release_updates": int(available_updates),
            "prior_policy": (
                "uniform_model_prior"
                if variant == "one_layer"
                else "uniform_family_uniform_inner"
            ),
            "posterior_update_score_basis": (
                "likelihood_only_release_normalized_bridge_evidence"
            ),
            "point_metrics_used_in_online_update": False,
            "fixed_share_used": False,
        }
        return posterior, initial, replay_validation

    @staticmethod
    def _cross_weights(
        forecast_ids: pd.Series,
        weights: pd.DataFrame,
    ) -> pd.DataFrame:
        ids = pd.DataFrame({"forecast_id": forecast_ids.astype(str).unique()})
        ids["__join__"] = 1
        work = weights[["model_id", "family", "weight"]].copy()
        work["__join__"] = 1
        return ids.merge(work, on="__join__", how="inner").drop(
            columns="__join__"
        )

    def _asof_weights(
        self,
        readout: pd.DataFrame,
        posterior: pd.DataFrame,
        initial: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        prior_mask = _bool_series(readout["used_prior_snapshot"])
        if prior_mask.any():
            rows.append(
                self._cross_weights(
                    readout.loc[prior_mask, "forecast_id"], initial
                )
            )
        snapshots = pd.to_datetime(
            readout.loc[~prior_mask, "posterior_snapshot_time"], errors="raise"
        )
        for snapshot_time in sorted(snapshots.unique()):
            timestamp = pd.Timestamp(snapshot_time)
            ids = readout.loc[
                ~prior_mask
                & pd.to_datetime(
                    readout["posterior_snapshot_time"], errors="coerce"
                ).eq(timestamp),
                "forecast_id",
            ]
            weights = posterior[
                pd.to_datetime(posterior["release_time"], errors="raise").eq(
                    timestamp
                )
            ]
            rows.append(self._cross_weights(ids, weights))
        if not rows:
            raise ValueError("as-of validation readout produced no model weights")
        all_weights = pd.concat(rows, ignore_index=True)
        native_pairs = self.metric_archive[
            ["forecast_id", "model_id"]
        ].drop_duplicates()
        all_weights = all_weights.merge(
            native_pairs, on=["forecast_id", "model_id"], how="inner"
        )
        mass = all_weights.groupby("forecast_id")["weight"].transform("sum")
        if (mass <= 0.0).any() or not np.isfinite(mass).all():
            raise ValueError("as-of validation weights have nonpositive native mass")
        all_weights["weight"] = all_weights["weight"].astype(float) / mass
        return all_weights

    def _exact_mixture_nll(
        self,
        scored: pd.DataFrame,
        weights: pd.DataFrame,
    ) -> pd.DataFrame:
        metric_ids = set(self.metric_ledger["forecast_id"].astype(str))
        rows = scored[
            scored["forecast_id"].astype(str).isin(metric_ids)
            & _bool_series(scored["observed_mask"])
        ].copy()
        rows["forecast_id"] = rows["forecast_id"].astype(str)
        rows["model_id"] = rows["model_id"].astype(str)
        rows = rows.merge(
            weights[["forecast_id", "model_id", "weight"]],
            on=["forecast_id", "model_id"],
            how="inner",
        )
        if rows.empty:
            raise ValueError("exact validation mixture NLL has no scored rows")
        particle_counts = rows.groupby(
            ["forecast_id", "model_id"]
        )["particle_id"].transform("nunique").astype(float)
        rows["log_term"] = (
            rows["log_score"].astype(float)
            + _exact_log_mixture_weights(rows["weight"])
            - np.log(particle_counts.clip(lower=1.0))
        )
        mixture = (
            rows.groupby("forecast_id", sort=False)["log_term"]
            .agg(lambda values: logsumexp(values.astype(float).to_numpy()))
            .rename("bridge_log_score")
            .reset_index()
        )
        expected = set(
            self.metric_ledger.loc[
                self.metric_ledger["observed_mask"].astype(bool), "forecast_id"
            ].astype(str)
        )
        missing = sorted(expected - set(mixture["forecast_id"].astype(str)))
        if missing:
            raise ValueError(
                "exact validation mixture NLL missing formal forecast_ids: "
                + ",".join(missing[:10])
            )
        mixture["bridge_nll"] = -mixture["bridge_log_score"].astype(float)
        return mixture

    def _evaluate_one_fold(
        self,
        state: ParameterState,
        *,
        variant: str,
        scored: pd.DataFrame | None = None,
    ) -> ReplayArtifacts:
        config = state.config()
        predictive_component_cache = (
            self._predictive_component_cache_for(config, state.family)
            if state.predictive_contract
            in {
                COHERENT_MEAN_PRESERVING_TRUNCATED_T,
                COHERENT_CENSORED_STUDENT_T,
                COHERENT_MEAN_PRESERVING_CENSORED_STUDENT_T,
            }
            else None
        )
        if scored is None:
            scored = self._scored_rows(config, state.family)
        posterior, initial, replay_validation = self._replay_posterior(
            scored, rho=state.rho, variant=variant
        )
        readout = posterior_predictive_readout_asof(
            self.metric_ledger,
            self.metric_archive,
            posterior,
            initial,
            posterior_update_policy="validation_causal_replay",
            release_availability_rule=(
                "date_only_release_time_no_later_than_forecast_origin"
            ),
            bridge_config=config,
            score_source=(
                "draw_kernel"
                if state.family == DRAW_KERNEL_T
                else "archive_moment"
            ),
            draws=self.draws if state.family == DRAW_KERNEL_T else None,
            predictive_component_cache=predictive_component_cache,
        )
        asof_weights = self._asof_weights(readout, posterior, initial)
        mixture_nll = self._exact_mixture_nll(scored, asof_weights)
        readout = readout.merge(
            mixture_nll[["forecast_id", "bridge_log_score", "bridge_nll"]],
            on="forecast_id",
            how="left",
        )
        interval_columns = {
            "lower_50",
            "upper_50",
            "lower_90",
            "upper_90",
        }
        if not interval_columns.issubset(readout.columns):
            if state.predictive_contract != alternate_ARCHIVE_MOMENT:
                missing = sorted(interval_columns - set(readout.columns))
                raise ValueError(
                    "predictive-contract readout omitted required validation "
                    f"intervals: {missing}"
                )
            sigma = np.sqrt(
                np.maximum(
                    pd.to_numeric(
                        readout["predictive_var"], errors="raise"
                    ).to_numpy(dtype=float),
                    0.0,
                )
            )
            mean = pd.to_numeric(
                readout["predictive_mean"], errors="raise"
            ).to_numpy(dtype=float)
            readout["lower_50"] = np.maximum(0.0, mean - Z50 * sigma)
            readout["upper_50"] = np.maximum(0.0, mean + Z50 * sigma)
            readout["lower_90"] = np.maximum(0.0, mean - Z90 * sigma)
            readout["upper_90"] = np.maximum(0.0, mean + Z90 * sigma)
        else:
            interval_values = readout[sorted(interval_columns)].apply(
                pd.to_numeric, errors="raise"
            )
            if not np.isfinite(interval_values.to_numpy(dtype=float)).all():
                raise ValueError("predictive-contract validation intervals are non-finite")
            if (
                (readout["lower_50"] > readout["upper_50"]).any()
                or (readout["lower_90"] > readout["upper_90"]).any()
            ):
                raise ValueError("predictive-contract validation intervals are inverted")
        readout["dataset"] = self.dataset_key
        readout["method"] = "caster_joint_validation"
        readout["method_group"] = "caster"
        readout["bridge_family"] = state.family
        readout["filter_variant"] = variant
        readout["rho"] = float(state.rho)
        readout["predictive_contract"] = state.predictive_contract
        readout["nll_score_basis"] = (
            "exact_asof_posterior_draw_kernel_mixture"
            if state.family == DRAW_KERNEL_T
            else "exact_asof_posterior_mixture_bridge"
        )

        origin = pd.to_datetime(readout["forecast_origin"], errors="raise")
        snapshot = pd.to_datetime(
            readout["posterior_snapshot_time"], errors="coerce"
        )
        used_prior = _bool_series(readout["used_prior_snapshot"])
        future = snapshot.notna() & (snapshot > origin) & ~used_prior
        self_release = pd.to_datetime(readout["release_time"], errors="coerce")
        self_update = snapshot.notna() & self_release.notna() & (
            self_release <= snapshot
        )
        if future.any() or self_update.any():
            raise ValueError(
                "validation causal replay violated as-of/self-target constraints"
            )
        replay_validation.update(
            {
                "metric_forecast_rows": int(len(readout)),
                "metric_observed_rows": int(
                    _bool_series(readout["observed_mask"]).sum()
                ),
                "readout_rows_using_prior_snapshot": int(used_prior.sum()),
                "readout_rows_future_snapshot_violation": int(future.sum()),
                "readout_rows_self_target_update_violation": int(
                    self_update.sum()
                ),
                "posterior_snapshot_policy": (
                    "latest_release_time_lte_forecast_origin_else_W0"
                ),
                "predictive_contract": state.predictive_contract,
                "validation_interval_source": (
                    "alternate_gaussian_raw_moment"
                    if state.predictive_contract == alternate_ARCHIVE_MOMENT
                    else "predictive_contract_mixture_quantiles"
                ),
            }
        )
        metrics, metric_validation, metric_slices = self.metric_evaluator(readout)
        return ReplayArtifacts(
            metrics=metrics,
            metric_validation=metric_validation,
            replay_validation=replay_validation,
            posterior_path=posterior,
            scored_readout=readout,
            metric_slices=metric_slices,
        )

    def _cached_fold_replays(self) -> list[tuple[str, "ExactValidationReplay"]]:
        if self._fold_replays is not None:
            return self._fold_replays
        replays: list[tuple[str, ExactValidationReplay]] = []
        folds = self.metric_ledger["__selection_fold"].drop_duplicates().tolist()
        for fold in folds:
            fold_name = str(fold)
            validation = self.validation_ledger[
                self.validation_ledger["__selection_fold"].eq(fold_name)
            ].drop(columns="__selection_fold")
            metric = self.metric_ledger[
                self.metric_ledger["__selection_fold"].eq(fold_name)
            ].drop(columns="__selection_fold")
            forecast_ids = set(validation["forecast_id"].astype(str)) | set(
                metric["forecast_id"].astype(str)
            )
            archive = self.archive[
                self.archive["forecast_id"].astype(str).isin(forecast_ids)
            ].copy()
            draws = (
                self.draws[
                    self.draws["forecast_id"].astype(str).isin(forecast_ids)
                ].copy()
                if "forecast_id" in self.draws.columns
                else self.draws.copy()
            )
            replays.append(
                (
                    fold_name,
                    ExactValidationReplay(
                        validation_ledger=validation,
                        metric_ledger=metric,
                        archive=archive,
                        draws=draws,
                        registry=self.registry,
                        dataset_key=self.dataset_key,
                        metric_evaluator=self.metric_evaluator,
                    ),
                )
            )
        self._fold_replays = replays
        return replays

    def evaluate(
        self,
        state: ParameterState,
        *,
        variant: str,
    ) -> ReplayArtifacts:
        ""

        if not self.fold_column:
            return self._evaluate_one_fold(state, variant=variant)
        fold_replays = self._cached_fold_replays()
        if len(fold_replays) == 1:
            return self._evaluate_one_fold(state, variant=variant)

        config = state.config()
        scored_all = self._scored_rows(config, state.family)
        fold_artifacts: list[tuple[str, ReplayArtifacts]] = []
        for fold_name, replay in fold_replays:
            forecast_ids = set(replay.validation_ledger["forecast_id"].astype(str)) | set(
                replay.metric_ledger["forecast_id"].astype(str)
            )
            fold_scored = scored_all[
                scored_all["forecast_id"].astype(str).isin(forecast_ids)
            ].copy()
            artifact = replay._evaluate_one_fold(
                state, variant=variant, scored=fold_scored
            )
            artifact.posterior_path["validation_fold"] = fold_name
            artifact.scored_readout["validation_fold"] = fold_name
            fold_artifacts.append((fold_name, artifact))

        posterior_frames = [
            artifact.posterior_path
            for _, artifact in fold_artifacts
            if not artifact.posterior_path.empty
        ]
        posterior = (
            pd.concat(posterior_frames, ignore_index=True)
            if posterior_frames
            else fold_artifacts[0][1].posterior_path.copy()
        )
        readout = pd.concat(
            [artifact.scored_readout for _, artifact in fold_artifacts],
            ignore_index=True,
        )
        metrics, metric_validation, metric_slices = self.metric_evaluator(readout)
        first_validation = dict(fold_artifacts[0][1].replay_validation)
        summed_keys = (
            "validation_release_count",
            "available_model_release_updates",
            "metric_forecast_rows",
            "metric_observed_rows",
            "readout_rows_using_prior_snapshot",
            "readout_rows_future_snapshot_violation",
            "readout_rows_self_target_update_violation",
        )
        for key in summed_keys:
            first_validation[key] = int(
                sum(int(artifact.replay_validation.get(key, 0)) for _, artifact in fold_artifacts)
            )
        first_validation.update(
            {
                "validation_fold_count": int(len(fold_artifacts)),
                "validation_fold_replay_policy": "independent_same_W0_per_fold",
                "validation_folds": [name for name, _ in fold_artifacts],
            }
        )
        return ReplayArtifacts(
            metrics=metrics,
            metric_validation=metric_validation,
            replay_validation=first_validation,
            posterior_path=posterior,
            scored_readout=readout,
            metric_slices=metric_slices,
        )


class JointParameterSelector:
    def __init__(
        self,
        replay: ExactValidationReplay,
        *,
        settings: JointSelectionSettings,
        transform: str = "log1p",
    ) -> None:
        self.replay = replay
        self.settings = settings.validate()
        self.transform = str(transform)
        self.anchors = calibrate_component_sigma(
            replay.metric_ledger,
            replay.metric_archive,
            transform=self.transform,
            min_sigma=self.settings.min_scale,
        )
        if not self.anchors:
            raise ValueError("joint selector could not initialize component scales")
        self.keys = sorted(self.anchors)
        self._cache: dict[str, tuple[dict[str, float], dict[str, object]]] = {}
        self._trace_rows: list[dict[str, object]] = []
        self._trace_id = 0

    def _cache_key(self, state: ParameterState, variant: str) -> str:
        return json.dumps(
            {"variant": variant, **state.serializable()},
            sort_keys=True,
            separators=(",", ":"),
        )

    def _raw_evaluate(
        self,
        state: ParameterState,
        *,
        variant: str,
        stage: str,
    ) -> tuple[dict[str, float], dict[str, object]]:
        key = self._cache_key(state, variant)
        cache_hit = key in self._cache
        if cache_hit:
            metrics, validation = self._cache[key]
        else:
            artifacts = self.replay.evaluate(state, variant=variant)
            metrics = dict(artifacts.metrics)
            validation = {**artifacts.metric_validation, **artifacts.replay_validation}
            self._cache[key] = (metrics, validation)
        self._trace_id += 1
        self._trace_rows.append(
            {
                "trace_id": self._trace_id,
                "stage": stage,
                "variant": variant,
                "bridge_family": state.family,
                "rho": float(state.rho),
                "parameters_json": json.dumps(
                    state.serializable(), sort_keys=True, separators=(",", ":")
                ),
                "cache_hit": bool(cache_hit),
                **{name: float(value) for name, value in metrics.items()},
                **validation,
            }
        )
        return metrics, validation

    def _state(
        self,
        family: str,
        *,
        scale_factor: float = 1.0,
        gamma: float = 1.0,
        nu: float = 10.0,
        rho: float = 0.5,
    ) -> ParameterState:
        return ParameterState(
            family=family,
            scales={
                key: max(
                    self.settings.min_scale,
                    float(self.anchors[key]) * float(scale_factor),
                )
                for key in self.keys
            },
            gammas=(
                {key: float(gamma) for key in self.keys}
                if family == MOMENT_T
                else {}
            ),
            nus={key: float(nu) for key in self.keys},
            rho=float(
                min(max(float(rho), self.settings.rho_bounds[0]), self.settings.rho_bounds[1])
            ),
        )

    def _pilot_states(self) -> list[ParameterState]:
        nu_reference = (
            10.0
            if any(float(value) == 10.0 for value in self.settings.nu_values)
            else float(self.settings.nu_values[0])
        )
        low_rho, high_rho = self.settings.rho_bounds
        states: list[ParameterState] = []
        for family in BRIDGE_FAMILIES:
            states.extend(
                [
                    self._state(family, nu=nu_reference, rho=0.5),
                    self._state(family, nu=nu_reference, rho=low_rho),
                    self._state(family, nu=nu_reference, rho=high_rho),
                    self._state(
                        family, scale_factor=0.5, nu=nu_reference, rho=0.5
                    ),
                    self._state(
                        family, scale_factor=2.0, nu=nu_reference, rho=0.5
                    ),
                ]
            )
            if family == MOMENT_T:
                states.extend(
                    [
                        self._state(family, gamma=0.5, nu=nu_reference, rho=0.5),
                        self._state(family, gamma=2.0, nu=nu_reference, rho=0.5),
                    ]
                )
            for nu in self.settings.nu_values:
                states.append(self._state(family, nu=float(nu), rho=0.5))
        unique: dict[str, ParameterState] = {}
        for state in states:
            unique[json.dumps(state.serializable(), sort_keys=True)] = state
        return list(unique.values())

    def _normalization(
        self,
        pilot_metrics: list[dict[str, float]],
        reference_metrics: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        reference = transformed_joint_metrics(
            reference_metrics,
            metric_epsilon=self.settings.metric_epsilon,
            coverage_target=self.settings.coverage_target,
            coverage_tolerance=self.settings.coverage_tolerance,
            coverage_upper_weight=self.settings.coverage_upper_weight,
        )
        transformed = [
            transformed_joint_metrics(
                metrics,
                metric_epsilon=self.settings.metric_epsilon,
                coverage_target=self.settings.coverage_target,
                coverage_tolerance=self.settings.coverage_tolerance,
                coverage_upper_weight=self.settings.coverage_upper_weight,
            )
            for metrics in pilot_metrics
        ]
        scales: dict[str, float] = {}
        for name in JOINT_METRICS:
            values = np.asarray(
                [row[name] for row in transformed if math.isfinite(row[name])],
                dtype=float,
            )
            if values.size:
                median = float(np.median(values))
                robust = float(1.4826 * np.median(np.abs(values - median)))
            else:
                robust = 0.0
            relative_floor = self.settings.robust_relative_floor * max(
                abs(float(reference[name])), self.settings.robust_scale_floor
            )
            scales[name] = float(
                max(
                    robust,
                    relative_floor,
                    self.settings.robust_scale_floor,
                )
            )
        return reference, scales

    def _objective(
        self,
        metrics: dict[str, float],
        reference: dict[str, float],
        scales: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        transformed = transformed_joint_metrics(
            metrics,
            metric_epsilon=self.settings.metric_epsilon,
            coverage_target=self.settings.coverage_target,
            coverage_tolerance=self.settings.coverage_tolerance,
            coverage_upper_weight=self.settings.coverage_upper_weight,
        )
        z = {
            name: (float(transformed[name]) - float(reference[name]))
            / float(scales[name])
            for name in JOINT_METRICS
        }
        if not all(math.isfinite(value) for value in z.values()):
            return float("inf"), z
        risk = sum(
            float(self.settings.objective_weights[name]) * float(z[name])
            for name in JOINT_METRICS
        )
        return float(risk), z

    def _evaluate_objective(
        self,
        state: ParameterState,
        *,
        variant: str,
        stage: str,
        reference: dict[str, float],
        scales: dict[str, float],
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        metrics, _ = self._raw_evaluate(state, variant=variant, stage=stage)
        objective, z = self._objective(metrics, reference, scales)
        self._trace_rows[-1].update(
            {
                "joint_risk": float(objective),
                **{f"z_{name}": float(value) for name, value in z.items()},
            }
        )
        return objective, metrics, z

    def _better(
        self,
        candidate_objective: float,
        candidate: ParameterState,
        incumbent_objective: float,
        incumbent: ParameterState,
    ) -> bool:
        tol = self.settings.exact_tie_tolerance
        if candidate_objective < incumbent_objective - tol:
            return True
        if abs(candidate_objective - incumbent_objective) <= tol:
            if candidate.rho < incumbent.rho - tol:
                return True
            if abs(candidate.rho - incumbent.rho) <= tol:
                return BRIDGE_FAMILIES.index(candidate.family) < BRIDGE_FAMILIES.index(
                    incumbent.family
                )
        return False

    def _coordinates(self, state: ParameterState) -> list[tuple[str, str]]:
        coordinates = [("scale", key) for key in self.keys]
        if state.family == MOMENT_T:
            coordinates.extend(("gamma", key) for key in self.keys)
        coordinates.append(("rho", "rho"))
        return coordinates

    def _coordinate_bounds(
        self, coordinate: tuple[str, str]
    ) -> tuple[float, float]:
        kind, key = coordinate
        if kind == "scale":
            anchor = float(self.anchors[key])
            multiplier = float(self.settings.scale_bound_multiplier)
            lower = max(float(self.settings.min_scale), anchor / multiplier)
            upper = max(lower * (1.0 + 1e-6), anchor * multiplier)
            return math.log(lower), math.log(upper)
        if kind == "gamma":
            return tuple(math.log(value) for value in self.settings.gamma_bounds)                              
        return tuple(math.log(value) for value in self.settings.rho_bounds)                              

    @staticmethod
    def _coordinate_value(state: ParameterState, coordinate: tuple[str, str]) -> float:
        kind, key = coordinate
        if kind == "scale":
            return math.log(float(state.scales[key]))
        if kind == "gamma":
            return math.log(float(state.gammas[key]))
        return math.log(float(state.rho))

    @staticmethod
    def _with_coordinate(
        state: ParameterState,
        coordinate: tuple[str, str],
        log_value: float,
    ) -> ParameterState:
        kind, key = coordinate
        value = float(math.exp(log_value))
        if kind == "scale":
            values = dict(state.scales)
            values[key] = value
            return replace(state, scales=values)
        if kind == "gamma":
            values = dict(state.gammas)
            values[key] = value
            return replace(state, gammas=values)
        return replace(state, rho=value)

    def _coordinate_search(
        self,
        start: ParameterState,
        *,
        variant: str,
        reference: dict[str, float],
        scales: dict[str, float],
        passes: int,
        stage_prefix: str,
    ) -> tuple[ParameterState, float, dict[str, float]]:
        current = start
        current_objective, current_metrics, _ = self._evaluate_objective(
            current,
            variant=variant,
            stage=f"{stage_prefix}_start",
            reference=reference,
            scales=scales,
        )
        step = float(self.settings.initial_log_step)
        for pass_index in range(int(passes)):
            for coordinate in self._coordinates(current):
                base = self._coordinate_value(current, coordinate)
                lower, upper = self._coordinate_bounds(coordinate)
                candidates: list[tuple[ParameterState, float, dict[str, float]]] = []
                for direction in (-1.0, 1.0):
                    trial_value = min(max(base + direction * step, lower), upper)
                    if abs(trial_value - base) <= 1e-14:
                        continue
                    trial = self._with_coordinate(current, coordinate, trial_value)
                    objective, metrics, _ = self._evaluate_objective(
                        trial,
                        variant=variant,
                        stage=(
                            f"{stage_prefix}_pass{pass_index + 1}_"
                            f"{coordinate[0]}_{coordinate[1]}"
                        ),
                        reference=reference,
                        scales=scales,
                    )
                    candidates.append((trial, objective, metrics))
                for trial, objective, metrics in candidates:
                    if self._better(
                        objective, trial, current_objective, current
                    ):
                        current = trial
                        current_objective = objective
                        current_metrics = metrics
            step *= 0.5
        return current, float(current_objective), current_metrics

    def _refine_nu(
        self,
        start: ParameterState,
        start_objective: float,
        start_metrics: dict[str, float],
        *,
        variant: str,
        reference: dict[str, float],
        scales: dict[str, float],
    ) -> tuple[ParameterState, float, dict[str, float]]:
        current = start
        current_objective = float(start_objective)
        current_metrics = start_metrics
        for key in self.keys:
            for nu in self.settings.nu_values:
                if float(nu) == float(current.nus[key]):
                    continue
                nus = dict(current.nus)
                nus[key] = float(nu)
                trial = replace(current, nus=nus)
                objective, metrics, _ = self._evaluate_objective(
                    trial,
                    variant=variant,
                    stage=f"outer_discrete_nu_{key}",
                    reference=reference,
                    scales=scales,
                )
                if self._better(objective, trial, current_objective, current):
                    current = trial
                    current_objective = objective
                    current_metrics = metrics
        return current, current_objective, current_metrics

    def select(self, *, variant: str) -> JointSelectionOutcome:
        if variant not in FILTER_VARIANTS:
            raise ValueError(f"unknown filtering variant {variant!r}")
        self._trace_rows = []
        self._trace_id = 0
        pilots = self._pilot_states()
        pilot_metrics = [
            self._raw_evaluate(state, variant=variant, stage="pilot")[0]
            for state in pilots
        ]
        reference_state = self._state(MOMENT_T, nu=10.0, rho=0.5)
        reference_metrics, _ = self._raw_evaluate(
            reference_state, variant=variant, stage="reference"
        )
        reference, normalization_scales = self._normalization(
            pilot_metrics, reference_metrics
        )

        scored_pilots: list[tuple[ParameterState, float, dict[str, float]]] = []
        for state, metrics in zip(pilots, pilot_metrics):
            objective, z = self._objective(metrics, reference, normalization_scales)
            scored_pilots.append((state, objective, metrics))
            self._trace_id += 1
            self._trace_rows.append(
                {
                    "trace_id": self._trace_id,
                    "stage": "pilot_rescored_after_frozen_normalization",
                    "variant": variant,
                    "bridge_family": state.family,
                    "rho": float(state.rho),
                    "parameters_json": json.dumps(
                        state.serializable(), sort_keys=True, separators=(",", ":")
                    ),
                    "cache_hit": True,
                    **metrics,
                    "joint_risk": float(objective),
                    **{f"z_{name}": float(value) for name, value in z.items()},
                }
            )

        family_best: dict[str, ParameterState] = {}
        family_objective: dict[str, float] = {}
        family_metrics: dict[str, dict[str, float]] = {}
        for family in BRIDGE_FAMILIES:
            candidates = [item for item in scored_pilots if item[0].family == family]
            start_state, start_objective, start_metrics = min(
                candidates, key=lambda item: (item[1], item[0].rho)
            )
            starts = [start_state]
            if self.settings.multi_starts > 1:
                midpoint_rho = math.sqrt(
                    self.settings.rho_bounds[0] * self.settings.rho_bounds[1]
                )
                second = self._state(
                    family,
                    scale_factor=math.sqrt(2.0),
                    gamma=1.0,
                    nu=float(next(iter(start_state.nus.values()))),
                    rho=midpoint_rho,
                )
                starts.append(second)
            while len(starts) < self.settings.multi_starts:
                factor = 2.0 ** (
                    (len(starts) - 1) / max(self.settings.multi_starts - 1, 1) - 0.5
                )
                starts.append(
                    self._state(
                        family,
                        scale_factor=factor,
                        nu=float(next(iter(start_state.nus.values()))),
                        rho=start_state.rho,
                    )
                )

            best_state = start_state
            best_objective = start_objective
            best_metrics = start_metrics
            for start_index, start in enumerate(starts):
                state, objective, metrics = self._coordinate_search(
                    start,
                    variant=variant,
                    reference=reference,
                    scales=normalization_scales,
                    passes=self.settings.coordinate_passes,
                    stage_prefix=f"continuous_start{start_index + 1}",
                )
                if self._better(objective, state, best_objective, best_state):
                    best_state, best_objective, best_metrics = state, objective, metrics

            best_state, best_objective, best_metrics = self._refine_nu(
                best_state,
                best_objective,
                best_metrics,
                variant=variant,
                reference=reference,
                scales=normalization_scales,
            )
            if self.settings.refinement_passes:
                best_state, best_objective, best_metrics = self._coordinate_search(
                    best_state,
                    variant=variant,
                    reference=reference,
                    scales=normalization_scales,
                    passes=self.settings.refinement_passes,
                    stage_prefix="continuous_after_nu_refinement",
                )
            family_best[family] = best_state
            family_objective[family] = float(best_objective)
            family_metrics[family] = best_metrics

        selected_family = min(
            BRIDGE_FAMILIES,
            key=lambda family: (
                family_objective[family],
                family_best[family].rho,
                BRIDGE_FAMILIES.index(family),
            ),
        )
        selected_state = family_best[selected_family]
        selected_objective = family_objective[selected_family]
        selected_metrics = family_metrics[selected_family]
        artifacts = self.replay.evaluate(selected_state, variant=variant)

        family_rows = []
        for family in BRIDGE_FAMILIES:
            state = family_best[family]
            family_rows.append(
                {
                    "rho_selection_variant": variant,
                    "bridge_family": family,
                    "distribution": "student_t",
                    "kernel_distribution": (
                        "student_t" if family == DRAW_KERNEL_T else ""
                    ),
                    "rho": float(state.rho),
                    "joint_risk": float(family_objective[family]),
                    "selected": family == selected_family,
                    "parameters_json": json.dumps(
                        state.serializable(), sort_keys=True, separators=(",", ":")
                    ),
                    **family_metrics[family],
                }
            )
        family_report = pd.DataFrame(family_rows)
        component_report = pd.DataFrame(
            [
                {
                    "rho_selection_variant": variant,
                    "bridge_family": selected_family,
                    "bridge_r_key": key,
                    "bridge_parameter_scope": "component_horizon",
                    "selected_sigma": (
                        float(selected_state.scales[key])
                        if selected_family == MOMENT_T
                        else float("nan")
                    ),
                    "selected_tau": (
                        float(selected_state.scales[key])
                        if selected_family == DRAW_KERNEL_T
                        else float("nan")
                    ),
                    "selected_gamma": (
                        float(selected_state.gammas[key])
                        if selected_family == MOMENT_T
                        else float("nan")
                    ),
                    "selected_nu": float(selected_state.nus[key]),
                    "selected_rho": float(selected_state.rho),
                    "sigma_selection_policy": (
                        "direct_continuous_log_scale_exact_joint_risk"
                        if selected_family == MOMENT_T
                        else "inactive_for_draw_kernel_family"
                    ),
                    "tau_selection_policy": (
                        "direct_continuous_log_scale_exact_joint_risk"
                        if selected_family == DRAW_KERNEL_T
                        else "inactive_for_moment_family"
                    ),
                    "gamma_selection_policy": (
                        "direct_continuous_log_scale_exact_joint_risk"
                        if selected_family == MOMENT_T
                        else "inactive_for_draw_kernel_family"
                    ),
                    "nu_selection_policy": "outer_discrete_exact_joint_risk",
                }
                for key in self.keys
            ]
        )
        trace = pd.DataFrame(self._trace_rows)
        if not trace.empty:
            trace["selected"] = False
            selected_json = json.dumps(
                selected_state.serializable(), sort_keys=True, separators=(",", ":")
            )
            matches = (
                trace["parameters_json"].astype(str).eq(selected_json)
                & trace["variant"].astype(str).eq(variant)
            )
            if matches.any():
                trace.loc[trace.index[matches][-1], "selected"] = True
        return JointSelectionOutcome(
            variant=variant,
            selected_state=selected_state,
            selected_config=selected_state.config(transform=self.transform),
            selected_metrics=selected_metrics,
            selected_objective=float(selected_objective),
            reference_metrics=reference_metrics,
            reference_transformed=reference,
            normalization_scales=normalization_scales,
            family_best_states=family_best,
            family_report=family_report,
            trace=trace,
            component_report=component_report,
            replay_artifacts=artifacts,
        )


def settings_payload(settings: JointSelectionSettings) -> dict[str, object]:
    payload = asdict(settings)
    payload["nu_values"] = [
        "inf" if math.isinf(float(value)) else float(value)
        for value in settings.nu_values
    ]
    return payload
