#!/usr/bin/env python3
""




from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


DATASET_KEY = "benchmark_b"
BENCHMARK_A_KEY = "benchmark_a"
CREATED_AT = "2026-05-20T00:00:00Z"
WINDOW_BEFORE = 4
WINDOW_AFTER = 8
VARIANT_SHARE_THRESHOLD = 0.40
REVISION_POSITIVE_QUANTILE = 0.90
BENCHMARK_A_WAVE_7D_TOTAL_THRESHOLD = 500.0
SPLIT_MARKER_RE = re.compile(
    r"(?:validation midpoint|test start|event_ledger_split|\bvalidation\b|\btest\b|\bsplit\b)",
    re.IGNORECASE,
)
EVENT_FACING_COLUMNS = ["event_id", "event_type", "label", "selection_rule"]

COMPONENTS = {
    "covid_adm_per100k": {
        "pathogen": "covid19",
        "label": "COVID-19 admissions per 100k",
        "preliminary_col": "preliminary_covid_adm_per100k",
        "finalized_col": "finalized_covid_adm_per100k",
    },
    "flu_adm_per100k": {
        "pathogen": "influenza",
        "label": "Influenza admissions per 100k",
        "preliminary_col": "preliminary_flu_adm_per100k",
        "finalized_col": "finalized_flu_adm_per100k",
    },
}

NO_LEAKAGE_FLAGS = {
    "posterior_used": False,
    "forecast_performance_used": False,
    "manual_visual_selection": False,
    "event_aligned_posterior_used": False,
    "CASTER_success_used": False,
    "test_metric_used": False,
    "posterior_movement_used": False,
    "not_used_for_training_or_metric_tuning": True,
    "use_for_posterior_diagnostic_only": True,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_gold_root() -> Path:
    return repo_root() / "data/benchmark_b/raw_all/gold"


def default_benchmark_a_panel() -> Path:
    return repo_root() / "data/benchmark_a/curated_full_v3_direct_rollout7/daily_panel.csv"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return text or "unknown"


def read_parquet_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=list(columns))
    except Exception:
        df = pd.read_parquet(path)
        return df[[c for c in columns if c in df.columns]].copy()


def event_row(
    *,
    unit: str,
    pathogen: str,
    component: str,
    event_time: object,
    event_type: str,
    label: str,
    selection_rule: str,
    source_path: Path,
    threshold: str,
    window_weeks_before: int,
    window_weeks_after: int,
    unit_scope: str,
    applies_to_all_units: bool,
    expansion_required_for_R3: bool,
    event_source_kind: str,
    event_role: str,
    not_model_input: bool,
) -> dict[str, object]:
    return {
        "event_id": "",
        "dataset_key": DATASET_KEY,
        "unit": unit,
        "pathogen": pathogen,
        "component": component,
        "event_time": pd.to_datetime(event_time).strftime("%Y-%m-%d"),
        "event_type": event_type,
        "label": label,
        "window_weeks_before": int(window_weeks_before),
        "window_weeks_after": int(window_weeks_after),
        "selection_rule": selection_rule,
        "source_path": str(source_path),
        "threshold": threshold,
        "created_at": CREATED_AT,
        "unit_scope": unit_scope,
        "applies_to_all_units": bool(applies_to_all_units),
        "expansion_required_for_R3": bool(expansion_required_for_R3),
        "event_source_kind": event_source_kind,
        "event_role": event_role,
        "not_model_input": bool(not_model_input),
    }


def build_winter_onset_events(gold_root: Path, before: int, after: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    source = gold_root / "panel_targets_long.parquet"
    if not source.exists():
        return [], {"source_path": str(source), "status": "missing"}
    columns = ["week_end", "target_name", "pathogen", "is_primary_benchmark_target"]
    df = read_parquet_columns(source, columns)
    required = set(columns)
    if not required <= set(df.columns):
        return [], {"source_path": str(source), "status": "missing_columns", "missing": sorted(required - set(df.columns))}
    work = df[df["target_name"].isin(COMPONENTS.keys())].copy()
    work = work[work["is_primary_benchmark_target"].astype(bool)]
    work["week_end"] = pd.to_datetime(work["week_end"], errors="coerce")
    work = work.dropna(subset=["week_end"])
    events: list[dict[str, object]] = []
    for component, g_component in work.groupby("target_name", dropna=False):
        for season_year, g_year in g_component[g_component["week_end"].dt.month.eq(10)].groupby(g_component["week_end"].dt.year):
            event_time = g_year["week_end"].min()
            spec = COMPONENTS[str(component)]
            events.append(
                event_row(
                    unit="all_units",
                    pathogen=spec["pathogen"],
                    component=str(component),
                    event_time=event_time,
                    event_type="winter_onset",
                    label=f"Winter onset {int(season_year)}: {spec['label']}",
                    selection_rule="calendar first October available week by component and season",
                    source_path=source,
                    threshold="calendar_month=10",
                    window_weeks_before=before,
                    window_weeks_after=after,
                    unit_scope="all_units",
                    applies_to_all_units=True,
                    expansion_required_for_R3=True,
                    event_source_kind="calendar_availability",
                    event_role="predeclared_calendar_shift_event",
                    not_model_input=True,
                )
            )
    meta = {
        "source_path": str(source),
        "status": "ok",
        "selection_rule": "first available October target week per season/component",
        "uses_target_value_for_ranking": False,
        "read_columns": columns,
        "count": len(events),
    }
    return events, meta


def build_variant_turnover_events(gold_root: Path, before: int, after: int, share_threshold: float) -> tuple[list[dict[str, object]], dict[str, object]]:
    source = gold_root / "aux_variants_summary.parquet"
    if not source.exists():
        return [], {"source_path": str(source), "status": "missing"}
    columns = ["usa_or_hhsregion", "week_ending", "modeltype", "time_interval", "top_variant", "max_share"]
    df = read_parquet_columns(source, columns)
    required = set(columns)
    if not required <= set(df.columns):
        return [], {"source_path": str(source), "status": "missing_columns", "missing": sorted(required - set(df.columns))}
    work = df[
        df["usa_or_hhsregion"].astype(str).eq("USA")
        & df["modeltype"].astype(str).eq("smoothed")
        & df["time_interval"].astype(str).eq("biweekly")
    ].copy()
    work["week_ending"] = pd.to_datetime(work["week_ending"], errors="coerce")
    work["max_share"] = pd.to_numeric(work["max_share"], errors="coerce")
    work = work.dropna(subset=["week_ending", "top_variant", "max_share"]).sort_values("week_ending")
    work["previous_top_variant"] = work["top_variant"].shift()
    selected = work[
        work["previous_top_variant"].notna()
        & ~work["top_variant"].astype(str).eq(work["previous_top_variant"].astype(str))
        & work["max_share"].ge(share_threshold)
    ].copy()
    events: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        new_variant = str(row["top_variant"])
        previous = str(row["previous_top_variant"])
        events.append(
            event_row(
                unit="all_units",
                pathogen="covid19",
                component="covid_adm_per100k",
                event_time=row["week_ending"],
                event_type="variant_turnover",
                label=f"Variant turnover to {new_variant} from {previous}",
                selection_rule="external surveillance top variant change with new top share threshold",
                source_path=source,
                threshold=f"new_top_share>={share_threshold:.2f}",
                window_weeks_before=before,
                window_weeks_after=after,
                unit_scope="all_units",
                applies_to_all_units=True,
                expansion_required_for_R3=True,
                event_source_kind="external_variant_surveillance",
                event_role="predeclared_external_shift_event",
                not_model_input=True,
            )
        )
    meta = {
        "source_path": str(source),
        "status": "ok",
        "selection_rule": "USA smoothed biweekly top-variant change, fixed new top share threshold",
        "variant_share_threshold": float(share_threshold),
        "count": len(events),
    }
    if not events:
        meta["limitation"] = "variant source did not yield a qualifying turnover under the fixed threshold; no variant event fabricated"
    return events, meta


def build_revision_spike_events(
    gold_root: Path,
    before: int,
    after: int,
    positive_quantile: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source = gold_root / "aux_preliminary_vs_finalized_paired.parquet"
    if not source.exists():
        return [], {"source_path": str(source), "status": "missing"}
    columns = ["jurisdiction", "week_end"]
    for spec in COMPONENTS.values():
        columns.extend([spec["preliminary_col"], spec["finalized_col"]])
    df = read_parquet_columns(source, columns)
    required = set(columns)
    if not required <= set(df.columns):
        return [], {"source_path": str(source), "status": "missing_columns", "missing": sorted(required - set(df.columns))}
    df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    events: list[dict[str, object]] = []
    thresholds: dict[str, object] = {}
    for component, spec in COMPONENTS.items():
        prelim = pd.to_numeric(df[spec["preliminary_col"]], errors="coerce")
        final = pd.to_numeric(df[spec["finalized_col"]], errors="coerce")
        revision = (final - prelim).abs()
        positive = revision[revision.gt(0)].dropna()
        if positive.empty:
            thresholds[component] = {"positive_count": 0, "threshold": None, "status": "no_positive_revision"}
            continue
        threshold = float(positive.quantile(positive_quantile))
        if threshold <= 0:
            thresholds[component] = {"positive_count": int(len(positive)), "threshold": threshold, "status": "nonpositive_threshold"}
            continue
        thresholds[component] = {
            "positive_count": int(len(positive)),
            "quantile": float(positive_quantile),
            "threshold": threshold,
            "threshold_scope": "full_revision_source_descriptive",
            "event_source_kind": "revision_derived",
            "event_role": "descriptive_shift_event",
            "not_model_input": True,
        }
        selected = df[revision.ge(threshold) & revision.gt(0) & df["week_end"].notna()].copy()
        selected["_revision_magnitude"] = revision.loc[selected.index]
        selected = selected.sort_values(["week_end", "_revision_magnitude", "jurisdiction"], ascending=[True, False, True])
        for _, row in selected.iterrows():
            events.append(
                event_row(
                    unit=str(row["jurisdiction"]),
                    pathogen=spec["pathogen"],
                    component=component,
                    event_time=row["week_end"],
                    event_type="large_revision_spike",
                    label=f"Large revision spike: {row['jurisdiction']} {spec['label']}",
                    selection_rule="positive preliminary finalized revision magnitude quantile by component",
                    source_path=source,
                    threshold=f"positive_revision_q{positive_quantile:.2f}={threshold:.6g}",
                    window_weeks_before=before,
                    window_weeks_after=after,
                    unit_scope="jurisdiction",
                    applies_to_all_units=False,
                    expansion_required_for_R3=False,
                    event_source_kind="revision_derived",
                    event_role="descriptive_shift_event",
                    not_model_input=True,
                )
            )
    meta = {
        "source_path": str(source),
        "status": "ok",
        "selection_rule": "component-wise positive preliminary-finalized revision quantile",
        "threshold_scope": "full_revision_source_descriptive",
        "not_used_for_training_or_metric_tuning": True,
        "use_for_posterior_diagnostic_only": True,
        "input_available_at_forecast_origin_claim": False,
        "count": len(events),
        "thresholds": thresholds,
    }
    if not events:
        meta["limitation"] = "revision source did not yield qualifying positive revision spikes; no revision event fabricated"
    return events, meta


def finalize_events(events: list[dict[str, object]], dataset_key: str = DATASET_KEY) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.DataFrame(events)
    df = df.sort_values(["event_time", "event_type", "component", "unit", "label"], kind="mergesort").reset_index(drop=True)
    df["event_id"] = [
        f"{dataset_key}_{i:04d}_{slug(row.event_type)}_{slug(row.component)}_{slug(row.unit)}_{str(row.event_time).replace('-', '')}"
        for i, row in enumerate(df.itertuples(index=False), start=1)
    ]
    return df[EVENT_COLUMNS]


EVENT_COLUMNS = [
    "event_id",
    "dataset_key",
    "unit",
    "pathogen",
    "component",
    "event_time",
    "event_type",
    "label",
    "window_weeks_before",
    "window_weeks_after",
    "selection_rule",
    "source_path",
    "threshold",
    "created_at",
    "unit_scope",
    "applies_to_all_units",
    "expansion_required_for_R3",
    "event_source_kind",
    "event_role",
    "not_model_input",
]


def build_manifest(
    *,
    run_root: Path,
    out: Path,
    manifest: Path,
    gold_root: Path,
    event_df: pd.DataFrame,
    source_meta: dict[str, object],
    before: int,
    after: int,
    revision_positive_quantile: float,
    variant_share_threshold: float,
) -> dict[str, object]:
    counts = event_df["event_type"].value_counts().sort_index().to_dict() if not event_df.empty else {}
    non_calendar = sum(int(counts.get(k, 0)) for k in ["variant_turnover", "large_revision_spike"])
    warnings = []
    if event_df.empty:
        warnings.append("No qualifying predeclared events were produced.")
    elif non_calendar == 0:
        warnings.append("Only calendar events were produced; posterior-adaptation evidence will be weak until variant or revision events are available.")
    return {
        "phase": "R2",
        "dataset_key": DATASET_KEY,
        "run_root": str(run_root),
        "gold_root": str(gold_root),
        "created_at": CREATED_AT,
        "predeclared_definition": "Rules are fixed before R3 event-aligned posterior construction and do not use CASTER outcomes.",
        "files_read": [
            str(gold_root / "panel_targets_long.parquet"),
            str(gold_root / "aux_variants_summary.parquet"),
            str(gold_root / "aux_preliminary_vs_finalized_paired.parquet"),
        ],
        "files_written": [str(out), str(manifest)],
        "no_event_aligned_posterior_written": True,
        "no_algorithm_scoring_tracker_files_modified": True,
        "event_counts_by_type": counts,
        "total_events": int(len(event_df)),
        "window_weeks_before": int(before),
        "window_weeks_after": int(after),
        "thresholds": {
            "variant_share_threshold": float(variant_share_threshold),
            "revision_positive_quantile": float(revision_positive_quantile),
            "revision_threshold_scope": "full_revision_source_descriptive",
        },
        "source_metadata": source_meta,
        "warnings": warnings,
        **NO_LEAKAGE_FLAGS,
    }


def build_events(
    *,
    run_root: Path,
    out: Path,
    manifest: Path,
    gold_root: Path,
    before: int,
    after: int,
    revision_positive_quantile: float,
    variant_share_threshold: float,
) -> int:
    winter_events, winter_meta = build_winter_onset_events(gold_root, before, after)
    variant_events, variant_meta = build_variant_turnover_events(gold_root, before, after, variant_share_threshold)
    revision_events, revision_meta = build_revision_spike_events(gold_root, before, after, revision_positive_quantile)
    event_df = finalize_events(winter_events + variant_events + revision_events)
    source_meta = {
        "winter_onset": winter_meta,
        "variant_turnover": variant_meta,
        "large_revision_spike": revision_meta,
    }
    manifest_doc = build_manifest(
        run_root=run_root,
        out=out,
        manifest=manifest,
        gold_root=gold_root,
        event_df=event_df,
        source_meta=source_meta,
        before=before,
        after=after,
        revision_positive_quantile=revision_positive_quantile,
        variant_share_threshold=variant_share_threshold,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    event_df.to_csv(out, index=False)
    manifest.write_text(json.dumps(to_jsonable(manifest_doc), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"events_csv={out}")
    print(f"manifest={manifest}")
    print(f"events={len(event_df)}")
    if not event_df.empty:
        print(event_df["event_type"].value_counts().sort_index().to_string())
    return 0


def benchmark_a_event_row(
    *,
    unit: str,
    component: str,
    event_time: object,
    event_type: str,
    label: str,
    selection_rule: str,
    source_path: Path,
    threshold: str,
    before: int,
    after: int,
    unit_scope: str,
    applies_to_all_units: bool,
) -> dict[str, object]:
    return {
        "event_id": "",
        "dataset_key": BENCHMARK_A_KEY,
        "unit": unit,
        "pathogen": "covid19",
        "component": component,
        "event_time": pd.to_datetime(event_time).strftime("%Y-%m-%d"),
        "event_type": event_type,
        "label": label,
        "window_weeks_before": int(before),
        "window_weeks_after": int(after),
        "selection_rule": selection_rule,
        "source_path": str(source_path),
        "threshold": threshold,
        "created_at": CREATED_AT,
        "unit_scope": unit_scope,
        "applies_to_all_units": bool(applies_to_all_units),
        "expansion_required_for_R3": bool(applies_to_all_units),
        "event_source_kind": "input_side_case_surveillance",
        "event_role": "predeclared_declared_wave_event",
        "not_model_input": True,
    }


def build_benchmark_a_declared_wave_events(
    panel_path: Path,
    before: int,
    after: int,
    wave_threshold: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not panel_path.exists():
        return [], {"source_path": str(panel_path), "status": "missing"}
    df = pd.read_csv(panel_path)
    required = {"country", "date", "component", "observed_value"}
    if not required <= set(df.columns):
        return [], {"source_path": str(panel_path), "status": "missing_columns", "missing": sorted(required - set(df.columns))}
    work = df[df["component"].astype(str).eq("cases")].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["observed_value"] = pd.to_numeric(work["observed_value"], errors="coerce")
    work = work.dropna(subset=["date", "country", "observed_value"])
    events: list[dict[str, object]] = []
    country_meta: dict[str, object] = {}
    for country, g in work.groupby("country", dropna=False):
        daily_total = g.groupby("date")["observed_value"].sum().sort_index()
        smoothed = daily_total.rolling(7, min_periods=3).mean()
        selected = smoothed[smoothed.ge(float(wave_threshold))]
        if selected.empty:
            country_meta[str(country)] = {"status": "no_threshold_crossing", "threshold": float(wave_threshold)}
            continue
        event_time = selected.index[0]
        country_meta[str(country)] = {
            "status": "ok",
            "event_time": pd.Timestamp(event_time).strftime("%Y-%m-%d"),
            "threshold": float(wave_threshold),
            "daily_total_7d_mean": float(selected.iloc[0]),
        }
        events.append(
            benchmark_a_event_row(
                unit="all_units",
                component="cases",
                event_time=event_time,
                event_type="declared_wave_onset",
                label=f"Declared epidemic wave onset: {country} cases",
                selection_rule="fixed country-level 7-day daily case total threshold from input-side Benchmark A panel",
                source_path=panel_path,
                threshold=f"country_total_cases_7d_mean>={wave_threshold:g}",
                before=before,
                after=after,
                unit_scope="country",
                applies_to_all_units=True,
            )
        )
    meta = {
        "source_path": str(panel_path),
        "status": "ok",
        "selection_rule": "first date per country where country-total 7-day mean cases crosses fixed threshold",
        "threshold": float(wave_threshold),
        "uses_posterior_movement": False,
        "uses_forecast_performance": False,
        "uses_test_metric_for_tuning": False,
        "count": len(events),
        "countries": country_meta,
    }
    if not events:
        meta["limitation"] = "Benchmark A panel did not yield qualifying declared-wave events under the fixed threshold; no event fabricated"
    return events, meta


def build_benchmark_a_events(
    *,
    run_root: Path,
    out: Path,
    panel_path: Path,
    before: int,
    after: int,
    wave_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    wave_events, wave_meta = build_benchmark_a_declared_wave_events(panel_path, before, after, wave_threshold)
    event_df = finalize_events(wave_events, dataset_key=BENCHMARK_A_KEY)
    out.parent.mkdir(parents=True, exist_ok=True)
    event_df.to_csv(out, index=False)
    manifest_doc = {
        "phase": "P4",
        "dataset_key": BENCHMARK_A_KEY,
        "run_root": str(run_root),
        "created_at": CREATED_AT,
        "predeclared_definition": "Benchmark A events are input-side declared-wave calendar/surveillance rules fixed before posterior validation.",
        "files_read": [str(panel_path)],
        "files_written": [str(out)],
        "event_counts_by_type": event_df["event_type"].value_counts().sort_index().to_dict() if not event_df.empty else {},
        "total_events": int(len(event_df)),
        "window_weeks_before": int(before),
        "window_weeks_after": int(after),
        "source_metadata": {"declared_wave_onset": wave_meta},
        "warnings": [] if not event_df.empty else ["No qualifying Benchmark A declared-wave events were produced."],
        "no_event_aligned_posterior_written": True,
        "no_algorithm_scoring_tracker_files_modified": True,
        **NO_LEAKAGE_FLAGS,
    }
    return event_df, manifest_doc


def build_benchmark_b_events_for_p4(
    *,
    run_root: Path,
    out: Path,
    manifest_path: Path,
    gold_root: Path,
    before: int,
    after: int,
    revision_positive_quantile: float,
    variant_share_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    winter_events, winter_meta = build_winter_onset_events(gold_root, before, after)
    variant_events, variant_meta = build_variant_turnover_events(gold_root, before, after, variant_share_threshold)
    revision_events, revision_meta = build_revision_spike_events(gold_root, before, after, revision_positive_quantile)
    event_df = finalize_events(winter_events + variant_events + revision_events, dataset_key=DATASET_KEY)
    out.parent.mkdir(parents=True, exist_ok=True)
    event_df.to_csv(out, index=False)
    manifest_doc = build_manifest(
        run_root=run_root,
        out=out,
        manifest=manifest_path,
        gold_root=gold_root,
        event_df=event_df,
        source_meta={
            "winter_onset": winter_meta,
            "variant_turnover": variant_meta,
            "large_revision_spike": revision_meta,
        },
        before=before,
        after=after,
        revision_positive_quantile=revision_positive_quantile,
        variant_share_threshold=variant_share_threshold,
    )
    manifest_doc["phase"] = "P4"
    return event_df, manifest_doc


def validate_no_split_markers(event_df: pd.DataFrame, dataset: str) -> list[str]:
    errors: list[str] = []
    if event_df.empty:
        return errors
    for col in [c for c in EVENT_FACING_COLUMNS if c in event_df.columns]:
        bad = event_df[col].astype(str).str.contains(SPLIT_MARKER_RE, na=False)
        if bad.any():
            errors.append(f"{dataset}: split-derived marker token found in {col}")
    return errors


def build_multi_dataset_events(
    *,
    run_root: Path,
    datasets: Sequence[str],
    out_root: Path,
    manifest: Path,
    gold_root: Path,
    benchmark_a_panel: Path,
    before: int,
    after: int,
    revision_positive_quantile: float,
    variant_share_threshold: float,
    benchmark_a_wave_threshold: float,
) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    combined: dict[str, object] = {
        "phase": "P4",
        "run_root": str(run_root),
        "created_at": CREATED_AT,
        "datasets": list(datasets),
        "dataset_manifests": {},
        "event_counts_by_dataset": {},
        "event_counts_by_dataset_type": {},
        "event_counts_by_type": {},
        "files_read": [],
        "files_written": [str(manifest)],
        "predeclared_definition": "Events are generated from input-side calendar/surveillance/revision sources before H3 posterior validation.",
        "no_event_aligned_posterior_written": True,
        "no_algorithm_scoring_tracker_files_modified": True,
        **NO_LEAKAGE_FLAGS,
    }
    errors: list[str] = []
    for dataset in datasets:
        dataset = dataset.strip()
        if not dataset:
            continue
        dataset_out = out_root / dataset / "events.csv"
        if dataset == BENCHMARK_A_KEY:
            event_df, doc = build_benchmark_a_events(
                run_root=run_root,
                out=dataset_out,
                panel_path=benchmark_a_panel,
                before=before,
                after=after,
                wave_threshold=benchmark_a_wave_threshold,
            )
        elif dataset == DATASET_KEY:
            event_df, doc = build_benchmark_b_events_for_p4(
                run_root=run_root,
                out=dataset_out,
                manifest_path=manifest,
                gold_root=gold_root,
                before=before,
                after=after,
                revision_positive_quantile=revision_positive_quantile,
                variant_share_threshold=variant_share_threshold,
            )
        else:
            errors.append(f"unsupported dataset: {dataset}")
            continue
        errors.extend(validate_no_split_markers(event_df, dataset))
        combined["dataset_manifests"][dataset] = doc
        combined["event_counts_by_dataset"][dataset] = int(len(event_df))
        combined["event_counts_by_dataset_type"][dataset] = event_df["event_type"].value_counts().sort_index().to_dict() if not event_df.empty else {}
        for path in doc.get("files_read", []):
            if path not in combined["files_read"]:
                combined["files_read"].append(path)
        for path in doc.get("files_written", [str(dataset_out)]):
            if path not in combined["files_written"]:
                combined["files_written"].append(path)
        for event_type, count in combined["event_counts_by_dataset_type"][dataset].items():
            combined["event_counts_by_type"][event_type] = int(combined["event_counts_by_type"].get(event_type, 0)) + int(count)
        print(f"{dataset}_events_csv={dataset_out}")
        print(f"{dataset}_events={len(event_df)}")
        if not event_df.empty:
            print(event_df["event_type"].value_counts().sort_index().to_string())
    combined["validation_errors"] = errors
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(to_jsonable(combined), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest={manifest}")
    if errors:
        for err in errors:
            print(f"EVENT_VALIDATION_ERROR {err}", file=sys.stderr)
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build predeclared benchmark_b shift events")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold-root", type=Path, default=default_gold_root())
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--benchmark-a-panel", type=Path, default=default_benchmark_a_panel())
    parser.add_argument("--benchmark-a-wave-threshold", type=float, default=BENCHMARK_A_WAVE_7D_TOTAL_THRESHOLD)
    parser.add_argument("--window-weeks-before", type=int, default=WINDOW_BEFORE)
    parser.add_argument("--window-weeks-after", type=int, default=WINDOW_AFTER)
    parser.add_argument("--revision-positive-quantile", type=float, default=REVISION_POSITIVE_QUANTILE)
    parser.add_argument("--variant-share-threshold", type=float, default=VARIANT_SHARE_THRESHOLD)
    args = parser.parse_args(argv)
    if args.datasets or args.out_root:
        datasets = [x.strip() for x in (args.datasets or DATASET_KEY).split(",") if x.strip()]
        out_root = args.out_root or (args.run_root / "events")
        return build_multi_dataset_events(
            run_root=args.run_root,
            datasets=datasets,
            out_root=out_root,
            manifest=args.manifest,
            gold_root=args.gold_root,
            benchmark_a_panel=args.benchmark_a_panel,
            before=args.window_weeks_before,
            after=args.window_weeks_after,
            revision_positive_quantile=args.revision_positive_quantile,
            variant_share_threshold=args.variant_share_threshold,
            benchmark_a_wave_threshold=args.benchmark_a_wave_threshold,
        )
    if args.out is None:
        parser.error("--out is required unless --datasets/--out-root is used")
    return build_events(
        run_root=args.run_root,
        out=args.out,
        manifest=args.manifest,
        gold_root=args.gold_root,
        before=args.window_weeks_before,
        after=args.window_weeks_after,
        revision_positive_quantile=args.revision_positive_quantile,
        variant_share_threshold=args.variant_share_threshold,
    )


if __name__ == "__main__":
    sys.exit(main())
