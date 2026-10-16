"""Point-in-time invitation-response modeling. Importing this module performs no I/O."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits


VERSION = "1.0"
HORIZONS = (90, 180, 365)
TARGETS = {90: "screening_90d", 180: "screening_180d", 365: "screening_365d"}
OUTPUTS = tuple(f"p_nonattendance_{h}d" for h in HORIZONS)
KEYS = ("invitation_id", "person_id", "invitation_date")
MANDATORY_MODELS = ("dummy", "logistic", "decision_tree", "random_forest",
                    "extra_trees", "hist_gradient_boosting", "mlp")
OPTIONAL_MODELS = ("lightgbm", "xgboost", "catboost")
ALIASES = {
    "screening_3m_flag": "screening_90d",
    "screening_6m_flag": "screening_180d",
    "screening_12m_flag": "screening_365d",
    "vzp_territorial_area": "territory",
    "never_screened_before_flag": "no_screen_observed_5y",
    "gyn_registered_flag": "gyn_registered",
}
BINARY = {"first_invitation", "gp_registered", "gyn_registered",
          "no_screen_observed_5y", *TARGETS.values()}
CATEGORICAL = {"age_band", "territory", "gp_reg_duration_band", "gyn_reg_duration_band"}
NUMERIC = {
    "prior_invitation_count", "invitation_sequence", "months_last_invitation",
    "healthcare_contact_12m_cnt", "insurance_duration_years", "prior_screen_count_5y",
    "months_since_last_screen", "gp_prevention_24m_cnt", "gyn_prevention_24m_cnt",
    "dental_prevention_24m_cnt", "other_screening_36m_cnt",
}
AMBIGUOUS = {"prior_invitation_count", "invitation_sequence", "months_last_invitation"}
DOMAINS = ("gp_prevention_24m_cnt", "gyn_prevention_24m_cnt",
           "dental_prevention_24m_cnt", "other_screening_36m_cnt")
SNAPSHOT_HISTORY = ("healthcare_contact_12m_cnt", "prior_screen_count_5y",
                    "gp_prevention_24m_cnt", "gyn_prevention_24m_cnt")
DEFINITIONS = {
    "age_band": ("demographics", "category", "Documented five-year age band; no exact age."),
    "territory": ("geography", "category", "VZP territorial office, not an administrative district."),
    "insurance_duration_years": ("registration", "whole years", "Insurance duration at T0."),
    "gp_registered": ("registration", "binary", "Valid GP registration at T0; specialties 001/002."),
    "gyn_registered": ("registration", "binary", "Valid gynecology registration at T0; 603/604."),
    "gp_reg_duration_band": ("registration", "years band", "Current GP registration: <2, 2-5, >5 years."),
    "gyn_reg_duration_band": ("registration", "years band", "Current gynecology registration duration band."),
    "prior_screen_count_5y": ("screening", "count/60 months", "Prior screening codes 89178,89223,89225,89314,89510,89814."),
    "months_since_last_screen": ("screening", "months", "Recency in documented 60-month lookback; missing when none observed."),
    "no_screen_observed_5y": ("screening", "binary", "Source NEVER flag restricted to documented 60-month observation."),
    "healthcare_contact_12m_cnt": ("engagement", "provider/day count/12 months", "Sum of unique provider identifiers per day; not distinct visits."),
    "first_invitation": ("invitation", "binary", "Source first-invitation flag; distinct from first observed input row."),
    "prior_invitation_count": ("invitation", "count", "Source count: inclusion of current invitation and coverage unresolved."),
    "invitation_sequence": ("invitation", "ordinal", "Source sequence: relationship to source count unresolved."),
    "months_last_invitation": ("invitation", "months", "Source invitation recency; time reference not explicitly specified."),
    "gp_prevention_24m_cnt": ("engagement", "count/24 months", "GP prevention codes 01021,01022 before T0."),
    "gyn_prevention_24m_cnt": ("engagement", "count/24 months", "Gynecology prevention codes 63050,63021 before T0."),
    "dental_prevention_24m_cnt": ("engagement", "count/24 months", "Dental prevention codes 00946,00901 before T0."),
    "other_screening_36m_cnt": ("engagement", "count/36 months", "Other screening, documented 36-month window and dictionary procedure list."),
}


class ContractError(ValueError):
    """Invalid data, temporal contract, or incompatible artifact."""


@dataclass
class Config:
    data: str = "data.csv"
    data_mode: str = "sample"
    allow_full_data: bool = False
    separator: str = ","
    encoding: str = "utf-8-sig"
    date_format: str = "%d.%m.%Y"
    schema_mapping: dict[str, str] = field(default_factory=dict)
    outcome_data_end: str | None = None
    reporting_lag_days: int = 0
    assume_t0_snapshots: bool = False
    test_year: int | None = None
    test_start: str | None = None
    test_end: str | None = None
    validation_start: str | None = None
    validation_end: str | None = None
    folds: int = 2
    cohort_policy: str = "horizon"
    historical_outcomes: str = "off"
    history_mode: str = "require"
    feature_sets: str = "raw_snapshot,snapshot_engineered"
    include_groups: str = ""
    exclude_groups: str = ""
    models: str = "logistic,decision_tree,hist_gradient_boosting"
    preset: str = "quick"
    search_method: str = "grid"
    budget_seconds: float | None = None
    max_trials: int = 60
    seed: int = 17
    n_jobs: int = 4
    memory_mb: int = 4096
    chunk_size: int = 100000
    sample_cap: int | None = None
    calibration: str = "none"
    calibration_days: int = 180
    bootstrap: int = 100
    explanation_rows: int = 150
    permutation_repeats: int = 3
    minimum_segment: int = 20
    strict_optional: bool = False
    patient_disjoint_stress: bool = False
    categorical_mode: str = "onehot"
    joint_interval: bool = False
    output_dir: str = "artifacts/run"
    provenance: str = "sample-specific; sampling mechanism unknown"

    def validate(self) -> None:
        if self.data_mode not in {"sample", "full"}:
            raise ContractError("data_mode must be sample or full.")
        if self.data_mode == "full" and not self.allow_full_data:
            raise ContractError("Full data requires --data-mode full --allow-full-data.")
        if self.reporting_lag_days < 0:
            raise ContractError("Reporting lag cannot be negative.")
        if self.folds < 1 or self.n_jobs < 1 or self.memory_mb < 1 or self.chunk_size < 1:
            raise ContractError("Folds, threads, memory and chunk size must be positive.")
        if self.budget_seconds is not None and self.budget_seconds <= 0:
            raise ContractError("Search budget must be positive.")
        if bool(self.test_start) != bool(self.test_end):
            raise ContractError("Both --test-start and --test-end are required together.")
        if bool(self.validation_start) != bool(self.validation_end):
            raise ContractError("Both validation boundaries are required together.")
        if self.test_year and self.test_start:
            raise ContractError("Use test year OR test boundaries.")
        if self.sample_cap is not None and self.sample_cap < 1:
            raise ContractError("Patient sample cap must be positive.")
        if len(self.separator) != 1:
            raise ContractError("CSV separator must be exactly one character.")
        if self.max_trials < 1 or self.calibration_days < 1:
            raise ContractError("Trial count and calibration window must be positive.")

    @property
    def budget(self) -> float:
        return self.budget_seconds if self.budget_seconds is not None else {
            "smoke": 60, "quick": 120, "standard": 1200, "extensive": 3600,
        }[self.preset]


def normalized_name(name: str) -> str:
    return "_".join(name.strip().lower().replace("-", "_").split())


def canonical_names(columns: Iterable[str], mapping: dict[str, str]) -> list[str]:
    aliases = dict(ALIASES)
    for key, value in mapping.items():
        norm = normalized_name(key)
        canonical = normalized_name(value)
        if norm in aliases and aliases[norm] != canonical:
            raise ContractError(f"Ambiguous alias override for {key}.")
        aliases[norm] = canonical
    result = [aliases.get(normalized_name(c), normalized_name(c)) for c in columns]
    if len(set(result)) != len(result):
        raise ContractError("Duplicate columns or normalization/alias collision.")
    return result


def validate_path(path: str, cfg: Config) -> Path:
    cfg.validate()
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ContractError(f"Input file does not exist: {p}")
    if cfg.data_mode == "sample":
        if p.name.casefold() == "rakhaton2026.csv" or p.stat().st_size > 50_000_000:
            raise ContractError("Refusing full/large CSV in sample mode before opening it.")
    return p


def parse_binary(series: pd.Series, name: str) -> pd.Series:
    mapping = {"a": 1., "y": 1., "true": 1., "1": 1., "1.0": 1.,
               "n": 0., "false": 0., "0": 0., "0.0": 0.}
    text = series.astype("string").str.strip().str.lower()
    missing = text.isna() | text.isin(["", "null", "na", "nan"])
    invalid = ~missing & ~text.isin(mapping)
    if invalid.any():
        raise ContractError(f"Unsupported binary encoding in {name}; {int(invalid.sum())} rows.")
    return text.map(mapping).astype(float)


def normalize_schema(frame: pd.DataFrame, cfg: Config, require_targets: bool = False) -> pd.DataFrame:
    df = frame.copy()
    df.columns = canonical_names(df.columns, cfg.schema_mapping)
    df = df.reset_index(drop=True)
    missing = [c for c in KEYS if c not in df]
    if require_targets:
        missing += [c for c in TARGETS.values() if c not in df]
    if missing:
        raise ContractError(f"Missing required columns: {missing}. Provide an explicit schema mapping.")
    for c in ("invitation_id", "person_id"):
        df[c] = df[c].astype("string").str.strip()
        if (df[c].isna() | df[c].eq("")).any():
            raise ContractError(f"Missing identifier: {c}.")
    if pd.api.types.is_datetime64_any_dtype(df["invitation_date"]):
        dates = df["invitation_date"]
    else:
        dates = pd.to_datetime(df["invitation_date"], format=cfg.date_format, errors="coerce")
    if dates.isna().any() or getattr(dates.dt, "tz", None) is not None:
        raise ContractError("Invalid invitation dates; specify the exact --date-format; timezone-free dates required.")
    df["invitation_date"] = dates
    for c in BINARY.intersection(df.columns):
        df[c] = parse_binary(df[c], c)
    for c in NUMERIC.intersection(df.columns):
        original = df[c].replace(r"^\s*$", np.nan, regex=True)
        number = pd.to_numeric(original, errors="coerce")
        if (original.notna() & number.isna()).any() or (number.dropna() < 0).any():
            raise ContractError(f"Invalid/negative numeric values: {c}.")
        if not np.isfinite(number.dropna()).all():
            raise ContractError(f"Nonfinite numeric values: {c}.")
        if (c.endswith("_cnt") or c in {"prior_screen_count_5y", *AMBIGUOUS - {"months_last_invitation"}}):
            if (number.dropna() % 1 != 0).any():
                raise ContractError(f"Nonintegral count: {c}.")
        df[c] = number.astype(float)
    for c in CATEGORICAL.intersection(df.columns):
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA).astype(object)
        df[c] = df[c].where(pd.notna(df[c]), np.nan)
    for c in {"gp_reg_duration_band", "gyn_reg_duration_band"}.intersection(df):
        if not df[c].dropna().isin(["<2", "2-5", ">5"]).all():
            raise ContractError(f"Unexpected documented duration band: {c}.")
    if "age_band" in df:
        allowed = {"<25", "< 25", "75+", *(f"{a}-{a+4}" for a in range(25, 75, 5))}
        if not df["age_band"].dropna().isin(allowed).all():
            raise ContractError("Unexpected age band; update the audited dictionary mapping explicitly.")
    if df["invitation_id"].duplicated().any():
        raise ContractError("Duplicate/conflicting invitation IDs; deduplicate explicitly before execution.")
    if df.duplicated(["person_id", "invitation_date"]).any():
        raise ContractError("Ambiguous same-person/same-time invitations; no real sequence documented.")
    if {"prior_screen_count_5y", "no_screen_observed_5y"}.issubset(df):
        known = df["prior_screen_count_5y"].notna() & df["no_screen_observed_5y"].notna()
        if (known & df["prior_screen_count_5y"].eq(0).ne(df["no_screen_observed_5y"].eq(1))).any():
            raise ContractError("Contradictory five-year screening count and no-screen-observed flag.")
    if {"no_screen_observed_5y", "months_since_last_screen"}.issubset(df):
        if (df["no_screen_observed_5y"].eq(1) & df["months_since_last_screen"].notna()).any():
            raise ContractError("No-screen-observed flag contradicts recorded screening recency.")
    for source in ("gp", "gyn"):
        flag, band = f"{source}_registered", f"{source}_reg_duration_band"
        if {flag, band}.issubset(df) and (df[flag].eq(0) & df[band].notna()).any():
            raise ContractError(f"Unregistered {source.upper()} record has a current-registration duration band.")
    validate_targets(df)
    return df


def load_csv(cfg: Config, path: str | None = None, require_targets: bool = False) -> pd.DataFrame:
    p = validate_path(path or cfg.data, cfg)
    with p.open(encoding=cfg.encoding, newline="") as handle:
        header = next(csv.reader(handle, delimiter=cfg.separator), [])
    names = canonical_names(header, cfg.schema_mapping)
    if not header:
        raise ContractError("CSV is empty.")
    approved = set(KEYS) | NUMERIC | BINARY | CATEGORICAL
    # Full mode materializes selected columns before global chronological grouping.
    usecols = [s for s, c in zip(header, names) if c in approved] if cfg.data_mode == "full" else None
    chunks: list[pd.DataFrame] = []
    used = 0
    for chunk in pd.read_csv(p, sep=cfg.separator, encoding=cfg.encoding, dtype=str,
                             keep_default_na=False, usecols=usecols, chunksize=cfg.chunk_size):
        used += int(chunk.memory_usage(deep=True).sum())
        if used * 2 > cfg.memory_mb * 1024**2:
            raise ContractError("Selected raw data exceed materialization memory guard. Increase --memory-mb in a suitable environment.")
        chunks.append(chunk)
    if not chunks:
        raise ContractError("CSV contains no rows.")
    df = normalize_schema(pd.concat(chunks, ignore_index=True), cfg, require_targets)
    if cfg.sample_cap and df["person_id"].nunique() > cfg.sample_cap:
        people = np.array(sorted(df["person_id"].unique()))
        chosen = np.random.default_rng(cfg.seed).choice(people, cfg.sample_cap, replace=False)
        df = df[df["person_id"].isin(chosen)].reset_index(drop=True)
    return df


def validate_targets(df: pd.DataFrame) -> None:
    for a, b in ((90, 180), (180, 365), (90, 365)):
        ca, cb = TARGETS[a], TARGETS[b]
        if ca in df and cb in df:
            observed = df[ca].notna() & df[cb].notna()
            bad = observed & (df[ca] > df[cb])
            if bad.any():
                raise ContractError(f"Contradictory cumulative labels {a}/{b} days: {int(bad.sum())} rows; no silent repair.")


def schema_report(df: pd.DataFrame) -> list[dict[str, Any]]:
    report = []
    for c in df:
        definition = DEFINITIONS.get(c)
        role = ("identifier" if c in KEYS[:2] else "timestamp" if c == KEYS[2]
                else "outcome" if c in TARGETS.values() else "predictor" if definition else "unknown")
        report.append({
            "name": c, "role": role, "dtype": str(df[c].dtype),
            "units": definition[1] if definition else "date" if role == "timestamp" else "not applicable",
            "missing_count": int(df[c].isna().sum()), "missing_fraction": float(df[c].isna().mean()),
            "cardinality": int(df[c].nunique()), "time_reference": "post-T0" if role == "outcome" else "T0" if definition else role,
            "source_definition": definition[2] if definition else "90/180/365-day recorded attendance" if role == "outcome" else role,
            "inclusion": "excluded: unresolved source history semantics" if c in AMBIGUOUS else
                         "approved T0 snapshot, subject to explicit backfill assumption" if definition else "excluded",
            "unresolved_questions": "Counter coverage/current-invitation semantics and T0 reference need source confirmation." if c in AMBIGUOUS else
                "Retrospective recording latency unavailable." if definition else
                "Verified observation cutoff and reporting delay required." if role == "outcome" else
                "Unapproved extra column; not a predictor." if role == "unknown" else "",
        })
    return report


def outcome_end(cfg: Config) -> pd.Timestamp:
    if not cfg.outcome_data_end:
        raise ContractError("A verified --outcome-data-end is required; insurance eligibility is not outcome ascertainment.")
    return iso_date(cfg.outcome_data_end)


def iso_date(value: str | pd.Timestamp) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ContractError(f"Invalid ISO date: {value}") from exc
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ContractError("Expected a timezone-free calendar date.")
    return result


def label_available_at(df: pd.DataFrame, horizon: int, cfg: Config) -> pd.Series:
    if horizon not in HORIZONS:
        raise ContractError("Unsupported target horizon.")
    return df["invitation_date"] + pd.to_timedelta(horizon + cfg.reporting_lag_days, unit="D")


def observed_mask(df: pd.DataFrame, horizon: int, cfg: Config, deadline: pd.Timestamp | None = None) -> pd.Series:
    if TARGETS[horizon] not in df:
        return pd.Series(False, index=df.index)
    cutoff = min(outcome_end(cfg), deadline) if deadline is not None else outcome_end(cfg)
    return df[TARGETS[horizon]].notna() & (label_available_at(df, horizon, cfg) < cutoff)


def train_mask(df: pd.DataFrame, horizon: int, origin: pd.Timestamp, cfg: Config) -> pd.Series:
    mask = (df["invitation_date"] < origin) & observed_mask(df, horizon, cfg, origin)
    if cfg.cohort_policy == "common":
        for h in HORIZONS:
            mask &= observed_mask(df, h, cfg, origin)
    return mask


def common_mask(df: pd.DataFrame, cfg: Config, deadline: pd.Timestamp | None = None) -> pd.Series:
    return pd.concat([observed_mask(df, h, cfg, deadline) for h in HORIZONS], axis=1).all(axis=1)


def fingerprint(df: pd.DataFrame, columns: Sequence[str] = KEYS) -> str:
    ordered = df.loc[:, list(columns)].sort_values(list(columns)).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).values.tobytes()).hexdigest()


def config_hash(cfg: Config) -> str:
    settings = asdict(cfg)
    for k in ("output_dir", "budget_seconds"):
        settings.pop(k)
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def reserve_test(df: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    cutoff = outcome_end(cfg)
    if cfg.test_start:
        start, end = iso_date(cfg.test_start), iso_date(cfg.test_end or "")
    else:
        if cfg.test_year:
            year = cfg.test_year
        else:
            years = sorted(df["invitation_date"].dt.year.unique(), reverse=True)
            eligible = [int(y) for y in years if pd.Timestamp(int(y), 12, 31)
                        + pd.Timedelta(days=365 + cfg.reporting_lag_days) < cutoff]
            if not eligible:
                raise ContractError("No complete invitation calendar year has verified 365-day follow-up.")
            year = eligible[0]
        start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
    if end < start or end + pd.Timedelta(days=365 + cfg.reporting_lag_days) >= cutoff:
        raise ContractError("Test interval does not have full conservative 365-day follow-up.")
    holdout = df[df["invitation_date"].between(start, end)]
    if holdout.empty:
        raise ContractError("Declared test interval has no invitations.")
    manifest = {
        "version": VERSION, "test_start": start.isoformat(), "test_end": end.isoformat(),
        "fit_origin": start.isoformat(), "selection_deadline": start.isoformat(),
        "test_fingerprint": fingerprint(holdout), "dataset_keys_fingerprint": fingerprint(df),
        "test_rows": len(holdout), "test_patients": int(holdout["person_id"].nunique()),
        "outcome_data_end": cutoff.isoformat(), "reporting_lag_days": cfg.reporting_lag_days,
        "boundary": "availability strictly before cutoff/origin", "provenance": cfg.provenance,
        "final_test_used_for_selection": False,
    }
    return manifest


@dataclass(frozen=True)
class Fold:
    start: pd.Timestamp
    end: pd.Timestamp
    selection_deadline: pd.Timestamp


def make_temporal_folds(df: pd.DataFrame, cfg: Config, manifest: dict[str, Any]) -> list[Fold]:
    deadline = iso_date(manifest["selection_deadline"])
    last_possible = deadline - pd.Timedelta(days=366 + cfg.reporting_lag_days)
    if cfg.validation_start:
        windows = [(iso_date(cfg.validation_start), iso_date(cfg.validation_end or ""))]
    else:
        last_year = last_possible.year if pd.Timestamp(last_possible.year, 12, 31) <= last_possible else last_possible.year - 1
        windows = [(pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31))
                   for y in range(last_year - cfg.folds + 1, last_year + 1)]
    folds = []
    for start, end in windows:
        if end < start or end >= deadline or end + pd.Timedelta(days=365 + cfg.reporting_lag_days) >= deadline:
            raise ContractError("Validation outcomes would not be available before the selection deadline.")
        validation = df["invitation_date"].between(start, end) & common_mask(df, cfg, deadline)
        if not validation.any():
            continue
        if not all(train_mask(df, h, start, cfg).any() for h in HORIZONS):
            continue
        folds.append(Fold(start, end, deadline))
    if not folds:
        raise ContractError("Insufficient chronological coverage: no mature expanding training/validation fold. No random fallback.")
    return folds


def audit_split(df: pd.DataFrame, fold: Fold, horizon: int, cfg: Config,
                training: pd.Series | None = None) -> dict[str, Any]:
    tr = train_mask(df, horizon, fold.start, cfg) if training is None else training
    va = df["invitation_date"].between(fold.start, fold.end) & common_mask(df, cfg, fold.selection_deadline)
    if ((label_available_at(df, horizon, cfg)[tr] >= fold.start).any()
            or (df.loc[tr, "invitation_date"] >= fold.start).any()
            or set(df.loc[tr, "invitation_id"]) & set(df.loc[va, "invitation_id"])):
        raise ContractError("Availability/split assertion detected training leakage.")
    return {
        "horizon": horizon, "fit_origin": fold.start.isoformat(),
        "validation_start": fold.start.isoformat(), "validation_end": fold.end.isoformat(),
        "selection_deadline": fold.selection_deadline.isoformat(),
        "train_rows": int(tr.sum()), "validation_rows": int(va.sum()),
        "train_patients": int(df.loc[tr, "person_id"].nunique()),
        "validation_patients": int(df.loc[va, "person_id"].nunique()),
        "overlap_patients": len(set(df.loc[tr, "person_id"]) & set(df.loc[va, "person_id"])),
        "immature_pre_origin": int(((df["invitation_date"] < fold.start) &
                                    ~(label_available_at(df, horizon, cfg) < fold.start)).sum()),
        "cohort_policy": cfg.cohort_policy,
    }


@dataclass(frozen=True)
class Feature:
    name: str
    description: str
    source_columns: tuple[str, ...]
    formula: str
    units: str = "dimensionless"
    dtype: str = "numeric"
    semantic_groups: tuple[str, ...] = ()
    transformation_tags: tuple[str, ...] = ()
    required_history: str = "none"
    availability_rule: str = "T0 snapshot; retrospective backfill assumption required"
    missing_value_meaning: str = "Unknown; not zero"
    leakage_risk: str = "Predictor recording latency is not observed"
    interpretation_caveat: str = "Association, not causation"
    evidence_or_hypothesis: str = "Prespecified engineering hypothesis"
    enabled_status: str = "enabled"


def build_feature_registry() -> dict[str, Feature]:
    registry: dict[str, Feature] = {}

    def add(name: str, sources: Sequence[str], formula: str, groups: Sequence[str],
            tags: Sequence[str], **kwargs: Any) -> None:
        if name in registry:
            raise ContractError(f"Duplicate feature definition: {name}")
        registry[name] = Feature(name, formula, tuple(sources), formula,
                                 semantic_groups=tuple(groups), transformation_tags=tuple(tags), **kwargs)

    for name, (group, units, definition) in DEFINITIONS.items():
        add(name, (name,), definition, (group, "snapshot"), ("raw",), units=units,
            dtype="categorical" if name in CATEGORICAL else "numeric",
            enabled_status="disabled: unresolved dictionary semantics" if name in AMBIGUOUS else "enabled")
        if name not in CATEGORICAL and name not in AMBIGUOUS:
            add(f"{name}__missing", (name,), f"is_missing({name})", (group,), ("missingness",))
        if name in NUMERIC and name not in AMBIGUOUS:
            add(f"{name}__log1p", (name,), f"log1p({name})", (group,), ("log1p",))
    for name, formula in {
        "invitation_month": "calendar month, categorical alternative to cyclic encoding",
        "invitation_quarter": "calendar quarter",
        "invitation_weekday": "calendar day of week",
        "month_sin": "sin(2*pi*(month-1)/12)",
        "month_cos": "cos(2*pi*(month-1)/12)",
        "year_trend": "calendar year minus 2020 (fixed reference, not fitted)",
    }.items():
        add(name, ("invitation_date",), formula, ("calendar",), ("cyclic" if "month_" in name else "calendar",),
            dtype="categorical" if name.startswith("invitation_") else "numeric",
            availability_rule="Invitation dispatch date known at T0",
            interpretation_caveat="Birthday-month selection and operational batching may explain calendar associations.")
    add("unknown_territory", ("territory",), "territory is missing", ("geography",), ("missingness",))
    add("age_band_order", ("age_band",), "Documented band ordinal position; <25=0, 25-29=1, ..., 75+=11",
        ("demographics", "age_order"), ("ordinal",),
        interpretation_caveat="Band order, NOT reconstructed exact age or a band midpoint.")
    add("age_band_order_x_screen_recency", ("age_band", "months_since_last_screen"),
        "Documented age-band order times recorded screening recency", ("demographics", "age_order", "screening"),
        ("interaction",), interpretation_caveat="Ordinal-band interaction, not an exact-age effect.")
    add("screen_count_category", ("prior_screen_count_5y",), "0 / 1 / multiple / missing", ("screening",), ("binned",), dtype="categorical")
    add("screen_recency_bin", ("months_since_last_screen",), "Prespecified <=24, (24,36], >36 months; missing separate",
        ("screening",), ("binned",), dtype="categorical")
    add("screen_gap_24m", ("months_since_last_screen",), "months_since_last_screen - 24", ("screening",), ("reference-gap",),
        units="months", interpretation_caveat="Behavioral reference only, not screening eligibility or clinical advice.")
    for name in DOMAINS:
        add(f"{name}__any", (name,), f"{name} > 0; preserve missing", ("engagement",), ("binary",))
        months = 36 if "36m" in name else 24
        add(f"{name}__annualized", (name,), f"{name} * 12/{months}", ("engagement",), ("rate",),
            units="recorded events/year")
    add("active_prevention_domains", DOMAINS, "Number of positive domains; missing if any domain unobserved",
        ("engagement",), ("count",))
    add("missing_prevention_domains", DOMAINS, "Number of missing preventive domains", ("engagement",), ("missingness",))
    add("prevention_contact_intensity_ratio", (*DOMAINS, "healthcare_contact_12m_cnt"),
        "Sum of annualized prevention counts / annual contacts; missing for zero denominator",
        ("engagement",), ("ratio",), interpretation_caveat="Different event definitions/windows; not a probability or bounded share.")
    for group in ("gp", "gyn"):
        add(f"{group}_registered_no_prevention", (f"{group}_registered", f"{group}_prevention_24m_cnt"),
            "Registered and zero recorded prevention; unknown preserved", ("registration", "engagement"), ("interaction",))
    add("registration_count", ("gp_registered", "gyn_registered"), "GP plus gynecology registration; unknown preserved",
        ("registration",), ("count",))
    add("screen_count_x_first_invitation", ("prior_screen_count_5y", "first_invitation"),
        "Prior screen count times source first-invitation flag", ("screening", "invitation"), ("interaction",))
    for name in ("observed_prior_rows", "days_since_previous_observed_invitation",
                 "days_since_first_observed_invitation", "prior_interval_mean", "prior_interval_min",
                 "prior_interval_max", "prior_interval_std", "observed_invitation_density",
                 "insufficient_observed_history"):
        add(name, ("person_id", "invitation_date"), "Strictly earlier input invitation summary: " + name,
            ("longitudinal", "invitation"), ("history",), required_history="earlier input rows",
            availability_rule="Only timestamps strictly before current T0", units="days" if "days" in name or "interval" in name else "count/rate",
            interpretation_caveat="Input history may be incomplete; not lifetime history.")
    for source in SNAPSHOT_HISTORY:
        for suffix in ("lag1", "lag2", "delta", "change_per_year", "past_mean", "past_min", "past_max", "past_std"):
            add(f"{source}__{suffix}", (source, "person_id", "invitation_date"),
                f"{suffix} of strictly earlier observed predictor snapshots" if suffix != "delta" else
                "Current minus previous rolling summary; NOT new medical-event counts",
                ("longitudinal", DEFINITIONS[source][0]), ("lag" if "lag" in suffix else "snapshot-summary",),
                required_history="earlier input snapshots", availability_rule="Strictly earlier T0 snapshots only")
    for h in HORIZONS:
        for suffix in ("mature_count", "last_nonattendance", "nonattendance_fraction"):
            add(f"historical_{h}d_{suffix}", ("person_id", "invitation_date", TARGETS[h]),
                "Earlier observed invitation outcomes with availability strictly before current T0: " + suffix,
                ("historical_outcomes",), ("mature-label-history",), required_history="mature earlier outcomes",
                availability_rule=f"earlier T0 + {h} days + reporting lag < current T0",
                enabled_status="opt-in", leakage_risk="Requires deployment outcome access; overlapping windows are not attribution.")
    return registry


PRESETS = ("minimal", "raw_snapshot", "demographics_calendar", "screening_invitation",
           "prevention_engagement", "registration_continuity", "snapshot_engineered",
           "longitudinal_no_labels", "longitudinal_with_mature_labels", "all_safe",
           "all_safe_no_geography", "all_safe_no_calendar", "all_safe_no_longitudinal")


def resolve_feature_set(preset: str, df: pd.DataFrame, cfg: Config) -> tuple[list[str], list[dict[str, str]]]:
    registry = build_feature_registry()
    if preset not in PRESETS:
        raise ContractError(f"Unknown feature preset: {preset}")
    minimal = {"age_band", "first_invitation", "prior_screen_count_5y", "months_since_last_screen",
               "gp_registered", "gyn_registered"}
    include, exclude = set(filter(None, cfg.include_groups.split(","))), set(filter(None, cfg.exclude_groups.split(",")))
    known_groups = {g for f in registry.values() for g in f.semantic_groups}
    if (include | exclude) - known_groups:
        raise ContractError(f"Unknown semantic groups: {(include | exclude) - known_groups}")
    selected, availability = [], []
    for name, f in registry.items():
        groups = set(f.semantic_groups)
        history, labels = "longitudinal" in groups, "historical_outcomes" in groups
        raw = "raw" in f.transformation_tags
        choose = {
            "minimal": name in minimal, "raw_snapshot": raw,
            "demographics_calendar": bool(groups & {"demographics", "geography", "calendar"}),
            "screening_invitation": bool(groups & {"screening", "invitation"}) and not history,
            "prevention_engagement": "engagement" in groups and not history,
            "registration_continuity": "registration" in groups,
            "snapshot_engineered": not history and not labels,
            "longitudinal_no_labels": not labels, "longitudinal_with_mature_labels": True,
            "all_safe": not labels, "all_safe_no_geography": not labels and "geography" not in groups,
            "all_safe_no_calendar": not labels and "calendar" not in groups,
            "all_safe_no_longitudinal": not labels and not history,
        }[preset]
        choose = (choose or bool(groups & include)) and not bool(groups & exclude)
        # Avoid simultaneously imposing alternative month encodings by default.
        if name in {"invitation_month", "invitation_quarter", "invitation_weekday"} and "calendar" not in include:
            choose = False
        if not choose:
            continue
        reason = ""
        if f.enabled_status.startswith("disabled"):
            reason = f.enabled_status
        elif labels and cfg.historical_outcomes != "on":
            reason = "Historical outcomes disabled; requires --historical-outcomes on"
        elif set(f.source_columns) - set(df.columns):
            reason = "Missing sources: " + ", ".join(sorted(set(f.source_columns) - set(df.columns)))
        if reason:
            availability.append({"name": name, "status": "unavailable", "reason": reason})
        else:
            selected.append(name)
            availability.append({"name": name, "status": "available", "reason": ""})
    if not selected:
        raise ContractError(f"Feature set {preset} resolves to zero usable features.")
    return selected, availability


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0)


def build_row_features(df: pd.DataFrame) -> pd.DataFrame:
    result: dict[str, Any] = {}
    for c in DEFINITIONS:
        if c not in df or c in AMBIGUOUS:
            continue
        result[c] = df[c]
        if c not in CATEGORICAL:
            result[f"{c}__missing"] = df[c].isna().astype(float)
        if c in NUMERIC:
            result[f"{c}__log1p"] = np.log1p(df[c])
    dates = df["invitation_date"].dt
    result.update(invitation_month=dates.month.astype(str), invitation_quarter=dates.quarter.astype(str),
                  invitation_weekday=dates.dayofweek.astype(str), year_trend=dates.year - 2020,
                  month_sin=np.sin(2 * np.pi * (dates.month - 1) / 12),
                  month_cos=np.cos(2 * np.pi * (dates.month - 1) / 12))
    if "territory" in df:
        result["unknown_territory"] = df["territory"].isna().astype(float)
    if "age_band" in df:
        order = {"<25": 0, "< 25": 0, **{f"{a}-{a+4}": (a-20)//5 for a in range(25, 75, 5)}, "75+": 11}
        result["age_band_order"] = df["age_band"].map(order).astype(float)
        if "months_since_last_screen" in df:
            result["age_band_order_x_screen_recency"] = result["age_band_order"] * df["months_since_last_screen"]
    if "prior_screen_count_5y" in df:
        x = df["prior_screen_count_5y"]
        result["screen_count_category"] = pd.Series(np.select([x.isna(), x.eq(0), x.eq(1)],
                                                       ["missing", "zero", "one"], default="multiple"), index=df.index)
    if "months_since_last_screen" in df:
        x = df["months_since_last_screen"]
        result["screen_recency_bin"] = pd.cut(x, [-np.inf, 24, 36, np.inf], labels=["<=24", "24-36", ">36"]).astype(object)
        result["screen_gap_24m"] = x - 24
    for c in DOMAINS:
        if c in df:
            result[f"{c}__any"] = (df[c] > 0).astype(float).where(df[c].notna())
            result[f"{c}__annualized"] = df[c] * 12 / (36 if "36m" in c else 24)
    if set(DOMAINS).issubset(df):
        values = df[list(DOMAINS)]
        result["active_prevention_domains"] = values.gt(0).sum(axis=1).astype(float).where(values.notna().all(axis=1))
        result["missing_prevention_domains"] = values.isna().sum(axis=1).astype(float)
        if "healthcare_contact_12m_cnt" in df:
            rate = pd.concat([result[f"{c}__annualized"] for c in DOMAINS], axis=1).sum(axis=1, min_count=4)
            result["prevention_contact_intensity_ratio"] = safe_ratio(rate, df["healthcare_contact_12m_cnt"])
    for group in ("gp", "gyn"):
        a, b = f"{group}_registered", f"{group}_prevention_24m_cnt"
        if a in df and b in df:
            result[f"{group}_registered_no_prevention"] = ((df[a] == 1) & (df[b] == 0)).astype(float).where(df[a].notna() & df[b].notna())
    if {"gp_registered", "gyn_registered"}.issubset(df):
        result["registration_count"] = df["gp_registered"] + df["gyn_registered"]
    if {"prior_screen_count_5y", "first_invitation"}.issubset(df):
        result["screen_count_x_first_invitation"] = df["prior_screen_count_5y"] * df["first_invitation"]
    return pd.DataFrame(result, index=df.index)


def build_patient_history(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    work = df.sort_values(["person_id", "invitation_date", "invitation_id"]).copy()
    group = work.groupby("person_id", sort=False)
    count = group.cumcount().astype(float)
    interval = group["invitation_date"].diff().dt.total_seconds() / 86400
    elapsed = (work["invitation_date"] - group["invitation_date"].transform("first")).dt.total_seconds() / 86400
    result = pd.DataFrame({
        "observed_prior_rows": count,
        "days_since_previous_observed_invitation": interval,
        "days_since_first_observed_invitation": elapsed.where(count > 0),
        "insufficient_observed_history": (count < 2).astype(float),
        "observed_invitation_density": count / (elapsed.clip(lower=365.25) / 365.25),
    }, index=work.index)
    prior_interval = interval.groupby(work["person_id"], sort=False).shift()
    for operation in ("mean", "min", "max", "std"):
        series = getattr(prior_interval.groupby(work["person_id"], sort=False).expanding(), operation)()
        result[f"prior_interval_{operation}"] = series.reset_index(level=0, drop=True)
    for c in SNAPSHOT_HISTORY:
        if c not in work:
            continue
        values = group[c]
        lag = values.shift()
        result[f"{c}__lag1"] = lag
        result[f"{c}__lag2"] = values.shift(2)
        result[f"{c}__delta"] = work[c] - lag
        result[f"{c}__change_per_year"] = safe_ratio(work[c] - lag, interval / 365.25)
        for operation in ("mean", "min", "max", "std"):
            series = getattr(lag.groupby(work["person_id"], sort=False).expanding(), operation)()
            result[f"{c}__past_{operation}"] = series.reset_index(level=0, drop=True)
    if cfg.historical_outcomes == "on":
        for h in HORIZONS:
            if TARGETS[h] not in work:
                continue
            # Availability is monotone within a person: an as-of join avoids quadratic histories.
            mature = work[work[TARGETS[h]].notna()].copy()
            mature["available"] = label_available_at(mature, h, cfg)
            mature = mature[mature["available"] < outcome_end(cfg)].copy()
            mature["nonattendance"] = 1 - mature[TARGETS[h]]
            mg = mature.groupby("person_id", sort=False)
            mature["mature_count"] = mg.cumcount() + 1.
            mature["nonattendance_fraction"] = mg["nonattendance"].cumsum() / mature["mature_count"]
            if mature.empty:
                for suffix in ("mature_count", "last_nonattendance", "nonattendance_fraction"):
                    result[f"historical_{h}d_{suffix}"] = 0. if suffix == "mature_count" else np.nan
                continue
            left = work[["person_id", "invitation_date"]].copy()
            left["_index"] = left.index
            merged = pd.merge_asof(
                left.sort_values("invitation_date"),
                mature[["person_id", "available", "mature_count", "nonattendance", "nonattendance_fraction"]].sort_values("available"),
                left_on="invitation_date", right_on="available", by="person_id", allow_exact_matches=False,
            ).set_index("_index")
            result[f"historical_{h}d_mature_count"] = merged["mature_count"].fillna(0)
            result[f"historical_{h}d_last_nonattendance"] = merged["nonattendance"]
            result[f"historical_{h}d_nonattendance_fraction"] = merged["nonattendance_fraction"]
    return result.reindex(df.index)


def build_features(df: pd.DataFrame, cfg: Config, names: Sequence[str]) -> pd.DataFrame:
    registry = build_feature_registry()
    unknown = set(names) - set(registry)
    if unknown:
        raise ContractError(f"Unregistered features: {unknown}")
    frames = [build_row_features(df)]
    if any(registry[n].required_history != "none" for n in names):
        frames.append(build_patient_history(df, cfg))
    result = pd.concat(frames, axis=1)
    missing = set(names) - set(result)
    if missing:
        raise ContractError(f"Training/inference feature schema mismatch: {sorted(missing)}")
    result = result[list(dict.fromkeys(names))].copy()
    for c in result:
        if registry[c].dtype == "numeric":
            result[c] = pd.to_numeric(result[c], errors="raise").astype(np.float32)
        else:
            result[c] = result[c].astype(object).where(pd.notna(result[c]), np.nan)
    if result.memory_usage(deep=True).sum() > cfg.memory_mb * 1024**2:
        raise ContractError("Feature frame exceeds configured memory ceiling.")
    return result


def project_horizon_probabilities(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ContractError("Probabilities must be a finite (n,3) array in [0,1].")
    # Three-point equal-weight PAV, vectorized across invitations.
    result = p.copy()
    bad01 = result[:, 0] < result[:, 1]
    result[bad01, :2] = result[bad01, :2].mean(axis=1, keepdims=True)
    bad12 = result[:, 1] < result[:, 2]
    result[bad12, 1:] = result[bad12, 1:].mean(axis=1, keepdims=True)
    pool_all = (bad01 & bad12) | (result[:, 0] < result[:, 1])
    result[pool_all] = p[pool_all].mean(axis=1, keepdims=True)
    return result


def positive_probability(estimator: Any, X: Any) -> np.ndarray:
    classes = np.asarray(estimator.classes_)
    values = estimator.predict_proba(X)
    positive = np.flatnonzero(classes == 1)
    if len(positive) == 1:
        return np.asarray(values[:, positive[0]], dtype=float)
    if len(classes) == 1 and classes[0] == 0:
        return np.zeros(values.shape[0])
    raise ContractError(f"Unsupported binary class mapping: {classes.tolist()}")


class MatrixGuard(TransformerMixin, BaseEstimator):
    def __init__(self, memory_mb: int = 4096, dense: bool = False):
        self.memory_mb = memory_mb
        self.dense = dense

    def fit(self, X: Any, y: Any = None) -> MatrixGuard:
        return self

    def transform(self, X: Any) -> Any:
        required = X.shape[0] * X.shape[1] * 4
        actual = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) if sparse.issparse(X) else X.nbytes
        if max(actual, required if self.dense else 0) > self.memory_mb * 1024**2:
            raise ContractError("Encoded matrix exceeds memory ceiling; reduce features/rows or increase --memory-mb.")
        return X.toarray().astype(np.float32) if self.dense and sparse.issparse(X) else X.astype(np.float32)


class NativeCategories(TransformerMixin, BaseEstimator):
    """Train-only category vocabulary; unknown and missing remain explicit categories."""

    def __init__(self, family: str, memory_mb: int = 4096):
        self.family = family
        self.memory_mb = memory_mb

    def fit(self, X: pd.DataFrame, y: Any = None) -> NativeCategories:
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.numeric_ = X.select_dtypes(include="number").columns.tolist()
        self.categories_ = {
            c: sorted(set(X[c].dropna().astype(str))) + ["__MISSING__", "__UNKNOWN__"]
            for c in X if c not in self.numeric_
        }
        if any(len(v) != len(set(v)) for v in self.categories_.values()):
            raise ContractError("Source categories collide with reserved native missing/unknown tokens.")
        self.medians_ = X[self.numeric_].median().fillna(0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if X.columns.tolist() != self.feature_names_in_.tolist():
            raise ContractError("Native feature schema/order differs from training.")
        result = X.copy()
        result[self.numeric_] = result[self.numeric_].fillna(self.medians_).astype(np.float32)
        for c, categories in self.categories_.items():
            values = result[c].where(result[c].notna(), "__MISSING__").astype(str)
            values = values.where(values.isin(categories), "__UNKNOWN__")
            result[c] = values if self.family == "catboost" else pd.Categorical(values, categories=categories)
        if result.memory_usage(deep=True).sum() > self.memory_mb * 1024**2:
            raise ContractError("Native categorical frame exceeds memory ceiling.")
        return result


def build_preprocessor(X: pd.DataFrame, family: str, cfg: Config) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X if c not in numeric]
    num_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
    ]
    if family in {"logistic", "mlp"}:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(num_steps), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2,
                                     sparse_output=True, dtype=np.float32)),
        ]), categorical),
    ], sparse_threshold=1.0, verbose_feature_names_out=True)


def optional_available(family: str) -> bool:
    return importlib.util.find_spec(family) is not None


def build_model(family: str, params: dict[str, Any], cfg: Config) -> Any:
    small = cfg.preset in {"smoke", "quick"}
    factories: dict[str, Any] = {
        "dummy": lambda: DummyClassifier(strategy="prior"),
        "logistic": lambda: LogisticRegression(max_iter=500, random_state=cfg.seed),
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=cfg.seed),
        "random_forest": lambda: RandomForestClassifier(n_estimators=30 if small else 200, max_depth=8,
                                                        min_samples_leaf=5, n_jobs=cfg.n_jobs, random_state=cfg.seed),
        "extra_trees": lambda: ExtraTreesClassifier(n_estimators=30 if small else 200, max_depth=8,
                                                    min_samples_leaf=5, n_jobs=cfg.n_jobs, random_state=cfg.seed),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=30 if small else 200,
                                                                        max_leaf_nodes=7, early_stopping=False, random_state=cfg.seed),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=35 if small else 300,
                                     early_stopping=False, batch_size="auto", random_state=cfg.seed),
    }
    if family in factories:
        model = factories[family]()
    elif family in OPTIONAL_MODELS:
        if not optional_available(family):
            raise ImportError(f"Optional model {family} unavailable; install requirements-optional.txt.")
        module = importlib.import_module(family)
        if family == "lightgbm":
            model = module.LGBMClassifier(n_estimators=40 if small else 300, n_jobs=cfg.n_jobs,
                                          random_state=cfg.seed, subsample_freq=1, verbosity=-1)
        elif family == "xgboost":
            model = module.XGBClassifier(n_estimators=40 if small else 300, tree_method="hist",
                                         n_jobs=cfg.n_jobs, random_state=cfg.seed, eval_metric="logloss")
        else:
            model = module.CatBoostClassifier(iterations=40 if small else 300, task_type="CPU",
                                               thread_count=cfg.n_jobs, random_seed=cfg.seed, verbose=False,
                                               allow_writing_files=False)
    else:
        raise ContractError(f"Unknown model family: {family}")
    model.set_params(**params)
    return model


def make_pipeline(X: pd.DataFrame, family: str, params: dict[str, Any], cfg: Config) -> Pipeline:
    if cfg.categorical_mode == "native" and family in OPTIONAL_MODELS:
        model = build_model(family, params, cfg)
        categories = X.select_dtypes(exclude="number").columns.tolist()
        if family == "catboost":
            model.set_params(cat_features=categories)
        if family == "xgboost":
            model.set_params(enable_categorical=True)
        return Pipeline([("preprocess", NativeCategories(family, cfg.memory_mb)), ("model", model)])
    return Pipeline([
        ("preprocess", build_preprocessor(X, family, cfg)),
        ("matrix", MatrixGuard(cfg.memory_mb, dense=family in {"hist_gradient_boosting", "catboost"})),
        ("model", build_model(family, params, cfg)),
    ])


def build_search_space(family: str, cfg: Config) -> list[dict[str, Any]]:
    if family == "dummy":
        return [{}]
    spaces = {
        "logistic": [{"C": c} for c in (.01, 1., 100.)],
        "decision_tree": [{"max_depth": d, "min_samples_leaf": leaf} for d in (3, 5, 8) for leaf in (5, 20)],
        "random_forest": [{"max_depth": d, "min_samples_leaf": leaf} for d in (6, 12, None) for leaf in (5, 25)],
        "extra_trees": [{"max_depth": d, "min_samples_leaf": leaf} for d in (6, 12, None) for leaf in (5, 25)],
        "hist_gradient_boosting": [{"learning_rate": lr, "max_leaf_nodes": leaves, "l2_regularization": l2}
                                    for lr in (.03, .1) for leaves in (7, 15, 31) for l2 in (0., 2.)],
        "mlp": [{"hidden_layer_sizes": shape, "alpha": alpha, "learning_rate_init": lr}
                for shape in ((32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32))
                for alpha in (1e-5, 1e-3, .1) for lr in (.0003, .003)],
        "lightgbm": [{"num_leaves": leaves, "min_child_samples": size, "learning_rate": lr,
                      "subsample": .8, "subsample_freq": 1, "colsample_bytree": .8}
                     for leaves in (7, 31, 63) for size in (20, 100) for lr in (.03, .1)],
        "xgboost": [{"max_depth": d, "min_child_weight": w, "learning_rate": lr,
                     "subsample": .8, "colsample_bytree": .8, "reg_lambda": 2.}
                    for d in (3, 5, 8) for w in (1, 10) for lr in (.03, .1)],
        "catboost": [{"depth": d, "learning_rate": lr, "l2_leaf_reg": l2}
                    for d in (4, 6, 8) for lr in (.03, .1) for l2 in (3, 10)],
    }
    if family not in spaces:
        raise ContractError(f"No search space for {family}")
    values = spaces[family]
    if cfg.preset == "smoke":
        return values[:1]
    if cfg.preset == "quick":
        if family == "mlp":
            return [{"hidden_layer_sizes": (32,), "alpha": .001},
                    {"hidden_layer_sizes": (64, 32), "alpha": .01}]
        return [values[0], values[-1]]
    if cfg.search_method == "random" or cfg.preset == "extensive":
        rng = np.random.default_rng(cfg.seed)
        return [values[i] for i in rng.permutation(len(values))]
    return values


@dataclass
class Calibrator:
    method: str
    estimator: Any

    def predict(self, p: np.ndarray) -> np.ndarray:
        if self.method == "sigmoid":
            clipped = np.clip(p, 1e-7, 1 - 1e-7)
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            return positive_probability(self.estimator, logits)
        return self.estimator.predict(p)


def fit_calibrator(p: np.ndarray, y: np.ndarray, method: str) -> Calibrator:
    minimum = 200 if method == "isotonic" else 30
    if len(y) < minimum or len(np.unique(y)) != 2:
        raise ContractError(f"{method} calibration requires >= {minimum} mature rows and both classes.")
    if method == "sigmoid":
        clipped = np.clip(p, 1e-7, 1 - 1e-7)
        estimator = LogisticRegression(C=1e6, max_iter=500).fit(np.log(clipped / (1 - clipped)).reshape(-1, 1), y)
    elif method == "isotonic":
        estimator = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    else:
        raise ContractError("Unknown calibration method.")
    return Calibrator(method, estimator)


@dataclass
class Component:
    horizon: int
    pipeline: Pipeline
    calibrator: Calibrator | None
    features: list[str]
    family: str
    params: dict[str, Any]
    fit_origin: str
    calibration_bounds: tuple[str, str] | None
    fit_key_hashes: set[str]
    fit_people_hashes: set[str]
    training_prevalence: float
    training_warnings: list[str]
    timings: dict[str, float]

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw = positive_probability(self.pipeline, X[self.features])
        return raw, self.calibrator.predict(raw) if self.calibrator else raw.copy()


def key_hashes(df: pd.DataFrame, column: str) -> set[str]:
    return {hashlib.sha256(str(v).encode()).hexdigest() for v in df[column]}


def fit_component(df: pd.DataFrame, X: pd.DataFrame, horizon: int, origin: pd.Timestamp,
                  family: str, params: dict[str, Any], cfg: Config,
                  disjoint_people: set[str] | None = None) -> Component:
    if not cfg.assume_t0_snapshots:
        raise ContractError("Training requires --assume-t0-snapshots: recording timestamps are unavailable.")
    base_origin = origin
    bounds = None
    calibration_mask = pd.Series(False, index=df.index)
    if cfg.calibration != "none":
        end = origin - pd.Timedelta(days=366 + cfg.reporting_lag_days)
        start = end - pd.Timedelta(days=cfg.calibration_days - 1)
        base_origin = start
        bounds = (start.isoformat(), end.isoformat())
        calibration_mask = df["invitation_date"].between(start, end) & observed_mask(df, horizon, cfg, origin)
    training = train_mask(df, horizon, base_origin, cfg)
    if disjoint_people:
        training &= ~df["person_id"].isin(disjoint_people)
        calibration_mask &= ~df["person_id"].isin(disjoint_people)
    if not training.any():
        raise ContractError("No mature training rows before the declared base fit origin.")
    y = (1 - df.loc[training, TARGETS[horizon]]).to_numpy(dtype=int)
    if family != "dummy" and len(np.unique(y)) != 2:
        raise ContractError("Single-class fit is only supported for the explicit dummy baseline.")
    pipeline = make_pipeline(X.loc[training], family, params, cfg)
    fit_started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=cfg.n_jobs):
        warnings.simplefilter("always")
        pipeline.fit(X.loc[training], y)
    fit_seconds = time.monotonic() - fit_started
    warning_messages = sorted({f"{type(w.message).__name__}: {w.message}" for w in caught})
    calibrator = None
    calibration_started = time.monotonic()
    if cfg.calibration != "none":
        if not calibration_mask.any():
            raise ContractError("Chronological calibration window has no mature outcomes.")
        probabilities = positive_probability(pipeline, X.loc[calibration_mask])
        cy = (1 - df.loc[calibration_mask, TARGETS[horizon]]).to_numpy(dtype=int)
        calibrator = fit_calibrator(probabilities, cy, cfg.calibration)
    used = training | calibration_mask
    return Component(horizon, pipeline, calibrator, X.columns.tolist(), family, params,
                     base_origin.isoformat(), bounds, key_hashes(df.loc[used], "invitation_id"),
                     key_hashes(df.loc[used], "person_id"), float(y.mean()), warning_messages,
                     {"base_fit_seconds": fit_seconds, "calibration_seconds": time.monotonic()-calibration_started if calibrator else 0.})


def evaluate_probabilities(y: np.ndarray, p: np.ndarray, baseline: float | None = None) -> dict[str, Any]:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ContractError("Invalid probability metric inputs.")
    if not len(y):
        return {"n": 0, "reason": "empty eligible cohort", "log_loss": None, "brier": None, "roc_auc": None}
    if not set(np.unique(y)).issubset({0, 1}):
        raise ContractError("Metrics require nonattendance labels in {0,1}.")
    two_classes = len(np.unique(y)) == 2
    pred = (p >= .5).astype(int)
    reliability = []
    for b in range(10):
        mask = (np.minimum((p * 10).astype(int), 9) == b)
        n = int(mask.sum())
        if n:
            rate = float(y[mask].mean())
            reliability.append({"left": b / 10, "right": (b + 1) / 10, "n": n,
                                "predicted": float(p[mask].mean()), "observed": rate,
                                "observed_wilson": wilson(int(y[mask].sum()), n)})
    metrics: dict[str, Any] = {
        "n": len(y), "prevalence_nonattendance": float(y.mean()),
        "no_skill_ap_nonattendance": float(y.mean()), "no_skill_ap_attendance": float(1 - y.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else None,
        "average_precision_nonattendance": float(average_precision_score(y, p)) if two_classes else None,
        "average_precision_attendance": float(average_precision_score(1 - y, 1 - p)) if two_classes else None,
        "undefined_reason": None if two_classes else "single-class cohort; AUC/AP not interpreted",
        "threshold": .5, "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if two_classes else None,
        "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "reliability": reliability,
        "ece_10_equal_width": sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in reliability) / len(y),
        "probability_quantiles": np.quantile(p, [0, .05, .25, .5, .75, .95, 1]).tolist(),
    }
    order = np.argsort(-p, kind="stable")
    metrics["ranking"] = {}
    for fraction in (.05, .1, .2):
        selected = y[order[:max(1, math.ceil(len(y) * fraction))]]
        precision = float(selected.mean())
        metrics["ranking"][str(fraction)] = {
            "n": len(selected), "precision": precision, "recall": float(selected.sum() / y.sum()) if y.sum() else None,
            "lift": precision / y.mean() if y.mean() else None,
            "caveat": "Risk ranking, not intervention benefit; ties resolved by original row order.",
        }
    if baseline is not None:
        baseline_p = np.full(len(y), baseline)
        metrics["baseline_log_loss"] = float(log_loss(y, baseline_p, labels=[0, 1]))
        metrics["baseline_brier"] = float(brier_score_loss(y, baseline_p))
        metrics["log_loss_improvement"] = metrics["baseline_log_loss"] - metrics["log_loss"]
        metrics["brier_improvement"] = metrics["baseline_brier"] - metrics["brier"]
    return metrics


def wilson(successes: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [center - margin, center + margin]


def vector_metrics(df: pd.DataFrame, raw: np.ndarray, calibrated: np.ndarray,
                   final: np.ndarray, baselines: Sequence[float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in (("raw", raw), ("calibrated", calibrated), ("final", final)):
        per_horizon = {str(h): evaluate_probabilities((1 - df[TARGETS[h]]).to_numpy(dtype=int),
                                                     values[:, j], baselines[j]) for j, h in enumerate(HORIZONS)}
        violation = np.maximum(values[:, 1:] - values[:, :-1], 0)
        result[name] = {
            "horizons": per_horizon,
            "macro_log_loss": float(np.mean([v["log_loss"] for v in per_horizon.values()])) if len(df) else None,
            "violation_frequency": float((violation > 1e-12).any(axis=1).mean()) if len(df) else None,
            "mean_positive_violation": float(violation.sum(axis=1).mean()) if len(df) else None,
            "max_violation": float(violation.max()) if len(df) else None,
        }
    result["mean_absolute_projection_change"] = float(np.abs(final - calibrated).mean()) if len(df) else None
    return result


def bootstrap_indices(people: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    ids = np.asarray(people)
    unique, counts = np.unique(ids, return_counts=True)
    if not len(unique):
        return np.array([], dtype=int)
    groups = np.split(np.argsort(ids, kind="stable"), np.cumsum(counts)[:-1])
    return np.concatenate([groups[i] for i in rng.integers(0, len(unique), len(unique))])


def bootstrap_by_patient(df: pd.DataFrame, p: np.ndarray, repeats: int, seed: int,
                         comparison: np.ndarray | None = None) -> dict[str, Any]:
    if df["person_id"].nunique() < 10 or repeats < 2:
        return {"status": "insufficient patients or disabled", "valid_repeats": 0}
    rng = np.random.default_rng(seed)
    codes, unique = pd.factorize(df["person_id"], sort=True)
    y = np.column_stack([(1 - df[TARGETS[h]]).to_numpy(dtype=int) for h in HORIZONS])
    samples, differences = [], []
    per_horizon: dict[str, dict[str, list[float]]] = {
        str(h): {"log_loss": [], "brier": [], "roc_auc": [], "ap_nonattendance": []} for h in HORIZONS}
    degenerate = {str(h): 0 for h in HORIZONS}
    for _ in range(repeats):
        patient_counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = patient_counts[codes]
        losses = []
        for j, h in enumerate(HORIZONS):
            ll = float(log_loss(y[:, j], p[:, j], labels=[0, 1], sample_weight=weights))
            losses.append(ll)
            stats = per_horizon[str(h)]
            stats["log_loss"].append(ll)
            stats["brier"].append(float(brier_score_loss(y[:, j], p[:, j], sample_weight=weights)))
            if len(np.unique(y[weights > 0, j])) < 2:
                degenerate[str(h)] += 1
            else:
                stats["roc_auc"].append(float(roc_auc_score(y[:, j], p[:, j], sample_weight=weights)))
                stats["ap_nonattendance"].append(float(average_precision_score(y[:, j], p[:, j], sample_weight=weights)))
        samples.append(float(np.mean(losses)))
        if comparison is not None:
            other = [log_loss(y[:, j], comparison[:, j], labels=[0, 1], sample_weight=weights) for j in range(3)]
            differences.append(float(np.mean(losses) - np.mean(other)))
    return {"status": "patient-cluster percentile CI; sample-specific", "valid_repeats": len(samples),
            "macro_log_loss_95ci": np.quantile(samples, [.025, .975]).tolist(),
            "paired_log_loss_difference_95ci": np.quantile(differences, [.025, .975]).tolist() if differences else None,
            "horizons": {h: {name: {"valid_repeats": len(v), "percentile_95ci": np.quantile(v, [.025, .975]).tolist() if v else None}
                             for name, v in stats.items()} for h, stats in per_horizon.items()},
            "single_class_resamples_auc_ap_excluded": degenerate,
            "implementation": "Whole-patient multiplicity weights; all invitations of a sampled person share its weight."}


def interval_labels(df: pd.DataFrame) -> np.ndarray:
    validate_targets(df)
    if df[list(TARGETS.values())].isna().any().any():
        raise ContractError("Interval labels require all three outcomes.")
    return np.select([df[TARGETS[90]].eq(1), df[TARGETS[180]].eq(1), df[TARGETS[365]].eq(1)], [0, 1, 2], default=3)


def interval_probabilities(masses: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    full = np.zeros((len(masses), 4))
    for j, c in enumerate(classes):
        if c not in (0, 1, 2, 3):
            raise ContractError("Invalid interval class.")
        full[:, c] = masses[:, j]
    if not np.isfinite(full).all() or (full < 0).any() or not np.allclose(full.sum(axis=1), 1):
        raise ContractError("Invalid interval probability masses.")
    return np.column_stack((full[:, 1:].sum(axis=1), full[:, 2:].sum(axis=1), full[:, 3]))


def run_interval_benchmark(df: pd.DataFrame, X: pd.DataFrame, folds: Sequence[Fold],
                           cfg: Config, deadline: float) -> list[dict[str, Any]]:
    records = []
    for params in build_search_space("logistic", cfg):
        if time.monotonic() >= deadline:
            records.append({"status": "not launched", "reason": "Shared search deadline", "params": params})
            break
        started = time.monotonic()
        predicted, indices = [], []
        for fold in folds:
            tr = (df["invitation_date"] < fold.start) & common_mask(df, cfg, fold.start)
            va = df["invitation_date"].between(fold.start, fold.end) & common_mask(df, cfg, fold.selection_deadline)
            y = interval_labels(df.loc[tr])
            if len(np.unique(y)) < 2:
                raise ContractError("Joint interval benchmark requires at least two mature interval classes.")
            pipeline = make_pipeline(X.loc[tr], "logistic", params, cfg)
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=cfg.n_jobs):
                warnings.simplefilter("always")
                pipeline.fit(X.loc[tr], y)
            p = interval_probabilities(pipeline.predict_proba(X.loc[va]), pipeline.classes_)
            predicted.append(p)
            indices.extend(np.flatnonzero(va))
            records.append({"status": "fold", "params": params, "origin": fold.start, "rows_train": int(tr.sum()),
                            "rows_validation": int(va.sum()), "warnings": [str(w.message) for w in caught]})
        matrix = np.vstack(predicted)
        cohort = df.iloc[indices]
        records.append({"status": "success", "family": "joint_interval_logistic", "params": params,
                        "features": X.columns.tolist(), "duration_seconds": time.monotonic()-started,
                        "policy": "Common 365-day-mature training; no calibration. Compare independent cohort-policy common for controlled training.",
                        "macro_log_loss": float(np.mean([log_loss(1-cohort[TARGETS[h]], matrix[:, j], labels=[0, 1])
                                                        for j, h in enumerate(HORIZONS)])),
                        "horizon_metrics": {str(h): evaluate_probabilities((1-cohort[TARGETS[h]]).to_numpy(dtype=int), matrix[:, j])
                                            for j, h in enumerate(HORIZONS)},
                        "validation_rows_fingerprint": fingerprint(cohort)})
    return records


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temp = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, default=json_default, allow_nan=False)
            handle.write("\n")
        except (TypeError, ValueError):
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)


def append_trial(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=json_default, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def environment_record() -> dict[str, str]:
    import sklearn
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def write_schema_outputs(df: pd.DataFrame, cfg: Config) -> None:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = schema_report(df)
    atomic_json(out / "schema_audit.json", report)
    pd.DataFrame(report).to_csv(out / "schema_audit.csv", index=False)
    atomic_json(out / "feature_registry.json", [asdict(f) for f in build_feature_registry().values()])
    atomic_json(out / "environment.json", environment_record())
    summary = {
        "rows": len(df), "patients": int(df["person_id"].nunique()),
        "date_min": df["invitation_date"].min().isoformat(), "date_max": df["invitation_date"].max().isoformat(),
        "invitation_dates": int(df["invitation_date"].nunique()),
        "rows_per_patient_distribution": df.groupby("person_id").size().value_counts().sort_index().to_dict(),
        "first_observed_input_rows": int(df["person_id"].nunique()),
        "source_first_invitation_rows": int(df["first_invitation"].eq(1).sum()) if "first_invitation" in df else None,
        "duplicate_ids": 0, "same_person_time_ambiguities": 0, "label_contradictions": 0,
        "sampling": cfg.provenance, "observed_history_incomplete": True,
        "outcome_data_end": cfg.outcome_data_end, "reporting_lag_days": cfg.reporting_lag_days,
        "unresolved": ["Outcome end must be independently verified.", "Source invitation counters disabled.",
                       "T0 predictor recording/backfill latency unknown.", "Sample selection mechanism unknown.",
                       "No observed prior row does not mean first-ever invitation."],
        "source_scope_not_verified_by_sample": {"documented_invitations": 1344556, "documented_people": 508527,
                                               "documented_invitation_years": "2020-2024"},
    }
    atomic_json(out / "inspection.json", summary)
    text = [
        "# Schema and sample audit", "",
        f"Rows: {len(df)}. Distinct people: {summary['patients']}. Invitation dates: {summary['date_min']} to {summary['date_max']}.",
        "", "These are sample-specific descriptions, not population estimates.",
        "The documented cohort requires continuous VZP insurance in 2015-2025 and applies anonymization filtering.",
        "Insurance eligibility is not verification of complete outcome ascertainment.",
        "The dictionary describes 1,344,556 invitations and 508,527 people; the full CSV was not inspected.",
        "", "## Unresolved definitions", *["- " + x for x in summary["unresolved"]],
        "", "All identifiers, current labels, unknown columns, and ambiguous source counters are excluded from predictors.",
        "The source NEVER flag is described as no screening observed in the documented five-year lookback, not lifetime nonattendance.",
        "See schema_audit.csv for every delivered column and feature_registry.json for feature provenance.",
    ]
    (out / "schema_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def model_families(cfg: Config) -> tuple[list[str], list[dict[str, str]]]:
    requested = list(MANDATORY_MODELS + OPTIONAL_MODELS) if cfg.models == "all-available" else list(dict.fromkeys(cfg.models.split(",")))
    unknown = set(requested) - set(MANDATORY_MODELS + OPTIONAL_MODELS)
    if unknown:
        raise ContractError(f"Unknown requested model families: {unknown}")
    available, skipped = [], []
    for family in ["dummy", *[f for f in requested if f != "dummy"]]:
        if family in OPTIONAL_MODELS and not optional_available(family):
            message = f"Install requirements-optional.txt to enable {family}."
            if cfg.strict_optional:
                raise ContractError(message)
            skipped.append({"family": family, "status": "skipped", "reason": message})
        else:
            available.append(family)
    return available, skipped


def trial_identity(family: str, feature_set: str, params: dict[str, Any], cfg: Config) -> str:
    payload = json.dumps([family, feature_set, params, cfg.calibration, cfg.cohort_policy, cfg.seed], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_trial(df: pd.DataFrame, X: pd.DataFrame, folds: Sequence[Fold], family: str,
              params: dict[str, Any], cfg: Config) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[Component]]:
    indices, raw_parts, calibrated_parts, fold_records = [], [], [], []
    last_components: list[Component] = []
    for fold in folds:
        mask = df["invitation_date"].between(fold.start, fold.end) & common_mask(df, cfg, fold.selection_deadline)
        idx = np.flatnonzero(mask)
        components, raw, calibrated = [], [], []
        for h in HORIZONS:
            component = fit_component(df, X, h, fold.start, family, params, cfg)
            r, c = component.predict(X.loc[mask])
            components.append(component)
            raw.append(r)
            calibrated.append(c)
            actual_training = train_mask(df, h, iso_date(component.fit_origin), cfg)
            audit = audit_split(df, fold, h, cfg, actual_training)
            audit["base_fit_origin"] = component.fit_origin
            audit["calibration_bounds"] = component.calibration_bounds
            audit["training_prevalence"] = component.training_prevalence
            audit["timings"] = component.timings
            yy = (1-df.loc[mask, TARGETS[h]]).to_numpy(dtype=int)
            audit["validation_raw_log_loss"] = float(log_loss(yy, r, labels=[0, 1]))
            audit["validation_calibrated_log_loss"] = float(log_loss(yy, c, labels=[0, 1]))
            audit["warnings"] = component.training_warnings
            fold_records.append(audit)
        indices.append(idx)
        raw_parts.append(np.column_stack(raw))
        calibrated_parts.append(np.column_stack(calibrated))
        last_components = components
    all_indices = np.concatenate(indices)
    if len(set(all_indices)) != len(all_indices):
        raise ContractError("Validation windows overlap; rows must not be scored twice.")
    raw_matrix, calibrated_matrix = np.vstack(raw_parts), np.vstack(calibrated_parts)
    final = project_horizon_probabilities(calibrated_matrix)
    metric = vector_metrics(df.iloc[all_indices], raw_matrix, calibrated_matrix, final,
                           [c.training_prevalence for c in last_components])
    # Each fold has its own training prior; use identical-row fold baselines for pooled evidence.
    for stage in ("raw", "calibrated", "final"):
        for j, h in enumerate(HORIZONS):
            baseline_losses, baseline_briers = [], []
            for fold_index, fold in enumerate(folds):
                mask = df["invitation_date"].between(fold.start, fold.end) & common_mask(df, cfg, fold.selection_deadline)
                prior = fold_records[fold_index*3+j]["training_prevalence"]
                y = (1 - df.loc[mask, TARGETS[h]]).to_numpy(dtype=int)
                baseline_losses.extend(-np.log(np.clip(np.where(y == 1, prior, 1-prior), 1e-15, 1)))
                baseline_briers.extend((y-prior)**2)
            m = metric[stage]["horizons"][str(h)]
            m["baseline_log_loss"] = float(np.mean(baseline_losses))
            m["baseline_brier"] = float(np.mean(baseline_briers))
            m["log_loss_improvement"] = m["baseline_log_loss"] - m["log_loss"]
            m["brier_improvement"] = m["baseline_brier"] - m["brier"]
    record = {
        "family": family, "params": params, "features": X.columns.tolist(), "folds": fold_records,
        "metrics": metric, "warnings": sorted({w for f in fold_records for w in f["warnings"]}),
        "fit_seconds": sum(f["timings"]["base_fit_seconds"] for f in fold_records),
        "calibration_seconds": sum(f["timings"]["calibration_seconds"] for f in fold_records),
        "validation_rows_fingerprint": fingerprint(df.iloc[all_indices]),
        "selection_losses": {str(h): metric["calibrated"]["horizons"][str(h)]["log_loss"] for h in HORIZONS},
        "preprocessing": "fold-local median+missing indicators; train-only rare OHE; scaling only linear/MLP",
    }
    predictions = np.stack((raw_matrix, calibrated_matrix), axis=0)
    return record, all_indices, predictions, last_components


def save_bundle(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
    try:
        joblib.dump(bundle, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_bundle(path: str) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("version") != VERSION:
        raise ContractError("Incompatible bundle version; retrain with this schema/code version.")
    if tuple(bundle.get("horizons", ())) != HORIZONS:
        raise ContractError("Bundle horizon order is incompatible.")
    saved = bundle.get("environment", {})
    current = environment_record()
    for key in ("scikit_learn", "numpy", "pandas"):
        if saved.get(key) != current[key]:
            raise ContractError(f"Incompatible {key} version: saved {saved.get(key)}, current {current[key]}. Restore the recorded environment.")
    if saved.get("code_sha256") != current["code_sha256"]:
        raise ContractError("Implementation hash changed since artifact creation; use its recorded code version or retrain.")
    return bundle


def bundle_metadata(cfg: Config, manifest: dict[str, Any], components: list[Component],
                    purpose: str) -> dict[str, Any]:
    registry = build_feature_registry()
    names = list(dict.fromkeys(f for c in components for f in c.features))
    return {
        "version": VERSION, "horizons": HORIZONS, "output_names": OUTPUTS,
        "positive_class": "1 = no recorded screening by horizon", "not_mutually_exclusive": True,
        "coherence": "deterministic equal-weight nonincreasing PAV after component calibration",
        "config": asdict(cfg), "config_hash": config_hash(cfg), "manifest": manifest,
        "environment": environment_record(), "components": components, "purpose": purpose,
        "feature_definitions": [asdict(registry[n]) for n in names],
        "history_strategy": "External earlier input predictor snapshots; no raw history embedded",
        "history_required": any(registry[n].required_history != "none" for n in names),
        "history_fields": sorted({s for n in names for s in registry[n].source_columns}),
        "schema_aliases": {**ALIASES, **cfg.schema_mapping},
        "partial_history_policy": cfg.history_mode,
    }


def compare_models(df: pd.DataFrame, cfg: Config, save_model: bool = False,
                   resume: bool = False) -> dict[str, Any]:
    start_time = time.monotonic()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = reserve_test(df, cfg)
    folds = make_temporal_folds(df, cfg, manifest)
    development = df[df["invitation_date"] < iso_date(manifest["test_start"])]
    identity = {
        "config_hash": config_hash(cfg), "data_development_fingerprint": fingerprint(development, development.columns.tolist()),
        "test_keys_fingerprint": manifest["test_fingerprint"], "environment": environment_record(),
        "folds": [{"start": f.start, "end": f.end, "deadline": f.selection_deadline} for f in folds],
    }
    identity = json.loads(json.dumps(identity, default=json_default))
    ledger = out / "trials.jsonl"
    if (out / "run_identity.json").exists():
        previous = json.loads((out / "run_identity.json").read_text())
        if not resume:
            raise ContractError("Existing experiment; use --resume or a new output directory.")
        if previous != identity:
            raise ContractError("Incompatible resume: data, folds, configuration, code or library versions changed.")
    elif resume:
        raise ContractError("No existing run to resume.")
    atomic_json(out / "test_manifest.json", manifest)
    atomic_json(out / "run_identity.json", identity)
    families, skipped = model_families(cfg)
    feature_sets = list(dict.fromkeys(cfg.feature_sets.split(",")))
    resolved, availability = {}, {}
    for preset in feature_sets:
        resolved[preset], availability[preset] = resolve_feature_set(preset, df, cfg)
    atomic_json(out / "feature_availability.json", availability)
    unique_sets = list(dict.fromkeys(tuple(resolved[p]) for p in feature_sets))
    scheduled_sets = [next(p for p in feature_sets if tuple(resolved[p]) == names) for names in unique_sets]
    aliases = {p: [alias for alias in feature_sets if resolved[alias] == resolved[p]] for p in scheduled_sets}
    atomic_json(out / "feature_set_aliases.json", aliases)
    feature_start = time.monotonic()
    union = list(dict.fromkeys(n for names in resolved.values() for n in names))
    X = build_features(df, cfg, union)
    feature_duration = time.monotonic() - feature_start
    successful: dict[str, dict[str, Any]] = {}
    attempted: set[str] = set()
    if resume and ledger.exists():
        lines = ledger.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if i == len(lines)-1:
                    raise ContractError("Truncated trial ledger; preserve it and remove only its incomplete last line before resume.") from exc
                raise ContractError("Corrupt trial ledger.") from exc
            if record["status"] in {"success", "failed", "skipped"}:
                attempted.add(record["trial_id"])
            if record["status"] == "success":
                successful[record["trial_id"]] = record
    plan = []
    spaces = {family: build_search_space(family, cfg) for family in families}
    # Round-robin exploration before deeper search; no family owns the full deadline.
    for depth in range(max(len(space) for space in spaces.values())):
        for preset in scheduled_sets:
            for family in families:
                if depth < len(spaces[family]):
                    params = spaces[family][depth]
                    tid = trial_identity(family, preset, params, cfg)
                    plan.append((tid, family, preset, params))
    search_start = time.monotonic()
    deadline = search_start + cfg.budget
    for tid, family, preset, params in plan:
        if tid in attempted:
            continue
        if time.monotonic() >= deadline or len(attempted) >= cfg.max_trials:
            break
        record = {
            "run_id": identity["config_hash"][:16], "trial_id": tid, "family": family,
            "feature_set": preset, "params": params, "seed": cfg.seed, "config_hash": config_hash(cfg),
            "feature_set_aliases": aliases[preset],
            "resources": {"threads": cfg.n_jobs, "memory_mb": cfg.memory_mb, "per_fit": "soft deadline; bounded estimator iterations",
                          "search_budget_seconds": cfg.budget},
            "cohort_policy": cfg.cohort_policy, "status": "running",
        }
        append_trial(ledger, record)
        trial_start = time.monotonic()
        try:
            result, idx, predictions, components = run_trial(df, X[resolved[preset]], folds, family, params, cfg)
            record.update(result)
            record["status"] = "success"
            checkpoint = out / "checkpoints" / tid
            checkpoint.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=checkpoint, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
            np.savez_compressed(temporary, indices=idx, probabilities=predictions)
            os.replace(temporary, checkpoint / "validation.npz")
            save_bundle(checkpoint / "last_fold.joblib", bundle_metadata(cfg, manifest, components, "held-out development diagnostics"))
        except (ValueError, MemoryError, ImportError, FloatingPointError, TimeoutError) as exc:
            record.update(status="failed", exception=f"{type(exc).__name__}: {exc}")
            print(f"Trial {tid} failed: {record['exception']}", file=sys.stderr)
        except KeyboardInterrupt:
            record.update(status="interrupted", exception="KeyboardInterrupt; completed earlier trials preserved",
                          duration_seconds=time.monotonic() - trial_start)
            append_trial(ledger, record)
            raise
        record["duration_seconds"] = time.monotonic() - trial_start
        append_trial(ledger, record)
        attempted.add(tid)
        if record["status"] == "success":
            successful[tid] = record
        print(f"{family}/{preset} {record['status']} {record['duration_seconds']:.3f}s", file=sys.stderr)
    for tid, family, preset, params in plan:
        if tid not in attempted:
            skipped.append({"family": family, "feature_set": preset, "trial_id": tid,
                            "status": "not launched", "reason": "Global search deadline or trial limit"})
    atomic_json(out / "skipped_models_trials.json", skipped)
    if not successful:
        raise ContractError("No successful validation trial; see the persisted trial ledger.")
    if cfg.joint_interval and not (resume and (out / "joint_interval_validation.json").exists()):
        if not cfg.assume_t0_snapshots:
            raise ContractError("Joint interval fitting requires --assume-t0-snapshots.")
        atomic_json(out / "joint_interval_validation.json", run_interval_benchmark(df, X[resolved[feature_sets[0]]], folds, cfg, deadline))
    search_duration = time.monotonic() - search_start
    cohorts = {r["validation_rows_fingerprint"] for r in successful.values()}
    if len(cohorts) != 1:
        raise ContractError("Candidate validation cohorts differ; refusing to pool/rank unmatched rows.")
    records = sorted(successful.values(), key=lambda r: (r["metrics"]["final"]["macro_log_loss"], r["trial_id"]))
    rows = [{
        "trial_id": r["trial_id"], "family": r["family"], "feature_set": r["feature_set"],
        "macro_final_log_loss": r["metrics"]["final"]["macro_log_loss"],
        "macro_raw_log_loss": r["metrics"]["raw"]["macro_log_loss"], "duration_seconds": r["duration_seconds"],
        **{f"log_loss_{h}d": r["selection_losses"][str(h)] for h in HORIZONS},
    } for r in records]
    pd.DataFrame(rows).to_csv(out / "validation_leaderboard.csv", index=False)
    choices = {str(h): min(records, key=lambda r: (r["selection_losses"][str(h)], r["trial_id"]))["trial_id"] for h in HORIZONS}
    per_family = {family: {str(h): min([r for r in records if r["family"] == family],
                                     key=lambda r: (r["selection_losses"][str(h)], r["trial_id"])) for h in HORIZONS}
                  for family in families if any(r["family"] == family for r in records)}
    atomic_json(out / "best_tested_by_family.json", {
        family: {h: {"trial_id": r["trial_id"], "params": r["params"], "feature_set": r["feature_set"],
                     "validation_log_loss": r["selection_losses"][h]} for h, r in horizons.items()}
        for family, horizons in per_family.items()
    })
    by_id = {r["trial_id"]: r for r in records}
    selection = {
        "version": VERSION, "rule": "Independently minimize pooled chronological common-cohort component log loss; equal horizon weights; ties by trial ID.",
        "coherence_not_optimized_on_test": "Equal-weight PAV fixed in advance; evaluate complete deployed vector on validation.",
        "choices": {h: {"trial_id": tid, "family": by_id[tid]["family"], "params": by_id[tid]["params"],
                        "feature_set": by_id[tid]["feature_set"], "features": by_id[tid]["features"]} for h, tid in choices.items()},
        "config": asdict(cfg), "config_hash": config_hash(cfg), "manifest": manifest,
        "data_development_fingerprint": identity["data_development_fingerprint"],
        "environment": environment_record(), "frozen_candidates": list(successful),
    }
    selected_raw, selected_calibrated, selected_components = [], [], []
    indices = None
    for j, h in enumerate(HORIZONS):
        tid = choices[str(h)]
        with np.load(out / "checkpoints" / tid / "validation.npz") as saved:
            if indices is None:
                indices = saved["indices"].copy()
            elif not np.array_equal(indices, saved["indices"]):
                raise ContractError("Validation checkpoint row mismatch.")
            selected_raw.append(saved["probabilities"][0, :, j])
            selected_calibrated.append(saved["probabilities"][1, :, j])
        diagnostic = load_bundle(str(out / "checkpoints" / tid / "last_fold.joblib"))
        selected_components.append(diagnostic["components"][j])
    assert indices is not None
    raw, calibrated = np.column_stack(selected_raw), np.column_stack(selected_calibrated)
    final = project_horizon_probabilities(calibrated)
    metrics = vector_metrics(df.iloc[indices], raw, calibrated, final, [c.training_prevalence for c in selected_components])
    for j, h in enumerate(HORIZONS):
        record = by_id[choices[str(h)]]
        priors = np.concatenate([np.full(f["validation_rows"], f["training_prevalence"]) for f in record["folds"] if f["horizon"] == h])
        yy = (1-df.iloc[indices][TARGETS[h]]).to_numpy(dtype=int)
        baseline_loss, baseline_brier = float(log_loss(yy, priors, labels=[0, 1])), float(brier_score_loss(yy, priors))
        for stage in ("raw", "calibrated", "final"):
            m = metrics[stage]["horizons"][str(h)]
            m.update(baseline_log_loss=baseline_loss, baseline_brier=baseline_brier,
                     log_loss_improvement=baseline_loss-m["log_loss"], brier_improvement=baseline_brier-m["brier"])
    metrics["baseline_note"] = "Exact horizon/fold-specific base-training priors on identical validation rows."
    atomic_json(out / "selected_validation_metrics.json", metrics)
    selection["validation_final_macro_log_loss"] = metrics["final"]["macro_log_loss"]
    atomic_json(out / "selection.json", selection)
    diagnostic_bundle = bundle_metadata(cfg, manifest, selected_components, "held-out development diagnostics")
    diagnostic_bundle["diagnostic_fold"] = {"start": folds[-1].start.isoformat(), "end": folds[-1].end.isoformat(),
                                          "deadline": folds[-1].selection_deadline.isoformat()}
    save_bundle(out / "validation_bundle.joblib", diagnostic_bundle)
    ablations = []
    for r in records:
        for base in records:
            if r["family"] != base["family"] or r["params"] != base["params"] or r["feature_set"] == base["feature_set"]:
                continue
            ablations.append({
                "family": r["family"], "reference_feature_set": base["feature_set"], "feature_set": r["feature_set"],
                "settings_fixed": True, "paired_rows": True, "params": r["params"],
                "final_macro_log_loss_difference": r["metrics"]["final"]["macro_log_loss"] - base["metrics"]["final"]["macro_log_loss"],
                "duration_difference_seconds": r["duration_seconds"] - base["duration_seconds"],
                "fold_log_loss_differences": [
                    a["validation_calibrated_log_loss"] - b["validation_calibrated_log_loss"]
                    for a, b in zip(r["folds"], base["folds"])],
                "fold_stability": "Per-horizon/fold paired differences; not a significance claim.",
            })
    atomic_json(out / "paired_feature_comparisons.json", ablations)
    plots_for_metrics(metrics, out, "validation")
    training_duration = 0.
    if save_model:
        fit_start = time.monotonic()
        train_selected(df, cfg, selection, out / "model_bundle.joblib")
        training_duration = time.monotonic() - fit_start
    timing = {"feature_seconds": feature_duration, "search_seconds": search_duration,
              "trial_base_fit_seconds": sum(r["fit_seconds"] for r in records),
              "trial_calibration_seconds": sum(r["calibration_seconds"] for r in records),
              "reporting_seconds": time.monotonic()-search_start-search_duration-training_duration,
              "refit_seconds": training_duration, "total_after_load_seconds": time.monotonic() - start_time,
              "budget_scope": "Search only; trial launch deadline is soft; an active bounded fit finishes.",
              "allocated_search_seconds": cfg.budget, "completed_trials": len(records), "skipped_trials": len(skipped)}
    atomic_json(out / "timings.json", timing)
    report = [
        "# Validation experiment", "", f"Provenance: {cfg.provenance}.", f"Completed trials: {len(records)}.",
        f"Allocated search budget: {cfg.budget:.1f}s. Actual search/reporting time: {timing['search_seconds']:.3f}s.",
        "Meaningful scheduled work can finish before the budget; no filler fits are run.",
        "All comparisons use the same fully observed validation invitations; the final test has not been scored.",
        "Component selection minimizes validation log loss independently; projection is fixed, not selected using test labels.",
        f"Selected bundle final validation macro log loss: {metrics['final']['macro_log_loss']:.6f}.",
        "", "## Interpretation and limitations",
        "Best tested configurations are not globally optimal. Feature additions are hypotheses, not guaranteed improvements.",
        "Metrics are sample-specific. Synthetic runs verify mechanics only and provide no empirical evidence about VZP.",
        "Predictor backfill latency is unobserved. A zero reporting lag is an assumption, not a verified property.",
        "Convergence warnings and all skipped/failed trials remain in the ledger.",
        "Feature comparisons with identical settings are recorded separately from independently retuned winners.",
        "See validation_leaderboard.csv, selection.json, selected_validation_metrics.json and paired_feature_comparisons.json.",
    ]
    (out / "experiment_report.md").write_text("\n".join(report) + "\n")
    return selection


def train_selected(df: pd.DataFrame, cfg: Config, selection: dict[str, Any], path: Path) -> dict[str, Any]:
    if selection.get("environment", {}).get("code_sha256") != environment_record()["code_sha256"]:
        raise ContractError("Selection code version differs; rerun the benchmark.")
    if selection["config_hash"] != config_hash(cfg):
        raise ContractError("Training configuration differs from the frozen selection; rerun benchmark for changed settings.")
    manifest = selection["manifest"]
    if fingerprint(df) != manifest["dataset_keys_fingerprint"]:
        raise ContractError("Training input invitation keys differ from the frozen selection manifest.")
    origin = iso_date(manifest["fit_origin"])
    development = df[df["invitation_date"] < origin]
    if fingerprint(development, development.columns.tolist()) != selection["data_development_fingerprint"]:
        raise ContractError("Development predictors/labels differ from the frozen selection input; rerun benchmark.")
    names = list(dict.fromkeys(n for choice in selection["choices"].values() for n in choice["features"]))
    X = build_features(df, cfg, names)
    components = []
    stress = []
    test_people = set(df.loc[df["invitation_date"].between(iso_date(manifest["test_start"]),
                                                          iso_date(manifest["test_end"])), "person_id"])
    for h in HORIZONS:
        choice = selection["choices"][str(h)]
        components.append(fit_component(df, X[choice["features"]], h, origin, choice["family"], choice["params"], cfg))
        if cfg.patient_disjoint_stress:
            stress.append(fit_component(df, X[choice["features"]], h, origin, choice["family"], choice["params"], cfg, test_people))
    bundle = bundle_metadata(cfg, manifest, components, "frozen final-test artifact")
    bundle["selection"] = selection
    if stress:
        bundle["patient_disjoint_components"] = stress
    save_bundle(path, bundle)
    return bundle


def prediction_matrices(bundle: dict[str, Any], df: pd.DataFrame,
                        cfg: Config | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    cfg = cfg or Config(**bundle["config"])
    components = bundle["components"]
    names = list(dict.fromkeys(n for c in components for n in c.features))
    X = build_features(df, cfg, names)
    raw, calibrated = [], []
    for c in components:
        r, p = c.predict(X)
        raw.append(r)
        calibrated.append(p)
    r, p = np.column_stack(raw), np.column_stack(calibrated)
    return r, p, project_horizon_probabilities(p), X


def predict_bundle(bundle: dict[str, Any], invitations: pd.DataFrame, history: pd.DataFrame | None = None,
                   data_as_of: str | None = None, partial_history: bool = False) -> pd.DataFrame:
    cfg = Config(**bundle["config"])
    if data_as_of is None:
        raise ContractError("Prediction requires --data-as-of to declare the available predictor snapshot cutoff.")
    cutoff = iso_date(data_as_of)
    if (invitations["invitation_date"] > cutoff).any():
        raise ContractError("Prediction invitation is after the declared data-as-of cutoff.")
    if bundle["history_required"] and history is None and not partial_history:
        raise ContractError("Model requires external past history. Supply --history-data or explicitly --partial-history.")
    if cfg.historical_outcomes == "on" and cutoff > outcome_end(cfg):
        raise ContractError("Historical-outcome bundle requires a newly verified observation policy for later scoring; retrain with updated configuration.")
    current = invitations.copy()
    n = len(current)
    if history is not None:
        if (history["invitation_date"] > cutoff).any():
            raise ContractError("History contains snapshots after --data-as-of.")
        overlap = set(current["invitation_id"]) & set(history["invitation_id"])
        if overlap:
            raise ContractError("History and current invitations contain duplicate invitation IDs.")
        current = pd.concat([current, history], ignore_index=True)
    current = normalize_schema(current, cfg)
    raw, calibrated, final, X = prediction_matrices(bundle, current, cfg)
    result = invitations[list(KEYS)].reset_index(drop=True).copy()
    for j, name in enumerate(OUTPUTS):
        result[name] = final[:n, j]
    for j, h in enumerate(HORIZONS):
        result[f"p_raw_nonattendance_{h}d"] = raw[:n, j]
        result[f"p_calibrated_nonattendance_{h}d"] = calibrated[:n, j]
        result[f"p_attendance_{h}d"] = 1 - final[:n, j]
    result["model_version"] = VERSION + ":" + bundle["environment"]["code_sha256"][:12]
    result["history_quality"] = "partial external/input history" if bundle["history_required"] else "snapshot-only"
    if "observed_prior_rows" in X:
        result["observed_prior_input_rows"] = X["observed_prior_rows"].iloc[:n].to_numpy()
    return result


def reject_fit_overlap(df: pd.DataFrame, components: Sequence[Component]) -> None:
    keys = key_hashes(df, "invitation_id")
    if any(keys & component.fit_key_hashes for component in components):
        raise ContractError("Holdout overlaps supervised training/calibration invitation IDs; in-sample evaluation is not a holdout.")
    for component in components:
        if (df["invitation_date"] < iso_date(component.fit_origin)).any():
            raise ContractError("Holdout precedes the frozen base fit origin.")


def evaluate_bundle(bundle: dict[str, Any], df: pd.DataFrame, output_dir: str,
                    bootstrap: int | None = None) -> dict[str, Any]:
    cfg = Config(**bundle["config"])
    if bundle["purpose"] != "frozen final-test artifact":
        raise ContractError("This artifact is for validation diagnostics, not the frozen final test.")
    manifest = bundle["manifest"]
    if bundle["history_required"] and fingerprint(df) != manifest["dataset_keys_fingerprint"]:
        raise ContractError("Longitudinal holdout evaluation requires the original complete input history stream, not a test-only subset.")
    selected = df["invitation_date"].between(iso_date(manifest["test_start"]), iso_date(manifest["test_end"]))
    heldout = df[selected]
    if fingerprint(heldout) != manifest["test_fingerprint"]:
        raise ContractError("Holdout does not reproduce the saved test manifest.")
    reject_fit_overlap(heldout, bundle["components"])
    mask = selected & common_mask(df, cfg)
    if not mask.any():
        raise ContractError("No fully observed eligible test rows.")
    raw, calibrated, final, X = prediction_matrices(bundle, df)
    eligible = df.loc[mask]
    priors = [c.training_prevalence for c in bundle["components"]]
    result = vector_metrics(eligible, raw[mask], calibrated[mask], final[mask], priors)
    result["cohort"] = {
        "reserved_rows": int(selected.sum()), "eligible_rows": int(mask.sum()),
        "missing_or_immature_exclusions": int((selected & ~common_mask(df, cfg)).sum()),
        "manifest": manifest, "selection_frozen": True, "provenance": cfg.provenance,
    }
    baseline = project_horizon_probabilities(np.tile(priors, (int(mask.sum()), 1)))
    result["patient_bootstrap"] = bootstrap_by_patient(eligible, final[mask], cfg.bootstrap if bootstrap is None else bootstrap, cfg.seed, baseline)
    result["horizon_specific_supplementary"] = {}
    for j, h in enumerate(HORIZONS):
        hm = selected & observed_mask(df, h, cfg)
        result["horizon_specific_supplementary"][str(h)] = evaluate_probabilities(
            (1-df.loc[hm, TARGETS[h]]).to_numpy(dtype=int), final[hm, j], priors[j])
    groups: dict[str, np.ndarray] = {}
    seen = np.array([hashlib.sha256(str(p).encode()).hexdigest() in bundle["components"][0].fit_people_hashes
                     for p in eligible["person_id"]])
    groups["seen_patients"], groups["unseen_patients"] = seen, ~seen
    if "first_invitation" in eligible:
        groups["source_first_invitation"] = eligible["first_invitation"].eq(1).to_numpy()
        groups["source_repeat_invitation"] = eligible["first_invitation"].eq(0).to_numpy()
    observed_count = build_patient_history(df, cfg)["observed_prior_rows"][mask].to_numpy()
    groups["no_observed_prior_input_row"] = observed_count == 0
    groups["sparse_input_history_0_or_1"] = observed_count < 2
    result["subgroups"] = {
        name: {"n": int(g.sum()), "small_sample": int(g.sum()) < cfg.minimum_segment,
               "horizons": {str(h): evaluate_probabilities((1-eligible.loc[g, TARGETS[h]]).to_numpy(dtype=int),
                                                          final[mask][g, j]) for j, h in enumerate(HORIZONS)}}
        for name, g in groups.items()
    }
    result["temporal_blocks"] = {
        str(period): {"n": int(g.sum()), "macro_log_loss": float(np.mean([
            log_loss(1 - eligible.loc[g, TARGETS[h]], final[mask][g, j], labels=[0, 1]) for j, h in enumerate(HORIZONS)]))}
        for period in eligible["invitation_date"].dt.to_period("Q").unique()
        for g in [(eligible["invitation_date"].dt.to_period("Q") == period).to_numpy()]
    }
    if "patient_disjoint_components" in bundle:
        stress_bundle = dict(bundle, components=bundle["patient_disjoint_components"])
        for c in stress_bundle["components"]:
            if key_hashes(heldout, "person_id") & c.fit_people_hashes:
                raise ContractError("Patient-disjoint stress artifact contains supervised-fit patient overlap.")
        sr, sc, sf, _ = prediction_matrices(stress_bundle, df)
        result["patient_disjoint_temporal_stress"] = vector_metrics(
            eligible, sr[mask], sc[mask], sf[mask], [c.training_prevalence for c in stress_bundle["components"]])
        result["patient_disjoint_history_policy"] = "All strictly earlier predictor snapshots allowed; no supervised-fit overlap."
    out = Path(output_dir)
    atomic_json(out / "final_test_metrics.json", result)
    plots_for_metrics(result, out, "final_test")
    atomic_json(out / "segments.json", summarize_segments(eligible, final[mask], cfg))
    (out / "evaluation_report.md").write_text(
        "# Locked final evaluation\n\n"
        f"Eligible invitations: {int(mask.sum())}. Provenance: {cfg.provenance}.\n"
        "This report does not select or replace the validation champion. No model refitting occurred.\n"
        f"Final coherent macro log loss: {result['final']['macro_log_loss']:.6f}.\n"
        "See final_test_metrics.json for raw/calibrated/final scores, calibration, ranking, subgroup and uncertainty diagnostics.\n"
        "Small groups and sample selection limit interpretation. Predictive associations do not identify intervention effects.\n")
    return result


def plots_for_metrics(metrics: dict[str, Any], out: Path, prefix: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, h in zip(axes, HORIZONS):
        ax.plot([0, 1], [0, 1], "--", color="gray")
        for stage in ("raw", "final"):
            bins = metrics[stage]["horizons"][str(h)].get("reliability", [])
            ax.plot([b["predicted"] for b in bins], [b["observed"] for b in bins], "o-", label=stage)
        ax.set(title=f"{h}-day nonattendance", xlabel="Predicted probability", ylabel="Observed fraction", xlim=(0, 1), ylim=(0, 1))
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_reliability.png", dpi=130)
    plt.close(fig)


def summarize_segments(df: pd.DataFrame, p: np.ndarray, cfg: Config) -> dict[str, Any]:
    rules: dict[str, tuple[pd.Series, str]] = {}
    if "healthcare_contact_12m_cnt" in df:
        rules["no_recorded_contacts"] = (df["healthcare_contact_12m_cnt"].eq(0), "Zero recorded provider/day contacts in 12 months.")
    if set(DOMAINS).issubset(df) and "healthcare_contact_12m_cnt" in df:
        rules["contact_without_recorded_prevention"] = (
            df["healthcare_contact_12m_cnt"].gt(0) & df[list(DOMAINS)].eq(0).all(axis=1),
            "Positive 12-month contacts and zero in each documented prevention window.")
    if {"prior_screen_count_5y", "months_since_last_screen"}.issubset(df):
        rules["prior_screen_long_recency"] = (df["prior_screen_count_5y"].gt(0) & df["months_since_last_screen"].gt(36),
                                               "Prior five-year screen count >0 and recorded recency >36 months.")
    for source in ("gp", "gyn"):
        a, b = f"{source}_registered", f"{source}_prevention_24m_cnt"
        if a in df and b in df:
            rules[f"{source}_registered_without_prevention"] = (df[a].eq(1) & df[b].eq(0),
                                                                f"{source.upper()} registered, no recorded 24-month prevention.")
    segments, unsupported = [], []
    for name, (mask, rule) in rules.items():
        if int(mask.sum()) < cfg.minimum_segment:
            unsupported.append({"segment": name, "support": int(mask.sum()), "reason": "Below prespecified minimum support"})
            continue
        segments.append({
            "segment": name, "rule": rule, "support": int(mask.sum()), "patients": int(df.loc[mask, "person_id"].nunique()),
            "horizons": {str(h): {
                "mean_predicted_nonattendance": float(p[mask, j].mean()),
                "observed_nonattendance": float((1-df.loc[mask, TARGETS[h]]).mean()),
                "wilson_descriptive_row_interval": wilson(int((1-df.loc[mask, TARGETS[h]]).sum()), int(mask.sum())),
            } for j, h in enumerate(HORIZONS)},
            "uncertainty_caveat": "Descriptive row interval; repeated invitations are dependent. Use patient bootstrap for confirmatory inference.",
            "observed_association": "Sample-specific observed risk in this rule-defined subgroup.",
            "model_association": "Mean predicted risk; not an explanation of causes.",
            "tentative_interpretation": "Possible engagement or access differences; unverified.",
            "intervention_hypothesis": "Offer navigation/support in a patient-randomized prospective experiment, not automatic outreach.",
            "additional_information": ["Consent and contact preferences", "Available channels", "Appointment capacity/access", "Reasons for nonparticipation"],
        })
    return {"segments": segments, "unsupported": unsupported, "overlap_allowed": True, "outcome_free_rules": True,
            "caveat": "No causal or treatment-benefit inference; rules fixed before inspecting test outcomes."}


def run_eda(df: pd.DataFrame, cfg: Config) -> None:
    manifest = reserve_test(df, cfg)
    out = Path(cfg.output_dir)
    atomic_json(out / "test_manifest.json", manifest)
    dev = df[df["invitation_date"] < iso_date(manifest["test_start"])].copy()
    if dev.empty:
        raise ContractError("No development rows remain after reserving the complete final test. Target-driven EDA refused.")
    summary: dict[str, Any] = {"provenance": cfg.provenance, "development_rows": len(dev), "test_excluded": True,
                               "numeric": {}, "subgroups": {}, "maturity": {}}
    for c in sorted(NUMERIC.intersection(dev) - AMBIGUOUS):
        values = dev[c].dropna()
        summary["numeric"][c] = {"n": len(values), "missing": int(dev[c].isna().sum()),
                                 "quantiles": values.quantile([0, .25, .5, .75, .95, 1]).to_dict() if len(values) else {},
                                 "invariant": bool(values.nunique() <= 1)}
    dev["invitation_period"] = dev["invitation_date"].dt.to_period("Q").astype(str)
    for h in HORIZONS:
        mask = observed_mask(dev, h, cfg, iso_date(manifest["selection_deadline"]))
        summary["maturity"][str(h)] = {
            "eligible": int(mask.sum()), "excluded_missing_immature": int((~mask).sum()),
            "nonattendance_prevalence": float((1-dev.loc[mask, TARGETS[h]]).mean()) if mask.any() else None}
        for column in ("invitation_period", "age_band", "territory", "first_invitation", "gp_registered", "gyn_registered", *DOMAINS):
            if column not in dev:
                continue
            entries = []
            for value, group in dev.loc[mask].groupby(column, dropna=False):
                if len(group) < cfg.minimum_segment:
                    continue
                successes = int((1-group[TARGETS[h]]).sum())
                entries.append({"category": str(value), "n": len(group), "nonattendance": successes/len(group),
                                "wilson_descriptive": wilson(successes, len(group))})
            summary["subgroups"][f"{column}_{h}d"] = entries
    numeric = sorted((NUMERIC - AMBIGUOUS).intersection(dev))
    dev[numeric].corr(method="spearman").to_csv(out / "development_spearman.csv")
    atomic_json(out / "development_eda.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if numeric:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        missingness = dev[numeric].isna().mean()
        ax.bar(missingness.index, missingness.values)
        ax.set_ylabel("Missing fraction")
        ax.tick_params(axis="x", labelrotation=90)
        fig.tight_layout()
        fig.savefig(out / "development_missingness.png", dpi=120)
        plt.close(fig)
    (out / "eda_report.md").write_text(
        "# Development-only exploration\n\n"
        f"Development rows: {len(dev)}. Provenance: {cfg.provenance}. Final test excluded.\n"
        "Rates include only observed labels mature before model-selection deadline. Small subgroups are suppressed.\n"
        "Intervals are descriptive; repeated-person dependence limits row-level Wilson intervals.\n"
        "Numeric associations use Spearman ranks; nominal categories are not converted to arbitrary correlation codes.\n"
        "No hypothesis tests were used, so there are no unadjusted multiple-testing p-values.\n"
        "Rolling-count differences are not incident event counts; unknown sample selection prevents population claims.\n"
        "Missing recency may mean no recorded five-year screen; registration is distinct from prevention.\n")


def explain_model(bundle: dict[str, Any], df: pd.DataFrame, output_dir: str,
                  rows: int = 150, repeats: int = 3, use_shap: bool = False) -> dict[str, Any]:
    if bundle["purpose"] != "held-out development diagnostics" or "diagnostic_fold" not in bundle:
        raise ContractError("Use validation_bundle.joblib for held-out development explanations; final refits do not make development rows held-out.")
    if rows < 1 or repeats < 1:
        raise ContractError("Explanation rows and repeats must be positive.")
    cfg = Config(**bundle["config"])
    if bundle["history_required"] and fingerprint(df) != bundle["manifest"]["dataset_keys_fingerprint"]:
        raise ContractError("Longitudinal explanation requires the original complete predictor-history stream.")
    fold = bundle["diagnostic_fold"]
    mask = df["invitation_date"].between(iso_date(fold["start"]), iso_date(fold["end"])) & common_mask(df, cfg, iso_date(fold["deadline"]))
    eligible = df.loc[mask]
    if eligible.empty:
        raise ContractError("No eligible held-out development explanation rows.")
    reject_fit_overlap(eligible, bundle["components"])
    chosen = np.random.default_rng(cfg.seed).choice(np.flatnonzero(mask), min(rows, int(mask.sum())), replace=False)
    chosen.sort()
    names = list(dict.fromkeys(n for c in bundle["components"] for n in c.features))
    all_X = build_features(df, cfg, names)
    X = all_X.iloc[chosen]
    sample = df.iloc[chosen]
    registry = build_feature_registry()

    def predict_frame(frame: pd.DataFrame) -> np.ndarray:
        p = np.column_stack([component.predict(frame)[1] for component in bundle["components"]])
        return project_horizon_probabilities(p)

    def loss(values: np.ndarray) -> float:
        return float(np.mean([log_loss(1-sample[TARGETS[h]], values[:, j], labels=[0, 1]) for j, h in enumerate(HORIZONS)]))

    baseline = predict_frame(X)
    baseline_loss = loss(baseline)
    rng = np.random.default_rng(cfg.seed)
    groups = sorted({g for n in names for g in registry[n].semantic_groups} - {"snapshot"})
    importance = []
    for group in groups:
        columns = [n for n in names if group in registry[n].semantic_groups]
        changes = []
        for _ in range(repeats):
            permuted = X.copy()
            permuted.loc[:, columns] = X[columns].iloc[rng.permutation(len(X))].to_numpy()
            changes.append(loss(predict_frame(permuted)) - baseline_loss)
        importance.append({"semantic_group": group, "features": columns, "repeats": repeats,
                           "mean_log_loss_increase": float(np.mean(changes)), "std": float(np.std(changes)),
                           "repeat_values": changes})
    coefficients = []
    shap_status: dict[str, Any] = {"status": "not requested"}
    for component in bundle["components"]:
        pipeline = component.pipeline
        model = pipeline.named_steps["model"]
        if hasattr(model, "coef_"):
            encoded = pipeline.named_steps["preprocess"].get_feature_names_out()
            coefficients.append({
                "horizon": component.horizon, "explained_output": "uncalibrated nonattendance logit",
                "encoding": "Standardized imputed numeric values; OHE categories, no reference category dropped; regularized conditional associations",
                "features": [{"encoded_name": n, "coefficient": float(v)} for n, v in zip(encoded, model.coef_[0])],
            })
    if use_shap:
        if not optional_available("shap"):
            shap_status = {"status": "skipped", "reason": "Optional SHAP unavailable; install requirements-optional.txt."}
        else:
            import shap
            supported = [c for c in bundle["components"] if c.family in {"decision_tree", "random_forest", "extra_trees"}]
            if not supported:
                shap_status = {"status": "skipped", "reason": "No supported tree finalist; final-wrapper permutation is available."}
            else:
                summaries = []
                for c in supported:
                    train = train_mask(df, c.horizon, iso_date(c.fit_origin), cfg)
                    if not train.any():
                        raise ContractError("Training-only SHAP background unavailable in supplied data.")
                    bg = all_X.loc[train, c.features].sample(min(50, int(train.sum())), random_state=cfg.seed)
                    prep = c.pipeline[:-1]
                    background, evaluation = prep.transform(bg), prep.transform(X[c.features].iloc[:min(30, len(X))])
                    guard = MatrixGuard(cfg.memory_mb, dense=True)
                    background, evaluation = guard.transform(background), guard.transform(evaluation)
                    try:
                        explainer = shap.TreeExplainer(c.pipeline.named_steps["model"], data=background)
                        values = np.asarray(explainer.shap_values(evaluation))
                        summaries.append({"horizon": c.horizon, "rows": len(evaluation),
                                          "explained_output": "Base tree probability, NOT calibration or final PAV wrapper",
                                          "mean_absolute_shap": np.abs(values).mean(axis=0).tolist(),
                                          "encoded_names": c.pipeline.named_steps["preprocess"].get_feature_names_out().tolist()})
                    except (ValueError, TypeError, NotImplementedError) as exc:
                        summaries.append({"horizon": c.horizon, "status": "unsupported", "reason": str(exc)})
                shap_status = {"status": "completed with bounded rows; no hard library timeout", "summaries": summaries}
    local_examples = [{
        "anonymous_example_number": j+1, "probabilities": dict(zip(OUTPUTS, p.tolist())),
        "note": "No identifiers or raw patient attributes. Group perturbations explain the aggregate fitted wrapper, not individual causes.",
    } for j, p in enumerate(baseline[:3])]
    earliest_fit = min(iso_date(c.fit_origin) for c in bundle["components"])
    training_background = pd.Series(True, index=df.index)
    for c in bundle["components"]:
        training_background &= train_mask(df, c.horizon, iso_date(c.fit_origin), cfg)
    if not training_background.any():
        raise ContractError("Earlier training predictor background is required for local sensitivity diagnostics.")
    representative = all_X.loc[training_background].sample(1, random_state=cfg.seed)
    for j, example in enumerate(local_examples):
        sensitivities = []
        for group in groups:
            columns = [n for n in names if group in registry[n].semantic_groups]
            perturbed = X.iloc[[j]].copy()
            perturbed.loc[:, columns] = representative[columns].to_numpy()
            difference = baseline[j] - predict_frame(perturbed)[0]
            sensitivities.append({"semantic_group": group, "probability_difference_from_training_reference": difference.tolist()})
        example["group_reference_sensitivity"] = sensitivities
        example["note"] = "Diagnostic reference replacement, not an additive attribution or plausible intervention; overlapping derived families may create unrealistic combinations."
    dependence = []
    for source in ("months_since_last_screen", "healthcare_contact_12m_cnt"):
        if source not in df:
            continue
        values = df.loc[training_background, source].dropna()
        if values.nunique() < 4:
            continue
        edges = np.unique(np.r_[-np.inf, values.quantile([.25, .5, .75]).to_numpy(), np.inf])
        binned = pd.cut(sample[source], edges)
        for interval in binned.cat.categories:
            supported = binned.eq(interval).to_numpy()
            if int(supported.sum()) < cfg.minimum_segment:
                continue
            dependence.append({"source": source, "bin": str(interval), "n": int(supported.sum()),
                               "mean_final_probabilities": baseline[supported].mean(axis=0).tolist(),
                               "meaning": "Conditional observed-bin association, not a manipulated counterfactual; cutpoints training-only."})
    segments = summarize_segments(sample, baseline, cfg)
    report = {
        "explained_output": "Final coherent probability vector via macro nonattendance log-loss permutation",
        "rows": len(X), "baseline_log_loss": baseline_loss, "permutation": importance,
        "permutation_caveat": "Derived/transformed columns perturbed jointly by overlapping semantic family. This diagnostic can create unrealistic combinations across families; it is not a valid intervention.",
        "coefficients": coefficients, "shap": shap_status, "local_examples": local_examples,
        "conditional_dependence": dependence,
        "segments": segments,
        "evidence_levels": ["Observed association in these development data", "Association used by this fitted predictive model",
                            "Tentative behavioral interpretation", "Intervention hypothesis requiring a prospective experiment"],
    }
    out = Path(output_dir)
    atomic_json(out / "explanations.json", report)
    atomic_json(out / "segments.json", segments)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ordered_importance = sorted(importance, key=lambda item: item["mean_log_loss_increase"])
    ax.barh([v["semantic_group"] for v in ordered_importance],
            [v["mean_log_loss_increase"] for v in ordered_importance],
            xerr=[v["std"] for v in ordered_importance])
    ax.set_xlabel("Final-wrapper validation log-loss increase (mean +/- SD)")
    fig.tight_layout()
    fig.savefig(out / "grouped_permutation.png", dpi=130)
    plt.close(fig)
    (out / "explainability_report.md").write_text(
        "# Held-out development explanation\n\n"
        f"Seeded held-out development rows: {len(X)}. Permutation repeats: {repeats}.\n"
        "Permutation evaluates the complete calibrated/coherent wrapper. Higher loss after perturbation indicates predictive use, not causation.\n"
        "Semantic families overlap. Derived-column perturbations may create unrealistic combinations and are diagnostic only.\n"
        "Logistic coefficients explain the uncalibrated logit; base-tree SHAP, if requested, does not explain the final wrapper.\n"
        "No raw identifiers or row attributes are included in local examples. Segment rules are outcome-free and prespecified.\n"
        "Stability across families/features is available from the paired feature comparisons; importance direction is not a treatment recommendation.\n"
        "See explanations.json and segments.json. A prospective experiment is required to test communication/support benefit.\n")
    return report


def self_test() -> None:
    assert np.allclose(project_horizon_probabilities(np.array([[.7, .8, .4]])), [[.75, .75, .4]])
    cfg = Config(outcome_data_end="2030-01-01", date_format="%Y-%m-%d")
    raw = pd.DataFrame({
        "INVITATION_ID": ["0001", "0002"], "PERSON_ID": ["000007", "000007"],
        "INVITATION_DATE": ["2020-01-01", "2021-01-01"],
        "HEALTHCARE_CONTACT_12M_CNT": [2, 5],
    })
    df = normalize_schema(raw, cfg)
    history = build_patient_history(df, cfg)
    assert history["observed_prior_rows"].tolist() == [0, 1]
    assert history["healthcare_contact_12m_cnt__lag1"].iloc[1] == 2
    assert label_available_at(df, 365, cfg).iloc[0] == pd.Timestamp("2020-12-31")
    print("Fast self-test passed. Run pytest for the full offline synthetic contract suite.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invitation-response probabilities; positive class is nonattendance.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test", help="Run fast synthetic diagnostics; no data files are read.")
    for command in ("inspect", "eda", "features", "benchmark", "train", "evaluate", "predict", "explain"):
        p = commands.add_parser(command, help={
            "inspect": "Validate the explicitly named CSV and write structural audits.",
            "eda": "Reserve final test, then summarize development data only.",
            "features": "Audit approved point-in-time features; optional local row-level export.",
            "benchmark": "Search chronological validation only; never automatically score final test.",
            "train": "Refit a frozen validation selection at its declared training origin.",
            "evaluate": "Explicit locked final-test evaluation of a trusted fixed artifact.",
            "predict": "Score new invitations without requiring current target columns.",
            "explain": "Explain the saved held-out development bundle.",
        }[command])
        p.add_argument("--data", default="data.csv", help="Explicit CSV path; no automatic CSV discovery.")
        p.add_argument("--data-mode", choices=("sample", "full"), default="sample")
        p.add_argument("--allow-full-data", action="store_true", help="Required together with --data-mode full.")
        p.add_argument("--separator", default=",")
        p.add_argument("--encoding", default="utf-8-sig")
        p.add_argument("--date-format", default="%d.%m.%Y")
        p.add_argument("--schema-mapping", help="JSON object mapping source column names to canonical names.")
        p.add_argument("--output-dir", default="artifacts/run")
        p.add_argument("--outcome-data-end", help="Independently verified ISO outcome ascertainment cutoff, exclusive.")
        p.add_argument("--reporting-lag-days", type=int, default=0, help="Explicit assumption when no delay is documented.")
        p.add_argument("--assume-t0-snapshots", action="store_true", help="Acknowledge retrospective predictor backfill limitation; required for fitting.")
        p.add_argument("--test-year", type=int)
        p.add_argument("--test-start")
        p.add_argument("--test-end")
        p.add_argument("--validation-start")
        p.add_argument("--validation-end")
        p.add_argument("--folds", type=int, default=2)
        p.add_argument("--cohort-policy", choices=("horizon", "common"), default="horizon")
        p.add_argument("--historical-outcomes", choices=("off", "on"), default="off")
        p.add_argument("--history-mode", choices=("require", "partial"), default="require")
        p.add_argument("--feature-sets", default="raw_snapshot,snapshot_engineered",
                       help="Comma-separated registry presets; use features --list-features for definitions.")
        p.add_argument("--include-groups", default="")
        p.add_argument("--exclude-groups", default="")
        p.add_argument("--list-features", action="store_true", help="Print the registry without loading any CSV.")
        p.add_argument("--models", default="logistic,decision_tree,hist_gradient_boosting",
                       help="Comma-separated families or all-available; dummy always included.")
        p.add_argument("--preset", choices=("smoke", "quick", "standard", "extensive"), default="quick")
        p.add_argument("--search-method", choices=("grid", "random"), default="grid")
        p.add_argument("--budget-seconds", type=float, help="Search launch budget only; active bounded fits may overrun.")
        p.add_argument("--max-trials", type=int, default=60)
        p.add_argument("--seed", type=int, default=17)
        p.add_argument("--n-jobs", type=int, default=4)
        p.add_argument("--memory-mb", type=int, default=4096, help="Materialized-frame and encoded-matrix guards, not a process RSS hard limit.")
        p.add_argument("--chunk-size", type=int, default=100000)
        p.add_argument("--sample-cap", type=int, help="Seeded whole-patient cap, not a row cap.")
        p.add_argument("--calibration", choices=("none", "sigmoid", "isotonic"), default="none")
        p.add_argument("--calibration-days", type=int, default=180)
        p.add_argument("--bootstrap", type=int, default=100)
        p.add_argument("--explanation-rows", type=int, default=150)
        p.add_argument("--permutation-repeats", type=int, default=3)
        p.add_argument("--minimum-segment", type=int, default=20)
        p.add_argument("--strict-optional", action="store_true")
        p.add_argument("--categorical-mode", choices=("onehot", "native"), default="onehot",
                       help="Native train-only category handling for installed LightGBM/XGBoost/CatBoost adapters.")
        p.add_argument("--joint-interval", action="store_true", help="Additional coherent four-class logistic validation benchmark; uses remaining search budget.")
        p.add_argument("--patient-disjoint-stress", action="store_true", help="Predeclare and fit a separate no-supervised-patient-overlap artifact.")
        p.add_argument("--provenance", default="sample-specific; sampling mechanism unknown")
        p.add_argument("--targets", default="90,180,365", help="This deployable three-horizon workflow requires exactly 90,180,365.")
        p.add_argument("--verbose", action="store_true")
        if command == "features":
            p.add_argument("--export-cache", action="store_true", help="Write a local ROW-LEVEL feature CSV to ignored output directory.")
        if command == "benchmark":
            p.add_argument("--save-model", action="store_true")
            p.add_argument("--resume", action="store_true")
        if command == "train":
            p.add_argument("--selection-path", required=True)
            p.add_argument("--model-path", default="artifacts/run/model_bundle.joblib")
        if command in {"evaluate", "predict", "explain"}:
            p.add_argument("--model-path", required=True, help="TRUSTED joblib path only; loading can execute Python code.")
        if command == "evaluate":
            p.add_argument("--split", choices=("test",), default="test", help="Reproduce the exact saved holdout definition.")
        if command == "predict":
            p.add_argument("--history-data")
            p.add_argument("--data-as-of", required=True)
            p.add_argument("--partial-history", action="store_true")
            p.add_argument("--output", required=True, help="Local row-level predictions; do not commit.")
        if command == "explain":
            p.add_argument("--shap", action="store_true")
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> Config:
    names = Config.__dataclass_fields__
    values = {k: v for k, v in vars(args).items() if k in names}
    if values.get("schema_mapping"):
        try:
            mapping = json.loads(Path(values["schema_mapping"]).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"Cannot read schema mapping: {exc}") from exc
        if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
            raise ContractError("Schema mapping must be a JSON object with string keys/values.")
        values["schema_mapping"] = mapping
    else:
        values["schema_mapping"] = {}
    cfg = Config(**values)
    cfg.validate()
    if getattr(args, "targets", "90,180,365") != "90,180,365":
        raise ContractError("The deployable vector requires --targets 90,180,365 in that order.")
    return cfg


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "self-test":
        self_test()
        return 0
    try:
        cfg = resolve_config(args)
        if args.list_features:
            print(json.dumps([asdict(f) for f in build_feature_registry().values()], indent=2))
            return 0
        load_start = time.monotonic()
        bundle = None
        if args.command in {"evaluate", "predict", "explain"}:
            bundle = load_bundle(args.model_path)
            # Parsing must use the artifact's persisted CSV/schema contract, never guessed CLI defaults.
            saved_cfg = Config(**bundle["config"])
            saved_cfg.data = cfg.data
            saved_cfg.data_mode, saved_cfg.allow_full_data = cfg.data_mode, cfg.allow_full_data
            saved_cfg.output_dir = cfg.output_dir
            if args.command == "predict":
                saved_cfg.sample_cap = None
            cfg = saved_cfg
        selection = None
        if args.command == "train":
            selection = json.loads(Path(args.selection_path).read_text())
            saved_cfg = Config(**selection["config"])
            saved_cfg.data = cfg.data
            saved_cfg.data_mode, saved_cfg.allow_full_data = cfg.data_mode, cfg.allow_full_data
            saved_cfg.output_dir = cfg.output_dir
            cfg = saved_cfg
        require_targets = args.command in {"eda", "benchmark", "train", "evaluate", "explain"}
        df = load_csv(cfg, require_targets=require_targets)
        loading_seconds = time.monotonic() - load_start
        if args.command in {"inspect", "features", "eda", "benchmark"}:
            write_schema_outputs(df, cfg)
        if args.command == "inspect":
            print(f"Schema valid: {len(df)} invitations; {df['person_id'].nunique()} people. Output: {cfg.output_dir}")
        elif args.command == "features":
            availability, names = {}, []
            for preset in cfg.feature_sets.split(","):
                selected, availability[preset] = resolve_feature_set(preset, df, cfg)
                names.extend(selected)
            X = build_features(df, cfg, list(dict.fromkeys(names)))
            atomic_json(Path(cfg.output_dir) / "feature_availability.json", availability)
            atomic_json(Path(cfg.output_dir) / "feature_summary.json", {
                "rows": len(X), "features": X.columns.tolist(), "config_hash": config_hash(cfg),
                "missing_fraction": X.isna().mean().to_dict(),
            })
            if args.export_cache:
                pd.concat([df[list(KEYS)], X], axis=1).to_csv(Path(cfg.output_dir) / "feature_cache.csv", index=False)
            print(f"Built {X.shape[1]} approved features for {len(X)} invitations.")
        elif args.command == "eda":
            run_eda(df, cfg)
        elif args.command == "benchmark":
            compare_models(df, cfg, args.save_model, args.resume)
        elif args.command == "train":
            assert selection is not None
            train_selected(df, cfg, selection, Path(args.model_path))
        elif args.command == "evaluate":
            assert bundle is not None
            evaluate_bundle(bundle, df, args.output_dir, args.bootstrap)
        elif args.command == "predict":
            assert bundle is not None
            history = load_csv(cfg, path=args.history_data) if args.history_data else None
            result = predict_bundle(bundle, df, history, args.data_as_of, args.partial_history or cfg.history_mode == "partial")
            output = Path(args.output)
            if output.resolve() == Path(cfg.data).resolve() or (args.history_data and output.resolve() == Path(args.history_data).resolve()):
                raise ContractError("Prediction output must not overwrite an input dataset.")
            output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output, index=False, date_format=cfg.date_format)
            print(f"Scored {len(result)} invitations with three coherent nonattendance probabilities.")
        elif args.command == "explain":
            assert bundle is not None
            explain_model(bundle, df, args.output_dir, args.explanation_rows, args.permutation_repeats, args.shap)
        atomic_json(Path(args.output_dir) / f"{args.command}_execution.json", {
            "loading_seconds": loading_seconds, "total_seconds": time.monotonic()-load_start,
            "command": args.command, "config_hash": config_hash(cfg),
        })
        return 0
    except (ContractError, OSError, ImportError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    # Canonical module identity keeps custom sklearn objects loadable in a fresh process.
    from mammography_pipeline import main as entrypoint
    raise SystemExit(entrypoint())
