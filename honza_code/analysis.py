#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rakathon 2026 – podrobná průzkumná analýza, asociace faktorů a segmentace.

Výstupy jsou agregované / anonymizované; identifikátory klientek se nezapisují do reportů.
Analýza je průzkumná: asociace nepředstavují důkaz příčinného vlivu.

Instalace:
    pip install pandas numpy scipy scikit-learn matplotlib tabulate

Příklad:
    python rakathon_mammo_analysis.py --input dataset.csv --output-dir analysis_out

Praktické volby:
    --target SCREENING_12M_FLAG      analyzovat jen jeden label (výchozí: všechny dostupné)
    --max-model-rows 80000           max. počet řádků pro interpretační logistický model na label
    --cluster-rows 100000            vzorek pro nalezení clusterů; přiřazení pak celé kohortě
    --clusters 5                     počet segmentů
    --skip-models                    přeskočit interpretační logistické modely
    --skip-clustering                přeskočit segmentaci pomocí clusterů
    --no-plots                       nevytvářet grafy

Hlavní výstupy:
    REPORT.md, data_profile.json, missingness.csv, numeric_summary.csv,
    label_prevalence.csv, categorical_rates__*.csv, categorical_associations__*.csv,
    numeric_bin_rates__*.csv, numeric_associations__*.csv, logistic_terms__*.csv,
    logistic_factor_ranking__*.csv, rule_segments__*.csv, cluster_profile.csv,
    cluster_assignments.csv, PNG grafy.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning

TARGET_HORIZONS = {
    "SCREENING_3M_FLAG": 90,
    "SCREENING_6M_FLAG": 180,
    "SCREENING_12M_FLAG": 365,
}
ID_COLUMNS = {"RC", "PERSON_ID", "INVITATION_ID", "_PATIENT_KEY", "_ROW_ID"}
FLAG_TRUE = {"A", "ANO", "YES", "Y", "TRUE", "T", "1", "1.0", "ATTENDED", "ÚČAST", "UCAST"}
FLAG_FALSE = {"N", "NE", "NO", "FALSE", "F", "0", "0.0", "NOT_ATTENDED", "NEÚČAST", "NEUCAST"}

NUMERIC_BASE = [
    "AGE", "HEALTHCARE_CONTACT_12M_CNT", "INSURANCE_DURATION_YEARS", "INSURANCE_DURATION_DAYS",
    "REG_DURATION", "GYN_REG_DURATION_YEARS", "PRIOR_SCREEN_COUNT_5Y", "MONTHS_SINCE_LAST_SCREEN",
    "GP_PREVENTION_24M_CNT", "GYN_PREVENTION_24M_CNT", "OTHER_SCREENING_36M_CNT",
    "DENTAL_PREVENTION_24M_CNT",
]
CATEGORICAL_BASE = [
    "AGE_BAND", "VZP_TERRITORIAL_AREA", "INSURANCE_DURATION_BAND", "GP_REGISTERED",
    "GP_REG_DURATION_BAND", "GYN_REGISTERED_FLAG", "GYN_REG_DURATION_BAND",
    "NEVER_SCREENED_BEFORE_FLAG", "FIRST_INVITATION",
]
NUMERIC_ENGINEERED = [
    "INVITATION_YEAR", "INVITATION_MONTH", "INVITATION_QUARTER", "INVITATION_DAY_OF_WEEK",
    "INVITATION_MONTH_SIN", "INVITATION_MONTH_COS", "INVITATION_DOW_SIN", "INVITATION_DOW_COS",
    "HIST_PRIOR_INVITATION_COUNT", "HIST_PRIOR_INVITATION_DATES", "HIST_DAYS_SINCE_PRIOR_INVITATION",
    "HIST_DAYS_SINCE_FIRST_INVITATION", "PREVENTION_TOTAL_CNT", "PREVENTION_DOMAINS_WITH_ACTIVITY",
    "PREVENTION_DATA_MISSING_COUNT", "ANY_GP_PREVENTION", "ANY_GYN_PREVENTION",
    "ANY_OTHER_SCREENING", "ANY_DENTAL_PREVENTION", "ANY_PRIOR_SCREENING_5Y",
    "HEALTHCARE_CONTACT_COUNT_LOG1P",
]
CATEGORICAL_ENGINEERED = [
    "INVITATION_SEASON", "PRIOR_INVITATION_COUNT_BAND", "DAYS_SINCE_PRIOR_INVITATION_BAND",
    "SCREENING_HISTORY_GROUP", "LAST_SCREEN_RECENCY_BAND", "PREVENTION_TOTAL_BAND",
    "PREVENTION_BREADTH_BAND", "HEALTHCARE_CONTACT_BAND", "REGISTRATION_PROFILE",
    "CARE_AND_SCREENING_PROFILE",
]


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Vstupní soubor neexistuje: {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p, low_memory=False, encoding="utf-8-sig",
                           dtype={"RC": "string", "PERSON_ID": "string", "INVITATION_ID": "string"})
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError("Podporované formáty: CSV, Parquet, Excel.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().upper() for c in out.columns]
    if out.columns.duplicated().any():
        raise ValueError(f"Duplicitní sloupce po normalizaci: {out.columns[out.columns.duplicated()].tolist()}")
    return out


def parse_binary(series: pd.Series) -> pd.Series:
    """Normalizuje labely A/N, Ano/Ne, yes/no a 1/0 na 1/0; neznámé hodnoty nechá jako NaN."""
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        return vals.where(vals.isin([0, 1]), np.nan).astype("float64")
    s = series.astype("string").str.strip().str.upper()
    s = s.str.replace(r"\s+", " ", regex=True)
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[s.isin(FLAG_TRUE)] = 1.0
    out.loc[s.isin(FLAG_FALSE)] = 0.0
    return out


def _to_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").astype("float64")


def _safe_band(series: pd.Series, bins: Sequence[float], labels: Sequence[str]) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True).astype("string").fillna("missing")


def engineer_features(raw: pd.DataFrame, patient_id_column: str = "PERSON_ID") -> Tuple[pd.DataFrame, Optional[str]]:
    """Rozšíří řádkový dataset o časové, behaviorální a historii-k-T0 featury.

    Historie je vypočtena pouze z pozvánek v přísně dřívějším datu. Více řádků stejné klientky
    ve stejný den si navzájem nepředávají informace. Žádný identifikátor klientky se nepoužívá
    jako prediktor ani se nevypisuje do reportů.
    """
    df = normalize_columns(raw)
    if "INVITATION_DATE" not in df.columns:
        raise ValueError("Chybí povinný sloupec INVITATION_DATE.")
    df["_ROW_ID"] = np.arange(len(df), dtype=np.int64)
    df["INVITATION_DATE"] = pd.to_datetime(df["INVITATION_DATE"], errors="coerce")
    invalid_date_n = int(df["INVITATION_DATE"].isna().sum())
    if invalid_date_n:
        logging.warning("%s řádků má neplatné datum pozvánky; pro časové analýzy budou vyřazeny.", f"{invalid_date_n:,}")
        df = df.loc[df["INVITATION_DATE"].notna()].copy()
    if df.empty:
        raise ValueError("Po vyřazení neplatných dat nezbyly žádné řádky.")

    d = df["INVITATION_DATE"]
    df["INVITATION_YEAR"] = d.dt.year.astype("float64")
    df["INVITATION_MONTH"] = d.dt.month.astype("float64")
    df["INVITATION_QUARTER"] = d.dt.quarter.astype("float64")
    df["INVITATION_DAY_OF_WEEK"] = d.dt.dayofweek.astype("float64")
    df["INVITATION_MONTH_SIN"] = np.sin(2 * np.pi * d.dt.month / 12.0)
    df["INVITATION_MONTH_COS"] = np.cos(2 * np.pi * d.dt.month / 12.0)
    df["INVITATION_DOW_SIN"] = np.sin(2 * np.pi * d.dt.dayofweek / 7.0)
    df["INVITATION_DOW_COS"] = np.cos(2 * np.pi * d.dt.dayofweek / 7.0)
    season_map = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                  6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
    df["INVITATION_SEASON"] = d.dt.month.map(season_map).astype("string")

    # Coerce known count/duration columns to numeric without destroying originals.
    for col in NUMERIC_BASE:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Historia klientky podle unikátních dat pozvánek; při chybějícím ID je řádek samostatný.
    requested = str(patient_id_column or "PERSON_ID").strip().upper()
    id_col = requested if requested in df.columns else next((x for x in ("PERSON_ID", "RC") if x in df.columns), None)
    if id_col is not None:
        if id_col != requested:
            logging.warning("ID sloupec %s chybí; historii počítám podle %s.", requested, id_col)
        pat = df[id_col].astype("string")
        missing_id = pat.isna() | pat.str.strip().eq("")
        # Nepropojujeme všechny chybějící identifikátory do jedné osoby.
        pat = pat.copy()
        pat.loc[missing_id] = "__MISSING_ROW_" + df.loc[missing_id, "_ROW_ID"].astype(str)
        df["_PATIENT_KEY"] = pat.astype("string")
        day = (df.groupby(["_PATIENT_KEY", "INVITATION_DATE"], dropna=False, sort=False)
                 .agg(_ROWS_ON_DATE=("_ROW_ID", "size"), _ROW_ID_MIN=("_ROW_ID", "min"))
                 .reset_index().sort_values(["_PATIENT_KEY", "INVITATION_DATE"], kind="mergesort"))
        g = day.groupby("_PATIENT_KEY", sort=False)
        day["HIST_PRIOR_INVITATION_COUNT"] = g["_ROWS_ON_DATE"].cumsum() - day["_ROWS_ON_DATE"]
        day["HIST_PRIOR_INVITATION_DATES"] = g.cumcount()
        day["_PREVIOUS_DATE"] = g["INVITATION_DATE"].shift(1)
        day["HIST_DAYS_SINCE_PRIOR_INVITATION"] = (day["INVITATION_DATE"] - day["_PREVIOUS_DATE"]).dt.days
        first_date = g["INVITATION_DATE"].transform("min")
        day["HIST_DAYS_SINCE_FIRST_INVITATION"] = (day["INVITATION_DATE"] - first_date).dt.days.where(day["HIST_PRIOR_INVITATION_COUNT"].gt(0))
        hist_cols = ["HIST_PRIOR_INVITATION_COUNT", "HIST_PRIOR_INVITATION_DATES",
                     "HIST_DAYS_SINCE_PRIOR_INVITATION", "HIST_DAYS_SINCE_FIRST_INVITATION"]
        df = df.merge(day[["_PATIENT_KEY", "INVITATION_DATE"] + hist_cols],
                      on=["_PATIENT_KEY", "INVITATION_DATE"], how="left", sort=False, validate="many_to_one")
        df.drop(columns=["_PATIENT_KEY"], inplace=True)
    else:
        logging.warning("Chybí ID klientky: historické featury budou nulové/NA a interpretujte je opatrně.")
        df["HIST_PRIOR_INVITATION_COUNT"] = 0
        df["HIST_PRIOR_INVITATION_DATES"] = 0
        df["HIST_DAYS_SINCE_PRIOR_INVITATION"] = np.nan
        df["HIST_DAYS_SINCE_FIRST_INVITATION"] = np.nan

    df["PRIOR_INVITATION_COUNT_BAND"] = _safe_band(df["HIST_PRIOR_INVITATION_COUNT"], [-0.1, 0.1, 1.1, 2.1, np.inf],
                                                     ["0", "1", "2", "3+"])
    df["DAYS_SINCE_PRIOR_INVITATION_BAND"] = pd.cut(
        df["HIST_DAYS_SINCE_PRIOR_INVITATION"], bins=[-0.1, 180, 365, 730, 1825, np.inf],
        labels=["<=6m", "6-12m", "1-2y", "2-5y", ">5y"], include_lowest=True
    ).astype("string").fillna("no_prior_invitation")

    # Prevenční chování: zachováme informaci o chybějících hodnotách, zároveň vytvoříme konzervativní součty.
    prevention_cols = ["GP_PREVENTION_24M_CNT", "GYN_PREVENTION_24M_CNT", "OTHER_SCREENING_36M_CNT", "DENTAL_PREVENTION_24M_CNT"]
    available_prevention = [c for c in prevention_cols if c in df.columns]
    if available_prevention:
        prevention_numeric = pd.DataFrame({c: _to_num(df, c) for c in available_prevention}, index=df.index)
        df["PREVENTION_DATA_MISSING_COUNT"] = prevention_numeric.isna().sum(axis=1)
        df["PREVENTION_TOTAL_CNT"] = prevention_numeric.fillna(0).clip(lower=0).sum(axis=1)
        df["PREVENTION_DOMAINS_WITH_ACTIVITY"] = (prevention_numeric.fillna(0).gt(0)).sum(axis=1)
        derived_map = {"GP_PREVENTION_24M_CNT": "ANY_GP_PREVENTION", "GYN_PREVENTION_24M_CNT": "ANY_GYN_PREVENTION",
                       "OTHER_SCREENING_36M_CNT": "ANY_OTHER_SCREENING", "DENTAL_PREVENTION_24M_CNT": "ANY_DENTAL_PREVENTION"}
        for c, out_col in derived_map.items():
            if c in prevention_numeric:
                df[out_col] = prevention_numeric[c].gt(0).astype("float64").where(prevention_numeric[c].notna())
    else:
        df["PREVENTION_DATA_MISSING_COUNT"] = np.nan
        df["PREVENTION_TOTAL_CNT"] = np.nan
        df["PREVENTION_DOMAINS_WITH_ACTIVITY"] = np.nan

    df["PREVENTION_TOTAL_BAND"] = pd.cut(df["PREVENTION_TOTAL_CNT"], bins=[-0.1, 0.1, 1.1, 3.1, 6.1, np.inf],
        labels=["0", "1", "2-3", "4-6", "7+"], include_lowest=True).astype("string").fillna("missing")
    df["PREVENTION_BREADTH_BAND"] = df["PREVENTION_DOMAINS_WITH_ACTIVITY"].map(
        lambda x: "missing" if pd.isna(x) else ("0 domains" if x == 0 else ("1 domain" if x == 1 else ("2 domains" if x == 2 else "3-4 domains"))))

    if "PRIOR_SCREEN_COUNT_5Y" in df.columns:
        prior_screen = _to_num(df, "PRIOR_SCREEN_COUNT_5Y")
        df["ANY_PRIOR_SCREENING_5Y"] = prior_screen.gt(0).astype("float64").where(prior_screen.notna())
        df["SCREENING_HISTORY_GROUP"] = pd.cut(prior_screen, bins=[-0.1, 0.1, 1.1, np.inf],
            labels=["0 in last 5y", "1 in last 5y", "2+ in last 5y"], include_lowest=True).astype("string").fillna("missing")
    else:
        df["ANY_PRIOR_SCREENING_5Y"] = np.nan
        df["SCREENING_HISTORY_GROUP"] = "missing"

    if "MONTHS_SINCE_LAST_SCREEN" in df.columns:
        months_last = _to_num(df, "MONTHS_SINCE_LAST_SCREEN")
        never = parse_binary(df["NEVER_SCREENED_BEFORE_FLAG"]) if "NEVER_SCREENED_BEFORE_FLAG" in df.columns else pd.Series(np.nan, index=df.index)
        recency = pd.cut(months_last, bins=[-0.1, 12, 24, 36, 60, np.inf],
                         labels=["0-12m", "13-24m", "25-36m", "37-60m", ">60m"], include_lowest=True).astype("string")
        recency = recency.mask(never.eq(1), "never_flagged")
        df["LAST_SCREEN_RECENCY_BAND"] = recency.fillna("missing_or_never")
    else:
        df["LAST_SCREEN_RECENCY_BAND"] = "missing"

    # Další behaviorální a kombinované featury.
    df["HEALTHCARE_CONTACT_COUNT_LOG1P"] = np.log1p(_to_num(df, "HEALTHCARE_CONTACT_12M_CNT").clip(lower=0))
    hcc = _to_num(df, "HEALTHCARE_CONTACT_12M_CNT")
    df["HEALTHCARE_CONTACT_BAND"] = pd.cut(hcc, bins=[-0.1, 0.1, 2, 5, 10, np.inf],
        labels=["0", "1-2", "3-5", "6-10", "11+"], include_lowest=True).astype("string").fillna("missing")
    gp_reg = parse_binary(df["GP_REGISTERED"]) if "GP_REGISTERED" in df.columns else pd.Series(np.nan, index=df.index)
    gyn_reg = parse_binary(df["GYN_REGISTERED_FLAG"]) if "GYN_REGISTERED_FLAG" in df.columns else pd.Series(np.nan, index=df.index)
    reg_profile = pd.Series("unknown", index=df.index, dtype="string")
    reg_profile.loc[gp_reg.eq(1) & gyn_reg.eq(1)] = "GP+GYN registered"
    reg_profile.loc[gp_reg.eq(1) & gyn_reg.eq(0)] = "GP only"
    reg_profile.loc[gp_reg.eq(0) & gyn_reg.eq(1)] = "GYN only"
    reg_profile.loc[gp_reg.eq(0) & gyn_reg.eq(0)] = "neither registered"
    df["REGISTRATION_PROFILE"] = reg_profile
    has_screen_info = df["SCREENING_HISTORY_GROUP"].astype("string")
    prevention_behavior = df["PREVENTION_TOTAL_BAND"].astype("string")
    reg_profile = df["REGISTRATION_PROFILE"].astype("string")
    df["CARE_AND_SCREENING_PROFILE"] = ("screen=" + has_screen_info + " | prev=" + prevention_behavior + " | reg=" + reg_profile).astype("string")

    # Existing supplied history counters are ambiguous in their exact semantics; use only newly reconstructed values.
    return df, id_col


def available_targets(df: pd.DataFrame, requested: Optional[str]) -> List[str]:
    if requested:
        t = requested.strip().upper()
        if t not in TARGET_HORIZONS:
            raise ValueError(f"Neznámý target {requested}; povolené: {', '.join(TARGET_HORIZONS)}")
        if t not in df.columns:
            raise ValueError(f"Ve vstupním datasetu chybí cílový sloupec {t}.")
        return [t]
    present = [t for t in TARGET_HORIZONS if t in df.columns]
    if not present:
        raise ValueError("Nenalezen žádný ze sloupců SCREENING_3M_FLAG, SCREENING_6M_FLAG, SCREENING_12M_FLAG.")
    return present


def feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numerics = [c for c in NUMERIC_BASE + NUMERIC_ENGINEERED if c in df.columns]
    categoricals = [c for c in CATEGORICAL_BASE + CATEGORICAL_ENGINEERED if c in df.columns]
    # Omezíme numerické sloupce s nulovou variabilitou; pro EDA se původní sloupce zachovají.
    numerics = [c for c in numerics if pd.to_numeric(df[c], errors="coerce").nunique(dropna=True) > 1]
    categoricals = [c for c in categoricals if df[c].nunique(dropna=True) > 1 and df[c].nunique(dropna=True) <= 200]
    return numerics, categoricals


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, (pd.Timestamp,)): return obj.isoformat()
    if pd.isna(obj) if not isinstance(obj, (list, dict, tuple, np.ndarray)) else False: return None
    return str(obj) if not isinstance(obj, (str, int, float, bool, list, dict, tuple)) else obj


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def wilson_interval(successes: np.ndarray, totals: np.ndarray, z: float = 1.96) -> Tuple[np.ndarray, np.ndarray]:
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = successes / totals
        denom = 1.0 + z*z / totals
        center = (p + z*z/(2*totals)) / denom
        half = (z * np.sqrt((p*(1-p) + z*z/(4*totals))/totals)) / denom
    low, high = center-half, center+half
    low[totals == 0] = np.nan; high[totals == 0] = np.nan
    return low, high


def data_profile(df: pd.DataFrame, original_columns: Sequence[str], id_col: Optional[str], outdir: Path) -> Dict[str, Any]:
    n = len(df)
    profile: Dict[str, Any] = {
        "rows_after_valid_date_filter": int(n),
        "columns_in_source": int(len(original_columns)),
        "unique_people": int(df[id_col].nunique(dropna=True)) if id_col and id_col in df.columns else None,
        "unique_invitation_ids": int(df["INVITATION_ID"].nunique(dropna=True)) if "INVITATION_ID" in df.columns else None,
        "date_min": df["INVITATION_DATE"].min(),
        "date_max": df["INVITATION_DATE"].max(),
        "rows_with_missing_patient_id": int(df[id_col].isna().sum()) if id_col and id_col in df.columns else None,
    }
    if id_col and id_col in df.columns:
        counts = df.groupby(id_col, dropna=True).size()
        profile["people_with_multiple_rows"] = int((counts > 1).sum())
        profile["share_rows_belonging_to_repeat_people"] = float(df[id_col].isin(counts[counts > 1].index).mean())
        profile["max_rows_per_person"] = int(counts.max()) if len(counts) else None
        profile["rows_per_person_distribution"] = {str(int(k)): int(v) for k, v in counts.value_counts().sort_index().items() if k <= 10}
    if "INVITATION_ID" in df.columns:
        profile["duplicate_invitation_id_rows_after_first"] = int(df["INVITATION_ID"].duplicated().sum())

    # Missingness and numerical summaries deliberately exclude direct identifiers.
    rows = []
    for c in original_columns:
        if c in ID_COLUMNS or c not in df.columns: continue
        s = df[c]
        rows.append({"feature": c, "dtype": str(s.dtype), "n_missing": int(s.isna().sum()),
                     "missing_pct": round(float(s.isna().mean()*100), 4),
                     "n_unique": int(s.nunique(dropna=True)),
                     "example_values": " | ".join(s.dropna().astype(str).drop_duplicates().head(5).tolist())})
    write_csv(pd.DataFrame(rows).sort_values("missing_pct", ascending=False), outdir / "missingness.csv")
    with (outdir / "data_profile.json").open("w", encoding="utf-8") as f:
        json.dump({k: json_safe(v) for k, v in profile.items()}, f, ensure_ascii=False, indent=2, default=json_safe)
    return profile


def numeric_summary(df: pd.DataFrame, numeric_cols: Sequence[str], outdir: Path) -> None:
    rows = []
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        desc = s.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
        rows.append({"feature": c, "n_nonmissing": int(s.notna().sum()), "n_missing": int(s.isna().sum()),
                     "mean": desc.get("mean"), "std": desc.get("std"), "min": desc.get("min"),
                     "p01": desc.get("1%"), "p05": desc.get("5%"), "p25": desc.get("25%"),
                     "median": desc.get("50%"), "p75": desc.get("75%"), "p95": desc.get("95%"),
                     "p99": desc.get("99%"), "max": desc.get("max"), "n_unique": int(s.nunique(dropna=True))})
    write_csv(pd.DataFrame(rows), outdir / "numeric_summary.csv")


def label_summary(df: pd.DataFrame, targets: Sequence[str], outdir: Path) -> pd.DataFrame:
    rows = []
    for target in targets:
        y = parse_binary(df[target])
        n = int(y.notna().sum()); yes = int(y.eq(1).sum()); no = int(y.eq(0).sum())
        rows.append({"target": target, "horizon_days": TARGET_HORIZONS[target], "rows_total": len(df),
                     "rows_labeled": n, "rows_missing_or_unrecognized": int(y.isna().sum()),
                     "participants": yes, "nonparticipants": no,
                     "participation_rate_pct": 100*yes/n if n else np.nan,
                     "nonparticipation_rate_pct": 100*no/n if n else np.nan})
    result = pd.DataFrame(rows)
    write_csv(result, outdir / "label_prevalence.csv")
    # Monthly and annual trends: complete rows for each target, label rate per period.
    trends = []
    for target in targets:
        tmp = pd.DataFrame({"date": df["INVITATION_DATE"], "y": parse_binary(df[target])}).dropna(subset=["y"])
        tmp["year_month"] = tmp["date"].dt.to_period("M").astype(str)
        m = tmp.groupby("year_month")["y"].agg(n="size", participants="sum", rate="mean").reset_index()
        m["target"] = target; m["participation_rate_pct"] = 100*m["rate"]
        trends.append(m.drop(columns="rate"))
    trend_df = pd.concat(trends, ignore_index=True) if trends else pd.DataFrame()
    write_csv(trend_df, outdir / "monthly_participation_trend.csv")
    return result


def categorical_associations(df: pd.DataFrame, target: str, categorical_cols: Sequence[str], outdir: Path,
                             min_group_count: int = 30) -> pd.DataFrame:
    y = parse_binary(df[target])
    valid = y.notna()
    overall_rate = float(y[valid].mean()) if valid.any() else np.nan
    detail_rows: List[Dict[str, Any]] = []
    assoc_rows: List[Dict[str, Any]] = []
    for col in categorical_cols:
        cat = df[col].astype("string").fillna("<MISSING>")
        # Příliš drobné úrovně sloučíme do OTHER jen pro statistickou tabulku, aby nevládly šumové skupiny.
        counts = cat[valid].value_counts(dropna=False)
        small_levels = set(counts[counts < min_group_count].index.astype(str))
        cat2 = cat.astype(str).where(~cat.astype(str).isin(small_levels), "<OTHER_RARE>")
        tmp = pd.DataFrame({"group": cat2, "y": y}).loc[valid]
        agg = tmp.groupby("group", dropna=False)["y"].agg(n="size", participants="sum", rate="mean").reset_index()
        low, high = wilson_interval(agg["participants"].to_numpy(), agg["n"].to_numpy())
        agg["feature"] = col
        agg["participation_rate_pct"] = 100*agg["rate"]
        agg["ci95_low_pct"] = 100*low
        agg["ci95_high_pct"] = 100*high
        agg["risk_difference_vs_overall_pp"] = 100*(agg["rate"]-overall_rate)
        agg["lift_vs_overall"] = agg["rate"]/overall_rate if overall_rate and not np.isnan(overall_rate) else np.nan
        detail_rows.extend(agg.drop(columns="rate").to_dict("records"))
        try:
            tab = pd.crosstab(cat2[valid], y[valid])
            if tab.shape[0] > 1 and tab.shape[1] > 1:
                chi2, p, _, _ = chi2_contingency(tab, correction=False)
                total = tab.to_numpy().sum()
                denom = min(tab.shape[0]-1, tab.shape[1]-1)
                v = math.sqrt(chi2/(total*denom)) if total and denom > 0 else np.nan
            else:
                chi2, p, v = np.nan, np.nan, np.nan
            rate_by_group = agg["rate"]
            assoc_rows.append({"feature": col, "n_levels_after_rare_merge": int(len(agg)), "n_labeled": int(len(tmp)),
                               "overall_rate_pct": overall_rate*100, "min_group_rate_pct": 100*rate_by_group.min() if len(agg) else np.nan,
                               "max_group_rate_pct": 100*rate_by_group.max() if len(agg) else np.nan,
                               "range_between_group_rates_pp": 100*(rate_by_group.max()-rate_by_group.min()) if len(agg) else np.nan,
                               "chi_square": chi2, "p_value": p, "cramers_v": v,
                               "note": "P-hodnota je při velkém N snadno velmi malá; pro velikost vztahu sledujte Cramérovo V a rozdíl sazeb."})
        except Exception as exc:
            logging.warning("Asociaci %s / %s nešlo spočítat: %s", col, target, exc)
    detail = pd.DataFrame(detail_rows)
    if not detail.empty:
        detail.sort_values(["feature", "n"], ascending=[True, False], inplace=True)
        write_csv(detail, outdir / f"categorical_rates__{target}.csv")
    assoc = pd.DataFrame(assoc_rows)
    if not assoc.empty:
        assoc.sort_values(["cramers_v", "range_between_group_rates_pp"], ascending=False, inplace=True, na_position="last")
        write_csv(assoc, outdir / f"categorical_associations__{target}.csv")
    return assoc


def numeric_associations(df: pd.DataFrame, target: str, numeric_cols: Sequence[str], outdir: Path,
                         max_corr_rows: int = 200000) -> pd.DataFrame:
    y_all = parse_binary(df[target])
    valid_idx = y_all[y_all.notna()].index
    if len(valid_idx) > max_corr_rows:
        # Deterministický vzorek pro korelace a kvantily; agregované bin-rate níže se počítají pro všechny řádky.
        rng = np.random.default_rng(2026)
        idx = np.sort(rng.choice(valid_idx.to_numpy(), size=max_corr_rows, replace=False))
    else:
        idx = valid_idx.to_numpy()
    sample_y = y_all.loc[idx]
    assoc_rows = []
    bin_rows = []
    overall = float(y_all.dropna().mean()) if y_all.notna().any() else np.nan
    for col in numeric_cols:
        x_all = pd.to_numeric(df[col], errors="coerce")
        x = x_all.loc[idx]
        m = x.notna() & sample_y.notna()
        if int(m.sum()) >= 20 and x.loc[m].nunique() > 1:
            try:
                rho, p = spearmanr(x.loc[m].to_numpy(), sample_y.loc[m].to_numpy())
            except Exception:
                rho, p = np.nan, np.nan
        else:
            rho, p = np.nan, np.nan
        assoc_rows.append({"feature": col, "n_corr_sample": int(m.sum()), "spearman_rho_with_participation": rho,
                           "p_value_sample": p,
                           "mean_among_participants": float(x_all[y_all.eq(1)].mean()) if y_all.eq(1).any() else np.nan,
                           "mean_among_nonparticipants": float(x_all[y_all.eq(0)].mean()) if y_all.eq(0).any() else np.nan,
                           "median_among_participants": float(x_all[y_all.eq(1)].median()) if y_all.eq(1).any() else np.nan,
                           "median_among_nonparticipants": float(x_all[y_all.eq(0)].median()) if y_all.eq(0).any() else np.nan})
        # Robust sample-derived quantile thresholds; rates are then computed over the complete labeled cohort.
        qsource = x_all.dropna()
        if len(qsource) > 200000:
            qsource = qsource.sample(n=200000, random_state=2026)
        if qsource.nunique() < 2:
            continue
        qs = np.unique(qsource.quantile([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]).to_numpy(dtype=float))
        if len(qs) < 3:
            continue
        # Open extreme edges enough to include numerical values equal to endpoints.
        bins = qs.copy(); bins[0] = -np.inf; bins[-1] = np.inf
        try:
            groups = pd.cut(x_all, bins=bins, include_lowest=True, duplicates="drop")
            tmp = pd.DataFrame({"bin": groups.astype("string").fillna("missing"), "y": y_all})
            tmp = tmp.dropna(subset=["y"])
            ag = tmp.groupby("bin", observed=False)["y"].agg(n="size", participants="sum", rate="mean").reset_index()
            ag = ag[ag["n"] > 0]
            ag["feature"] = col
            ag["participation_rate_pct"] = 100*ag["rate"]
            ag["risk_difference_vs_overall_pp"] = 100*(ag["rate"]-overall)
            ag["lift_vs_overall"] = ag["rate"]/overall if overall and not np.isnan(overall) else np.nan
            bin_rows.extend(ag.drop(columns="rate").to_dict("records"))
        except Exception as exc:
            logging.debug("Kvantily pro %s nešly vytvořit: %s", col, exc)
    assoc = pd.DataFrame(assoc_rows)
    if not assoc.empty:
        assoc["abs_spearman_rho"] = assoc["spearman_rho_with_participation"].abs()
        assoc.sort_values("abs_spearman_rho", ascending=False, inplace=True, na_position="last")
        write_csv(assoc, outdir / f"numeric_associations__{target}.csv")
    bins = pd.DataFrame(bin_rows)
    if not bins.empty:
        write_csv(bins, outdir / f"numeric_bin_rates__{target}.csv")
    return assoc


def get_preprocessor(numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                             ("scaler", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_pipe, list(numeric_cols)), ("cat", cat_pipe, list(categorical_cols))],
                             remainder="drop", sparse_threshold=0.3, verbose_feature_names_out=True)


def fit_interpretable_logistic(df: pd.DataFrame, target: str, numeric_cols: Sequence[str], categorical_cols: Sequence[str],
                               outdir: Path, max_rows: int = 80000) -> Optional[pd.DataFrame]:
    y_all = parse_binary(df[target])
    valid = y_all.notna()
    if y_all[valid].nunique() < 2 or int(valid.sum()) < 100:
        logging.warning("Příliš málo označených řádků / tříd pro interpretační logistickou regresi (%s).", target)
        return None
    cols = list(numeric_cols) + list(categorical_cols)
    cols = [c for c in cols if c in df.columns]
    if not cols: return None
    rows = df.index[valid].to_numpy()
    if len(rows) > max_rows:
        # Stratifikovaný vzorek udržuje přibližnou prevalenci obou tříd a je reprodukovatelný.
        rng = np.random.default_rng(2026)
        selected = []
        for klass in (0.0, 1.0):
            class_rows = y_all.loc[rows][y_all.loc[rows].eq(klass)].index.to_numpy()
            n_class = max(1, int(round(max_rows * len(class_rows) / len(rows))))
            selected.extend(rng.choice(class_rows, size=min(n_class, len(class_rows)), replace=False).tolist())
        rows = np.array(sorted(set(selected)))
    sub = df.loc[rows, cols].copy()
    y = y_all.loc[rows].astype(int)
    # Drop entirely empty or constant columns on the analysis sample.
    nums = [c for c in numeric_cols if c in sub.columns and pd.to_numeric(sub[c], errors="coerce").nunique(dropna=True) > 1]
    cats = [c for c in categorical_cols if c in sub.columns and sub[c].nunique(dropna=True) > 1]
    if not nums and not cats: return None
    prep = get_preprocessor(nums, cats)
    model = Pipeline([("preprocessor", prep), ("classifier", LogisticRegression(
        solver="saga", C=0.5, max_iter=250, tol=1e-3, random_state=42, n_jobs=-1,
        class_weight=None, fit_intercept=True
    ))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(sub[nums + cats], y)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            logging.warning("Logistický model pro %s nedokonvergoval v max_iter; interpretujte koeficienty opatrně.", target)
    classifier = model.named_steps["classifier"]
    names = model.named_steps["preprocessor"].get_feature_names_out()
    coefs = classifier.coef_.ravel()
    terms = pd.DataFrame({"encoded_term": names, "coefficient_log_odds": coefs})
    terms["odds_ratio_per_encoded_unit"] = np.exp(np.clip(terms["coefficient_log_odds"], -20, 20))
    terms["abs_coefficient"] = terms["coefficient_log_odds"].abs()
    terms["target"] = target
    terms["n_rows_fit"] = len(y)
    terms["positive_rate_fit_pct"] = 100*y.mean()
    # Translate encoded dummy terms back to the source feature for aggregate ranking.
    source = []
    for name in names:
        clean = str(name)
        if clean.startswith("num__"):
            src = clean[5:]
        elif clean.startswith("cat__"):
            rest = clean[5:]
            src = next((c for c in sorted(cats, key=len, reverse=True) if rest == c or rest.startswith(c + "_")), rest)
        else:
            src = clean
        source.append(src)
    terms["source_feature"] = source
    terms.sort_values("abs_coefficient", ascending=False, inplace=True)
    write_csv(terms, outdir / f"logistic_terms__{target}.csv")
    factor = (terms.groupby("source_feature").agg(max_abs_coefficient=("abs_coefficient", "max"),
              rms_coefficient=("coefficient_log_odds", lambda s: float(np.sqrt(np.mean(np.square(s))))),
              number_of_encoded_terms=("encoded_term", "count"))
              .reset_index().sort_values("max_abs_coefficient", ascending=False))
    factor["target"] = target
    write_csv(factor, outdir / f"logistic_factor_ranking__{target}.csv")
    return factor



def fit_nonlinear_forest(df: pd.DataFrame, target: str, numeric_cols: Sequence[str], categorical_cols: Sequence[str],
                         outdir: Path, max_rows: int = 80000) -> Optional[pd.DataFrame]:
    """Exploratorní stromový model pro nelineární vztahy a interakce; reportuje impurity importance.

    Důležitosti jsou pouze orientační (u kategorií s mnoha úrovněmi mohou být zvýhodněné).
    Nejsou kauzální a nejsou ekvivalentem změny pravděpodobnosti při zásahu do atributu.
    """
    y_all = parse_binary(df[target])
    valid = y_all.notna()
    if y_all[valid].nunique() < 2 or int(valid.sum()) < 100:
        logging.warning("Příliš málo označených řádků / tříd pro Random Forest (%s).", target)
        return None
    nums = [c for c in numeric_cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").nunique(dropna=True) > 1]
    cats = [c for c in categorical_cols if c in df.columns and df[c].nunique(dropna=True) > 1]
    cols = nums + cats
    if not cols:
        return None
    rows = df.index[valid].to_numpy()
    if len(rows) > max_rows:
        rng = np.random.default_rng(42026)
        selected = []
        y_valid = y_all.loc[rows]
        for klass in (0.0, 1.0):
            class_rows = y_valid[y_valid.eq(klass)].index.to_numpy()
            n_class = max(1, int(round(max_rows * len(class_rows) / len(rows))))
            selected.extend(rng.choice(class_rows, size=min(n_class, len(class_rows)), replace=False).tolist())
        rows = np.array(sorted(set(selected)))
    sub = df.loc[rows, cols].copy()
    y = y_all.loc[rows].astype(int)
    prep = get_preprocessor(nums, cats)
    X = prep.fit_transform(sub)
    # Omezujeme hloubku a velikost listu kvůli rychlosti/stabilitě na velkých kohortách.
    min_leaf = max(30, int(round(len(y) * 0.001)))
    forest = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=min_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    forest.fit(X, y)
    names = prep.get_feature_names_out()
    imp = pd.DataFrame({"encoded_term": names, "impurity_importance": forest.feature_importances_})
    imp["target"] = target
    imp["n_rows_fit"] = int(len(y))
    imp.sort_values("impurity_importance", ascending=False, inplace=True)
    write_csv(imp, outdir / f"forest_encoded_importance__{target}.csv")
    source = []
    for name in names:
        clean = str(name)
        if clean.startswith("num__"):
            src = clean[5:]
        elif clean.startswith("cat__"):
            rest = clean[5:]
            src = next((c for c in sorted(cats, key=len, reverse=True) if rest == c or rest.startswith(c + "_")), rest)
        else:
            src = clean
        source.append(src)
    imp["source_feature"] = source
    ranking = (imp.groupby("source_feature").agg(total_impurity_importance=("impurity_importance", "sum"),
                max_encoded_importance=("impurity_importance", "max"), number_of_encoded_terms=("encoded_term", "count"))
               .reset_index().sort_values("total_impurity_importance", ascending=False))
    ranking["target"] = target
    ranking["n_rows_fit"] = int(len(y))
    write_csv(ranking, outdir / f"forest_factor_ranking__{target}.csv")
    return ranking


def create_rule_segments(df: pd.DataFrame, target: str, outdir: Path, min_segment_size: int = 100) -> pd.DataFrame:
    """Interpretable segmenty z předem srozumitelných behaviorálních a demografických skupin."""
    group_cols = [c for c in ["AGE_BAND", "SCREENING_HISTORY_GROUP", "PREVENTION_TOTAL_BAND", "REGISTRATION_PROFILE"] if c in df.columns]
    if len(group_cols) < 2: return pd.DataFrame()
    y = parse_binary(df[target])
    tmp = df[group_cols].copy()
    for c in group_cols: tmp[c] = tmp[c].astype("string").fillna("missing")
    tmp["_Y"] = y
    tmp = tmp.dropna(subset=["_Y"])
    agg = tmp.groupby(group_cols, dropna=False, observed=True)["_Y"].agg(n="size", participants="sum", participation_rate="mean").reset_index()
    agg = agg[agg["n"] >= min_segment_size].copy()
    if agg.empty: return agg
    overall = float(tmp["_Y"].mean())
    low, high = wilson_interval(agg["participants"].to_numpy(), agg["n"].to_numpy())
    agg["participation_rate_pct"] = 100*agg["participation_rate"]
    agg["risk_difference_vs_overall_pp"] = 100*(agg["participation_rate"]-overall)
    agg["lift_vs_overall"] = agg["participation_rate"]/overall if overall else np.nan
    agg["ci95_low_pct"] = 100*low; agg["ci95_high_pct"] = 100*high
    # Score combines sample size and distance from the overall rate; not a causal priority score.
    agg["segment_priority_score"] = agg["risk_difference_vs_overall_pp"].abs() * np.sqrt(agg["n"])
    agg["target"] = target
    agg.sort_values(["segment_priority_score", "n"], ascending=False, inplace=True)
    write_csv(agg, outdir / f"rule_segments__{target}.csv")
    return agg


def build_cluster_matrix(df: pd.DataFrame, numeric_cols: Sequence[str], categorical_cols: Sequence[str],
                         sample_idx: np.ndarray, seed: int = 42):
    cols = list(numeric_cols) + list(categorical_cols)
    sample = df.loc[sample_idx, cols].copy()
    prep = get_preprocessor(numeric_cols, categorical_cols)
    Xs = prep.fit_transform(sample)
    # SVD reduces sparse one-hot space before clustering; target labels are never included.
    max_components = min(20, Xs.shape[1]-1, Xs.shape[0]-1)
    if max_components < 2:
        return None, None, None, None
    svd = TruncatedSVD(n_components=max_components, n_iter=7, random_state=seed)
    Zs = svd.fit_transform(Xs)
    return prep, svd, Zs, cols


def cluster_segments(df: pd.DataFrame, targets: Sequence[str], numeric_cols: Sequence[str], categorical_cols: Sequence[str],
                     outdir: Path, n_clusters: int = 5, sample_size: int = 100000, batch_size: int = 50000) -> Optional[pd.DataFrame]:
    cols = [c for c in list(numeric_cols) + list(categorical_cols) if c in df.columns]
    if len(cols) < 2 or len(df) < max(50, n_clusters*5):
        logging.warning("Málo vhodných dat pro clusterování; vynechávám.")
        return None
    numeric_cols = [c for c in numeric_cols if c in cols and pd.to_numeric(df[c], errors="coerce").nunique(dropna=True) > 1]
    categorical_cols = [c for c in categorical_cols if c in cols and df[c].nunique(dropna=True) > 1]
    rng = np.random.default_rng(2026)
    n_sample = min(len(df), sample_size)
    sample_idx = np.sort(rng.choice(df.index.to_numpy(), size=n_sample, replace=False)) if len(df) > n_sample else df.index.to_numpy()
    prep, svd, Zs, cols = build_cluster_matrix(df, numeric_cols, categorical_cols, sample_idx)
    if prep is None:
        logging.warning("Nelze vytvořit matici pro clustering.")
        return None
    clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=4096, n_init=5, max_iter=100,
                                reassignment_ratio=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clusterer.fit(Zs)
    # Přiřaď segment všem řádkům v dávkách, nikoli jednou velkou maticí.
    labels = np.empty(len(df), dtype=np.int32)
    positions = np.arange(len(df))
    for start in range(0, len(df), batch_size):
        stop = min(start + batch_size, len(df))
        batch = df.iloc[start:stop][cols]
        xb = prep.transform(batch)
        zb = svd.transform(xb)
        labels[start:stop] = clusterer.predict(zb)
        if stop == len(df) or stop % (batch_size*4) == 0:
            logging.info("Clusterování: přiřazeno %s / %s řádků", f"{stop:,}", f"{len(df):,}")
    out = pd.DataFrame({"_ROW_ID": df["_ROW_ID"].to_numpy(), "cluster": labels,
                        "INVITATION_DATE": df["INVITATION_DATE"].dt.strftime("%Y-%m-%d").to_numpy()})
    for target in targets:
        out[target] = parse_binary(df[target]).to_numpy() if target in df.columns else np.nan
    # assignments contain only source row number, date, cluster and outcome flags (never patient IDs).
    write_csv(out, outdir / "cluster_assignments.csv")

    profile_rows = []
    for cl, sub_idx in pd.Series(labels).groupby(pd.Series(labels)).groups.items():
        positions_cl = np.asarray(list(sub_idx), dtype=int)
        sub = df.iloc[positions_cl]
        row: Dict[str, Any] = {"cluster": int(cl), "n_rows": int(len(sub)), "share_rows_pct": 100*len(sub)/len(df)}
        if "AGE" in sub.columns:
            row["age_median"] = float(_to_num(sub, "AGE").median())
            row["age_mean"] = float(_to_num(sub, "AGE").mean())
        for c in ["AGE_BAND", "VZP_TERRITORIAL_AREA", "SCREENING_HISTORY_GROUP", "PREVENTION_TOTAL_BAND", "REGISTRATION_PROFILE", "HEALTHCARE_CONTACT_BAND"]:
            if c in sub.columns:
                modes = sub[c].astype("string").fillna("missing").value_counts(normalize=True)
                row[f"top_{c.lower()}"] = str(modes.index[0]) if len(modes) else None
                row[f"top_{c.lower()}_pct"] = 100*float(modes.iloc[0]) if len(modes) else np.nan
        for target in targets:
            y = parse_binary(sub[target])
            row[f"{target}_n_labeled"] = int(y.notna().sum())
            row[f"{target}_participation_rate_pct"] = 100*float(y.mean()) if y.notna().any() else np.nan
        profile_rows.append(row)
    profile = pd.DataFrame(profile_rows).sort_values("cluster")
    write_csv(profile, outdir / "cluster_profile.csv")
    # Save compact clustering metadata (no serialized transformer needed for this exploratory task).
    meta = {"clusters": int(n_clusters), "sample_used_for_cluster_fit": int(n_sample),
            "rows_assigned": int(len(df)), "svd_components": int(svd.n_components),
            "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
            "features_used": cols,
            "warning": "Klastry jsou exploratorní segmenty podle vstupních featur; labely nebyly použity k jejich vytvoření."}
    (outdir / "cluster_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return profile


def make_plots(df: pd.DataFrame, targets: Sequence[str], outdir: Path,
               factor_tables: Dict[str, pd.DataFrame], cluster_profile: Optional[pd.DataFrame]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning("Grafy přeskočeny, matplotlib není dostupný: %s", exc)
        return
    # Monthly participation trend.
    fig, ax = plt.subplots(figsize=(12, 6))
    for target in targets:
        tmp = pd.DataFrame({"date": df["INVITATION_DATE"], "y": parse_binary(df[target])}).dropna(subset=["y"])
        tmp["month"] = tmp["date"].dt.to_period("M").dt.to_timestamp()
        rates = tmp.groupby("month")["y"].mean()*100
        ax.plot(rates.index, rates.values, marker=".", linewidth=1.5, label=target)
    ax.set_title("Účast podle měsíce odeslání pozvánky")
    ax.set_xlabel("Měsíc pozvánky"); ax.set_ylabel("Účast (%)"); ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "plot_monthly_participation.png", dpi=150); plt.close(fig)

    # Top multivariable logistic source factors per target.
    for target, factors in factor_tables.items():
        if factors is None or factors.empty: continue
        p = factors.sort_values("max_abs_coefficient", ascending=True).tail(18)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(p["source_feature"].astype(str), p["max_abs_coefficient"])
        ax.set_title(f"Nejsilnější vysvětlující faktory – logistický model ({target})")
        ax.set_xlabel("Max. absolutní koeficient (exploratorní skóre)")
        fig.tight_layout(); fig.savefig(outdir / f"plot_logistic_factors__{target}.png", dpi=150); plt.close(fig)
    if cluster_profile is not None and not cluster_profile.empty:
        target = next((t for t in targets if f"{t}_participation_rate_pct" in cluster_profile.columns), None)
        if target:
            p = cluster_profile.sort_values("cluster")
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(p["cluster"].astype(str), p[f"{target}_participation_rate_pct"])
            ax.set_title(f"Účast podle clusteru ({target})")
            ax.set_xlabel("Cluster"); ax.set_ylabel("Účast (%)"); fig.tight_layout()
            fig.savefig(outdir / f"plot_cluster_participation__{target}.png", dpi=150); plt.close(fig)


def create_report(df: pd.DataFrame, targets: Sequence[str], profile: Dict[str, Any],
                  label_df: pd.DataFrame, factor_tables: Dict[str, pd.DataFrame], forest_tables: Dict[str, pd.DataFrame],
                  cat_assocs: Dict[str, pd.DataFrame], num_assocs: Dict[str, pd.DataFrame],
                  cluster_profile: Optional[pd.DataFrame], outdir: Path) -> None:
    lines = ["# Rakathon 2026 – automatický analytický report", "",
             "> Jde o průzkumnou analýzu asociací. Výsledky nejsou důkazem kauzality a samy o sobě neprokazují, že změna faktoru změní účast.", "",
             "## 1. Velikost a časové pokrytí", "",
             f"- Řádky po vyřazení neplatných dat pozvánky: **{profile.get('rows_after_valid_date_filter', 0):,}**".replace(",", " "),
             f"- Unikátní klientky podle dostupného ID: **{profile.get('unique_people', 'nelze zjistit')}**",
             f"- Unikátní ID pozvánek: **{profile.get('unique_invitation_ids', 'sloupec chybí')}**",
             f"- Rozsah dat pozvánek: **{profile.get('date_min')} – {profile.get('date_max')}**",
             f"- Klientky s více řádky: **{profile.get('people_with_multiple_rows', 'nelze zjistit')}**",
             f"- Podíl řádků patřících klientkám s více pozvánkami: **{100*profile.get('share_rows_belonging_to_repeat_people', 0):.2f}%**" if profile.get("share_rows_belonging_to_repeat_people") is not None else "- Opakované pozvánky: ID klientky nebylo možné použít.",
             "", "## 2. Základní prevalence cílových labelů", ""]
    if not label_df.empty:
        lines.append(label_df[["target", "horizon_days", "rows_labeled", "participants", "nonparticipants", "participation_rate_pct"]].round(3).to_markdown(index=False))
    lines += ["", "Interpretace: srovnávejte labely s ohledem na délku časového okna. Účast do 12 měsíců přirozeně může být vyšší než do 3 měsíců.", "",
              "## 3. Nejvýraznější kategoriální asociace", ""]
    for target in targets:
        a = cat_assocs.get(target)
        lines.append(f"### {target}")
        if a is None or a.empty:
            lines.append("Kategorie nebyly k dispozici.")
        else:
            lines.append(a[["feature", "cramers_v", "range_between_group_rates_pp", "min_group_rate_pct", "max_group_rate_pct"]].head(12).round(4).to_markdown(index=False))
        lines.append("")
    lines += ["## 4. Numerické atributy s nejsilnější monotónní asociací", ""]
    for target in targets:
        a = num_assocs.get(target)
        lines.append(f"### {target}")
        if a is None or a.empty:
            lines.append("Numerické asociace nebyly k dispozici.")
        else:
            lines.append(a[["feature", "spearman_rho_with_participation", "mean_among_participants", "mean_among_nonparticipants"]].head(12).round(4).to_markdown(index=False))
        lines.append("")
    lines += ["## 5. Víceproměnná logistická regrese – nejsilnější faktory", "",
              "Koeficienty jsou odhadnuty na reprodukovatelném vzorku označených řádků. Číselné proměnné jsou škálované; u kategorií jde o kontrast proti referenční úrovni. Jde o asociace po zohlednění ostatních featur v modelu, nikoli o kauzální účinky.", ""]
    for target in targets:
        a = factor_tables.get(target)
        lines.append(f"### {target}")
        if a is None or a.empty:
            lines.append("Model nebyl natrénován nebo nevrátil koeficienty.")
        else:
            lines.append(a[["source_feature", "max_abs_coefficient", "rms_coefficient", "number_of_encoded_terms"]].head(15).round(4).to_markdown(index=False))
        lines.append("")
    lines += ["## 6. Nelineární model (Random Forest) – důležitosti faktorů", "",
              "Random Forest může zachytit nelinearity a interakce, které logistická regrese s lineárními členy nezachytí. Zde uváděné impurity importance jsou orientační a mohou zvýhodňovat atributy s více kategoriemi; ověřujte je proti univariačním tabulkám, logistické regresi a stabilitě v čase.", ""]
    for target in targets:
        a = forest_tables.get(target)
        lines.append(f"### {target}")
        if a is None or a.empty:
            lines.append("Model nebyl natrénován nebo nevrátil důležitosti.")
        else:
            lines.append(a[["source_feature", "total_impurity_importance", "max_encoded_importance", "number_of_encoded_terms"]].head(15).round(5).to_markdown(index=False))
        lines.append("")
    lines += ["## 7. Interpretovatelné segmenty", "",
              "Soubor `rule_segments__*.csv` obsahuje kombinace věkové skupiny, historie screeningu v posledních 5 letech, celkového prevenčního chování a registrace u praktika/gynekologa. Drobné skupiny jsou filtrovány; pořadí zohledňuje velikost segmentu i odchylku účasti od průměru.", "",
              "## 8. Segmentace pomocí clusterů", ""]
    if cluster_profile is not None and not cluster_profile.empty:
        show_cols = [c for c in ["cluster", "n_rows", "share_rows_pct", "age_median", "top_screening_history_group", "top_prevention_total_band", "top_registration_profile"] + [f"{targets[-1]}_participation_rate_pct" if targets else ""] if c in cluster_profile.columns]
        lines.append(cluster_profile[show_cols].round(3).to_markdown(index=False))
        lines.append("")
        lines.append("Clustering používá pouze prediktorové featury a nesleduje labely při tvorbě skupin. Labely se připojí až pro profilaci clusterů. Klastry jsou statistické segmenty, nikoli automaticky klinicky smysluplné typy klientek.")
    else:
        lines.append("Clustering byl přeskočen nebo nebyl proveditelný.")
    lines += ["", "## 9. Nově vytvořené featury", "",
              "- Cyklické časové proměnné měsíce / dne v týdnu a roční období.",
              "- Předchozí pozvánky a počet předchozích dat pozvánek, odstup od poslední a od první pozvánky; řádky stejného dne se navzájem nepočítají do historie.",
              "- Prevenční součet, počet oblastí prevence s aktivitou a explicitní počet chybějících prevenčních atributů.",
              "- Binární indikátory prevence, skupiny intenzity předchozího screeningu, recence screeningu, kontaktů se zdravotnictvím.",
              "- Kombinovaný profil registrace u praktika/gynekologa a čitelný behaviorální profil pro segmentaci.",
              "", "## 10. Důležité metodické limity", "",
              "1. Opakované pozvánky jedné klientky znamenají, že řádky nejsou nezávislé. Zde se vztahy zkoumají na úrovni pozvánky; pro intervaly spolehlivosti a testy hypotéz by bylo třeba zohlednit shlukování podle osoby.",
              "2. U velmi velkého N mohou být p-hodnoty miniaturní i při slabém vztahu. Sledujte velikost efektu (Cramérovo V, rozdíly v procentních bodech, lift), stabilitu v čase a velikost skupiny.",
              "3. Missing neznamená automaticky nulu. Součty prevence používají nulu pouze jako pomocný výpočet, zároveň je vytvořena featura `PREVENTION_DATA_MISSING_COUNT`.",
              "4. `MONTHS_SINCE_LAST_SCREEN` a `NEVER_SCREENED_BEFORE_FLAG` mohou popisovat pouze vymezené pětileté pozorovací okno; neinterpretujte je bez ověření definice jako celoživotní historii.",
              "5. Pozvánky v novějších obdobích musí mít dokončené celé 90/180/365denní okno. Neúplné labely mohou vychýlit sazby; ověřte datum úplnosti vykázaných výkonů.",
              "6. Některé featury vznikají přímo z původních indikátorů, proto jejich jednotlivé efekty nejsou nezávislé. Logistická regrese je regularizovaná, ale multikolinearita interpretaci stále ovlivňuje.",
              "", "## 11. Kde hledat podrobné výstupy", "",
              "- `missingness.csv`, `numeric_summary.csv`: kvalita dat a distribuce.",
              "- `categorical_rates__*.csv`, `numeric_bin_rates__*.csv`: konkrétní skupiny / intervaly s mírou účasti, počty a odchylkou od průměru.",
              "- `categorical_associations__*.csv`, `numeric_associations__*.csv`: souhrn síly univariačních asociací.",
              "- `logistic_terms__*.csv`, `logistic_factor_ranking__*.csv`: víceproměnné koeficienty a pořadí vstupních faktorů.",
              "- `forest_encoded_importance__*.csv`, `forest_factor_ranking__*.csv`: nelineární stromové důležitosti (orientační).",
              "- `rule_segments__*.csv`: srozumitelné kombinace faktorů.",
              "- `cluster_profile.csv`, `cluster_assignments.csv`: popis clusterů a přiřazení každého řádku.",
              "- `monthly_participation_trend.csv` a `plot_*.png`: časový trend a grafy.", ""]
    (outdir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Podrobná EDA, hledání asociací a segmentace Rakathon datasetu.")
    p.add_argument("--input", required=True, help="CSV, Parquet nebo Excel dataset.")
    p.add_argument("--output-dir", default="rakathon_analysis_out", help="Složka pro reporty, CSV a grafy.")
    p.add_argument("--target", choices=list(TARGET_HORIZONS), default=None, help="Volitelně analyzovat jeden label; jinak všechny dostupné.")
    p.add_argument("--patient-id-column", default="PERSON_ID", help="ID pro historické featury; ID nikdy není modelový vstup.")
    p.add_argument("--max-model-rows", type=int, default=80000, help="Vzorek pro každý interpretační logistický model.")
    p.add_argument("--cluster-rows", type=int, default=100000, help="Vzorek pro učení clusterů; segment se přiřadí všem řádkům.")
    p.add_argument("--clusters", type=int, default=5, help="Počet clusterů.")
    p.add_argument("--min-segment-size", type=int, default=100, help="Minimální počet řádků v pravidlovém segmentu.")
    p.add_argument("--min-category-count", type=int, default=30, help="Kategorie s menším počtem řádků se při testu asociace sloučí.")
    p.add_argument("--skip-models", action="store_true", help="Přeskočí víceproměnnou logistickou regresi.")
    p.add_argument("--skip-clustering", action="store_true", help="Přeskočí MiniBatchKMeans segmentaci.")
    p.add_argument("--no-plots", action="store_true", help="Nevytváří PNG grafy.")
    p.add_argument("--verbose", action="store_true", help="Detailnější logování.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    if args.clusters < 2:
        raise ValueError("--clusters musí být alespoň 2.")
    if args.max_model_rows < 100:
        raise ValueError("--max-model-rows musí být alespoň 100.")
    if args.cluster_rows < 100:
        raise ValueError("--cluster-rows musí být alespoň 100.")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    logging.info("Načítám %s", args.input)
    raw = load_table(args.input)
    original_columns = [str(c).strip().upper() for c in raw.columns]
    logging.info("Načteno %s řádků × %s sloupců", f"{len(raw):,}", len(raw.columns))
    df, id_col = engineer_features(raw, patient_id_column=args.patient_id_column)
    targets = available_targets(df, args.target)
    # Vytvoření metadat/profilu neukládá hodnoty identifikátorů.
    profile = data_profile(df, original_columns, id_col, outdir)
    num_cols, cat_cols = feature_columns(df)
    numeric_summary(df, [c for c in NUMERIC_BASE + NUMERIC_ENGINEERED if c in df.columns], outdir)
    label_df = label_summary(df, targets, outdir)

    cat_assocs: Dict[str, pd.DataFrame] = {}
    num_assocs: Dict[str, pd.DataFrame] = {}
    factors: Dict[str, pd.DataFrame] = {}
    forest_factors: Dict[str, pd.DataFrame] = {}
    for target in targets:
        logging.info("Analyzuji vztahy k labelu %s", target)
        cat_assocs[target] = categorical_associations(df, target, cat_cols, outdir, min_group_count=args.min_category_count)
        num_assocs[target] = numeric_associations(df, target, num_cols, outdir)
        seg = create_rule_segments(df, target, outdir, min_segment_size=args.min_segment_size)
        if not seg.empty:
            logging.info("Pro %s vytvořeno %s pravidlových segmentů", target, len(seg))
        if args.skip_models:
            factors[target] = pd.DataFrame()
            forest_factors[target] = pd.DataFrame()
        else:
            factor = fit_interpretable_logistic(df, target, num_cols, cat_cols, outdir, max_rows=args.max_model_rows)
            factors[target] = factor if factor is not None else pd.DataFrame()
            forest_factor = fit_nonlinear_forest(df, target, num_cols, cat_cols, outdir, max_rows=args.max_model_rows)
            forest_factors[target] = forest_factor if forest_factor is not None else pd.DataFrame()

    cluster_profile = None
    if not args.skip_clustering:
        logging.info("Spouštím unsupervised segmentaci (%s clusterů)", args.clusters)
        cluster_profile = cluster_segments(df, targets, num_cols, cat_cols, outdir,
                                            n_clusters=args.clusters, sample_size=args.cluster_rows)

    if not args.no_plots:
        make_plots(df, targets, outdir, factors, cluster_profile)
    create_report(df, targets, profile, label_df, factors, forest_factors, cat_assocs, num_assocs, cluster_profile, outdir)

    logging.info("Hotovo. Výstupy jsou ve složce: %s", outdir.resolve())
    logging.info("Základní prevalence labelů:\n%s", label_df.to_string(index=False))
    for target in targets:
        if target in cat_assocs and not cat_assocs[target].empty:
            logging.info("Top kategoriální asociace %s:\n%s", target,
                         cat_assocs[target][["feature", "cramers_v", "range_between_group_rates_pp"]].head(5).to_string(index=False))
        if target in factors and not factors[target].empty:
            logging.info("Top víceproměnné faktory %s:\n%s", target,
                         factors[target][["source_feature", "max_abs_coefficient"]].head(5).to_string(index=False))
        if target in forest_factors and not forest_factors[target].empty:
            logging.info("Top nelineární stromové faktory %s:\n%s", target,
                         forest_factors[target][["source_feature", "total_impurity_importance"]].head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Přerušeno uživatelem.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        logging.exception("Analýza skončila chybou: %s", exc)
        raise SystemExit(1)
