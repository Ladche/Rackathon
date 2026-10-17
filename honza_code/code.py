#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rakathon 2026 – leak-safe baseline pro predikci účasti na mamografickém screeningu.

Jednosouborový skript s režimy train / evaluate / predict.

Příklady:
  # Trénink + vyhodnocení na posledním roce, model se uloží:
  python rakathon_mammo_model.py --input dataset.csv --mode train --target SCREENING_12M_FLAG --save-model model_12m.joblib

  # Samostatné vyhodnocení uloženého modelu:
  python rakathon_mammo_model.py --input dataset.csv --mode evaluate --load-model model_12m.joblib

  # Predikce (vstup musí obsahovat historii klientek, pokud ji chcete využít):
  python rakathon_mammo_model.py --input novy_dataset.csv --mode predict --load-model model_12m.joblib --output predikce.csv

  # Výslovné datum začátku testovacího období a datum, do kdy jsou labely pozorovány:
  python rakathon_mammo_model.py --input dataset.csv --mode train --test-start 2023-12-01 \\
      --observation-end-date 2025-12-31 --target SCREENING_12M_FLAG --save-model model.joblib

Instalace závislostí:
  pip install pandas numpy scikit-learn joblib

Poznámky k metodice:
- INVITATION_DATE je čas T0; surové datum ani ID klientky nejsou vstupem modelu.
- Vstupní prediktory jsou definovány allow-listem. Cílové labely ani neznámé sloupce se nepoužijí.
- Historické featury používají pouze záznamy se striktně dřívějším datem než T0.
- Test tvoří poslední období dle data pozvánky. Tréninkové řádky se navíc purgují tak,
  aby celé cílové okno skončilo nejpozději na začátku testu (label maturity / temporal purge).
- Standardně se z tréninku vyřadí klientky, které se vyskytují v testu (--allow-patient-overlap
  tuto přísnější kontrolu vypne). Je to konzervativní test generalizace na neviděné klientky.
- Skript spoléhá na správnost dodaných labelů. Pokud labely nejsou dostupné až do konce
  cílového okna, nastavte --observation-end-date; jinak mohou být pozdní testovací labely neúplné.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_HORIZON_DAYS: Dict[str, int] = {
    "SCREENING_3M_FLAG": 90,
    "SCREENING_6M_FLAG": 180,
    "SCREENING_12M_FLAG": 365,
}

TARGET_COLUMNS = list(TARGET_HORIZON_DAYS)
IDENTIFIER_COLUMNS = {"INVITATION_ID", "RC", "PERSON_ID"}

# Používáme allow-list známých prediktorů. Neznámé sloupce se automaticky ignorují,
# což chrání před náhodným použitím budoucího labelu či ID v nové verzi souboru.
STATIC_NUMERIC_FEATURES = [
    "AGE",
    "HEALTHCARE_CONTACT_12M_CNT",
    "INSURANCE_DURATION_YEARS",
    "INSURANCE_DURATION_DAYS",
    "REG_DURATION",
    "GYN_REG_DURATION_YEARS",
    "PRIOR_SCREEN_COUNT_5Y",
    "MONTHS_SINCE_LAST_SCREEN",
    "GP_PREVENTION_24M_CNT",
    "GYN_PREVENTION_24M_CNT",
    "OTHER_SCREENING_36M_CNT",
    "DENTAL_PREVENTION_24M_CNT",
]
STATIC_CATEGORICAL_FEATURES = [
    "AGE_BAND",
    "VZP_TERRITORIAL_AREA",
    "INSURANCE_DURATION_BAND",
    "GP_REGISTERED",
    "GP_REG_DURATION_BAND",
    "GYN_REGISTERED_FLAG",
    "GYN_REG_DURATION_BAND",
    "NEVER_SCREENED_BEFORE_FLAG",
    "FIRST_INVITATION",
]
TIME_FEATURES = [
    "INVITATION_YEAR",
    "INVITATION_MONTH",
    "INVITATION_QUARTER",
    "INVITATION_DAY_OF_WEEK",
    "INVITATION_MONTH_SIN",
    "INVITATION_MONTH_COS",
    "INVITATION_DOW_SIN",
    "INVITATION_DOW_COS",
]
HISTORY_SOURCE_FEATURES = [
    "PRIOR_SCREEN_COUNT_5Y",
    "GP_PREVENTION_24M_CNT",
    "GYN_PREVENTION_24M_CNT",
    "OTHER_SCREENING_36M_CNT",
    "DENTAL_PREVENTION_24M_CNT",
    "HEALTHCARE_CONTACT_12M_CNT",
]
# Tyto sloupce jsou v popisu datasetu sice přítomné, ale jejich přesná semantika
# (počítají-li i právě zpracovávanou pozvánku) nemusí být zřejmá. Proto je znovu
# vytváříme z pořadí dat, místo abychom použili dodané čítače.
RECOMPUTED_OR_AMBIGUOUS_COLUMNS = {
    "PRIOR_INVITATION_COUNT",
    "INVITATION_SEQUENCE",
    "MONTHS_LAST_INVITATION",
}
ENGINEERED_HISTORY_FEATURES = [
    "HIST_PRIOR_INVITATION_COUNT",
    "HIST_MONTHS_SINCE_PRIOR_INVITATION",
    "HIST_DAYS_SINCE_FIRST_OBSERVED_INVITATION",
]

NUMERIC_MODEL_CANDIDATES = set(STATIC_NUMERIC_FEATURES + TIME_FEATURES + ENGINEERED_HISTORY_FEATURES)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_table(path: str) -> pd.DataFrame:
    """Načte CSV, Parquet nebo Excel; ID při CSV načítá jako text."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Vstupní soubor neexistuje: {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        # Výslovně jako string, aby se ID omylem nezměnilo na float / neztratilo přesnost.
        return pd.read_csv(
            p,
            low_memory=False,
            encoding="utf-8-sig",
            dtype={"RC": "string", "PERSON_ID": "string", "INVITATION_ID": "string"},
        )
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError("Nepodporovaný formát vstupu. Použijte CSV, Parquet nebo Excel.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().upper() for c in out.columns]
    if out.columns.duplicated().any():
        duplicates = out.columns[out.columns.duplicated()].tolist()
        raise ValueError(f"Po normalizaci názvů vznikly duplicitní sloupce: {duplicates}")
    return out


def choose_patient_key_column(df: pd.DataFrame, requested: str) -> Optional[str]:
    requested = str(requested or "PERSON_ID").strip().upper()
    if requested in df.columns:
        return requested
    for fallback in ("PERSON_ID", "RC"):
        if fallback in df.columns:
            logging.warning("Sloupec %s není dostupný; pro historické featury použiji %s.", requested, fallback)
            return fallback
    logging.warning("Nenašel se žádný identifikátor klientky. Historické featury budou počítány po řádcích.")
    return None


def _numeric_or_nan(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def build_features(raw: pd.DataFrame, patient_id_column: str = "PERSON_ID") -> Tuple[pd.DataFrame, str, Optional[str]]:
    """Vytvoří časové a historické featury pouze z informace dostupné k T0.

    Vrací kompletní rámec s původními labely (pro vyhodnocení), interní skupinový klíč
    a skutečný sloupec ID použitý pro historii. Interní klíč není featurou modelu.
    """
    df = normalize_columns(raw)
    if "INVITATION_DATE" not in df.columns:
        raise ValueError("Chybí povinný sloupec INVITATION_DATE.")

    df = df.copy()
    df["_SOURCE_ROW"] = np.arange(len(df), dtype=np.int64)
    df["INVITATION_DATE"] = pd.to_datetime(df["INVITATION_DATE"], errors="coerce")
    invalid_dates = int(df["INVITATION_DATE"].isna().sum())
    if invalid_dates:
        logging.warning("Vyřazuji %s řádků s neplatným INVITATION_DATE.", f"{invalid_dates:,}")
        df = df.loc[df["INVITATION_DATE"].notna()].copy()
    if df.empty:
        raise ValueError("Po vyřazení neplatných dat nezbyly žádné řádky.")

    # Časové featury (bez surového data jako vstupu modelu).
    dt = df["INVITATION_DATE"]
    df["INVITATION_YEAR"] = dt.dt.year.astype("float64")
    df["INVITATION_MONTH"] = dt.dt.month.astype("float64")
    df["INVITATION_QUARTER"] = dt.dt.quarter.astype("float64")
    df["INVITATION_DAY_OF_WEEK"] = dt.dt.dayofweek.astype("float64")
    df["INVITATION_MONTH_SIN"] = np.sin(2.0 * np.pi * dt.dt.month / 12.0)
    df["INVITATION_MONTH_COS"] = np.cos(2.0 * np.pi * dt.dt.month / 12.0)
    df["INVITATION_DOW_SIN"] = np.sin(2.0 * np.pi * dt.dt.dayofweek / 7.0)
    df["INVITATION_DOW_COS"] = np.cos(2.0 * np.pi * dt.dt.dayofweek / 7.0)

    id_col = choose_patient_key_column(df, patient_id_column)
    # Chybějící ID nesmí všechny spadnout do jedné fiktivní klientky; každý takový řádek
    # dostane vlastní skupinu a žádnou historii jiné klientky nezdědí.
    if id_col is None:
        df["_PATIENT_KEY"] = "__ROW__" + df["_SOURCE_ROW"].astype(str)
    else:
        ids = df[id_col].astype("string").str.strip()
        missing = ids.isna() | ids.eq("")
        ids = ids.astype("object")
        ids.loc[missing] = "__MISSING_ROW__" + df.loc[missing, "_SOURCE_ROW"].astype(str)
        df["_PATIENT_KEY"] = ids.astype(str)

    # Date-level agregace je důležitá: více řádků stejné klientky ve stejný den
    # si nesmí navzájem předávat informace.
    key_cols = ["_PATIENT_KEY", "INVITATION_DATE"]
    date_counts = (
        df.groupby(key_cols, dropna=False, sort=False)
        .size()
        .rename("_N_ROWS_ON_DATE")
        .reset_index()
        .sort_values(key_cols, kind="mergesort")
        .reset_index(drop=True)
    )
    grouped_counts = date_counts.groupby("_PATIENT_KEY", sort=False)["_N_ROWS_ON_DATE"]
    date_counts["HIST_PRIOR_INVITATION_COUNT"] = grouped_counts.cumsum() - date_counts["_N_ROWS_ON_DATE"]
    prev_date = date_counts.groupby("_PATIENT_KEY", sort=False)["INVITATION_DATE"].shift(1)
    date_counts["HIST_MONTHS_SINCE_PRIOR_INVITATION"] = (
        (date_counts["INVITATION_DATE"] - prev_date).dt.total_seconds() / (86400.0 * 30.4375)
    )
    first_date = date_counts.groupby("_PATIENT_KEY", sort=False)["INVITATION_DATE"].transform("min")
    date_counts["HIST_DAYS_SINCE_FIRST_OBSERVED_INVITATION"] = (
        date_counts["INVITATION_DATE"] - first_date
    ).dt.total_seconds() / 86400.0

    # Historické snapshoty povolených prediktorů z dřívějších pozvánek.
    # Neagregujeme žádný label a aktuální řádek se nikdy nezapočítá do vlastních lagů.
    existing_history_sources = [c for c in HISTORY_SOURCE_FEATURES if c in df.columns]
    for col in existing_history_sources:
        temp = df[key_cols + [col]].copy()
        temp[col] = _numeric_or_nan(temp[col])
        per_date = (
            temp.groupby(key_cols, dropna=False, sort=False)[col]
            .mean()
            .rename(col)
            .reset_index()
        )
        date_counts = date_counts.merge(per_date, on=key_cols, how="left", validate="one_to_one", sort=False)
        prior_last = date_counts.groupby("_PATIENT_KEY", sort=False)[col].shift(1)
        # Expanding mean před aktuálním datem, počítaná přes kumulativní součet / počet známých hodnot.
        vals = date_counts[col]
        vals0 = vals.fillna(0.0)
        cumulative_sum = vals0.groupby(date_counts["_PATIENT_KEY"], sort=False).cumsum()
        valid = vals.notna().astype(np.int64)
        cumulative_count = valid.groupby(date_counts["_PATIENT_KEY"], sort=False).cumsum()
        prior_sum = cumulative_sum - vals0
        prior_count = cumulative_count - valid
        safe_mean = prior_sum / prior_count.replace(0, np.nan)
        date_counts[f"HIST_PRIOR_LAST_{col}"] = prior_last
        date_counts[f"HIST_PRIOR_MEAN_{col}"] = safe_mean

    hist_cols = [
        "HIST_PRIOR_INVITATION_COUNT",
        "HIST_MONTHS_SINCE_PRIOR_INVITATION",
        "HIST_DAYS_SINCE_FIRST_OBSERVED_INVITATION",
    ]
    for col in existing_history_sources:
        hist_cols.extend([f"HIST_PRIOR_LAST_{col}", f"HIST_PRIOR_MEAN_{col}"])

    lookup_cols = key_cols + hist_cols
    df = df.merge(
        date_counts[lookup_cols],
        on=key_cols,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    df = df.sort_values("_SOURCE_ROW", kind="mergesort").reset_index(drop=True)

    # Sjednotíme známé číselné featury (případné neparsovatelné hodnoty jdou do imputace).
    numeric_cols = set(STATIC_NUMERIC_FEATURES + TIME_FEATURES + ENGINEERED_HISTORY_FEATURES)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _numeric_or_nan(df[col])
    # Jednotný stringový typ kategorií před OneHotEncoderem; zabrání chybě při směsi
    # hodnot typu číslo/text (například věkové pásmo načtené z různých exportů).
    for col in STATIC_CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].map(lambda value: np.nan if pd.isna(value) else str(value).strip())
    return df, "_PATIENT_KEY", id_col


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    known = set(STATIC_NUMERIC_FEATURES + STATIC_CATEGORICAL_FEATURES + TIME_FEATURES + ENGINEERED_HISTORY_FEATURES)
    for col in HISTORY_SOURCE_FEATURES:
        known.add(f"HIST_PRIOR_LAST_{col}")
        known.add(f"HIST_PRIOR_MEAN_{col}")
    # ID, datum, labely, původní nejednoznačné čítače ani interní sloupce nejsou ve výběru.
    forbidden = set(IDENTIFIER_COLUMNS) | set(TARGET_COLUMNS) | RECOMPUTED_OR_AMBIGUOUS_COLUMNS
    forbidden |= {"INVITATION_DATE", "_SOURCE_ROW", "_PATIENT_KEY"}
    columns = [c for c in df.columns if c in known and c not in forbidden]
    if not columns:
        raise ValueError("Nenašly se žádné známé prediktory. Zkontrolujte názvy sloupců v datasetu.")
    return columns


def parse_binary_label(series: pd.Series) -> pd.Series:
    """Mapuje A/N, Ano/Ne, bool či 0/1 na 1/0; neznámé hodnoty zůstanou NaN."""
    yes = {"A", "ANO", "YES", "Y", "TRUE", "T", "1", "1.0"}
    no = {"N", "NE", "NO", "FALSE", "F", "0", "0.0"}

    def one(value: Any) -> float:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        if isinstance(value, (int, np.integer, float, np.floating)):
            if value == 1:
                return 1.0
            if value == 0:
                return 0.0
            return np.nan
        value_s = str(value).strip().upper()
        if value_s in yes:
            return 1.0
        if value_s in no:
            return 0.0
        return np.nan

    return series.map(one).astype("float64")


def make_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = [c for c in X.columns if c in NUMERIC_MODEL_CANDIDATES or c.startswith("HIST_PRIOR_LAST_") or c.startswith("HIST_PRIOR_MEAN_")]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # kompatibilita se staršími scikit-learn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]
    )
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)
    return preprocessor, numeric_cols, categorical_cols


def build_model(X: pd.DataFrame, random_state: int, max_iter: int, class_weight: Optional[str]) -> Tuple[Pipeline, List[str], List[str]]:
    preprocessor, numeric_cols, categorical_cols = make_preprocessor(X)
    # SAGA optimalizuje logistickou regresi přímo nad řídkou one-hot maticí;
    # oproti základní SGD je vhodnější baseline i pro pravděpodobnostní výstup.
    classifier = LogisticRegression(
        solver="saga",
        C=1.0,
        max_iter=max_iter,
        tol=1e-3,
        random_state=random_state,
        class_weight=class_weight,
    )
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    return model, numeric_cols, categorical_cols


def determine_cutoff(df: pd.DataFrame, test_years: int, explicit_start: Optional[str]) -> pd.Timestamp:
    if explicit_start:
        cutoff = pd.Timestamp(explicit_start)
    else:
        max_date = pd.Timestamp(df["INVITATION_DATE"].max())
        cutoff = max_date - pd.DateOffset(years=test_years)
    return pd.Timestamp(cutoff).normalize()


def make_temporal_split(
    df: pd.DataFrame,
    target: str,
    cutoff: pd.Timestamp,
    horizon_days: int,
    group_col: str,
    allow_patient_overlap: bool,
    observation_end_date: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Vrátí poziční indexy train/test s purgem délky cílového okna."""
    dates = pd.to_datetime(df["INVITATION_DATE"])
    y = parse_binary_label(df[target])
    test_mask = dates.ge(cutoff)
    train_mask = dates.add(pd.to_timedelta(horizon_days, unit="D")).le(cutoff)

    if observation_end_date:
        observation_end = pd.Timestamp(observation_end_date).normalize()
        matured = dates.add(pd.to_timedelta(horizon_days, unit="D")).le(observation_end)
        test_mask &= matured
        logging.info("Label maturity filtr: testovací řádky musí mít konec okna nejpozději %s.", observation_end.date())

    # Hodnoty labelu neznámé nebo nekódované jako A/N či 0/1 nejdou do metrik.
    train_mask &= y.notna()
    test_mask &= y.notna()

    groups_test = set(df.loc[test_mask, group_col].astype(str).unique())
    if not allow_patient_overlap and groups_test:
        train_mask &= ~df[group_col].astype(str).isin(groups_test)

    train_idx = np.flatnonzero(train_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())
    stats = {
        "cutoff": cutoff.isoformat(),
        "horizon_days": int(horizon_days),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "max_train_date": str(dates.iloc[train_idx].max().date()) if len(train_idx) else None,
        "min_test_date": str(dates.iloc[test_idx].min().date()) if len(test_idx) else None,
        "group_overlap_allowed": bool(allow_patient_overlap),
        "test_unique_groups": int(df.iloc[test_idx][group_col].nunique()) if len(test_idx) else 0,
    }
    if not allow_patient_overlap and len(train_idx) and len(test_idx):
        train_groups = set(df.iloc[train_idx][group_col].astype(str).unique())
        test_groups = set(df.iloc[test_idx][group_col].astype(str).unique())
        stats["train_test_group_overlap"] = int(len(train_groups.intersection(test_groups)))
    return train_idx, test_idx, stats


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
    title: str = "Vyhodnocení",
    metrics_out: Optional[str] = None,
    top_features: int = 20,
) -> Dict[str, Any]:
    valid = y.notna()
    Xv = X.loc[valid]
    yv = y.loc[valid].astype(int)
    if len(yv) == 0:
        raise ValueError("Ve vyhodnocovací sadě nejsou žádné známé labely.")
    if yv.nunique() < 2:
        logging.warning("Vyhodnocovací sada obsahuje pouze jednu třídu; ROC-AUC / PR-AUC nemusí být definované.")

    if hasattr(model, "predict_proba"):
        classes = list(model.named_steps["classifier"].classes_)
        positive_index = classes.index(1) if 1 in classes else int(np.argmax(classes))
        proba = model.predict_proba(Xv)[:, positive_index]
    else:
        raise RuntimeError("Model neumí predict_proba; tento skript očekává probabilistický klasifikátor.")
    pred = (proba >= threshold).astype(int)

    result: Dict[str, Any] = {
        "title": title,
        "n": int(len(yv)),
        "positive_rate": float(yv.mean()),
        "mean_predicted_probability": float(np.mean(proba)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(yv, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yv, pred)),
        "precision": float(precision_score(yv, pred, zero_division=0)),
        "recall": float(recall_score(yv, pred, zero_division=0)),
        "f1": float(f1_score(yv, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(yv, proba)),
        "log_loss": float(log_loss(yv, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "confusion_matrix_labels_0_1": confusion_matrix(yv, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(yv, pred, labels=[0, 1], target_names=["N (0)", "A (1)"], zero_division=0, output_dict=True),
    }
    try:
        result["roc_auc"] = float(roc_auc_score(yv, proba))
    except ValueError:
        result["roc_auc"] = None
    try:
        result["average_precision_pr_auc"] = float(average_precision_score(yv, proba))
    except ValueError:
        result["average_precision_pr_auc"] = None

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Počet řádků:               {result['n']:,}")
    print(f"Podíl pozitivních labelů:  {result['positive_rate']:.4f}")
    print(f"Průměr predikované prav.:  {result['mean_predicted_probability']:.4f}")
    for key, label in [
        ("roc_auc", "ROC-AUC"),
        ("average_precision_pr_auc", "Average precision / PR-AUC"),
        ("brier_score", "Brier score (nižší je lepší)"),
        ("log_loss", "Log loss (nižší je lepší)"),
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("precision", f"Precision při threshold={threshold:.2f}"),
        ("recall", f"Recall při threshold={threshold:.2f}"),
        ("f1", f"F1 při threshold={threshold:.2f}"),
    ]:
        value = result.get(key)
        print(f"{label:34s} {value if value is not None else 'n/a'}")
    print("Matice záměn (řádky=skutečnost 0/1, sloupce=predikce 0/1):")
    print(np.asarray(result["confusion_matrix_labels_0_1"]))
    print("\nClassification report:")
    print(classification_report(yv, pred, labels=[0, 1], target_names=["N (0)", "A (1)"], zero_division=0))

    # Koeficienty logistické regrese: orientační asociace, ne kauzální efekt.
    if top_features > 0:
        try:
            clf = model.named_steps["classifier"]
            pre = model.named_steps["preprocessor"]
            if hasattr(clf, "coef_"):
                names = pre.get_feature_names_out()
                coefs = clf.coef_[0]
                k = min(top_features, len(coefs))
                idx = np.argsort(np.abs(coefs))[-k:][::-1]
                print(f"\nTop {k} koeficientů podle absolutní hodnoty (orientační):")
                for i in idx:
                    print(f"  {names[i]}: {coefs[i]:+.5f}")
        except Exception as exc:  # vyhodnocení nemá selhat kvůli volitelnému reportu koeficientů
            logging.debug("Nepodařilo se vypsat koeficienty: %s", exc)

    if metrics_out:
        out_path = Path(metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)
        logging.info("Metriky zapsány do: %s", out_path)
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Nelze serializovat typ {type(value)} do JSON.")


def save_artifact(path: str, artifact: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out, compress=3)
    logging.info("Model a metadata uloženy do: %s", out)


def load_artifact(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model neexistuje: {p}")
    artifact = joblib.load(p)
    if not isinstance(artifact, dict) or "model" not in artifact or "target" not in artifact:
        raise ValueError("Soubor neobsahuje očekávaný artefakt tohoto skriptu.")
    return artifact


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leak-safe trénink a vyhodnocení predikce účasti na mamografickém screeningu.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Vstupní dataset (CSV, Parquet nebo Excel).")
    parser.add_argument("--mode", choices=["train", "evaluate", "predict"], default="train", help="Režim běhu.")
    parser.add_argument("--target", choices=TARGET_COLUMNS, default=None, help="Cílový label; při tréninku je výchozí 12M.")
    parser.add_argument("--load-model", default=None, help="Cesta k uloženému modelu (evaluate/predict).")
    parser.add_argument("--save-model", default=None, help="Volitelná cesta pro uložení natrénovaného modelu.")
    parser.add_argument("--output", default="predictions.csv", help="Výstupní CSV pro režim predict.")
    parser.add_argument("--predict-from-date", default=None, help="V režimu predict vypíše jen pozvánky od data YYYY-MM-DD; starší řádky se přesto použijí pro historii.")
    parser.add_argument("--test-years", type=int, default=1, help="Délka testovacího období v letech, počítaná zpětně od max. data.")
    parser.add_argument("--test-start", default=None, help="Explicitní začátek testu, například 2023-12-01.")
    parser.add_argument("--observation-end-date", default=None, help="Poslední datum kompletně pozorovaných labelů (volitelné, YYYY-MM-DD).")
    parser.add_argument("--patient-id-column", default="PERSON_ID", help="Sloupec pro historii klientky; nikdy se nepoužije jako prediktor.")
    parser.add_argument("--allow-patient-overlap", action="store_true", help="Povolí stejnou klientku v train i test; výchozí nastavení je přísně bez překryvu.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Práh pro převod pravděpodobnosti na třídu při vyhodnocení.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximální počet iterací solveru logistické regrese.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed pro reprodukovatelnost.")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none", help="Váhy tříd; balanced často zvýší recall, ale zhorší kalibraci pravděpodobností.")
    parser.add_argument("--metrics-out", default=None, help="Volitelná cesta pro JSON s metrikami.")
    parser.add_argument("--top-features", type=int, default=20, help="Kolik nejsilnějších koeficientů vypsat; 0 vypne.")
    parser.add_argument("--verbose", action="store_true", help="Detailní logování.")
    args = parser.parse_args(argv)
    if args.test_years < 1:
        parser.error("--test-years musí být celé číslo >= 1.")
    if not (0.0 < args.threshold < 1.0):
        parser.error("--threshold musí být mezi 0 a 1.")
    if args.mode in {"evaluate", "predict"} and not args.load_model:
        parser.error(f"Režim --mode {args.mode} vyžaduje --load-model.")
    if args.mode == "predict" and not args.output:
        parser.error("Režim predict vyžaduje --output.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    logging.info("Python %s | pandas %s | scikit-learn %s", sys.version.split()[0], pd.__version__, sklearn.__version__)
    logging.info("Načítám dataset: %s", args.input)
    raw = load_table(args.input)
    logging.info("Načteno %s řádků a %s sloupců.", f"{len(raw):,}", len(raw.columns))
    if args.mode in {"train", "evaluate"} and not args.observation_end_date:
        logging.warning("Není zadáno --observation-end-date. Předpokládám, že dodané labely jsou kompletní; ověřte úplné follow-up okno zejména pro poslední pozvánky.")

    artifact: Optional[Dict[str, Any]] = None
    if args.mode in {"evaluate", "predict"}:
        artifact = load_artifact(args.load_model)
        target = args.target or artifact["target"]
        if args.mode == "evaluate" and target != artifact["target"]:
            raise ValueError(f"Model byl trénován pro {artifact['target']}, ne pro {target}.")
        if args.mode == "predict" and args.target and args.target != artifact["target"]:
            logging.warning("Zadaný target se nepoužije; uložený model predikuje %s.", artifact["target"])
            target = artifact["target"]
        # Použijeme stejný feature engineering jako při tréninku.
        feature_frame, group_col, actual_id_col = build_features(raw, artifact.get("patient_id_column", args.patient_id_column))
        feature_cols = artifact["feature_columns"]
        X = feature_frame.reindex(columns=feature_cols)
        model = artifact["model"]

        if args.mode == "predict":
            proba = model.predict_proba(X)
            classes = list(model.named_steps["classifier"].classes_)
            positive_index = classes.index(1) if 1 in classes else int(np.argmax(classes))
            prediction_mask = pd.Series(True, index=feature_frame.index)
            if args.predict_from_date:
                prediction_mask &= feature_frame["INVITATION_DATE"].ge(pd.Timestamp(args.predict_from_date).normalize())
            rows_to_write = np.flatnonzero(prediction_mask.to_numpy())
            output = pd.DataFrame({
                "SOURCE_ROW": feature_frame.iloc[rows_to_write]["_SOURCE_ROW"].to_numpy(),
                "INVITATION_DATE": feature_frame.iloc[rows_to_write]["INVITATION_DATE"].dt.strftime("%Y-%m-%d").to_numpy(),
                "PREDICTED_PROBABILITY": proba[rows_to_write, positive_index],
                "PREDICTED_FLAG_AT_THRESHOLD": np.where(proba[rows_to_write, positive_index] >= args.threshold, "A", "N"),
            })
            # Pro kontrolní párování přidáme technické ID, pokud existuje. Není součástí modelu.
            id_col = artifact.get("patient_id_column")
            if id_col and id_col in feature_frame.columns:
                output.insert(1, id_col, feature_frame.iloc[rows_to_write][id_col].astype("string").to_numpy())
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(out, index=False, encoding="utf-8-sig")
            logging.info("Predikce (%s řádků) zapsány do %s", f"{len(output):,}", out)
            return 0

        # evaluate: test začíná na cut-off uloženém při tréninku, pokud nebyl zadán nový.
        if target not in feature_frame.columns:
            raise ValueError(f"V datasetu chybí cílový sloupec {target}.")
        if args.test_start:
            cutoff = pd.Timestamp(args.test_start).normalize()
        elif artifact.get("split_cutoff"):
            cutoff = pd.Timestamp(artifact["split_cutoff"]).normalize()
        else:
            cutoff = determine_cutoff(feature_frame, args.test_years, None)
        dates = pd.to_datetime(feature_frame["INVITATION_DATE"])
        y = parse_binary_label(feature_frame[target])
        test_mask = dates.ge(cutoff) & y.notna()
        if args.observation_end_date:
            mature = dates.add(pd.to_timedelta(TARGET_HORIZON_DAYS[target], unit="D")).le(pd.Timestamp(args.observation_end_date).normalize())
            test_mask &= mature
        X_test = X.loc[test_mask].reset_index(drop=True)
        y_test = y.loc[test_mask].reset_index(drop=True)
        if X_test.empty:
            raise ValueError("Testovací sada je prázdná. Zkontrolujte --test-start, labely a observation-end-date.")
        logging.info("Vyhodnocení od %s; %s testovacích řádků.", cutoff.date(), f"{len(X_test):,}")
        evaluate_model(model, X_test, y_test, args.threshold, f"TEST – {target} od {cutoff.date()}", args.metrics_out, args.top_features)
        return 0

    # Režim train
    target = args.target or "SCREENING_12M_FLAG"
    if target not in normalize_columns(raw).columns:
        raise ValueError(f"V datasetu chybí cílový sloupec {target}.")
    data, group_col, actual_id_col = build_features(raw, args.patient_id_column)
    feature_cols = get_feature_columns(data)
    X_all = data[feature_cols].copy()
    y_all = parse_binary_label(data[target])
    cutoff = determine_cutoff(data, args.test_years, args.test_start)
    horizon_days = TARGET_HORIZON_DAYS[target]
    train_idx, test_idx, split_stats = make_temporal_split(
        data,
        target,
        cutoff,
        horizon_days,
        group_col,
        args.allow_patient_overlap,
        args.observation_end_date,
    )
    logging.info("Časový split: %s", json.dumps(split_stats, ensure_ascii=False))
    if len(train_idx) == 0:
        raise ValueError("Trénovací sada je prázdná po temporal purge / group splitu. Zvolte jiné --test-start nebo --allow-patient-overlap.")
    if len(test_idx) == 0:
        raise ValueError("Testovací sada je prázdná. Zkontrolujte datum, cílový label a observation-end-date.")

    X_train = X_all.iloc[train_idx].reset_index(drop=True)
    y_train = y_all.iloc[train_idx].astype(int).reset_index(drop=True)
    X_test = X_all.iloc[test_idx].reset_index(drop=True)
    y_test = y_all.iloc[test_idx].astype(int).reset_index(drop=True)
    if y_train.nunique() < 2:
        raise ValueError("Trénovací sada obsahuje pouze jednu třídu; nelze natrénovat binární klasifikátor.")

    class_weight = None if args.class_weight == "none" else "balanced"
    model, numeric_cols, categorical_cols = build_model(X_train, args.random_state, args.max_iter, class_weight)
    logging.info("Trénuji na %s řádcích; numeric=%s, categorical=%s, prediktorů=%s.",
                 f"{len(X_train):,}", len(numeric_cols), len(categorical_cols), len(feature_cols))
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test, args.threshold, f"TEMPORAL TEST – {target} od {cutoff.date()}", args.metrics_out, args.top_features)

    if args.save_model:
        saved = {
            "model": model,
            "target": target,
            "target_horizon_days": horizon_days,
            "feature_columns": feature_cols,
            "numeric_feature_columns": numeric_cols,
            "categorical_feature_columns": categorical_cols,
            "split_cutoff": cutoff.isoformat(),
            "split_stats": split_stats,
            "patient_id_column": actual_id_col,
            "class_weight": class_weight,
            "threshold_default": args.threshold,
            "random_state": args.random_state,
            "sklearn_version": sklearn.__version__,
            "training_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
        }
        save_artifact(args.save_model, saved)
    else:
        logging.info("Model nebyl uložen (nezadali jste --save-model).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logging.error("Přerušeno uživatelem.")
        raise SystemExit(130)
    except Exception as exc:
        logging.exception("Chyba: %s", exc)
        raise SystemExit(1)
