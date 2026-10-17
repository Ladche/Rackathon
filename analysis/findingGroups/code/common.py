"""Shared descriptive-analysis contracts; no estimators or clustering."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mammography_pipeline as source  # noqa: E402

HORIZONS = (90, 180, 365)
CARE_LEVELS = ("low_contact", "care_no_prevention", "other_prevention", "medical_prevention", "unknown")
HISTORY_LEVELS = ("repeated_zero", "single_zero", "delayed_response", "early_response", "response_timing_unknown", "unobserved")


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--results", type=Path, default=BASE / "results")
    return result


def connect(results: Path) -> sqlite3.Connection:
    path = results / "private" / "analysis.sqlite"
    if not path.is_file():
        raise FileNotFoundError(f"Run 01_audit.py first: {path}")
    audit = results / "audit.json"
    if not audit.is_file() or read_json(audit).get("status") != "validated":
        raise ValueError("No successful audit manifest; a partial database cannot be analyzed.")
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((BASE / "code").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update((ROOT / "mammography_pipeline.py").read_bytes())
    return digest.hexdigest()


def wilson(successes: int, n: int) -> tuple[float, float]:
    if not n:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - radius, center + radius


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Wilson intervals apply to the one-invitation-per-person main cohort."""
    rows = []
    groups = frame.groupby(keys, dropna=False, observed=True, sort=True) if keys else [((), frame)]
    for values, group in groups:
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(keys, values))
        row.update(n=len(group), people=group["person_id"].nunique())
        for horizon in HORIZONS:
            outcome = group[f"screening_{horizon}d"].dropna()
            n, successes = len(outcome), int(outcome.sum())
            low, high = wilson(successes, n)
            row.update({
                f"known_{horizon}d": n, f"attended_{horizon}d": successes,
                f"rate_{horizon}d": successes / n if n else math.nan,
                f"ci_low_{horizon}d": low, f"ci_high_{horizon}d": high,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def save_table(results: Path, name: str, table: pd.DataFrame, minimum_cell: int = 20) -> None:
    """Do not publish small group rows; no identifiers are accepted."""
    if {"person_id", "invitation_id"}.intersection(table):
        raise ValueError("Patient-level identifiers must not be exported.")
    original = table
    if "n" in table:
        table = table.loc[table["n"].ge(minimum_cell)].copy()
    table.to_csv(results / name, index=False, float_format="%.8f")
    manifest_path = results / "export_suppression.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    manifest[name] = {
        "minimum_row_support": minimum_cell, "input_rows": len(original), "published_rows": len(table),
        "suppressed_rows": len(original) - len(table),
        "suppressed_support": int(original["n"].sum() - table["n"].sum()) if "n" in table else 0,
    }
    write_json(manifest_path, manifest)


def care_profile(frame: pd.DataFrame, low_contact: int = 2) -> pd.Series:
    domains = list(source.DOMAINS)
    values = frame[domains]
    medical = values[domains[:2]].gt(0).any(axis=1)
    medical_zero = values[domains[:2]].eq(0).all(axis=1)
    other = values[domains[2:]].gt(0).any(axis=1)
    known = values.notna().all(axis=1)
    contacts = frame["healthcare_contact_12m_cnt"]
    return pd.Series(np.select(
        [medical, medical_zero & other, known & contacts.le(low_contact), known & contacts.gt(low_contact)],
        ["medical_prevention", "other_prevention", "low_contact", "care_no_prevention"],
        default="unknown",
    ), index=frame.index, dtype="string")


def history_profile(frame: pd.DataFrame, lag: int = 30) -> pd.Series:
    known = frame[f"prior_known_{lag}"]
    responses = frame[f"prior_yes_{lag}"]
    early = frame[f"prior_early_{lag}"]
    # Any past response wins over a zero-response pattern; no labels from T0 enter.
    return pd.Series(np.select(
        [early.gt(0), frame[f"prior_late_{lag}"].gt(0), responses.gt(0), known.ge(2), known.eq(1)],
        ["early_response", "delayed_response", "response_timing_unknown", "repeated_zero", "single_zero"],
        default="unobserved",
    ), index=frame.index, dtype="string")


def add_profiles(frame: pd.DataFrame, low_contact: int = 2, lag: int = 30) -> pd.DataFrame:
    frame = frame.copy()
    frame["care_profile"] = care_profile(frame, low_contact)
    frame["history_profile"] = history_profile(frame, lag)
    months = frame["months_since_last_screen"]
    frame["screen_recency"] = np.select(
        [frame["no_screen_observed_5y"].eq(1), months.lt(24), months.lt(36), months.lt(60), months.ge(60)],
        ["none_lookback", "under_24m", "24_35m", "36_59m", "60m_plus"], default="unknown",
    )
    frame["invitation_status"] = frame["first_invitation"].map({1: "source_first", 0: "source_repeat"}).fillna("unknown")
    frame["registration_profile"] = np.select(
        [frame["gp_registered"].eq(1) & frame["gyn_registered"].eq(1),
         frame["gp_registered"].eq(1) & frame["gyn_registered"].eq(0),
         frame["gp_registered"].eq(0) & frame["gyn_registered"].eq(1),
         frame["gp_registered"].eq(0) & frame["gyn_registered"].eq(0)],
        ["both", "gp_only", "gyn_only", "neither"], default="unknown",
    )
    return frame


def read_cohort(connection: sqlite3.Connection, policy: str = "index_year", year: int | None = None) -> pd.DataFrame:
    where = ""
    if policy == "index_year":
        if year is None:
            year = int(connection.execute("SELECT MAX(substr(invitation_date,1,4)) FROM features").fetchone()[0])
        where = f"WHERE substr(invitation_date,1,4)='{int(year):04d}'"
        policy = "latest"
    order = {"latest": "DESC", "earliest": "ASC"}.get(policy)
    if order is None:
        raise ValueError("Cohort policy must be latest or earliest.")
    query = f"""
        SELECT * FROM (
            SELECT f.*, ROW_NUMBER() OVER (
                PARTITION BY person_id ORDER BY invitation_date {order}
            ) AS person_rank
            FROM features f {where}
        ) WHERE person_rank=1
    """
    frame = pd.read_sql_query(query, connection, parse_dates=["invitation_date"])
    for col in ["age_band", "territory", "gp_reg_duration_band", "gyn_reg_duration_band"]:
        frame[col] = frame[col].fillna("unknown").astype("category")
    return add_profiles(frame)
