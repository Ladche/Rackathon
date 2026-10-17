"""Synthetic-only tests, run with the repository's existing pytest tool."""
from __future__ import annotations

import importlib
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    add_profiles, care_profile, connect, history_profile, read_cohort, save_table,
    source, summarize, wilson, write_json,
)

audit = importlib.import_module("01_audit")
features = importlib.import_module("02_features")
groups = importlib.import_module("03_groups")


def raw_rows(dates=("2020-01-01", "2021-01-01", "2022-01-01")):
    rows = []
    for i, date in enumerate(dates):
        row = {name: "0" for name in source.NUMERIC}
        row.update({
            "invitation_id": f"SYNTH-I-{i}", "person_id": "SYNTH-P-1", "invitation_date": date,
            "age_band": "50-54", "territory": "SYNTH-AREA",
            "gp_reg_duration_band": ">5", "gyn_reg_duration_band": ">5",
            "gp_registered": "A", "gyn_registered": "A", "first_invitation": "A" if i == 0 else "N",
            "no_screen_observed_5y": "A", "months_since_last_screen": "",
            "screening_90d": "N", "screening_180d": "N", "screening_365d": "N",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def normalized(rows):
    return source.normalize_schema(rows, source.Config(date_format="%Y-%m-%d"), require_targets=True)


def workspace(tmp_path, rows):
    private = tmp_path / "private"
    private.mkdir()
    frame = normalized(rows)
    frame["invitation_date"] = frame["invitation_date"].dt.strftime("%Y-%m-%d")
    with sqlite3.connect(private / "analysis.sqlite") as con:
        frame.to_sql("invitations", con, index=False)
    write_json(tmp_path / "audit.json", {"status": "validated"})
    features.build_features(tmp_path)
    return tmp_path


@pytest.mark.parametrize("contacts,expected", [(0, "low_contact"), (2, "low_contact"), (3, "care_no_prevention")])
def test_care_thresholds(contacts, expected):
    frame = normalized(raw_rows()).iloc[:1].copy()
    frame["healthcare_contact_12m_cnt"] = contacts
    assert care_profile(frame).iloc[0] == expected


def test_care_does_not_replace_unknown_by_zero():
    frame = normalized(raw_rows()).iloc[:1].copy()
    frame["gp_prevention_24m_cnt"] = np.nan
    frame["dental_prevention_24m_cnt"] = 1
    assert care_profile(frame).iloc[0] == "unknown"
    frame["gyn_prevention_24m_cnt"] = 1
    assert care_profile(frame).iloc[0] == "medical_prevention"


def test_care_never_adds_different_window_counts():
    frame = normalized(raw_rows()).iloc[:1].copy()
    frame["other_screening_36m_cnt"] = 100
    assert care_profile(frame).iloc[0] == "other_prevention"
    frame["gp_prevention_24m_cnt"] = 1
    assert care_profile(frame).iloc[0] == "medical_prevention"


def test_history_strict_boundary_and_no_current_or_future_labels(tmp_path):
    dates = ["2020-01-01", "2020-12-31", "2021-01-01", "2021-01-30", "2021-01-31", "2022-01-01"]
    workspace(tmp_path, raw_rows(dates))
    with connect(tmp_path) as con:
        frame = pd.read_sql_query("SELECT * FROM features ORDER BY invitation_date", con)
    assert frame["prior_known_0"].tolist()[:3] == [0, 0, 1]
    assert frame["prior_known_30"].tolist()[:5] == [0, 0, 0, 0, 1]
    assert frame["observed_prior"].tolist() == list(range(6))
    assert frame["prior_known_30"].iloc[0] == 0


def test_missing_past_outcome_not_a_nonresponse(tmp_path):
    rows = raw_rows()
    rows.loc[0, "screening_365d"] = ""
    workspace(tmp_path, rows)
    with connect(tmp_path) as con:
        frame = read_cohort(con)
    assert frame["prior_known_30"].iloc[0] == 0
    assert frame["history_profile"].iloc[0] == "unobserved"


def test_known_response_with_unknown_timing_is_not_delayed(tmp_path):
    rows = raw_rows()
    rows.loc[0, "screening_90d"] = ""
    rows.loc[0, "screening_180d"] = ""
    rows.loc[0, "screening_365d"] = "A"
    workspace(tmp_path, rows)
    with connect(tmp_path) as con:
        frame = read_cohort(con)
    assert frame["history_profile"].iloc[0] == "response_timing_unknown"


def test_fixed_year_cohort_does_not_condition_on_future_invitations(tmp_path):
    rows = raw_rows(["2020-01-01", "2021-01-01", "2022-01-01"])
    workspace(tmp_path, rows)
    with connect(tmp_path) as con:
        frame = read_cohort(con, year=2021)
    assert len(frame) == 1
    assert frame["invitation_date"].dt.year.iloc[0] == 2021
    assert frame["observed_prior"].iloc[0] == 1


def test_safety_override_and_current_outcome_invariance(tmp_path):
    workspace(tmp_path, raw_rows())
    with connect(tmp_path) as con:
        frame = read_cohort(con)
    mapping = groups.discover_mapping(frame, minimum=500)
    original = groups.assign(frame, mapping)
    changed = frame.copy()
    for target in source.TARGETS.values():
        changed[target] = 1 - changed[target]
    assert groups.assign(changed, mapping).equals(original)
    changed["months_since_last_screen"] = 0
    assert groups.assign(changed, mapping).iloc[0] == "recent_screen"


def test_sparse_pooling_is_target_independent(tmp_path):
    workspace(tmp_path, raw_rows())
    with connect(tmp_path) as con:
        frame = read_cohort(con)
    first = groups.discover_mapping(frame)
    for target in source.TARGETS.values():
        frame[target] = 1
    assert groups.discover_mapping(frame) == first
    assert list(first.values()) == ["single_zero__no_prevention"]


def test_audit_detects_duplicates_across_chunk_boundary(tmp_path):
    frame = raw_rows(["01.01.2020", "01.01.2021"])
    frame.loc[1, "invitation_id"] = frame.loc[0, "invitation_id"]
    data = tmp_path / "synthetic.csv"
    frame.to_csv(data, index=False)
    with pytest.raises(sqlite3.IntegrityError):
        audit.audit(data, tmp_path / "results", chunk_size=1)
    assert not (tmp_path / "results" / "audit.json").exists()


def test_audit_rejects_duplicate_headers(tmp_path):
    data = tmp_path / "synthetic.csv"
    data.write_text("PERSON_ID,PERSON_ID\nSYNTH-1,SYNTH-1\n")
    with pytest.raises(source.ContractError):
        audit.audit(data, tmp_path / "results")


def test_export_rejects_identifiers_and_suppresses_small_rows(tmp_path):
    with pytest.raises(ValueError, match="identifiers"):
        save_table(tmp_path, "bad.csv", pd.DataFrame({"person_id": ["SYNTH-1"]}))
    save_table(tmp_path, "aggregate.csv", pd.DataFrame({"cell": ["a", "b"], "n": [19, 20]}))
    assert pd.read_csv(tmp_path / "aggregate.csv")["cell"].tolist() == ["b"]


def test_summary_preserves_missing_outcomes_and_wilson_denominator():
    frame = pd.DataFrame({
        "person_id": ["SYNTH-1", "SYNTH-2"],
        "screening_90d": [0, np.nan], "screening_180d": [1, np.nan], "screening_365d": [1, np.nan],
    })
    row = summarize(frame, []).iloc[0]
    assert row["n"] == 2
    assert row["known_365d"] == 1
    assert row["rate_365d"] == 1
    low, high = wilson(0, 100)
    assert 0 <= low < high < .04
    assert math.isnan(wilson(0, 0)[0])


def test_power_scenarios_are_feasible_and_multiplicity_costs_sample():
    assert groups.sample_size(.2, .05, .05 / 19) > groups.sample_size(.2, .05)
    assert groups.sample_size(.99, .05) is None
    assert groups.sample_size(.2, .02) > groups.sample_size(.2, .05)


def test_partial_audit_database_cannot_be_consumed(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    sqlite3.connect(private / "analysis.sqlite").close()
    with pytest.raises(ValueError, match="successful audit"):
        connect(tmp_path)


def test_current_and_future_labels_cannot_change_earlier_features(tmp_path):
    rows = raw_rows()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    workspace(a, rows)
    rows.loc[1:, list(source.TARGETS.values())] = "A"
    workspace(b, rows)
    with connect(a) as con:
        before = pd.read_sql_query("SELECT * FROM features ORDER BY invitation_date", con)
    with connect(b) as con:
        after = pd.read_sql_query("SELECT * FROM features ORDER BY invitation_date", con)
    history_columns = [c for c in before if c.startswith(("prior_known_", "prior_yes_", "prior_early_", "prior_late_"))]
    pd.testing.assert_frame_equal(before[history_columns], after[history_columns])
