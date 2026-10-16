from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

import mammography_pipeline as m


@pytest.mark.parametrize("value,expected", [
    ("A", 1), ("N", 0), ("Y", 1), ("n", 0), (True, 1), (False, 0),
    ("1", 1), (0, 0), ("  a ", 1), ("NULL", np.nan), ("", np.nan), (None, np.nan),
])
def test_binary(value, expected):
    result = m.parse_binary(pd.Series([value]), "test").iloc[0]
    assert np.isnan(result) if np.isnan(expected) else result == expected


@pytest.mark.parametrize("value", ["unknown", "ANO", 2, -1, "yes"])
def test_binary_unknown_fails(value):
    with pytest.raises(m.ContractError, match="Unsupported binary"):
        m.parse_binary(pd.Series([value]), "test")


@pytest.mark.parametrize("names", [
    ["AGE_BAND", " age_band "], ["screening_3m_flag", "screening_90d"], ["A", "A"],
])
def test_collision(names):
    with pytest.raises(m.ContractError, match="collision"):
        m.canonical_names(names, {})


def test_alias_override_fails():
    with pytest.raises(m.ContractError, match="Ambiguous"):
        m.canonical_names(["SCREENING_3M_FLAG"], {"screening_3m_flag": "something_else"})


def test_csv_ids_format_encoding(tmp_path, cfg):
    p = tmp_path / "sample.csv"
    p.write_text("INVITATION_ID;PERSON_ID;INVITATION_DATE;VZP_TERRITORIAL_AREA\n0001;0000009;31.01.2020;Area\n", encoding="utf-8-sig")
    result = m.load_csv(replace(cfg, data=str(p), separator=";", date_format="%d.%m.%Y"))
    assert result["invitation_id"].iloc[0] == "0001"
    assert result["person_id"].iloc[0] == "0000009"
    assert result["invitation_date"].iloc[0] == pd.Timestamp("2020-01-31")


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.drop(columns="person_id"), "Missing required"),
    (lambda d: d.assign(person_id=""), "Missing identifier"),
    (lambda d: d.assign(invitation_date="not-a-date"), "Invalid invitation"),
    (lambda d: d.assign(gp_prevention_24m_cnt=-1), "negative"),
    (lambda d: d.assign(gp_prevention_24m_cnt=.5), "Nonintegral"),
    (lambda d: d.assign(gp_prevention_24m_cnt=np.inf), "Nonfinite"),
    (lambda d: d.assign(age_band="unverified"), "Unexpected age"),
    (lambda d: d.assign(gp_reg_duration_band="long"), "Unexpected documented"),
    (lambda d: d.assign(prior_screen_count_5y=1., no_screen_observed_5y=1.), "Contradictory five-year"),
    (lambda d: d.assign(no_screen_observed_5y=1., prior_screen_count_5y=0., months_since_last_screen=12.), "contradicts recorded"),
    (lambda d: d.assign(gp_registered=0., gp_reg_duration_band=">5"), "Unregistered"),
    (lambda d: d.assign(invitation_id="DUP"), "Duplicate/conflicting"),
    (lambda d: d.assign(person_id="same", invitation_date=pd.Timestamp("2020-01-01")), "same-person"),
])
def test_schema_failures(synthetic, cfg, mutate, match):
    with pytest.raises(m.ContractError, match=match):
        m.normalize_schema(mutate(synthetic.iloc[:2].copy()), cfg)


def test_duplicate_indices_reset(synthetic, cfg):
    small = synthetic.iloc[:2].copy()
    small.index = [8, 8]
    assert m.normalize_schema(small, cfg).index.tolist() == [0, 1]


def test_all_targets_not_required_in_prediction(synthetic, cfg):
    no_labels = synthetic.drop(columns=list(m.TARGETS.values()))
    m.normalize_schema(no_labels, cfg)
    with pytest.raises(m.ContractError, match="Missing required"):
        m.normalize_schema(no_labels, cfg, require_targets=True)


def test_non_nested_labels_fail(synthetic, cfg):
    small = synthetic.iloc[:1].copy()
    small["screening_90d"], small["screening_365d"] = 1., 0.
    with pytest.raises(m.ContractError, match="Contradictory"):
        m.normalize_schema(small, cfg)


def test_schema_unknowns_excluded(synthetic, cfg):
    synthetic["future_label_proxy"] = 999
    report = m.schema_report(synthetic)
    unknown = next(r for r in report if r["name"] == "future_label_proxy")
    assert unknown["inclusion"] == "excluded"
    names, availability = m.resolve_feature_set("all_safe", synthetic, cfg)
    assert not (set(m.KEYS) | set(m.TARGETS.values()) | m.AMBIGUOUS | {"future_label_proxy"}) & set(names)
    assert all(next(a for a in availability if a["name"] == c)["status"] == "unavailable" for c in m.AMBIGUOUS)


def test_full_sentinel_is_never_opened(tmp_path, cfg, monkeypatch):
    sentinel = tmp_path / "Rakhaton2026.csv"
    sentinel.touch()
    from pathlib import Path
    original = Path.open

    def guarded(self, *args, **kwargs):
        assert self.resolve() != sentinel.resolve(), "Full sentinel was opened!"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(m.ContractError, match="Refusing full"):
        m.load_csv(replace(cfg, data=str(sentinel)))
    with pytest.raises(m.ContractError, match="requires"):
        m.load_csv(replace(cfg, data=str(sentinel), data_mode="full"))


def test_small_chunk_full_path_matches_materialized(tmp_path, synthetic, cfg):
    p = tmp_path / "synthetic.csv"
    synthetic.to_csv(p, index=False)
    sample = m.load_csv(replace(cfg, data=str(p), chunk_size=10000))
    chunked = m.load_csv(replace(cfg, data=str(p), data_mode="full", allow_full_data=True, chunk_size=7))
    pd.testing.assert_frame_equal(sample, chunked)
    names, _ = m.resolve_feature_set("longitudinal_no_labels", sample, cfg)
    pd.testing.assert_frame_equal(m.build_features(sample, cfg, names), m.build_features(chunked, cfg, names))


@pytest.mark.parametrize("settings", [
    {"reporting_lag_days": -1}, {"folds": 0}, {"n_jobs": 0}, {"memory_mb": 0},
    {"chunk_size": 0}, {"budget_seconds": 0}, {"test_start": "2020-01-01"},
    {"validation_end": "2020-01-01"}, {"test_year": 2024, "test_start": "2024-01-01", "test_end": "2024-12-31"},
    {"sample_cap": 0}, {"separator": ",;"}, {"max_trials": 0}, {"calibration_days": 0},
])
def test_invalid_config(cfg, settings):
    with pytest.raises(m.ContractError):
        replace(cfg, **settings).validate()


def test_whole_patient_sample(tmp_path, synthetic, cfg):
    p = tmp_path / "synthetic.csv"
    synthetic.to_csv(p, index=False)
    result = m.load_csv(replace(cfg, data=str(p), sample_cap=3))
    assert result["person_id"].nunique() == 3
    assert result.groupby("person_id").size().tolist() == [10, 10, 10]
    assert len(result) == 30


def test_memory_guard(tmp_path, synthetic, cfg):
    p = tmp_path / "synthetic.csv"
    pd.concat([synthetic] * 10).to_csv(p, index=False)
    with pytest.raises(m.ContractError, match="memory guard"):
        m.load_csv(replace(cfg, data=str(p), memory_mb=1, chunk_size=100))


def test_mapping_file_errors(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('["wrong"]')
    args = m.parse_args(["inspect", "--schema-mapping", str(p)])
    with pytest.raises(m.ContractError, match="JSON object"):
        m.resolve_config(args)
    p.write_text(json.dumps({"person": "person_id"}))
    assert m.resolve_config(args).schema_mapping == {"person": "person_id"}
