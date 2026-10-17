from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import mammography_pipeline as m


def test_hand_features(synthetic, cfg):
    d = synthetic.iloc[:1].copy()
    d["gp_prevention_24m_cnt"] = 2.
    d["gyn_prevention_24m_cnt"] = 4.
    d["dental_prevention_24m_cnt"] = 0.
    d["other_screening_36m_cnt"] = 3.
    d["healthcare_contact_12m_cnt"] = 8.
    d["months_since_last_screen"] = 36.
    d["gp_registered"] = 1.
    features = m.build_row_features(d)
    assert features["active_prevention_domains"].iloc[0] == 3
    assert features["prevention_contact_intensity_ratio"].iloc[0] == .5
    assert features["screen_gap_24m"].iloc[0] == 12
    assert features["screen_recency_bin"].iloc[0] == "24-36"
    assert features["gp_prevention_24m_cnt__log1p"].iloc[0] == pytest.approx(np.log(3))
    assert features["month_sin"].iloc[0] == 0
    d["healthcare_contact_12m_cnt"] = 0.
    d["gyn_prevention_24m_cnt"] = np.nan
    features = m.build_row_features(d)
    assert np.isnan(features["prevention_contact_intensity_ratio"].iloc[0])
    assert np.isnan(features["active_prevention_domains"].iloc[0])
    assert features["missing_prevention_domains"].iloc[0] == 1


def history_fixture(cfg):
    d = pd.DataFrame({
        "invitation_id": ["a", "b", "c", "d"], "person_id": ["p"] * 4,
        "invitation_date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"]),
        "healthcare_contact_12m_cnt": [10., 20., 5., 100.],
        "screening_90d": [0., 1., 0., 1.], "screening_180d": [0., 1., 1., 1.],
        "screening_365d": [1., 1., 1., 1.],
    })
    return m.normalize_schema(d, cfg)


def test_history_hand_calculation(cfg):
    d = history_fixture(cfg)
    x = m.build_patient_history(d, cfg)
    assert x["observed_prior_rows"].tolist() == [0, 1, 2, 3]
    assert np.isnan(x["healthcare_contact_12m_cnt__lag1"].iloc[0])
    assert x["healthcare_contact_12m_cnt__lag1"].iloc[2] == 20
    assert x["healthcare_contact_12m_cnt__lag2"].iloc[2] == 10
    assert x["healthcare_contact_12m_cnt__past_mean"].iloc[3] == pytest.approx(35/3)
    assert x["healthcare_contact_12m_cnt__delta"].iloc[2] == -15
    assert x["days_since_previous_observed_invitation"].iloc[1] == 366
    assert x["prior_interval_mean"].iloc[2] == 366
    assert np.isnan(x["prior_interval_mean"].iloc[1])
    assert x["observed_invitation_density"].iloc[0] == 0


def test_future_and_input_order_invariance(cfg):
    d = history_fixture(cfg)
    names, _ = m.resolve_feature_set("all_safe", d, cfg)
    x = m.build_features(d.iloc[:3], cfg, names)
    longer = d.copy()
    longer.loc[3, "healthcare_contact_12m_cnt"] = 999999
    actual = m.build_features(longer, cfg, names).iloc[:3]
    pd.testing.assert_frame_equal(x, actual)
    permuted = longer.iloc[[2, 0, 3, 1]].copy()
    shuffled = m.build_features(permuted, cfg, names)
    pd.testing.assert_frame_equal(m.build_features(longer, cfg, names).sort_index(), shuffled.sort_index())
    fixed = m.build_preprocessor(x, "logistic", cfg).fit(x)
    a, b = fixed.transform(x), fixed.transform(actual)
    np.testing.assert_allclose(a.toarray() if hasattr(a, "toarray") else a, b.toarray() if hasattr(b, "toarray") else b)


def test_default_targets_have_no_effect(cfg):
    d = history_fixture(cfg)
    names, _ = m.resolve_feature_set("all_safe", d, cfg)
    original = m.build_features(d, cfg, names)
    d[list(m.TARGETS.values())] = 1
    pd.testing.assert_frame_equal(original, m.build_features(d, cfg, names))


def test_immature_outcome_not_used(cfg):
    cfg = replace(cfg, historical_outcomes="on")
    d = history_fixture(cfg).iloc[:2].copy()
    d.loc[1, "invitation_date"] = pd.Timestamp("2020-02-01")
    a = m.build_patient_history(d, cfg)
    d.loc[0, list(m.TARGETS.values())] = 1
    b = m.build_patient_history(d, cfg)
    pd.testing.assert_frame_equal(a, b)
    assert a["historical_90d_mature_count"].iloc[1] == 0
    d.loc[1, "invitation_date"] = pd.Timestamp("2020-04-01")
    later = m.build_patient_history(d, cfg)
    assert later["historical_90d_mature_count"].iloc[1] == 1
    assert later["historical_90d_last_nonattendance"].iloc[1] == 0


@pytest.mark.parametrize("horizon", m.HORIZONS)
@pytest.mark.parametrize("lag", [0, 7])
def test_exclusive_maturity_boundary(horizon, lag, synthetic, cfg):
    cfg = replace(cfg, reporting_lag_days=lag)
    d = synthetic.iloc[:1].copy()
    d["invitation_date"] = pd.Timestamp("2020-02-29")
    available = pd.Timestamp("2020-02-29") + pd.Timedelta(days=horizon+lag)
    assert not m.train_mask(d, horizon, available, cfg).iloc[0]
    assert m.train_mask(d, horizon, available + pd.Timedelta(days=1), cfg).iloc[0]
    d[m.TARGETS[horizon]] = np.nan
    assert not m.observed_mask(d, horizon, cfg).iloc[0]


def test_folds_label_deadlines_and_patient_overlap(synthetic, cfg):
    manifest = m.reserve_test(synthetic, cfg)
    folds = m.make_temporal_folds(synthetic, cfg, manifest)
    assert [f.start.year for f in folds] == [2021, 2022]
    for fold in folds:
        for horizon in m.HORIZONS:
            audit = m.audit_split(synthetic, fold, horizon, cfg)
            assert audit["overlap_patients"] > 0
            assert audit["train_rows"] > 0
            assert m.label_available_at(synthetic, horizon, cfg)[m.train_mask(synthetic, horizon, fold.start, cfg)].max() < fold.start


def test_deliberately_leaky_mask_detected(synthetic, cfg):
    fold = m.Fold(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"), pd.Timestamp("2024-01-01"))
    bad = synthetic["invitation_date"] < fold.start
    with pytest.raises(m.ContractError, match="leakage"):
        m.audit_split(synthetic, fold, 365, cfg, bad)


def test_missing_cutoff_and_short_sample_fail(synthetic, cfg):
    with pytest.raises(m.ContractError, match="verified"):
        m.reserve_test(synthetic, replace(cfg, outcome_data_end=None))
    tiny = synthetic[synthetic["invitation_date"].dt.year == 2020]
    cfg = replace(cfg, test_year=2020)
    manifest = m.reserve_test(tiny, cfg)
    with pytest.raises(m.ContractError, match="Insufficient chronological"):
        m.make_temporal_folds(tiny, cfg, manifest)


def test_no_early_availability_for_positive_labels(synthetic, cfg):
    d = synthetic.iloc[:2].copy()
    d["invitation_date"] = pd.Timestamp("2020-01-01")
    d["screening_90d"] = [0, 1]
    assert m.label_available_at(d, 90, cfg).nunique() == 1


def test_common_cohort_stricter(synthetic, cfg):
    origin = pd.Timestamp("2021-01-01")
    shorter = m.train_mask(synthetic, 90, origin, cfg)
    common = m.train_mask(synthetic, 90, origin, replace(cfg, cohort_policy="common"))
    assert common.sum() < shorter.sum()
    assert (~common | shorter).all()


@pytest.mark.parametrize("preset", m.PRESETS)
def test_registry_presets_deduplicate(synthetic, cfg, preset):
    names, _ = m.resolve_feature_set(preset, synthetic, replace(cfg, include_groups="screening"))
    assert len(names) == len(set(names))
    assert not set(m.TARGETS.values()) & set(names)
    assert m.build_features(synthetic, cfg, names).columns.tolist() == names


def test_registry_missing_and_empty(synthetic, cfg):
    with pytest.raises(m.ContractError, match="zero usable"):
        m.resolve_feature_set("registration_continuity", synthetic[list(m.KEYS)], cfg)
    with pytest.raises(m.ContractError, match="Unknown semantic"):
        m.resolve_feature_set("all_safe", synthetic, replace(cfg, include_groups="made_up"))
    with pytest.raises(m.ContractError, match="Unknown feature"):
        m.resolve_feature_set("made_up", synthetic, cfg)
    with pytest.raises(m.ContractError, match="Unregistered"):
        m.build_features(synthetic, cfg, ["future_count"])


def test_geography_calendar_ablations(synthetic, cfg):
    for preset, group in [("all_safe_no_geography", "geography"), ("all_safe_no_calendar", "calendar"),
                          ("all_safe_no_longitudinal", "longitudinal")]:
        names, _ = m.resolve_feature_set(preset, synthetic, cfg)
        registry = m.build_feature_registry()
        assert all(group not in registry[n].semantic_groups for n in names)
