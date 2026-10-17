from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

import mammography_pipeline as m


def test_native_category_vocabulary_and_unknowns():
    train = pd.DataFrame({"number": [1., 3., np.nan], "cat": ["A", "B", None]})
    new = pd.DataFrame({"number": [np.nan, 10.], "cat": ["NEW", None]})
    for family in ("lightgbm", "xgboost", "catboost"):
        transformer = m.NativeCategories(family).fit(train)
        result = transformer.transform(new)
        assert result["number"].tolist() == [2., 10.]
        assert result["cat"].tolist() == ["__UNKNOWN__", "__MISSING__"]
        assert not isinstance(result["cat"].dtype, pd.CategoricalDtype) if family == "catboost" else isinstance(result["cat"].dtype, pd.CategoricalDtype)
        with pytest.raises(m.ContractError, match="schema/order"):
            transformer.transform(new[["cat", "number"]])
    with pytest.raises(m.ContractError, match="reserved"):
        m.NativeCategories("catboost").fit(pd.DataFrame({"c": ["__MISSING__"]}))
    transformer = m.NativeCategories("catboost", memory_mb=0).fit(train)
    with pytest.raises(m.ContractError, match="memory ceiling"):
        transformer.transform(new)


def test_interval_benchmark_is_runnable(synthetic, cfg):
    cfg = replace(cfg, preset="quick")
    manifest = m.reserve_test(synthetic, cfg)
    folds = m.make_temporal_folds(synthetic, cfg, manifest)
    names, _ = m.resolve_feature_set("minimal", synthetic, cfg)
    X = m.build_features(synthetic, cfg, names)
    records = m.run_interval_benchmark(synthetic, X, folds, cfg, time.monotonic()+20)
    successful = [r for r in records if r["status"] == "success"]
    assert len(successful) == 2
    assert all(r["horizon_metrics"]["90"]["n"] == 72 for r in successful)
    skipped = m.run_interval_benchmark(synthetic, X, folds, cfg, time.monotonic()-1)
    assert skipped[0]["status"] == "not launched"


def test_history_all_missing_outcomes(synthetic, cfg):
    cfg = replace(cfg, historical_outcomes="on")
    small = synthetic.iloc[:2].copy()
    small[list(m.TARGETS.values())] = np.nan
    result = m.build_patient_history(small, cfg)
    for h in m.HORIZONS:
        assert result[f"historical_{h}d_mature_count"].tolist() == [0, 0]
        assert result[f"historical_{h}d_nonattendance_fraction"].isna().all()


def test_historical_exact_boundary_and_observation_end(cfg):
    cfg = replace(cfg, historical_outcomes="on", outcome_data_end="2020-12-01")
    small = pd.DataFrame({
        "invitation_id": ["a","b","c"], "person_id": ["p"]*3,
        "invitation_date": pd.to_datetime(["2020-01-01","2020-03-31","2020-04-01"]),
        "screening_90d": [1.,np.nan,np.nan],
    })
    result = m.build_patient_history(small, cfg)
    assert result["historical_90d_mature_count"].tolist() == [0, 0, 1]
    cfg = replace(cfg, outcome_data_end="2020-03-01")
    result = m.build_patient_history(small, cfg)
    assert result["historical_90d_mature_count"].tolist() == [0, 0, 0]


def test_reserve_default_and_bad_windows(synthetic, cfg):
    manifest = m.reserve_test(synthetic, replace(cfg, test_year=None))
    assert manifest["test_start"].startswith("2024")
    with pytest.raises(m.ContractError, match="No complete"):
        m.reserve_test(synthetic, replace(cfg, test_year=None, outcome_data_end="2015-02-01"))
    with pytest.raises(m.ContractError, match="full conservative"):
        m.reserve_test(synthetic, replace(cfg, outcome_data_end="2024-10-01"))
    with pytest.raises(m.ContractError, match="no invitations"):
        m.reserve_test(synthetic, replace(cfg, test_year=2010))
    invalid = replace(cfg, validation_start="2023-01-01", validation_end="2023-12-31")
    with pytest.raises(m.ContractError, match="selection deadline"):
        m.make_temporal_folds(synthetic, invalid, m.reserve_test(synthetic, cfg))
    explicit = replace(cfg, validation_start="2021-01-01", validation_end="2021-06-30")
    assert len(m.make_temporal_folds(synthetic, explicit, manifest)) == 1


@pytest.mark.parametrize("date", ["bad", "2020-01-01T12:00:00", "2020-01-01T00:00:00Z", "NaT"])
def test_bad_iso_date(date):
    with pytest.raises(m.ContractError):
        m.iso_date(date)


def test_single_class_and_empty_fit_fail(synthetic, cfg):
    names, _ = m.resolve_feature_set("minimal", synthetic, cfg)
    X = m.build_features(synthetic, cfg, names)
    small = synthetic.copy()
    small[list(m.TARGETS.values())] = 1.
    with pytest.raises(m.ContractError, match="Single-class"):
        m.fit_component(small, X, 90, pd.Timestamp("2024-01-01"), "logistic", {}, cfg)
    with pytest.raises(m.ContractError, match="No mature training"):
        m.fit_component(small, X, 90, pd.Timestamp("2010-01-01"), "dummy", {}, cfg)
    with pytest.raises(m.ContractError, match="assume-t0"):
        m.fit_component(small, X, 90, pd.Timestamp("2024-01-01"), "dummy", {}, replace(cfg, assume_t0_snapshots=False))
    with pytest.raises(m.ContractError, match="no mature outcomes"):
        m.fit_component(small, X, 90, pd.Timestamp("2030-01-01"), "dummy", {}, replace(cfg, calibration="sigmoid"))


def test_missing_sources_and_manual_calendar_variants(synthetic, cfg):
    small = synthetic[list(m.KEYS)].copy()
    names, reasons = m.resolve_feature_set("demographics_calendar", small, replace(cfg, include_groups="calendar"))
    assert "invitation_month" in names
    assert any(r["reason"].startswith("Missing sources") for r in reasons)
    with pytest.raises(m.ContractError, match="schema mismatch"):
        m.build_features(small, cfg, ["gp_registered"])
    with pytest.raises(m.ContractError, match="memory ceiling"):
        m.build_features(synthetic, replace(cfg, memory_mb=0), ["territory"])


def test_atomic_json_invalid_is_explicit(tmp_path):
    with pytest.raises(ValueError):
        m.atomic_json(tmp_path / "bad.json", {"value": np.nan})
    assert not (tmp_path / "bad.json").exists()
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(TypeError):
        m.json_default(object())


def test_cli_config_errors_and_list_without_data(tmp_path, capsys):
    assert m.main(["inspect", "--list-features", "--data", "/does/not/exist"]) == 0
    assert "availability_rule" in capsys.readouterr().out
    assert m.main(["inspect", "--targets", "365"]) == 2
    assert m.main(["inspect", "--data", "/does/not/exist"]) == 2
    with pytest.raises(m.ContractError, match="Unsupported target"):
        m.label_available_at(pd.DataFrame(), 100, m.Config())
    assert m.observed_mask(pd.DataFrame(index=[0]), 90, m.Config()).tolist() == [False]
    with pytest.raises(m.ContractError, match="Unknown requested"):
        m.model_families(m.Config(models="invented"))


def test_schema_empty_header_and_duplicate_csv(tmp_path, cfg):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(m.ContractError, match="empty"):
        m.load_csv(replace(cfg, data=str(p)))
    p.write_text("PERSON_ID,PERSON_ID\nx,x\n")
    with pytest.raises(m.ContractError, match="collision"):
        m.load_csv(replace(cfg, data=str(p)))


def test_reporting_outputs_are_structural_only(synthetic, cfg, tmp_path):
    cfg = replace(cfg, output_dir=str(tmp_path))
    m.write_schema_outputs(synthetic, cfg)
    summary = json.loads((tmp_path / "inspection.json").read_text())
    assert summary["patients"] == 36
    assert "prevalence" not in json.dumps(summary)
    assert "SYNTH-P" not in (tmp_path / "schema_report.md").read_text()


@pytest.mark.parametrize("mutation,reason", [
    ({"horizons": (365, 180, 90)}, "horizon order"),
    ({"environment": {**m.environment_record(), "numpy": "incompatible"}}, "Incompatible numpy"),
    ({"environment": {**m.environment_record(), "code_sha256": "changed"}}, "Implementation hash"),
])
def test_persistence_version_contract(tmp_path, mutation, reason):
    bundle = {"version": m.VERSION, "horizons": m.HORIZONS, "environment": m.environment_record(), **mutation}
    path = tmp_path / "bundle.joblib"
    m.save_bundle(path, bundle)
    with pytest.raises(m.ContractError, match=reason):
        m.load_bundle(str(path))


def test_atomic_model_failure_preserves_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "existing.joblib"
    path.write_bytes(b"existing artifact")

    def fail(*a, **kw):
        raise OSError("Controlled serialization failure")

    monkeypatch.setattr(m.joblib, "dump", fail)
    with pytest.raises(OSError, match="serialization"):
        m.save_bundle(path, {})
    assert path.read_bytes() == b"existing artifact"
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_estimator_classes_are_not_assumed_positive():
    class InvalidClasses:
        classes_ = np.array([2])

        def predict_proba(self, X):
            return np.ones((len(X), 1))

    with pytest.raises(m.ContractError, match="Unsupported binary class"):
        m.positive_probability(InvalidClasses(), [1])
