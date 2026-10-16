from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import mammography_pipeline as m


def test_import_no_data_io(tmp_path):
    script = """
import builtins
old = builtins.open
def guarded(file, *a, **kw):
    if str(file).endswith('.csv'):
        raise AssertionError('CSV read on import')
    return old(file, *a, **kw)
builtins.open = guarded
import mammography_pipeline
assert not hasattr(mammography_pipeline, 'args')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_help_and_invalid_args():
    for arguments in (["--help"], ["predict", "--help"], ["self-test"]):
        completed = subprocess.run([sys.executable, "mammography_pipeline.py", *arguments], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
    result = subprocess.run([sys.executable, "mammography_pipeline.py", "not-a-command"], capture_output=True, text=True)
    assert result.returncode == 2


def test_full_cli_workflow_fresh_process(tmp_path, synthetic):
    data = tmp_path / "SYNTHETIC.csv"
    synthetic.to_csv(data, index=False)
    out = tmp_path / "run"

    def run(*arguments, success=True):
        completed = subprocess.run([sys.executable, "mammography_pipeline.py", *arguments],
                                   capture_output=True, text=True, timeout=90)
        assert completed.returncode == (0 if success else 2), completed.stdout + completed.stderr
        return completed

    shared = ["--data", str(data), "--date-format", "%Y-%m-%d", "--output-dir", str(out),
              "--outcome-data-end", "2026-01-02", "--test-year", "2024", "--n-jobs", "1",
              "--provenance", "SYNTHETIC integration mechanics only"]
    run("inspect", *shared)
    run("features", *shared, "--feature-sets", "all_safe")
    run("eda", *shared)
    run("benchmark", *shared, "--models", "logistic,decision_tree", "--feature-sets", "minimal,raw_snapshot",
        "--assume-t0-snapshots", "--budget-seconds", "30")
    assert not (out / "final_test_metrics.json").exists()
    selection = json.loads((out / "selection.json").read_text())
    assert len(selection["frozen_candidates"]) == 10
    bundle_path = out / "model_bundle.joblib"
    run("train", *shared, "--selection-path", str(out / "selection.json"), "--model-path", str(bundle_path))
    bundle = m.load_bundle(str(bundle_path))
    local = m.prediction_matrices(bundle, synthetic)[2]
    # A fresh interpreter must reconstruct canonical custom classes, not __main__ objects.
    script = (
        "import mammography_pipeline as m; import numpy as np; "
        f"b=m.load_bundle({str(bundle_path)!r}); c=m.Config(**b['config']); "
        f"d=m.load_csv(c,{str(data)!r}); np.save({str(tmp_path / 'fresh.npy')!r}, m.prediction_matrices(b,d)[2])"
    )
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    np.testing.assert_allclose(local, np.load(tmp_path / "fresh.npy"), atol=1e-12)
    eval_out = tmp_path / "evaluation"
    run("evaluate", "--data", str(data), "--model-path", str(bundle_path),
        "--output-dir", str(eval_out), "--bootstrap", "5")
    metrics = json.loads((eval_out / "final_test_metrics.json").read_text())
    assert metrics["final"]["violation_frequency"] == 0
    new = synthetic.iloc[[-1, -5, -2]].drop(columns=list(m.TARGETS.values()))
    new_path = tmp_path / "new.csv"
    new.to_csv(new_path, index=False)
    predictions = tmp_path / "predictions.csv"
    run("predict", "--data", str(new_path), "--model-path", str(bundle_path), "--partial-history",
        "--data-as-of", "2025-01-01", "--output", str(predictions), "--output-dir", str(tmp_path / "pred"))
    scored = pd.read_csv(predictions)
    assert scored["invitation_id"].tolist() == new["invitation_id"].tolist()
    assert scored.columns[3:6].tolist() == list(m.OUTPUTS)
    assert set(m.OUTPUTS).issubset(scored)
    assert (np.diff(scored[list(m.OUTPUTS)].values) <= 1e-12).all()
    run("explain", "--data", str(data), "--model-path", str(out / "validation_bundle.joblib"),
        "--output-dir", str(tmp_path / "explain"), "--explanation-rows", "12", "--permutation-repeats", "2", "--shap")
    explanation = json.loads((tmp_path / "explain" / "explanations.json").read_text())
    assert explanation["permutation"]
    assert explanation["shap"]["status"] in {"skipped", "completed with bounded rows; no hard library timeout"}


def test_selection_does_not_use_test_labels(synthetic, cfg):
    cfg = replace(cfg, models="logistic", feature_sets="minimal,raw_snapshot", folds=1, max_trials=6)
    first = m.compare_models(synthetic, cfg)
    changed = synthetic.copy()
    test = changed["invitation_date"].dt.year == 2024
    changed.loc[test, list(m.TARGETS.values())] = 1.
    second_cfg = replace(cfg, output_dir=cfg.output_dir + "_changed")
    second = m.compare_models(changed, second_cfg)
    assert first["choices"] == second["choices"]
    assert first["validation_final_macro_log_loss"] == second["validation_final_macro_log_loss"]
    first_bundle = m.train_selected(synthetic, cfg, first, Path(cfg.output_dir) / "one.joblib")
    second_bundle = m.train_selected(changed, second_cfg, second, Path(second_cfg.output_dir) / "two.joblib")
    np.testing.assert_allclose(m.prediction_matrices(first_bundle, synthetic)[2],
                               m.prediction_matrices(second_bundle, synthetic)[2], atol=1e-12)


def test_resume_and_incompatible_resume(synthetic, cfg, monkeypatch):
    cfg = replace(cfg, models="logistic", folds=1, max_trials=6)
    first = m.compare_models(synthetic, cfg)

    def forbidden(*a, **kw):
        raise AssertionError("Completed fit was repeated during compatible resume")

    monkeypatch.setattr(m, "run_trial", forbidden)
    second = m.compare_models(synthetic, cfg, resume=True)
    assert first["choices"] == second["choices"]
    with pytest.raises(m.ContractError, match="Incompatible resume"):
        m.compare_models(synthetic, replace(cfg, seed=99), resume=True)
    with pytest.raises(m.ContractError, match="Existing experiment"):
        m.compare_models(synthetic, cfg)


def test_failed_timeout_and_interrupted_trials_persist(synthetic, cfg, monkeypatch):
    cfg = replace(cfg, models="logistic", folds=1, max_trials=2, feature_sets="minimal")

    def timeout(*args, **kwargs):
        raise TimeoutError("Synthetic controlled fit timeout")

    monkeypatch.setattr(m, "run_trial", timeout)
    with pytest.raises(m.ContractError, match="No successful"):
        m.compare_models(synthetic, cfg)
    ledger = [json.loads(x) for x in (Path(cfg.output_dir) / "trials.jsonl").read_text().splitlines()]
    assert any(r["status"] == "failed" and "TimeoutError" in r["exception"] for r in ledger)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(m, "run_trial", interrupted)
    next_cfg = replace(cfg, output_dir=cfg.output_dir + "_interrupted")
    with pytest.raises(KeyboardInterrupt):
        m.compare_models(synthetic, next_cfg)
    assert '"status": "interrupted"' in (Path(next_cfg.output_dir) / "trials.jsonl").read_text()


def test_global_deadline_does_not_launch_work(synthetic, cfg):
    cfg = replace(cfg, budget_seconds=1e-12)
    with pytest.raises(m.ContractError, match="No successful"):
        m.compare_models(synthetic, cfg)
    skipped = json.loads((Path(cfg.output_dir) / "skipped_models_trials.json").read_text())
    assert any(r["reason"] == "Global search deadline or trial limit" for r in skipped)


def test_history_parity_and_holdout_overlap(synthetic, cfg, tmp_path):
    cfg = replace(cfg, feature_sets="longitudinal_no_labels", models="dummy", folds=1)
    selection = m.compare_models(synthetic, cfg)
    bundle = m.train_selected(synthetic, cfg, selection, tmp_path / "bundle.joblib")
    invitations = synthetic.iloc[[-1, -2]].drop(columns=list(m.TARGETS.values()))
    with pytest.raises(m.ContractError, match="requires external"):
        m.predict_bundle(bundle, invitations, data_as_of="2025-01-01")
    with pytest.raises(m.ContractError, match="data-as-of"):
        m.predict_bundle(bundle, invitations, partial_history=True)
    with pytest.raises(m.ContractError, match="after"):
        m.predict_bundle(bundle, invitations, data_as_of="2010-01-01", partial_history=True)
    history = synthetic.iloc[:-2]
    result = m.predict_bundle(bundle, invitations, history, "2025-01-01")
    assert result["observed_prior_input_rows"].tolist() == [9, 9]
    assert result["invitation_id"].tolist() == invitations["invitation_id"].tolist()
    with pytest.raises(m.ContractError, match="duplicate"):
        m.predict_bundle(bundle, invitations, synthetic, "2025-01-01")
    train = synthetic.iloc[:36]
    with pytest.raises(m.ContractError, match="overlaps"):
        m.reject_fit_overlap(train, bundle["components"])
    with pytest.raises(m.ContractError, match="held-out development"):
        m.explain_model(bundle, synthetic, str(tmp_path / "explain"))
    changed = synthetic.copy()
    changed.loc[0, "gp_registered"] = 1 - changed.loc[0, "gp_registered"]
    with pytest.raises(m.ContractError, match="Development predictors"):
        m.train_selected(changed, cfg, selection, tmp_path / "bad.joblib")
    broken = dict(bundle, version="future")
    m.save_bundle(tmp_path / "wrong.joblib", broken)
    with pytest.raises(m.ContractError, match="Incompatible"):
        m.load_bundle(str(tmp_path / "wrong.joblib"))


def test_empty_segments_and_undefined_reports(synthetic, cfg, tmp_path):
    d = synthetic.iloc[:1]
    result = m.summarize_segments(d, np.array([[.8, .7, .6]]), cfg)
    assert result["segments"] == []
    metric = m.vector_metrics(d.iloc[:0], np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), [.5]*3)
    m.atomic_json(tmp_path / "empty.json", metric)
    m.plots_for_metrics(metric, tmp_path, "empty")
    assert (tmp_path / "empty_reliability.png").exists()


def test_patient_disjoint_fit(synthetic, cfg):
    people = set(synthetic["person_id"].unique()[:5])
    names, _ = m.resolve_feature_set("minimal", synthetic, cfg)
    X = m.build_features(synthetic, cfg, names)
    c = m.fit_component(synthetic, X, 365, pd.Timestamp("2024-01-01"), "logistic", {}, cfg, people)
    forbidden = m.key_hashes(pd.DataFrame({"person_id": list(people)}), "person_id")
    assert not c.fit_people_hashes & forbidden
