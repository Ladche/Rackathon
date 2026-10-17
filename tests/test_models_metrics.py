from __future__ import annotations

from dataclasses import replace

from hypothesis import given, strategies as st
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.isotonic import IsotonicRegression

import mammography_pipeline as m


@given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=3, max_size=3))
def test_projection_properties(values):
    p = np.array([values])
    projected = m.project_horizon_probabilities(p)
    reference = IsotonicRegression(increasing=False).fit_transform(np.arange(3), values)
    np.testing.assert_allclose(projected[0], reference, atol=1e-12)
    np.testing.assert_allclose(m.project_horizon_probabilities(projected), projected)
    assert (np.diff(projected) <= 1e-12).all()
    assert (projected >= 0).all() and (projected <= 1).all()


def test_projection_not_sorting_or_normalization():
    p = np.array([[.7, .8, .4], [.9, .8, .7]])
    out = m.project_horizon_probabilities(p)
    np.testing.assert_allclose(out, [[.75, .75, .4], [.9, .8, .7]])
    assert not np.allclose(out.sum(axis=1), 1)


@pytest.mark.parametrize("p", [np.array([.1,.2,.3]), np.array([[1.1,0,0]]), np.array([[np.nan,0,0]])])
def test_invalid_projection(p):
    with pytest.raises(m.ContractError):
        m.project_horizon_probabilities(p)


class ReversedClasses:
    classes_ = np.array([1, 0])

    def predict_proba(self, X):
        return np.tile([.8, .2], (len(X), 1))


def test_class_mapping():
    assert m.positive_probability(ReversedClasses(), [0, 1]).tolist() == [.8, .8]
    estimator = m.DummyClassifier(strategy="prior").fit([[0], [1]], [0, 0])
    assert m.positive_probability(estimator, [[0]]).tolist() == [0]
    estimator.fit([[0], [1]], [1, 1])
    assert m.positive_probability(estimator, [[0]]).tolist() == [1]


@pytest.mark.parametrize("family", m.MANDATORY_MODELS)
def test_every_family_three_horizons(synthetic, cfg, family):
    cfg = replace(cfg, preset="smoke")
    names, _ = m.resolve_feature_set("minimal", synthetic, cfg)
    X = m.build_features(synthetic, cfg, names)
    predictions = []
    for h in m.HORIZONS:
        c = m.fit_component(synthetic, X, h, pd.Timestamp("2024-01-01"), family, {}, cfg)
        p, q = c.predict(X.iloc[-3:])
        assert p.shape == (3,) and np.isfinite(q).all()
        assert ((p >= 0) & (p <= 1)).all()
        predictions.append(q)
    final = m.project_horizon_probabilities(np.column_stack(predictions))
    assert final.shape == (3, 3) and (np.diff(final) <= 1e-12).all()


@pytest.mark.parametrize("shape,alpha", [((32,), .001), ((64,32), .01)])
def test_mlp_architectures_and_warnings(synthetic, cfg, shape, alpha):
    names, _ = m.resolve_feature_set("minimal", synthetic, cfg)
    X = m.build_features(synthetic, cfg, names)
    c = m.fit_component(synthetic, X, 90, pd.Timestamp("2024-01-01"), "mlp",
                        {"hidden_layer_sizes": shape, "alpha": alpha, "max_iter": 2}, cfg)
    assert c.pipeline.named_steps["model"].hidden_layer_sizes == shape
    assert not c.pipeline.named_steps["model"].early_stopping
    assert any("ConvergenceWarning" in w for w in c.training_warnings)


def test_train_only_preprocessing_and_unseen_category(cfg):
    train = pd.DataFrame({"number": [0., 0., np.nan], "category": ["A", "A", np.nan],
                          "empty": [np.nan]*3})
    validation = pd.DataFrame({"number": [1000.], "category": ["UNSEEN"], "empty": [4.]})
    pipeline = m.make_pipeline(train, "logistic", {}, cfg).fit(train, [0, 1, 0])
    imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_.tolist() == [0., 0.]
    assert np.isfinite(pipeline.predict_proba(validation)).all()


def test_memory_limit_dense():
    guard = m.MatrixGuard(memory_mb=1, dense=True)
    with pytest.raises(m.ContractError, match="memory ceiling"):
        guard.transform(sparse.csr_matrix((10000, 1000)))


def test_optional_dependency_skip(cfg, monkeypatch):
    monkeypatch.setattr(m, "optional_available", lambda family: False)
    available, skipped = m.model_families(replace(cfg, models="lightgbm,xgboost,catboost"))
    assert available == ["dummy"] and len(skipped) == 3
    with pytest.raises(m.ContractError, match="Install"):
        m.model_families(replace(cfg, models="catboost", strict_optional=True))
    with pytest.raises(ImportError, match="unavailable"):
        m.build_model("xgboost", {}, cfg)


def test_search_quick_two_configs(cfg):
    for family in m.MANDATORY_MODELS:
        space = m.build_search_space(family, cfg)
        assert len(space) >= (1 if family == "dummy" else 2)
        for params in space:
            m.build_model(family, params, cfg)


def test_metrics_semantics():
    y = np.array([1, 1, 0, 0])
    p = np.array([.9, .8, .2, .1])
    result = m.evaluate_probabilities(y, p, .5)
    assert result["roc_auc"] == 1
    assert result["average_precision_attendance"] == 1
    assert result["brier"] == pytest.approx(.025)
    assert result["prevalence_nonattendance"] == .5
    assert result["log_loss_improvement"] > 0
    assert m.evaluate_probabilities(np.array([1,1]), np.array([.5,.9]))["roc_auc"] is None
    assert m.evaluate_probabilities(np.array([]), np.array([]))["reason"] == "empty eligible cohort"
    with pytest.raises(m.ContractError):
        m.evaluate_probabilities(np.array([0]), np.array([np.nan]))


def test_interval_model_conversion(synthetic):
    labels = m.interval_labels(synthetic)
    assert set(labels) == {0, 1, 2, 3}
    out = m.interval_probabilities(np.array([[.1, .2, .3, .4]]), [0, 1, 2, 3])
    np.testing.assert_allclose(out, [[.9, .7, .4]])
    with pytest.raises(m.ContractError):
        m.interval_probabilities(np.array([[.1, .2]]), [0, 1])


def test_patient_bootstrap_preserves_groups(synthetic, cfg):
    people = ["a", "a", "b", "b", "b"]
    for seed in range(10):
        idx = m.bootstrap_indices(people, np.random.default_rng(seed))
        assert sum(idx == 0) == sum(idx == 1)
        assert sum(idx == 2) == sum(idx == 3) == sum(idx == 4)
    p = np.tile([.8, .7, .6], (len(synthetic), 1))
    result = m.bootstrap_by_patient(synthetic, p, 10, cfg.seed, p)
    assert result["paired_log_loss_difference_95ci"] == [0, 0]
    assert m.bootstrap_by_patient(synthetic.iloc[:1], p[:1], 10, cfg.seed)["valid_repeats"] == 0


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_explicit_and_monotone(method):
    p = np.tile([.2, .3, .7, .8], 100)
    y = np.tile([0, 1, 0, 1], 100)
    c = m.fit_calibrator(p, y, method)
    out = c.predict(np.linspace(0, 1, 20))
    assert np.isfinite(out).all() and (np.diff(out) >= -1e-9).all()
    with pytest.raises(m.ContractError, match="requires"):
        m.fit_calibrator(p[:10], y[:10], method)


def test_calibration_partition(synthetic, cfg):
    cfg = replace(cfg, calibration="sigmoid", calibration_days=365)
    names, _ = m.resolve_feature_set("minimal", synthetic, cfg)
    X = m.build_features(synthetic, cfg, names)
    origin = pd.Timestamp("2024-01-01")
    c = m.fit_component(synthetic, X, 90, origin, "logistic", {}, cfg)
    assert c.calibrator is not None
    start, end = map(pd.Timestamp, c.calibration_bounds)
    assert pd.Timestamp(c.fit_origin) == start
    assert end + pd.Timedelta(days=365) < origin
    assert (m.label_available_at(synthetic, 90, cfg)[m.train_mask(synthetic, 90, start, cfg)] < start).all()
