import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.model_selection import train_test_split

import explainability as xa


@pytest.fixture
def screening_data():
    rng = np.random.default_rng(87)
    size = 240
    signal = rng.normal(size=size)
    age = rng.integers(0, 3, size)
    region = rng.integers(0, 3, size)
    frame = pd.DataFrame({
        "SIGNAL": signal,
        "CORRELATED_SIGNAL": signal + rng.normal(0, 0.03, size),
        "CONTACT_COUNT": rng.integers(0, 20, size),
        "NOISE": rng.normal(size=size),
        "AGE_BAND_B": (age == 1).astype(int),
        "AGE_BAND_C": (age == 2).astype(int),
        "VZP_TERRITORIAL_AREA_B": (region == 1).astype(int),
        "VZP_TERRITORIAL_AREA_C": (region == 2).astype(int),
    })
    labels = pd.Series(
        (rng.random(size) < expit(3 * signal + 0.05 * frame.CONTACT_COUNT)).astype(int),
        index=frame.index,
    )
    return train_test_split(frame, labels, test_size=0.3, stratify=labels, random_state=42)


@pytest.fixture(autouse=True)
def close_plots(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    yield
    plt.close("all")


@pytest.mark.parametrize("kind", ["xgboost", "catboost"])
def test_entire_notebook_section(screening_data, kind):
    training, evaluation, train_target, test_target = screening_data
    if kind == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=20, max_depth=3, n_jobs=1, random_state=42)
    else:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=20, depth=3, thread_count=1, verbose=False,
            random_seed=42, allow_writing_files=False,
        )
    model.fit(training, train_target)
    before_training, before_test = training.copy(deep=True), evaluation.copy(deep=True)
    probabilities_before = model.predict_proba(evaluation).copy()
    notebook = json.loads((Path(__file__).parent / "program.ipynb").read_text())
    context = {
        "pd": pd, "np": np, "plt": plt,
        "X_train": training, "X_test": evaluation,
        "y_test": test_target, "final_best_model": model, "winner_name": kind,
        "X_num": training[["SIGNAL", "CONTACT_COUNT", "NOISE"]],
    }
    executed = []
    for cell in notebook["cells"]:
        if cell.get("id", "").startswith("explainability-") and cell["cell_type"] == "code":
            code = "".join(cell["source"])
            exec(compile(code, f"notebook:{cell['id']}", "exec"), context)
            executed.append(cell["id"])
    assert len(executed) == 6
    pd.testing.assert_frame_equal(training, before_training)
    pd.testing.assert_frame_equal(evaluation, before_test)
    np.testing.assert_array_equal(model.predict_proba(evaluation), probabilities_before)
    risk = context["xa_risk"]
    np.testing.assert_allclose(risk, model.predict_proba(evaluation)[:, 0])
    shap_values = context["xa_shap_values"]
    np.testing.assert_allclose(
        expit(shap_values.base_values + shap_values.values.sum(axis=1)), risk, atol=2e-5
    )
    assert context["xa_consensus"].shape[0] == training.shape[1]
    assert np.isfinite(context["xa_consensus"].to_numpy()).all()
    assert any("SIGNAL" in name for name in context["xa_grouped"].head(2).index)
    assert context["xa_interactions"].shape[0] == 3
    assert context["xa_local_stability"].shape == (training.shape[1], 3)
    assert context["xa_shap_table"].top10_frequency.between(0, 1).all()
    assert (context["xa_scenarios"].risk_difference < 0).all()
    for result in context["xa_effects"].values():
        assert abs(np.average(result["ale"].ale, weights=result["ale"]["count"])) < 1e-10


class AdditiveModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        risk = 0.3 + 0.2 * frame["x"].to_numpy()
        return np.column_stack([risk, 1 - risk])


def test_ale_matches_linear_probability_effect():
    frame = pd.DataFrame({"x": np.linspace(0, 1, 201), "irrelevant": np.ones(201)})
    result = xa.ale_1d(AdditiveModel(), frame, "x")
    expected = 0.2 * result.value
    expected -= np.average(expected, weights=result["count"])
    np.testing.assert_allclose(result.ale, expected, atol=1e-12)


def test_permutation_signal_and_constant():
    frame = pd.DataFrame({"x": np.linspace(0, 1, 100), "irrelevant": np.ones(100)})
    target = (frame.x > 0.5).astype(int)
    first = xa.permutation_table(AdditiveModel(), frame, target, repeats=5)
    second = xa.permutation_table(AdditiveModel(), frame, target, repeats=5)
    pd.testing.assert_frame_equal(first, second)
    assert first.loc["x", "auc_drop"] > 0.3
    assert first.loc["x", "logloss_increase"] > 0
    assert first.loc["irrelevant", "auc_drop"] == 0
    assert first.loc["irrelevant", "logloss_increase"] == 0


def test_class_order_and_invalid_classes():
    class ReversedModel:
        classes_ = np.array([1, 0])

        def predict_proba(self, frame):
            return np.tile([0.8, 0.2], (len(frame), 1))

    frame = pd.DataFrame({"x": [0, 1]})
    np.testing.assert_array_equal(xa.nonattendance_probability(ReversedModel(), frame), [0.2, 0.2])
    model = ReversedModel()
    model.classes_ = np.array([1, 2])
    with pytest.raises(ValueError, match="binary classes"):
        xa.nonattendance_probability(model, frame)


def test_groups_keep_dummy_blocks_and_correlated_features(screening_data):
    training = screening_data[0]
    groups = xa.feature_groups(training)
    assert set(groups["AGE_BAND"]) == {"AGE_BAND_B", "AGE_BAND_C"}
    assert set(groups["VZP_TERRITORIAL_AREA"]) == {
        "VZP_TERRITORIAL_AREA_B", "VZP_TERRITORIAL_AREA_C",
    }
    assert any(set(cols) == {"SIGNAL", "CORRELATED_SIGNAL"} for cols in groups.values())
    flattened = [col for columns in groups.values() for col in columns]
    assert len(flattened) == len(set(flattened)) == training.shape[1]


def test_prepare_data_stratifies_and_preserves_alignment(screening_data):
    training, evaluation, _, target = screening_data
    sample, outcome = xa.prepare_data(AdditiveModelForSetup(), training, evaluation, target, max_rows=20)
    assert len(sample) == 20
    assert set(outcome) == {0, 1}
    pd.testing.assert_series_equal(outcome, (target.loc[sample.index] == 0).astype(int))
    with pytest.raises(ValueError, match="aligned"):
        xa.prepare_data(AdditiveModelForSetup(), training, evaluation, target.iloc[::-1])
    invalid = evaluation.copy()
    invalid.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        xa.prepare_data(AdditiveModelForSetup(), training, invalid, target)


class AdditiveModelForSetup:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        return np.tile([0.3, 0.7], (len(frame), 1))


def test_no_lower_risk_profile_is_explicitly_empty():
    frame = pd.DataFrame({"x": [0.0, 0.5, 1.0]})
    result = xa.observed_scenarios(AdditiveModel(), frame, frame.iloc[[0]])
    assert result.empty
    assert "risk_difference" in result


def test_constant_ale_fails_explicitly():
    with pytest.raises(ValueError, match="nonconstant"):
        xa.ale_1d(AdditiveModel(), pd.DataFrame({"x": [1, 1, 1]}), "x")
