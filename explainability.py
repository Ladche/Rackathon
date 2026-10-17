"""Diagnostic explanations of a frozen binary screening model (class 0 = nonattendance)."""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree


def nonattendance_probability(model, frame):
    classes = np.asarray(model.classes_)
    if len(classes) != 2 or set(classes.tolist()) != {0, 1}:
        raise ValueError("Expected binary classes 0 (nonattendance), 1 (attendance).")
    return np.asarray(model.predict_proba(frame), dtype=float)[:, np.flatnonzero(classes == 0)[0]]


def prepare_data(model, X_train, X_test, y_test, max_rows=400, seed=42):
    if not X_train.columns.equals(X_test.columns):
        raise ValueError("Training and test features must have identical names and order.")
    if not X_train.columns.is_unique or not X_train.index.is_unique or not X_test.index.is_unique:
        raise ValueError("Feature names and training/test indices must be unique.")
    if not y_test.index.equals(X_test.index):
        raise ValueError("y_test must be aligned with X_test before sampling.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train.dtypes):
        raise ValueError("Explain the numeric, encoded inputs actually used by the model.")
    if not np.isfinite(X_train.to_numpy()).all() or not np.isfinite(X_test.to_numpy()).all():
        raise ValueError("Inputs must be finite; apply the training imputation first.")
    if set(y_test.unique()) != {0, 1}:
        raise ValueError("Both classes are required for ROC-AUC.")
    if not 4 <= max_rows:
        raise ValueError("max_rows must be at least 4.")
    if len(X_test) > max_rows:
        frame, _ = train_test_split(
            X_test, train_size=max_rows, stratify=y_test, random_state=seed
        )
    else:
        frame = X_test.copy()
    target = (y_test.loc[frame.index] == 0).astype(int)
    nonattendance_probability(model, frame)
    return frame.copy(), target


def feature_groups(frame, prefixes=("AGE_BAND_", "VZP_TERRITORIAL_AREA_"), correlation=0.8):
    """Keep dummy blocks intact and join connected, correlated numeric features."""
    groups = {}
    remaining = list(frame.columns)
    for prefix in prefixes:
        columns = [col for col in remaining if col.startswith(prefix)]
        if columns:
            groups[prefix.rstrip("_")] = columns
            remaining = [col for col in remaining if col not in columns]
    correlations = frame[remaining].corr(method="spearman").abs()
    while remaining:
        component = [remaining.pop(0)]
        frontier = component.copy()
        while frontier:
            current = frontier.pop()
            neighbours = [
                col for col in remaining if correlations.loc[current, col] >= correlation
            ]
            component.extend(neighbours)
            frontier.extend(neighbours)
            remaining = [col for col in remaining if col not in neighbours]
        groups[" + ".join(component)] = component
    return groups


def permutation_table(model, frame, target, groups=None, repeats=8, seed=42):
    """Positive values mean worse held-out performance after permutation."""
    if repeats < 2:
        raise ValueError("Use at least two permutations to measure variability.")
    if groups is None:
        groups = {col: [col] for col in frame.columns}
    rng = np.random.default_rng(seed)
    baseline = nonattendance_probability(model, frame)
    auc = roc_auc_score(target, baseline)
    loss = log_loss(target, baseline, labels=[0, 1])
    records = []
    for name, columns in groups.items():
        auc_drops, loss_increases = [], []
        for _ in range(repeats):
            changed = frame.copy()
            changed.loc[:, columns] = frame[columns].iloc[
                rng.permutation(len(frame))
            ].to_numpy()
            probability = nonattendance_probability(model, changed)
            auc_drops.append(auc - roc_auc_score(target, probability))
            loss_increases.append(log_loss(target, probability, labels=[0, 1]) - loss)
        records.append({
            "feature": name,
            "auc_drop": np.mean(auc_drops),
            "auc_std": np.std(auc_drops, ddof=1),
            "auc_q025": np.quantile(auc_drops, 0.025),
            "auc_q975": np.quantile(auc_drops, 0.975),
            "logloss_increase": np.mean(loss_increases),
        })
    return pd.DataFrame(records).set_index("feature").sort_values("auc_drop", ascending=False)


def native_importance(model, columns):
    importance = np.asarray(model.feature_importances_)
    if importance.shape != (len(columns),):
        raise ValueError("Unexpected native feature importance shape.")
    return pd.Series(importance, index=columns, name="native").sort_values(ascending=False)


def shap_analysis(model, frame, bootstrap=100, seed=42):
    """Orient raw tree SHAP towards class 0 and verify the log-odds link."""
    import shap

    if bootstrap < 2:
        raise ValueError("Use at least two bootstrap resamples.")
    explanation = shap.TreeExplainer(model, model_output="raw")(frame)
    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    class_zero = int(np.flatnonzero(np.asarray(model.classes_) == 0)[0])
    if values.ndim == 3:
        values = values[:, :, class_zero]
        base = base[:, class_zero]
    elif values.ndim == 2:
        sign = 1 if class_zero == 1 else -1
        values = sign * values
        base = sign * base
    else:
        raise ValueError(f"Unsupported SHAP shape: {values.shape}")
    base = np.broadcast_to(base, (len(frame),))
    probability = nonattendance_probability(model, frame)
    reconstructed = expit(base + values.sum(axis=1))
    if not np.allclose(reconstructed, probability, atol=2e-5, rtol=2e-5):
        raise ValueError("Raw SHAP is not binary log-odds for this model; do not label it as risk.")
    risk_shap = shap.Explanation(
        values=values, base_values=base, data=frame.to_numpy(),
        feature_names=frame.columns.tolist(),
    )
    absolute = np.abs(values)
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        absolute[rng.integers(0, len(frame), len(frame))].mean(axis=0)
        for _ in range(bootstrap)
    ])
    top_k = min(10, frame.shape[1])
    frequencies = np.zeros(frame.shape[1])
    for sample in boot_means:
        frequencies[np.argsort(-sample)[:top_k]] += 1
    table = pd.DataFrame({
        "mean_abs_shap": absolute.mean(axis=0),
        "bootstrap_q025": np.quantile(boot_means, 0.025, axis=0),
        "bootstrap_q975": np.quantile(boot_means, 0.975, axis=0),
        "top10_frequency": frequencies / bootstrap,
    }, index=frame.columns).sort_values("mean_abs_shap", ascending=False)
    shap.plots.bar(risk_shap, max_display=15, show=False)
    plt.title("SHAP: global importance for nonattendance (log-odds)")
    plt.show()
    shap.plots.beeswarm(risk_shap, max_display=15, show=False)
    plt.show()
    for col in table.index[:2]:
        shap.plots.scatter(risk_shap[:, col], show=False)
        plt.show()
    positions = [int(np.argmax(probability)), int(np.argmin(probability))]
    local = []
    for position in dict.fromkeys(positions):
        shap.plots.waterfall(risk_shap[position], max_display=15, show=False)
        plt.show()
        local.append(pd.DataFrame({
            "row": frame.index[position],
            "feature": frame.columns,
            "value": frame.iloc[position].to_numpy(),
            "risk_shap_logodds": values[position],
            "p_nonattendance": probability[position],
        }).assign(abs_impact=lambda data: data.risk_shap_logodds.abs())
                     .sort_values("abs_impact", ascending=False))
    return table, risk_shap, pd.concat(local, ignore_index=True)


def ale_1d(model, frame, feature, bins=10):
    """First-order ALE with quantile bins, observation-weighted centering."""
    edges = np.unique(np.quantile(frame[feature], np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        raise ValueError(f"ALE needs a nonconstant feature: {feature}")
    membership = np.clip(
        np.searchsorted(edges, frame[feature], side="right") - 1, 0, len(edges) - 2
    )
    increments, counts = [], []
    for idx in range(len(edges) - 1):
        rows = frame.loc[membership == idx].copy()
        counts.append(len(rows))
        if rows.empty:
            increments.append(0.0)
            continue
        low, high = rows.copy(), rows.copy()
        low[feature], high[feature] = edges[idx], edges[idx + 1]
        increments.append(np.mean(
            nonattendance_probability(model, high) - nonattendance_probability(model, low)
        ))
    # Bin midpoint approximation of the accumulated local effect.
    effects = np.cumsum(increments) - np.asarray(increments) / 2
    effects -= np.average(effects, weights=counts)
    return pd.DataFrame({
        "value": (edges[:-1] + edges[1:]) / 2,
        "ale": effects, "count": counts,
    })


def effect_analysis(model, frame, features, points=10, ice_rows=25):
    """PDP/ICE and ALE for numeric, non-dummy inputs selected by the notebook."""
    tables = {}
    for col in features:
        grid = np.unique(np.quantile(frame[col], np.linspace(0.05, 0.95, points)))
        curves = []
        for value in grid:
            changed = frame.copy()
            changed[col] = value
            curves.append(nonattendance_probability(model, changed))
        curves = np.asarray(curves).T
        ale = ale_1d(model, frame, col)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(grid, curves[:ice_rows].T, color="gray", alpha=0.2)
        axes[0].plot(grid, curves.mean(axis=0), color="navy", linewidth=3, label="PDP")
        axes[0].set(xlabel=col, ylabel="P(nonattendance)", title="PDP + individual ICE")
        axes[0].legend()
        axes[1].plot(ale.value, ale.ale, marker="o")
        axes[1].axhline(0, color="gray", linewidth=1)
        axes[1].set(xlabel=col, ylabel="Centered probability effect", title="ALE")
        fig.tight_layout()
        plt.show()
        tables[col] = {"grid": grid, "ice": curves, "ale": ale}
    return tables


def interaction_analysis(model, frame, features, points=6):
    """Nonadditivity of 2D PDP, not a causal interaction or formal SHAP interaction."""
    records = []
    for first, second in combinations(features, 2):
        grid_a = np.unique(np.quantile(frame[first], np.linspace(0.1, 0.9, points)))
        grid_b = np.unique(np.quantile(frame[second], np.linspace(0.1, 0.9, points)))
        surface = np.empty((len(grid_a), len(grid_b)))
        for i, a in enumerate(grid_a):
            for j, b in enumerate(grid_b):
                changed = frame.copy()
                changed[first], changed[second] = a, b
                surface[i, j] = nonattendance_probability(model, changed).mean()
        residual = surface - surface.mean(axis=0) - surface.mean(axis=1)[:, None] + surface.mean()
        records.append({
            "first": first, "second": second,
            "pdp_nonadditivity_rms": np.sqrt(np.mean(residual ** 2)),
        })
        fig, ax = plt.subplots(figsize=(6, 4))
        image = ax.pcolormesh(grid_b, grid_a, surface, shading="auto", vmin=0, vmax=1)
        ax.set(xlabel=second, ylabel=first, title="2D PDP: P(nonattendance)")
        fig.colorbar(image, ax=ax)
        plt.show()
    return pd.DataFrame(records, columns=["first", "second", "pdp_nonadditivity_rms"])


def global_surrogate(model, training, evaluation, seed=42):
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=0.05, random_state=seed)
    tree.fit(training, nonattendance_probability(model, training))
    reference = nonattendance_probability(model, evaluation)
    approximation = tree.predict(evaluation)
    fidelity = pd.Series({
        "heldout_R2": r2_score(reference, approximation),
        "heldout_MAE": mean_absolute_error(reference, approximation),
    })
    fig, ax = plt.subplots(figsize=(22, 8))
    plot_tree(tree, feature_names=list(training.columns), filled=True, rounded=True, ax=ax)
    ax.set_title("Surrogate tree: approximate P(nonattendance), not the original model")
    plt.show()
    return fidelity, export_text(tree, feature_names=list(training.columns)), tree


def local_surrogate(model, training, patient, groups, seed=42, samples=800):
    """LIME-style grouped empirical perturbations, with separate fidelity evaluation."""
    rng = np.random.default_rng(seed)
    scale = training.std(ddof=0).replace(0, 1)
    center = patient.iloc[0]
    original = training.sample(samples, replace=True, random_state=seed).reset_index(drop=True)
    for columns in groups.values():
        keep = rng.random(samples) < 0.5
        original.loc[keep, columns] = center[columns].to_numpy()
    design = (original - center) / scale
    distance = np.sqrt(np.mean(design.to_numpy() ** 2, axis=1))
    bandwidth = max(float(np.quantile(distance, 0.75)), 1e-6)
    weights = np.exp(-(distance / bandwidth) ** 2)
    train_idx, test_idx = train_test_split(np.arange(samples), test_size=0.3, random_state=seed)
    labels = nonattendance_probability(model, original)
    regression = Ridge(alpha=1.0)
    regression.fit(design.iloc[train_idx], labels[train_idx], sample_weight=weights[train_idx])
    predicted = regression.predict(design.iloc[test_idx])
    fidelity = pd.Series({
        "local_heldout_weighted_R2": r2_score(
            labels[test_idx], predicted, sample_weight=weights[test_idx]
        ),
        "local_heldout_weighted_MAE": mean_absolute_error(
            labels[test_idx], predicted, sample_weight=weights[test_idx]
        ),
        "patient_actual_probability": nonattendance_probability(model, patient)[0],
        "patient_surrogate_probability": regression.predict((patient - center) / scale)[0],
    })
    coefficients = pd.Series(
        regression.coef_, index=training.columns, name="risk_change_per_training_SD"
    )
    return coefficients.sort_values(key=abs, ascending=False), fidelity


def observed_scenarios(model, training, patient, limit=5):
    """Nearest observed lower-risk profiles; no claim of feasible interventions."""
    probability = nonattendance_probability(model, training)
    current = nonattendance_probability(model, patient)[0]
    scale = training.std(ddof=0).replace(0, 1)
    distances = np.sqrt((((training - patient.iloc[0]) / scale) ** 2).mean(axis=1))
    candidates = pd.DataFrame({
        "p_nonattendance": probability, "standardized_distance": distances,
    }, index=training.index)
    candidates = candidates.loc[candidates.p_nonattendance < current].nsmallest(
        limit, "standardized_distance"
    )
    candidates["risk_difference"] = candidates.p_nonattendance - current
    changes = []
    for position in candidates.index:
        difference = training.loc[position].ne(patient.iloc[0])
        changes.append({
            col: (patient.iloc[0][col], training.loc[position, col])
            for col in training.columns[difference]
        })
    candidates["changed_values_(patient,reference)"] = changes
    return candidates


def consensus_table(native, permutation, shap_table):
    scores = pd.concat([
        native.rename("native"),
        permutation.auc_drop.rename("permutation_auc"),
        permutation.logloss_increase.rename("permutation_logloss"),
        shap_table.mean_abs_shap.rename("shap"),
    ], axis=1)
    ranks = scores.rank(ascending=False, method="average").add_suffix("_rank")
    result = pd.concat([scores, ranks], axis=1)
    result["mean_rank"] = ranks.mean(axis=1)
    result["rank_spread"] = ranks.max(axis=1) - ranks.min(axis=1)
    return result.sort_values("mean_rank")
