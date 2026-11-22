"""The challenge metric and cross-validation that holds out whole sites."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def score(y, p):
    """Challenge score: 0.6 F1 at the fixed 0.5 cut + 0.4 ROC-AUC."""
    return 0.6 * f1_score(y, p >= 0.5) + 0.4 * roc_auc_score(y, p)


def cross_val_predict(make_model, X, y, sites=None, repeats=10, folds=5):
    """Out-of-fold probabilities, one row per repeat.

    With `sites`, every fold holds out whole sites, as the test set does;
    without them, records are split at random.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    predictions = np.zeros((repeats, len(y)))
    for r in range(repeats):
        if sites is None:
            splits = StratifiedKFold(folds, shuffle=True, random_state=r).split(X, y)
        else:
            splits = StratifiedGroupKFold(folds, shuffle=True, random_state=r).split(X, y, sites)
        for fit_rows, held_out in splits:
            model = make_model(seed=r).fit(X[fit_rows], y[fit_rows])
            predictions[r, held_out] = model.predict_proba(X[held_out])[:, 1]
    return predictions


def summarize(predictions, y):
    """Mean and standard deviation over repeats of the score, F1 and AUC."""
    rows = [(score(y, p), f1_score(y, p >= 0.5), roc_auc_score(y, p)) for p in predictions]
    table = pd.DataFrame(rows, columns=["score", "F1", "AUC"])
    return table.agg(["mean", "std"])


def compare(baseline, candidate, y, sites, n_boot=2000, seed=0):
    """Score difference candidate − baseline, on the same folds.

    The interval resamples whole sites, so it reflects how the difference
    would vary on another set of villages.
    """
    y = np.asarray(y)
    per_repeat = [score(y, c) - score(y, b) for b, c in zip(baseline, candidate)]
    base, cand = baseline.mean(0), candidate.mean(0)
    rows_of = {s: np.flatnonzero(np.asarray(sites) == s) for s in np.unique(sites)}
    names = list(rows_of)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        rows = np.concatenate([rows_of[s] for s in rng.choice(names, len(names))])
        boot.append(score(y[rows], cand[rows]) - score(y[rows], base[rows]))
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"difference": np.mean(per_repeat), "low": low, "high": high,
            "P(better)": np.mean(np.array(boot) > 0)}
