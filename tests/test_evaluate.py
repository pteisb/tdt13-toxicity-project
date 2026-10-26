"""Tests for src/evaluate.py, using tiny handmade examples where the right answer is known.

Run with: uv run pytest
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

import evaluate


def make_preds(labels, preds, folds=None, probs=None, ids=None, system="test", setting="clean"):
    """Build a predictions table in the standard format."""
    n = len(labels)
    return pd.DataFrame({
        "id": ids if ids is not None else range(n),
        "fold": folds if folds is not None else [0] * n,
        "label": labels,
        "prob_toxic": probs if probs is not None else [np.nan] * n,
        "pred": preds,
        "system": system,
        "setting": setting,
    })


def folds_for(preds, keys=None):
    """A fold file that matches a predictions table."""
    return pd.DataFrame({
        "id": preds["id"], "message": "x", "is_toxic": preds["label"], "fold": preds["fold"],
        "key": keys if keys is not None else [f"text {i}" for i in preds["id"]],
    })


# ---------- Metrics ----------

def test_perfect_classifier():
    scores = evaluate.classification_scores([1, 1, 0, 0, 0], [1, 1, 0, 0, 0])
    assert scores == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0}


def test_always_non_toxic():
    # Never predicting toxic: no true positives, so precision, recall and F1 are 0,
    # but accuracy is still the share of non-toxic comments (3 of 5).
    scores = evaluate.classification_scores([1, 1, 0, 0, 0], [0, 0, 0, 0, 0])
    assert scores["precision"] == 0.0
    assert scores["recall"] == 0.0
    assert scores["f1"] == 0.0
    assert scores["accuracy"] == pytest.approx(0.6)


def test_handmade_example():
    # 2 true positives, 1 false positive, 1 false negative, 1 true negative
    scores = evaluate.classification_scores([1, 1, 1, 0, 0], [1, 1, 0, 1, 0])
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(2 / 3)
    assert scores["f1"] == pytest.approx(2 / 3)
    assert scores["accuracy"] == pytest.approx(3 / 5)


def test_fast_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y_true = rng.integers(0, 2, 200)
        y_pred = rng.integers(0, 2, 200)
        assert evaluate.f1_from_counts(y_true, y_pred) == pytest.approx(f1_score(y_true, y_pred))
    assert evaluate.f1_from_counts(np.array([0, 0]), np.array([0, 0])) == 0.0


def test_per_fold_scores_and_summary():
    # Fold 0 is perfect (F1 = 1), fold 1 misses its only toxic comment (F1 = 0)
    preds = make_preds(labels=[1, 0, 1, 0], preds=[1, 0, 0, 0], folds=[0, 0, 1, 1])
    folds = evaluate.per_fold_scores(preds)
    assert list(folds["f1"]) == [1.0, 0.0]

    summary = evaluate.summarize(preds)
    assert summary["n"] == 4
    assert summary["n_folds"] == 2
    assert summary["f1_mean"] == pytest.approx(0.5)
    assert summary["f1_pooled"] == pytest.approx(2 / 3)   # tp=1, fn=1, fp=0
    assert np.isnan(summary["ap"])                         # no probabilities given


def test_average_precision():
    perfect_ranking = make_preds([1, 1, 0, 0], [1, 1, 0, 0], probs=[0.9, 0.8, 0.2, 0.1])
    assert evaluate.average_precision(perfect_ranking) == pytest.approx(1.0)
    assert np.isnan(evaluate.average_precision(make_preds([1, 0], [1, 0])))


# ---------- Bootstrap ----------

def test_bootstrap_ci_perfect_classifier():
    labels = [1, 0] * 50
    assert evaluate.bootstrap_f1_ci(labels, labels) == (1.0, 1.0)


def test_bootstrap_ci_contains_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 500)
    y_pred = np.where(rng.random(500) < 0.8, y_true, 1 - y_true)   # about 80% correct
    low, high = evaluate.bootstrap_f1_ci(y_true, y_pred)
    assert low < f1_score(y_true, y_pred) < high
    assert (low, high) == evaluate.bootstrap_f1_ci(y_true, y_pred)  # same seed, same result


def test_paired_bootstrap():
    labels = [1, 0] * 50
    perfect = make_preds(labels, labels)
    never_toxic = make_preds(labels, [0] * 100)

    same = evaluate.paired_bootstrap(perfect, perfect)
    assert (same["f1_diff"], same["ci_low"], same["ci_high"]) == (0.0, 0.0, 0.0)

    better = evaluate.paired_bootstrap(perfect, never_toxic)
    assert better["f1_diff"] == pytest.approx(1.0)
    assert better["ci_low"] > 0                  # interval excludes 0: reliable difference


def test_paired_bootstrap_requires_same_ids():
    a = make_preds([1, 0], [1, 0], ids=[1, 2])
    b = make_preds([1, 0], [1, 0], ids=[1, 3])
    with pytest.raises(ValueError):
        evaluate.paired_bootstrap(a, b)


# ---------- Validation and filtering ----------

def test_valid_predictions_pass():
    preds = make_preds([1, 0, 1], [1, 0, 0], probs=[0.9, 0.1, 0.4])
    evaluate.validate_predictions(preds, folds_for(preds))


def test_subset_of_fold_file_is_allowed():
    preds = make_preds([1, 0, 1], [1, 0, 0])
    evaluate.validate_predictions(preds.iloc[:2], folds_for(preds))


@pytest.mark.parametrize("change, message", [
    (lambda p: p.drop(columns="prob_toxic"), "missing columns"),
    (lambda p: p.assign(id=[0, 0, 1]), "not unique"),
    (lambda p: p.assign(id=[0, 1, 99]), "not in the fold file"),
    (lambda p: p.assign(label=[0, 0, 1]), "label does not match"),
    (lambda p: p.assign(fold=[1, 0, 0]), "fold does not match"),
    (lambda p: p.assign(pred=[1, 0, 2]), "pred must be 0 or 1"),
    (lambda p: p.assign(setting="other"), "setting must be"),
    (lambda p: p.assign(system=["a", "b", "a"]), "exactly one value"),
    (lambda p: p.assign(prob_toxic=[0.5, np.nan, 0.5]), "some rows but not"),
    (lambda p: p.assign(prob_toxic=[0.5, 1.5, 0.5]), "between 0 and 1"),
])
def test_invalid_predictions_are_rejected(change, message):
    preds = make_preds([1, 0, 1], [1, 0, 0])
    folds = folds_for(preds)
    with pytest.raises(ValueError, match=message):
        evaluate.validate_predictions(change(preds), folds)


def test_filter_to_ids():
    preds = make_preds([1, 0, 1, 0], [1, 0, 0, 0])
    subset = evaluate.filter_to_ids(preds, [1, 2])
    assert list(subset["id"]) == [1, 2]
    with pytest.raises(ValueError):
        evaluate.filter_to_ids(preds, [1, 99])     # id 99 has no prediction


# ---------- Seen / unseen ----------

def test_seen_in_training():
    # Fold 0 holds texts a, b; fold 1 holds a, c. Text "a" is in both folds, so it is "seen"
    # (its copy is in the training data) wherever it is tested; b and c are unseen.
    preds = make_preds([1, 0, 1, 0], [1, 0, 1, 0], folds=[0, 0, 1, 1], setting="leaky")
    folds = folds_for(preds, keys=["a", "b", "a", "c"])
    seen = evaluate.seen_in_training(folds)
    assert seen.to_dict() == {0: True, 1: False, 2: True, 3: False}


def test_seen_unseen_scores():
    # Seen comments are predicted perfectly, unseen ones are all wrong.
    preds = make_preds(labels=[1, 0, 1, 1], preds=[1, 0, 1, 0], folds=[0, 0, 1, 1],
                       setting="leaky")
    folds = folds_for(preds, keys=["a", "b", "a", "c"])
    rows = {row["group"]: row for row in evaluate.seen_unseen_scores(preds, folds)}
    assert rows["seen"]["n"] == 2 and rows["seen"]["f1"] == 1.0
    assert rows["unseen"]["n"] == 2 and rows["unseen"]["f1"] == 0.0
    assert rows["all"]["n"] == 4


# ---------- LaTeX ----------

def test_latex_table():
    table = pd.DataFrame([evaluate.summarize(make_preds([1, 0, 1, 0], [1, 0, 0, 0],
                                                        folds=[0, 0, 1, 1], system="tfidf_lr"))])
    latex = evaluate.to_latex(table)
    assert r"\toprule" in latex and r"\bottomrule" in latex
    assert r"tfidf\_lr" in latex
