"""Tests for the threshold calibration in src/detoxify_eval.py.

Run with: uv run pytest
"""

import pandas as pd
import pytest

import detoxify_eval


def test_best_threshold_handmade():
    # Thresholds and resulting F1 (toxic if score >= t):
    #   t=0.35 -> preds [0,1,1,1]: P=2/3, R=1,   F1=0.8   <- best
    #   t=0.40 -> preds [0,1,0,1]: P=1/2, R=1/2, F1=0.5
    #   t=0.80 -> preds [0,0,0,1]: P=1,   R=1/2, F1=0.667
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    assert detoxify_eval.best_threshold(labels, scores) == pytest.approx(0.35)


def test_calibration_uses_only_other_folds():
    # Fold 0 is perfectly separated at 0.9; fold 1 at 0.2. Each fold must get the threshold
    # learned from the OTHER fold, so both folds end up with wrong predictions here.
    data = pd.DataFrame({
        "fold":     [0,   0,    1,   1],
        "is_toxic": [0,   1,    0,   1],
    })
    scores = pd.Series([0.5, 0.95, 0.1, 0.3])
    pred, thresholds = detoxify_eval.calibrated_predictions(data, scores, n_folds=2)
    assert thresholds == {0: pytest.approx(0.3), 1: pytest.approx(0.95)}
    assert list(pred) == [1, 1, 0, 0]
