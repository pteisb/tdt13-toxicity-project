"""Off-the-shelf Detoxify models (no training on our data) on all clean comments.

Systems written:
  detoxify_original       Detoxify('original'): BERT-base trained on Jigsaw Wikipedia comments
  detoxify_unbiased       Detoxify('unbiased'): RoBERTa-base trained on Jigsaw Civil Comments
  detoxify_original_cal   the same scores, with a calibrated threshold (below)
  detoxify_unbiased_cal   the same, for 'unbiased'

prob_toxic = Detoxify's 'toxicity' score. Uncalibrated systems predict toxic if the score is >= 0.5.
Calibrated systems: for each fold, the threshold that gives the highest F1 on the OTHER four folds
is chosen and then applied to the fold itself, so the test fold never influences its threshold.
Comments are truncated to 2,000 characters before scoring.

Reads:  data/clean_folds.csv
Writes: data/detoxify_scores.csv           cached toxicity scores (id, original, unbiased), so the
                                           slow scoring runs once; delete it to rescore
        preds/detoxify_<model>_clean.csv, preds/detoxify_<model>_cal_clean.csv
        results/detoxify_thresholds.csv    calibrated threshold per model and fold

Run:  uv run python src/detoxify_eval.py                    (CPU, about 30-60 minutes the first time)
      uv run python src/detoxify_eval.py --batch-size 64
"""

import argparse
import time

import numpy as np
import pandas as pd
from detoxify import Detoxify
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

from config import CLEAN_FOLDS, DATA_DIR, N_FOLDS, PRED_COLUMNS, PREDS_DIR, RESULTS_DIR

MODELS = ["original", "unbiased"]
MAX_CHARS = 2000
SCORES_FILE = DATA_DIR / "detoxify_scores.csv"
THRESHOLDS_FILE = RESULTS_DIR / "detoxify_thresholds.csv"


def score_comments(model_name, texts, batch_size):
    """Detoxify 'toxicity' score for each text, in the original order.

    Texts are processed shortest first: each batch is padded to its longest text, so batching
    texts of similar length is much faster. The scores themselves do not depend on the order.
    """
    model = Detoxify(model_name, device="cpu")
    order = np.argsort([len(t) for t in texts], kind="stable")
    scores = np.empty(len(texts))
    for start in tqdm(range(0, len(texts), batch_size), desc=f"Detoxify {model_name}", unit="batch"):
        batch_index = order[start:start + batch_size]
        result = model.predict([texts[i] for i in batch_index])
        scores[batch_index] = result["toxicity"]
    return scores


def load_or_compute_scores(data, batch_size):
    """Read cached scores and compute any model that is missing from the cache."""
    if SCORES_FILE.exists():
        scores = pd.read_csv(SCORES_FILE)
        if not scores["id"].equals(data["id"]):
            raise SystemExit(f"{SCORES_FILE} does not match the fold file; delete it to rescore.")
    else:
        scores = data[["id"]].copy()

    texts = data["message"].str.slice(0, MAX_CHARS).tolist()
    for model_name in MODELS:
        if model_name in scores.columns:
            print(f"Using cached scores for '{model_name}'")
            continue
        start = time.time()
        scores[model_name] = score_comments(model_name, texts, batch_size)
        scores.to_csv(SCORES_FILE, index=False)    # save after each model, so a crash keeps it
        print(f"Scored '{model_name}' in {(time.time() - start) / 60:.1f} minutes")
    return scores


def best_threshold(labels, scores):
    """The threshold t that maximizes toxic-class F1 when predicting toxic for score >= t."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # precision and recall have one more entry than thresholds (the end point); drop it
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    return thresholds[np.argmax(f1)]


def calibrated_predictions(data, scores, n_folds=N_FOLDS):
    """Per fold: threshold chosen on the other folds, applied to this fold."""
    pred = pd.Series(0, index=data.index)
    thresholds = {}
    for fold in range(n_folds):
        train = data["fold"] != fold
        threshold = best_threshold(data.loc[train, "is_toxic"], scores[train])
        thresholds[fold] = threshold
        pred[~train] = (scores[~train] >= threshold).astype(int)
    return pred, thresholds


def prediction_table(data, scores, pred, system):
    return pd.DataFrame({
        "id": data["id"],
        "fold": data["fold"],
        "label": data["is_toxic"],
        "prob_toxic": scores,
        "pred": pred,
        "system": system,
        "setting": "clean",
    })[PRED_COLUMNS]


def main():
    parser = argparse.ArgumentParser(description="Score clean comments with Detoxify.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data = pd.read_csv(CLEAN_FOLDS)
    scores = load_or_compute_scores(data, args.batch_size)

    PREDS_DIR.mkdir(exist_ok=True)
    written, threshold_rows = [SCORES_FILE], []
    for model_name in MODELS:
        model_scores = scores[model_name]
        variants = {f"detoxify_{model_name}": (model_scores >= 0.5).astype(int)}
        variants[f"detoxify_{model_name}_cal"], thresholds = calibrated_predictions(data, model_scores)
        threshold_rows += [{"model": model_name, "fold": f, "threshold": t} for f, t in thresholds.items()]
        print(f"'{model_name}' calibrated thresholds per fold: "
              + ", ".join(f"{t:.3f}" for t in thresholds.values()))
        for system, pred in variants.items():
            path = PREDS_DIR / f"{system}_clean.csv"
            prediction_table(data, model_scores, pred, system).to_csv(path, index=False)
            written.append(path)

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(threshold_rows).round(4).to_csv(THRESHOLDS_FILE, index=False)
    written.append(THRESHOLDS_FILE)

    print("\nWrote " + "\n      ".join(str(p) for p in written))
    print("Metrics: uv run python src/evaluate.py --preds preds/*.csv --out results/")


if __name__ == "__main__":
    main()
