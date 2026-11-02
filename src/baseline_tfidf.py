"""TF-IDF + logistic regression baseline (system "tfidf_lr"), on the clean and leaky folds.

For each of the 5 folds: fit the TF-IDF vectorizer and the classifier on the other four folds
only, then predict the held-out fold. The regularization strength C is chosen per fold by an
inner stratified 3-fold cross-validation on the training folds (scoring: toxic-class F1).

Settings:
  TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
  LogisticRegression(class_weight="balanced", max_iter=2000, tol=1e-6), L2 penalty, with C chosen from
  [0.25, 0.5, 1, 2, 4, 8] by GridSearchCV (scoring="f1", stratified 3-fold, shuffled, seed 42)
  and refit on all training folds with the best C
  prob_toxic = predicted probability of class 1; pred = prob_toxic >= 0.5

Reads:  data/clean_folds.csv, data/leaky_folds.csv
Writes: preds/tfidf_lr_<setting>.csv       predictions (id, fold, label, prob_toxic, pred, system, setting)
        results/tfidf_lr_log.csv           chosen C, inner-CV F1 and vocabulary size per fold

Run:  uv run python src/baseline_tfidf.py                    (both settings)
      uv run python src/baseline_tfidf.py --settings clean   (one setting)
"""

import argparse
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from config import (
    CLEAN_FOLDS,
    LEAKY_FOLDS,
    N_FOLDS,
    PRED_COLUMNS,
    PREDS_DIR,
    RESULTS_DIR,
    SEED,
)

SYSTEM = "tfidf_lr"
C_GRID = [0.25, 0.5, 1, 2, 4, 8]
FOLD_FILES = {"clean": CLEAN_FOLDS, "leaky": LEAKY_FOLDS}
LOG_FILE = RESULTS_DIR / "tfidf_lr_log.csv"
# Optimizer tolerance. scikit-learn's default (1e-4) stops before the solution is fully
# converged; 1e-6 gives the same predictions as 1e-8, so the result no longer depends on it.
TOL = 1e-6


def train_and_predict(train, test):
    """Fit on the training rows, return toxic probabilities for the test rows and a log entry."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train["message"])   # vocabulary learned from train only
    X_test = vectorizer.transform(test["message"])

    # Try every C with an inner 3-fold CV on the training rows and keep the C with the best
    # mean F1. The default penalty is L2.
    search = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=2000, tol=TOL),
        param_grid={"C": C_GRID},
        scoring="f1",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        refit=False,
    )
    search.fit(X_train, train["is_toxic"])
    inner_f1 = search.cv_results_["mean_test_score"]   # one value per C, in C_GRID order
    chosen_C = search.best_params_["C"]

    # Final model: all training rows, chosen C
    model = LogisticRegression(C=chosen_C, class_weight="balanced", max_iter=2000, tol=TOL)
    model.fit(X_train, train["is_toxic"])
    log = {
        "chosen_C": chosen_C,
        "inner_cv_f1": search.best_score_,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "n_train": len(train),
    }
    log.update({f"inner_f1_C={c}": f1 for c, f1 in zip(C_GRID, inner_f1)})
    return model.predict_proba(X_test)[:, 1], log


def run_setting(setting):
    """Cross-validated predictions for one fold file. Returns (predictions, log rows)."""
    data = pd.read_csv(FOLD_FILES[setting])
    pred_parts: list[pd.DataFrame] = []
    log_rows: list[dict] = []
    for fold in range(N_FOLDS):
        start = time.time()
        train = data[data["fold"] != fold]
        test = data[data["fold"] == fold]
        prob, log = train_and_predict(train, test)
        pred_parts.append(pd.DataFrame({
            "id": test["id"],
            "fold": fold,
            "label": test["is_toxic"],
            "prob_toxic": prob,
            "pred": (prob >= 0.5).astype(int),
            "system": SYSTEM,
            "setting": setting,
        }))
        log_rows.append({"setting": setting, "fold": fold, **log,
                         "seconds": round(time.time() - start, 1)})
        print(f"  {setting} fold {fold}: chosen C = {log['chosen_C']}, "
              f"inner-CV F1 = {log['inner_cv_f1']:.3f}, {log_rows[-1]['seconds']} s")
    return pd.concat(pred_parts)[PRED_COLUMNS], log_rows


def main():
    parser = argparse.ArgumentParser(description="TF-IDF + logistic regression baseline.")
    parser.add_argument("--settings", nargs="+", choices=list(FOLD_FILES), default=list(FOLD_FILES))
    args = parser.parse_args()

    PREDS_DIR.mkdir(exist_ok=True)
    all_logs, written = [], []
    for setting in args.settings:
        print(f"Setting: {setting}")
        preds, log_rows = run_setting(setting)
        path = PREDS_DIR / f"{SYSTEM}_{setting}.csv"
        preds.to_csv(path, index=False)
        written.append(path)
        all_logs += log_rows

    # Keep log rows from settings that were not rerun this time
    log = pd.DataFrame(all_logs)
    if LOG_FILE.exists():
        old = pd.read_csv(LOG_FILE)
        log = pd.concat([old[~old["setting"].isin(args.settings)], log])
    log.round(4).to_csv(LOG_FILE, index=False)
    written.append(LOG_FILE)

    print("\nWrote " + "\n      ".join(str(p) for p in written))
    print("Metrics: uv run python src/make_results.py")


if __name__ == "__main__":
    main()
