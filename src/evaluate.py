"""Compute all metrics from prediction files. Model scripts never compute metrics themselves.

Reads:  one or more prediction CSVs (columns: id, fold, label, prob_toxic, pred, system,
        setting) and the matching fold file in data/, used to validate ids, folds and labels
        and for the seen/unseen split.
Writes: nothing by default (prints the tables). With --out:
        <out>/<name>.csv               one row per prediction file, sorted by pooled F1
        <out>/<name>.tex               the same table in LaTeX (booktabs)
        <out>/<name>_per_fold.csv      precision, recall, F1 and accuracy for every fold
        <out>/<name>_seen_unseen.csv   F1 on seen vs unseen comments (only for leaky files)

The final results package (all tables and figures) is built by src/make_results.py, which uses the
functions in this file. This command line is for quick checks of individual prediction files.

All metrics are for the toxic class (label 1):
  - precision, recall and F1 per fold, reported as mean and standard deviation over folds
  - precision, recall and F1 on all predictions pooled, with a 95% bootstrap CI for F1
  - accuracy, and average precision (AP) when prob_toxic is filled in

Run:  uv run python src/evaluate.py --preds preds/roberta_base_clean.csv preds/tfidf_lr_clean.csv
      uv run python src/evaluate.py --preds "preds/*_v1_clean.csv" --subset data/llm_eval_ids.csv
      (add --out <folder> to also save the tables)
"""

import argparse
import functools
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
)

from config import CLEAN_FOLDS, LEAKY_FOLDS, PRED_COLUMNS, SEED, SETTINGS

FOLD_FILES = {"clean": CLEAN_FOLDS, "leaky": LEAKY_FOLDS}
N_RESAMPLES = 2000


# ---------- Loading and validation ----------

@functools.cache
def load_folds(setting):
    """Read the fold file for 'clean' or 'leaky' (read once, then reused)."""
    return pd.read_csv(FOLD_FILES[setting])


def validate_predictions(preds, folds):
    """Check a predictions table against the schema and the fold file. Raises ValueError."""
    missing = [col for col in PRED_COLUMNS if col not in preds.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for col in ["system", "setting"]:
        if preds[col].nunique() != 1:
            raise ValueError(f"column '{col}' must contain exactly one value")
    if preds["setting"].iloc[0] not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}")
    if not preds["id"].is_unique:
        raise ValueError("ids are not unique")
    if not preds["pred"].isin([0, 1]).all():
        raise ValueError("pred must be 0 or 1")

    prob = preds["prob_toxic"]
    if prob.notna().any():
        if prob.isna().any():
            raise ValueError("prob_toxic is filled in for some rows but not for others")
        if not prob.between(0, 1).all():
            raise ValueError("prob_toxic must be between 0 and 1")

    merged = preds.merge(folds[["id", "fold", "is_toxic"]], on="id", how="left",
                         suffixes=("", "_file"), indicator=True)
    n_unknown = (merged["_merge"] == "left_only").sum()
    if n_unknown:
        raise ValueError(f"{n_unknown} ids are not in the fold file")
    if (merged["fold"] != merged["fold_file"]).any():
        raise ValueError("fold does not match the fold file")
    if (merged["label"] != merged["is_toxic"]).any():
        raise ValueError("label does not match the fold file")


def load_predictions(path):
    """Read a predictions CSV and validate it against its fold file."""
    preds = pd.read_csv(path)
    setting = preds["setting"].iloc[0] if "setting" in preds.columns and len(preds) else None
    if setting not in SETTINGS:
        raise ValueError(f"{path}: setting must be one of {SETTINGS}")
    try:
        validate_predictions(preds, load_folds(setting))
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None
    return preds


def filter_to_ids(preds, ids):
    """Keep only the given ids. Raises ValueError if any of them has no prediction."""
    ids = set(ids)
    missing = ids - set(preds["id"])
    if missing:
        raise ValueError(f"{len(missing)} of the requested ids have no prediction")
    return preds[preds["id"].isin(ids)].reset_index(drop=True)


# ---------- Metrics ----------

def classification_scores(y_true, y_pred):
    """Precision, recall and F1 for the toxic class, plus accuracy."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy_score(y_true, y_pred)}


def f1_from_counts(y_true, y_pred):
    """Toxic-class F1 computed directly from counts. Same result as sklearn, but much faster,
    which matters when it is called thousands of times in the bootstrap."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def confusion_counts(y_true, y_pred):
    """True/false positives and negatives, with toxic as the positive class."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"tp": int(np.sum((y_true == 1) & (y_pred == 1))), "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
            "fn": int(np.sum((y_true == 1) & (y_pred == 0))), "tn": int(np.sum((y_true == 0) & (y_pred == 0)))}


def false_positive_rate(y_true, y_pred):
    """Share of non-toxic comments predicted toxic, or NaN if there are no non-toxic comments."""
    counts = confusion_counts(y_true, y_pred)
    negatives = counts["fp"] + counts["tn"]
    return counts["fp"] / negatives if negatives else float("nan")


def per_fold_scores(preds):
    """One row of scores per fold."""
    rows = {fold: classification_scores(group["label"], group["pred"])
            for fold, group in preds.groupby("fold")}
    return pd.DataFrame.from_dict(rows, orient="index")


def average_precision(preds):
    """Average precision (area under the precision-recall curve), or NaN if there are no
    probabilities (LLMs) or only one class is present."""
    if preds["prob_toxic"].isna().any() or preds["label"].nunique() < 2:
        return float("nan")
    return average_precision_score(preds["label"], preds["prob_toxic"])


def bootstrap_f1_ci(y_true, y_pred, n_resamples=N_RESAMPLES, seed=SEED):
    """95% confidence interval for F1: resample the comments with replacement many times,
    compute F1 each time, and take the 2.5th and 97.5th percentiles."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.integers(0, n, n)
        scores[i] = f1_from_counts(y_true[sample], y_pred[sample])
    low, high = np.percentile(scores, [2.5, 97.5])
    return low, high


def paired_bootstrap(preds_a, preds_b, n_resamples=N_RESAMPLES, seed=SEED):
    """F1 of system A minus F1 of system B, with a 95% CI.

    Each resample draws the same comments for both systems, so the interval describes the
    difference itself. If the interval does not contain 0, the difference is reliable.
    """
    merged = preds_a[["id", "label", "pred"]].merge(
        preds_b[["id", "pred"]], on="id", suffixes=("_a", "_b"))
    if not len(merged) == len(preds_a) == len(preds_b):
        raise ValueError("the two systems must have predictions for exactly the same ids")
    y_true = merged["label"].to_numpy()
    pred_a, pred_b = merged["pred_a"].to_numpy(), merged["pred_b"].to_numpy()

    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.integers(0, n, n)
        diffs[i] = (f1_from_counts(y_true[sample], pred_a[sample])
                    - f1_from_counts(y_true[sample], pred_b[sample]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    observed = f1_from_counts(y_true, pred_a) - f1_from_counts(y_true, pred_b)
    return {"n": n, "f1_diff": observed, "ci_low": low, "ci_high": high}


def summarize(preds):
    """All main-table numbers for one predictions table."""
    folds = per_fold_scores(preds)
    pooled = classification_scores(preds["label"], preds["pred"])
    ci_low, ci_high = bootstrap_f1_ci(preds["label"], preds["pred"])
    return {
        "system": preds["system"].iloc[0],
        "setting": preds["setting"].iloc[0],
        "n": len(preds),
        "n_folds": len(folds),
        # standard deviation over folds uses ddof=1 (the pandas default)
        "precision_mean": folds["precision"].mean(),
        "precision_sd": folds["precision"].std(),
        "recall_mean": folds["recall"].mean(),
        "recall_sd": folds["recall"].std(),
        "f1_mean": folds["f1"].mean(),
        "f1_sd": folds["f1"].std(),
        "precision_pooled": pooled["precision"],
        "recall_pooled": pooled["recall"],
        "f1_pooled": pooled["f1"],
        "f1_ci_low": ci_low,
        "f1_ci_high": ci_high,
        "accuracy": pooled["accuracy"],
        "ap": average_precision(preds),
    }


# ---------- Seen / unseen split (leaky setting) ----------

def seen_in_training(folds):
    """For each id: True if the same normalized text (key) appears in another fold, i.e. in the
    training data of the model that predicted this comment. Returns a Series indexed by id."""
    seen = pd.Series(False, index=folds.index)
    for fold in folds["fold"].unique():
        in_fold = folds["fold"] == fold
        train_keys = set(folds.loc[~in_fold, "key"])
        seen[in_fold] = folds.loc[in_fold, "key"].isin(train_keys)
    return pd.Series(seen.to_numpy(), index=folds["id"], name="seen")


def seen_unseen_scores(preds, folds):
    """Precision, recall and F1 (with CI) for all, seen and unseen comments."""
    seen = preds["id"].map(seen_in_training(folds))
    groups = {"all": preds, "seen": preds[seen], "unseen": preds[~seen]}
    rows = []
    for group, part in groups.items():
        scores = classification_scores(part["label"], part["pred"])
        ci_low, ci_high = bootstrap_f1_ci(part["label"], part["pred"])
        rows.append({"system": preds["system"].iloc[0], "setting": preds["setting"].iloc[0],
                     "group": group, "n": len(part), "toxic": int(part["label"].sum()),
                     "precision": scores["precision"], "recall": scores["recall"],
                     "f1": scores["f1"], "f1_ci_low": ci_low, "f1_ci_high": ci_high})
    return rows


# ---------- Output ----------

def to_latex(table):
    """The main table as a LaTeX tabular (booktabs). Wrap it in a table environment in the
    report with \\input{...}."""
    def mean_sd(row, metric):
        return f"{row[metric + '_mean']:.3f} $\\pm$ {row[metric + '_sd']:.3f}"

    lines = [
        r"\begin{tabular}{llrccccc}",
        r"\toprule",
        r"System & Setting & $n$ & Precision & Recall & F1 & Pooled F1 [95\% CI] & AP \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        system = str(row["system"]).replace("_", r"\_")
        ap = "--" if pd.isna(row["ap"]) else f"{row['ap']:.3f}"
        pooled = f"{row['f1_pooled']:.3f} [{row['f1_ci_low']:.3f}, {row['f1_ci_high']:.3f}]"
        lines.append(f"{system} & {row['setting']} & {row['n']:,} & {mean_sd(row, 'precision')} & "
                     f"{mean_sd(row, 'recall')} & {mean_sd(row, 'f1')} & {pooled} & {ap} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def expand_paths(patterns):
    """Expand glob patterns ourselves, because PowerShell does not expand preds/*.csv."""
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            sys.exit(f"No file matches: {pattern}")
        paths.extend(matches)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Compute metrics from prediction files.")
    parser.add_argument("--preds", nargs="+", required=True, help="prediction CSVs; globs allowed")
    parser.add_argument("--out", help="folder to save the tables in (default: only print them)")
    parser.add_argument("--subset", help="CSV with an 'id' column: score only these ids")
    parser.add_argument("--name", default="main_table", help="base name of the output files")
    args = parser.parse_args()

    subset_ids = pd.read_csv(args.subset)["id"] if args.subset else None
    rows, fold_rows, seen_rows = [], [], []
    for path in expand_paths(args.preds):
        try:
            preds = load_predictions(path)
            if subset_ids is not None:
                preds = filter_to_ids(preds, subset_ids)
        except ValueError as error:
            sys.exit(f"Invalid predictions file. {error}")
        setting = preds["setting"].iloc[0]
        if subset_ids is None and len(preds) < len(load_folds(setting)):
            print(f"Warning: {path} covers {len(preds):,} of {len(load_folds(setting)):,} "
                  f"comments in the {setting} fold file")
        rows.append(summarize(preds))
        folds = per_fold_scores(preds).rename_axis("fold").reset_index()
        folds.insert(0, "system", preds["system"].iloc[0])
        folds.insert(1, "setting", setting)
        fold_rows.append(folds)
        if setting == "leaky":
            seen_rows += seen_unseen_scores(preds, load_folds(setting))

    table = pd.DataFrame(rows).sort_values("f1_pooled", ascending=False)
    show = table[["system", "setting", "n", "f1_mean", "f1_sd", "f1_pooled", "f1_ci_low",
                  "f1_ci_high", "precision_pooled", "recall_pooled", "ap"]]
    print(show.to_string(index=False, float_format="{:.3f}".format))

    per_fold = pd.concat(fold_rows)
    print("\nF1 per fold:")
    print(per_fold.pivot(index=["system", "setting"], columns="fold", values="f1")
          .to_string(float_format="{:.3f}".format))

    seen_table = pd.DataFrame(seen_rows)
    if seen_rows:
        print("\nLeaky setting, seen vs unseen comments:")
        print(seen_table.to_string(index=False, float_format="{:.3f}".format))

    if not args.out:
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = [out / f"{args.name}.csv", out / f"{args.name}.tex", out / f"{args.name}_per_fold.csv"]
    table.round(4).to_csv(written[0], index=False)
    written[1].write_text(to_latex(table), encoding="utf-8")
    per_fold.round(4).to_csv(written[2], index=False)
    if seen_rows:
        written.append(out / f"{args.name}_seen_unseen.csv")
        seen_table.round(4).to_csv(written[-1], index=False)
    print("\nWrote " + "\n      ".join(str(p) for p in written))


if __name__ == "__main__":
    main()
