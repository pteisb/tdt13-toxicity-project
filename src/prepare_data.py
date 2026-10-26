"""Download the ToxiCR code review dataset and create the two frozen fold files.

Reads:  data/raw/code-review-dataset-full.xlsx (downloaded from a pinned ToxiCR commit if missing)
Writes: data/clean_folds.csv   one row per normalized text, conflicting labels removed
        data/leaky_folds.csv   all non-empty rows, no deduplication (mimics the original protocol)
        results/data_stats.csv the summary printed at the end

Both fold files have the columns: id, message, is_toxic, fold, key
  id    = row position in the original xlsx (0-based), the same in both files
  key   = normalized text: lowercase, all whitespace collapsed to single spaces
  fold  = 0-4, from StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on is_toxic

The fold files must never change once created, so the script refuses to overwrite them.

Run:  uv run python src/prepare_data.py           (first time)
      uv run python src/prepare_data.py --force   (overwrite existing fold files)
"""

import argparse
import hashlib
import sys

import pandas as pd
import requests
from sklearn.model_selection import StratifiedKFold

from config import CLEAN_FOLDS, LEAKY_FOLDS, N_FOLDS, RAW_DIR, RESULTS_DIR, SEED

# Pinned to a fixed commit of https://github.com/WSU-SEAL/ToxiCR (master on 26 Oct 2026),
# so the exact same file is downloaded every time.
TOXICR_COMMIT = "e71370aa175aa057d505229e4f25fcc687cf4a2b"
DATA_URL = (
    f"https://raw.githubusercontent.com/WSU-SEAL/ToxiCR/{TOXICR_COMMIT}"
    "/models/code-review-dataset-full.xlsx"
)
RAW_FILE = RAW_DIR / "code-review-dataset-full.xlsx"
STATS_FILE = RESULTS_DIR / "data_stats.csv"
FOLD_COLUMNS = ["id", "message", "is_toxic", "fold", "key"]


def download_raw_file():
    """Download the xlsx into data/raw/ unless it is already there."""
    if RAW_FILE.exists():
        print(f"Raw file already present, skipping download: {RAW_FILE}")
        return
    print(f"Downloading {DATA_URL}")
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    # An xlsx file is a zip archive, so it starts with "PK". Anything else (an HTML error
    # page, a Git LFS pointer) means the download went wrong.
    if not response.content.startswith(b"PK"):
        sys.exit("Downloaded file is not an xlsx file. Check the URL.")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RAW_FILE.write_bytes(response.content)


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    """Lowercase and collapse all whitespace (spaces, tabs, newlines) to single spaces."""
    return " ".join(text.lower().split())


def assign_folds(df):
    """Return a copy of df with a 'fold' column (0-4), stratified on is_toxic."""
    df = df.sort_values("id").reset_index(drop=True)
    df["fold"] = -1
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, test_index) in enumerate(splitter.split(df, df["is_toxic"])):
        df.loc[test_index, "fold"] = fold
    return df


def seen_share_per_fold(df):
    """For each fold: the share of its rows whose key also appears in the other four folds."""
    shares = {}
    for fold in range(N_FOLDS):
        test = df[df["fold"] == fold]
        train_keys = set(df.loc[df["fold"] != fold, "key"])
        shares[fold] = test["key"].isin(train_keys).mean()
    return shares


def fold_counts(df):
    """Rows and toxic rows per fold, as a small table."""
    return df.groupby("fold")["is_toxic"].agg(rows="size", toxic="sum")


def main():
    parser = argparse.ArgumentParser(description="Download ToxiCR and create the fold files.")
    parser.add_argument("--force", action="store_true", help="overwrite existing fold files")
    args = parser.parse_args()

    existing = [p for p in (CLEAN_FOLDS, LEAKY_FOLDS) if p.exists()]
    if existing and not args.force:
        names = ", ".join(str(p) for p in existing)
        sys.exit(f"Refusing to overwrite existing fold files: {names}\n"
                 "These files are frozen. Use --force only if you are sure.")

    download_raw_file()
    checksum = sha256_of(RAW_FILE)

    # Load and drop empty messages
    raw = pd.read_excel(RAW_FILE)
    raw["id"] = range(len(raw))
    raw_rows = len(raw)
    is_empty = raw["message"].isna() | (raw["message"].astype(str).str.strip() == "")
    data = raw[~is_empty].copy()
    data["message"] = data["message"].astype(str)
    data["is_toxic"] = data["is_toxic"].astype(int)
    data["key"] = data["message"].map(normalize)

    # Keys whose rows carry both labels are dropped entirely
    labels_per_key = data.groupby("key")["is_toxic"].nunique()
    conflicting_keys = set(labels_per_key[labels_per_key > 1].index)
    is_conflict = data["key"].isin(conflicting_keys)

    # Clean data: no conflicts, one row per key (the first occurrence in the file)
    clean = data[~is_conflict].drop_duplicates(subset="key", keep="first")
    clean = assign_folds(clean)[FOLD_COLUMNS]

    # Leaky data: every non-empty row, same fold procedure
    leaky = assign_folds(data)[FOLD_COLUMNS]
    seen = seen_share_per_fold(leaky)

    clean.to_csv(CLEAN_FOLDS, index=False)
    leaky.to_csv(LEAKY_FOLDS, index=False)

    # Collect the summary as rows of (dataset, fold, stat, value)
    stats = [
        ("raw", "all", "source_commit", TOXICR_COMMIT),
        ("raw", "all", "sha256", checksum),
        ("raw", "all", "rows", raw_rows),
        ("raw", "all", "toxic", int(raw["is_toxic"].sum())),
        ("raw", "all", "empty_rows", int(is_empty.sum())),
        ("raw", "all", "unique_keys", data["key"].nunique()),
        ("raw", "all", "conflicting_keys", len(conflicting_keys)),
        ("raw", "all", "conflicting_rows", int(is_conflict.sum())),
    ]
    for name, df in [("clean", clean), ("leaky", leaky)]:
        stats.append((name, "all", "rows", len(df)))
        stats.append((name, "all", "toxic", int(df["is_toxic"].sum())))
        for fold, row in fold_counts(df).iterrows():
            stats.append((name, fold, "rows", int(row["rows"])))
            stats.append((name, fold, "toxic", int(row["toxic"])))
    for fold, share in seen.items():
        stats.append(("leaky", fold, "seen_share", round(share, 4)))
    pooled_seen = sum(seen[f] * (leaky["fold"] == f).sum() for f in seen) / len(leaky)
    stats.append(("leaky", "all", "seen_share", round(pooled_seen, 4)))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stats, columns=["dataset", "fold", "stat", "value"]).to_csv(STATS_FILE, index=False)

    # Print the summary
    print()
    print(f"Source commit:     {TOXICR_COMMIT}")
    print(f"SHA-256 of xlsx:   {checksum}")
    print(f"Raw rows:          {raw_rows:,} ({int(raw['is_toxic'].sum()):,} toxic)")
    print(f"Empty rows:        {int(is_empty.sum())}")
    print(f"Unique keys:       {data['key'].nunique():,}")
    print(f"Conflicting keys:  {len(conflicting_keys)} ({int(is_conflict.sum())} rows)")
    print(f"Clean rows:        {len(clean):,} ({int(clean['is_toxic'].sum()):,} toxic)")
    print(f"Leaky rows:        {len(leaky):,} ({int(leaky['is_toxic'].sum()):,} toxic)")
    print()
    table = pd.concat(
        {"clean": fold_counts(clean), "leaky": fold_counts(leaky)}, axis=1
    )
    table[("leaky", "seen_share")] = pd.Series(seen).map("{:.1%}".format)
    print("Per fold:")
    print(table.to_string())
    print(f"\nLeaky seen share, all folds: {pooled_seen:.1%}")
    print(f"\nWrote {CLEAN_FOLDS}\n      {LEAKY_FOLDS}\n      {STATS_FILE}")


if __name__ == "__main__":
    main()
