"""Draw the fixed 2,000-comment LLM evaluation subset and the few-shot pool.

LLMs are too slow to run on all 12,722 clean comments, so they are evaluated on a stratified random
sample of 2,000 (seed 42, same toxic share as the full data). All other systems are also scored on
these ids, using their out-of-fold predictions. Few-shot examples and pilot runs must come from the
remaining comments (the few-shot pool), so nothing used to design prompts is in the evaluation set.

Reads:  data/clean_folds.csv
Writes: data/llm_eval_ids.csv    id, fold, is_toxic   (2,000 rows)
        data/fewshot_pool.csv    all other clean comments, same columns as clean_folds.csv

Both files are frozen once LLM runs use them, so the script refuses to overwrite them.

Run:  uv run python src/make_llm_subset.py           (first time)
      uv run python src/make_llm_subset.py --force   (overwrite existing files)
"""

import argparse
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from config import CLEAN_FOLDS, FEWSHOT_POOL, LLM_EVAL_IDS, SEED

SUBSET_SIZE = 2000
# Comment ids sent to the API in the smoke test (fold 0 toxic comments, chosen before this subset
# existed). They were only used to check API settings, not to design prompts; reported for the record.
SMOKE_TEST_IDS = [16823, 2187, 15460, 18294, 15150, 9655, 856, 6115, 19193, 6441]


def main():
    parser = argparse.ArgumentParser(description="Draw the LLM evaluation subset and few-shot pool.")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    existing = [p for p in (LLM_EVAL_IDS, FEWSHOT_POOL) if p.exists()]
    if existing and not args.force:
        sys.exit(f"Refusing to overwrite: {', '.join(str(p) for p in existing)}\n"
                 "These files are frozen once LLM runs use them. Use --force only if you are sure.")

    data = pd.read_csv(CLEAN_FOLDS)
    subset, pool = train_test_split(data, train_size=SUBSET_SIZE, stratify=data["is_toxic"],
                                    random_state=SEED)
    subset = subset.sort_values("id")
    pool = pool.sort_values("id")

    subset[["id", "fold", "is_toxic"]].to_csv(LLM_EVAL_IDS, index=False)
    pool.to_csv(FEWSHOT_POOL, index=False)

    smoke_in_subset = sorted(set(SMOKE_TEST_IDS) & set(subset["id"]))
    print(f"LLM evaluation subset: {len(subset):,} comments, {subset['is_toxic'].sum()} toxic "
          f"({subset['is_toxic'].mean():.1%}; full clean data {data['is_toxic'].mean():.1%})")
    print("  per fold:", subset.groupby("fold").size().to_dict())
    print(f"Few-shot pool:         {len(pool):,} comments, {pool['is_toxic'].sum()} toxic")
    print(f"Overlap check:         {len(set(subset['id']) & set(pool['id']))} ids in both (must be 0)")
    print(f"Smoke-test comments in the evaluation subset: {smoke_in_subset or 'none'}")
    print(f"\nWrote {LLM_EVAL_IDS}\n      {FEWSHOT_POOL}")


if __name__ == "__main__":
    main()
