"""Draw the 50 comments for the LLM pilot, used to check the prompts before the full runs.

They come from the few-shot pool, so nothing in the evaluation subset influences the prompts. The 8
comments used as examples in prompt C are left out, because the model would see their answers.

Reads:  data/fewshot_pool.csv
Writes: data/llm_pilot_ids.csv    id, fold, is_toxic   (50 rows)

Run:  uv run python src/make_llm_pilot_ids.py
"""

import pandas as pd

from config import DATA_DIR, FEWSHOT_POOL, LLM_EVAL_IDS, PROMPTS_DIR, SEED

PILOT_SIZE = 50
PILOT_IDS = DATA_DIR / "llm_pilot_ids.csv"
FEWSHOT_IDS = [7884, 6429, 3005, 600, 1411, 4408, 18544, 19324]    # the examples in fewshot_v1.txt


def main():
    pool = pd.read_csv(FEWSHOT_POOL)
    examples = pool[pool["id"].isin(FEWSHOT_IDS)]
    # Compared with whitespace collapsed, because the prompt file holds the examples on single lines.
    fewshot_text = " ".join((PROMPTS_DIR / "fewshot_v1.txt").read_text(encoding="utf-8").split())
    if (len(examples) != len(FEWSHOT_IDS)
            or not all(" ".join(m.split()) in fewshot_text for m in examples["message"])):
        raise SystemExit("FEWSHOT_IDS do not match the comments in prompts/fewshot_v1.txt")

    pilot = pool[~pool["id"].isin(FEWSHOT_IDS)].sample(PILOT_SIZE, random_state=SEED).sort_values("id")
    pilot[["id", "fold", "is_toxic"]].to_csv(PILOT_IDS, index=False)

    overlap = set(pilot["id"]) & set(pd.read_csv(LLM_EVAL_IDS)["id"])
    print(f"Pilot: {len(pilot)} comments, {pilot['is_toxic'].sum()} toxic")
    print(f"In the evaluation subset: {len(overlap)} (must be 0) | few-shot examples: "
          f"{len(set(pilot['id']) & set(FEWSHOT_IDS))} (must be 0)")
    print(f"\nWrote {PILOT_IDS}")


if __name__ == "__main__":
    main()
