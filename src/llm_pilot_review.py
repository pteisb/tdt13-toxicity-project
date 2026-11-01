"""Put the pilot answers of all prompt conditions side by side, to read them before freezing the prompts.

Reads:  data/llm_pilot_ids.csv, data/clean_folds.csv, llm_raw/pilot/<model>_<condition>_v1.jsonl
Writes: results/llm_pilot/<model>_review.csv    one row per comment: text, label, each condition's
                                                 answer (raw text), status and completion tokens

Prints precision, recall and F1 per condition. With 50 comments these are only a rough sanity check.

Run:  uv run python src/llm_pilot_review.py
      uv run python src/llm_pilot_review.py --model kimik26 --prompt-version v1
"""

import argparse
import json

import pandas as pd

from config import CLEAN_FOLDS
from evaluate import classification_scores
from llm_classify import PILOT_PREDS_DIR, PILOT_RAW_DIR
from make_llm_pilot_ids import PILOT_IDS

CONDITIONS = ["A", "B", "C"]


def read_final_answers(path):
    """id -> final record of a raw file."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {record["id"]: record for record in records if record["final"]}


def main():
    parser = argparse.ArgumentParser(description="Side-by-side view of the pilot answers.")
    parser.add_argument("--model", default="gptoss120b", help="short model name in the file names")
    parser.add_argument("--prompt-version", default="v1")
    args = parser.parse_args()

    data = pd.read_csv(CLEAN_FOLDS)
    review = data[data["id"].isin(pd.read_csv(PILOT_IDS)["id"])][["id", "is_toxic", "message"]]
    review = review.rename(columns={"is_toxic": "label"}).sort_values("id").reset_index(drop=True)

    print(f"{'condition':10s} {'precision':>9s} {'recall':>7s} {'F1':>6s}  unparsed  refused  mean tokens")
    for condition in CONDITIONS:
        system = f"{args.model}_{condition}_{args.prompt_version}"
        answers = read_final_answers(PILOT_RAW_DIR / f"{system}.jsonl")
        finals = [answers[comment_id] for comment_id in review["id"]]
        review[f"{condition}_pred"] = [0 if r["label"] is None else r["label"] for r in finals]
        review[f"{condition}_status"] = [r["status"] for r in finals]
        review[f"{condition}_answer"] = [r["content"] for r in finals]
        review[f"{condition}_tokens"] = [r["completion_tokens"] for r in finals]

        scores = classification_scores(review["label"], review[f"{condition}_pred"])
        statuses = review[f"{condition}_status"]
        print(f"{condition:10s} {scores['precision']:9.3f} {scores['recall']:7.3f} {scores['f1']:6.3f}  "
              f"{(statuses == 'unparsed').sum():8d} {(statuses == 'refused').sum():8d} "
              f"{review[f'{condition}_tokens'].mean():12.0f}")

    PILOT_PREDS_DIR.mkdir(parents=True, exist_ok=True)
    path = PILOT_PREDS_DIR / f"{args.model}_review.csv"
    review.to_csv(path, index=False)
    print(f"\nWrote {path} ({len(review)} comments)")


if __name__ == "__main__":
    main()
