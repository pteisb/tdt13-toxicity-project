"""Classify comments as toxic or not with a prompted LLM through NTNU's API.

For each id: build the messages from the prompt files, ask the model, and parse the one-word answer.
An answer that cannot be parsed (or a refusal) is asked once more. Every response is appended to the
raw file straight away, and ids that already have a final answer there are skipped, so running the same
command again after a crash continues where it stopped. In the predictions file, unparsed answers and
refusals count as non-toxic (0); their numbers are printed at the end.

Reads:  data/clean_folds.csv, the --ids CSV (column "id"), prompts/, NTNU_API_TOKEN from .env
Writes: llm_raw/<system>.jsonl       every response (the comment id, not its text; the reasoning may quote it)
        preds/<system>_clean.csv     one row per id with a final answer (prob_toxic empty)
With --pilot the same files go to llm_raw/pilot/ and results/llm_pilot/ instead, so that pilot answers
never mix with the real runs. <system> looks like gptoss120b_B_v1.

Run (NTNU network or VPN):
  uv run python src/llm_classify.py --model openai/gpt-oss-120b --condition B --prompt-version v1 \
      --ids data/llm_eval_ids.csv
"""

import argparse
import hashlib
import json
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

from config import CLEAN_FOLDS, LLM_EVAL_IDS, LLM_RAW_DIR, NON_TOXIC, PRED_COLUMNS, PREDS_DIR, PROMPTS_DIR, RESULTS_DIR, SEED, TOXIC
from llm_client import LLMError, chat, reasoning_tokens

MAX_ATTEMPTS = 2                # one retry on an unparsed answer or a refusal
MAX_CONSECUTIVE_ERRORS = 10     # the API is probably down: stop instead of working through every id
PILOT_RAW_DIR = LLM_RAW_DIR / "pilot"
PILOT_PREDS_DIR = RESULTS_DIR / "llm_pilot"

# Models often write hyphens as other Unicode characters, for example "NON‑TOXIC".
DASHES = re.compile("[‐‑‒–—―−]")
NEGATED_TOXIC = re.compile(r"\b(?:NON|NOT)[\s-]*TOXIC\b")
PLAIN_TOXIC = re.compile(r"\bTOXIC\b")
REFUSAL_START = re.compile(
    r"(?:i\s+cannot|i\s+can\s*not|i\s+can't|i\s+won't|i\s+will\s+not|i\s+am\s+(?:unable|not able)"
    r"|i'm\s+(?:unable|not able|sorry)|i\s+am\s+sorry|sorry\b|unable\s+to)")


def parse_answer(content):
    """1 = TOXIC, 0 = NON-TOXIC, None = neither, or both. The negative forms are handled first,
    because "NON-TOXIC" contains "TOXIC"."""
    text = DASHES.sub("-", content).upper()
    has_negative = bool(NEGATED_TOXIC.search(text))
    has_positive = bool(PLAIN_TOXIC.search(NEGATED_TOXIC.sub(" ", text)))
    if has_negative and has_positive:
        return None
    if has_negative:
        return NON_TOXIC
    if has_positive:
        return TOXIC
    return None


def is_refusal(content, finish_reason=None):
    """True if the model declined to answer: the provider's filter stopped it, or the answer starts
    with a typical refusal phrase."""
    if finish_reason == "content_filter":
        return True
    text = content.strip().lstrip("*_ ").lower().replace("’", "'")
    return bool(REFUSAL_START.match(text))


def read_answer(content, finish_reason=None):
    """Returns (label, status) with status "parsed", "refused" or "unparsed"; label is None unless parsed."""
    if is_refusal(content, finish_reason):
        return None, "refused"
    label = parse_answer(content)
    return label, "unparsed" if label is None else "parsed"


def system_name(model, condition, version):
    """openai/gpt-oss-120b, B, v1 -> gptoss120b_B_v1"""
    short = re.sub(r"[^a-z0-9]", "", model.split("/")[-1].lower())
    return f"{short}_{condition}_{version}"


def system_prompt(condition, version, prompts_dir=PROMPTS_DIR):
    """The system message. Condition C fills its template by plain string replacement, so braces in
    the example comments are safe."""
    def read(name):
        return (prompts_dir / f"{name}_{version}.txt").read_text(encoding="utf-8").strip()

    if condition != "C":
        return read(condition)
    return read("C").replace("{rubric}", read("B")).replace("{examples}", read("fewshot"))


def build_messages(prompt, comment):
    return [{"role": "system", "content": prompt},
            {"role": "user", "content": f"<comment>{comment}</comment>"}]


def classify_comment(comment_id, comment, settings, save_record):
    """Ask about one comment, saving every response. Returns the final status, or "error" if the
    request failed (that id then has no final answer and is tried again on the next run)."""
    messages = build_messages(settings["prompt"], comment)
    base = {"id": comment_id, "model": settings["model"], "prompt_sha256": settings["prompt_sha256"],
            "params": settings["params"]}
    status = "error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        record = {**base, "attempt": attempt, "time": datetime.now().isoformat(timespec="seconds")}
        try:
            result = chat(messages, model=settings["model"], **settings["params"])
        except LLMError as error:
            save_record({**record, "status": "error", "final": False, "error": str(error)})
            return "error"
        label, status = read_answer(result["content"], result["finish_reason"])
        final = status == "parsed" or attempt == MAX_ATTEMPTS
        save_record({
            **record, "status": status, "final": final, "label": label,
            "content": result["content"], "reasoning_content": result["reasoning_content"],
            "reasoning_chars": len(result["reasoning_content"]),
            "reasoning_tokens": reasoning_tokens(result["usage"]),
            "completion_tokens": result["usage"].get("completion_tokens"),
            "finish_reason": result["finish_reason"], "latency": round(result["latency"], 3),
            "http_attempts": result["attempts"], "served_by": result["model"],
        })
        if final:
            break
    return status


def load_records(raw_path, settings):
    """All records in an existing raw file. Refuses to continue a file made with another prompt or
    other settings, because that would silently mix two experiments."""
    if not raw_path.exists():
        return []
    records = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"Warning: skipping a damaged line in {raw_path.name} (interrupted write?)")
    for record in records:
        if any(record[key] != settings[key] for key in ("model", "prompt_sha256", "params")):
            sys.exit(f"{raw_path} was made with a different prompt or different settings.\n"
                     "Move it away if it is only a pilot; otherwise create a new prompt version.")
    return records


def final_records(records):
    """The final answer per id."""
    return {record["id"]: record for record in records if record["final"]}


def predictions_table(comments, finals, system):
    """One row per comment with a final answer. Unparsed answers and refusals count as non-toxic."""
    answered = comments[comments["id"].isin(finals)].sort_values("id")
    labels = [finals[comment_id].get("label") for comment_id in answered["id"]]
    return pd.DataFrame({
        "id": answered["id"].to_numpy(),
        "fold": answered["fold"].to_numpy(),
        "label": answered["is_toxic"].to_numpy(),
        "prob_toxic": None,
        "pred": [NON_TOXIC if label is None else label for label in labels],
        "system": system,
        "setting": "clean",
    })[PRED_COLUMNS]


def run_requests(todo, settings, raw_path, workers):
    """Classify the comments in parallel, appending to the raw file. Returns the number of ids that
    ended in an error."""
    lock = threading.Lock()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and raw_path.stat().st_size and not raw_path.read_bytes().endswith(b"\n"):
        with raw_path.open("a", encoding="utf-8") as raw_file:
            raw_file.write("\n")

    with raw_path.open("a", encoding="utf-8") as raw_file, ThreadPoolExecutor(workers) as pool:
        def save_record(record):
            with lock:
                raw_file.write(json.dumps(record) + "\n")
                raw_file.flush()

        futures = [pool.submit(classify_comment, int(row.id), row.message, settings, save_record)
                   for row in todo.itertuples()]
        errors = consecutive_errors = 0
        try:
            for done, future in enumerate(as_completed(futures), 1):
                if future.result() == "error":
                    errors += 1
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0
                if done % 100 == 0 or done == len(futures):
                    print(f"  {done:,} of {len(futures):,} done ({errors} errors)", flush=True)
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"Stopping: {consecutive_errors} requests in a row failed. Run the same "
                          "command again later to continue.")
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
        except KeyboardInterrupt:
            print("Interrupted. Run the same command again to continue.")
            pool.shutdown(wait=False, cancel_futures=True)
    return errors


def print_summary(records, finals, n_requested):
    """Counts of the final answers, and speed and token numbers over all responses."""
    answers = [record for record in records if record["status"] != "error"]
    statuses = [record["status"] for record in finals.values()]
    print(f"\nFinal answers: {len(finals):,} of {n_requested:,} ids | parsed {statuses.count('parsed'):,}, "
          f"unparsed {statuses.count('unparsed'):,}, refused {statuses.count('refused'):,}")
    print(f"Responses: {len(answers):,} (retries {sum(r['attempt'] > 1 for r in answers):,}), "
          f"failed requests {len(records) - len(answers):,}")
    if not answers:
        return
    print(f"Mean latency: {statistics.mean(r['latency'] for r in answers):.2f} s")
    reported = [r["reasoning_tokens"] for r in answers if r["reasoning_tokens"] is not None]
    if reported:
        print(f"Mean reasoning tokens: {statistics.mean(reported):.0f}")
    else:
        print("Mean reasoning tokens: not reported by the API; mean completion tokens (reasoning "
              f"included) {statistics.mean(r['completion_tokens'] or 0 for r in answers):.0f}, "
              f"mean reasoning length {statistics.mean(r['reasoning_chars'] for r in answers):.0f} characters")


def main():
    parser = argparse.ArgumentParser(description="Classify comments with a prompted LLM.")
    parser.add_argument("--model", required=True, help='API model id, e.g. "openai/gpt-oss-120b"')
    parser.add_argument("--condition", required=True, choices=["A", "B", "C"])
    parser.add_argument("--prompt-version", required=True, help='e.g. "v1"')
    parser.add_argument("--ids", required=True, help='CSV with a column "id"')
    parser.add_argument("--workers", type=int, default=4, help="parallel requests")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                        help="not set = the API's default, which is medium")
    parser.add_argument("--pilot", action="store_true",
                        help=f"write to {PILOT_RAW_DIR.name}/ and {PILOT_PREDS_DIR.name}/, not the real locations")
    args = parser.parse_args()

    ids = pd.read_csv(args.ids)["id"]
    if args.pilot and ids.isin(pd.read_csv(LLM_EVAL_IDS)["id"]).any():
        sys.exit("Pilot ids must not overlap the evaluation subset: it is never used to design prompts.")
    data = pd.read_csv(CLEAN_FOLDS)
    comments = data[data["id"].isin(ids)]
    if len(comments) != ids.nunique():
        sys.exit(f"{ids.nunique() - len(comments)} ids in {args.ids} are not in {CLEAN_FOLDS.name}")

    system = system_name(args.model, args.condition, args.prompt_version)
    raw_path = (PILOT_RAW_DIR if args.pilot else LLM_RAW_DIR) / f"{system}.jsonl"
    preds_path = (PILOT_PREDS_DIR if args.pilot else PREDS_DIR) / f"{system}_clean.csv"

    prompt = system_prompt(args.condition, args.prompt_version)
    params = {"temperature": 0, "seed": SEED}
    if args.reasoning_effort:
        params["reasoning_effort"] = args.reasoning_effort
    settings = {"model": args.model, "prompt": prompt, "params": params,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}

    done = final_records(load_records(raw_path, settings))
    todo = comments[~comments["id"].isin(done)]
    print(f"{system}: {len(comments):,} ids, {len(comments) - len(todo):,} already answered, "
          f"{len(todo):,} to do ({args.workers} workers)", flush=True)
    start = time.time()
    if len(todo):
        run_requests(todo, settings, raw_path, args.workers)
    elapsed = time.time() - start

    records = [r for r in load_records(raw_path, settings) if r["id"] in set(comments["id"])]
    finals = final_records(records)
    print_summary(records, finals, len(comments))
    print(f"Time for this run: {elapsed / 60:.1f} min")

    preds_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_table(comments, finals, system).to_csv(preds_path, index=False)
    print(f"\nWrote {raw_path}\n      {preds_path} ({len(finals):,} rows)")
    if len(finals) < len(comments):
        print(f"WARNING: {len(comments) - len(finals):,} ids have no answer yet. Run the same command again.")


if __name__ == "__main__":
    main()
