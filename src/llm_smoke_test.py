"""Smoke test for the NTNU LLM API before any real LLM runs.

Checks:
  1. Which model ids work (gpt-oss-120b with and without the "openai/" prefix, Kimi-K2.6).
  2. Caching and temperature, on gpt-oss: the same request is sent 3 times
     (a) with the proxy's cache allowed, (b) without cache and default settings,
     (c) without cache and temperature=0. Identical reasoning across the 3 answers means the
     output did not vary. (a) shows the cache; (c) shows whether temperature=0 is deterministic.
  3. reasoning_effort, on gpt-oss: 10 toxic comments (fold 0, random, seed 42) at temperature=0
     without cache, with reasoning_effort unset, "low", "medium" and "high". The setting is honored
     if the number of completion tokens changes.
  4. Refusals: the same 10 comments with the draft rubric prompt, for each working model;
     counts TOXIC / NON-TOXIC / unparseable answers.

Reads:  data/clean_folds.csv, NTNU_API_TOKEN from .env (never printed)
Writes: llm_raw/smoke_test.jsonl            every response (no comment text, only ids)
        results/llm_smoke_test.txt          the printed summary

Run (NTNU network or VPN):  uv run python src/llm_smoke_test.py
"""

import json
import statistics
from datetime import datetime

import pandas as pd

from config import CLEAN_FOLDS, LLM_RAW_DIR, RESULTS_DIR, SEED
from llm_client import LLMError, chat

GPT_OSS = "openai/gpt-oss-120b"
KIMI = "moonshotai/Kimi-K2.6"
MODEL_IDS_TO_TRY = ["gpt-oss-120b", GPT_OSS, KIMI]
REPEATS = 3
RAW_FILE = LLM_RAW_DIR / "smoke_test.jsonl"
SUMMARY_FILE = RESULTS_DIR / "llm_smoke_test.txt"

# Draft rubric prompt (condition B draft). The final prompt files are written in a later step.
DRAFT_RUBRIC = (
    "You classify code review comments from open-source software projects. A comment is TOXIC if it "
    "contains any of: (1) profanity or swearing, even when used positively or jokingly; (2) abbreviations "
    "of swear words such as \"wtf\" or \"stfu\"; (3) insults or demeaning remarks about another person or "
    "about their work or code; (4) attacks on someone's identity (race, religion, nationality, gender, "
    "sexual orientation); (5) aggressive or threatening language; (6) references to sexual activity; "
    "(7) flirting. Demeaning words that writers use about themselves or their own work (\"stupid me\") are "
    "NOT toxic. Technical terms like \"kill the process\" or \"dead code\" are NOT toxic. Everything else is "
    "NON-TOXIC. Answer with exactly one word: TOXIC or NON-TOXIC."
)
# A made-up borderline comment for the caching/temperature test (not from the dataset)
SETTINGS_COMMENT = "This is a stupid hack. Please do it properly this time."
CACHE_VARIANTS = {
    "cache allowed, default": {"use_cache": True},
    "no cache, default": {},
    "no cache, temperature=0": {"temperature": 0},
}
EFFORT_VARIANTS = {
    "effort not set": {},
    "effort low": {"reasoning_effort": "low"},
    "effort medium": {"reasoning_effort": "medium"},
    "effort high": {"reasoning_effort": "high"},
}

summary_lines = []


def report(line=""):
    print(line, flush=True)
    summary_lines.append(line)


def rubric_messages(comment):
    return [{"role": "system", "content": DRAFT_RUBRIC},
            {"role": "user", "content": f"<comment>{comment}</comment>"}]


def parse_answer(content):
    """1 = TOXIC, 0 = NON-TOXIC, None = neither. NON-TOXIC is checked first, because the
    string "NON-TOXIC" contains "TOXIC"."""
    text = content.upper().replace("NON TOXIC", "NON-TOXIC")
    if "NON-TOXIC" in text:
        return 0
    if "TOXIC" in text:
        return 1
    return None


def save_raw(raw_file, test, model, params, result, comment_id=None, error=None):
    record = {"time": datetime.now().isoformat(timespec="seconds"), "test": test, "model": model,
              "params": params, "id": comment_id, "error": error}
    if result:
        record.update({key: result[key] for key in
                       ["content", "reasoning_content", "finish_reason", "usage", "latency"]})
    raw_file.write(json.dumps(record) + "\n")
    raw_file.flush()


def ask(raw_file, test, model, messages, params, comment_id=None):
    """One request, logged to the raw file. Returns the result, or None on an error."""
    try:
        result = chat(messages, model=model, **params)
    except LLMError as error:
        save_raw(raw_file, test, model, params, None, comment_id, error=str(error))
        return None
    save_raw(raw_file, test, model, params, result, comment_id)
    return result


def test_model_ids(raw_file):
    """Returns the model ids that answered. A short timeout and a single retry keep a model
    that is down from blocking the whole test."""
    report("1. Model ids (60 s timeout, 1 retry)")
    working = set()
    for model in MODEL_IDS_TO_TRY:
        try:
            result = chat([{"role": "user", "content": "Reply with the single word: ok"}], model=model,
                          timeout=60, max_retries=1)
            save_raw(raw_file, "model_id", model, {}, result)
            report(f"   {model:24s} works   (answer {result['content']!r}, {result['latency']:.1f} s)")
            working.add(model)
        except LLMError as error:
            save_raw(raw_file, "model_id", model, {}, None, error=str(error))
            report(f"   {model:24s} FAILS   ({str(error)[:70]})")
    return working


def test_cache_and_temperature(raw_file):
    report(f"\n2. Caching and temperature on {GPT_OSS} ({REPEATS} identical requests each)")
    report(f"   {'variant':26s} {'compl. tokens':20s} {'latency (s)':20s} {'distinct reasoning':>18s}")
    for name, params in CACHE_VARIANTS.items():
        attempts = [ask(raw_file, "cache_temperature", GPT_OSS, rubric_messages(SETTINGS_COMMENT), params)
                    for _ in range(REPEATS)]
        results = [r for r in attempts if r is not None]
        if len(results) < REPEATS:
            report(f"   {name:26s} ERROR (see {RAW_FILE.name})")
            continue
        tokens = str([r["usage"].get("completion_tokens") for r in results])
        latencies = str([round(r["latency"], 1) for r in results])
        distinct = len({r["reasoning_content"] for r in results})
        report(f"   {name:26s} {tokens:20s} {latencies:20s} {distinct:>13d} of {REPEATS}")


def test_reasoning_effort(raw_file, comments):
    report(f"\n3. reasoning_effort on {GPT_OSS} ({len(comments)} toxic comments, temperature=0, no cache)")
    for name, effort in EFFORT_VARIANTS.items():
        results = [ask(raw_file, "reasoning_effort", GPT_OSS, rubric_messages(row["message"]),
                       {"temperature": 0, **effort}, int(row["id"]))
                   for _, row in comments.iterrows()]
        results = [r for r in results if r is not None]
        if not results:
            report(f"   {name:16s} all requests failed")
            continue
        tokens = [r["usage"].get("completion_tokens", 0) for r in results]
        labels = [parse_answer(r["content"]) for r in results]
        report(f"   {name:16s} completion tokens mean {statistics.mean(tokens):5.0f} "
               f"(min {min(tokens)}, max {max(tokens)}), latency {statistics.mean(r['latency'] for r in results):4.1f} s, "
               f"answered TOXIC {labels.count(1)} of {len(results)}")


def test_refusals(raw_file, comments, working_models):
    report(f"\n4. Refusals: {len(comments)} toxic comments with the draft rubric prompt (temperature=0, no cache)")
    report(f"   comment ids: {', '.join(str(i) for i in comments['id'])}")
    for model in [GPT_OSS, KIMI]:
        if model not in working_models:
            report(f"   {model:24s} skipped (did not answer in test 1)")
            continue
        counts = {"TOXIC": 0, "NON-TOXIC": 0, "unparseable": 0, "error": 0}
        latencies, unparseable = [], []
        for _, row in comments.iterrows():
            result = ask(raw_file, "refusal", model, rubric_messages(row["message"]),
                         {"temperature": 0}, int(row["id"]))
            if result is None:
                counts["error"] += 1
                continue
            latencies.append(result["latency"])
            label = parse_answer(result["content"])
            if label is None:
                counts["unparseable"] += 1
                unparseable.append(f"id {row['id']}: {result['content'].strip()[:80]!r} "
                                   f"(finish_reason={result['finish_reason']})")
            else:
                counts["TOXIC" if label == 1 else "NON-TOXIC"] += 1
        mean_latency = statistics.mean(latencies) if latencies else float("nan")
        report(f"   {model:24s} " + ", ".join(f"{k} {v}" for k, v in counts.items())
               + f"  (mean latency {mean_latency:.1f} s)")
        for line in unparseable:
            report(f"      {line}")


def main():
    data = pd.read_csv(CLEAN_FOLDS)
    toxic_fold0 = data[(data["fold"] == 0) & (data["is_toxic"] == 1)]
    comments = toxic_fold0.sample(10, random_state=SEED)

    LLM_RAW_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    report(f"LLM API smoke test, {datetime.now():%Y-%m-%d %H:%M}\n")
    with open(RAW_FILE, "a", encoding="utf-8") as raw_file:
        working_models = test_model_ids(raw_file)
        if GPT_OSS in working_models:
            test_cache_and_temperature(raw_file)
            test_reasoning_effort(raw_file, comments)
        test_refusals(raw_file, comments, working_models)

    SUMMARY_FILE.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RAW_FILE}\n      {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
