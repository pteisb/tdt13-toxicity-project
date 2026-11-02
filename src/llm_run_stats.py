"""Summarize the full LLM runs: answer counts, retries, speed and token use per system.

Reads:  llm_raw/<system>.jsonl for the systems listed in SYSTEMS
Writes: results/llm_run_stats.csv    one row per system

Run:  uv run python src/llm_run_stats.py
"""

import json
from datetime import datetime

import pandas as pd

from config import LLM_RAW_DIR, RESULTS_DIR

SYSTEMS = [f"{model}_{condition}_v1" for model in ("gptoss120b", "kimik26") for condition in "ABC"]


def run_stats(system):
    records = [json.loads(line) for line in (LLM_RAW_DIR / f"{system}.jsonl").read_text(encoding="utf-8").splitlines()]
    answers = [r for r in records if r["status"] != "error"]
    finals = [r for r in records if r["final"]]
    times = [datetime.fromisoformat(r["time"]) for r in records]
    latency = pd.Series([r["latency"] for r in answers])
    return {
        "system": system,
        "ids": len(finals),
        "parsed": sum(r["status"] == "parsed" for r in finals),
        "unparsed": sum(r["status"] == "unparsed" for r in finals),
        "refused": sum(r["status"] == "refused" for r in finals),
        "retries": sum(r["attempt"] > 1 for r in answers),
        "failed_requests": len(records) - len(answers),
        "http_retries": sum(r["http_attempts"] - 1 for r in answers),
        "latency_mean_s": latency.mean(),
        "latency_median_s": latency.median(),
        "latency_max_s": latency.max(),
        "completion_tokens_mean": pd.Series([r["completion_tokens"] for r in answers]).mean(),
        "reasoning_chars_mean": pd.Series([r["reasoning_chars"] for r in answers]).mean(),
        "first_request": min(times).isoformat(),
        "wall_clock_h": (max(times) - min(times)).total_seconds() / 3600,
    }


def main():
    table = pd.DataFrame([run_stats(system) for system in SYSTEMS])
    path = RESULTS_DIR / "llm_run_stats.csv"
    table.round(3).to_csv(path, index=False)
    print(table.round(2).to_string(index=False))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
