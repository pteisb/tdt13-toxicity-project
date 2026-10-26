"""Small client for NTNU's LLM API (OpenAI-compatible, models served on the IDUN cluster).

Reads:  NTNU_API_TOKEN from .env (never printed or logged)
Writes: nothing

Usage from other scripts:
    from llm_client import chat
    result = chat([{"role": "user", "content": "Hello"}], model="openai/gpt-oss-120b")
    result["content"], result["reasoning_content"], result["usage"], result["latency"]

Extra request fields (temperature, reasoning_effort, seed, ...) are passed as keyword arguments.

The API's proxy caches responses: an identical request returns the stored answer instead of
asking the model again. chat() therefore bypasses the cache by default (use_cache=False), so every
answer is really computed and latencies are real.
Do not set a small max_tokens: reasoning models spend output tokens on reasoning first, and the
final answer comes back empty if the limit is reached.

The API is only reachable from the NTNU network or VPN.
Quick check:  uv run python src/llm_client.py
"""

import os
import random
import time

import requests
from dotenv import load_dotenv

from config import ROOT

API_URL = "https://llm.hpc.ntnu.no/v1/chat/completions"
RETRY_STATUSES = {429, 500, 502, 503, 504}
PLACEHOLDER_TOKEN = "your_token_here"


class LLMError(Exception):
    """The API returned an error that retrying will not fix, or all retries failed."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def load_token():
    """Read the API token from .env. Exits with a clear message if it is missing."""
    load_dotenv(ROOT / ".env")
    token = os.environ.get("NTNU_API_TOKEN", "").strip()
    if not token or token == PLACEHOLDER_TOKEN:
        raise SystemExit("NTNU_API_TOKEN is not set in .env (see .env.example).")
    return token


def backoff_seconds(attempt, response=None):
    """Wait time before the next retry: the server's Retry-After if given, otherwise
    exponential backoff (2, 4, 8, ... seconds) with a little random jitter."""
    if response is not None and response.headers.get("Retry-After", "").isdigit():
        return int(response.headers["Retry-After"])
    return 2 ** (attempt + 1) + random.uniform(0, 1)


def chat(messages, model, timeout=300, max_retries=5, use_cache=False, **params):
    """Send one chat request and return a dict with:
    content, reasoning_content, finish_reason, usage, latency (seconds), model, attempts.

    Retries with backoff on HTTP 429/5xx, timeouts and connection errors. Other errors
    (for example 400 for an unknown model) raise LLMError immediately.
    """
    headers = {"Authorization": f"Bearer {load_token()}"}
    payload = {"model": model, "messages": messages, **params}
    if not use_cache:
        payload["cache"] = {"no-cache": True}    # tells the API's proxy not to answer from its cache

    problem = "no attempt made"
    for attempt in range(max_retries + 1):
        start = time.time()
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as error:
            problem, response = f"{type(error).__name__}", None
        else:
            if response.ok:
                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]
                return {
                    "content": message.get("content") or "",
                    "reasoning_content": message.get("reasoning_content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "usage": data.get("usage") or {},
                    "latency": time.time() - start,
                    "model": data.get("model"),
                    "attempts": attempt + 1,
                }
            if response.status_code not in RETRY_STATUSES:
                raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}",
                               status=response.status_code)
            problem = f"HTTP {response.status_code}"

        if attempt < max_retries:
            wait = backoff_seconds(attempt, response)
            print(f"  {problem}; retrying in {wait:.0f} s (attempt {attempt + 1} of {max_retries})")
            time.sleep(wait)

    raise LLMError(f"gave up after {max_retries} retries ({problem})")


def reasoning_tokens(usage):
    """Reasoning-token count if the API reports one, else None. (The NTNU API currently only
    reports completion_tokens, which includes the reasoning.)"""
    details = usage.get("completion_tokens_details") or {}
    return details.get("reasoning_tokens")


if __name__ == "__main__":
    result = chat([{"role": "user", "content": "Reply with the single word: ok"}],
                  model="openai/gpt-oss-120b")
    print(f"content={result['content']!r}  latency={result['latency']:.1f} s  "
          f"completion_tokens={result['usage'].get('completion_tokens')}")
