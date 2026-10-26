"""Tests for the retry logic in src/llm_client.py, with fake responses (no network, no token).

Run with: uv run pytest
"""

import pytest
import requests

import llm_client


class FakeResponse:
    def __init__(self, status, body=None, headers=None):
        self.status_code = status
        self.ok = status == 200
        self.headers = headers or {}
        self._body = body or {}
        self.text = str(self._body)

    def json(self):
        return self._body


OK_BODY = {
    "model": "openai/gpt-oss-120b",
    "choices": [{"finish_reason": "stop",
                 "message": {"content": "TOXIC", "reasoning_content": "thinking..."}}],
    "usage": {"completion_tokens": 12},
}


@pytest.fixture
def fake_api(monkeypatch):
    """Replace the token, sleeping and HTTP calls. Returns a function that sets the sequence of
    responses (or exceptions) the fake API gives, and the list of calls made."""
    calls = []
    monkeypatch.setattr(llm_client, "load_token", lambda: "fake-token")
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: None)

    def set_responses(*responses):
        queue = list(responses)

        def fake_post(url, headers, json, timeout):
            calls.append(json)
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(llm_client.requests, "post", fake_post)
        return calls

    return set_responses


def test_success_returns_content_and_reasoning(fake_api):
    calls = fake_api(FakeResponse(200, OK_BODY))
    result = llm_client.chat([{"role": "user", "content": "hi"}], model="m", temperature=0)
    assert result["content"] == "TOXIC"
    assert result["reasoning_content"] == "thinking..."
    assert result["usage"]["completion_tokens"] == 12
    assert result["attempts"] == 1
    assert calls[0]["temperature"] == 0          # extra settings are passed through
    assert calls[0]["cache"] == {"no-cache": True}   # the proxy's cache is bypassed by default


def test_cache_can_be_allowed(fake_api):
    calls = fake_api(FakeResponse(200, OK_BODY))
    llm_client.chat([], model="m", use_cache=True)
    assert "cache" not in calls[0]


def test_retries_on_429_and_server_errors(fake_api):
    fake_api(FakeResponse(429), FakeResponse(503), requests.Timeout(), FakeResponse(200, OK_BODY))
    result = llm_client.chat([], model="m")
    assert result["content"] == "TOXIC"
    assert result["attempts"] == 4


def test_client_error_is_not_retried(fake_api):
    calls = fake_api(FakeResponse(400, {"error": "Invalid model name"}), FakeResponse(200, OK_BODY))
    with pytest.raises(llm_client.LLMError) as error:
        llm_client.chat([], model="unknown")
    assert error.value.status == 400
    assert len(calls) == 1


def test_gives_up_after_max_retries(fake_api):
    calls = fake_api(*[FakeResponse(503)] * 3)
    with pytest.raises(llm_client.LLMError, match="gave up"):
        llm_client.chat([], model="m", max_retries=2)
    assert len(calls) == 3


def test_retry_after_header_is_respected():
    assert llm_client.backoff_seconds(0, FakeResponse(429, headers={"Retry-After": "7"})) == 7
    assert 2 <= llm_client.backoff_seconds(0) < 3
    assert 8 <= llm_client.backoff_seconds(2) < 9
