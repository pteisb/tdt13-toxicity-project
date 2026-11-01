"""Tests for src/llm_classify.py: answer parsing, prompts, resuming and the predictions file.
No network and no token: the API call is replaced by a fake.

Run with: uv run pytest
"""

import json

import pandas as pd
import pytest

import evaluate
import llm_classify
from config import CLEAN_FOLDS, PROMPTS_DIR
from llm_client import LLMError


@pytest.mark.parametrize("content, expected", [
    ("NON-TOXIC", 0),
    ("non-toxic.", 0),
    ("Non toxic", 0),
    ("NONTOXIC", 0),
    ("NON‑TOXIC", 0),              # non-breaking hyphen, common in model output
    ("NON–TOXIC", 0),              # en dash
    ("**NON-TOXIC**", 0),
    ("This is not toxic.", 0),
    ("TOXIC", 1),
    ("Toxic", 1),
    ("**TOXIC**", 1),
    ("toxic.\n", 1),
    ("Answer: TOXIC", 1),
    ("", None),
    ("I cannot help", None),
    ("The author seems intoxicated", None),
    ("TOXIC or NON-TOXIC", None),       # both words: ambiguous
])
def test_parse_answer(content, expected):
    assert llm_classify.parse_answer(content) == expected


@pytest.mark.parametrize("content, finish_reason, label, status", [
    ("NON-TOXIC", "stop", 0, "parsed"),
    ("**TOXIC**", "stop", 1, "parsed"),
    ("I cannot help", "stop", None, "refused"),
    ("I can’t help with that.", "stop", None, "refused"),
    ("I'm sorry, but I can't assist with that.", "stop", None, "refused"),
    ("Sorry, I cannot classify this as TOXIC.", "stop", None, "refused"),
    ("", "content_filter", None, "refused"),
    ("", "length", None, "unparsed"),
    ("Maybe?", "stop", None, "unparsed"),
])
def test_read_answer(content, finish_reason, label, status):
    assert llm_classify.read_answer(content, finish_reason) == (label, status)


def test_system_name():
    assert llm_classify.system_name("openai/gpt-oss-120b", "B", "v1") == "gptoss120b_B_v1"
    assert llm_classify.system_name("moonshotai/Kimi-K2.6", "C", "v1") == "kimik26_C_v1"


def test_condition_c_prompt_contains_rubric_and_examples():
    prompt = llm_classify.system_prompt("C", "v1")
    assert llm_classify.system_prompt("B", "v1") in prompt
    assert (PROMPTS_DIR / "fewshot_v1.txt").read_text(encoding="utf-8").strip() in prompt
    assert "{rubric}" not in prompt and "{examples}" not in prompt
    assert prompt.count("<comment>") == 9    # the 8 examples and the sentence that explains the tags


def test_prompt_a_and_b_differ_only_in_the_definition():
    a, b = llm_classify.system_prompt("A", "v1"), llm_classify.system_prompt("B", "v1")
    assert a.split("\n\n")[0] == b.split("\n\n")[0]
    assert a.split("\n\n")[-2:] == b.split("\n\n")[-2:]


def test_user_message_contains_only_the_tagged_comment():
    messages = llm_classify.build_messages("the prompt", "fix this")
    assert messages == [{"role": "system", "content": "the prompt"},
                        {"role": "user", "content": "<comment>fix this</comment>"}]


SETTINGS = {"model": "m", "prompt": "p", "prompt_sha256": "abc", "params": {"temperature": 0}}


def fake_chat(answers):
    """A replacement for chat() that gives the listed answers, one per call."""
    queue = list(answers)

    def chat(messages, model, **params):
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return {"content": answer, "reasoning_content": "because", "finish_reason": "stop",
                "usage": {"completion_tokens": 10}, "latency": 0.5, "model": model, "attempts": 1}

    return chat


def classify(monkeypatch, answers):
    monkeypatch.setattr(llm_classify, "chat", fake_chat(answers))
    records = []
    status = llm_classify.classify_comment(7, "text", SETTINGS, records.append)
    return status, records


def test_good_answer_is_final_without_retry(monkeypatch):
    status, records = classify(monkeypatch, ["TOXIC"])
    assert status == "parsed"
    assert len(records) == 1
    assert records[0]["final"] and records[0]["label"] == 1 and records[0]["id"] == 7
    assert records[0]["reasoning_chars"] == len("because")


def test_unparsed_answer_is_retried_once_and_can_recover(monkeypatch):
    status, records = classify(monkeypatch, ["hmm", "NON-TOXIC"])
    assert status == "parsed"
    assert [r["final"] for r in records] == [False, True]
    assert records[1]["label"] == 0


def test_two_unparsed_answers_end_as_unparsed(monkeypatch):
    status, records = classify(monkeypatch, ["hmm", "hmm"])
    assert status == "unparsed"
    assert len(records) == 2 and records[-1]["final"] and records[-1]["label"] is None


def test_refusal_is_retried_and_reported(monkeypatch):
    status, records = classify(monkeypatch, ["I cannot help", "I cannot help"])
    assert status == "refused"
    assert len(records) == 2 and records[-1]["final"]


def test_request_error_leaves_the_id_without_a_final_answer(monkeypatch):
    status, records = classify(monkeypatch, [LLMError("gave up")])
    assert status == "error"
    assert not records[0]["final"]
    assert llm_classify.final_records(records) == {}


def test_raw_file_is_resumable_and_keeps_only_final_answers(tmp_path):
    raw = tmp_path / "raw.jsonl"
    lines = [{**SETTINGS, "id": 1, "final": True, "status": "parsed", "label": 1},
             {**SETTINGS, "id": 2, "final": False, "status": "error"}]
    raw.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    assert set(llm_classify.final_records(llm_classify.load_records(raw, SETTINGS))) == {1}


def test_damaged_last_line_is_skipped(tmp_path, capsys):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({**SETTINGS, "id": 1, "final": True}) + '\n{"id": 2, "fin', encoding="utf-8")
    assert len(llm_classify.load_records(raw, SETTINGS)) == 1
    assert "damaged" in capsys.readouterr().out


def test_refuses_to_mix_a_changed_prompt_into_a_raw_file(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({**SETTINGS, "id": 1, "final": True}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="different prompt"):
        llm_classify.load_records(raw, {**SETTINGS, "prompt_sha256": "changed"})


def test_predictions_table_passes_the_evaluation_checks():
    folds = pd.read_csv(CLEAN_FOLDS)
    comments = folds.head(4)
    ids = comments["id"].tolist()
    finals = {ids[0]: {"label": 1}, ids[1]: {"label": 0}, ids[2]: {"label": None}}    # ids[3]: no answer yet
    table = llm_classify.predictions_table(comments, finals, "gptoss120b_B_v1")
    assert table["id"].tolist() == ids[:3]
    assert table["pred"].tolist() == [1, 0, 0]      # unparsed counts as non-toxic
    assert table["prob_toxic"].isna().all()
    evaluate.validate_predictions(table, folds)
