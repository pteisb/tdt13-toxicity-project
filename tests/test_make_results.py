"""Tests for src/make_results.py: the jargon-word matcher and the system list.

Run with: uv run pytest
"""

import pytest

import make_results
from config import PREDS_DIR


@pytest.mark.parametrize("text", [
    "kill the process", "Kill it", "the thread DIES here", "dead code", "dead-code", "it killed the build",
    "killing the worker", "execute the query", "executed twice", "the child died", "remove this garbage",
    "an empty dummy file", "junk data", "the dirty bit", "just trash it", "restart the daemon",
    "naked pointers", "json.dump(x)", "core dumped", "open a bug", "fixes two bugs",
])
def test_jargon_words_are_found(text):
    assert make_results.has_jargon(text)


@pytest.mark.parametrize("text", [
    "skill", "killer feature", "killall", "diesel", "deadline", "deadlock", "executor", "junkyard",
    "dumpster", "debug", "buggy", "daemonize", "trashcan", "abort", "crash", "", None,
])
def test_only_whole_words_count(text):
    assert not make_results.has_jargon(text)


def test_system_list_is_complete_and_unique():
    keys = [system.key for system in make_results.SYSTEMS]
    assert len(keys) == len(set(keys)) == 21
    assert all(system.family in make_results.FAMILIES for system in make_results.SYSTEMS)
    for system in make_results.SYSTEMS:
        assert (PREDS_DIR / f"{system.key}_clean.csv").exists(), system.key


def test_learning_rate_text():
    assert make_results.lr_text(2e-5) == "2e-5"
    assert make_results.lr_text(5e-6) == "5e-6"
