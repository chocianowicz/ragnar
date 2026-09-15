import pytest

from generation import followup
from generation.answerer import recent_history, HISTORY_TURNS


class ScriptedLLM:
    def __init__(self, reply=""):
        self.reply = reply
        self.calls = []

    def generate(self, system, user, **kwargs):
        self.calls.append(user)
        return self.reply


class FailingLLM:
    def generate(self, *a, **k):
        raise RuntimeError("ollama is down")


HISTORY = [
    {"role": "user", "content": "What is Norway's 2035 target?"},
    {"role": "assistant", "content": "At least 70-75% below 1990 levels.",
     "citations": [{"label": "x"}], "trace": {"kept": 5}},
]


def test_a_follow_up_is_resolved_against_the_conversation():
    """The failure this exists for: asked after a question about Norway,
    'the base year for that' retrieved base years for four other countries
    because nothing in the query said Norway."""
    llm = ScriptedLLM("What is the base year for Norway's 2035 target?")

    resolved, changed = followup.resolve(llm, "And the base year for that?",
                                         HISTORY)

    assert changed
    assert "Norway" in resolved


def test_the_resolver_is_shown_the_conversation():
    llm = ScriptedLLM("What is the base year for Norway's 2035 target?")

    followup.resolve(llm, "And the base year for that?", HISTORY)

    assert "Norway's 2035 target" in llm.calls[0]


def test_no_history_means_no_model_call():
    """The first question of a chat has nothing to resolve against."""
    llm = ScriptedLLM("something")

    resolved, changed = followup.resolve(llm, "What is the notice period?", [])

    assert (resolved, changed) == ("What is the notice period?", False)
    assert llm.calls == []


def test_an_unchanged_question_is_not_reported_as_resolved():
    question = "What is Norway's 2035 target?"
    llm = ScriptedLLM(question)

    resolved, changed = followup.resolve(llm, question, HISTORY)

    assert (resolved, changed) == (question, False)


def test_a_truncated_resolution_is_discarded():
    """A resolution should expand the question, not replace it."""
    llm = ScriptedLLM("Norway")

    resolved, changed = followup.resolve(
        llm, "And what exactly is the base year used for that target?",
        HISTORY)

    assert not changed
    assert resolved.startswith("And what exactly")


def test_a_failing_model_costs_context_not_the_answer():
    resolved, changed = followup.resolve(
        FailingLLM(), "And the base year?", HISTORY)

    assert (resolved, changed) == ("And the base year?", False)


def test_resolver_sees_only_recent_turns():
    long_history = [
        {"role": "user", "content": f"question {i}"} for i in range(20)
    ]
    llm = ScriptedLLM("resolved question that is long enough")

    followup.resolve(llm, "and that?", long_history)

    assert "question 19" in llm.calls[0]
    assert "question 0" not in llm.calls[0]


# --- the window sent to the answering model ----------------------------

def test_history_is_reduced_to_role_and_content():
    """Citations and traces are for the UI; sending them would be noise."""
    [_, assistant] = recent_history(HISTORY)

    assert set(assistant) == {"role", "content"}


def test_history_is_windowed():
    long_history = [{"role": "user", "content": f"q{i}"} for i in range(40)]

    assert len(recent_history(long_history)) == HISTORY_TURNS


def test_empty_history_is_empty():
    assert recent_history(None) == []
    assert recent_history([]) == []


def test_messages_without_content_are_dropped():
    assert recent_history([{"role": "user", "content": ""}]) == []
