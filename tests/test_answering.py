"""Unit tests for the answer pipeline, with no Streamlit involved.

This is the gap this module was created to close: the same logic inside
ui/app.py could only be reached through AppTest, and shipped once with a
NameError that the whole unit suite passed straight through.
"""
from core.models import Chunk, SearchResult
from generation.answering import Settings, answer
from generation.prompts import NO_ANSWER
from retrieval.search import SearchOutcome, SearchTrace
from tests.fakes import ScriptedLLM


class StubJob:
    """The parts of ui.jobs.Job the pipeline touches."""

    def __init__(self):
        self.status = ""
        self.chunks = []
        self.citations = []
        self.trace = {}

    def append(self, piece):
        self.chunks.append(piece)

    @property
    def text(self):
        return "".join(self.chunks)


class StubSearch:
    def __init__(self, outcome):
        self._outcome = outcome
        self.questions = []

    def find(self, question, **kwargs):
        self.questions.append(question)
        return self._outcome


class StubAnswerer:
    def __init__(self, pieces):
        self._pieces = pieces
        self.histories = []

    def stream(self, question, results, **kwargs):
        self.histories.append(kwargs.get("history"))
        yield from self._pieces


def _outcome(*texts, refused=False, related=()):
    return SearchOutcome(
        results=[SearchResult(chunk=Chunk(doc_id="d", filename="f.pdf",
                                          text=t, chunk_index=i, page=1),
                              score=0.9)
                 for i, t in enumerate(texts)],
        related=list(related), refused=refused,
        trace=SearchTrace(kept=len(texts)),
    )


def _settings(**over):
    base = dict(model="m", temperature=0.0, floor=0.5, candidates=25,
                use_reranker=True, follow_up=False,
                multi_query=False, multi_hop=False, self_correct=False)
    base.update(over)
    return Settings(**base)


def _run(job, question="q", *, history=None, settings=None, search=None,
         answerer=None, llm=None):
    answer(job, question, history=history or [], doc_ids=None,
           settings=settings or _settings(),
           search=search or StubSearch(_outcome("the passage")),
           answerer=answerer or StubAnswerer(["ok"]),
           agentic=None, llm=llm or ScriptedLLM(), publish=None)


def test_a_plain_answer_streams_and_carries_citations():
    job = StubJob()
    _run(job, answerer=StubAnswerer(["Three ", "months."]))

    assert job.text == "Three months."
    assert [c["label"] for c in job.citations] == ["f.pdf, p. 1"]
    assert job.trace["kept"] == 1


def test_nothing_retrieved_refuses_without_calling_the_model():
    """The refusal must stay free."""
    answerer = StubAnswerer(["should not be reached"])
    job = StubJob()

    _run(job, search=StubSearch(_outcome(refused=True)), answerer=answerer)

    assert "could not find" in job.text
    assert job.citations == []
    assert answerer.histories == []          # never streamed


def test_a_declined_answer_drops_its_sources():
    """Retrieval cleared the floor, the model found no answer in it. The
    passages did not produce this, so they must not be cited."""
    job = StubJob()
    _run(job, answerer=StubAnswerer([NO_ANSWER]))

    assert job.citations == []
    assert NO_ANSWER not in job.text
    assert "could not find" in job.text
    assert job.trace["model_declined"] is True
    assert job.trace["related"] == ["f.pdf, p. 1"]


def test_the_sentinel_never_reaches_the_job_even_partially():
    """It is held back while the first tokens decide, so it cannot flash
    on screen."""
    job = StubJob()
    _run(job, answerer=StubAnswerer(list(NO_ANSWER)))   # one char at a time

    assert NO_ANSWER not in job.text


def test_a_short_answer_is_not_swallowed_by_the_decision_buffer():
    """Shorter than the sentinel, so the stream ends mid-decision."""
    job = StubJob()
    _run(job, answerer=StubAnswerer(["Yes."]))

    assert job.text == "Yes."


def test_history_reaches_the_answerer_when_context_is_remembered():
    job = StubJob()
    history = [{"role": "user", "content": "earlier"}]
    answerer = StubAnswerer(["ok"])

    _run(job, history=history, answerer=answerer,
         settings=_settings(follow_up=True))

    assert answerer.histories[0] == history


def test_without_remember_context_the_answer_sees_no_chat():
    """The checkbox governs every use of the conversation, not only the
    follow-up rewrite: off means each question stands alone."""
    job = StubJob()
    answerer = StubAnswerer(["ok"])
    llm = ScriptedLLM()

    _run(job, history=[{"role": "user", "content": "Poland is in the EU."}],
         answerer=answerer, llm=llm, settings=_settings(follow_up=False))

    assert answerer.histories[0] == []
    assert llm.calls == []                   # no follow-up rewrite either


def test_a_follow_up_is_resolved_before_searching():
    search = StubSearch(_outcome("x"))
    job = StubJob()

    _run(job, "And the base year?",
         history=[{"role": "user", "content": "Norway's target?"},
                  {"role": "assistant", "content": "70-75%."}],
         settings=_settings(follow_up=True), search=search,
         answerer=StubAnswerer(["1990."]),
         llm=ScriptedLLM(["What is the base year for Norway's target?"]))

    assert search.questions == ["What is the base year for Norway's target?"]
    assert job.trace["resolved_question"] == \
        "What is the base year for Norway's target?"


def test_the_question_the_user_typed_is_what_gets_answered():
    """Only the search query is rewritten; the model answers what was asked."""
    job = StubJob()

    _run(job, "And the base year?",
         history=[{"role": "user", "content": "Norway's target?"},
                  {"role": "assistant", "content": "70-75%."}],
         settings=_settings(follow_up=True), answerer=StubAnswerer(["1990."]),
         llm=ScriptedLLM(["What is the base year for Norway's target?"]))

    assert job.text == "1990."


def test_status_is_reported_as_the_work_proceeds():
    job = StubJob()
    _run(job)

    assert job.status == "Writing the answer"
