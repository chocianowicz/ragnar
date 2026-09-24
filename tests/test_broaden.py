"""Answering through a broader subject: the Poland / EU case.

The documents give the EU target and never name Poland. The model may
propose "the EU" and the link, but the link only counts when the chat
quotes it verbatim or the documents state it; these tests pin both halves
of that.
"""
from core.models import Chunk, SearchResult
from generation import broaden
from generation.answering import Settings, answer
from generation.prompts import NO_ANSWER
from retrieval.search import SearchOutcome, SearchTrace
from tests.fakes import FailingLLM, ScriptedLLM

POLAND = "What is Poland's net zero target?"
EU = "What is the EU's net zero target?"
LINK = "Poland is a member state of the EU."
PROPOSAL = f"Broader: {EU}\nLink: {LINK}\nQuote: Poland is a member of the EU"
MEMBERSHIP_CHAT = [
    {"role": "user", "content": "Is Poland in the EU?"},
    {"role": "assistant", "content": "Yes, Poland is a member of the EU."},
]


def _found(*texts, filename="eu.pdf"):
    return SearchOutcome(
        results=[SearchResult(chunk=Chunk(doc_id=filename, filename=filename,
                                          text=t, chunk_index=i, page=i + 1),
                              score=0.8)
                 for i, t in enumerate(texts)],
        trace=SearchTrace(candidates=25, kept=len(texts)),
    )


def _refused():
    return SearchOutcome(refused=True,
                         trace=SearchTrace(candidates=25, kept=0))


class ByQuestionSearch:
    """Answers each query from a table; anything unlisted is refused."""

    def __init__(self, table):
        self._table = table
        self.questions = []

    def find(self, question, **kwargs):
        self.questions.append(question)
        return self._table.get(question, _refused())


class StubAnswerer:
    def __init__(self, direct=("NO_ANSWER_IN_EXCERPTS",),
                 indirect=("The documents do not state it for Poland.",)):
        self._direct, self._indirect = list(direct), list(indirect)
        self.bridges = []

    def stream(self, question, results, **kwargs):
        yield from self._direct

    def stream_indirect(self, question, bridge, results, **kwargs):
        self.bridges.append(bridge)
        yield from self._indirect


class StubJob:
    def __init__(self):
        self.status, self.chunks, self.citations, self.trace = "", [], [], {}

    def append(self, piece):
        self.chunks.append(piece)

    @property
    def text(self):
        return "".join(self.chunks)


def _settings(**over):
    base = dict(model="m", temperature=0.0, floor=0.55, candidates=25,
                use_reranker=True, follow_up=True, multi_query=False,
                multi_hop=False, self_correct=False, broaden=True)
    base.update(over)
    return Settings(**base)


def _run(search, *, history=(), llm=None, answerer=None, settings=None,
         question=POLAND):
    job = StubJob()
    answerer = answerer or StubAnswerer()
    settings = settings or _settings()
    if llm is None:
        # With a chat and "Remember context" on, the follow-up resolver
        # asks first; an empty reply leaves the question as typed.
        resolver = [""] if history and settings.follow_up else []
        llm = ScriptedLLM(resolver + [PROPOSAL])
    answer(job, question, history=list(history), doc_ids=None,
           settings=settings, search=search, answerer=answerer,
           agentic=None, llm=llm, publish=None)
    return job, answerer


def _broaden_calls(llm):
    return [c for c in llm.calls if c[0] == broaden.BROADEN_SYSTEM]


# -- the link, from the chat ---------------------------------------------

def test_a_link_quoted_from_the_chat_answers_through_the_eu():
    search = ByQuestionSearch({EU: _found("EU: climate neutrality by 2050")})

    job, answerer = _run(search, history=MEMBERSHIP_CHAT)

    assert job.text == "The documents do not state it for Poland."
    assert [c["label"] for c in job.citations] == ["eu.pdf, p. 1"]
    indirect = job.trace["indirect"]
    assert indirect["source"] == "conversation"
    assert indirect["speaker"] == "assistant"
    assert indirect["quote"] == "Poland is a member of the EU"
    assert indirect["broader_question"] == EU
    # The chat settled it, so the link was never searched for.
    assert search.questions == [POLAND, EU]


def test_a_quote_the_chat_does_not_contain_is_not_a_link():
    """The model saying the chat established it is not enough."""
    search = ByQuestionSearch({EU: _found("EU: climate neutrality by 2050")})

    job, answerer = _run(search, history=[
        {"role": "user", "content": "Tell me about the EU target."}])

    assert "indirect" not in job.trace
    assert answerer.bridges == []
    assert "could not find" in job.text
    assert "neither this chat nor the documents" in job.trace["broader_attempt"]


def test_quote_matching_ignores_case_and_spacing_but_not_words():
    history = [{"role": "assistant", "content": "Yes -  Poland IS a\nmember"
                                                " of the EU."}]
    assert broaden.quoted_in("poland is a member of the EU", "q",
                             history) == "assistant"
    assert broaden.quoted_in("Poland is a founding member of the EU", "q",
                             history) is None


def test_a_link_in_the_question_itself_counts_as_the_users():
    question = "Poland is in the EU, so what is its net zero target?"
    assert broaden.quoted_in("Poland is in the EU", question, []) == "user"


def test_words_inside_a_question_prove_nothing():
    """Asking whether Poland is in the EU does not establish that it is."""
    history = [{"role": "user", "content": "Is Poland in the EU?"}]
    assert broaden.quoted_in("Is Poland in the EU?", "q", history) is None
    assert broaden.quoted_in("Poland in the EU", "q", history) is None
    assert broaden.quoted_in("Poland is in the EU", "q",
                             [{"role": "user",
                               "content": "Poland is in the EU?"}]) is None


def test_a_rewritten_follow_up_is_not_something_the_user_said():
    """The resolved question is model output; only the typed one counts."""
    search = ByQuestionSearch({EU: _found("EU: climate neutrality by 2050")})
    llm = ScriptedLLM([
        "What is the net zero goal of Poland, an EU member state?",
        f"Broader: {EU}\nLink: {LINK}\n"
        f"Quote: Poland, an EU member state",
    ])

    job, answerer = _run(search, question="And its net zero goal?",
                         history=[{"role": "user", "content": "Poland?"},
                                  {"role": "assistant", "content": "Hm."}],
                         llm=llm, settings=_settings(follow_up=True))

    assert "indirect" not in job.trace
    assert answerer.bridges == []


def test_a_trivially_short_quote_proves_nothing():
    assert broaden.quoted_in("yes", "q", MEMBERSHIP_CHAT) is None


# -- the link, from the documents ----------------------------------------

def test_a_link_the_documents_state_is_cited_with_the_answer():
    search = ByQuestionSearch({
        EU: _found("EU: climate neutrality by 2050"),
        LINK: _found("Member states: ... Poland ...", filename="treaty.pdf"),
    })

    job, answerer = _run(search)          # no chat at all

    indirect = job.trace["indirect"]
    assert indirect["source"] == "documents"
    assert indirect["link_citations"] == ["treaty.pdf, p. 1"]
    assert [c["label"] for c in job.citations] == \
        ["eu.pdf, p. 1", "treaty.pdf, p. 1"]
    assert answerer.bridges[0].link_results   # the model saw the passage


# -- when the refusal must stand -----------------------------------------

def test_nothing_for_the_broader_subject_either_keeps_the_refusal():
    job, answerer = _run(ByQuestionSearch({}), history=MEMBERSHIP_CHAT)

    assert "could not find" in job.text
    assert answerer.bridges == []
    assert "nothing cleared the floor" in job.trace["broader_attempt"]


def test_no_proposal_keeps_the_refusal():
    job, _ = _run(ByQuestionSearch({}),
                  llm=ScriptedLLM(["Broader: none\nLink: none\nQuote: none"]))

    assert "could not find" in job.text
    assert job.trace["broader_attempt"] == "no broader subject to try"


def test_a_failing_model_keeps_the_refusal():
    job, _ = _run(ByQuestionSearch({}), llm=FailingLLM())

    assert "could not find" in job.text


def test_a_declined_indirect_answer_keeps_the_refusal_and_no_sources():
    search = ByQuestionSearch({EU: _found("EU: unrelated passage")})

    job, _ = _run(search, history=MEMBERSHIP_CHAT,
                  answerer=StubAnswerer(indirect=[NO_ANSWER]))

    assert NO_ANSWER not in job.text
    assert "could not find" in job.text
    assert job.citations == []
    assert "indirect" not in job.trace


def test_switched_off_it_costs_nothing():
    llm = ScriptedLLM(["", PROPOSAL])     # resolver, then proposal
    search = ByQuestionSearch({EU: _found("EU: climate neutrality by 2050")})

    job, _ = _run(search, history=MEMBERSHIP_CHAT, llm=llm,
                  settings=_settings(broaden=False))

    assert _broaden_calls(llm) == []
    assert search.questions == [POLAND]
    assert "could not find" in job.text


def test_an_empty_scope_is_not_broadened():
    """Nothing retrieved at all is not a gap a broader subject can fill."""
    llm = ScriptedLLM(["", PROPOSAL])     # resolver, then proposal
    search = ByQuestionSearch({POLAND: SearchOutcome(refused=True)})

    _run(search, history=MEMBERSHIP_CHAT, llm=llm)

    assert _broaden_calls(llm) == []


def test_a_direct_answer_never_broadens():
    llm = ScriptedLLM([PROPOSAL])
    search = ByQuestionSearch({POLAND: _found("Poland: 2050", filename="pl.pdf")})

    job, answerer = _run(search, llm=llm,
                         answerer=StubAnswerer(direct=["Poland aims for 2050."]))

    assert job.text == "Poland aims for 2050."
    assert llm.calls == []
    assert answerer.bridges == []


# -- the model declined the direct passages, then broadening ------------

def test_broadening_also_follows_a_declined_direct_answer():
    """The floor let an EU passage through for the Poland question, and the
    model rightly declined it; with the link, it is an answer after all."""
    eu = _found("EU: climate neutrality by 2050")
    search = ByQuestionSearch({POLAND: eu, EU: eu})

    job, _ = _run(search, history=MEMBERSHIP_CHAT)

    assert job.trace["indirect"]["source"] == "conversation"
    assert "model_declined" not in job.trace
    assert job.citations


def test_a_proposal_that_is_just_the_question_is_ignored():
    llm = ScriptedLLM([f"Broader: {POLAND}\nLink: x is y\nQuote: none"])
    assert broaden.propose(llm, POLAND, []) is None


def test_with_reranking_off_the_documents_cannot_prove_a_link():
    """No reranker means no floor: the link search would return something
    whatever the documents say, so it must not count as proof."""
    search = ByQuestionSearch({
        EU: _found("EU: climate neutrality by 2050"),
        LINK: _found("Anything at all", filename="noise.pdf"),
    })

    job, answerer = _run(search, settings=_settings(use_reranker=False))

    assert "indirect" not in job.trace
    assert answerer.bridges == []
    assert LINK not in search.questions
    assert "re-ranking off" in job.trace["broader_attempt"]


def test_with_reranking_off_a_chat_link_still_works():
    search = ByQuestionSearch({EU: _found("EU: climate neutrality by 2050")})

    job, _ = _run(search, history=MEMBERSHIP_CHAT,
                  settings=_settings(use_reranker=False))

    assert job.trace["indirect"]["source"] == "conversation"


def test_without_remember_context_the_chat_cannot_supply_a_link():
    """The same chat that answers through the EU with the box ticked gives
    a refusal with it unticked: the chat is not consulted at all."""
    search = ByQuestionSearch({EU: _found("EU: climate neutrality by 2050")})

    job, answerer = _run(search, history=MEMBERSHIP_CHAT,
                         settings=_settings(follow_up=False))

    assert "indirect" not in job.trace
    assert answerer.bridges == []
    assert "could not find" in job.text
