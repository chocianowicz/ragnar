# Phase 1 — Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the codebase testable and DRY without changing a single observable behaviour — chiefly by moving the answer orchestration out of the Streamlit script, where the unit suite cannot reach it.

**Architecture:** `ui/app.py` (501 lines) currently mixes module-level script flow with four render functions and a 97-line `answer_job`. That orchestration moves to `generation/answering.py` as a plain function over injected services, testable with fakes. The remaining rendering splits into `ui/chat.py` and `ui/reader.py`, leaving `app.py` as wiring. Alongside: three pieces of duplicated logic collapse, side-effecting I/O leaves the render path, and `retrieval/migrate.py` gets its first tests.

**Tech Stack:** Python 3.11, pytest, Streamlit 1.63 (`streamlit.testing.v1.AppTest`), Qdrant (in-container, for migrate tests). Everything runs inside the `app` container.

**Proof of no behaviour change:** every task ends with the full unit suite at **286 passed, 25 skipped** (Phase 0's number) plus whatever that task adds. The final task runs the integration suite, which must stay at **311 passed**. A behaviour change shows up as a failure in a test nobody touched.

---

## Before you start

```bash
cd /Users/denis/Documents/AI/GitHub/ragnar
RUN="docker compose exec -T app"
docker compose up -d
```

**Commit trailer.** End every commit with the attribution line your harness specifies.

**Branch:** `git switch -c phase-1-structure phase-0-foundations`

**Line numbers** are as of the end of Phase 0. Always locate by the quoted anchor text.

**Smoke test used repeatedly below** (referred to as *the AppTest smoke*):

```bash
$RUN python -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('ui/app.py', default_timeout=200); at.run()
print('exception:', at.exception[0].message if at.exception else 'none')
print('chat input:', len(at.chat_input) > 0)"
```
Expected every time: `exception: none`, `chat input: True`.

## File structure

| File | Responsibility | Change |
|---|---|---|
| `core/models.py` | dataclasses | add `Chunk.key()` |
| `retrieval/search.py` | retrieve / rerank / floor | `SearchTrace.as_dict()` |
| `retrieval/agentic.py` | optional extra stages | one `_fuse`; `_widen` stops mutating its argument; `AgenticTrace.as_dict()` |
| `generation/answering.py` | **new** — the whole answer, start to finish | moved out of `ui/app.py` |
| `generation/answerer.py` | citations | citations carry `url` |
| `ui/chat.py` | **new** — transcript, sources, trace, live answer | moved out of `ui/app.py` |
| `ui/reader.py` | **new** — the `?doc=` reader page | moved out of `ui/app.py` |
| `ui/app.py` | wiring and dispatch only | shrinks to < 150 lines |
| `tests/fakes.py` | shared doubles | gains `ScriptedLLM`, `RecordingStore`, `PassThroughReranker` |
| `tests/test_answering.py` | **new** | unit tests for the extracted orchestration |
| `tests/test_migrate.py` | **new** | first tests for the index rebuild |

---

### Task 0: Baseline and branch

- [ ] **Step 1: Confirm Phase 0's numbers**

Run: `$RUN python -m pytest tests -q 2>&1 | tail -1`
Expected: `286 passed, 25 skipped`.

- [ ] **Step 2: Branch**

```bash
git switch -c phase-1-structure phase-0-foundations
```

---

### Task 1: `Chunk.key()` — one definition of chunk identity

`f"{doc_id}:{chunk_index}"` is written in three places: `store._point_id`, and twice in `agentic.py`. If they drift, deduplication and point identity disagree silently.

**Files:**
- Modify: `core/models.py` (`Chunk`), `retrieval/store.py` (`_point_id`), `retrieval/agentic.py` (`_pool`, `_widen`)
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py`:

```python
def test_chunk_key_identifies_a_chunk_within_its_document():
    from core.models import Chunk

    a = Chunk(doc_id="d1", filename="f.pdf", text="x", chunk_index=3)
    b = Chunk(doc_id="d1", filename="OTHER.pdf", text="y", chunk_index=3)
    c = Chunk(doc_id="d2", filename="f.pdf", text="x", chunk_index=3)
    d = Chunk(doc_id="d1", filename="f.pdf", text="x", chunk_index=4)

    assert a.key() == b.key()      # identity is doc + index, nothing else
    assert a.key() != c.key()
    assert a.key() != d.key()
```

- [ ] **Step 2: Run to verify it fails**

Run: `$RUN python -m pytest tests/test_models.py -k chunk_key -v`
Expected: FAIL, `AttributeError: 'Chunk' object has no attribute 'key'`.

- [ ] **Step 3: Implement**

In `core/models.py`, add to `Chunk` (above `citation_label`):

```python
    def key(self) -> str:
        """Identity of this chunk within the corpus.

        The store derives its point id from this, and the agentic layer
        deduplicates on it. One definition, so those two can never
        disagree about what "the same chunk" means.
        """
        return f"{self.doc_id}:{self.chunk_index}"
```

- [ ] **Step 4: Use it everywhere**

In `retrieval/store.py`, `_point_id` becomes:

```python
        return str(uuid.uuid5(NAMESPACE, chunk.key()))
```

In `retrieval/agentic.py`, replace both occurrences of
`key = f"{result.chunk.doc_id}:{result.chunk.chunk_index}"` with
`key = result.chunk.key()`.

- [ ] **Step 5: Run the suite**

Run: `$RUN python -m pytest tests -q 2>&1 | tail -1`
Expected: `287 passed, 25 skipped`. The store's integration tests are skipped here; Task 10 runs them.

- [ ] **Step 6: Commit**

```bash
git add core/models.py retrieval/store.py retrieval/agentic.py tests/test_models.py
git commit -m "refactor: one definition of chunk identity

doc_id:chunk_index was written out in three places. If they ever drifted,
point identity and deduplication would disagree with nothing failing."
```

---

### Task 2: One `_fuse` in the agentic layer

`_pool` and `_widen` both dedupe by chunk key, sort by score and cap — written out twice. `_widen` also mutates the caller's list via `pool.extend(...)`, a side effect invisible in its signature.

**Files:**
- Modify: `retrieval/agentic.py` (`_pool`, `_widen`, and `_follow_up`'s call sites)
- Test: `tests/test_agentic.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agentic.py`:

```python
def test_fuse_keeps_the_best_score_per_chunk_and_caps():
    from retrieval.agentic import _fuse

    results = [_result("a", index=0, score=0.2),
               _result("a-again", index=0, score=0.9),   # same key, better
               _result("b", index=1, score=0.5),
               _result("c", index=2, score=0.1)]

    fused = _fuse(results, cap=2)

    assert [r.score for r in fused] == [0.9, 0.5]
    assert fused[0].chunk.text == "a-again"


def test_widening_does_not_mutate_the_pool_it_was_given():
    """It used to extend the caller's list in place, which is invisible
    from the signature and surprising on the second hop."""
    store = RecordingStore([_result("extra", index=9)])
    agentic, _ = _agentic(store, ScriptedLLM())
    pool = [_result("original", index=0)]
    before = list(pool)

    agentic._widen("q", "follow up", pool, None, None, 0.3, True,
                   lambda _label: None)

    assert pool == before
```

- [ ] **Step 2: Run to verify they fail**

Run: `$RUN python -m pytest tests/test_agentic.py -k "fuse or mutate" -v`
Expected: 2 failed — `ImportError: cannot import name '_fuse'`, and the mutation test failing because `pool` grew.

- [ ] **Step 3: Implement `_fuse` as a module-level function**

In `retrieval/agentic.py`, add above the `AgenticSearch` class:

```python
def _fuse(results: list[SearchResult], cap: int) -> list[SearchResult]:
    """Best-scoring instance of each chunk, highest first, capped.

    Used for both the multi-query union and each widening hop: the same
    three lines were written out in both places, and a difference between
    them would have been a silent behaviour change.
    """
    best: dict[str, SearchResult] = {}
    for result in results:
        current = best.get(result.chunk.key())
        if current is None or result.score > current.score:
            best[result.chunk.key()] = result
    return sorted(best.values(), key=lambda r: r.score, reverse=True)[:cap]
```

- [ ] **Step 4: Use it in both places**

`_pool`'s body becomes:

```python
        found: list[SearchResult] = []
        for query in queries:
            found += self._search.retrieve(query, doc_ids=doc_ids,
                                           limit=limit)
        # Keep the pool the same size the reranker would have seen anyway,
        # so extra phrasings improve what is in the pool without making the
        # expensive stage any slower.
        cap = limit if limit is not None else self._search.candidates
        return _fuse(found, cap)
```

`_widen`'s body becomes:

```python
        """Rerank the pool plus one query's extra candidates.

        Returns a new pool rather than growing the caller's: the previous
        version extended the list it was passed, which is invisible from
        the signature and compounds over hops.
        """
        extra = self._search.retrieve(extra_query, doc_ids=doc_ids,
                                      limit=limit)
        cap = limit if limit is not None else self._search.candidates
        widened = _fuse(list(pool) + extra, cap)
        return self._search.narrow(question, widened,
                                   score_floor=score_floor,
                                   use_reranker=use_reranker, step=step)
```

- [ ] **Step 5: Run to verify they pass**

Run: `$RUN python -m pytest tests/test_agentic.py -v 2>&1 | tail -3`
Expected: all pass. The multi-hop tests exercise `_widen` through `find()`, so they prove the behaviour is unchanged.

- [ ] **Step 6: Full suite and commit**

Run: `$RUN python -m pytest tests -q 2>&1 | tail -1` → `289 passed, 25 skipped`.

```bash
git add retrieval/agentic.py tests/test_agentic.py
git commit -m "refactor: one fusion helper, and stop widening in place

_pool and _widen each wrote out dedupe-by-key, sort, cap. _widen also
extended the caller's list, which the signature does not admit to and
which compounds over hops. Both now go through _fuse and return new lists."
```

---

### Task 3: Traces serialise themselves

`answer_job` copies every field of `SearchTrace` and `AgenticTrace` into a dict by hand. Add a field to either dataclass and the UI silently stops showing it.

**Files:**
- Modify: `retrieval/search.py` (`SearchTrace`), `retrieval/agentic.py` (`AgenticTrace`)
- Test: `tests/test_search.py`, `tests/test_agentic.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py`:

```python
def test_search_trace_serialises_every_field():
    """Hand-copying the fields meant a new one silently never reached the
    UI. asdict cannot forget."""
    import dataclasses
    from retrieval.search import SearchTrace

    trace = SearchTrace(candidates=25, reranked=25, kept=5, floor=0.55,
                        best_score=0.73, hybrid=True, seconds={"embed": 2.0})
    data = trace.as_dict()

    assert set(data) == {f.name for f in dataclasses.fields(SearchTrace)}
    assert data["kept"] == 5
    assert data["seconds"] == {"embed": 2.0}
```

Append to `tests/test_agentic.py`:

```python
def test_agentic_trace_serialises_every_field():
    import dataclasses
    from retrieval.agentic import AgenticTrace

    trace = AgenticTrace(rewritten_query="x", queries=["a", "b"], hops=1,
                         self_corrected=True, pool_size=25, llm_calls=2,
                         notes=["n"])
    data = trace.as_dict()

    assert set(data) == {f.name for f in dataclasses.fields(AgenticTrace)}
    assert data["queries"] == ["a", "b"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `$RUN python -m pytest tests/test_search.py tests/test_agentic.py -k serialises -v`
Expected: 2 failed, `AttributeError: … has no attribute 'as_dict'`.

- [ ] **Step 3: Implement**

In `retrieval/search.py`, add `import dataclasses` and a method to `SearchTrace`:

```python
    def as_dict(self) -> dict:
        """Every field, for the UI and the chat history.

        dataclasses.asdict rather than a hand-written literal: the
        previous version named each field, so adding one meant the UI
        quietly never showed it.
        """
        return dataclasses.asdict(self)
```

Add the identical method to `AgenticTrace` in `retrieval/agentic.py` (with `import dataclasses` at the top there too).

- [ ] **Step 4: Use them in `answer_job`**

In `ui/app.py`, replace the whole `job.trace = { … }` literal with:

```python
    job.trace = outcome.trace.as_dict()
    job.trace["resolved_question"] = search_question if resolved else None
    if agentic is not None:
        job.trace["agentic"] = agentic.as_dict()
```

- [ ] **Step 5: Verify the UI still shows a trace**

Run the AppTest smoke, then:

```bash
$RUN python -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('ui/app.py', default_timeout=900); at.run()
at.chat_input[0].set_value(\"What is Norway's emission reduction target for 2035?\").run()
t = at.session_state['messages'][-1]['trace']
print('trace keys:', sorted(t))
print('kept:', t['kept'], '| hybrid:', t['hybrid'], '| seconds:', sorted(t['seconds']))"
```
Expected: keys include `candidates, kept, floor, best_score, hybrid, seconds, reranked, resolved_question`; `kept` is 5; seconds has `embed`, `rerank`, `search`. This is the first end-to-end answer in this phase and takes ~60s.

- [ ] **Step 6: Full suite and commit**

Run: `$RUN python -m pytest tests -q 2>&1 | tail -1` → `291 passed, 25 skipped`.

```bash
git add retrieval/search.py retrieval/agentic.py ui/app.py tests/test_search.py tests/test_agentic.py
git commit -m "refactor: traces serialise themselves

answer_job named every field of both trace dataclasses when building the
dict for the UI, so adding a field meant the UI silently never showed it."
```

---

### Task 4: Publish source files when the citation is built, not while rendering

`sources.publish()` runs inside `render_sources` — filesystem I/O per citation per rerun, and while polling that is once a second. It belongs where the citation is made, and the URL belongs in the citation dict so it persists into history.

**Files:**
- Modify: `generation/answerer.py` (`build_citations`), `ui/app.py` (`render_sources`, `answer_job`)
- Test: `tests/test_answerer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_answerer.py`:

```python
def test_citations_carry_a_url_when_a_publisher_is_given():
    """Publishing during rendering meant filesystem I/O per citation per
    rerun — once a second while an answer streams."""
    from generation.answerer import build_citations
    from core.models import Chunk, SearchResult

    result = SearchResult(chunk=Chunk(doc_id="abc", filename="a.pdf",
                                      text="t", chunk_index=0, page=2),
                          score=0.9)
    calls = []

    def publish(doc_id, filename):
        calls.append((doc_id, filename))
        return "/app/static/abc.pdf"

    [citation] = build_citations([result], publish=publish)

    assert citation["url"] == "/app/static/abc.pdf"
    assert calls == [("abc", "a.pdf")]


def test_citations_have_no_url_without_a_publisher():
    from generation.answerer import build_citations
    from core.models import Chunk, SearchResult

    [citation] = build_citations([SearchResult(
        chunk=Chunk(doc_id="a", filename="a.pdf", text="t", chunk_index=0),
        score=0.5)])

    assert citation["url"] is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `$RUN python -m pytest tests/test_answerer.py -k url -v`
Expected: 2 failed — `TypeError: build_citations() got an unexpected keyword argument 'publish'` and `KeyError: 'url'`.

- [ ] **Step 3: Implement**

`build_citations` gains a parameter:

```python
def build_citations(results: list[SearchResult], publish=None) -> list[dict]:
```

Extend the docstring with: *"`publish(doc_id, filename) -> str | None` makes the original file reachable and returns its URL. Injected rather than imported so this module stays free of UI concerns, and called once here rather than on every redraw."*

and the appended dict gains:

```python
            "url": publish(chunk.doc_id, chunk.filename) if publish else None,
```

- [ ] **Step 4: Call it from `answer_job` and simplify `render_sources`**

In `ui/app.py`, `answer_job`'s citation line becomes:

```python
    job.citations = build_citations(
        outcome.results,
        publish=lambda doc_id, filename: sources.publish(
            svc["storage"].archived_path(filename, doc_id), doc_id, filename),
    )
```

In `render_sources`, delete the `published = sources.publish(...)` block and use the stored value:

```python
                filename = citation.get("filename", "")
                page = citation.get("page")
                published = citation.get("url")
                url = f"?doc={quote(doc_id)}"
                if page:
                    url += f"&page={quote(str(page))}"
```

(The rest of the block — the two `st.link_button` calls — is unchanged.)

- [ ] **Step 5: Verify the links still work end to end**

```bash
$RUN python -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('ui/app.py', default_timeout=900); at.run()
at.chat_input[0].set_value(\"What is Norway's emission reduction target for 2035?\").run()
print('exception:', at.exception[0].message if at.exception else 'none')
for l in at.get('link_button')[:2]: print(' ', l.label, '->', l.url)
print('urls persisted on citations:',
      [bool(c.get('url')) for c in at.session_state['messages'][-1]['citations']])"
```
Expected: no exception; an `Open the original ↗` link pointing at `/app/static/<id>.pdf#page=…`; all citations report `True`.

- [ ] **Step 6: Full suite and commit**

Run: `$RUN python -m pytest tests -q 2>&1 | tail -1` → `293 passed, 25 skipped`.

```bash
git add generation/answerer.py ui/app.py tests/test_answerer.py
git commit -m "perf: publish a source file once, when its citation is built

publish() ran inside the render loop: filesystem I/O per citation per
rerun, once a second while an answer streams. It now runs once, where the
citation is made, and the URL travels with the citation into history."
```

---

### Task 5: Promote the test doubles into `tests/fakes.py`

`test_agentic.py` defines `ScriptedLLM`, `RecordingStore` and `PassThroughReranker`. Task 6 needs all three.

**Files:**
- Modify: `tests/fakes.py`, `tests/test_agentic.py`

- [ ] **Step 1: Move them**

Cut the three classes (`ScriptedLLM`, `FailingLLM`, `RecordingStore`, `PassThroughReranker`) from `tests/test_agentic.py` into `tests/fakes.py` verbatim, and in `test_agentic.py` import them:

```python
from tests.fakes import (FakeEmbedder, ScriptedLLM, FailingLLM,
                         RecordingStore, PassThroughReranker)
```

Add a module docstring line to `tests/fakes.py`: *"Shared test doubles. Anything used by more than one test module lives here."*

- [ ] **Step 2: Verify nothing changed**

Run: `$RUN python -m pytest tests -q 2>&1 | tail -1`
Expected: `293 passed, 25 skipped` — exactly the same count. This task adds no tests and must break none.

- [ ] **Step 3: Commit**

```bash
git add tests/fakes.py tests/test_agentic.py
git commit -m "test: move the shared doubles into tests/fakes.py"
```

---

### Task 6: Extract the answer orchestration out of the UI

The reason this phase exists. `answer_job` is 97 lines of orchestration inside a Streamlit script; when it shipped with `declined` missing from the imports, 253 unit tests passed while every answer failed.

**Files:**
- Create: `generation/answering.py`, `tests/test_answering.py`
- Modify: `ui/app.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_answering.py`:

```python
"""Unit tests for the answer pipeline, with no Streamlit involved.

This is the gap this module was created to close: the same logic inside
ui/app.py could only be reached through AppTest, and shipped once with a
NameError that the whole unit suite passed straight through.
"""
import pytest

from core.models import Chunk, SearchResult
from generation.answering import Settings, answer
from generation.prompts import NO_ANSWER
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
    from retrieval.search import SearchOutcome, SearchTrace
    return SearchOutcome(
        results=[SearchResult(chunk=Chunk(doc_id="d", filename="f.pdf",
                                          text=t, chunk_index=i, page=1),
                              score=0.9)
                 for i, t in enumerate(texts)],
        related=list(related), refused=refused, trace=SearchTrace(kept=len(texts)),
    )


def _settings(**over):
    base = dict(model="m", temperature=0.0, floor=0.5, candidates=25,
                use_reranker=True, follow_up=False, rewrite=False,
                multi_query=False, multi_hop=False, self_correct=False)
    base.update(over)
    return Settings(**base)


def test_a_plain_answer_streams_and_carries_citations():
    job = StubJob()
    answer(job, "q", history=[], doc_ids=None, settings=_settings(),
           search=StubSearch(_outcome("the passage")),
           answerer=StubAnswerer(["Three ", "months."]),
           agentic=None, llm=ScriptedLLM(), publish=None)

    assert job.text == "Three months."
    assert [c["label"] for c in job.citations] == ["f.pdf, p. 1"]
    assert job.trace["kept"] == 1


def test_nothing_retrieved_refuses_without_calling_the_model():
    """The refusal must stay free."""
    answerer = StubAnswerer(["should not be reached"])
    job = StubJob()

    answer(job, "q", history=[], doc_ids=None, settings=_settings(),
           search=StubSearch(_outcome(refused=True)),
           answerer=answerer, agentic=None, llm=ScriptedLLM(), publish=None)

    assert "could not find" in job.text
    assert job.citations == []
    assert answerer.histories == []          # never streamed


def test_a_declined_answer_drops_its_sources():
    """Retrieval cleared the floor, the model found no answer in it. The
    passages did not produce this, so they must not be cited."""
    job = StubJob()

    answer(job, "q", history=[], doc_ids=None, settings=_settings(),
           search=StubSearch(_outcome("looks relevant")),
           answerer=StubAnswerer([NO_ANSWER]),
           agentic=None, llm=ScriptedLLM(), publish=None)

    assert job.citations == []
    assert NO_ANSWER not in job.text
    assert "could not find" in job.text
    assert job.trace["model_declined"] is True
    assert job.trace["related"] == ["f.pdf, p. 1"]


def test_the_sentinel_never_reaches_the_job_even_partially():
    """It is held back while the first tokens decide, so it cannot flash
    on screen."""
    job = StubJob()

    answer(job, "q", history=[], doc_ids=None, settings=_settings(),
           search=StubSearch(_outcome("x")),
           answerer=StubAnswerer(list(NO_ANSWER)),   # one character at a time
           agentic=None, llm=ScriptedLLM(), publish=None)

    assert NO_ANSWER not in job.text


def test_a_short_answer_is_not_swallowed_by_the_decision_buffer():
    """Shorter than the sentinel, so the stream ends mid-decision."""
    job = StubJob()

    answer(job, "q", history=[], doc_ids=None, settings=_settings(),
           search=StubSearch(_outcome("x")),
           answerer=StubAnswerer(["Yes."]),
           agentic=None, llm=ScriptedLLM(), publish=None)

    assert job.text == "Yes."


def test_history_reaches_the_answerer():
    job = StubJob()
    history = [{"role": "user", "content": "earlier"}]

    answer(job, "q", history=history, doc_ids=None, settings=_settings(),
           search=StubSearch(_outcome("x")), answerer=(a := StubAnswerer(["ok"])),
           agentic=None, llm=ScriptedLLM(), publish=None)

    assert a.histories[0] == history


def test_a_follow_up_is_resolved_before_searching():
    search = StubSearch(_outcome("x"))
    job = StubJob()

    answer(job, "And the base year?",
           history=[{"role": "user", "content": "Norway's target?"},
                    {"role": "assistant", "content": "70-75%."}],
           doc_ids=None, settings=_settings(follow_up=True), search=search,
           answerer=StubAnswerer(["1990."]), agentic=None,
           llm=ScriptedLLM("What is the base year for Norway's target?"),
           publish=None)

    assert search.questions == ["What is the base year for Norway's target?"]
    assert job.trace["resolved_question"] == \
        "What is the base year for Norway's target?"


def test_the_question_the_user_typed_is_what_gets_answered():
    """Only the search query is rewritten; the model answers what was asked."""
    answerer = StubAnswerer(["1990."])
    job = StubJob()

    answer(job, "And the base year?",
           history=[{"role": "user", "content": "Norway's target?"},
                    {"role": "assistant", "content": "70-75%."}],
           doc_ids=None, settings=_settings(follow_up=True),
           search=StubSearch(_outcome("x")), answerer=answerer, agentic=None,
           llm=ScriptedLLM("What is the base year for Norway's target?"),
           publish=None)

    assert job.text == "1990."


def test_status_is_reported_as_the_work_proceeds():
    job = StubJob()
    answer(job, "q", history=[], doc_ids=None, settings=_settings(),
           search=StubSearch(_outcome("x")), answerer=StubAnswerer(["ok"]),
           agentic=None, llm=ScriptedLLM(), publish=None)

    assert job.status == "Writing the answer"
```

- [ ] **Step 2: Run to verify they fail**

Run: `$RUN python -m pytest tests/test_answering.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'generation.answering'`.

- [ ] **Step 3: Create the module**

Create `generation/answering.py` by moving `answer_job` out of `ui/app.py` essentially verbatim, with services injected instead of read from `svc`, and `query` replaced by a typed `Settings`:

```python
"""The whole answer, start to finish, with nothing from Streamlit in it.

This lived inside ui/app.py, reachable only through AppTest. It shipped
once with a name missing from the imports and the entire unit suite passed
while every answer failed. Services are injected so it can be driven by
fakes; the caller supplies a `job` to report into, which is the only
mutable thing it touches.
"""
from __future__ import annotations

from dataclasses import dataclass

from generation import followup
from generation.answerer import (
    AnswerMode, build_citations, citation_labels, classify, declined,
    NO_RESULTS_MESSAGE,
)
from generation.guards import aggregation_refusal
from generation.prompts import NO_ANSWER


@dataclass(frozen=True)
class Settings:
    """Per-question settings, as chosen in the Settings panel.

    Frozen and passed by value: two people using the app must never be
    able to change each other's choices, which is why none of this is
    written back onto a shared service.
    """
    model: str
    temperature: float
    floor: float
    candidates: int
    use_reranker: bool
    follow_up: bool
    rewrite: bool
    multi_query: bool
    multi_hop: bool
    self_correct: bool

    @property
    def wants_extra_stages(self) -> bool:
        return any((self.rewrite, self.multi_query,
                    self.multi_hop, self.self_correct))


def answer(job, question: str, *, history: list[dict],
           doc_ids: list[str] | None, settings: Settings,
           search, answerer, agentic, llm, publish) -> None:
    """Retrieve, decide, and stream an answer into `job`.

    Reports progress by assigning to job.status and appends text as it
    arrives. Touches no Streamlit API: this runs on a background thread
    that outlives the script run which started it.
    """
    search_question, resolved = question, False
    if history and settings.follow_up:
        job.status = "Working out what the question refers to"
        search_question, resolved = followup.resolve(
            llm, question, history, model=settings.model)

    def step(label: str) -> None:
        job.status = label

    common = dict(doc_ids=doc_ids, score_floor=settings.floor,
                  candidates=settings.candidates,
                  use_reranker=settings.use_reranker, on_step=step)

    if settings.wants_extra_stages and agentic is not None:
        outcome, agentic_trace = agentic.find(
            search_question, rewrite=settings.rewrite,
            multi_query=settings.multi_query, multi_hop=settings.multi_hop,
            self_correct=settings.self_correct, **common)
    else:
        outcome = search.find(search_question, **common)
        agentic_trace = None

    mode = classify(question, outcome.refused, outcome.results)
    job.trace = outcome.trace.as_dict()
    job.trace["resolved_question"] = search_question if resolved else None
    if agentic_trace is not None:
        job.trace["agentic"] = agentic_trace.as_dict()

    if mode is AnswerMode.NO_RESULTS:
        job.append(NO_RESULTS_MESSAGE)
        job.trace["related"] = citation_labels(outcome.related)
        return
    if mode is AnswerMode.AGGREGATION_REFUSED:
        job.append(aggregation_refusal(outcome.results))
        return

    job.citations = build_citations(outcome.results, publish=publish)
    job.status = "Writing the answer"

    # Hold the opening back until it is clear whether this is an answer or
    # the model declining, so the sentinel never appears on screen. It is
    # the first thing emitted when it is emitted at all, so a short buffer
    # settles it.
    buffer, deciding = "", True
    for piece in answerer.stream(question, outcome.results,
                                 model=settings.model,
                                 temperature=settings.temperature,
                                 history=history):
        if deciding:
            buffer += piece
            if declined(buffer):
                break
            if len(buffer.strip()) < len(NO_ANSWER):
                continue
            deciding = False
            job.append(buffer)
            continue
        job.append(piece)

    if declined(buffer):
        # The passages looked relevant but did not answer. Not an answer,
        # so no sources: they did not produce this.
        job.chunks.clear()
        job.citations = []
        job.append(NO_RESULTS_MESSAGE)
        job.trace["model_declined"] = True
        job.trace["related"] = citation_labels(outcome.results)
    elif deciding:
        job.append(buffer)        # stream ended inside the buffer
```

- [ ] **Step 4: Run to verify they pass**

Run: `$RUN python -m pytest tests/test_answering.py -v`
Expected: 9 passed.

- [ ] **Step 5: Call it from the UI**

In `ui/app.py`: delete the whole `answer_job` function; import `from generation.answering import Settings, answer as run_answer`; and change the `jobs.start(...)` call to:

```python
    settings = Settings(
        model=query["model"], temperature=query["temperature"],
        floor=query["floor"], candidates=query["candidates"],
        use_reranker=query["use_reranker"], follow_up=query["follow_up"],
        rewrite=query["rewrite"], multi_query=query["multi_query"],
        multi_hop=query["multi_hop"], self_correct=query["self_correct"])

    jobs.start(
        st.session_state.current_chat_id,
        lambda job: run_answer(
            job, question, history=history, doc_ids=doc_ids_filter,
            settings=settings, search=svc["search"],
            answerer=svc["answerer"], agentic=svc["agentic"], llm=svc["llm"],
            publish=lambda doc_id, filename: sources.publish(
                svc["storage"].archived_path(filename, doc_id),
                doc_id, filename),
        ),
    )
```

Remove imports left unused (`pyflakes` in Step 7 will name them).

- [ ] **Step 6: Verify end to end**

Run the AppTest smoke, then a real question as in Task 3 Step 5, plus a deliberate refusal:

```bash
$RUN python -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('ui/app.py', default_timeout=900); at.run()
at.chat_input[0].set_value('What is the capital of France?').run()
m = at.session_state['messages'][-1]
print('refusal:', m['content'][:60])
print('citations:', len(m['citations']), '| related:', len(m['trace'].get('related') or []))"
```
Expected: the standard refusal, 0 citations, some related documents.

- [ ] **Step 7: Full suite, pyflakes, commit**

Run:
```bash
$RUN python -m pytest tests -q 2>&1 | tail -1      # 302 passed, 25 skipped
$RUN python -m pyflakes generation/answering.py ui/app.py
```
Expected: the suite passes; pyflakes prints nothing.

```bash
git add generation/answering.py ui/app.py tests/test_answering.py
git commit -m "refactor: move the answer pipeline out of the Streamlit script

answer_job was 97 lines of orchestration inside ui/app.py, reachable only
through AppTest. It shipped once with a name missing from the imports and
the whole unit suite passed while every answer failed. It is now a plain
function over injected services with nine unit tests, including the
declined-answer path that regression proved was untested."
```

---

### Task 7: Split the rendering out of `ui/app.py`

**Files:**
- Create: `ui/chat.py`, `ui/reader.py`
- Modify: `ui/app.py`

- [ ] **Step 1: Move the reader page**

Create `ui/reader.py` containing `render_document_page`, taking `svc` as its first argument instead of closing over it:

```python
"""The standalone document page a citation opens in a new tab."""
import streamlit as st

from ui import sources


def render_document_page(svc, doc_id: str, page: str | None) -> None:
    ...   # body moved verbatim from ui/app.py
```

- [ ] **Step 2: Move the chat rendering**

Create `ui/chat.py` containing `render_sources`, `render_trace`, and a new `render_transcript(messages)` holding the history loop. `render_sources` needs `svc` only for nothing now (Task 4 removed the publish call), so it keeps its current signature.

- [ ] **Step 3: Rewire `ui/app.py`**

`app.py` keeps: page config, `svc`, the job registry, `commit`/`reap_finished_jobs`, the Ollama health check, the query-param dispatch, the sidebar, the document filter, the chat input handler, the live-answer block and the polling. It imports the rest.

- [ ] **Step 4: Verify**

Run the AppTest smoke, and:

```bash
$RUN python -m pytest tests -q 2>&1 | tail -1
$RUN python -m pyflakes ui/*.py ui/panels/*.py
wc -l ui/app.py ui/chat.py ui/reader.py
```
Expected: `302 passed, 25 skipped`; pyflakes silent; `app.py` under 150 lines.

Also check the reader page still renders, using a real doc id:

```bash
$RUN python -c "
from streamlit.testing.v1 import AppTest
from core.config import Config
from ingestion.registry_db import Registry
doc = Registry(Config().data_dir / 'registry.db').all()[0]
at = AppTest.from_file('ui/app.py', default_timeout=200)
at.query_params['doc'] = doc.doc_id; at.run()
print('exception:', at.exception[0].message if at.exception else 'none')
print('title:', [t.value for t in at.title])"
```
Expected: no exception; the title is the document's filename.

- [ ] **Step 5: Commit**

```bash
git add ui/app.py ui/chat.py ui/reader.py
git commit -m "refactor: split rendering out of ui/app.py

501 lines mixing script flow with four render functions. app.py is now
wiring and dispatch; the transcript, sources and trace live in ui/chat.py
and the reader page in ui/reader.py."
```

---

### Task 8: Redraw only the answer — ATTEMPTED AND REVERTED

The intent: replace `time.sleep(1); st.rerun()` at the bottom of `ui/app.py`
with `@st.fragment(run_every="1s")` around the answer area, so a streaming
answer redraws one block instead of the whole page.

**It was implemented and reverted.** `st.fragment(run_every=…)` does not
auto-rerun under `streamlit.testing.v1.AppTest`: the fragment executes once
inline, the job is still running, and nothing polls again. The module-level
`st.rerun()` is what made AppTest loop until the answer finished. With the
fragment in place the end-to-end check came back with the *user* turn as the
last message — the answer never committed.

That is an AppTest limitation rather than a browser bug, but the trade is
bad either way: this phase and Phase 0 both caught real regressions with
exactly that end-to-end check (a missing import that the whole unit suite
passed, and a chat-switch bug). Losing the ability to drive an answer to
completion in a test costs more than a once-a-second full redraw of a
single-user local app.

Revisit only with a way to verify an answer end to end that does not depend
on the module-level rerun — e.g. driving a real browser session, or an
explicit "run until the job finishes" helper the test can call.

### Task 9: First tests for the index rebuild

`retrieval/migrate.py` rewrites the index and has never been tested. It ran once, successfully; that is not the same thing.

**Files:**
- Create: `tests/test_migrate.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_migrate.py`:

```python
"""Tests for the dense -> hybrid migration.

It rewrites the index, so the one thing that must never happen is a
partial copy being mistaken for a complete one.
"""
import os
import uuid

import pytest

from core.config import Config
from core.models import Chunk
from retrieval.migrate import migrate, _dense_of
from retrieval.store import QdrantStore, DENSE


def _chunk(doc_id, text, index=0):
    return Chunk(doc_id=doc_id, filename="f.pdf", text=text, chunk_index=index)


@pytest.fixture
def dense_collection():
    """A pre-hybrid collection with a few points, dropped afterwards."""
    name = f"test_{uuid.uuid4().hex[:8]}"
    store = QdrantStore(os.environ["QDRANT_URL"], name, dim=8, hybrid=False)
    store.ensure_collection()
    store.upsert([_chunk("d1", "ammonium sulphate 31022100", 0),
                  _chunk("d1", "notice period three months", 1)],
                 [[0.1] * 8, [0.2] * 8])
    yield name, store
    for n in (name, f"{name}_hybrid"):
        try:
            QdrantStore(os.environ["QDRANT_URL"], n, dim=8).drop_collection()
        except Exception:
            pass


@pytest.mark.integration
def test_dry_run_changes_nothing(dense_collection, capsys):
    name, store = dense_collection
    cfg = Config()
    cfg._raw["storage"]["collection"] = name
    cfg._raw["models"]["embedding_dim"] = 8

    migrate(cfg, apply=False)

    assert "Would build" in capsys.readouterr().out
    assert not store.is_hybrid


@pytest.mark.integration
def test_apply_builds_a_hybrid_copy_and_leaves_the_source_alone(dense_collection):
    name, store = dense_collection
    cfg = Config()
    cfg._raw["storage"]["collection"] = name
    cfg._raw["models"]["embedding_dim"] = 8

    migrate(cfg, apply=True)

    target = QdrantStore(os.environ["QDRANT_URL"], f"{name}_hybrid", dim=8)
    assert target.is_hybrid
    assert target._client.get_collection(f"{name}_hybrid").points_count == 2
    # the source is untouched and still serving
    assert not store.is_hybrid
    assert store._client.get_collection(name).points_count == 2


@pytest.mark.integration
def test_the_lexical_half_is_built_from_the_stored_text(dense_collection):
    """The whole point: no re-parse, no re-embed — the text is already in
    the payload, so an exact identifier is findable afterwards."""
    name, _ = dense_collection
    cfg = Config()
    cfg._raw["storage"]["collection"] = name
    cfg._raw["models"]["embedding_dim"] = 8
    migrate(cfg, apply=True)

    target = QdrantStore(os.environ["QDRANT_URL"], f"{name}_hybrid", dim=8)
    hits = target.search([0.15] * 8, limit=2, text="31022100")

    assert "31022100" in hits[0].chunk.text


@pytest.mark.integration
def test_migrating_an_already_hybrid_collection_is_a_no_op(dense_collection, capsys):
    name, _ = dense_collection
    cfg = Config()
    cfg._raw["storage"]["collection"] = name
    cfg._raw["models"]["embedding_dim"] = 8
    migrate(cfg, apply=True)

    cfg._raw["storage"]["collection"] = f"{name}_hybrid"
    migrate(cfg, apply=True)

    assert "Nothing to do" in capsys.readouterr().out


@pytest.mark.integration
def test_refuses_to_overwrite_an_existing_target(dense_collection):
    name, _ = dense_collection
    cfg = Config()
    cfg._raw["storage"]["collection"] = name
    cfg._raw["models"]["embedding_dim"] = 8
    migrate(cfg, apply=True)

    with pytest.raises(SystemExit, match="already exists"):
        migrate(cfg, apply=True)


def test_dense_of_accepts_both_stored_shapes():
    """Points may carry a bare vector or a named one."""
    class P:
        def __init__(self, v):
            self.vector = v

    assert _dense_of(P([0.1, 0.2])) == [0.1, 0.2]
    assert _dense_of(P({DENSE: [0.3], "sparse": None})) == [0.3]
```

- [ ] **Step 2: Run them**

Run: `$RUN python -m pytest tests/test_migrate.py -v --run-integration`
Expected: 6 passed. If `test_refuses_to_overwrite_an_existing_target` fails because the fixture's cleanup already dropped the target, check the fixture teardown order.

- [ ] **Step 3: Confirm they are skipped without the flag**

Run: `$RUN python -m pytest tests/test_migrate.py -q`
Expected: `1 passed, 5 skipped` (only `_dense_of` needs no Qdrant).

- [ ] **Step 4: Commit**

```bash
git add tests/test_migrate.py
git commit -m "test: first tests for the hybrid migration

It rewrites the index and had only ever been verified by running it once."
```

---

### Task 10: Full verification

- [ ] **Step 1: Unit suite**

Run: `$RUN python -m pytest tests -q 2>&1 | tail -1`
Expected: `308 passed, 30 skipped` (302 + the 6 migrate tests, 5 of which skip without the flag — so `303 passed, 30 skipped`; record the actual and reconcile).

- [ ] **Step 2: Integration suite**

Run: `$RUN python -m pytest tests -q --run-integration 2>&1 | tail -1`
Expected: all pass, zero failures. Phase 0 ended at 311; this phase adds 9 (Task 6) + 6 (Task 9) + 5 (Tasks 1–4) = **331 passed**.

- [ ] **Step 3: Confirm the behaviour is genuinely unchanged**

Ask the three questions this phase has used throughout and compare with Phase 0's answers:

```bash
$RUN python -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('ui/app.py', default_timeout=900); at.run()
for q in [\"What is Norway's emission reduction target for 2035?\",
          'What is the Column A benchmark value for CN code 31022100?',
          'What is the capital of France?']:
    at.chat_input[0].set_value(q).run()
    m = at.session_state['messages'][-1]
    print(f'{q[:45]:47s} -> {m[\"content\"][:55]} | {len(m[\"citations\"])} sources')"
```
Expected: 70–75% vs 1990 with 5 sources; 0.022 tCO2e/t with sources; the refusal with 0 sources.

- [ ] **Step 4: Line counts**

Run: `wc -l ui/app.py ui/chat.py ui/reader.py generation/answering.py`
Expected: `app.py` under 150; nothing over 250.

- [ ] **Step 5: Report**

State: test counts before and after, the three answers, final line counts, and anything that behaved differently from Phase 0. Nothing is pushed.

---

## Not in this plan

- Settings regrouping, and the decision on `rewrite` / `self_correct` — Phase 2, which needs the golden set.
- The reranker service, helper model, adaptive candidates — Phase 3.
- Heading-aware chunking, cross-page prose, `rebuild-index` — Phase 4, which needs a re-ingest.
- Any change to what an answer says. If one of the three questions in Task 10 Step 3 answers differently, that is a bug in this phase, not a new feature.
