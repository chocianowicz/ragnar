"""The Settings panel: a short list for asking, the rest behind a toggle.

Three rules decide where a control goes.

A control is in the user panel only if changing it makes a visible
difference to *this question*. Chunking does not — it affects documents
ingested later — so it sits with ingestion, next to the button that
applies it.

Every control says what it changes and what it costs. The re-ranker costs
about a second per candidate; extra reasoning steps cost model calls
before the answer starts. Those numbers belong in the help text, where the
choice is made.

Nothing in the user panel can switch off the refusal guarantee. Turning
off re-ranking disables the similarity floor, which is the property the
whole product rests on, so it lives behind the advanced toggle with a
warning rather than one click from every question.
"""
import streamlit as st

from ingestion.chunkers.registry import build_chunker
from ui.services import list_chat_models, list_loaded_models

# The similarity floor, as three named stops. PROVISIONAL: these are the
# shipped 0.55 either side of a margin, not calibrated values. Calibration
# needs a verified golden set (eval/golden_set.real.yaml is drafted, not
# reviewed) and `run_eval.py --calibrate`. The help text says so rather
# than implying a rigour that does not exist yet.
STRICTNESS = {
    "Strict": 0.65,
    "Balanced": 0.55,
    "Lenient": 0.45,
}


def _closest_stop(floor: float) -> str:
    return min(STRICTNESS, key=lambda name: abs(STRICTNESS[name] - floor))


def _apply_chunker(svc, strategy: str, chunk_size: int, overlap_pct: int,
                   table_rows: int) -> None:
    """Point the ingestion pipeline at the current chunk settings.

    Called from the Re-chunk button rather than on every rerun: it mutates
    the pipeline every session shares, and doing that while merely
    rendering a panel is how one person's slider changes another person's
    ingest.
    """
    svc["pipeline"].set_chunker(build_chunker({
        "strategy": strategy,
        "target_tokens": chunk_size,
        "overlap_tokens": round(chunk_size * overlap_pct / 100),
        "table_rows_per_group": table_rows,
    }))


def render(svc) -> dict:
    """Render Settings and return the per-request query settings.

    Returning them (rather than mutating shared service objects) is what
    keeps concurrent users from clobbering each other's choices — each
    request carries its own.
    """
    cfg = svc["cfg"]
    prefs = svc["prefs"]

    with st.expander("Settings", expanded=True):
        # ── Ask: the four things that change this question ──────────
        available_models = list_chat_models(
            cfg.ollama_url, cfg.embedding_model
        ) or [cfg.llm_model]
        default_model = (
            cfg.llm_model if cfg.llm_model in available_models
            else available_models[0]
        )
        model = st.selectbox(
            "Answer model", options=available_models,
            index=available_models.index(default_model),
            help="The model that writes the answer from the retrieved "
                 "passages. A larger model is slower and more careful. "
                 "Switching to one that is not already loaded makes the "
                 "next question pay a load of up to a minute.",
        )

        strictness = st.select_slider(
            "Strictness", options=list(STRICTNESS),
            value=_closest_stop(float(prefs.get("floor", cfg.score_floor))),
            help="How sure the app must be before it answers at all. "
                 "Stricter refuses more and is safer against questions the "
                 "documents do not cover; more lenient answers more and "
                 "risks a confident answer built from a weak match. These "
                 "three stops are provisional — the shipped default either "
                 "side of a margin, not calibrated numbers.",
        )
        floor = STRICTNESS[strictness]
        st.caption(f"Relevance floor: {floor:.2f}")

        thorough = st.checkbox(
            "Thorough search", value=bool(prefs.get("thorough", False)),
            help="Searches several rewordings of your question and, if the "
                 "first pass comes back thin, looks for what is missing and "
                 "searches again. For questions your documents phrase "
                 "differently than you do, or that need two passages. Costs "
                 "one or two extra model calls before the answer starts; "
                 "the trace shows what it did.",
        )
        follow_up = st.checkbox(
            "Remember this conversation",
            value=bool(prefs.get("follow_up", True)),
            help="Resolves what a follow-up refers to before searching, so "
                 "\"and the base year for that?\" searches for the thing you "
                 "were discussing rather than the words you typed. Costs one "
                 "model call per follow-up. The answer sees the recent "
                 "conversation either way.",
        )

        # ── Everything else ─────────────────────────────────────────
        st.divider()
        admin = st.toggle(
            "Show advanced settings", value=False,
            help="Tuning and ingestion. These are deployment choices, not "
                 "per-question ones.",
        )

        candidates = int(prefs.get("candidates", cfg.candidates))
        use_reranker = bool(prefs.get("use_reranker", True))
        temperature = float(prefs.get("temperature", 0.0))
        rewrite = bool(prefs.get("rewrite", False))
        self_correct = bool(prefs.get("self_correct", False))

        if admin:
            st.markdown("**Retrieval**")
            candidates = st.slider(
                "Candidates considered", min_value=10, max_value=100,
                value=candidates, step=5,
                help="How many passages are fetched before re-ranking picks "
                     "the best few. The re-ranker scores every one of them — "
                     "roughly a second each — so doubling this roughly "
                     "doubles the wait. It cannot rescue an answer the "
                     "search ranks far down: an exact code in a large table "
                     "sat 291st on this corpus, which no setting here reaches.",
            )
            prefs.set("candidates", int(candidates))

            use_reranker = st.checkbox(
                "Re-rank results", value=use_reranker,
                help="A second, more careful pass over the retrieved "
                     "passages. Turning it off also disables the relevance "
                     "floor — the app will then answer whenever anything is "
                     "retrieved, which removes the guarantee that it refuses "
                     "rather than guesses.",
            )
            prefs.set("use_reranker", bool(use_reranker))

            st.caption(
                "Search mode: "
                + ("meaning + wording (hybrid)" if svc["store"].is_hybrid
                   else "meaning only — run `python -m retrieval.migrate` "
                        "to add lexical search")
            )

            st.markdown("**Generation**")
            temperature = st.slider(
                "Temperature", min_value=0.0, max_value=1.0,
                value=temperature, step=0.1,
                help="0 keeps the answer as close to the excerpts as the "
                     "model can manage, which is what this tool is for. "
                     "Higher values wander.",
            )
            prefs.set("temperature", float(temperature))

            st.markdown("**Extra reasoning steps**")
            st.caption(
                "Thorough search above runs the two with a case for them. "
                "These two are unvalidated: no golden-set evidence says they "
                "help, and each costs model calls before the answer starts."
            )
            rewrite = st.checkbox(
                "Reword the question for search", value=rewrite,
                help="Largely redundant now: follow-up resolution already "
                     "rewrites references, and lexical search already finds "
                     "the identifiers this was meant for.",
            )
            prefs.set("rewrite", bool(rewrite))
            self_correct = st.checkbox(
                "Check the draft answer", value=self_correct,
                help="Drafts an answer, judges whether the excerpts support "
                     "it, and searches again if not. The most expensive "
                     "option: two model calls before the answer you see.",
            )
            prefs.set("self_correct", bool(self_correct))

            _render_ingestion(svc, cfg, prefs)
            _render_diagnostics(svc, cfg)

    if not use_reranker:
        st.warning(
            "Re-ranking is off, so the relevance floor does not apply: the "
            "app will answer whenever anything is retrieved rather than "
            "refusing when nothing is a good match.",
            icon="⚠️",
        )

    prefs.set("floor", float(floor))
    prefs.set("thorough", bool(thorough))
    prefs.set("follow_up", bool(follow_up))

    return {
        "model": model, "temperature": temperature,
        "floor": floor, "candidates": candidates,
        "use_reranker": use_reranker, "follow_up": follow_up,
        # Thorough search is the two stages with a case for them; the other
        # two stay individually switchable in advanced until there is
        # evidence either way.
        "multi_query": thorough, "multi_hop": thorough,
        "rewrite": rewrite, "self_correct": self_correct,
    }


def _render_ingestion(svc, cfg, prefs) -> None:
    """Chunking, and the button that applies it.

    Here rather than in the user panel because none of it affects the
    question being asked — only documents ingested after it changes.
    """
    st.markdown("**Ingestion**")
    default_tokens = int(prefs.get("chunk_size",
                                   cfg.chunking.get("target_tokens", 500)))
    chunk_size = st.number_input(
        "Chunk size (tokens)", min_value=100, max_value=2000,
        value=default_tokens, step=50,
        help="Target size for a passage of prose. Tables follow the row "
             "rule below and are bounded by this as a ceiling.",
    )
    overlap_pct = st.number_input(
        "Overlap (%)", min_value=0, max_value=50,
        value=int(prefs.get("overlap_pct", round(
            100 * cfg.chunking.get("overlap_tokens", 50) / default_tokens))),
        step=1,
        help="How much each passage repeats of the one before, so a "
             "sentence split across a boundary still appears whole "
             "somewhere.",
    )
    table_rows = st.number_input(
        "Table rows per chunk", min_value=5, max_value=200,
        value=int(prefs.get("table_rows",
                            cfg.chunking.get("table_rows_per_group", 20))),
        step=5,
        help="How many rows go into one passage, as a maximum — a wide "
             "table will use fewer to stay within the chunk size.",
    )
    for key, value in (("chunk_size", chunk_size),
                       ("overlap_pct", overlap_pct),
                       ("table_rows", table_rows)):
        prefs.set(key, int(value))

    st.caption(
        "Applies to documents ingested from now on. Documents already "
        "indexed keep the chunking they were ingested with — re-chunk them "
        "below to apply these settings."
    )

    if st.button("Re-chunk all documents", use_container_width=True):
        _apply_chunker(svc, cfg.chunking.get("strategy", "structural"),
                       int(chunk_size), int(overlap_pct), int(table_rows))
        requeued = missing = 0
        for doc in svc["registry"].all():
            if doc.status.value != "done":
                continue
            if svc["storage"].restore_to_inbox(doc.filename, doc.doc_id):
                svc["registry"].requeue(doc.doc_id)
                requeued += 1
            else:
                missing += 1
        st.success(f"Re-queued {requeued} document(s) for re-chunking")
        if missing:
            st.warning(f"{missing} skipped — original file not found")
        st.rerun()


def _render_diagnostics(svc, cfg) -> None:
    """What is loaded and what is indexed — the two things to check first
    when the app is slow or an answer is missing."""
    st.markdown("**Diagnostics**")
    loaded = list_loaded_models(cfg.ollama_url)
    for name in (cfg.llm_model, cfg.embedding_model):
        resident = any(name.split(":")[0] in m for m in loaded)
        st.caption(("🟢 " if resident else "⚪ ") + name
                   + (" — loaded" if resident else
                      " — not loaded; the next question that needs it "
                      "pays the load"))

    try:
        count = svc["store"]._client.get_collection(
            cfg.collection).points_count
        st.caption(f"Index: {cfg.collection} · {count:,} passages · "
                   f"{len(svc['registry'].all())} documents")
    except Exception:
        st.caption(f"Index: {cfg.collection} — not reachable")
