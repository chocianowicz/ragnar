"""Rendering the conversation: transcript, sources and the trace."""
from urllib.parse import quote

import streamlit as st

from ui import sources


def render_sources(citations: list) -> None:
    """Show each cited passage, with a way into the document it came from.

    Citations saved before this existed are plain strings; render those as
    the captions they used to be rather than dropping older conversations.
    """
    flagged = [c for c in citations
               if isinstance(c, dict) and c.get("flags")]
    if flagged:
        st.warning(
            f"{len(flagged)} of the sources below contain text that reads "
            "as instructions to an AI rather than to a reader. The model "
            "may have followed it. Check the flagged passage before "
            "relying on this answer."
        )

    for citation in citations:
        if not isinstance(citation, dict):
            st.caption(str(citation))
            continue

        label = citation.get("label", "source")
        score = citation.get("score")
        heading = f"{label}" + (f"  ·  {score:.2f}" if score is not None else "")
        with st.expander(heading):
            # The passage the model was actually given, not a fresh lookup:
            # this is what the answer was built from.
            st.markdown(
                f"> {citation.get('text', '').strip()[:1500]}"
                .replace("\n", "\n> ")
            )
            for snippet in citation.get("flags") or []:
                # st.text, never markdown: this is document text and must
                # not be able to render markup.
                st.error("Instruction-like text in this passage:", icon="⚠️")
                st.text(snippet)
            doc_id = citation.get("doc_id")
            if doc_id:
                # Links, not buttons: st.link_button opens a new tab, so
                # checking a source does not cost you your place in the
                # chat.
                filename = citation.get("filename", "")
                page = citation.get("page")
                url = f"?doc={quote(doc_id)}"
                if page:
                    url += f"&page={quote(str(page))}"

                published = citation.get("url")

                cols = st.columns(2)
                with cols[0]:
                    if published and sources.viewable(filename):
                        # #page= is understood by the PDF viewers built into
                        # every current browser, so this lands on the cited
                        # page of the real document rather than near it.
                        anchor = f"{published}#page={page}" if page else published
                        st.link_button("Open the original ↗", anchor,
                                       help="The real file, at the cited page")
                    elif published:
                        st.link_button("Open the original ↗", published,
                                       help="Downloads in a new tab")
                with cols[1]:
                    st.link_button("Converted text ↗", url,
                                   help="What the page number indexes")


def render_trace(trace, key: str) -> None:
    """How the answer was found — shown because a local question takes tens
    of seconds, and because a refusal is only actionable if you can see
    whether nothing matched or everything scored just under the floor.

    Wrapped in keyed containers so CSS can render the whole thing smaller
    and dimmed: it is supporting detail, and should not compete with the
    answer for attention.
    """
    if trace is None:
        return
    with st.container(key=f"tracewrap_{key}"), \
            st.expander("How this answer was found"), \
            st.container(key=f"trace_{key}"):
        bits = [
            f"**{trace.get('candidates', 0)}** passages retrieved"
            + (" by meaning and wording" if trace.get("hybrid") else ""),
        ]
        if trace.get("reranked"):
            bits.append(f"**{trace['reranked']}** re-checked by the ranker")
        bits.append(
            f"**{trace.get('kept', 0)}** cleared the relevance floor "
            f"({trace.get('floor', 0):.2f})"
        )
        if trace.get("best_score") is not None:
            bits.append(f"best score **{trace['best_score']:.3f}**")
        st.markdown(" · ".join(bits))
        seconds = trace.get("seconds") or {}
        if seconds:
            st.caption("  ".join(f"{k} {v:.1f}s" for k, v in seconds.items()))

        if trace.get("resolved_question"):
            st.caption("Searched for the follow-up as:")
            # st.text: this is model output and must not be able to inject
            # markup into the page.
            st.text(trace["resolved_question"])

        agentic = trace.get("agentic")
        if agentic:
            st.divider()
            st.markdown(
                f"**{agentic.get('llm_calls', 0)}** extra model call(s) "
                f"before the answer · pool widened to "
                f"**{agentic.get('pool_size', 0)}** passages"
            )
            if agentic.get("rewritten_query"):
                # st.text, not markdown: this is model output and must not
                # be able to inject formatting or markup into the page.
                st.caption("Searched instead for:")
                st.text(agentic["rewritten_query"])
            queries = agentic.get("queries") or []
            if len(queries) > 1:
                st.caption(f"{len(queries)} phrasings searched:")
                for q in queries:
                    st.text(f"• {q}")
            if agentic.get("hops"):
                st.caption(f"Followed up {agentic['hops']} time(s) "
                           f"after finding gaps.")
            if agentic.get("self_corrected"):
                st.caption("Draft answer judged incomplete; searched again.")
            for note in agentic.get("notes") or []:
                st.caption(f"⚠ {note}")


def render_transcript(messages: list[dict]) -> None:
    """Replay the conversation so far."""
    for turn, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                st.caption("Sources")
                render_sources(message["citations"])
            else:
                # No answer came out of the passages, so no sources — but the
                # near misses are still worth offering as somewhere to look.
                related = (message.get("trace") or {}).get("related") or []
                if related:
                    with st.expander("Related documents you might check"):
                        for label in related:
                            st.caption(label)
            render_trace(message.get("trace"), key=f"h{turn}")
