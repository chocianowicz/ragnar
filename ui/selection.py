"""Which documents are ticked, kept outside the checkbox widgets.

Streamlit drops a widget's session-state key when the widget is not
rendered. A folded folder renders no checkboxes, so without this its
documents would read back as the `True` default and silently rejoin the
search — the user folds a folder to tidy the list and quietly widens what
their next question can see.

So the truth lives in one dict that nothing stops rendering, and the
checkboxes are a view of it.
"""
from __future__ import annotations

import streamlit as st

_KEY = "doc_selected"


def _store() -> dict:
    return st.session_state.setdefault(_KEY, {})


def is_selected(doc_id: str) -> bool:
    """Ticked unless we have been told otherwise. A document that has
    never been seen is included, which is what the flat list always did."""
    return _store().get(doc_id, True)


def remember(doc_id: str) -> None:
    """Copy a rendered checkbox's value into the store.

    Called right after the widget renders, so a click made on the previous
    run — which Streamlit has already written into the widget's key — is
    what gets kept.
    """
    widget_key = f"sel_{doc_id}"
    if widget_key in st.session_state:
        _store()[doc_id] = bool(st.session_state[widget_key])


def set_selected(doc_id: str, value: bool) -> None:
    """Tick or untick a document directly.

    Writes both the store and the widget key: the key so a checkbox
    already on screen redraws correctly this run, the store so it survives
    the folder being folded away.
    """
    _store()[doc_id] = bool(value)
    st.session_state[f"sel_{doc_id}"] = bool(value)


def selected_ids(docs) -> set[str]:
    return {d.doc_id for d in docs if is_selected(d.doc_id)}
