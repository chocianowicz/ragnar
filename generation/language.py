"""The language of the question, for the sentences the app writes itself.

The model is told to answer in the language of the question, and mostly
does. Refusals are different: they are fixed text, written by code so
that no model call can embellish them, which also meant they were always
English, whatever the question was asked in. On a team that asks in
Polish as often as in English, most refusals came back in the wrong
language.

English and Polish, the languages this team asks in, have hand-written
sentences: detection is deterministic and no model is called, so those
refusals stay free and predictable. A question in any other language gets
the English sentence translated by the model (translate()), with a check
in code that every file name and page it names came through unchanged;
if one did not, the English original is used rather than a translation
that might name a different document.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Letters that occur in Polish and in no English word.
_POLISH_LETTERS = set("ąćęłńśźż")

# Short, frequent Polish words that are not English words. One is enough
# in a question: "czy", "jest" and "jaki" do not occur in English text.
# "co" and "mi" are left out although common: "CO emissions" (carbon
# monoxide) and "mi" (miles) are English on this corpus.
_POLISH_WORDS = {
    "jaki", "jaka", "jakie", "jakich", "jakim", "jest", "czy", "jak",
    "dla", "nie", "który", "która", "które", "ile", "gdzie", "kiedy",
    "dlaczego", "przez", "oraz", "czym", "podaj", "opisz", "roku", "cel",
    "celem", "jakiej", "jakiego", "mamy", "moze",
}

# Frequent English words that are not words in the other languages
# people are likely to ask in. Shared ones are left out: "was" and "will"
# (German), "is" and "of" (Dutch), "in", "a" and "die". Mistaking English
# for another language costs one translation call on a refusal, which then
# comes back in English anyway; the reverse gives the wrong language.
_ENGLISH_WORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "are", "were", "the", "does", "do", "did", "can", "could", "should",
    "would", "for", "with", "about", "and", "or", "this", "that", "these",
    "those", "there", "list", "show", "tell", "explain", "describe", "give",
    "much", "many", "any", "its", "their",
}

TEXT = {
    "en": {
        "no_answer": "I could not find an answer to this in the indexed "
                     "documents.",
        "and": " and ",
        "near_miss_declined": "The closest passages were in {listed}, but "
                              "they do not answer it.",
        "near_miss_floor": "The closest passages were in {listed}, but none "
                           "matched closely enough to answer from.",
        "missing_subject": "None of the passages found mention {subject}.",
        "aggregation": "This looks like a question that requires calculating "
                       "across a whole table. I can only read individual "
                       "rows, so any total I gave you could be wrong.\n\n"
                       "The relevant data is in: {listed}",
        "sheet": "{filename} (sheet {sheet})",
    },
    "pl": {
        "no_answer": "Nie udało się znaleźć odpowiedzi na to pytanie w "
                     "zindeksowanych dokumentach.",
        "and": " i ",
        "near_miss_declined": "Najbliższe fragmenty są w: {listed}, ale nie "
                              "odpowiadają na to pytanie.",
        "near_miss_floor": "Najbliższe fragmenty są w: {listed}, ale żaden "
                           "nie pasuje wystarczająco, by na jego podstawie "
                           "odpowiedzieć.",
        # The subject is quoted as the question wrote it, already inflected
        # ("Polski"), so it stands in quotation marks rather than inside a
        # phrase that would need its own grammatical case.
        "missing_subject": "Żaden ze znalezionych fragmentów nie dotyczy "
                           "tematu pytania („{subject}”).",
        "aggregation": "To pytanie wymaga obliczeń na całej tabeli. Czytam "
                       "tylko pojedyncze wiersze, więc podana przeze mnie "
                       "suma mogłaby być błędna.\n\n"
                       "Odpowiednie dane są w: {listed}",
        "sheet": "{filename} (arkusz {sheet})",
    },
}


OTHER = "other"


def of(text: str) -> str:
    """"pl", "en", or OTHER for a question in any other language."""
    lowered = text.casefold()
    words = set(re.findall(r"\w+", lowered))
    if _POLISH_LETTERS & set(lowered) or _POLISH_WORDS & words:
        return "pl"
    if _ENGLISH_WORDS & words:
        return "en"
    return OTHER


def text(lang: str, key: str, **values) -> str:
    """A fixed sentence in `lang`, or in English where there is none.

    For OTHER that English sentence is what translate() starts from.
    """
    table = TEXT.get(lang, TEXT["en"])
    return table.get(key, TEXT["en"][key]).format(**values)


TRANSLATE_SYSTEM = """\
You translate a short message into the language the question is written
in.

Rules:
- Translate the message only. Do not answer the question.
- Keep every file name, page number and anything in quotation marks
  exactly as written.
- Add nothing, leave nothing out.
- Output only the translation.
"""


def translate(llm, question: str, message: str, keep: list[str], *,
              model: str | None = None) -> str:
    """`message` in the language of `question`, or `message` unchanged.

    Only used for OTHER, where no hand-written sentence exists. `keep` is
    every citation label and quoted name the message contains; a
    translation missing any of them is discarded, since a refusal that
    names the wrong document sends the user to read the wrong thing.
    """
    try:
        out = llm.generate(TRANSLATE_SYSTEM,
                           f"Question: {question}\n\nMessage: {message}",
                           model=model).strip()
    except Exception as exc:
        logger.warning("refusal translation failed: %s", exc)
        return message
    if not out or any(item not in out for item in keep):
        return message
    return out
