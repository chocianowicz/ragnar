"""Answering in the background, so the UI stays usable while it runs.

Streamlit runs one script per session, and cancels it the moment the user
touches a widget. With generation inline that meant opening another chat,
or nudging a slider, threw away a question that had been running for the
best part of a minute — and left the question in the transcript with no
reply, because the answer is only appended once streaming finishes.

So the work happens on a thread and writes into a Job. The script renders
whatever the Job has so far and re-runs itself while it is unfinished. A
cancelled script now costs a redraw rather than the answer: the thread
does not know or care that it happened, and the finished answer is picked
up by whichever run notices it first — including one in a different tab,
or minutes later after visiting another chat.

Threads here never touch Streamlit. They mutate their own Job and nothing
else; session state and every st.* call stay on the script's thread, which
is the only place they are safe.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Job:
    """One question being answered."""
    chat_id: str
    question: str
    started_at: float = field(default_factory=time.time)
    status: str = "Working"        # last reported stage, for the UI
    chunks: list[str] = field(default_factory=list)
    done: bool = False
    error: str | None = None
    # Filled in once retrieval finishes; the UI needs them to render the
    # finished turn.
    citations: list[dict] = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    mode: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def text(self) -> str:
        with self._lock:
            return "".join(self.chunks)

    def append(self, piece: str) -> None:
        with self._lock:
            self.chunks.append(piece)

    def elapsed(self) -> float:
        return time.time() - self.started_at


class JobRegistry:
    """The in-flight answers, keyed by chat.

    One per chat, deliberately: a second question in the same conversation
    while the first is still running would race to append two turns in an
    order neither the user nor the transcript could predict.
    """

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, chat_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(chat_id)

    def running(self, chat_id: str) -> bool:
        job = self.get(chat_id)
        return job is not None and not job.done

    def any_running(self) -> bool:
        with self._lock:
            return any(not j.done for j in self._jobs.values())

    def finished(self) -> list[Job]:
        """Completed jobs, whichever chat they belong to."""
        with self._lock:
            return [j for j in self._jobs.values() if j.done]

    def pop(self, chat_id: str) -> Job | None:
        with self._lock:
            return self._jobs.pop(chat_id, None)

    def start(self, chat_id: str, question: str, work) -> Job:
        """Run `work(job)` on a thread. Returns the Job immediately.

        `work` is handed the Job and reports into it. Any exception it
        raises is recorded rather than lost on a thread nobody is
        watching.
        """
        job = Job(chat_id=chat_id, question=question)
        with self._lock:
            self._jobs[chat_id] = job

        def run() -> None:
            try:
                work(job)
            except Exception as exc:                  # noqa: BLE001
                job.error = str(exc)
            finally:
                job.done = True

        threading.Thread(target=run, daemon=True,
                         name=f"answer-{chat_id[:8]}").start()
        return job


def new_chat_id() -> str:
    return uuid.uuid4().hex
