import threading
import time

import pytest

from ui.jobs import Job, JobRegistry, new_chat_id


def test_a_job_runs_without_blocking_the_caller():
    """The whole point: the script returns while the answer continues."""
    registry = JobRegistry()
    release = threading.Event()

    def work(job):
        release.wait(timeout=5)
        job.append("done")

    started = time.time()
    registry.start("chat1", work)
    assert time.time() - started < 0.5      # start() returned immediately
    assert registry.running("chat1")

    release.set()
    _wait_done(registry, "chat1")
    assert registry.get("chat1").text == "done"


def test_partial_text_is_readable_while_the_job_runs():
    """So the UI can render an answer as it arrives, not only at the end."""
    registry = JobRegistry()
    seen = threading.Event()

    def work(job):
        job.append("half ")
        seen.set()
        time.sleep(0.2)
        job.append("and half")

    registry.start("chat1", work)
    seen.wait(timeout=5)
    assert registry.get("chat1").text == "half "

    _wait_done(registry, "chat1")
    assert registry.get("chat1").text == "half and half"


def test_a_failing_job_records_the_error_and_still_finishes():
    """An exception on a thread nobody watches would otherwise vanish, and
    the UI would wait for an answer that is never coming."""
    registry = JobRegistry()

    def work(job):
        raise RuntimeError("ollama is down")

    registry.start("chat1", work)
    _wait_done(registry, "chat1")

    job = registry.get("chat1")
    assert job.done
    assert "ollama is down" in job.error


def test_jobs_are_isolated_per_chat():
    registry = JobRegistry()
    registry.start("a", lambda job: job.append("first"))
    registry.start("b", lambda job: job.append("second"))
    _wait_done(registry, "a")
    _wait_done(registry, "b")

    assert registry.get("a").text == "first"
    assert registry.get("b").text == "second"


def test_popping_a_job_clears_it():
    registry = JobRegistry()
    registry.start("a", lambda job: job.append("x"))
    _wait_done(registry, "a")

    popped = registry.pop("a")

    assert popped.text == "x"
    assert registry.get("a") is None
    assert not registry.running("a")


def test_running_is_false_for_an_unknown_chat():
    assert not JobRegistry().running("never-seen")


def test_status_is_readable_while_running():
    registry = JobRegistry()
    reached = threading.Event()

    def work(job):
        job.status = "Re-checking the 25 closest passages"
        reached.set()
        time.sleep(0.1)

    registry.start("a", work)
    reached.wait(timeout=5)
    assert "25 closest" in registry.get("a").status
    _wait_done(registry, "a")


def test_appending_from_several_threads_loses_nothing():
    job = Job(chat_id="a")
    threads = [threading.Thread(target=lambda: [job.append("x")
                                                for _ in range(200)])
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(job.text) == 800


def test_chat_ids_are_unique():
    assert new_chat_id() != new_chat_id()


def _wait_done(registry, chat_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = registry.get(chat_id)
        if job is not None and job.done:
            return
        time.sleep(0.01)
    raise AssertionError(f"job {chat_id} did not finish")


def test_a_finished_job_is_reported_whatever_chat_is_on_screen():
    """The walk-away case: the answer finishes while you are elsewhere, and
    something has to notice and file it."""
    registry = JobRegistry()
    registry.start("chat-a", lambda job: job.append("answer a"))
    _wait_done(registry, "chat-a")

    [finished] = registry.finished()

    assert finished.chat_id == "chat-a"
    assert finished.text == "answer a"


def test_finished_does_not_report_jobs_still_running():
    registry = JobRegistry()
    release = threading.Event()
    registry.start("a", lambda job: release.wait(timeout=5))

    assert registry.finished() == []

    release.set()
    _wait_done(registry, "a")
    assert len(registry.finished()) == 1
