# The in-app ingest hang: what was ruled out, and what is left

Read-only investigation. Nothing here has been changed in the app; the
findings are recorded so the next person does not repeat the elimination.

## What happened

Twice, the in-app ingest worker went silent with a document stuck in
`processing`, both times on `python data analysis.pdf`. The Streamlit process
showed 5 threads and near-zero CPU. A host re-ingest of the same document
finished in 195 s (582 chunks, on MPS), so the document itself is fine.

## Ruled out

**The document.** Re-ingested on the host in 195 s. A deterministic parse
failure would reproduce there.

**Streamlit's `StopException`.** It subclasses `BaseException`, and
`IngestWorker._loop` catches only `Exception`, so a `StopException` raised
inside the worker would kill the thread silently — which matches the symptom
exactly. But nothing in `ingestion/` imports Streamlit or touches `st.*`
(checked every module), and the worker touches only `registry`, `storage` and
`pipeline`. No path to it was found.

**A crashed worker thread.** `reset_stale_processing()` on `start()` would
recover it on the next app start, and the row stayed stuck — so either the
thread was alive but not progressing, or the app never restarted.

## What the evidence does point at

Docling's `AcceleratorOptions.num_threads` **defaults to 4**, regardless of
how many CPUs the machine has, and reads `DOCLING_NUM_THREADS` /
`OMP_NUM_THREADS` for an override. On this machine:

- Docker Desktop's VM has **16** CPUs (`docker info --format '{{.NCPU}}'`)
- the host has 18 logical / 6 performance cores
- `DOCLING_NUM_THREADS` is set in `docker-compose.yml` but this branch did not
  carry that commit when the hang was observed, and Docling's own default is 4

That is consistent with the observed symptom in a specific way: "5 threads and
near-zero CPU" is what a thread pool looks like when the pool is *contending or
deadlocked* rather than working. A parse that is merely slow still burns CPU.
Pin all its workers to a small subset of cores and add the ONNX/Metal
fallback's own internal threading, and an oversubscribed pool can spin without
making progress.

This is a hypothesis, not a finding. It is testable and cheap to test:

1. Start the container with `DOCLING_NUM_THREADS` matched to what the VM
   actually has, and re-ingest the same document in-app.
2. If it still wedges, dump the threads while it is stuck —
   `py-spy dump --pid <pid>` from inside the container, or
   `faulthandler.dump_traceback_later()` around the parse — and read where the
   worker is parked. That distinguishes "waiting on a lock" from "spinning" from
   "blocked in a syscall", which no amount of reading the source can.
3. Compare against `--device cpu` on the host: if the host with the same thread
   count is fine, the difference is the container's cgroup limits, not Docling.

## Why it is not fixed here

The parse now moves to `host_server.py` over HTTP when `PARSER_URL` is set,
which is the configuration this branch is for. The in-container path remains as
the fallback, so the bug is not gone — but reproducing it needs a live worker
and a shared `registry.db`, and the handoff explicitly rules both out while
another session is ingesting.

## One thing worth knowing about the cache

`data/converted/` on this checkout contained no `.blocks.json` files at all
before this work, so there was nothing from August in an older format to
contaminate a rebuild. `Storage.read_parsed` returns `None` (a cache miss, never
an exception) for a file that is truncated, is not JSON, is missing
`low_confidence`, or carries a block field the current `Block` does not have —
so an old-format file costs a re-parse and cannot build an index from partial
blocks.
