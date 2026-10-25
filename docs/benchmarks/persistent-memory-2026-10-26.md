# Persistent source retention — 2026-10-26 (Asia/Tokyo)

A persistent conversation could lose access to an old primary source while that
source was still present in PostgreSQL. Both the live FIFO and cold session load
shared a 1,024-entry budget with ordinary derived answers. Once those answers
filled the recent window, explicit source questions returned no evidence and an
explicit correction failed validation because its target was no longer resident.

Conversation mode now evicts derived/inactive entries before active sources.
PostgreSQL restores active primary sources **before** applying the resident cap,
using a partial `(session_id, ts DESC, id DESC)` index. Expired, superseded and
answer-only records cannot consume that cap. A shared decoder preserves existing
metadata and the legacy default for rows without origin metadata. The ordinary
`load_session` history API still exposes all stored QAs; history is not deleted.

## Actual PostgreSQL experiment

The same benchmark script ran before and after the production change. Baseline
commit `1ce5c9b` contains the diagnostic script with the original production code;
implementation commit `288caae` contains the new retention and restore path.
[Metrics](persistent-memory-2026-10-26.metrics.json) record full revisions, identical
script SHA-256 and per-size fixture hashes. PostgreSQL was 16.15, without pgvector,
over a private Unix socket. Each run used a fresh disposable table prefix in the
isolated test database. The published after-run was measured without concurrent
regression tests; an earlier overlapping run was excluded from this comparison.

Each size has eight authored, scoped primary sources, the stated number of later
derived QAs, and two newer inactive decoys (one expired, one superseded). All
original sources fit within the unchanged primary budget. Exact source qids and
answers are graded, including scope and lifecycle exclusions. There are **zero
provider calls** and **no final-answer quality evaluation**. This is evidence
selection and persistence diagnosis, not a paid language-understanding benchmark.

| Later derived QAs | Live recall before → after | Restart recall before → after | Correction accepted before → after | Restart-resident traces before → after |
| ---: | :---: | :---: | :---: | :---: |
| 128 | 8/8 → 8/8 | 8/8 → 8/8 | yes → yes | 138 → 8 |
| 1,024 | 0/8 → 8/8 | 0/8 → 8/8 | no → yes | 1,024 → 8 |
| 8,192 | 0/8 → 8/8 | 0/8 → 8/8 | no → yes | 1,024 → 8 |
| 32,768 | 0/8 → 8/8 | 0/8 → 8/8 | no → yes | 1,024 → 8 |

All three cold session recreations per size produced the same recall. Neither arm
returned derived, inactive or wrong-scope decoys. At every size the new arm
committed two explicit corrections, replayed the original operation's receipt,
and still selected only the newest revision. Cancelling a pending correction
restored that newest source's availability without a write. The initial operation
result legitimately replays its historical answer; it does not reactivate its
historical source. Unfocused pronouns are separately withheld after explicitly
clearing the benchmark focus; this is not a test of durable focus restoration.

## Latency and costs

Medians are milliseconds. Retrieval has 72 samples per size (eight questions ×
nine repeats) on the final cold session. Hydration has only three samples and
includes DB access, outbox recovery checks and EngineSession construction after
one unmeasured process/model warmup. It is not a full OS/process or storage cold
start. There are no episodic rows in this fixture, so it does not measure the
existing all-episode restore/embedding migration path.

| Derived QAs | Restart retrieval before → after | Session hydration before → after |
| ---: | :---: | :---: |
| 128 | 0.104 → 0.071 | 15.21 → 19.39 |
| 1,024 | 0.287 → 0.072 | 28.31 → 17.62 |
| 8,192 | 0.294 → 0.070 | 43.42 → 19.49 |
| 32,768 | 0.285 → 0.068 | 23.06 → 19.54 |

Smaller restored working sets reduce scan overhead in this fixture. These short
single-machine runs do not establish a general latency speedup: the small case
hydrates more slowly, and startup now uses separate source and episode reads.
Warm retrieval at the largest size rose from 0.282 ms for an empty evidence result
to 0.341 ms for the correct evidence. Bulk live-history construction rose from
8.52 to 76.50 ms for 32,778 entries because retention now searches for an evictable
non-source. That synthetic bulk load is not the steady per-turn persistence path.
The change prioritizes correct source retention, with bounded residency.

## Reproduction and coverage

With dependencies from `requirements-dev.txt` and `requirements-pg.txt` installed,
set `DUALBRAIN_TEST_PG_DSN` to an isolated test database. Do not point it at an
application database. Run from the checkout root with a new output directory:

```bash
python sr-dual-brain-llm/scripts/run_offline_checks.py -- \
  python sr-dual-brain-llm/scripts/benchmark_persistent_memory.py \
  --output-dir target/benchmarks/persistent-memory-local
```

The benchmark creates/drops only its unique prefix. Its local SQLite outbox is in
the output directory; no DSN/credential values enter the report. The baseline and
new measurements require their recorded code revisions, not a simulated reference
adapter. PostgreSQL cache state and host scheduling are not controlled.

Local validation passed **716 Python tests**, including **13 real PostgreSQL
cases**, with no skips; **17 Gateway tests** passed (two binary-on-stdio cases are
unsupported/skipped); **13 Node UI tests** passed. New regressions cover live
retention, pending-write preservation, inactive eviction, source-cap overflow,
legacy FIFO/history compatibility, stable timestamp ties, legacy metadata and
actual EngineSession restart/correction/replay beyond the former recent window.

The 1,024-entry primary cap remains: more than 1,024 active sources still evict
older sources. Many-source, many-expired-source, large-document and pgvector
performance are not measured. Focus/support caches and other control state remain
transient. Cross-process resident-cache invalidation is unchanged. Custom stores
can implement `load_conversation_sources(session_id, limit=...)` to get the new
restore policy; adapters without it retain their compatible recent-history load.
