# Conversation memory commit recovery

With `DUALBRAIN_MEMORY_CONVERSATION_CONTEXT=1` and `DUALBRAIN_PG_DSN` configured,
the stdio engine separates successful generation from confirmed persistence.
An incomplete source generation never prepares a memory operation. For a complete
generation, the engine commits the exact payload to a local SQLite outbox before
attempting PostgreSQL. The new traces, old-source supersession and durable receipt
then commit together in one PostgreSQL transaction. Receipt lookup and retry never
call a model.

## Caller contract

Supply a stable `memory_operation_id` in `process` or `process_stream` params and
reuse it, with the same generation inputs, when retrying an uncertain response:

```json
{
  "session_id": "alpha-discussion",
  "qid": "alpha-correction-2",
  "memory_operation_id": "93addf5d-8aa5-458e-8a6b-3c4cfd652481",
  "question": "Project Alpha's deadline is now seven days. Repeat and remember it.",
  "memory_context": {
    "id": "Alpha", "aliases": ["Alpha"], "remember": true,
    "supersedes": ["alpha-correction-1"]
  }
}
```

The operation ID is a nonempty string of at most 128 characters, scoped to the
session and database. Use a new ID for each intentional revision. The controller's
existing unique source `qid` rule remains; the database does **not** make historical
`session_id + qid` pairs unique. Distinct stored revisions may share those values.
Replacement matching includes the exact previous timestamp, question and answer.

The payload fingerprint is SHA-256 of canonical versioned JSON containing every
new trace and replacement target, including metadata. Identical retries return the
original `memory_commit` receipt, whose `state` is `committed` and whose fields
include `operation_id`, `payload_sha256`, `memory_ids` and `superseded_ids`.
A different payload or replacement set with the same ID raises
`MemoryCommitConflict` without a memory mutation. Conflicting request redelivery
is rejected before recovery, including during session creation.

After a store error, `MemoryCommitPending` identifies the unconfirmed operation.
This means the commit may have succeeded; it does not mean generation failed or
that the old source is still current in PostgreSQL. Retry the same request or use
the stdio recovery method:

```json
{"id": 2, "method": "reconcile_memory", "params": {
  "session_id": "alpha-discussion",
  "memory_operation_id": "93addf5d-8aa5-458e-8a6b-3c4cfd652481"
}}
```

Omit `memory_operation_id` from `reconcile_memory` to recover all pending operations
for that session, in preparation order. This method returns `receipts`. Session
creation and subsequent process requests also reconcile pending operations before
allowing new generation. A replayed process response includes `replayed: true`,
the saved answer/qid and the original receipt. It does not regenerate streaming
tokens, telemetry, dialogue flows or metrics (`metrics` is empty on replay).
In-memory sources are refreshed from current PostgreSQL rows. Replaying an old
receipt after a newer correction cannot reactivate the old source. A receipt proves
a past commit, not that its source remains active.

Without a caller ID the engine generates a UUID at preparation time. That protects
the prepared database write across local retries/restarts, but does not deduplicate
a newly submitted process request. Callers needing RPC retry safety must retain
their own stable ID and request (including `qid`).

## Durable state and boundaries

`DUALBRAIN_MEMORY_OUTBOX_PATH` selects the persistent SQLite file; the default is
`~/.local/state/sr-dual-brain/memory-outbox.sqlite3`. New files use mode 0600. Keep
the file on persistent local storage and reuse it after restart. It contains the
exact source text, targets, original request and basic result; treat it as private
conversation data. It is retained after acknowledgement to support response replay.
No provider credentials are journaled. Custom paths should be excluded from version
control and included in the application's private backup/retention policy.

Outbox entries are namespaced by non-secret DSN target fields and table prefix;
password rotation does not change the namespace. Keep the same explicit target
configuration on restart. Moving to another host requires moving the outbox too.
Changing database routing through environment variables, service definitions or
DNS is outside this identity guarantee. The outbox is not a distributed job queue.

If the process dies before the outbox commit, no PostgreSQL attempt has occurred;
that generated text is not recoverable. After preparation the ID and exact payload
survive a process restart independently of in-memory queues. If the outbox is lost,
durable receipts alone cannot reconstruct an uncommitted operation or its response.
An existing receipt without its original outbox prevents regeneration with that
ID; `PostgresStateStore.memory_commit_receipt` supports read-only receipt inspection.

Competing corrections serialize on the session row. A new operation targeting a
superseded, expired, ambiguous or wrong-scope source is rejected transactionally.
A pending conflict blocks recovery for that session; it is not automatically
rebased or discarded. Resolve the original operation before continuing, or use a
new session after inspecting durable state. Reset refuses locally pending operations.
An explicit reset removes the session's durable receipts; acknowledged local
operations then reject replay instead of recreating deleted memory. Coordinate
resets with all writers. Reset, database restore, disk loss and cross-host failover
are not covered by the retained-receipt guarantee.

This behavior is limited to persistent, opt-in conversation memory. Transient
sessions and legacy append APIs retain their behavior. Existing rows require no
new uniqueness rules or backfill. Custom persistence adapters for this path must
implement `memory_namespace`, `commit_memory_operation`, `memory_commit_receipt`
and `load_session` with the same contract. A configured PostgreSQL initialization
failure stops this mode instead of silently creating transient memory.
Episodes, telemetry and the rest of engine state remain outside the memory commit
transaction. Their failure does not undo a committed memory receipt. Generation
success still does not establish factual truth or semantic completeness.

## Model-free verification

`tests/test_memory_commits.py` exercises engine integration with deterministic
fault injection, immutable outbox payloads, restart recovery, source completion
gating and conflicting requests. Fake-store tests do not establish database atomicity.

`tests/test_memory_commit_postgres.py` uses real PostgreSQL and unique disposable
table prefixes. Set `DUALBRAIN_TEST_PG_DSN` to an **isolated test database**, install
`requirements-dev.txt` and `requirements-pg.txt`, then run:

```sh
pytest -q tests/test_memory_commit_postgres.py
```

The database tests cover rollback before commit, commit success followed by lost
acknowledgement, outbox reopening, immutable payload conflicts, concurrent retries,
legitimate corrections sharing qids, stale receipt replay and legacy episode SQL.
Adversarial cases also exit a child process immediately after a real commit,
recover that unacknowledged operation after a newer correction, race different
operations against one source, and reject a later target after an earlier status
update to verify transaction rollback.
They passed locally on PostgreSQL 16.2 without pgvector, using a private Unix
socket with TCP disabled. CI runs the same suite against an isolated PostgreSQL 16
service. Fresh schema creation exposed an existing unquoted `leading` column;
its SQL references are now quoted. pgvector, production databases, power loss and
distributed failover have not been exercised by these tests.
