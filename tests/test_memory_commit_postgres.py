"""Real PostgreSQL integration tests; use only an isolated test database.

Run with DUALBRAIN_TEST_PG_DSN set and requirements-pg.txt installed. No model calls.
"""
import asyncio
from copy import deepcopy
from dataclasses import replace
import os
import uuid

import pytest

from core.memory_commits import MemoryCommitConflict, MemoryCommitPending, MemoryOutbox, memory_payload, reconcile_memory
from core.postgres_state_store import PostgresStateStore
from core.shared_memory import MemoryTrace
from test_conversation_memory import memory, retrieve


@pytest.fixture
def pg_store():
    dsn = os.environ.get("DUALBRAIN_TEST_PG_DSN")
    if not dsn:
        pytest.skip("isolated PostgreSQL DSN not configured")
    pytest.importorskip("psycopg")
    store = PostgresStateStore(dsn, table_prefix="retry_test_" + uuid.uuid4().hex[:12])
    asyncio.run(store.ensure_schema())
    try:
        yield store
    finally:
        with store._connect() as conn, conn.cursor() as cur:
            for name in ("memory_commits", "shared_memory", "hippocampal_episodes", "telemetry_events",
                         "schema_memory", "consolidation_state", "sessions"):
                cur.execute(f"DROP TABLE IF EXISTS {store._table(name)} CASCADE")


def source(answer, ts, qid="shared-qid"):
    return MemoryTrace("Project Alpha deadline", answer, qid=qid, timestamp=ts,
                       retrieval_scope="Alpha", scope_aliases=("Alpha",))


def load(store):
    return asyncio.run(store.load_session("s"))[0]


def count_receipts(store):
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {store._table('memory_commits')}")
        return cur.fetchone()[0]


@pytest.mark.parametrize("fault", ["before", "after"])
def test_real_commit_fault_restart_and_old_receipt_replay(pg_store, tmp_path, monkeypatch, fault):
    store = pg_store
    old, new, newest = source("14 days", 1), source("7 days", 2), source("3 days", 3)
    asyncio.run(store.append_memory_traces("s", [old]))
    payload = memory_payload([new], [replace(old, evidence_status="superseded")])
    path = tmp_path / "outbox.sqlite3"
    outbox = MemoryOutbox(path, store.memory_namespace)
    outbox.prepare("s", "first", payload)
    connect = store._connect

    class FaultConnection:
        def __init__(self): self.conn = connect()
        def __enter__(self): self.conn.__enter__(); return self
        def __exit__(self, *args): return self.conn.__exit__(*args)
        def cursor(self): return self.conn.cursor()
        def commit(self):
            if fault == "before":
                raise OSError("injected before transaction commit")
            self.conn.commit()
            raise OSError("injected lost commit acknowledgement")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_connect", FaultConnection)
        with pytest.raises(MemoryCommitPending):
            asyncio.run(reconcile_memory(outbox, store, "s"))
    if fault == "before":
        assert load(store) == [old] and count_receipts(store) == 0
    else:
        assert len(load(store)) == 2 and count_receipts(store) == 1
        assert load(store)[0].evidence_status == "superseded"
    durable_receipt = asyncio.run(store.memory_commit_receipt("s", "first"))

    restarted = MemoryOutbox(path, store.memory_namespace)
    receipt = asyncio.run(reconcile_memory(restarted, store, "s"))[0]
    assert durable_receipt is None or receipt == durable_receipt
    assert len(load(store)) == 2 and count_receipts(store) == 1
    assert receipt["memory_ids"] and receipt["superseded_ids"]

    # Distinct revisions intentionally share session_id + qid; exact targets include ts/text.
    next_payload = memory_payload([newest], [replace(new, evidence_status="superseded")])
    asyncio.run(store.commit_memory_operation("s", "second", next_payload))
    before = load(store)
    assert asyncio.run(store.commit_memory_operation("s", "first", payload)) == receipt
    assert asyncio.run(reconcile_memory(restarted, store, "s", operation_id="first")) == [receipt]
    assert load(store) == before and count_receipts(store) == 2
    m = memory()
    m.past_qas = load(store)
    assert [t.answer for t in retrieve(m, "Project Alpha deadline?")] == ["3 days"]


@pytest.mark.parametrize("change", ["answer", "targets"])
def test_real_conflicting_operation_never_mutates_memory(pg_store, change):
    old, new = source("14 days", 1), source("7 days", 2)
    asyncio.run(pg_store.append_memory_traces("s", [old]))
    payload = memory_payload([new], [replace(old, evidence_status="superseded")])
    receipt = asyncio.run(pg_store.commit_memory_operation("s", "operation", payload))
    before = load(pg_store)
    conflict = deepcopy(payload)
    if change == "answer":
        conflict["traces"][0]["answer"] = "9 days"
    else:
        conflict["updates"] = []
    with pytest.raises(MemoryCommitConflict):
        asyncio.run(pg_store.commit_memory_operation("s", "operation", conflict))
    assert load(pg_store) == before and count_receipts(pg_store) == 1
    assert asyncio.run(pg_store.memory_commit_receipt("s", "operation")) == receipt


def test_concurrent_duplicate_and_competing_corrections(pg_store):
    old = source("14 days", 1)
    asyncio.run(pg_store.append_memory_traces("s", [old]))
    payload = memory_payload([source("7 days", 2)], [replace(old, evidence_status="superseded")])

    async def duplicate():
        return await asyncio.gather(*(pg_store.commit_memory_operation("s", "same", payload) for _ in range(4)))

    receipts = asyncio.run(duplicate())
    assert all(receipt == receipts[0] for receipt in receipts)
    assert len(load(pg_store)) == 2 and count_receipts(pg_store) == 1
    # A different operation cannot supersede an already replaced source.
    with pytest.raises(MemoryCommitConflict):
        asyncio.run(pg_store.commit_memory_operation("s", "stale", payload))
    assert len(load(pg_store)) == 2 and count_receipts(pg_store) == 1


def test_legacy_duplicate_qids_remain_allowed(pg_store):
    asyncio.run(pg_store.append_memory_traces("s", [source("one", 1), source("two", 2)]))
    assert len(load(pg_store)) == 2 and count_receipts(pg_store) == 0


def test_existing_episode_sql_roundtrip_with_quoted_leading_column(pg_store):
    import numpy as np
    from core.temporal_hippocampal_indexing import EpisodicTrace
    episode = EpisodicTrace("episode", "question", "answer", np.ones(128), leading="left")
    asyncio.run(pg_store.upsert_episodes("s", [episode]))
    # Exercise both initial insertion and the existing ON CONFLICT path.
    asyncio.run(pg_store.upsert_episodes("s", [replace(episode, leading="right")]))
    assert asyncio.run(pg_store.load_session("s"))[1][0].leading == "right"
    assert asyncio.run(pg_store.load_session("s", episode_limit=1))[1][0].leading == "right"
    assert asyncio.run(pg_store.search_episodes("s", np.ones(128)))[0][1].leading == "right"
    assert asyncio.run(pg_store.fetch_episodes_after_id("s", after_id=0, limit=5))[0]["leading"] == "right"
