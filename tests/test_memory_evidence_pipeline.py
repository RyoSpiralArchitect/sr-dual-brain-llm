import asyncio
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))
from engine_stdio import _right_worker  # noqa: E402
from run_server import right_worker  # noqa: E402
from core.callosum import Callosum  # noqa: E402
from core.llm_client import LLMConfig  # noqa: E402
from core.models import RightBrainModel  # noqa: E402
from core.postgres_state_store import PostgresStateStore  # noqa: E402
from core.shared_memory import MemoryTrace, SharedMemory  # noqa: E402


@pytest.mark.parametrize("worker_fn", [_right_worker, right_worker])
@pytest.mark.parametrize("request_type", ["ASK_DETAIL", "ASK_CRITIC"])
@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_both_workers_deliver_final_qa_and_provenance_to_actual_model_prompt(worker_fn, request_type, guarded, complete):
    prompts = []

    class CaptureClient:
        async def complete(self, prompt, **kwargs):
            prompts.append((prompt, kwargs["system"]))
            return '{"verdict":"ok","issues":[],"fixes":[],"confidence_r":0.8}'

    model = RightBrainModel(LLMConfig("openai", "fixture", "NOT_A_KEY"))
    model._llm_client = CaptureClient()
    from core.memory_answerability import AnswerabilityGate
    from test_memory_answerability import Checks
    memory = SharedMemory(answerability_gate=AnswerabilityGate(Checks()) if guarded else None,
                          conversation_context=complete)
    answer = "The recorded retention is 48 hours."
    if complete:
        answer = "General policy. " * 50 + " Late exception: " + answer
    memory.store(MemoryTrace("Earlier Project Helix question", answer,
                             qid="original-qa", seed_id="seed", retrieval_scope="helix", scope_aliases=("Helix",)))
    if guarded:
        memory.store(MemoryTrace("Earlier Project Helix question", "DENY-MARKER wrong facet",
                                 qid="denied-qa", seed_id="denied-seed", retrieval_scope="helix", scope_aliases=("Helix",)))
        asyncio.run(memory.prepare_retrieval("What did the earlier Helix record say?"))

    async def run():
        callosum = Callosum(slot_ms=0)
        task = asyncio.create_task(worker_fn(callosum, memory, model))
        try:
            return await callosum.ask_detail({
                "qid": "new-question", "type": request_type, "question": "What did the earlier Helix record say?",
                "draft": "Unknown", "draft_sum": "Unknown", "context": "Base context",
                "memory_evidence": memory.evidence_for_seeds(["seed", "denied-seed"], question="What did the earlier Helix record say?"),
                "memory_evidence_format": "complete_v1" if complete else "legacy",
            }, timeout_ms=1000)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    response = asyncio.run(run())
    assert response["qid"] == "new-question" and len(prompts) == 1
    prompt, system = prompts[0]
    assert "48 hours" in system and '"source_qid": "original-qa"' in system
    assert '"status": "unverified"' in system and "not confirmed facts" in system
    assert '"retrieval_scope": "helix"' in system
    assert "Base context" in system and "NOT_A_KEY" not in prompt + system
    assert "DENY-MARKER" not in prompt + system
    if complete:
        assert answer in system and '"truncated": false' in system


def test_postgres_metadata_write_load_and_legacy_defaults_without_database(monkeypatch):
    class Cursor:
        def __init__(self):
            self.sql = ""
            self.rows = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def execute(self, sql, params=()):
            self.sql = sql
            if "INSERT INTO srdb_shared_memory" in sql:
                self.rows.append(params[1:])

        def fetchall(self):
            return self.rows if "FROM srdb_shared_memory" in self.sql else []

    cursor = Cursor()

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def cursor(self):
            return cursor

        def commit(self):
            pass

    store = PostgresStateStore("NOT_A_REAL_DSN")
    monkeypatch.setattr(store, "_connect", Connection)
    traces = [MemoryTrace("Q", "Final answer", qid="original", seed_id="seed",
                           evidence_status="superseded", valid_until=50,
                           retrieval_scope="fixture", scope_aliases=("Fixture", "フィクスチャ"))]
    store._append_memory_traces_sync("session", traces, None)
    loaded, episodes = store._load_session_sync("session", None, None)
    assert loaded == traces and not episodes
    loaded_limited, _ = store._load_session_sync("session", 10, None)
    assert loaded_limited == traces
    cursor.rows = [("old", "Q", "A", 10, '[]', '[]', '{}')]
    loaded, _ = store._load_session_sync("session", None, None)
    assert loaded[0].seed_id is None and loaded[0].evidence_status == "unverified"
    assert loaded[0].retrieval_scope is None and not loaded[0].scope_aliases


@pytest.mark.parametrize("worker_fn", [_right_worker, right_worker])
def test_missing_context_preserves_legacy_memory_lookup_but_explicit_empty_is_gated(worker_fn):
    model = RightBrainModel()
    model._llm_client = None
    memory = SharedMemory()
    memory.store({"Q": "incident records", "A": "lookup-sentinel"})

    async def run():
        callosum = Callosum(slot_ms=0)
        task = asyncio.create_task(worker_fn(callosum, memory, model))
        try:
            base = {"type": "ASK_DETAIL", "question": "incident records", "draft_sum": "draft"}
            missing = await callosum.ask_detail({**base, "qid": "missing"}, timeout_ms=1500)
            empty = await callosum.ask_detail({**base, "qid": "empty", "context": ""}, timeout_ms=1500)
            return missing, empty
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    missing, empty = asyncio.run(run())
    assert "lookup-sentinel" in missing["notes_sum"]
    assert "lookup-sentinel" not in empty["notes_sum"]


def test_optional_vector_failure_does_not_abort_required_metadata_migration(monkeypatch):
    class TransactionCursor:
        aborted = False

        def __init__(self):
            self.statements = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def execute(self, sql, _params=()):
            self.statements.append(sql)
            if "ROLLBACK TO SAVEPOINT" in sql:
                self.aborted = False
            elif self.aborted:
                raise RuntimeError("transaction aborted")
            elif "CREATE EXTENSION" in sql:
                self.aborted = True
                raise RuntimeError("optional extension unavailable")

    cursor = TransactionCursor()

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def cursor(self):
            return cursor

        def commit(self):
            assert not cursor.aborted

    store = PostgresStateStore("NOT_A_REAL_DSN")
    monkeypatch.setattr(store, "_connect", Connection)
    store._ensure_schema_sync()
    store._ensure_schema_sync()
    assert store._has_pgvector is False
    assert sum("ADD COLUMN IF NOT EXISTS evidence_metadata" in sql for sql in cursor.statements) == 2
    assert sum("ROLLBACK TO SAVEPOINT" in sql for sql in cursor.statements) == 2
