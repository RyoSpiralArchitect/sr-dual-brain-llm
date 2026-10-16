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
def test_both_workers_deliver_final_qa_and_provenance_to_actual_model_prompt(worker_fn, request_type):
    prompts = []

    class CaptureClient:
        async def complete(self, prompt, **kwargs):
            prompts.append((prompt, kwargs["system"]))
            return '{"verdict":"ok","issues":[],"fixes":[],"confidence_r":0.8}'

    model = RightBrainModel(LLMConfig("openai", "fixture", "NOT_A_KEY"))
    model._llm_client = CaptureClient()
    memory = SharedMemory()
    memory.store(MemoryTrace("Earlier question", "The recorded retention is 48 hours.", qid="original-qa", seed_id="seed"))

    async def run():
        callosum = Callosum(slot_ms=0)
        task = asyncio.create_task(worker_fn(callosum, memory, model))
        try:
            return await callosum.ask_detail({
                "qid": "new-question", "type": request_type, "question": "What did the earlier record say?",
                "draft": "Unknown", "draft_sum": "Unknown", "context": "Base context",
                "memory_evidence": memory.evidence_for_seeds(["seed"]),
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
    assert "Base context" in system and "NOT_A_KEY" not in prompt + system


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
                           evidence_status="superseded", valid_until=50)]
    store._append_memory_traces_sync("session", traces, None)
    loaded, episodes = store._load_session_sync("session", None, None)
    assert loaded == traces and not episodes
    loaded_limited, _ = store._load_session_sync("session", 10, None)
    assert loaded_limited == traces
    cursor.rows = [("old", "Q", "A", 10, '[]', '[]', '{}')]
    loaded, _ = store._load_session_sync("session", None, None)
    assert loaded[0].seed_id is None and loaded[0].evidence_status == "unverified"
