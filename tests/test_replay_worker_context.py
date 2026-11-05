"""Recheck copied replay context at the real queued-worker consumer boundary."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))

from core.callosum import Callosum
from core.llm_client import LLMConfig
from core.memory_answerability import AnswerabilityGate
from core.models import RightBrainModel
from core.right_worker import right_worker
from core.shared_memory import MemoryTrace, SharedMemory
from offline_provider_guard import forbid_provider_network


REPLAY = "Repeat Project Cedar recorded answer verbatim."
SEMANTIC = "Summarize Project Cedar policy records."
FIRST = "WORKER_FIRST_RECORD: retain the item for 19 days, except archived items."
SECOND = "WORKER_SECOND_RECORD: backup responsibility belongs to the operations team."
OTHER_CONTEXT = "Caller context: use plain English."
CONDITION = "Conversation missing-evidence condition:\n"
KINDS = ["ASK_DETAIL", "ASK_CRITIC", "ASK_LEAD"]


class ApproveAll:
    identity = "queued-replay-fixed-approval-v1"

    def __init__(self):
        self.requests = []

    async def evaluate(self, request):
        self.requests.append(deepcopy(request))
        return json.dumps({"decisions": [
            {"id": record["id"], "supported": True, "quote": record["answer"]}
            for record in request["records"]
        ]})


class CaptureClient:
    """Capture actual RightBrainModel prompts with a fixed local completion."""
    def __init__(self, prompts):
        self.prompts = prompts

    async def complete(self, prompt, **kwargs):
        self.prompts.append({"prompt": prompt, "system": kwargs.get("system") or ""})
        return '{"verdict":"ok","issues":[],"fixes":[]}'


def source(qid="worker-first", answer=FIRST):
    return MemoryTrace("Project Cedar policy?", answer, qid=qid, seed_id=f"seed-{qid}",
                       retrieval_scope="Cedar", scope_aliases=("Cedar",))


def packets(text):
    return [json.loads(part.splitlines()[0]) for part in text.split(CONDITION)[1:]]


@pytest.fixture(autouse=True)
def offline_environment():
    with patch.dict(os.environ, {}, clear=True), forbid_provider_network() as audit:
        yield audit
        assert audit.blocked_requests == 0


async def prepared_request(kind, *, question=REPLAY, records=None):
    memory = SharedMemory(conversation_context=True, answerability_gate=AnswerabilityGate(ApproveAll()))
    records = [source()] if records is None else records
    for record in records:
        memory.store(record)
    memory.begin_turn(question, "queued-turn")
    await memory.prepare_retrieval(question)
    context = memory.retrieve_related(question)
    evidence = memory.evidence_for_seeds([record.seed_id for record in records], question=question)
    assert context and evidence
    return memory, {
        "type": kind, "qid": "queued-turn", "question": question,
        "draft": "Prior generated text without a selected record.",
        "draft_sum": "Prior generated text without a selected record.",
        # Replay context is the admitted source body. Ordinary semantic calls
        # retain their unrelated context under the existing worker contract.
        "context": (OTHER_CONTEXT + "\n\n" if question == SEMANTIC else "") + context,
        "memory_evidence": evidence, "memory_evidence_format": "complete_v1",
    }


async def consume_after_change(memory, request, change):
    """Admit the real copied packet, then release the real worker after mutation."""
    prompts = []
    model = RightBrainModel(LLMConfig("openai", "worker-context-fake", "FAKE_NO_CREDENTIAL",
                                     api_base="https://fake.invalid/v1", auto_continue=False))
    model._llm_client = CaptureClient(prompts)
    bus = Callosum(slot_ms=0)
    release = asyncio.Event()

    async def consume():
        await release.wait()
        await right_worker(bus, memory, model)

    worker = asyncio.create_task(consume())
    ask = asyncio.create_task(bus.ask_detail(request, timeout_ms=3000))
    before = deepcopy(request)
    try:
        await asyncio.wait_for(bus._available.wait(), timeout=1)
        assert bus.snapshot()["queued"] == 1 and bus.snapshot()["running"] == 0
        assert not prompts  # No consumer/model input has been created yet.
        if change == "second_record":
            memory.store(source("worker-second", SECOND))
        elif change == "expire":
            memory.past_qas[0] = replace(memory.past_qas[0], valid_until=0)
        else:
            assert change == "unchanged"
        release.set()
        response = await asyncio.wait_for(ask, timeout=3)
        assert response["qid"] == request["qid"] and "error" not in response
        await bus.wait_idle()
        assert bus.snapshot()["queued"] == bus.snapshot()["pending"] == bus.snapshot()["running"] == 0
    finally:
        release.set()
        await bus.close()
        await asyncio.gather(worker, ask, return_exceptions=True)
    assert request == before  # Consumer changes apply to its copied request only.
    assert len(prompts) == 1 and len(memory.answerability_gate.backend.requests) == 1
    return prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("change", ["second_record", "expire"])
async def test_queued_replay_revalidates_body_and_seed_evidence_before_model_dispatch(kind, change):
    memory, request = await prepared_request(kind)
    assert FIRST in request["context"] and request["memory_evidence"][0]["answer"] == FIRST
    captured = await consume_after_change(memory, request, change)
    combined = captured["system"] + "\n" + captured["prompt"]
    conditions = packets(combined)
    expected_action = "clarify_record" if change == "second_record" else "clarify_reference"
    assert conditions and all(condition["action"] == expected_action for condition in conditions)
    assert all(condition["operation"] == "recorded_answer_replay" for condition in conditions)
    assert FIRST not in combined and SECOND not in combined
    assert "worker-first" not in combined and "seed-worker-first" not in combined
    assert not memory.retrieve_related_traces(REPLAY)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_queued_unique_approved_replay_preserves_complete_body_and_provenance(kind):
    memory, request = await prepared_request(kind)
    captured = await consume_after_change(memory, request, "unchanged")
    combined = captured["system"] + "\n" + captured["prompt"]
    assert FIRST in captured["system"]
    assert '"source_qid": "worker-first"' in captured["system"]
    assert "historical quotation, not independent factual verification" in captured["system"]
    assert not packets(combined)
    assert memory.replay_intent(REPLAY) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_queued_semantic_multirecord_request_preserves_ordinary_evidence(kind):
    memory, request = await prepared_request(kind, question=SEMANTIC,
                                             records=[source(), source("worker-second", SECOND)])
    captured = await consume_after_change(memory, request, "unchanged")
    combined = captured["system"] + "\n" + captured["prompt"]
    assert FIRST in captured["system"] and SECOND in captured["system"]
    assert OTHER_CONTEXT in captured["system"]
    assert all(f'"source_qid": "{qid}"' in captured["system"] for qid in ("worker-first", "worker-second"))
    assert not packets(combined) and not memory.replay_resolution(SEMANTIC).requested
