import asyncio
from dataclasses import replace
import json

import pytest

from core.memory_answerability import AnswerabilityGate, OpenAIAnswerability, answerability_gate_from_environment
from core.llm_client import LLMConfig
from core.shared_memory import MemoryTrace, SharedMemory


class Checks:
    identity = "fake-support-v1"
    calls = 0
    input_tokens = output_tokens = 0
    served_model = "fake"

    async def evaluate(self, request):
        self.calls += 1
        return json.dumps({"decisions": [{"id": r["id"], "supported": "DENY" not in r["answer"],
                                          "quote": r["answer"] if "DENY" not in r["answer"] else ""}
                                         for r in request["records"]]})


def source(qid="source", answer="At least three characters.", **kwargs):
    return MemoryTrace("What is the minimum query length for Project Alpha?", answer,
                       qid=qid, seed_id="seed-" + qid, retrieval_scope="alpha", scope_aliases=("Alpha",), **kwargs)


def test_preparation_is_required_and_cache_binds_question_source_status_and_model():
    backend = Checks()
    memory = SharedMemory(answerability_gate=AnswerabilityGate(backend))
    memory.store(source())
    q = "What is the minimum query length for Alpha?"
    assert not memory.retrieve_related_traces(q) and not memory.get_context()
    asyncio.run(memory.prepare_retrieval(q))
    assert [t.qid for t in memory.retrieve_related_traces(q)] == ["source"]
    assert memory.evidence_for_seeds(["seed-source"], question=q)
    assert not memory.evidence_for_seeds(["seed-source"])
    asyncio.run(memory.prepare_retrieval(q))
    assert backend.calls == 1
    assert not memory.retrieve_related_traces(q + " Is it a maximum?")
    memory.past_qas[0] = replace(memory.past_qas[0], answer="Changed answer")
    assert not memory.retrieve_related_traces(q)
    asyncio.run(memory.prepare_retrieval(q))
    assert backend.calls == 2
    memory.past_qas[0] = replace(memory.past_qas[0], evidence_status="unverified")
    assert not memory.retrieve_related_traces(q)
    backend.identity = "different-model"
    assert not memory.retrieve_related_traces(q)
    assert asyncio.run(memory.prepare_retrieval(q))["answerability"]["status"] == "withheld"


@pytest.mark.parametrize("decisions", [
    [], [{"id": "other", "supported": True, "quote": "At least three characters."}],
    [{"id": "m0", "supported": "true", "quote": "At least three characters."}],
    [{"id": "m0", "supported": True, "quote": "At most three characters."}],
    [{"id": "m0", "supported": True, "quote": ""}],
    [{"id": "m0", "supported": True, "quote": " "}],
])
def test_malformed_or_ungrounded_decisions_cannot_approve(decisions):
    class Bad(Checks):
        async def evaluate(self, request):
            return json.dumps({"decisions": decisions})
    memory = SharedMemory(answerability_gate=AnswerabilityGate(Bad()))
    memory.store(source())
    q = "Alpha minimum query length"
    result = asyncio.run(memory.prepare_retrieval(q))
    assert result["answerability"]["status"] == "withheld"
    assert not memory.retrieve_related(q)


def test_provider_error_is_fail_closed_and_does_not_echo_private_error():
    class Broken(Checks):
        async def evaluate(self, request):
            raise RuntimeError("Bearer SECRET private answer")
    memory = SharedMemory(answerability_gate=AnswerabilityGate(Broken()))
    memory.store(source())
    result = asyncio.run(memory.prepare_retrieval("Alpha minimum query length"))
    assert "SECRET" not in str(result) and "private answer" not in str(result)
    assert not memory.retrieve_related("Alpha minimum query length")


def test_scope_expiry_and_visible_excerpt_limits_cannot_be_bypassed():
    backend = Checks()
    memory = SharedMemory(answerability_gate=AnswerabilityGate(backend))
    memory.store(source(valid_until=0))
    memory.store(source("long", answer="x" * 641))
    for q in ["Alpha query length", "Unknown query length", "Alpha " + "x" * 6000]:
        asyncio.run(memory.prepare_retrieval(q))
        assert not memory.retrieve_related_traces(q)
    assert backend.calls == 0


def test_actual_openai_payload_is_bounded_and_usage_is_recorded():
    backend = OpenAIAnswerability(LLMConfig("openai", "old", "NOT_A_KEY"), "gpt-6-luna")
    captured = []
    async def post(url, payload, headers, **kwargs):
        captured.append((payload, kwargs))
        return {"model": "gpt-6-luna", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [{"finish_reason": "stop", "message": {"content": '{"decisions":[]}'}}]}
    backend.client._post_json = post
    assert asyncio.run(backend.evaluate({"question": "Q", "records": []})) == '{"decisions":[]}'
    assert backend.calls == 1 and backend.input_tokens == 10 and backend.output_tokens == 5
    payload, kwargs = captured[0]
    assert payload["max_completion_tokens"] == 4096 and "temperature" not in payload
    assert payload["response_format"] == {"type": "json_object"}
    assert kwargs["timeout_seconds"] <= 20 and not backend.config.auto_continue


def test_gate_is_opt_in_and_uses_an_explicit_model(monkeypatch):
    monkeypatch.delenv("DUALBRAIN_MEMORY_ANSWERABILITY_MODEL", raising=False)
    assert answerability_gate_from_environment() is None
    monkeypatch.setenv("DUALBRAIN_MEMORY_ANSWERABILITY_MODEL", "support-model")
    gate = answerability_gate_from_environment(LLMConfig("openai", "chat-model", "NOT_A_KEY"))
    assert gate.backend.config.model == "support-model"
