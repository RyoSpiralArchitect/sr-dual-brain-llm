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
    memory.past_qas[0] = replace(memory.past_qas[0], evidence_status="verified")
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
    identity = backend.identity
    backend.config.model = "different-model"
    assert backend.identity != identity


def test_gate_is_opt_in_and_uses_an_explicit_model(monkeypatch):
    monkeypatch.delenv("DUALBRAIN_MEMORY_ANSWERABILITY_MODEL", raising=False)
    assert answerability_gate_from_environment() is None
    monkeypatch.setenv("DUALBRAIN_MEMORY_ANSWERABILITY_MODEL", "support-model")
    gate = answerability_gate_from_environment(LLMConfig("openai", "chat-model", "NOT_A_KEY"))
    assert gate.backend.config.model == "support-model"


def test_widening_recovers_independent_fact_pruned_by_relative_similarity():
    from core.memory_retrieval import QuestionSemanticIndex
    semantic = Checks()
    index = QuestionSemanticIndex(semantic, scope_min_similarity=.25)
    memory = SharedMemory(semantic_index=index, answerability_gate=AnswerabilityGate(Checks()))
    first = replace(source("duration", "14 days"), question="Project Alpha duration")
    second = replace(source("currency", "EUR"), question="Project Alpha currency")
    memory.store(first)
    memory.store(second)
    query = "Project Alpha duration and currency"
    resolver = memory.scope_resolver()
    index.vectors.update({resolver.query(query).content: (1., 0.),
                          resolver.source(first.question).content: (1., 0.),
                          resolver.source(second.question).content: (.3, .9539392014169457)})
    assert [t.qid for t in memory._retrieve_related_traces(query)] == ["duration"]
    asyncio.run(memory.prepare_retrieval(query))
    assert {t.qid for t in memory.retrieve_related_traces(query)} == {"duration", "currency"}
    assert len(memory.evidence_for_seeds(["seed-duration", "seed-currency"], question=query)) == 2


def test_unannotated_seed_route_also_requires_support_and_query():
    memory = SharedMemory(answerability_gate=AnswerabilityGate(Checks()))
    memory.store(MemoryTrace("minimum input length", "DENY unrelated value", qid="x", seed_id="sx"))
    asyncio.run(memory.prepare_retrieval("minimum input length"))
    assert not memory.evidence_for_seeds(["sx"], question="minimum input length")
    assert not memory.evidence_for_seeds(["sx"])


def test_support_allowlist_blocks_identical_hash_seed_and_is_in_field_snapshot():
    from test_memory_retrieval import seed_field
    field, question, _ = seed_field()
    field.recall_evidence_allowlist = frozenset()
    summary = field.summary(field.analyse(question=question, draft=""))
    assert not summary.emergent_ideas
    assert "evidence_support_not_approved" in summary.harvest_attempts[0].failure_reasons
    assert field.memory_state()["recall_evidence_allowlist"] == []
    assert field.fork().memory_state() == field.memory_state()


def test_duplicate_json_fields_cannot_override_an_unsupported_decision():
    class Duplicate(Checks):
        async def evaluate(self, request):
            return '{"decisions":[{"id":"m0","supported":false,"supported":true,"quote":"At least three characters."}]}'
    memory = SharedMemory(answerability_gate=AnswerabilityGate(Duplicate()))
    memory.store(source())
    result = asyncio.run(memory.prepare_retrieval("Alpha minimum query length"))
    assert result["answerability"]["status"] == "withheld"
    assert not memory.retrieve_related_traces("Alpha minimum query length")


def test_sanitization_must_not_remove_a_condition_before_support_checking():
    backend = Checks()
    memory = SharedMemory(answerability_gate=AnswerabilityGate(backend))
    memory.store(source(answer="If the user is in Europe, use this policy.\nAt least three characters."))
    q = "Alpha minimum query length"
    result = asyncio.run(memory.prepare_retrieval(q))
    assert result["answerability"]["status"] == "empty" and backend.calls == 0
    assert not memory.retrieve_related_traces(q)
    assert not memory.evidence_for_seeds(["seed-source"], question=q)


def test_served_model_drift_permanently_invalidates_previous_approvals():
    backend = OpenAIAnswerability(LLMConfig("openai", "requested", "NOT_A_KEY"), "requested")
    models = iter(["served-v1", "served-v2"])
    async def post(*args, **kwargs):
        return {"model": next(models), "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"decisions": [
                    {"id": "m0", "supported": True, "quote": "At least three characters."}]})}}]}
    backend.client._post_json = post
    memory = SharedMemory(answerability_gate=AnswerabilityGate(backend))
    memory.store(source())
    q = "Alpha minimum query length"
    asyncio.run(memory.prepare_retrieval(q))
    assert memory.retrieve_related_traces(q)
    old_identity = backend.identity
    result = asyncio.run(memory.prepare_retrieval("Alpha minimum input length"))
    assert result["answerability"]["status"] == "withheld" and backend.identity != old_identity
    assert not memory.retrieve_related_traces(q)
    assert asyncio.run(memory.prepare_retrieval(q))["answerability"]["error_type"] == "IdentityChanged"
    assert backend.calls == 2


class RecordingChecks(Checks):
    def __init__(self):
        self.requests = []

    async def evaluate(self, request):
        self.requests.append(request)
        return await super().evaluate(request)


def visible(trace):
    return SharedMemory._answerability_source(trace)


def test_growing_candidates_reuses_positive_and_negative_checks_without_cutting_conditions():
    backend = RecordingChecks()
    gate = AnswerabilityGate(backend)
    memory = SharedMemory(answerability_gate=gate, conversation_context=True)
    memory.store(source("minimum"))
    memory.store(source("fee", answer="DENY: fees have not been supplied."))
    question = "Alpha minimum query length and regional policy"
    asyncio.run(memory.prepare_retrieval(question))
    long_answer = "The normal policy applies. " * 40 + "Except in Europe, use seven days."
    memory.store(source("region", answer=long_answer))
    result = asyncio.run(memory.prepare_retrieval(question))["answerability"]
    assert len(backend.requests) == 2
    assert backend.requests[-1]["records"] == [
        {"id": "m0", "question": source().question, "answer": long_answer}]
    assert result == {"status": "ready", "candidates": 3, "cached_candidates": 2,
                      "checked_candidates": 1, "approved": 2}
    assert {t.qid for t in memory.retrieve_related_traces(question)} == {"minimum", "region"}
    assert {r["source_qid"] for r in memory.evidence_for_seeds(
        ["seed-minimum", "seed-fee", "seed-region"], question=question)} == {"minimum", "region"}
    cached = asyncio.run(memory.prepare_retrieval(question))["answerability"]
    assert cached["status"] == "cached" and cached["checked_candidates"] == 0
    assert len(backend.requests) == 2


@pytest.mark.parametrize("change", [
    {"answer": "At least seven characters."}, {"question": "Alpha maximum query length?"},
    {"status": "verified"}, {"recorded_at": 1.}, {"valid_until": 9999999999.},
    {"retrieval_scope": "beta"}, {"qid": "new-id"}, {"seed_id": "new-seed"},
])
def test_changed_source_identity_rechecks_only_that_source(change):
    backend = RecordingChecks()
    gate = AnswerabilityGate(backend)
    stable, changed = visible(source("stable")), visible(source("changing"))
    asyncio.run(gate.prepare("Alpha minimum", [stable, changed]))
    revised = dict(changed, **change)
    result = asyncio.run(gate.prepare("Alpha minimum", [revised, stable]))
    assert result["cached_candidates"] == result["checked_candidates"] == 1
    assert backend.requests[-1]["records"] == [
        {"id": "m0", "question": revised["question"], "answer": revised["answer"]}]
    assert gate.allows("Alpha minimum", revised) and gate.allows("Alpha minimum", stable)
    assert not gate.allows("Alpha minimum, but is it a maximum?", stable)


@pytest.mark.parametrize("failure", ["transport", "quote", "coverage"])
def test_failed_incremental_batch_preserves_hits_but_approves_no_new_records(failure):
    backend = RecordingChecks()
    gate = AnswerabilityGate(backend)
    yes, no = visible(source("yes")), visible(source("no", answer="DENY unrelated"))
    fresh = [visible(source("fresh-1")), visible(source("fresh-2"))]
    asyncio.run(gate.prepare("Alpha minimum", [yes, no]))

    async def broken(request):
        assert len(request["records"]) == 2
        if failure == "transport":
            raise TimeoutError("SECRET private text")
        decisions = [{"id": "m0", "supported": True, "quote": fresh[0]["answer"]}]
        if failure == "quote":
            decisions.append({"id": "m1", "supported": True, "quote": "invented quote"})
        return json.dumps({"decisions": decisions})

    backend.evaluate = broken
    result = asyncio.run(gate.prepare("Alpha minimum", [yes, fresh[0], no, fresh[1]]))
    assert result["status"] == "withheld" and result["cached_candidates"] == 2
    assert "SECRET" not in str(result)
    assert gate.allows("Alpha minimum", yes) and not gate.allows("Alpha minimum", no)
    assert all(not gate.allows("Alpha minimum", s) for s in fresh)
    backend.evaluate = RecordingChecks.evaluate.__get__(backend)
    retried = asyncio.run(gate.prepare("Alpha minimum", [yes, fresh[0], no, fresh[1]]))
    assert retried["checked_candidates"] == 2 and retried["approved"] == 3


def test_model_change_during_incremental_check_invalidates_cache_hits_too():
    backend = RecordingChecks()
    gate = AnswerabilityGate(backend)
    old, fresh = visible(source("old")), visible(source("fresh"))
    asyncio.run(gate.prepare("Alpha minimum", [old]))

    async def changed(request):
        result = await RecordingChecks.evaluate(backend, request)
        backend.identity = "changed-model"
        return result

    backend.evaluate = changed
    result = asyncio.run(gate.prepare("Alpha minimum", [old, fresh]))
    assert result["status"] == "withheld"
    assert not gate.allows("Alpha minimum", old) and not gate.allows("Alpha minimum", fresh)


def test_incremental_check_survives_concurrent_lru_eviction_and_keeps_the_bound():
    async def scenario():
        backend = RecordingChecks()
        gate = AnswerabilityGate(backend, max_items=6)
        yes, no = visible(source("yes")), visible(source("no", answer="DENY unrelated"))
        fresh = visible(source("fresh"))
        await gate.prepare("Alpha minimum", [yes, no])
        started, release = asyncio.Event(), asyncio.Event()

        async def blocked(request):
            if request["question"] == "Alpha minimum":
                started.set()
                await release.wait()
            return await RecordingChecks.evaluate(backend, request)

        backend.evaluate = blocked
        task = asyncio.create_task(gate.prepare("Alpha minimum", [yes, fresh, no]))
        await started.wait()
        await gate.prepare("Other question", [visible(source(str(i))) for i in range(6)])
        assert not gate.allows("Alpha minimum", yes)
        release.set()
        result = await task
        assert result["status"] == "ready" and result["approved"] == 2
        assert gate.allows("Alpha minimum", yes) and gate.allows("Alpha minimum", fresh)
        assert not gate.allows("Alpha minimum", no) and len(gate.cache) == 6

    asyncio.run(scenario())


def test_cancelled_incremental_check_does_not_publish_pending_decisions():
    async def scenario():
        backend = RecordingChecks()
        gate = AnswerabilityGate(backend)
        old, fresh = visible(source("old")), visible(source("fresh"))
        await gate.prepare("Alpha minimum", [old])
        started = asyncio.Event()

        async def blocked(request):
            started.set()
            await asyncio.Event().wait()

        backend.evaluate = blocked
        task = asyncio.create_task(gate.prepare("Alpha minimum", [old, fresh]))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert gate.allows("Alpha minimum", old) and not gate.allows("Alpha minimum", fresh)

    asyncio.run(scenario())


def test_cached_support_cannot_revive_expired_or_superseded_memory(monkeypatch):
    backend = RecordingChecks()
    memory = SharedMemory(answerability_gate=AnswerabilityGate(backend))
    monkeypatch.setattr("core.shared_memory.time.time", lambda: 100.)
    memory.store(source("expires", valid_until=101.))
    memory.store(source("replaced"))
    question = "Alpha minimum query length"
    asyncio.run(memory.prepare_retrieval(question))
    assert {t.qid for t in memory.retrieve_related_traces(question)} == {"expires", "replaced"}
    monkeypatch.setattr("core.shared_memory.time.time", lambda: 102.)
    memory.past_qas[1] = replace(memory.past_qas[1], evidence_status="superseded")
    assert not memory.retrieve_related_traces(question)
    assert asyncio.run(memory.prepare_retrieval(question))["answerability"]["status"] == "empty"
    assert len(backend.requests) == 1
