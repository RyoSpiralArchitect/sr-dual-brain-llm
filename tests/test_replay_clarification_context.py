"""Literal record choice survives the real EngineSession generation inputs."""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))

from core.llm_client import LLMClient, LLMConfig
from core.memory_answerability import AnswerabilityGate
from core.shared_memory import MemoryTrace
from core.thalamic_reticular_nucleus import (
    ACC_CONSULT_PROMOTION, ACC_SYSTEM2_PROMOTION, ACC_TEMPERATURE_ADJUSTMENT, DEFAULT_REGISTRY,
)
from engine_stdio import EngineSession, _handle_process
from offline_provider_guard import forbid_provider_network
from test_dual_brain import AlwaysConsultPolicy


REPLAY = "Repeat Project Cedar recorded answer verbatim."
CONDITION = "Conversation missing-evidence condition:\n"
FIRST = "CEDAR_FIRST_RECORD: retain the item for 17 days, except archived items."
SECOND = "CEDAR_SECOND_RECORD: review is owned by the operations team."
OTHER_CONTEXT = "Caller-permitted context: use plain English."


def source(qid="first", *, answer=FIRST):
    return MemoryTrace("Project Cedar retained policy?", answer, qid=qid,
                       retrieval_scope="Cedar", scope_aliases=("Cedar",), seed_id="seed-" + qid)


class ApproveAll:
    identity = "replay-context-fake-support-v1"

    def __init__(self):
        self.requests = []

    async def evaluate(self, request):
        self.requests.append(deepcopy(request))
        return json.dumps({"decisions": [
            {"id": record["id"], "supported": True, "quote": record["answer"]}
            for record in request["records"]
        ]})


def packets(text):
    return [json.loads(part.splitlines()[0]) for part in text.split(CONDITION)[1:]]


class ScriptedClient(LLMClient):
    """Capture actual HTTP-shaped prompts; the transport stays entirely local."""
    def __init__(self, role, calls, *, revise=False, draft_barrier=None, critic_barrier=None):
        super().__init__(LLMConfig("openai", "fake-replay-context", "FAKE_NO_CREDENTIAL",
            api_base="https://fake.invalid/v1", auto_continue=False, max_continuations=0))
        self.role, self.calls = role, calls
        self.revise, self.draft_barrier = revise, draft_barrier
        self.critic_barrier = critic_barrier
        self.critic_calls = 0

    async def _post_json(self, _url, payload, _headers, **_kwargs):
        messages = deepcopy(payload["messages"])
        system = messages[0]["content"]
        prompt = "\n\n".join(message["content"] for message in messages[1:])
        combined = system + "\n\n" + prompt
        kind = ("critic" if self.role == "right" and "rigorous reviewer (critic)" in system
                else "detail" if self.role == "right"
                else "integration" if "Internal collaborator notes (do not output directly):" in prompt
                else "draft")
        self.calls.append({"role": self.role, "kind": kind, "system": system, "prompt": prompt})
        if kind == "critic":
            self.critic_calls += 1
            needs_revision = self.revise and self.critic_calls == 1
            answer = json.dumps({"verdict": "issues" if needs_revision else "ok",
                "issues": ["The draft must distinguish project identity from record selection."] if needs_revision else [],
                "fixes": ["Preserve the unresolved request and ask which complete record is intended."] if needs_revision else []})
        elif kind == "detail":
            answer = "The project identity is already resolved. The complete record to quote still needs an explicit selection."
        elif (conditions := packets(combined)):
            answer = {
                "clarify_record": "Which complete recorded answer do you mean?",
                "clarify_reference": "Which project does the recorded-answer request concern?",
                "request_current_record": "Please supply a current complete record for this replay request.",
                "preserve_unknown": "Support for the requested recorded answer could not be established.",
            }[conditions[-1]["action"]]
        elif kind == "integration":
            answer = prompt.split("Draft answer:\n", 1)[1].split("\n\nInternal collaborator notes", 1)[0]
        elif "Recorded-answer replay (historical quotation" in combined:
            encoded = next(line for line in combined.splitlines() if line.startswith('{"source_question":'))
            answer = json.loads(encoded)["recorded_answer"]
        else:
            answer = "The supplied records can be summarized as unverified conversation-recorded information."
        if kind == "draft" and self.draft_barrier is not None:
            entered, release = self.draft_barrier
            entered.set()
            await release.wait()
        if kind == "critic" and self.critic_calls == 1 and self.critic_barrier is not None:
            entered, release = self.critic_barrier
            entered.set()
            await release.wait()
        return {"choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    async def _openai_style_stream_call(self, messages, *, on_delta, **kwargs):
        response = await self._post_json("unused", {"messages": messages}, {})
        text = response["choices"][0]["message"]["content"]
        if on_delta is not None:
            result = on_delta(text)
            if asyncio.iscoroutine(result):
                await result
        return text, "stop"


@pytest.fixture(autouse=True)
def offline_environment():
    environment = {
        "DUALBRAIN_MEMORY_CONVERSATION_CONTEXT": "1",
        "DUALBRAIN_SYSTEM2_PRIORITY_ON": "balanced",
        "DUALBRAIN_SYSTEM2_ROUND_TARGET_MIN": "2",
        "DUALBRAIN_SYSTEM2_LOW_SIGNAL_FILTER": "0",
    }
    for module in (ACC_CONSULT_PROMOTION, ACC_SYSTEM2_PROMOTION, ACC_TEMPERATURE_ADJUSTMENT):
        environment[DEFAULT_REGISTRY[module].mode_env] = "off"
    with patch.dict(os.environ, environment, clear=True), forbid_provider_network() as audit:
        yield audit
        assert audit.blocked_requests == 0


@asynccontextmanager
async def engine(calls, *, revise=False, draft_barrier=None, critic_barrier=None):
    session = await EngineSession.create("replay-clarification-context")
    session.memory.answerability_gate = AnswerabilityGate(ApproveAll())
    session.left._llm_client = ScriptedClient("left", calls, draft_barrier=draft_barrier)
    session.right._llm_client = ScriptedClient("right", calls, revise=revise, critic_barrier=critic_barrier)
    session.left.uses_external_llm = session.right.uses_external_llm = True
    session.controller.policy = AlwaysConsultPolicy()
    try:
        yield session
    finally:
        await session.close()


async def process(session, tmp_path, mode, question=REPLAY, **options):
    return await _handle_process(session, {"question": question, "qid": "probe", "leading_brain": "left",
        "system2_mode": mode, "executive_mode": "off", "executive_observer_mode": "off", **options},
        blob_root=tmp_path, blob_index={}, vision_cache={}, vision_cache_limit=0)


def assert_unbound_input(call, *, action="clarify_record"):
    conditions = packets(call["system"] + "\n" + call["prompt"])
    assert conditions and all(packet["action"] == action for packet in conditions)
    assert all(packet["operation"] == "recorded_answer_replay" for packet in conditions)
    assert all(packet["parameter_status"] == "not_applicable" for packet in conditions)
    assert "Prior draft text, reviewer notes and earlier quotations do not select or approve a record." in (
        call["system"] + call["prompt"])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "on"])
@pytest.mark.parametrize("unrelated_context", [False, True])
async def test_approved_ambiguous_replay_stays_explicit_in_draft_consult_and_revision(
        tmp_path, monkeypatch, mode, unrelated_context):
    calls = []
    async with engine(calls, revise=mode == "on") as session:
        records = [source(), source("second", answer=SECOND)]
        for record in records:
            session.memory.store(record)
        gate = session.memory.answerability_gate
        # Preserve the old ordinary-positive state for this exact request. It
        # must not silently supply a selection for a whole-record replay.
        await gate.prepare(REPLAY, [session.memory._answerability_source(record) for record in records])
        assert all(gate.allows(REPLAY, session.memory._answerability_source(record)) for record in records)
        if unrelated_context:
            retrieve = session.memory.retrieve_related

            def retain_other_context(question, n=3):
                return retrieve(question, n) or OTHER_CONTEXT

            monkeypatch.setattr(session.memory, "retrieve_related", retain_other_context)
        result = await process(session, tmp_path, mode)
        assert result["answer"] == "Which complete recorded answer do you mean?"
        assert len(gate.backend.requests) == 1  # Only the explicit ordinary-cache preparation above.
        assert session.memory.replay_resolution(REPLAY).binding_status == "ambiguous_record"
    assert [call["kind"] for call in calls] == (
        ["draft", "detail", "integration"] if mode == "off"
        else ["draft", "critic", "integration", "critic"])
    for call in calls:
        assert_unbound_input(call)
        assert FIRST not in call["system"] + call["prompt"]
        assert SECOND not in call["system"] + call["prompt"]
    if unrelated_context:
        assert OTHER_CONTEXT in calls[0]["system"]
        assert OTHER_CONTEXT in calls[1]["system"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "on"])
@pytest.mark.parametrize("change", ["second_record", "expire"])
async def test_replay_binding_is_rechecked_after_awaited_draft_without_new_calls(tmp_path, mode, change):
    calls = []
    entered, release = asyncio.Event(), asyncio.Event()
    async with engine(calls, revise=mode == "on", draft_barrier=(entered, release)) as session:
        session.memory.store(source())
        task = asyncio.create_task(process(session, tmp_path, mode))
        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
            assert FIRST in calls[0]["system"]
            if change == "second_record":
                session.memory.store(source("second", answer=SECOND))
            else:
                session.memory.past_qas[0] = replace(session.memory.past_qas[0], valid_until=0)
            release.set()
            result = await asyncio.wait_for(task, timeout=5)
            assert len(session.memory.answerability_gate.backend.requests) == 1
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    expected_action = "clarify_record" if change == "second_record" else "clarify_reference"
    assert [call["kind"] for call in calls] == (
        ["draft", "detail", "integration"] if mode == "off"
        else ["draft", "critic", "integration", "critic"])
    for call in calls[1:]:
        assert_unbound_input(call, action=expected_action)
        # Prior generated text remains a labeled draft, not current evidence.
        assert FIRST not in call["system"] and SECOND not in call["system"] + call["prompt"]
        if call["kind"] == "integration":
            notes = call["prompt"].split("Internal collaborator notes (do not output directly):", 1)[1]
            assert FIRST not in notes
    assert FIRST not in result["answer"] and SECOND not in result["answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "on"])
async def test_unique_approved_literal_record_remains_exact_without_clarification(tmp_path, mode):
    calls = []
    async with engine(calls) as session:
        session.memory.store(source())
        result = await process(session, tmp_path, mode)
        assert result["answer"] == FIRST
        assert len(session.memory.answerability_gate.backend.requests) == 1
        assert session.memory.answerability_gate.backend.requests[0]["operation"]["kind"] == "recorded_answer_replay"
    assert FIRST in calls[0]["system"] and FIRST in calls[1]["system"]
    assert all(not packets(call["system"] + call["prompt"]) for call in calls)
    assert "historical quotation, not independent factual verification" in calls[0]["system"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "on"])
async def test_ordinary_semantic_multirecord_context_is_preserved(tmp_path, mode):
    calls = []
    question = "Summarize Project Cedar retained policy records."
    async with engine(calls) as session:
        session.memory.store(source())
        session.memory.store(source("second", answer=SECOND))
        await process(session, tmp_path, mode, question=question)
        assert not session.memory.replay_resolution(question).requested
        assert len(session.memory.retrieve_related_traces(question)) == 2
    for call in calls[:2]:
        assert FIRST in call["system"] and SECOND in call["system"]
        assert "unverified answers are not confirmed facts" in call["system"]
    assert all(not packets(call["system"] + call["prompt"]) for call in calls)


@pytest.mark.asyncio
async def test_approved_context_is_rechecked_before_initial_generation_after_director_stage(tmp_path, monkeypatch):
    calls = []
    async with engine(calls) as session:
        session.memory.store(source())
        director_stage = session.controller._run_pre_draft_director_stage

        async def change_after_context_was_prepared(**kwargs):
            result = await director_stage(**kwargs)
            assert FIRST in result.context_frame.context
            session.memory.store(source("second", answer=SECOND))
            return result

        monkeypatch.setattr(session.controller, "_run_pre_draft_director_stage", change_after_context_was_prepared)
        result = await process(session, tmp_path, "off")
        assert result["answer"] == "Which complete recorded answer do you mean?"
        assert len(session.memory.answerability_gate.backend.requests) == 1
    assert [call["kind"] for call in calls] == ["draft", "detail", "integration"]
    for call in calls:
        assert_unbound_input(call)
        assert FIRST not in call["system"] + call["prompt"]
        assert SECOND not in call["system"] + call["prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["second_record", "expire"])
async def test_copied_critic_request_revalidates_seed_evidence_after_prior_review_await(tmp_path, monkeypatch, change):
    calls, requests = [], []
    entered, release = asyncio.Event(), asyncio.Event()
    async with engine(calls, revise=True, critic_barrier=(entered, release)) as session:
        session.memory.store(source())
        build = session.controller._build_consult_request
        ask = session.callosum.ask_detail

        def include_current_seed_evidence(**kwargs):
            plan = build(**kwargs)
            evidence = session.memory.evidence_for_seeds(["seed-first"], question=REPLAY)
            assert evidence and evidence[0]["answer"] == FIRST
            plan.payload.update(memory_evidence=evidence, memory_evidence_format="complete_v1")
            return plan

        async def record_request(payload, **kwargs):
            requests.append(deepcopy(payload))
            return await ask(payload, **kwargs)

        monkeypatch.setattr(session.controller, "_build_consult_request", include_current_seed_evidence)
        monkeypatch.setattr(session.callosum, "ask_detail", record_request)
        task = asyncio.create_task(process(session, tmp_path, "on"))
        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
            assert requests[0]["memory_evidence"][0]["answer"] == FIRST
            assert "Historical memory evidence" in calls[1]["system"]
            if change == "second_record":
                session.memory.store(source("second", answer=SECOND))
            else:
                session.memory.past_qas[0] = replace(session.memory.past_qas[0], valid_until=0)
            release.set()
            result = await asyncio.wait_for(task, timeout=5)
            assert len(session.memory.answerability_gate.backend.requests) == 1
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    expected_action = "clarify_record" if change == "second_record" else "clarify_reference"
    assert FIRST not in result["answer"] and SECOND not in result["answer"]
    assert [call["kind"] for call in calls] == ["draft", "critic", "integration", "critic"]
    assert len(requests) == 2 and "memory_evidence" not in requests[1]
    assert FIRST not in requests[1]["context"] and SECOND not in requests[1]["context"]
    for call in calls[2:]:
        assert_unbound_input(call, action=expected_action)
        assert FIRST not in call["system"] and SECOND not in call["system"] + call["prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "on"])
async def test_existing_direct_fallback_rechecks_binding_after_failed_consultation(tmp_path, monkeypatch, mode):
    calls, requests = [], []
    async with engine(calls, revise=mode == "on") as session:
        session.memory.store(source())
        ask = session.callosum.ask_detail

        async def fail_initial_request_after_source_changes(payload, **kwargs):
            requests.append(deepcopy(payload))
            if len(requests) == 1:
                session.memory.store(source("second", answer=SECOND))
                return {"error": "ScriptedConsultationFailure"}
            return await ask(payload, **kwargs)

        monkeypatch.setattr(session.callosum, "ask_detail", fail_initial_request_after_source_changes)
        result = await process(session, tmp_path, mode)
        assert result["answer"] == "Which complete recorded answer do you mean?"
        assert len(session.memory.answerability_gate.backend.requests) == 1
    assert [call["kind"] for call in calls] == (
        ["draft", "detail", "integration"] if mode == "off"
        else ["draft", "critic", "integration", "critic"])
    assert len(requests) == (1 if mode == "off" else 2)
    for call in calls[1:]:
        assert_unbound_input(call)
        assert FIRST not in call["system"] and SECOND not in call["system"] + call["prompt"]
