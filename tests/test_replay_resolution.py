"""Requested whole-record replay is independent of its current exact binding."""
import asyncio
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from core.memory_answerability import AnswerabilityGate
from core.shared_memory import MemoryTrace, SharedMemory


QUESTION = "Repeat Project Rowan recorded answer verbatim."
DECLARATION = {"id": "rowan", "aliases": ["Rowan"], "remember": False}


class Approvals:
    identity = "replay-resolution-fixed-approval-v1"

    def __init__(self, during_check=None):
        self.requests = []
        self.during_check = during_check

    async def evaluate(self, request):
        self.requests.append(request)
        if self.during_check is not None:
            self.during_check()
        return json.dumps({"decisions": [{"id": record["id"], "supported": True,
                                         "quote": record["answer"]}
                                        for record in request["records"]]})


def source(qid="first", **updates):
    trace = MemoryTrace("Project Rowan retention?", "PRIVATE_ROWAN_19_DAYS", qid=qid,
                        seed_id=f"seed-{qid}", retrieval_scope="rowan", scope_aliases=("Rowan",))
    return replace(trace, **updates)


def memory(*traces, gate=True, conversation=True):
    result = SharedMemory(conversation_context=conversation,
                          answerability_gate=AnswerabilityGate(Approvals()) if gate else None)
    for trace in traces:
        result.store(trace)
    return result


def test_bound_resolution_is_immutable_local_binding_without_approval():
    m = memory(source())
    m.begin_turn(QUESTION, "probe")
    resolution = m.replay_resolution(QUESTION)
    assert resolution.requested and resolution.binding_status == "bound"
    assert resolution.intent == m.replay_intent(QUESTION)
    assert resolution.intent.source_qid == "first"
    assert not m.retrieve_related_traces(QUESTION)  # Binding alone never approves the source.
    assert not m.answerability_gate.backend.requests
    with pytest.raises(FrozenInstanceError):
        resolution.binding_status = "not_requested"
    m.store(source("second", answer="PRIVATE_SECOND_RECORD"))
    assert resolution.binding_status == "bound"  # A value object, never mutable cached state.
    assert m.replay_resolution(QUESTION).binding_status == "ambiguous_record"
    assert m.replay_intent(QUESTION) is None


@pytest.mark.parametrize("followup", [
    "Repeat  that recorded answer verbatim.",
    "Please\tquote that stored text exactly.",
    "Ｒｅｐｅａｔ　ｔｈａｔ　ｒｅｃｏｒｄｅｄ　ａｎｓｗｅｒ　ｖｅｒｂａｔｉｍ。",
    "その記録をそのまま再掲して",
])
@pytest.mark.parametrize("caller_declared", [False, True])
def test_normalized_pronoun_requests_remain_requested_without_a_unique_record(followup, caller_declared):
    m = memory(source(), source("second", answer="PRIVATE_SECOND_RECORD"))
    m.begin_turn("Project Rowan retention?", "focus")
    m.begin_turn(followup, "probe", DECLARATION if caller_declared else None)
    resolution = m.replay_resolution(followup)
    assert resolution.requested and resolution.binding_status == "ambiguous_record"
    assert resolution.intent is None and m.replay_intent(followup) is None
    condition = m.missing_evidence(followup)
    assert condition.operation == "recorded_answer_replay" and condition.action == "clarify_record"
    assert condition.source_availability == "current_unselected" and condition.source_approval == "unknown"
    assert not m.answerability_gate.backend.requests


@pytest.mark.parametrize("state", ["missing", "lost_focus", "ambiguous_alias"])
def test_requested_state_survives_missing_or_unresolved_reference(state):
    m = memory()
    question = QUESTION
    metadata = DECLARATION
    expected = "missing_record"
    if state == "lost_focus":
        question, metadata, expected = "Repeat that recorded answer verbatim.", None, "unresolved_reference"
    elif state == "ambiguous_alias":
        m.store(source(scope_aliases=("Shared",)))
        m.store(source("other", retrieval_scope="birch", scope_aliases=("Shared",)))
        question, metadata, expected = "Repeat Project Shared recorded answer verbatim.", None, "unresolved_reference"
    m.begin_turn(question, "probe", metadata)
    resolution = m.replay_resolution(question)
    assert resolution.requested and resolution.binding_status == expected and resolution.intent is None
    condition = m.missing_evidence(question)
    assert condition.operation == "recorded_answer_replay"
    assert condition.action == ("request_current_record" if state == "missing" else "clarify_reference")
    assert not m.retrieve_related(question)


@pytest.mark.parametrize("failure", ["gate_disabled", "too_long", "sanitized", "missing_qid", "duplicate_qid"])
def test_unique_but_unavailable_binding_preserves_existing_source_uncertainty(failure):
    trace = source()
    if failure == "too_long":
        trace = replace(trace, answer="x" * 2049)
    elif failure == "sanitized":
        trace = replace(trace, answer="PRIVATE_ROWAN_19_DAYS\ntelemetry (raw) omitted content")
    elif failure == "missing_qid":
        trace = replace(trace, qid=None)
    m = memory(trace, gate=failure != "gate_disabled")
    if failure == "duplicate_qid":
        m.store(replace(trace, retrieval_scope="birch", scope_aliases=("Birch",), valid_until=0))
    m.begin_turn(QUESTION, "probe")
    resolution = m.replay_resolution(QUESTION)
    assert resolution.requested and resolution.binding_status == "unavailable_binding"
    assert resolution.intent is None and not m.retrieve_related(QUESTION)
    assert not m.evidence_for_seeds([trace.seed_id], question=QUESTION)
    assert not m.evidence_for_seeds([trace.seed_id])
    assert not m.get_context()
    condition = m.missing_evidence(QUESTION)
    assert condition.action == "preserve_unknown" and condition.source_approval == "unknown"
    assert condition.source_availability == "current_unapproved"


@pytest.mark.parametrize("question", [
    "What is Project Rowan retention?",
    "Summarize Project Rowan recorded answer.",
    "Repeat Project Rowan recorded answer verbatim and verify it.",
    "Repeat Project Rowan recorded answer verbatim. Then calculate the limit.",
    "Project Rowanの記録をそのまま再掲して、比較して",
    "Compare Project Rowan and Project Birch records.",
])
def test_other_requests_do_not_acquire_literal_replay_state(question):
    m = memory(source(), source("second", answer="PRIVATE_SECOND_RECORD"))
    m.begin_turn(question, "probe")
    resolution = m.replay_resolution(question)
    assert not resolution.requested and resolution.binding_status == "not_requested"
    assert resolution.intent is None and m.replay_intent(question) is None


@pytest.mark.parametrize("excluded", ["primary_write", "legacy"])
def test_primary_write_and_nonconversation_routes_remain_outside_replay_resolution(excluded):
    m = memory(source(), conversation=excluded != "legacy")
    m.begin_turn(QUESTION, "probe", {**DECLARATION, "remember": True} if excluded == "primary_write" else None)
    resolution = m.replay_resolution(QUESTION)
    assert not resolution.requested and resolution.intent is None
    assert m.answerability_candidates(QUESTION)


def test_binding_is_revalidated_when_support_await_adds_another_current_record():
    m = memory(source())
    m.begin_turn(QUESTION, "probe")
    before = m.replay_resolution(QUESTION)
    gate = m.answerability_gate
    gate.backend.during_check = lambda: m.store(source("second", answer="PRIVATE_SECOND_RECORD"))
    prepared = asyncio.run(m.prepare_retrieval(QUESTION))
    assert prepared["answerability"]["approved"] == 1  # This is the completed old snapshot's decision.
    assert gate.decision(QUESTION, m._answerability_source(m.past_qas[0]), intent=before.intent) is True
    after = m.replay_resolution(QUESTION)
    assert after.requested and after.binding_status == "ambiguous_record" and after.intent is None
    assert not m.retrieve_related_traces(QUESTION) and not m.retrieve_related(QUESTION)
    assert not m.evidence_for_seeds(["seed-first", "seed-second"], question=QUESTION)
    assert m.missing_evidence(QUESTION).action == "clarify_record"
    assert len(gate.backend.requests) == 1
