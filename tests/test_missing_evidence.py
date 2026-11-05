"""Explicit missing conditions through memory, prefrontal context and both brains."""
import asyncio
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
from core.prefrontal_cortex import PrefrontalCortex
from core.shared_memory import MemoryTrace, SharedMemory
from engine_stdio import EngineSession, _handle_process


DECLARATION = {"id": "Cedar", "aliases": ["Cedar"], "remember": False}


class Decisions:
    identity = "missing-evidence-support-v1"

    def __init__(self, approve=False):
        self.approve, self.requests = approve, []

    async def evaluate(self, request):
        self.requests.append(request)
        return json.dumps({"decisions": [{"id": row["id"], "supported": self.approve,
            "quote": row["answer"] if self.approve else ""} for row in request["records"]]})


def memory(*, approve=False):
    return SharedMemory(conversation_context=True, answerability_gate=AnswerabilityGate(Decisions(approve)))


def source(*, qid="source", scope="Cedar", alias="Cedar", answer="PRIVATE_CURRENT_17_DAYS", **kwargs):
    return MemoryTrace(f"Project {alias} data retention period?", answer, qid=qid,
                       retrieval_scope=scope, scope_aliases=(alias,), **kwargs)


def prepare(m, question, metadata=None):
    m.begin_turn(question, "probe", metadata)
    return asyncio.run(m.prepare_retrieval(question))


@pytest.mark.parametrize("question,parameter,action", [
    ("Project Cedar data retention period?", "data_retention_period", "request_current_source"),
    ("What is the data-retention period for Project Cedar?", "data_retention_period", "request_current_source"),
    ("What is Project Cedar's retention rate?", "retention_rate", "request_current_source"),
    ("Project Cedar retention?", None, "clarify_parameter"),
    ("Ｐｒｏｊｅｃｔ Ｃｅｄａｒ　ｒｅｔｅｎｔｉｏｎ　ｒａｔｅ？", "retention_rate", "request_current_source"),
    ("Cedarのデータ保持期間は？", "data_retention_period", "request_current_source"),
    ("Cedarのデータ保存期間を教えてください。", "data_retention_period", "request_current_source"),
    ("Cedarのリテンション率は？", "retention_rate", "request_current_source"),
    ("Cedarのリテンションは？", None, "clarify_parameter"),
    ("What is its data retention period?", "data_retention_period", "request_current_source"),
    ("That project retention rate?", "retention_rate", "request_current_source"),
    ("そのプロジェクトのデータ保持期間は？", "data_retention_period", "request_current_source"),
])
def test_caller_declared_identity_preserves_explicit_missing_conditions(question, parameter, action):
    m = memory()
    result = prepare(m, question, DECLARATION)
    condition = m.missing_evidence(question)
    assert m.reference_resolution(question) == "resolved"
    assert m._focus_scope == "Cedar"
    assert condition.action == action and condition.parameter == parameter
    assert condition.identity_basis == "caller_declaration" and condition.source_availability == "none_current"
    assert not m.scope_resolver().query(question).scopes  # No factual alias was registered.
    assert result["answerability"] == {"status": "empty", "candidates": 0}
    assert not m.answerability_gate.backend.requests
    assert not m.retrieve_related(question) and not m.replay_intent(question)
    assert "Cedar" not in json.dumps(condition.to_dict())  # No private identity/source text in the packet.


@pytest.mark.parametrize("lifecycle", ["expired", "superseded", "recalled"])
@pytest.mark.parametrize("declared", [False, True])
def test_inactive_sources_never_supply_identity_parameter_units_or_values(lifecycle, declared):
    m = memory()
    trace = source(answer="PRIVATE_HISTORY_17_DAYS")
    m.store(replace(trace, **{"expired": {"valid_until": 0}, "superseded": {"evidence_status": "superseded"},
                             "recalled": {"recalled_answer": True}}[lifecycle]))
    question = "Project Cedar retention?"
    prepare(m, question, DECLARATION if declared else None)
    condition = m.missing_evidence(question)
    assert condition.action == ("clarify_parameter" if declared else "clarify_reference")
    assert condition.reference == ("resolved" if declared else "unresolved")
    assert condition.parameter is None and condition.parameter_status == "underspecified"
    assert not m.retrieve_related(question) and not m.answerability_gate.backend.requests
    rendered = PrefrontalCortex.clarification_context(condition)
    assert "PRIVATE_HISTORY" not in rendered and "17" not in rendered and "days" not in rendered


@pytest.mark.parametrize("record_count", [1, 6])
def test_current_unapproved_record_and_empty_memory_have_distinct_availability(record_count):
    m = memory()
    for index in range(record_count):
        m.store(source(qid=f"source-{index}"))
    question = "Project Cedar data retention period?"
    prepare(m, question)
    condition = m.missing_evidence(question)
    assert condition.source_availability == "current_unapproved"
    assert condition.identity_basis == "available_aliases" and condition.action == "request_current_source"
    assert condition.source_approval == "denied"
    assert len(m.answerability_gate.backend.requests) == 1
    assert "PRIVATE_CURRENT" not in PrefrontalCortex.clarification_context(condition)


@pytest.mark.parametrize("unprepared", ["incomplete", "candidate_cap"])
def test_unchecked_current_records_prevent_aggregate_denial(unprepared):
    m = memory()
    if unprepared == "incomplete":
        m.store(source(qid="incomplete", answer="PRIVATE_CURRENT\n[telemetry (raw)] omitted content"))
        m.store(source(qid="checked"))
        expected_checked = 1
    else:
        for index in range(m.answerability_gate.candidate_limit + 1):
            m.store(source(qid=f"source-{index}"))
        expected_checked = m.answerability_gate.candidate_limit
    question = "Project Cedar data retention period?"
    prepared = prepare(m, question)
    assert prepared["answerability"]["checked_candidates"] == expected_checked
    assert prepared["answerability"]["approved"] == 0
    gate = m.answerability_gate
    decisions = [gate.decision(question, m._answerability_source(trace)) for trace in m.past_qas]
    assert decisions.count(False) == expected_checked and decisions.count(None) == 1
    cache = dict(gate.cache)
    condition = m.missing_evidence(question)
    assert condition.source_availability == "current_unapproved"
    assert condition.source_approval == "unknown" and condition.action == "preserve_unknown"
    assert not m.retrieve_related(question)
    assert dict(gate.cache) == cache and len(gate.backend.requests) == 1
    assert "PRIVATE_" not in PrefrontalCortex.clarification_context(condition)


@pytest.mark.parametrize("failure", ["not_prepared", "provider_error", "changed_identity", "incomplete_source"])
def test_unavailable_support_preserves_unknown_instead_of_claiming_missing_records(failure):
    m = memory()
    trace = source()
    if failure == "incomplete_source":
        trace = replace(trace, answer="PRIVATE_CURRENT\n[telemetry (raw)] omitted content")
    m.store(trace)
    question = "Project Cedar data retention period?"
    if failure == "provider_error":
        async def broken(_request):
            raise RuntimeError("PRIVATE_PROVIDER_ERROR")
        m.answerability_gate.backend.evaluate = broken
    m.begin_turn(question, "probe")
    if failure != "not_prepared":
        asyncio.run(m.prepare_retrieval(question))
    if failure == "changed_identity":
        m.answerability_gate.backend.identity = "changed"
    condition = m.missing_evidence(question)
    assert condition.action == "preserve_unknown" and condition.source_approval == "unknown"
    assert condition.source_availability == "current_unapproved"
    assert not m.retrieve_related(question)
    assert "PRIVATE_" not in PrefrontalCortex.clarification_context(condition)


def test_exact_support_accessor_distinguishes_denial_unknown_and_replay_policy():
    m = memory()
    trace = source()
    m.store(trace)
    q = "Repeat Project Cedar recorded answer verbatim."
    prepare(m, q)
    gate, intent = m.answerability_gate, m.replay_intent(q)
    packet = m._answerability_source(trace)
    assert gate.decision(q, packet, intent=intent) is False
    assert gate.decision(q, packet) is None  # An ordinary lookup has a separate policy binding.
    assert gate.decision(q + " Changed", packet, intent=intent) is None
    assert gate.decision(q, {**packet, "answer": "Changed"}, intent=intent) is None
    assert not gate.allows(q, packet, intent=intent)


def test_caller_identity_does_not_keep_an_expired_cached_approval_alive():
    m = memory(approve=True)
    m.store(source())
    question = "Project Cedar data retention period?"
    prepare(m, question, DECLARATION)
    assert m.retrieve_related(question) and m.missing_evidence(question) is None
    cache = dict(m.answerability_gate.cache)
    m.past_qas = [replace(m.past_qas[0], valid_until=0)]
    assert m.reference_resolution(question) == "resolved"
    assert not m.retrieve_related(question)
    condition = m.missing_evidence(question)
    assert condition.source_availability == "none_current" and condition.action == "request_current_source"
    assert dict(m.answerability_gate.cache) == cache and len(m.answerability_gate.backend.requests) == 1


@pytest.mark.parametrize("alias", ["Blue Rowan", "Blue Project Rowan", "Data Retention"])
def test_declared_whole_alias_is_separate_from_parameter_words(alias):
    m = memory()
    question = f"Project {alias} retention rate?"
    prepare(m, question, {"id": "Cedar", "aliases": [alias], "remember": False})
    condition = m.missing_evidence(question)
    assert condition.reference == "resolved" and condition.parameter == "retention_rate"
    assert condition.action == "request_current_source"


def test_ambiguous_alias_is_not_resolved_by_the_requested_measure():
    m = memory()
    m.store(source(alias="Shared"))
    m.store(source(qid="other", scope="Birch", alias="Shared", answer="PRIVATE_OTHER_29_PERCENT"))
    question = "Project Shared retention rate?"
    prepare(m, question)
    condition = m.missing_evidence(question)
    assert condition.reference == "ambiguous" and condition.action == "clarify_reference"
    assert condition.source_availability == "unknown"
    assert condition.parameter == "retention_rate"
    assert not m.retrieve_related(question) and not m.answerability_gate.backend.requests


def test_declared_identity_rechecks_conflicts_lifecycle_and_turn_ownership():
    m = memory()
    question = "Project Cedar data retention period?"
    prepare(m, question, DECLARATION)
    m.store(source(scope="Birch"))  # A new conflicting alias after an await.
    assert m.reference_resolution(question) == "ambiguous"
    assert m.missing_evidence(question).action == "clarify_reference"
    m.past_qas = [replace(m.past_qas[0], valid_until=0)]
    assert m.reference_resolution(question) == "resolved"  # Caller declaration is still explicit.
    m.begin_turn(question, "next")
    assert m.reference_resolution(question) is None
    assert m.missing_evidence(question).action == "clarify_reference"
    prepare(m, question, DECLARATION)
    m.abort_turn(None)
    assert m.reference_resolution(question) is None and m.missing_evidence(question) is None
    prepare(m, question, DECLARATION)
    m.restore_committed_traces([])
    assert m.reference_resolution(question) is None and m.missing_evidence(question) is None


@pytest.mark.parametrize("question,alias", [
    ("That project retention rate?", "That project"),
    ("そのプロジェクトのデータ保持期間は？", "そのプロジェクト"),
])
@pytest.mark.parametrize("lifecycle", ["expired", "superseded"])
def test_caller_pronoun_identity_cannot_switch_to_a_new_live_alias(question, alias, lifecycle):
    m = memory()
    prepare(m, question, DECLARATION)
    assert m._reference_state(question)[:3] == ("resolved", "caller_declaration", frozenset({"Cedar"}))
    m.store(source(scope="Birch", alias=alias))  # A new alias arrives after the turn began.
    assert m.scope_resolver().query(question).scopes == {"Birch"}
    assert m.reference_resolution(question) == "ambiguous"
    condition = m.missing_evidence(question)
    assert condition.reference == "ambiguous" and condition.action == "clarify_reference"
    assert condition.source_availability == "unknown" and condition.source_approval == "not_applicable"
    assert m._turn_metadata[0].retrieval_scope == "Cedar"
    m.past_qas = [replace(m.past_qas[0], **({"valid_until": 0} if lifecycle == "expired"
                                           else {"evidence_status": "superseded"}))]
    assert m._reference_state(question)[:3] == ("resolved", "caller_declaration", frozenset({"Cedar"}))
    assert not m.retrieve_related(question) and not m.answerability_gate.backend.requests


@pytest.mark.parametrize("support", ["unchecked", "cached_approval", "disabled"])
def test_conflicting_caller_reference_withholds_candidates_and_cached_approval(support):
    m = SharedMemory(conversation_context=True) if support == "disabled" else memory(approve=True)
    question = "That project data retention period?"
    trace = source(scope="Birch", alias="That project", answer="PRIVATE_BIRCH_29_DAYS")
    gate = m.answerability_gate
    if support == "cached_approval":
        m.store(trace)
        prepare(m, question)  # A prior undeclared turn approved this exact question/source.
        assert m.retrieve_related_traces(question) == [trace]
        m.past_qas.clear()
    m.begin_turn(question, "declared", DECLARATION)
    m.store(trace)
    cache = dict(gate.cache) if gate is not None else None
    requests = len(gate.backend.requests) if gate is not None else 0
    if support == "cached_approval":
        assert gate.decision(question, m._answerability_source(trace)) is True
    assert not m.answerability_candidates(question)
    assert not m.retrieve_related_traces(question) and not m.retrieve_related(question)
    prepared = asyncio.run(m.prepare_retrieval(question))
    if gate is not None:
        assert prepared["answerability"] == {"status": "empty", "candidates": 0}
        assert dict(gate.cache) == cache and len(gate.backend.requests) == requests
    condition = m.missing_evidence(question)
    assert condition.action == "clarify_reference" and condition.source_availability == "unknown"
    assert "PRIVATE_" not in PrefrontalCortex.clarification_context(condition)


@pytest.mark.parametrize("question", [
    "Project Cedar retention rate? Project Cedar",
    "Project Cedar data retention period and price?",
    "Project Cedar retention rate? Then calculate growth.",
    "Cedarのリテンション率を教えて、比較して。",
    "Summarize Project Cedar's retention policy.",
    "What is Project Cedar's deadline?",
    "Repeat Project Cedar recorded answer verbatim and verify it.",
    "Project Cedar data retention period for Project Unknown?",
    "What is the weather?",
    "Tell me a story.",
])
def test_extra_clauses_and_unrecognized_questions_receive_no_new_targeted_prompt(question):
    m = memory()
    prepare(m, question, DECLARATION)
    assert m.missing_evidence(question) is None


@pytest.mark.parametrize("question", ["retention?", "What is a retention rate?", "そのリテンションは？"])
def test_unqualified_questions_without_declared_reference_are_ordinary_questions(question):
    m = memory()
    prepare(m, question)
    assert m.missing_evidence(question) is None


def test_explicit_multiple_scopes_are_not_mislabeled_as_an_ambiguous_reference():
    m = memory()
    m.store(source())
    m.store(source(qid="birch", scope="Birch", alias="Birch"))
    q = "What is the data retention period for Project Cedar and Project Birch?"
    prepare(m, q)
    assert m.reference_resolution(q) == "multiple_explicit" and m.missing_evidence(q) is None
    with pytest.raises(ValueError, match="declared scope disagree"):
        m.begin_turn("Project Birch retention?", "conflict", DECLARATION)
    q = "Project Unknown retention?"
    prepare(m, q, DECLARATION)
    assert m.reference_resolution(q) is None
    assert m.missing_evidence(q).action == "clarify_reference"


@pytest.mark.parametrize("question", [
    "Repeat Project Cedar recorded answer verbatim.",
    "Repeat that recorded answer verbatim.",
    "その記録をそのまま再掲して",
])
def test_missing_literal_record_does_not_become_a_parameter_lookup(question):
    m = memory()
    prepare(m, question, DECLARATION)
    condition = m.missing_evidence(question)
    assert condition.operation == "recorded_answer_replay" and condition.action == "request_current_record"
    assert condition.parameter is None and condition.parameter_status == "not_applicable"
    assert not m.replay_intent(question) and not m.retrieve_related(question)


def test_multiple_current_records_require_record_identity_for_literal_replay():
    m = memory()
    m.store(source())
    m.store(source(qid="second", answer="PRIVATE_SECOND_RECORD"))
    q = "Repeat Project Cedar recorded answer verbatim."
    prepare(m, q)
    assert not m.replay_intent(q)
    condition = m.missing_evidence(q)
    assert condition.reference == "resolved" and condition.action == "clarify_record"


def test_approved_source_and_primary_writes_preserve_their_existing_paths():
    m = memory(approve=True)
    m.store(source())
    q = "Project Cedar data retention period?"
    prepare(m, q)
    assert m.retrieve_related_traces(q) and m.missing_evidence(q) is None
    replay = "Repeat Project Cedar recorded answer verbatim."
    prepare(m, replay)
    assert m.replay_intent(replay) and m.missing_evidence(replay) is None
    assert m.past_qas[0].evidence_status == "unverified"
    m.begin_turn(q, "write", {**DECLARATION, "remember": True})
    assert m.missing_evidence(q) is None
    empty = memory()
    empty.begin_turn(q, "new-source", {**DECLARATION, "remember": True})
    assert empty.reference_resolution(q) is None and empty.missing_evidence(q) is None


def test_bounded_current_source_followup_keeps_reference_and_parameter_independent():
    m = memory()
    m.store(source())
    prepare(m, "Project Cedar data retention period?")
    q = "That retention?"
    prepare(m, q)
    assert m.missing_evidence(q).action == "clarify_parameter"
    assert m.missing_evidence(q).parameter is None
    m.past_qas = [replace(m.past_qas[0], valid_until=0)]
    assert m.reference_resolution(q) is None
    assert m.missing_evidence(q).action == "clarify_reference"


class ScriptedClient(LLMClient):
    """Generate only from the actual context; all replies have terminal receipts."""
    def __init__(self, role, calls):
        super().__init__(LLMConfig("openai", "fake-missing-evidence", "FAKE_NO_CREDENTIAL",
            api_base="https://fake.invalid/v1", auto_continue=False, max_continuations=0))
        self.role, self.calls = role, calls

    async def _post_json(self, _url, payload, _headers, **_kwargs):
        system = payload["messages"][0]["content"]
        context = "\n".join(message["content"] for message in payload["messages"])
        self.calls.append((self.role, context))
        if self.role == "right" and "rigorous reviewer (critic)" in system:
            answer = '{"verdict":"ok","issues":[],"fixes":[]}'
        elif self.role == "right":
            answer = "An up-to-date document or recorded statement could supply the missing evidence."
        else:
            prefix = "Conversation missing-evidence condition:\n"
            if prefix in context:
                packet = json.loads(context.split(prefix, 1)[1].splitlines()[0])
                answer = {
                    "request_current_source": "Please provide a current source for the explicit parameter.",
                    "clarify_parameter": "Do you mean a data-retention period or a retention rate?",
                    "clarify_reference": "Which project does this request concern?",
                    "clarify_record": "Which complete recorded answer do you mean?",
                    "request_current_record": "Please provide the current complete recorded answer.",
                    "preserve_unknown": "Support for the recorded information could not be established.",
                }[packet["action"]]
            else:
                answer = "The bounded clarification condition is unavailable."
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


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "on"])
@pytest.mark.parametrize("question,declared,action,parameter", [
    ("Project Cedar data-retention period?", True, "request_current_source", "data_retention_period"),
    ("Project Cedar retention rate?", True, "request_current_source", "retention_rate"),
    ("Project Cedar retention?", True, "clarify_parameter", None),
    ("What is its data retention period?", True, "request_current_source", "data_retention_period"),
    ("Project Cedar data retention period?", False, "clarify_reference", "data_retention_period"),
    ("Repeat that recorded answer verbatim.", True, "request_current_record", None),
])
async def test_missing_condition_reaches_engine_generation_and_critic_without_historical_values(
        tmp_path, mode, question, declared, action, parameter):
    from test_dual_brain import AlwaysConsultPolicy
    calls = []
    with patch.dict(os.environ, {"DUALBRAIN_MEMORY_CONVERSATION_CONTEXT": "1"}, clear=True), \
            patch("core.llm_client.aiohttp.ClientSession", side_effect=AssertionError("External transport forbidden")):
        session = await EngineSession.create("missing-condition")
        try:
            session.memory.answerability_gate = AnswerabilityGate(Decisions())
            session.memory.store(source(valid_until=0, answer="PRIVATE_EXPIRED_17_DAYS"))
            session.left._llm_client = ScriptedClient("left", calls)
            session.right._llm_client = ScriptedClient("right", calls)
            session.left.uses_external_llm = session.right.uses_external_llm = True
            session.controller.policy = AlwaysConsultPolicy()
            result = await _handle_process(session, {
                "question": question, "qid": "probe", "leading_brain": "left", "system2_mode": mode,
                "executive_mode": "off", "executive_observer_mode": "off",
                **({"memory_context": DECLARATION} if declared else {}),
            }, blob_root=tmp_path, blob_index={}, vision_cache={}, vision_cache_limit=0)
            assert result["answer"] and not session.memory.retrieve_related(question)
            assert not session.memory.answerability_gate.backend.requests
            assert session.memory.past_qas[0].valid_until == 0
        finally:
            await session.close()
    assert "left" in {role for role, _ in calls}
    assert "right" in {role for role, _ in calls}
    if mode == "off":
        assert sum(role == "left" for role, _ in calls) >= 2  # Draft and actual final integration.
    prefix = "Conversation missing-evidence condition:\n"
    for _, system in calls:
        packet = json.loads(system.split(prefix, 1)[1].splitlines()[0])
        assert packet["action"] == action and packet["parameter"] == parameter
        assert "PRIVATE_EXPIRED_17_DAYS" not in system
        assert packet["identity_basis"] == ("caller_declaration" if declared else "unresolved")
    if action == "clarify_parameter":
        assert "data-retention period or a retention rate" in result["answer"]
    elif action == "request_current_source":
        assert "current source" in result["answer"]


@pytest.mark.asyncio
async def test_failed_support_status_survives_final_integration_without_reenter_request(tmp_path):
    from test_dual_brain import AlwaysConsultPolicy
    calls = []
    backend = Decisions()
    async def broken(request):
        backend.requests.append(request)
        raise RuntimeError("PRIVATE_PROVIDER_ERROR")
    backend.evaluate = broken
    q = "Project Cedar data retention period?"
    with patch.dict(os.environ, {"DUALBRAIN_MEMORY_CONVERSATION_CONTEXT": "1"}, clear=True), \
            patch("core.llm_client.aiohttp.ClientSession", side_effect=AssertionError("External transport forbidden")):
        session = await EngineSession.create("unknown-approval")
        try:
            session.memory.answerability_gate = AnswerabilityGate(backend)
            session.memory.store(source())
            session.left._llm_client, session.right._llm_client = ScriptedClient("left", calls), ScriptedClient("right", calls)
            session.left.uses_external_llm = session.right.uses_external_llm = True
            session.controller.policy = AlwaysConsultPolicy()
            result = await _handle_process(session, {"question": q, "qid": "probe", "leading_brain": "left",
                "system2_mode": "off", "executive_mode": "off", "executive_observer_mode": "off"},
                blob_root=tmp_path, blob_index={}, vision_cache={}, vision_cache_limit=0)
            assert result["answer"] == "Support for the recorded information could not be established."
            assert not session.memory.retrieve_related(q) and len(backend.requests) == 1
        finally:
            await session.close()
    assert sum(role == "left" for role, _ in calls) >= 2
    for _, context in calls:
        packet = json.loads(context.split("Conversation missing-evidence condition:\n", 1)[1].splitlines()[0])
        assert packet["action"] == "preserve_unknown" and packet["source_approval"] == "unknown"
        assert "PRIVATE_" not in context
