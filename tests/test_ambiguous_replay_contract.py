"""Public memory boundaries for literal requests whose record is not selected.

The support backend is deterministic. Ordinary approvals are explicitly primed
to exercise existing receipts even when runtime preparation withholds a request.
This file is self-contained so the same cases can run against a frozen baseline.
"""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from core.memory_answerability import AnswerabilityGate
from core.shared_memory import MemoryTrace, SharedMemory


REPLAY = "Repeat Project Larch recorded answer verbatim."
DECLARATION = {"id": "Larch", "aliases": ["Larch"], "remember": False}


class FixedSupport:
    identity = "ambiguous-replay-contract-v1"

    def __init__(self, *, decisions=None, unknown=()):
        self.decisions = dict(decisions or {})
        self.unknown = set(unknown)
        self.requests = []

    async def evaluate(self, request):
        self.requests.append(deepcopy(request))
        if any(row["answer"] in self.unknown for row in request["records"]):
            raise RuntimeError("Scripted unavailable support")
        return json.dumps({"decisions": [
            {"id": row["id"], "supported": self.decisions.get(row["answer"], True),
             "quote": row["answer"] if self.decisions.get(row["answer"], True) else ""}
            for row in request["records"]
        ]})


def records():
    return (
        MemoryTrace("Project Larch data retention period?",
                    "PRIVATE_PERIOD: retention is 19 days.\nException: previews use 2 days — 記録のまま。",
                    qid="period", seed_id="period-seed", timestamp=1.,
                    retrieval_scope="Larch", scope_aliases=("Larch",)),
        MemoryTrace("Project Larch backup location?",
                    "PRIVATE_BACKUP: backups stay in the north vault.",
                    qid="backup", seed_id="backup-seed", timestamp=2.,
                    retrieval_scope="Larch", scope_aliases=("Larch",)),
    )


def memory(traces, backend=None, *, conversation=True):
    result = SharedMemory(conversation_context=conversation,
                          answerability_gate=AnswerabilityGate(backend or FixedSupport()))
    for trace in traces:
        result.store(trace)
    return result


def begin_replay(m, form="named"):
    question = {
        "named": REPLAY,
        "english_followup": "Repeat that recorded answer verbatim.",
        "japanese_followup": "その記録をそのまま再掲して",
        "caller_declared": "Repeat the recorded answer verbatim.",
    }[form]
    if form in {"english_followup", "japanese_followup"}:
        m.begin_turn("Project Larch data retention period?", "anchor")
    m.begin_turn(question, "replay-probe", DECLARATION if form == "caller_declared" else None)
    return question


def prime_ordinary(m, question, traces):
    """Retain an ordinary-policy decision for the exact effective query/source."""
    result = asyncio.run(m.answerability_gate.prepare(
        m.retrieval_question(question), [m._answerability_source(trace) for trace in traces]))
    assert result["status"] in {"ready", "cached"}
    assert all("operation" not in request for request in m.answerability_gate.backend.requests)


def assert_record_selection_required(m, question):
    assert m.reference_resolution(question) == "resolved"
    assert m.replay_intent(question) is None
    condition = m.missing_evidence(question)
    assert condition is not None
    assert condition.reference == "resolved" and condition.action == "clarify_record"
    assert condition.operation == "recorded_answer_replay"
    assert condition.parameter is None and condition.parameter_status == "not_applicable"
    assert "PRIVATE_" not in json.dumps(condition.to_dict())
    # Neither presentation limits nor the latent-seed path may select a record.
    assert m.retrieve_related_traces(question, n=1) == []
    assert m.retrieve_related_traces(question, n=len(m.past_qas)) == []
    assert m.retrieve_related(question) == ""
    assert m.evidence_for_seeds([trace.seed_id for trace in m.past_qas], question=question) == []


@pytest.mark.parametrize("form", ["named", "english_followup", "japanese_followup", "caller_declared"])
@pytest.mark.parametrize("approval", ["both", "one_denied", "one_unknown"])
def test_ordinary_support_never_selects_the_target_of_a_whole_record_request(form, approval):
    period, backup = records()
    backend = FixedSupport(
        decisions={backup.answer: False} if approval == "one_denied" else None,
        unknown=(backup.answer,) if approval == "one_unknown" else (),
    )
    m = memory((period, backup), backend)
    question = begin_replay(m, form)
    prime_ordinary(m, question, (period,) if approval == "one_unknown" else (period, backup))
    key = m.retrieval_question(question)
    assert m.answerability_gate.decision(key, m._answerability_source(period)) is True
    expected_second = {"both": True, "one_denied": False, "one_unknown": None}[approval]
    assert m.answerability_gate.decision(key, m._answerability_source(backup)) is expected_second
    # Runtime may withhold support preparation altogether. If it does prepare,
    # neither another decision nor one cached positive establishes user intent.
    asyncio.run(m.prepare_retrieval(question))
    assert_record_selection_required(m, question)


@pytest.mark.parametrize("omission", ["candidate_limit", "ineligible_record"])
def test_records_omitted_from_support_candidates_still_prevent_selection(omission):
    period, backup = records()
    if omission == "candidate_limit":
        traces = tuple(replace(period, qid=f"record-{i}", seed_id=f"seed-{i}", timestamp=float(i + 1),
                               question=f"Project Larch recorded document {i}?", answer=f"PRIVATE_DOCUMENT_{i}")
                       for i in range(7))
        checked = traces[1:]
    else:
        traces = (period, replace(backup, answer="PRIVATE_BACKUP\n[telemetry (raw)] omitted content"))
        checked = (period,)
    m = memory(traces)
    question = begin_replay(m)
    prime_ordinary(m, question, checked)
    gate, key = m.answerability_gate, m.retrieval_question(question)
    states = [gate.decision(key, m._answerability_source(trace)) for trace in traces]
    assert states.count(True) == len(checked) and states.count(None) == 1
    asyncio.run(m.prepare_retrieval(question))
    assert_record_selection_required(m, question)


@pytest.mark.parametrize("transition", ["expired", "superseded", "recalled"])
def test_becoming_unique_requires_a_new_complete_replay_approval(transition):
    period, backup = records()
    m = memory((period, backup))
    question = begin_replay(m)
    prime_ordinary(m, question, (period, backup))
    change = {"expired": {"valid_until": 0.}, "superseded": {"evidence_status": "superseded"},
              "recalled": {"recalled_answer": True}}[transition]
    m.past_qas[1] = replace(backup, **change)
    intent = m.replay_intent(question)
    assert intent is not None and intent.source_qid == period.qid
    gate, key = m.answerability_gate, m.retrieval_question(question)
    source = m._answerability_source(period)
    assert gate.decision(key, source) is True
    assert gate.decision(key, source, intent=intent) is None
    assert m.retrieve_related_traces(question) == []
    assert m.evidence_for_seeds([period.seed_id, backup.seed_id], question=question) == []
    prior_calls = len(gate.backend.requests)
    prepared = asyncio.run(m.prepare_retrieval(question))
    assert prepared["answerability"]["approved"] == 1
    assert len(gate.backend.requests) == prior_calls + 1
    request = gate.backend.requests[-1]
    assert request["operation"] == {"kind": "recorded_answer_replay", "record_id": "m0"}
    assert len(request["records"]) == 1 and request["records"][0]["answer"] == period.answer
    assert m.retrieve_related_traces(question) == [period]
    quoted = json.loads(m.retrieve_related(question).split("\n", 1)[1])["recorded_answer"]
    assert quoted.encode("utf-8") == period.answer.encode("utf-8")
    assert [row["source_qid"] for row in m.evidence_for_seeds(
        [period.seed_id, backup.seed_id], question=question)] == [period.qid]
    assert "PRIVATE_BACKUP" not in m.retrieve_related(question)


def test_a_second_current_record_revokes_exposure_despite_old_replay_and_ordinary_approvals():
    period, backup = records()
    m = memory((period,))
    question = begin_replay(m)
    prime_ordinary(m, question, (period,))
    asyncio.run(m.prepare_retrieval(question))
    old_intent = m.replay_intent(question)
    assert old_intent is not None and m.retrieve_related_traces(question) == [period]
    m.store(backup)
    # A historical receipt may remain in the exact cache, but it does not prove
    # that the user's request still has one current possible target.
    key, source = m.retrieval_question(question), m._answerability_source(period)
    assert m.answerability_gate.decision(key, source, intent=old_intent) is True
    assert m.answerability_gate.decision(key, source) is True
    assert_record_selection_required(m, question)


@pytest.mark.parametrize("kind", ["single_fact", "compound_facts", "extra_clause"])
def test_ordinary_semantic_requests_keep_their_approved_source_path(kind):
    period, backup = records()
    question = {
        "single_fact": "What is Project Larch's data retention period?",
        "compound_facts": "What are Project Larch's data retention period and backup location?",
        "extra_clause": "Repeat Project Larch recorded answer verbatim and verify it.",
    }[kind]
    backend = FixedSupport(decisions={backup.answer: False} if kind == "single_fact" else None)
    m = memory((period, backup), backend)
    m.begin_turn(question, "semantic-probe")
    asyncio.run(m.prepare_retrieval(question))
    expected = {period.qid} if kind == "single_fact" else {period.qid, backup.qid}
    assert m.replay_intent(question) is None and m.missing_evidence(question) is None
    assert {trace.qid for trace in m.retrieve_related_traces(question)} == expected
    assert {row["source_qid"] for row in m.evidence_for_seeds(
        [period.seed_id, backup.seed_id], question=question)} == expected
    assert "PRIVATE_PERIOD" in m.retrieve_related(question)
    assert ("PRIVATE_BACKUP" in m.retrieve_related(question)) == (backup.qid in expected)
    assert all("operation" not in request for request in backend.requests)
    assert all(trace.evidence_status == "unverified" for trace in m.past_qas)


def test_disabled_checker_does_not_turn_lexical_relevance_into_record_selection():
    period, backup = records()
    m = SharedMemory(conversation_context=True)
    m.store(replace(period, question="Project Larch recorded answer about retention?"))
    m.store(replace(backup, question="Project Larch recorded answer about backups?"))
    question = begin_replay(m)
    asyncio.run(m.prepare_retrieval(question))
    assert_record_selection_required(m, question)


def test_legacy_memory_retains_its_ordinary_approved_qa_behavior():
    period, backup = records()
    m = memory((period, backup), conversation=False)
    m.begin_turn(REPLAY, "legacy-probe")
    asyncio.run(m.prepare_retrieval(REPLAY))
    assert m.replay_intent(REPLAY) is None and m.missing_evidence(REPLAY) is None
    assert {trace.qid for trace in m.retrieve_related_traces(REPLAY)} == {period.qid, backup.qid}
    assert "PRIVATE_PERIOD" in m.retrieve_related(REPLAY) and "PRIVATE_BACKUP" in m.retrieve_related(REPLAY)
    assert all("operation" not in request for request in m.answerability_gate.backend.requests)
