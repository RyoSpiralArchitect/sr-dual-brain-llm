"""Factual approval cannot choose an unspecified whole record for the caller."""
import asyncio
from dataclasses import replace
import json
import os
from unittest.mock import patch

import pytest

from core.memory_answerability import AnswerabilityGate
from core.prefrontal_cortex import PrefrontalCortex
from core.shared_memory import MemoryTrace, SharedMemory
from test_missing_evidence import Decisions, ScriptedClient


QUESTION = 'Repeat Project Copper Finch recorded answer verbatim.'
DECLARATION = {'id': 'copper', 'aliases': ['Copper Finch', '銅雀'], 'remember': False}


def source(qid='period', **kwargs):
    fields = dict(question='Project Copper Finch recorded answer for data retention period?',
                  answer='PRIVATE_PERIOD_21_DAYS', qid=qid, seed_id=qid + '-seed',
                  retrieval_scope='copper', scope_aliases=('Copper Finch', '銅雀'))
    fields.update(kwargs)
    return MemoryTrace(**fields)


def memory(checker=True, **kwargs):
    result = SharedMemory(conversation_context=True,
        answerability_gate=AnswerabilityGate(Decisions(True)) if checker else None, **kwargs)
    result.store(source())
    result.store(source('backup', question='Project Copper Finch recorded answer for backup location?',
                        answer='PRIVATE_BACKUP_OSAKA'))
    return result


def prepare(m, question=QUESTION, metadata=None):
    m.begin_turn(question, 'probe', metadata)
    return asyncio.run(m.prepare_retrieval(question))


@pytest.mark.parametrize('checker', [False, True])
@pytest.mark.parametrize('question,declared,focus', [
    (QUESTION, False, False),
    ('Project Copper Finch: repeat the recorded answer verbatim.', False, False),
    ('銅雀の記録された回答をそのまま再掲して。', False, False),
    ('Repeat that recorded answer verbatim.', False, True),
    ('Ｐｌｅａｓｅ　ｒｅｐｅａｔ　ｔｈａｔ　ｒｅｃｏｒｄｅｄ　ａｎｓｗｅｒ　ｖｅｒｂａｔｉｍ．', False, True),
    ('Repeat the recorded answer verbatim.', True, False),
])
def test_approved_records_cannot_replace_missing_record_selection(checker, question, declared, focus):
    m = memory(checker)
    if focus:
        m.begin_turn('Project Copper Finch retention?', 'focus')
    if checker:
        m.begin_turn(question, 'prepare-cache', DECLARATION if declared else None)
        asyncio.run(m.answerability_gate.prepare(m.retrieval_question(question),
                    [m._answerability_source(t) for t in m.past_qas]))
    prepare(m, question, DECLARATION if declared else None)
    if checker:
        gate, query = m.answerability_gate, m.retrieval_question(question)
        assert all(gate.decision(query, m._answerability_source(t)) is True for t in m.past_qas)
        before = dict(gate.cache)
    assert m.reference_resolution(question) == 'resolved' and m.replay_intent(question) is None
    for count in (1, 3, 100):
        assert m.retrieve_related_traces(question, count) == []
        assert m.retrieve_related(question, count) == ''
    assert m.evidence_for_seeds(['period-seed', 'backup-seed'], question=question) == []
    condition = m.missing_evidence(question)
    assert condition.action == 'clarify_record' and condition.operation == 'recorded_answer_replay'
    assert condition.reference == 'resolved' and condition.parameter is None
    assert 'PRIVATE_' not in PrefrontalCortex.clarification_context(condition)
    if checker:
        assert dict(gate.cache) == before  # Withhold delivery, retain the actual receipts.


@pytest.mark.parametrize('expansion', [False, True])
@pytest.mark.parametrize('candidate_limit', [1, 3])
def test_candidate_caps_and_ineligible_second_source_do_not_manufacture_uniqueness(expansion, candidate_limit):
    m = memory(answerability_scope_expansion=expansion)
    m.answerability_gate.candidate_limit = candidate_limit
    m.past_qas[1] = replace(m.past_qas[1], answer='PRIVATE_LONG_' + 'x' * 2100)
    # A current ordinary approval already exists even if the ranking cap later
    # chooses only an ineligible record. It must not become a selection oracle.
    asyncio.run(m.answerability_gate.prepare(QUESTION, [m._answerability_source(m.past_qas[0])]))
    prepare(m)
    assert m.answerability_gate.backend.requests
    assert len(m.answerability_gate.backend.requests[0]['records']) == 1
    assert not m.retrieve_related(QUESTION, 1) and m.replay_intent(QUESTION) is None
    assert m.missing_evidence(QUESTION).action == 'clarify_record'


@pytest.mark.parametrize('inactive', ['expired', 'superseded', 'derived', 'other_scope'])
def test_only_current_primary_sources_in_the_resolved_scope_count_for_selection(inactive):
    m = memory()
    changes = {'expired': {'valid_until': 0}, 'superseded': {'evidence_status': 'superseded'},
               'derived': {'recalled_answer': True},
               'other_scope': {'retrieval_scope': 'maple', 'scope_aliases': ('Maple',)}}
    m.past_qas[1] = replace(m.past_qas[1], **changes[inactive])
    prepare(m)
    assert m.replay_intent(QUESTION).source_qid == 'period'
    assert [t.qid for t in m.retrieve_related_traces(QUESTION)] == ['period']
    assert m.missing_evidence(QUESTION) is None
    assert 'recorded_answer' in m.retrieve_related(QUESTION)


@pytest.mark.asyncio
async def test_late_source_insertion_vetoes_an_awaited_full_replay_approval_and_rechecks_expiry():
    started, release = asyncio.Event(), asyncio.Event()
    backend = Decisions(True)
    evaluate = backend.evaluate
    async def delayed(request):
        started.set()
        await release.wait()
        return await evaluate(request)
    backend.evaluate = delayed
    m = SharedMemory(conversation_context=True, answerability_gate=AnswerabilityGate(backend))
    m.store(source())
    m.begin_turn(QUESTION, 'probe')
    intent = m.replay_intent(QUESTION)
    task = asyncio.create_task(m.prepare_retrieval(QUESTION))
    await started.wait()
    m.store(source('backup', question='Project Copper Finch backup location?', answer='PRIVATE_BACKUP_OSAKA'))
    release.set()
    await task
    assert m.answerability_gate.decision(QUESTION, m._answerability_source(m.past_qas[0]), intent=intent) is True
    assert not m.retrieve_related(QUESTION) and m.replay_intent(QUESTION) is None
    assert m.missing_evidence(QUESTION).action == 'clarify_record'
    assert not m.evidence_for_seeds(['period-seed'], question=QUESTION)
    m.past_qas[1] = replace(m.past_qas[1], valid_until=0)
    assert m.replay_intent(QUESTION) == intent
    assert [t.qid for t in m.retrieve_related_traces(QUESTION)] == ['period']
    assert m.missing_evidence(QUESTION) is None and len(backend.requests) == 1


@pytest.mark.parametrize('question', [
    'Project Copper Finch data retention period?',
    'Repeat Project Copper Finch recorded answer verbatim. Then summarize it.',
])
def test_factual_and_extra_clause_requests_keep_their_existing_nonliteral_path(question):
    m = memory()
    prepare(m, question)
    assert m.replay_intent(question) is None
    assert m.retrieve_related_traces(question)
    assert m.missing_evidence(question) is None


def test_selection_guard_is_opt_in_and_does_not_replace_explicit_primary_write_authority():
    legacy = SharedMemory()
    legacy.store(source())
    legacy.store(source('backup', answer='PRIVATE_BACKUP_OSAKA'))
    assert legacy.retrieve_related_traces(QUESTION)
    m = memory()
    prepare(m, QUESTION, {**DECLARATION, 'remember': True})
    assert m.primary_write_pending and m.replay_intent(QUESTION) is None
    assert m.retrieve_related_traces(QUESTION) and m.missing_evidence(QUESTION) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['off', 'on'])
async def test_selection_condition_survives_both_brains_and_final_integration(tmp_path, mode):
    from engine_stdio import EngineSession, _handle_process
    from test_dual_brain import AlwaysConsultPolicy
    calls = []
    with patch.dict(os.environ, {'DUALBRAIN_MEMORY_CONVERSATION_CONTEXT': '1'}, clear=True), \
            patch('core.llm_client.aiohttp.ClientSession', side_effect=AssertionError('External HTTP forbidden')):
        session = await EngineSession.create('record-selection')
        try:
            configured = memory()
            session.memory.answerability_gate = configured.answerability_gate
            for trace in configured.past_qas:
                session.memory.store(trace)
            await session.memory.answerability_gate.prepare(QUESTION,
                [session.memory._answerability_source(t) for t in session.memory.past_qas])
            session.left._llm_client = ScriptedClient('left', calls)
            session.right._llm_client = ScriptedClient('right', calls)
            session.left.uses_external_llm = session.right.uses_external_llm = True
            session.controller.policy = AlwaysConsultPolicy()
            result = await _handle_process(session, {
                'question': QUESTION, 'qid': 'probe', 'leading_brain': 'left', 'system2_mode': mode,
                'executive_mode': 'off', 'executive_observer_mode': 'off',
            }, blob_root=tmp_path, blob_index={}, vision_cache={}, vision_cache_limit=0)
            assert result['answer'] == 'Which complete recorded answer do you mean?'
            assert len(session.memory.answerability_gate.backend.requests) == 1
        finally:
            await session.close()
    assert {'left', 'right'} <= {role for role, _ in calls}
    if mode == 'off':
        assert sum(role == 'left' for role, _ in calls) >= 2
    prefix = 'Conversation missing-evidence condition:\n'
    for _, context in calls:
        assert 'PRIVATE_' not in context
        assert json.loads(context.split(prefix, 1)[1].splitlines()[0])['action'] == 'clarify_record'
