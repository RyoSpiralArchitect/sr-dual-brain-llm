import asyncio
from dataclasses import replace
import json

import pytest

from core.memory_answerability import AnswerabilityGate, POLICY, REPLAY_POLICY, policy_for, request_for
from core.memory_intent import RecordReplayIntent
from core.consultation_router import route_conversation_lookup
from core.shared_memory import MemoryTrace, SharedMemory
from test_memory_answerability import Checks


class ReplayChecks(Checks):
    def __init__(self, *, partial=False, deny=False):
        self.requests, self.partial, self.deny = [], partial, deny

    async def evaluate(self, request):
        self.requests.append(request)
        replay = 'operation' in request
        return json.dumps({'decisions': [{'id': r['id'], 'supported': replay and not self.deny,
            'quote': (r['answer'][:1] if self.partial else r['answer']) if replay and not self.deny else ''}
            for r in request['records']]})


def memory(backend=None):
    result = SharedMemory(conversation_context=True, answerability_gate=AnswerabilityGate(backend or ReplayChecks()))
    result.store(MemoryTrace('Project Rowan retention?', 'Retention is 8 days.', qid='original',
                            retrieval_scope='rowan', scope_aliases=('Rowan',)))
    return result


def prepare(m, question, qid='probe'):
    m.begin_turn(question, qid)
    return asyncio.run(m.prepare_retrieval(question))


def test_literal_operation_survives_correction_pronoun_and_exact_cache_binding():
    m = memory()
    q = 'Repeat Project Rowan recorded answer verbatim.'
    prepare(m, q)
    assert m.retrieve_related_traces(q)[0].qid == 'original'
    m.begin_turn('Project Rowan retention?', 'corrected', {
        'id': 'rowan', 'aliases': ['Rowan'], 'remember': True, 'supersedes': ['original']})
    m.store({'Q': 'Project Rowan retention?', 'A': 'Retention is 5 days.'}, qid='corrected')
    followup = 'その記録をそのまま再掲して'
    result = prepare(m, followup)
    assert result['answerability']['approved'] == 1
    intent = m.replay_intent(followup)
    assert intent.source_qid == 'corrected'
    backend = m.answerability_gate.backend
    assert backend.requests[-1]['operation'] == {'kind': 'recorded_answer_replay', 'record_id': 'm0'}
    assert backend.requests[-1]['records'][0]['answer'] == 'Retention is 5 days.'
    assert policy_for(backend.requests[-1]) == REPLAY_POLICY
    assert 'Recorded-answer replay' in m.retrieve_related(followup)
    assert json.loads(m.retrieve_related(followup).split('\n', 1)[1])['recorded_answer'] == 'Retention is 5 days.'
    before = len(backend.requests)
    prepare(m, followup, 'again')
    assert len(backend.requests) == before
    assert not m.answerability_gate.allows(m.retrieval_question(followup),
        m._answerability_source(m.past_qas[-1]))  # ordinary factual approval is distinct
    assert m.past_qas[0].evidence_status == 'superseded'


@pytest.mark.parametrize('followup', ['Repeat that recorded answer verbatim.', 'Please quote that stored text exactly.'])
def test_bounded_english_deictic_replay_carries_only_existing_unambiguous_focus(followup):
    m = memory()
    prepare(m, 'Project Rowan retention?')
    prepare(m, followup)
    assert m.reference_resolution(followup) == 'resolved' and m.replay_intent(followup)
    prepare(m, 'Different topic.', 'interrupt')
    prepare(m, followup, 'lost')
    assert m.replay_intent(followup) is None


@pytest.mark.parametrize('question', [
    'Project Rowan retention?', 'Summarize Project Rowan recorded answer.',
    'Repeat Project Rowan recorded answer verbatim and verify it.',
    'Repeat Project Rowan recorded answer verbatim. Then change the limit.',
    'Project Rowanの記録をそのまま再掲して、合計も計算して',
    'Repeat Project Unknown recorded answer verbatim.',
    'Repeat Project Rowan and Project Unknown recorded answer verbatim.',
])
def test_no_literal_operation_for_nonliteral_extra_clause_or_unknown_scope(question):
    m = memory()
    prepare(m, question)
    assert m.replay_intent(question) is None
    assert all('operation' not in request for request in m.answerability_gate.backend.requests)


@pytest.mark.parametrize('change', ['expired', 'superseded', 'derived', 'long', 'sanitized', 'duplicate', 'multiple', 'many'])
def test_replay_declines_incomplete_inactive_and_nonunique_sources_before_ranking(change):
    m = memory()
    source = m.past_qas[0]
    if change in {'expired', 'superseded', 'derived', 'long', 'sanitized'}:
        updates = {'expired': {'valid_until': 0}, 'superseded': {'evidence_status': 'superseded'},
            'derived': {'recalled_answer': True}, 'long': {'answer': 'x' * 2049},
            'sanitized': {'answer': 'Retention is 8 days.\ntelemetry (raw) hidden'}}
        m.past_qas[0] = replace(source, **updates[change])
    else:
        count = 7 if change == 'many' else 1
        for i in range(count):
            m.store(replace(source, qid=source.qid if change == 'duplicate' else 'other-' + str(i),
                            question='Project Rowan owner?', answer='Mina'))
    q = 'Repeat Project Rowan recorded answer verbatim.'
    prepare(m, q)
    assert m.replay_intent(q) is None
    assert all('operation' not in request for request in m.answerability_gate.backend.requests)


@pytest.mark.parametrize('partial,deny', [(True, False), (False, True)])
def test_partial_quote_and_denial_never_deliver_a_replay(partial, deny):
    m = memory(ReplayChecks(partial=partial, deny=deny))
    q = 'Repeat Project Rowan recorded answer verbatim.'
    result = prepare(m, q)
    assert result['answerability']['status'] == ('withheld' if partial else 'ready')
    assert not m.retrieve_related(q)


def test_changed_source_query_and_plain_decision_cannot_reuse_replay_approval():
    m = memory()
    q = 'Repeat Project Rowan recorded answer verbatim.'
    prepare(m, q)
    source = m._answerability_source(m.past_qas[0])
    intent = m.replay_intent(q)
    gate = m.answerability_gate
    assert gate.allows(q, source, intent=intent)
    assert not gate.allows(q, source)
    for changed in ({**source, 'answer': 'Retention is 4 days.'}, {**source, 'valid_until': 0}, {**source, 'qid': 'other'}):
        assert not gate.allows(q, changed, intent=intent)
    assert not gate.allows(q + ' Also explain.', source, intent=intent)
    result = asyncio.run(gate.prepare(q, [source, source], intent=intent))
    assert result == {'status': 'withheld', 'error_type': 'IntentBindingChanged'}
    assert policy_for(request_for(q, [source])) == POLICY


def test_primary_write_intent_never_gets_reinterpreted_as_replay():
    m = memory()
    q = 'Repeat Project Rowan recorded answer verbatim.'
    m.begin_turn(q, 'new', {'id': 'rowan', 'aliases': ['Rowan'], 'remember': True})
    assert m.primary_write_pending and m.replay_intent(q) is None


def test_complete_typed_quotation_can_preserve_qualitative_text_without_numeric_inference():
    source = MemoryTrace('Project Rowan color?', 'The recorded color is indigo.')
    args = dict(question='Repeat Project Rowan recorded answer verbatim.', draft=source.answer,
        confidence=.9, risk=0., q_type='easy', reference='resolved', approved_records=[source])
    assert route_conversation_lookup(**args).action == 'defer'
    assert route_conversation_lookup(**args, replay_intent=True).reason == 'supported_literal_replay'
    assert route_conversation_lookup(**{**args, 'draft': source.answer + ' Added.'}, replay_intent=True).action == 'defer'
    assert route_conversation_lookup(**{**args, 'risk': .1}, replay_intent=True).action == 'defer'


@pytest.mark.parametrize('question', [
    'Repeat Project Rowan recorded answer verbatim. Explain.',
    'Repeat the recorded answer verbatim. Explain.',
    'Explain. Repeat the recorded answer verbatim.',
    'Repeat and verify recorded answer verbatim.',
])
def test_an_alias_cannot_erase_an_extra_instruction_to_create_literal_intent(question):
    m = memory()
    m.past_qas[0] = replace(m.past_qas[0], scope_aliases=('Rowan', 'Explain', 'and verify'))
    prepare(m, question)
    assert m.replay_intent(question) is None
    assert all('operation' not in r for r in m.answerability_gate.backend.requests)


@pytest.mark.parametrize('question', [
    'Repeat the recorded answer for Project Rowan verbatim.',
    'Quote the stored text exactly for Rowan.', "Repeat Rowan's recorded answer verbatim.",
    'Project Rowan: Repeat the recorded answer verbatim.', 'プロジェクトRowanの記録をそのまま再掲して',
])
def test_explicit_identity_slots_preserve_bounded_supported_forms(question):
    m = memory()
    prepare(m, question)
    assert m.replay_intent(question) is not None
