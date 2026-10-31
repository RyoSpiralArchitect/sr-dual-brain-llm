import asyncio
from dataclasses import replace

import pytest

from core.consultation_router import route_conversation_lookup
from core.shared_memory import MemoryTrace, SharedMemory
from core.memory_answerability import AnswerabilityGate
from test_memory_answerability import Checks
from test_dual_brain import make_stage_controller, make_decision


def route(**overrides):
    values = dict(question='What is Project Amber 31 retention?', draft='The recorded retention is 11 days.',
                  confidence=.9, risk=0., q_type='easy', reference='resolved',
                  approved_records=[MemoryTrace('Project Amber 31 retention', 'Retention is 11 days.')])
    values.update(overrides)
    return route_conversation_lookup(**values)


def test_identifier_digits_are_not_an_arithmetic_request_and_records_are_not_approved():
    trace = MemoryTrace('Project Amber 31 retention', 'Retention is 11 days.')
    assert route(approved_records=[trace]).action == 'skip'
    assert route(q_type='medium', approved_records=[trace]).action == 'skip'
    assert trace.evidence_status == 'unverified'
    assert route(question='Amberの保存期間は何日？', draft='記録では１１日です。',
                 approved_records=[replace(trace, answer='保存期間は11日です。')]).action == 'skip'


@pytest.mark.parametrize('changes', [
    {'approved_records': []}, {'approved_records': [MemoryTrace('a','1'),MemoryTrace('b','2')]},
    {'reference':'ambiguous'}, {'reference':None}, {'confidence':.4}, {'confidence':float('nan')},
    {'risk':.1}, {'risk':float('nan')}, {'draft':''}, {'draft':'I cannot find a record.'},
    {'approved_records':[MemoryTrace('status','It is blue.')], 'draft':'It is blue.'},
])
def test_no_fast_path_for_missing_ambiguous_incomplete_or_uncertain_evidence(changes):
    assert route(**changes).action == 'defer'


@pytest.mark.parametrize('question', [
    'Verify Project Amber 31 retention.', 'Project Amber 31の保存期間が正しいか検証して。',
    'What is the total retention across 3 sequential runs?', 'Calculate 11 * 3.',
    'Compare Amber and Indigo.', 'If the contract changes, what follows?',
])
def test_explicit_review_and_reasoning_need_consultation_even_at_high_confidence(question):
    assert route(question=question).action == 'consult'


def test_new_numeric_claim_requests_a_critic_instead_of_skipping():
    assert route(draft='The recorded retention is 19 days.').reason == 'numeric_drift'


def test_grouped_numbers_are_whole_values_and_other_formats_never_silently_skip():
    source=MemoryTrace('Project Amber request limit','The limit is 1,000 requests.')
    assert route(approved_records=[source],draft='The recorded limit is 1,000,000 requests.').reason=='numeric_drift'
    assert route(approved_records=[source],draft='The recorded limit is 1000 requests.').action=='skip'
    for text in ('1,00 requests','1 000 requests','1/2 requests','1e3 requests','1.2.3 requests'):
        assert route(approved_records=[replace(source,answer=text)],draft=text).reason=='unsupported_numeric_format'


async def routing_stage(monkeypatch, **changes):
    monkeypatch.setenv('DUALBRAIN_CONVERSATION_SELECTIVE_CONSULT', '1')
    m = SharedMemory(conversation_context=True, answerability_gate=AnswerabilityGate(Checks()))
    m.store(MemoryTrace('Project Amber 31 retention', 'Retention is 11 days.', qid='a',
                       retrieval_scope='amber', scope_aliases=('Amber', 'Amber 31')))
    q = 'What is Project Amber 31 retention?'
    m.begin_turn(q, 'probe')
    await m.prepare_retrieval(q)
    controller = make_stage_controller(memory=m)
    parts = {'memory':m.retrieve_related(q)}
    frame = controller._build_context_frame(question=q, context_parts=parts)
    decision = make_decision(action=1)
    decision.state['q_type']='easy'
    if changes.pop('primary_write', False):
        m._turn_metadata = ('amber', (), True)
    decision.state.update(changes.pop('state', {}))
    options = dict(question=q, decision=decision, draft='According to the record, retention is 11 days.',
        confidence=.9, affect={'risk':0.}, salience_signal=None, context_parts=parts,
        context_frame=frame, system2_mode='auto', system2_priority='balanced', system2_active=True,
        system2_capable=True, system2_reason='numeric_question', precision_priority=False,
        system2_low_signal_filter=True, director_task=None, director_payload=None, director_control=None,
        explicit_leading=True, force_right_lead=False, is_trivial_chat=False,
        phase_add=lambda *_:None, context_signal_len=len(parts['memory']), leading_brain='left')
    options.update(changes)
    return await controller._run_routing_stage(**options), controller, m


def test_post_draft_route_can_override_prior_memory_driven_consultation_without_approving_facts(monkeypatch):
    result, controller, m = asyncio.run(routing_stage(monkeypatch))
    assert result.decision.action == 0 and not result.system2_active
    assert m.past_qas[0].evidence_status == 'unverified'
    assert [t.qid for t in m.retrieve_related_traces('What is Project Amber 31 retention?')] == ['a']
    event = [p for name,p in controller.telemetry.events if name=='system2_mode'][-1]
    assert event['enabled'] is False and event['reason']=='selective_supported_lookup'


@pytest.mark.parametrize('changes', [
    {'system2_mode':'on'}, {'system2_mode':'off'}, {'leading_brain':'right'},
    {'force_right_lead':True}, {'precision_priority':True},
    {'director_control':{'consult':'force'}}, {'director_control':{'consult':'skip'}},
    {'primary_write':True}, {'state':{'acc_override_consult':True}},
    {'state':{'amygdala_override':True}},
])
def test_explicit_modes_leads_priority_and_director_controls_retain_authority(monkeypatch, changes):
    result, _, _ = asyncio.run(routing_stage(monkeypatch, **changes))
    assert 'selective_consultation' not in result.decision.state
    assert result.decision.action == (0 if changes.get('director_control',{}).get('consult')=='skip' else 1)
