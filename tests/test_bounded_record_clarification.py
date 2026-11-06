"""Only exact, freshly justified source-free record questions can skip review."""
from dataclasses import replace
import os
from unittest.mock import patch

import pytest

from core.memory_clarification import record_choice_clarifications
from core.prefrontal_cortex import PrefrontalCortex
from core.thalamic_reticular_nucleus import (ThalamicReticularNucleus, GateFacts, GateConfig,
    GateMode, CONVERSATION_SELECTIVE_CONSULTATION)
from test_consultation_router import make_stage_controller, make_decision
from test_record_choices import memory, ref_for, begin, Q, META


async def routed(*, mode='auto', selection=False, change=None, draft_change=None, **changes):
    m = memory()
    if selection:
        ref = ref_for(m)
        begin(m, ref)
        m.past_qas[1] = replace(m.past_qas[1], answer='PRIVATE_CHANGED')
    else:
        m.begin_turn(Q, 'clarify', META)
    condition = m.missing_evidence(Q)
    replies = record_choice_clarifications(condition)
    draft = replies[1 if changes.pop('japanese', False) else 0]
    assert all(reply in PrefrontalCortex.clarification_context(condition) for reply in replies)
    if draft_change:
        draft = draft_change(draft)
    if change == 'unique':
        m.past_qas.pop()
    if change == 'extra_clause':
        m.begin_turn(Q + ' Then compare the records.', 'extra', META)
    if change == 'none_current':
        m.past_qas.clear()
    controller = make_stage_controller(memory=m)
    decision = make_decision(action=1); decision.state['q_type'] = 'easy'
    decision.state.update(changes.pop('state', {}))
    parts = {'memory': m.retrieve_related(Q)}
    options = dict(question=Q if change != 'extra_clause' else Q + ' Then compare the records.',
        decision=decision, draft=draft, confidence=.9, affect={'risk':0.}, salience_signal=None,
        context_parts=parts, context_frame=controller._build_context_frame(question=Q, context_parts=parts),
        system2_mode='auto', system2_priority='balanced', system2_active=True, system2_capable=True,
        system2_reason='prior_memory_route', precision_priority=False, system2_low_signal_filter=True,
        director_task=None, director_payload=None, director_control=None, explicit_leading=True,
        force_right_lead=False, is_trivial_chat=False, phase_add=lambda *_:None,
        context_signal_len=0, leading_brain='left')
    options.update(changes)
    with patch.dict(os.environ, {'DUALBRAIN_MODULE_CONVERSATION_SELECTIVE_CONSULTATION_MODE':mode}):
        result = await controller._run_routing_stage(**options)
    return result, controller, m


@pytest.mark.asyncio
@pytest.mark.parametrize('selection', [False, True])
@pytest.mark.parametrize('japanese', [False, True])
async def test_exact_current_record_clarification_skips_without_approval(selection, japanese):
    result, controller, m = await routed(selection=selection, japanese=japanese)
    assert result.decision.action == 0 and not result.system2_active
    route = result.decision.state['selective_consultation']
    assert route['reason'] == 'bounded_record_clarification' and route['approved_records'] == 0
    assert result.system2_reason == 'selective_record_clarification'
    assert not m.answerability_gate.backend.requests and not m.answerability_gate.cache
    assert not m.pending_memory_writes and not m.pending_memory_updates


@pytest.mark.asyncio
async def test_reselection_stays_bounded_after_all_sources_disappear():
    result, _, _ = await routed(selection=True, change='none_current')
    assert result.decision.action == 0 and not result.system2_active


@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [
    {'mode':'off'}, {'system2_mode':'on'}, {'system2_mode':'off'},
    {'precision_priority':True}, {'leading_brain':'right'}, {'force_right_lead':True},
    {'director_control':{'consult':'force'}}, {'state':{'amygdala_override':True}},
    {'state':{'acc_override_consult':True}}, {'confidence':.8}, {'affect':{'risk':.1}},
    {'change':'unique'}, {'change':'extra_clause'},
    {'draft_change':lambda text:text+' The stored period is 91 days.'},
    {'draft_change':lambda text:'According to the record: '+text},
    {'draft_change':lambda _text:'Please paste the entire record here.'},
])
async def test_protected_modes_state_changes_and_noncanonical_answers_do_not_skip(changes):
    result, _, _ = await routed(**changes)
    assert result.decision.action == 1
    assert (result.decision.state.get('selective_consultation') or {}).get('reason') != 'bounded_record_clarification'


def test_gate_requires_complete_typed_facts_without_changing_legacy_default():
    n = ThalamicReticularNucleus()
    facts = GateFacts(eligible=True, has_draft=True, confidence=.9, risk=0.,
                      reference_resolved=True, bounded_clarification_complete=True)
    auto = GateConfig(GateMode.AUTO, 'explicit', True)
    assert n.decide(CONVERSATION_SELECTIVE_CONSULTATION, facts=facts, config=auto).enabled
    for fields in ({'bounded_clarification_complete':'true'}, {'approved_records':1},
                   {'reference_resolved':False}, {'confidence':None}):
        assert not n.decide(CONVERSATION_SELECTIVE_CONSULTATION,
                            facts=replace(facts, **fields), config=auto).enabled
    assert n.configure(CONVERSATION_SELECTIVE_CONSULTATION, environment={}).mode is GateMode.OFF
