import asyncio
import json
from pathlib import Path

import pytest

from core.consultation_router import literal_record_replay_requested
from core.memory_answerability import AnswerabilityGate
from core.shared_memory import SharedMemory, MemoryTrace
from core.thalamic_reticular_nucleus import CONVERSATION_SELECTIVE_CONSULTATION as SELECTIVE
from test_memory_answerability import Checks
from test_dual_brain import make_stage_controller, make_decision


@pytest.mark.parametrize('text', [
    'repeat the recorded answer verbatim', 'quote the stored text exactly',
    'repeat the recorded answer for verbatim.', 'の記録をそのまま再掲して',
    '記録された回答をそのまま再掲してください。',
])
def test_only_positive_literal_replay_shapes_are_admitted(text):
    assert literal_record_replay_requested(text)


@pytest.mark.parametrize('text', [
    'What is retention?', 'repeat the recorded answer and compute its difference',
    'repeat the recorded answer verbatim and verify it', 'summarize the recorded answer',
    '記録をそのまま再掲して、通常と緊急の差も計算して', '記録された条件を比較して', '',
])
def test_extra_requests_and_semantic_transformations_are_not_literal_replay(text):
    assert not literal_record_replay_requested(text)


SOURCE='Normal retention is 11 days. Urgent retention is 2 days.'
REPEAT='Repeat the recorded answer for Project Amber verbatim.'


async def route_stage(controller, question, draft, before_route=None, **changes):
    memory=controller.memory
    memory.begin_turn(question, changes.pop('qid','probe'))
    await memory.prepare_retrieval(question)
    parts={'memory':memory.retrieve_related(question)}
    frame=controller._build_context_frame(question=question,context_parts=parts)
    decision=make_decision(action=1)
    decision.state['q_type']='easy'
    args=dict(question=question,decision=decision,draft=draft,confidence=.9,affect={'risk':0.},
        salience_signal=None,context_parts=parts,context_frame=frame,system2_mode='auto',
        system2_priority='balanced',system2_active=True,system2_capable=True,system2_reason='memory',
        precision_priority=False,system2_low_signal_filter=True,director_task=None,director_payload=None,
        director_control=None,explicit_leading=True,force_right_lead=False,is_trivial_chat=False,
        phase_add=lambda *_:None,context_signal_len=len(parts['memory']),leading_brain='left')
    args.update(changes)
    if before_route is not None:
        before_route()
    return await controller._run_routing_stage(**args)


def controller():
    m=SharedMemory(conversation_context=True,answerability_gate=AnswerabilityGate(Checks()))
    m.store(MemoryTrace('Project Amber retention',SOURCE,qid='primary',retrieval_scope='amber',scope_aliases=('Amber',)))
    return make_stage_controller(memory=m)


@pytest.mark.parametrize('repeat_question', [REPEAT, 'Amberの記録をそのまま再掲して'])
def test_router_automatically_turns_on_and_off_on_successive_turns(monkeypatch, repeat_question):
    monkeypatch.setenv('DUALBRAIN_MODULE_CONVERSATION_SELECTIVE_CONSULTATION_MODE','auto')
    c=controller()
    async def run():
        first=await route_stage(c,repeat_question,SOURCE,qid='first')
        assert first.decision.state['module_gates'][SELECTIVE]['enabled']
        assert first.decision.action==0 and not first.system2_active
        second=await route_stage(c,'What is Project Amber retention?',SOURCE,qid='second')
        gate=second.decision.state['module_gates'][SELECTIVE]
        assert not gate['enabled'] and gate['reason']=='literal_replay_not_requested'
        assert second.decision.action==1 and second.system2_active
    asyncio.run(run())
    assert c.memory.past_qas[0].evidence_status=='unverified'


@pytest.mark.parametrize('draft', [
    'Normal retention is 11 days.',
    'Normal retention is 2 days. Urgent retention is 11 days.',
    'Normal retention is 11 weeks. Urgent retention is 2 weeks.',
])
def test_same_numbers_do_not_establish_complete_replay(monkeypatch,draft):
    monkeypatch.setenv('DUALBRAIN_MODULE_CONVERSATION_SELECTIVE_CONSULTATION_MODE','auto')
    r=asyncio.run(route_stage(controller(),REPEAT,draft))
    assert not r.decision.state['module_gates'][SELECTIVE]['enabled']
    assert r.decision.action==1 and r.system2_active


@pytest.mark.parametrize('changes', [
    {'system2_mode':'on'}, {'system2_mode':'off'}, {'precision_priority':True},
    {'leading_brain':'right'}, {'director_control':{'consult':'force'}},
])
def test_auto_never_overrides_existing_explicit_controls(monkeypatch,changes):
    monkeypatch.setenv('DUALBRAIN_MODULE_CONVERSATION_SELECTIVE_CONSULTATION_MODE','auto')
    r=asyncio.run(route_stage(controller(),REPEAT,SOURCE,**changes))
    assert not r.decision.state['module_gates'][SELECTIVE]['eligible']
    assert 'selective_consultation' not in r.decision.state


@pytest.mark.parametrize('child_action', ['consult', 'defer'])
def test_auto_does_not_apply_child_escalation_or_deferral(monkeypatch, child_action):
    from core.consultation_router import ConsultationRoute
    monkeypatch.setenv('DUALBRAIN_MODULE_CONVERSATION_SELECTIVE_CONSULTATION_MODE','auto')
    monkeypatch.setattr('core.dual_brain.route_conversation_lookup',lambda **_:ConsultationRoute(child_action,'numeric_drift'))
    r=asyncio.run(route_stage(controller(),REPEAT,SOURCE,system2_active=False))
    assert r.decision.state['module_gates'][SELECTIVE]['enabled']
    assert not r.decision.state['selective_consultation']['applied']
    assert r.decision.action==1 and not r.system2_active


def test_canonical_off_overrides_legacy_on(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_CONVERSATION_SELECTIVE_CONSULT','1')
    monkeypatch.setenv('DUALBRAIN_MODULE_CONVERSATION_SELECTIVE_CONSULTATION_MODE','off')
    r=asyncio.run(route_stage(controller(),REPEAT,SOURCE))
    assert r.decision.state['module_gates'][SELECTIVE]['reason']=='forced_off'
    assert r.decision.action==1


def test_off_does_not_prepare_selective_router_facts(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_CONVERSATION_SELECTIVE_CONSULTATION_MODE', 'off')
    c = controller()

    def unexpected(*args, **kwargs):
        raise AssertionError('Disabled router must not prepare admission facts')

    # Normal context preparation still resolves identity before this stage.
    result = asyncio.run(route_stage(c, REPEAT, SOURCE,
        before_route=lambda: monkeypatch.setattr(c.memory, 'reference_resolution', unexpected)))
    assert not result.decision.state['module_gates'][SELECTIVE]['enabled']


def test_gate_telemetry_failure_preserves_admission_and_state():
    from core.thalamic_reticular_nucleus import ACC_MONITOR, GateConfig, GateFacts, GateMode

    class FailingTelemetry:
        def log(self, *args, **kwargs):
            raise RuntimeError('Telemetry unavailable')

    c = controller()
    c.telemetry = FailingTelemetry()
    decision = make_decision(action=0)
    result = c._module_gate(ACC_MONITOR, facts=GateFacts(eligible=True), decision=decision,
        config=GateConfig(GateMode.ON, 'explicit', True))
    assert result.enabled
    assert decision.state['module_gates'][ACC_MONITOR] == result.to_payload()


@pytest.mark.asyncio
async def test_real_engine_uses_auto_admission_without_an_extra_model_classifier(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
    from engine_stdio import EngineSession,_handle_process
    from core.thalamic_reticular_nucleus import DEFAULT_REGISTRY
    for spec in DEFAULT_REGISTRY.values():
        monkeypatch.setenv(spec.mode_env, 'auto')
    monkeypatch.setenv('DUALBRAIN_MEMORY_CONVERSATION_CONTEXT','1')
    monkeypatch.setenv('DUALBRAIN_SYSTEM2_PRIORITY_AUTO','balanced')
    monkeypatch.setenv('DUALBRAIN_MODULE_CONVERSATION_SELECTIVE_CONSULTATION_MODE','auto')
    calls=[]
    class Client:
        def __init__(self,role):self.role=role
        async def complete(self,*args,**kwargs):
            calls.append(self.role)
            return SOURCE if self.role=='left' else json.dumps({'verdict':'ok','issues':[],'fixes':[]})
    session=await EngineSession.create('module-gates')
    try:
        session.memory.store(MemoryTrace('Project Amber retention',SOURCE,qid='primary',
            retrieval_scope='amber',scope_aliases=('Amber',)))
        session.memory.answerability_gate=AnswerabilityGate(Checks())
        session.left._llm_client,session.right._llm_client=Client('left'),Client('right')
        session.left.uses_external_llm=session.right.uses_external_llm=True
        options=dict(blob_root=tmp_path,blob_index={},vision_cache={},vision_cache_limit=0)
        first=await _handle_process(session,{'question':REPEAT,'qid':'first','leading_brain':'left','system2_mode':'auto'},**options)
        assert first['metrics']['modules']['gates'][SELECTIVE]['enabled']
        assert first['metrics']['policy']['consultation_route']['applied']
        assert calls==['left']
        second=await _handle_process(session,{'question':'Compare normal and urgent retention for Project Amber, and explain the difference between those conditions.',
            'qid':'second','leading_brain':'left','system2_mode':'auto'},**options)
        assert not second['metrics']['modules']['gates'][SELECTIVE]['enabled']
        assert 'right' in calls
        text=json.dumps(second['metrics']['modules']['gates'])
        assert 'Amber' not in text and SOURCE not in text and 'primary' not in text
    finally:
        await session.close()


def test_shared_conflict_modules_have_independent_auto_gates(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_ACC_MONITOR_MODE','auto')
    monkeypatch.setenv('DUALBRAIN_MODULE_CEREBELLAR_CORRECTION_MODE','auto')
    class Left:
        uses_external_llm=True
        integrations=0
        async def integrate_info_async(self,**kwargs):
            self.integrations+=1
            return '4'
    left=Left()
    c=make_stage_controller(left_model=left)
    async def regulate(question,draft):
        async def reset():pass
        decision=make_decision(action=0)
        result=await c._run_conflict_regulation_stage(question=question,decision=decision,
            final_answer=draft,confidence=.95,detail_notes=None,system2_active=False,system2_mode='auto',
            is_trivial_chat=False,emit_reset_if_needed=reset,delta_cb=None,stream_final_only=False,phase_add=lambda *_:None)
        return result,decision.state['module_gates']
    async def run():
        _,first=await regulate('Tell me a greeting.','Hello.')
        assert not first['acc_monitor']['enabled'] and not first['cerebellar_correction']['enabled']
        result,second=await regulate('Compute 2+2?','2+2=5')
        assert second['acc_monitor']['enabled'] and second['cerebellar_correction']['enabled']
        assert result.final_answer=='4' and left.integrations==1
    asyncio.run(run())
