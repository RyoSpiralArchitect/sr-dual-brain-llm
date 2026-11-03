from dataclasses import replace

import pytest

from core.director_controls import build_local_director_plan
from core.director_reasoner import DirectorAdvice
from core.thalamic_reticular_nucleus import DIRECTOR_ADVICE
from core.thalamus import ThalamicRelay
from test_advisor_timing import BlockingDirector, setup
from test_dual_brain import CaptureLeftModel, OkCriticCallosum, make_stage_controller


def relay(*, notes=('memory_gate:suppress_long_term',), **overrides):
    return replace(ThalamicRelay(target_network='executive_control', relay_gain=.8,
        keep_working_memory=True, keep_long_term_memory=False, keep_schema_memory=False,
        keep_pitfall_memory=False, keep_hippocampal_memory=False,
        suppress_default_mode=False, notes=notes), **overrides)


async def pre_draft(c, stage, *, parts=None, trivial=False, plan=None):
    parts = {} if parts is None else parts
    frame = c._build_context_frame(question='Explain the current constraints.', context_parts=parts)
    return await c._run_pre_draft_director_stage(question='Explain the current constraints.',
        qid='local-controls', is_trivial_chat=trivial, context_frame=frame,
        context_parts=parts, director_task=stage.director_task,
        director_requested=stage.use_director, phase_add=lambda *_:None,
        local_director_plan=stage.local_director_plan if plan is None else plan)


@pytest.mark.asyncio
async def test_auto_known_controls_bypass_remote_and_apply_before_the_actual_primary_draft(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE', 'auto')
    observed = []
    c = None

    class InspectingLeft(CaptureLeftModel):
        async def generate_answer(self, input_text, context, *, vision_images=None, on_delta=None):
            local = [p for e,p in c.telemetry.events if e=='director_local_controls']
            assert local and local[-1]['applied_before_draft'] is True
            assert local[-1]['source']=='local_facts' and local[-1]['controls']['consult']=='force'
            assert not [p for e,p in c.telemetry.events if e=='advisor_work' and p.get('module')==DIRECTOR_ADVICE]
            observed.append(context)
            return 'The supplied constraints remain explicit.'

    c = make_stage_controller(left_model=InspectingLeft(confidence=.9))
    c.director_model = model = BlockingDirector()
    c.right_model.uses_external_llm = True
    c.callosum = OkCriticCallosum()
    routing = c._run_routing_stage
    routed = []

    async def capture_routing(**kwargs):
        result = await routing(**kwargs)
        routed.append(result)
        return result

    c._run_routing_stage = capture_routing
    result = await c.process('Explain why an independently checked plan must retain its supplied constraints.',
        system2_mode='on', executive_observer_mode='director')
    assert result and observed and model.calls==0
    gates = [p for e,p in c.telemetry.events if e=='module_gate' and p.get('module')==DIRECTOR_ADVICE]
    assert gates[-1]['reason']=='local_controls_ready' and gates[-1]['enabled'] is False
    assert routed and routed[0].director_control['consult']=='force'
    assert routed[0].decision.state['director_consult']=='force'
    assert c.callosum.payloads and c.callosum.payloads[0]['payload']['type']=='ASK_CRITIC'
    assert c.hypothalamus.advisor_timing._history=={}


@pytest.mark.parametrize('mode', [None, 'on', 'off'])
@pytest.mark.asyncio
async def test_manual_and_default_modes_keep_the_custom_remote_api_while_off_creates_no_work(monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE', raising=False)
    else:
        monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE', mode)
    calls = []

    class StrictDirector:
        async def advise(self, *, question, context, signals=None, temperature=.15):
            calls.append((question, signals))
            return DirectorAdvice('Existing custom API.', {'max_chars':600}, .8, 1., 'test')

    c = make_stage_controller()
    c.director_model = StrictDirector()
    stage = setup(c, system2_mode='on', system2_capable=True, system2_active=True)
    assert stage.local_director_plan is None
    if mode=='off':
        assert stage.director_task is None and not calls
    else:
        result = await c._wait_advisor(stage.director_task, timeout_ms=1600.)
        assert result.control=={'max_chars':600} and len(calls)==1
    assert c.hypothalamus.advisor_timing._history=={}


@pytest.mark.parametrize('candidate', [None, relay(notes=()), relay(keep_schema_memory=True),
    relay(notes=('memory_gate:working_only', 'fallback_long_term')),
    relay(keep_long_term_memory=0)])
def test_absent_or_mixed_relay_and_system2_off_leave_ordinary_consultation_neutral(monkeypatch, candidate):
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE', 'auto')
    c = make_stage_controller()
    plan = c._local_director_plan(system2_mode='off', system2_capable=True,
        system2_active=False, thalamic_relay=candidate)
    assert not plan.nonneutral and plan.controls=={}
    assert plan.merge({'consult':'force', 'memory':{'working_memory':'keep'}})=={
        'consult':'force', 'memory':{'working_memory':'keep'}}


@pytest.mark.parametrize('notes', [('memory_gate:suppress_long_term',), ('memory_gate:working_only',)])
@pytest.mark.asyncio
async def test_existing_all_stream_restriction_applies_before_drafting_without_dropping_working_memory(monkeypatch, notes):
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE', 'auto')
    c = make_stage_controller()
    c.director_model = model = BlockingDirector()
    parts = {'working_memory':'Current working context.', 'memory':'Old source.',
             'schema':'Old schema.', 'pitfalls':'Old pitfalls.', 'hippocampal':'Old episode.', 'replay':'Old replay.'}
    stage = setup(c, context_parts=parts, thalamic_relay=relay(notes=notes))
    assert stage.director_task is None and stage.local_director_plan.nonneutral
    result = await pre_draft(c, stage, parts=parts)
    assert result.director_control=={'memory':{'long_term':'drop'}}
    assert result.director_payload['source']=='local_facts'
    assert parts['working_memory']=='Current working context.'
    assert all(parts[key]=='' for key in ('memory','schema','pitfalls','hippocampal','replay'))
    assert model.calls==0 and c.hypothalamus.advisor_timing._history=={}


@pytest.mark.asyncio
async def test_expired_deadline_cannot_apply_a_previously_derived_local_plan(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE', 'auto')
    c = make_stage_controller()
    c.director_model = model = BlockingDirector()
    plan = build_local_director_plan(system2_mode='on', system2_capable=True, system2_active=True)
    monkeypatch.setattr('core.dual_brain.remaining_turn_ms', lambda:0.)
    stage = setup(c, system2_mode='on', system2_capable=True, system2_active=True)
    assert stage.local_director_plan is None and stage.director_task is None
    assert stage.module_gates[DIRECTOR_ADVICE]['reason']=='turn_deadline_elapsed'
    result = await pre_draft(c, stage, plan=plan)
    assert result.director_control is None and model.calls==0
    assert not [p for e,p in c.telemetry.events if e=='director_local_controls']


@pytest.mark.asyncio
async def test_existing_trivial_guard_remains_separate_from_new_local_fact_controls(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE', 'auto')
    c = make_stage_controller()
    c.director_model = model = BlockingDirector()
    stage = setup(c, is_trivial_chat=True, system2_mode='on', system2_capable=True, system2_active=True)
    assert stage.local_director_plan is None and stage.director_task is None
    result = await pre_draft(c, stage, trivial=True)
    assert result.director_payload['source']=='fast_path'
    assert result.director_control['consult']=='skip'
    assert model.calls==0 and c.hypothalamus.advisor_timing._history=={}
    assert not [p for e,p in c.telemetry.events if e=='director_local_controls']
