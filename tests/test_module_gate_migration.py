"""Exercise admission at its production call sites without provider requests."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from core.anterior_cingulate import ConflictSignal
from core.default_mode_network import DefaultModeNetwork
from core.prefrontal_cortex import FocusSummary, PrefrontalCortex
from core.schema import UnconsciousSummaryModel
from core.thalamic_reticular_nucleus import (
    DEFAULT_REGISTRY, DEFAULT_MODE_REFLECTION_INJECTION as DMN,
    SYSTEM2_STALLED_FOLLOWUP as STALLED, EXECUTIVE_ADVICE as EXECUTIVE,
    DIRECTOR_ADVICE as DIRECTOR, EXECUTIVE_METRICS_OBSERVATION as METRICS,
    ACC_CONSULT_PROMOTION as ACC_CONSULT, ACC_SYSTEM2_PROMOTION as ACC_SYSTEM2,
    ACC_TEMPERATURE_ADJUSTMENT as ACC_TEMPERATURE,
    BASAL_CONSULT_PROMOTION as BASAL,
)
from test_dual_brain import (
    InstantExecutiveModel, InstantDirectorModel, make_stage_controller, make_decision,
)
from test_module_gate_integration import route_stage


def configure(monkeypatch, module, mode):
    monkeypatch.setenv(DEFAULT_REGISTRY[module].mode_env, mode)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['off', 'on', 'auto'])
@pytest.mark.parametrize('selected', [False, True])
@pytest.mark.parametrize('trivial', [False, True])
async def test_advice_and_director_gate_before_creating_owned_tasks(monkeypatch, mode, selected, trivial):
    c = make_stage_controller()
    c.executive_model, c.director_model = InstantExecutiveModel(), InstantDirectorModel()
    for module in (EXECUTIVE, DIRECTOR):
        configure(monkeypatch, module, mode)
    setup = c._prepare_observer_stage(
        qid='private-turn', question='private question', context='', focus=None,
        focus_keywords=[], context_parts={}, is_trivial_chat=trivial,
        hemisphere_mode='balanced', hemisphere_bias=0., leading='left', collaborative_lead=False,
        collaboration_profile=SimpleNamespace(strength=0.),
        executive_mode='assist' if selected else 'off',
        executive_observer_mode='both' if selected else 'off',
    )
    expected = selected and mode != 'off' and (mode != 'auto' or not trivial)
    tasks = [t for t in (setup.executive_task, setup.director_task) if t is not None]
    try:
        assert bool(setup.executive_task) == bool(setup.director_task) == expected
        assert setup.executive_mode == ('assist' if selected else 'off')
        assert all(gate['enabled'] == expected for gate in setup.module_gates.values())
        assert 'private' not in json.dumps(setup.module_gates)
    finally:
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['off', 'on', 'auto'])
@pytest.mark.parametrize('question,expected', [('hi', False), ('Explain how waves propagate.', True)])
async def test_metrics_modes_control_the_post_answer_call(monkeypatch, mode, question, expected):
    configure(monkeypatch, METRICS, mode)
    expected = mode != 'off' and (mode != 'auto' or expected)
    c = make_stage_controller(prefrontal_cortex=PrefrontalCortex())
    calls = []

    class Executive(InstantExecutiveModel):
        async def advise(self, **kwargs):
            calls.append(kwargs)
            return await super().advise(**kwargs)

    c.executive_model = Executive()
    await c.process(question, leading_brain='left', system2_mode='off', executive_observer_mode='metrics')
    gates = [payload for event, payload in c.telemetry.events if event == 'module_gate']
    metrics_gate = next(g for g in gates if g['module'] == METRICS)
    assert metrics_gate['enabled'] == expected
    assert len(calls) == int(expected)
    assert next(g for g in gates if g['module'] == EXECUTIVE)['reason'] == 'caller_mode_off'


@pytest.mark.asyncio
@pytest.mark.parametrize('question,expected', [('hi', False), ('Describe tidal motion.', True)])
async def test_all_auto_preserves_requested_advice_and_director_modes(monkeypatch, question, expected):
    for module in DEFAULT_REGISTRY:
        configure(monkeypatch, module, 'auto')
    c = make_stage_controller(prefrontal_cortex=PrefrontalCortex())
    calls = []

    class Executive(InstantExecutiveModel):
        async def advise(self, **kwargs):
            calls.append('executive')
            return await super().advise(**kwargs)

    class Director(InstantDirectorModel):
        async def advise(self, **kwargs):
            calls.append('director')
            return await super().advise(**kwargs)

    c.executive_model, c.director_model = Executive(), Director()
    original_route = c._run_routing_stage
    captured = {}

    async def route(**kwargs):
        captured.update(kwargs['decision'].state['module_gates'])
        return await original_route(**kwargs)

    c._run_routing_stage = route
    await c.process(question, leading_brain='left', system2_mode='off',
                    executive_mode='observe', executive_observer_mode='both')
    gates = {p['module']: p for event, p in c.telemetry.events if event == 'module_gate'}
    assert calls.count('executive') == (2 if expected else 0)
    assert calls.count('director') == int(expected)
    for module in (EXECUTIVE, DIRECTOR, METRICS):
        assert gates[module]['enabled'] == expected
    assert captured[EXECUTIVE]['enabled'] == captured[DIRECTOR]['enabled'] == expected


class Unconscious:
    def analyse(self, **kwargs):
        return None

    def summary(self, profile):
        return UnconsciousSummaryModel(
            top_k=['self'], archetype_map=[{'id': 'self', 'label': 'Self', 'intensity': .95}],
            emergent_ideas=[], stress_released=0., cache_depth=7, psychoid_signal=None,
        )


def reflect(c, *, risk=0., suppressed=False):
    decision = make_decision()
    decision.state['q_type'] = 'easy' if suppressed else 'medium'
    result = c._run_unconscious_resonance_stage(
        question='Imagine an ocean journey.', draft='A boat crosses the sea.', decision=decision,
        focus=FocusSummary(('sea',), .0 if suppressed else .5, .0),
        affect={'arousal': 0., 'risk': risk}, salience_signal=None, acc_signal=None,
        basal_signal=None, predictive_frame=None, thalamic_relay=None,
    )
    return result, decision.state['module_gates'][DMN]


def test_dmn_off_preserves_history_and_cooldown_and_on_preserves_suppression(monkeypatch):
    c = make_stage_controller(unconscious_field=Unconscious())
    c.default_mode_network = DefaultModeNetwork(min_cache_depth=0, activation_bias=0., cooldown_steps=1)
    configure(monkeypatch, DMN, 'off')
    first, gate = reflect(c)
    assert first.default_mode_reflections and first.suppress_default_mode_reflections
    assert not gate['enabled']
    configure(monkeypatch, DMN, 'on')
    second, gate = reflect(c)
    assert not second.default_mode_reflections and not gate['enabled']
    third, gate = reflect(c)
    assert third.default_mode_reflections and gate['enabled']
    assert len(c.default_mode_network._history) == 3
    c.default_mode_network.cooldown_steps = 0
    c.default_mode_network._cooldown = 0
    c._default_mode_low_focus_streak = 1
    fourth, gate = reflect(c, suppressed=True)
    assert fourth.default_mode_reflections and not gate['eligible'] and not gate['enabled']
    assert gate['reason'] == 'default_mode_suppressed'


def test_dmn_auto_turns_injection_off_on_risk_without_losing_reflections(monkeypatch):
    configure(monkeypatch, DMN, 'auto')
    c = make_stage_controller(unconscious_field=Unconscious())
    c.default_mode_network = DefaultModeNetwork(min_cache_depth=0, activation_bias=0., cooldown_steps=0)
    assert reflect(c)[1]['enabled']
    result, gate = reflect(c, risk=.2)
    assert result.default_mode_reflections and result.suppress_default_mode_reflections
    assert not gate['enabled'] and gate['reason'] == 'reflection_signal_low'


@pytest.mark.parametrize('system2_mode', ['auto', 'on', 'off'])
def test_acc_interventions_are_independent_and_preserve_explicit_system2(monkeypatch, system2_mode):
    for module in (ACC_CONSULT, ACC_SYSTEM2, ACC_TEMPERATURE):
        configure(monkeypatch, module, 'auto')
    monkeypatch.setenv('DUALBRAIN_ACC_TEMPERATURE_DROP', '.2')
    c = make_stage_controller()
    c.anterior_cingulate_cortex.monitor = lambda **_: ConflictSignal(.95)
    decision = make_decision(action=0)
    decision.temperature = .55
    result = asyncio.run(route_stage(c, 'Compute 2+2', '2+2=5', decision=decision,
        system2_active=False, system2_mode=system2_mode))
    gates = result.decision.state['module_gates']
    assert gates[ACC_CONSULT]['enabled'] and gates[ACC_TEMPERATURE]['enabled']
    assert gates[ACC_SYSTEM2]['enabled'] == (system2_mode == 'auto')
    assert bool(result.decision.state.get('acc_system2_bump')) == (system2_mode == 'auto')
    assert result.decision.state['acc_override_consult']
    assert result.decision.state['acc_temperature']['adjusted'] < .55


def test_acc_canonical_off_overrides_legacy_and_avoids_signal_work(monkeypatch):
    for module in (ACC_CONSULT, ACC_SYSTEM2, ACC_TEMPERATURE):
        configure(monkeypatch, module, 'off')
    monkeypatch.setenv('DUALBRAIN_ACC_OVERRIDE_CONSULT', '1')
    monkeypatch.setenv('DUALBRAIN_ACC_SYSTEM2_BUMP', '1')
    monkeypatch.setenv('DUALBRAIN_ACC_TEMPERATURE_DROP', '.2')
    c = make_stage_controller()
    calls = []

    def unexpected(**kwargs):
        calls.append(kwargs)
        raise AssertionError('No optional ACC signal work when all controls are off')

    monkeypatch.setattr('core.dual_brain.micro_criticise_reasoning', unexpected)
    result = asyncio.run(route_stage(c, 'Compute 2+2', '2+2=5', system2_active=False))
    assert all(not result.decision.state['module_gates'][module]['enabled']
               for module in (ACC_CONSULT, ACC_SYSTEM2, ACC_TEMPERATURE))
    assert not calls


@pytest.mark.parametrize('mode,conflict,expected', [('off', .95, False), ('auto', .95, True), ('auto', .7, False)])
def test_basal_consult_gate_cannot_reintroduce_a_declined_weak_acc_intervention(monkeypatch, mode, conflict, expected):
    configure(monkeypatch, BASAL, mode)
    configure(monkeypatch, ACC_CONSULT, 'off')
    configure(monkeypatch, ACC_SYSTEM2, 'off')
    configure(monkeypatch, ACC_TEMPERATURE, 'auto')
    monkeypatch.setenv('DUALBRAIN_ACC_TEMPERATURE_DROP', '.2')
    monkeypatch.setenv('DUALBRAIN_ACC_CONFLICT_THRESHOLD', '.6')
    c = make_stage_controller()
    c.anterior_cingulate_cortex.monitor = lambda **_: ConflictSignal(conflict, recommended_control='consult')
    c.basal_ganglia.evaluate = lambda **_: SimpleNamespace(recommended_action=1, to_dict=lambda: {})
    result = asyncio.run(route_stage(c, 'Compute 2+2', '2+2=5', decision=make_decision(action=0), system2_active=False))
    assert result.decision.state['module_gates'][BASAL]['enabled'] == expected
    assert bool(result.decision.state.get('basal_loop_override')) == expected
    assert result.decision.action == int(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize('progress', [False, True])
@pytest.mark.parametrize('mode,confidence,budget,expected', [
    ('auto', .9, 3, True), ('off', .9, 3, False), ('auto', None, 3, False),
    ('auto', .9, 2, False), ('on', None, 3, True),
])
async def test_stalled_followup_controls_only_the_stalled_override(monkeypatch, mode, confidence, budget, expected, progress):
    configure(monkeypatch, STALLED, mode)
    c = make_stage_controller()
    decision = make_decision(action=1)
    issues = ['Tax rate omits the exemption threshold.',
              'Queue capacity exceeds the memory budget.',
              'Shutdown leaks the running worker.',
              'Currency conversion uses the wrong exchange date.']
    decision.state.update(critic_verdict='issues', critic_kind='external', critic_sum='Four constraints need repair.')
    calls = []

    async def critic(payload, **kwargs):
        calls.append(payload['round'])
        return {'verdict': 'issues' if payload['round'] == 2 else 'ok',
                'issues': (issues[:-1] if progress else issues) if payload['round'] == 2 else [],
                'critic_sum': 'Four constraints need repair.' if payload['round'] == 2 else 'No issues detected.',
                'critic_kind': 'external', 'confidence_r': confidence}

    async def reset():
        pass

    result = await c._run_system2_refinement_stage(
        question='Explain the policy.', decision=decision, draft='A draft.', final_answer='A revised draft.',
        payload={}, context='', psychoid_projection=None, system2_mode='auto',
        system2_round_target=budget, system2_draft_limit=1600, timeout_ms_base=3000,
        timeout_multiplier=1., timeout_max_ms=9000, system2_priority='balanced',
        system2_low_signal_filter=False, system2_followup_new_issue_cap=2,
        system2_followup_new_issue_min_score=1., system2_followup_min_overlap=.5,
        system2_followup_max_remaining=4, system2_followup_min_progress=1,
        critic_issues=issues, system2_initial_issue_count=4, critic_needs_revision=True,
        system2_critic_unhealthy=False, system2_critic_unhealthy_reason=None,
        ask_callosum_with_timeout=critic, emit_reset_if_needed=reset, delta_cb=None,
        stream_final_only=False, phase_add=lambda *_: None,
    )
    followup_expected = budget >= 3 and (progress or expected)
    assert decision.state['module_gates'][STALLED]['enabled'] == (expected and not progress)
    assert result.system2_followup_revision == followup_expected
    assert calls == ([2, 3] if followup_expected else [2])
    assert len(c.left_model.integrations) == int(followup_expected)
