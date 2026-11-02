import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from core.advisor_timing import AdvisorTiming, model_profile
from core.director_reasoner import DirectorAdvice
from core.executive_reasoner import ExecutiveAdvice
from core.llm_client import LLMConfig
from core.thalamic_reticular_nucleus import (
    ThalamicReticularNucleus, GateFacts, GateConfig, GateMode,
    DIRECTOR_ADVICE as DIRECTOR, EXECUTIVE_ADVICE as EXECUTIVE,
    EXECUTIVE_METRICS_OBSERVATION as METRICS,
)
from core.turn_lifecycle import cancel_task, TurnDispatcher, remaining_turn_ms, begin_turn_commit, turn_abandoned
from test_dual_brain import make_stage_controller, AlwaysConsultPolicy


PROFILE = 'a' * 64


def test_recent_max_retains_censoring_and_expiry_per_role_and_model():
    now = [0.]
    ledger = AdvisorTiming(clock=lambda: now[0])
    ledger.record(DIRECTOR, PROFILE, 300.)
    ledger.record(DIRECTOR, PROFILE, 1600., censored=True)
    estimate = ledger.estimate(DIRECTOR, PROFILE)
    assert (estimate.ms, estimate.kind, estimate.completed_samples, estimate.censored_samples) == (1600., 'censored_bound', 1, 1)
    assert ledger.estimate(EXECUTIVE, PROFILE).ms is None
    assert ledger.estimate(DIRECTOR, 'b' * 64).ms is None
    now[0] = 60.
    assert ledger.estimate(DIRECTOR, PROFILE).ms is None


def test_history_and_profile_counts_are_bounded_and_ignore_invalid_observations():
    ledger = AdvisorTiming(max_profiles=2, max_samples=2)
    for duration in (900., 400., 100.):
        ledger.record(DIRECTOR, PROFILE, duration)
    assert ledger.estimate(DIRECTOR, PROFILE).ms == 400.
    for invalid in (True, -1, float('nan'), float('inf'), 'private', 10 ** 1000):
        ledger.record(DIRECTOR, PROFILE, invalid)
    assert ledger.estimate(DIRECTOR, PROFILE).completed_samples == 2
    ledger.record(DIRECTOR, 'b' * 64, 1.)
    ledger.record(DIRECTOR, 'c' * 64, 1.)
    assert ledger.estimate(DIRECTOR, PROFILE).ms is None
    assert len(ledger._history) == 2


def test_profile_uses_effective_model_options_without_auth_or_conversation():
    config = LLMConfig('openai', 'model-a', 'PRIVATE_KEY', api_base='https://example.invalid')
    model = SimpleNamespace(uses_external_llm=True, llm_config=config)
    profile = model_profile(model)
    model.llm_config = replace(config, api_key='DIFFERENT_KEY', extra_headers={'Private': 'VALUE'})
    assert model_profile(model) == profile
    model.llm_config = replace(config, model='model-b')
    assert model_profile(model) != profile
    model._llm_client = SimpleNamespace(config=config)
    assert model_profile(model) == profile
    model.uses_external_llm = False
    assert model_profile(model) is None


@pytest.mark.parametrize('module', [DIRECTOR, EXECUTIVE, METRICS])
def test_time_policy_modes_unknown_fallback_and_reserve(module):
    nucleus = ThalamicReticularNucleus()
    facts = GateFacts(eligible=True, has_input=True, has_draft=True,
                      remaining_budget_ms=1600., estimated_latency_ms=1600.)
    decide = lambda mode, f=facts: nucleus.decide(module, facts=f, config=GateConfig(mode, 'explicit', True))
    assert decide(GateMode.ON).enabled and not decide(GateMode.OFF).enabled
    assert decide(GateMode.AUTO).reason == 'insufficient_time_budget'
    assert decide(GateMode.AUTO, replace(facts, estimated_latency_ms=1500.)).enabled
    assert not decide(GateMode.AUTO, replace(facts, estimated_latency_ms=1550.)).enabled
    assert decide(GateMode.AUTO, replace(facts, estimated_latency_ms=None)).enabled
    assert decide(GateMode.AUTO, replace(facts, remaining_budget_ms=None)).enabled
    assert not decide(GateMode.AUTO, replace(facts, remaining_budget_ms=50., estimated_latency_ms=None)).enabled
    assert not decide(GateMode.ON, replace(facts, eligible=False, blocked_reason='turn_deadline_elapsed')).enabled
    for value in (-1., float('nan'), float('inf'), True):
        assert not decide(GateMode.AUTO, replace(facts, estimated_latency_ms=value)).enabled


class BlockingDirector:
    uses_external_llm = True
    llm_config = LLMConfig('openai', 'timing-test', 'PRIVATE_KEY')

    def __init__(self):
        self.started = asyncio.Event()
        self.calls = 0

    async def advise(self, **kwargs):
        self.calls += 1
        self.started.set()
        await asyncio.Event().wait()


def setup(c, **overrides):
    args = dict(qid='timing-probe', question='PRIVATE_QUESTION', context='', focus=None, focus_keywords=[],
                context_parts={}, is_trivial_chat=False, hemisphere_mode='balanced', hemisphere_bias=0.,
                leading='left', collaborative_lead=False, collaboration_profile=SimpleNamespace(strength=0.),
                executive_mode='off', executive_observer_mode='director')
    args.update(overrides)
    return c._prepare_observer_stage(**args)


@pytest.mark.asyncio
async def test_cold_probe_timeout_then_warm_inhibition_and_expiry(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE', 'auto')
    elapsed, age = [0.], [0.]
    monkeypatch.setattr('core.dual_brain.time.perf_counter', lambda: elapsed[0])
    c, model = make_stage_controller(), BlockingDirector()
    c.director_model = model
    c.hypothalamus.advisor_timing = AdvisorTiming(clock=lambda: age[0])
    first = setup(c)
    await model.started.wait()
    elapsed[0] = 2.
    with pytest.raises(asyncio.TimeoutError):
        await c._wait_advisor(first.director_task, timeout_ms=1.)
    estimate = c.hypothalamus.advisor_timing.estimate(DIRECTOR, model_profile(model))
    assert estimate.ms == 2000. and estimate.kind == 'censored_bound'
    second = setup(c)
    assert second.director_task is None
    assert second.module_gates[DIRECTOR]['reason'] == 'insufficient_time_budget'
    assert model.calls == 1
    age[0] = 60.
    third = setup(c)
    assert third.director_task is not None
    c._cancel_advisor(third.director_task, 'test_cleanup')
    await asyncio.gather(third.director_task, return_exceptions=True)
    public = [{k:v for k,v in p.items() if k != 'qid'} for event,p in c.telemetry.events
              if event in {'advisor_work', 'advisor_budget'}]
    assert 'PRIVATE_KEY' not in json.dumps(public) and 'PRIVATE_QUESTION' not in json.dumps(public)


@pytest.mark.asyncio
async def test_direct_owner_cancellation_does_not_learn_cleanup_as_model_latency(monkeypatch):
    clock = [0.]
    monkeypatch.setattr('core.dual_brain.time.perf_counter', lambda: clock[0])
    c, model = make_stage_controller(), BlockingDirector()
    cleaning, release = asyncio.Event(), asyncio.Event()

    async def advise():
        model.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            clock[0] = 12.

    task = c._start_advisor(DIRECTOR, model, qid='cleanup', factory=advise)
    await model.started.wait()
    clock[0] = 2.
    cancel_task(task)  # Uses the existing lifecycle owner, not a controller helper.
    await cleaning.wait()
    cancel_task(task)
    release.set()
    await asyncio.gather(task, return_exceptions=True)
    estimate = c.hypothalamus.advisor_timing.estimate(DIRECTOR, model_profile(model))
    assert estimate.ms == 2000.
    terminal = [p for event,p in c.telemetry.events if event == 'advisor_work' and p['state'] != 'started']
    assert len(terminal) == 1 and terminal[0]['cleanup_ms'] == 10000.


@pytest.mark.asyncio
async def test_usable_completion_trains_but_model_error_and_prestart_cancellation_do_not(monkeypatch):
    clock = [0.]
    monkeypatch.setattr('core.dual_brain.time.perf_counter', lambda: clock[0])
    c, model = make_stage_controller(), BlockingDirector()

    async def completed():
        clock[0] += .3
        return DirectorAdvice('Valid guidance.', {'consult':'auto'}, .9, 300., 'llm')

    task = c._start_advisor(DIRECTOR, model, qid='complete', factory=completed)
    await c._wait_advisor(task, timeout_ms=1600.)
    assert c.hypothalamus.advisor_timing.estimate(DIRECTOR, model_profile(model)).ms == 300.

    async def error():
        return ExecutiveAdvice('PRIVATE_ERROR', {}, 0., 1., 'error')

    task = c._start_advisor(EXECUTIVE, model, qid='error', factory=error)
    await c._wait_advisor(task, timeout_ms=1000.)
    assert c.hypothalamus.advisor_timing.estimate(EXECUTIVE, model_profile(model)).ms is None
    task = c._start_advisor(METRICS, model, qid='prestart', factory=model.advise)
    cancel_task(task)
    await asyncio.gather(task, return_exceptions=True)
    assert model.calls == 0 and c.hypothalamus.advisor_timing.estimate(METRICS, model_profile(model)).ms is None
    assert any(p.get('state') == 'not_started' and p['usage_status'] == 'not_started'
               for event,p in c.telemetry.events if event == 'advisor_work')


@pytest.mark.asyncio
async def test_output_returned_during_cancel_cleanup_remains_censored_and_unusable(monkeypatch):
    clock = [0.]
    monkeypatch.setattr('core.dual_brain.time.perf_counter', lambda: clock[0])
    c, model = make_stage_controller(), BlockingDirector()

    async def late_output():
        model.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            clock[0] = 8.
            return DirectorAdvice('Late guidance.', {}, .9, 8000., 'llm')

    task = c._start_advisor(DIRECTOR, model, qid='late', factory=late_output)
    await model.started.wait()
    clock[0] = 2.
    with pytest.raises(asyncio.TimeoutError):
        await c._wait_advisor(task, timeout_ms=1.)
    estimate = c.hypothalamus.advisor_timing.estimate(DIRECTOR, model_profile(model))
    assert estimate.ms == 2000. and estimate.kind == 'censored_bound'
    terminal = [p for event,p in c.telemetry.events if event == 'advisor_work' and p['state'] != 'started']
    assert len(terminal) == 1 and terminal[0]['state'] == 'timed_out'
    assert terminal[0]['usable_output'] is False and terminal[0]['cleanup_ms'] == 6000.


@pytest.mark.asyncio
async def test_executive_overlap_is_not_assumed_to_have_only_its_later_wait_budget(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_EXECUTIVE_ADVICE_MODE', 'auto')
    c, model = make_stage_controller(), BlockingDirector()
    c.executive_model = model
    c.hypothalamus.advisor_timing.record(EXECUTIVE, model_profile(model), 5000.)
    result = setup(c, executive_mode='observe', executive_observer_mode='off')
    assert result.executive_task is not None  # No outer RPC deadline is known.
    c._cancel_advisor(result.executive_task, 'test_cleanup')
    await asyncio.gather(result.executive_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_rpc_remaining_budget_is_read_only_and_exhaustion_vetoes_manual_on(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE', 'on')
    responses, seen = [], []
    c = make_stage_controller()
    c.director_model = BlockingDirector()
    assert remaining_turn_ms() is None

    async def handle(req):
        seen.append(remaining_turn_ms())
        begin_turn_commit()  # Deadline cancellation is deferred while finishing a write.
        dispatcher.cancel(req['id'], 'DeadlineExceeded')
        assert remaining_turn_ms() == 0.
        result = setup(c)
        assert result.director_task is None
        assert result.module_gates[DIRECTOR]['reason'] == 'turn_deadline_elapsed'
        return {'id':req['id'], 'ok':True}

    dispatcher = TurnDispatcher(handle, responses.append, timeout_seconds=2.)
    try:
        dispatcher.submit({'id':'deadline', 'method':'process'})
        while not responses:
            await asyncio.sleep(.001)
        assert 0 < seen[0] <= 2000.
        assert responses[0]['error']['type'] == 'DeadlineExceeded'
        assert c.director_model.calls == 0
    finally:
        await dispatcher.close()


@pytest.mark.asyncio
async def test_local_advisor_timeouts_do_not_consume_the_turn_cancellation_guard():
    c, model, responses = make_stage_controller(), BlockingDirector(), []

    async def delayed():
        await asyncio.sleep(.06)
        return DirectorAdvice('Later output.', {}, .9, 60., 'llm')

    async def handle(req):
        for module in (DIRECTOR, METRICS):
            task = c._start_advisor(module, model, qid=req['id'], factory=delayed)
            with pytest.raises(asyncio.TimeoutError):
                await c._wait_advisor(task, timeout_ms=5.)
        return {'id':req['id'], 'ok':True}

    dispatcher = TurnDispatcher(handle, responses.append, timeout_seconds=1.)
    try:
        dispatcher.submit({'id':'successive-timeouts','method':'process'})
        while not responses:
            await asyncio.sleep(.001)
        assert responses[0]['ok']
        outcomes = [p['state'] for event,p in c.telemetry.events if event == 'advisor_work' and p['state'] != 'started']
        assert outcomes == ['timed_out', 'timed_out']
    finally:
        await dispatcher.close()


@pytest.mark.asyncio
async def test_rpc_deadline_still_cancels_after_an_optional_advisor_timeout():
    c, model, responses, continued = make_stage_controller(), BlockingDirector(), [], []

    async def delayed():
        await asyncio.sleep(.1)

    async def handle(req):
        task = c._start_advisor(DIRECTOR, model, qid=req['id'], factory=delayed)
        with pytest.raises(asyncio.TimeoutError):
            await c._wait_advisor(task, timeout_ms=5.)
        await asyncio.sleep(.1)
        continued.append(True)
        return {'id':req['id'], 'ok':True}

    dispatcher = TurnDispatcher(handle, responses.append, timeout_seconds=.03)
    try:
        dispatcher.submit({'id':'deadline-after-timeout','method':'process'})
        while not responses:
            await asyncio.sleep(.001)
        assert responses[0]['error']['type'] == 'DeadlineExceeded'
        assert not responses[0]['error']['may_have_committed'] and not continued
    finally:
        await dispatcher.close()


@pytest.mark.asyncio
async def test_inhibiting_unaffordable_nontrivial_director_does_not_add_replacement_work(monkeypatch):
    results = {}
    for mode in ('on', 'auto'):
        monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE', mode)
        c, model = make_stage_controller(), BlockingDirector()
        c.director_model = model
        c.policy = AlwaysConsultPolicy()
        c.hypothalamus.advisor_timing.record(DIRECTOR, model_profile(model), 1600., censored=True)
        original = c._wait_advisor

        async def brief_director_wait(task, *, timeout_ms):
            if getattr(task, '_advisor_work', None).module == DIRECTOR:
                timeout_ms = 1.
            return await original(task, timeout_ms=timeout_ms)

        c._wait_advisor = brief_director_wait
        answer = await c.process('Describe how tides work.', leading_brain='left', system2_mode='off',
                                 executive_observer_mode='director')
        results[mode] = (answer, model.calls, len(c.callosum.payloads), len(c.left_model.integrations))
    assert results['on'][1] == 1 and results['auto'][1] == 0
    assert results['on'][0] == results['auto'][0]
    assert results['on'][2:] == results['auto'][2:] == (1, 1)
