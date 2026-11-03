import asyncio
from dataclasses import replace

import pytest

from core.advisor_timing import AdvisorTiming, model_profile
from core.director_reasoner import DirectorAdvice
from core.executive_reasoner import ExecutiveAdvice
from core.thalamic_reticular_nucleus import GateFacts, GateConfig, GateMode, ThalamicReticularNucleus
from core.turn_lifecycle import TurnDispatcher, cancel_task
from test_advisor_timing import BlockingDirector, setup, DIRECTOR, EXECUTIVE, METRICS, PROFILE
from test_dual_brain import make_stage_controller


@pytest.mark.parametrize('module,floor', [(DIRECTOR,1600.),(METRICS,2500.),(EXECUTIVE,2000.)])
def test_probe_requires_known_ample_budget_and_quiet_history_without_spending_on_reads(module,floor):
    now = [0.]
    ledger = AdvisorTiming(clock=lambda:now[0])
    ledger.record(module, PROFILE, floor+100., censored=True)
    assert ledger.recovery_status(module, PROFILE, floor).reason == 'cooldown'
    now[0] = 15.
    for budget in (None,True,float('nan'),float('inf'),-1.):
        assert not ledger.recovery_status(module, PROFILE, budget).allowed
    assert ledger.recovery_status(module, PROFILE, floor-1.).reason == 'budget_too_small'
    for _ in range(20):
        assert ledger.recovery_status(module, PROFILE, floor).allowed
    token = ledger.start_probe(module, PROFILE, floor)
    assert token is not None and ledger.start_probe(module, PROFILE, floor) is None
    assert ledger.recovery_status(module, PROFILE, floor).reason == 'in_flight'


def test_fast_probe_replaces_old_bounds_then_relapse_obeys_the_rolling_start_limit():
    now = [0.]
    ledger = AdvisorTiming(clock=lambda:now[0], confirmation_enabled=False)
    ledger.record(DIRECTOR, PROFILE, 2000., censored=True)
    now[0] = 16.
    token = ledger.start_probe(DIRECTOR, PROFILE, 1600.)
    assert ledger.finish_probe(token, 100., usable=True) == 'recovered'
    estimate = ledger.estimate(DIRECTOR, PROFILE)
    assert (estimate.ms,estimate.kind,estimate.completed_samples,estimate.censored_samples) == (100.,'completed_max',1,0)
    assert ledger.finish_probe(token, 1., usable=True) == 'profile_unavailable'
    now[0] = 18.
    ledger.record(DIRECTOR, PROFILE, 2000., censored=True)
    now[0] = 33.
    assert ledger.recovery_status(DIRECTOR, PROFILE, 1600.).reason == 'rate_limited'
    now[0] = 75.999
    assert ledger.start_probe(DIRECTOR, PROFILE, 1600.) is None
    now[0] = 76.
    assert ledger.start_probe(DIRECTOR, PROFILE, 1600.) is not None


@pytest.mark.parametrize('usable,censored,outcome', [(False,False,'unusable'),(False,True,'cancelled'),(True,False,'still_slow')])
def test_failed_empty_cancelled_or_near_limit_probe_does_not_erase_slow_history(usable,censored,outcome):
    now = [0.]
    ledger = AdvisorTiming(clock=lambda:now[0])
    ledger.record(DIRECTOR, PROFILE, 2000., censored=True)
    now[0] = 16.
    token = ledger.start_probe(DIRECTOR, PROFILE, 1600.)
    assert ledger.finish_probe(token, 1570., usable=usable, censored=censored) == outcome
    assert ledger.estimate(DIRECTOR, PROFILE).ms == 2000.
    now[0] = 32.
    assert ledger.recovery_status(DIRECTOR, PROFILE, 1600.).reason == 'rate_limited'


def test_expiry_cannot_reset_a_recent_probe_allowance_and_profiles_remain_bounded():
    now = [0.]
    ledger = AdvisorTiming(clock=lambda:now[0], max_profiles=2)
    ledger.record(DIRECTOR, PROFILE, 2000., censored=True)
    now[0] = 55.
    token = ledger.start_probe(DIRECTOR, PROFILE, 1600.)
    ledger.finish_probe(token, 1.)  # Unusable output cannot renew old samples.
    now[0] = 61.
    assert ledger.estimate(DIRECTOR, PROFILE).ms is None
    ledger.record(DIRECTOR, PROFILE, 2000., censored=True)  # New ordinary cold probe.
    now[0] = 76.
    assert ledger.recovery_status(DIRECTOR, PROFILE, 1600.).reason == 'rate_limited'
    assert not ledger.recovery_status(METRICS, PROFILE, 2500.).allowed
    ledger.record(METRICS, PROFILE, 3000., censored=True)
    ledger.record(DIRECTOR, 'b'*64, 2000., censored=True)
    assert len(ledger._history) == 2
    assert ledger.finish_probe(token, 1., usable=True) == 'profile_unavailable'


def test_fast_probe_preserves_observations_that_arrive_after_its_claim():
    now = [0.]
    ledger = AdvisorTiming(clock=lambda:now[0], confirmation_enabled=False)
    ledger.record(DIRECTOR, PROFILE, 3000., censored=True)
    now[0] = 16.
    token = ledger.start_probe(DIRECTOR, PROFILE, 1600.)
    ledger.record(DIRECTOR, PROFILE, 2000., censored=True)
    ledger.finish_probe(token, 100., usable=True)
    estimate = ledger.estimate(DIRECTOR, PROFILE)
    assert estimate.ms == 2000. and estimate.censored_samples == 1


def test_v3_remains_pure_with_mode_precedence_and_v2_equivalent_resource_baseline():
    gate = ThalamicReticularNucleus()
    facts = GateFacts(eligible=True,has_input=True,has_draft=True,
                      remaining_budget_ms=1600.,estimated_latency_ms=2000.,recovery_probe_allowed=True)
    decide = lambda mode,f=facts:gate.decide(DIRECTOR,facts=f,config=GateConfig(mode,'explicit',True))
    assert decide(GateMode.AUTO).reason == 'bounded_recovery_probe'
    assert decide(GateMode.ON).reason == 'forced_on' and not decide(GateMode.OFF).enabled
    assert not decide(GateMode.AUTO,replace(facts,is_trivial=True)).enabled
    assert not decide(GateMode.ON,replace(facts,eligible=False)).enabled
    assert not decide(GateMode.AUTO,replace(facts,remaining_budget_ms=50.)).enabled
    assert not decide(GateMode.AUTO,replace(facts,recovery_probe_allowed=1)).enabled
    old = AdvisorTiming(recovery_enabled=False)
    old.record(DIRECTOR,PROFILE,2000.,censored=True)
    assert old.recovery_status(DIRECTOR,PROFILE,1600.).reason == 'disabled'


@pytest.mark.asyncio
async def test_cancel_before_start_off_and_profile_change_spend_no_probe(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE','auto')
    age = [0.]
    c,model = make_stage_controller(),BlockingDirector()
    c.director_model = model
    c.hypothalamus.advisor_timing = ledger = AdvisorTiming(clock=lambda:age[0])
    profile = model_profile(model)
    ledger.record(DIRECTOR,profile,2000.,censored=True)
    age[0] = 16.
    first = setup(c)
    assert first.module_gates[DIRECTOR]['reason'] == 'bounded_recovery_probe'
    cancel_task(first.director_task)
    await asyncio.gather(first.director_task,return_exceptions=True)
    assert model.calls == 0 and ledger.recovery_status(DIRECTOR,profile,1600.).allowed
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE','off')
    assert setup(c).director_task is None
    assert ledger.recovery_status(DIRECTOR,profile,1600.).allowed
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE','auto')
    second = setup(c)
    model.llm_config = replace(model.llm_config,model='changed-profile')
    await second.director_task
    assert model.calls == 0 and ledger.recovery_status(DIRECTOR,profile,1600.).allowed


@pytest.mark.asyncio
async def test_usable_probe_restores_admission_without_waiting_for_ttl(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE','auto')
    age,elapsed = [0.],[0.]
    monkeypatch.setattr('core.dual_brain.time.perf_counter',lambda:elapsed[0])
    c,model = make_stage_controller(),BlockingDirector()
    c.director_model = model
    c.hypothalamus.advisor_timing = ledger = AdvisorTiming(clock=lambda:age[0], confirmation_enabled=False)
    profile = model_profile(model)
    ledger.record(DIRECTOR,profile,2000.,censored=True)
    age[0] = 16.

    async def fast(**kwargs):
        elapsed[0] += .08
        return DirectorAdvice('Usable guidance.',{},.9,80.,'llm')

    model.advise = fast
    result = setup(c)
    await c._wait_advisor(result.director_task,timeout_ms=1600.)
    assert ledger.estimate(DIRECTOR,profile).ms == 80.
    assert setup(c).module_gates[DIRECTOR]['reason'] == 'optional_observation_needed'
    # Drain the ordinary task as well; it is not a second recovery probe.
    await asyncio.sleep(0)
    terminal = [p for e,p in c.telemetry.events if e == 'advisor_work' and p['state']=='completed']
    assert sum(p['recovery_probe_started'] for p in terminal) == 1
    assert terminal[0]['recovery_outcome'] == 'recovered'


@pytest.mark.asyncio
async def test_probe_timeout_excludes_cleanup_and_spends_one_allowance(monkeypatch):
    age,elapsed = [0.],[0.]
    monkeypatch.setattr('core.dual_brain.time.perf_counter',lambda:elapsed[0])
    c,model = make_stage_controller(),BlockingDirector()
    c.hypothalamus.advisor_timing = ledger = AdvisorTiming(clock=lambda:age[0])
    profile = model_profile(model)
    ledger.record(DIRECTOR,profile,2000.,censored=True)
    age[0] = 16.

    async def blocked():
        model.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            elapsed[0] = 10.

    task = c._start_advisor(DIRECTOR,model,qid='probe-cleanup',factory=blocked,
                            recovery_probe=True,phase_budget_ms=1600.)
    await model.started.wait()
    elapsed[0] = 3.
    with pytest.raises(asyncio.TimeoutError):
        await c._wait_advisor(task,timeout_ms=1.)
    assert ledger.estimate(DIRECTOR,profile).ms == 3000.
    assert ledger.recovery_status(DIRECTOR,profile,1600.).reason == 'rate_limited'
    terminal = [p for e,p in c.telemetry.events if e=='advisor_work' and p['state']=='timed_out'][0]
    assert terminal['cleanup_ms']==7000. and terminal['recovery_outcome']=='cancelled'


@pytest.mark.asyncio
async def test_tight_rpc_budget_vetoes_due_executive_probe(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_EXECUTIVE_ADVICE_MODE','auto')
    age = [0.]
    c,model,responses = make_stage_controller(),BlockingDirector(),[]
    c.executive_model = model
    c.hypothalamus.advisor_timing = ledger = AdvisorTiming(clock=lambda:age[0])
    ledger.record(EXECUTIVE,model_profile(model),5000.)
    age[0] = 16.

    async def handle(req):
        result = setup(c,executive_mode='observe',executive_observer_mode='off')
        assert result.executive_task is None
        assert result.module_gates[EXECUTIVE]['reason']=='insufficient_time_budget'
        return {'id':req['id'],'ok':True}

    dispatcher = TurnDispatcher(handle,responses.append,timeout_seconds=.4)
    try:
        dispatcher.submit({'id':'tight-probe','method':'process'})
        while not responses:
            await asyncio.sleep(.001)
        assert responses[0]['ok'] and model.calls == 0
    finally:
        await dispatcher.close()


@pytest.mark.asyncio
async def test_known_ample_rpc_allows_one_executive_recovery_probe(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_EXECUTIVE_ADVICE_MODE','auto')
    age = [0.]
    c,model,responses = make_stage_controller(),BlockingDirector(),[]
    c.executive_model = model
    c.hypothalamus.advisor_timing = ledger = AdvisorTiming(clock=lambda:age[0], confirmation_enabled=False)
    profile = model_profile(model)
    ledger.record(EXECUTIVE,profile,5000.)
    age[0] = 16.

    async def fast(**kwargs):
        return ExecutiveAdvice('Recovered executive guidance.',{},.9,0.,'llm')

    model.advise = fast

    async def handle(req):
        result = setup(c,executive_mode='observe',executive_observer_mode='off')
        assert result.module_gates[EXECUTIVE]['reason']=='bounded_recovery_probe'
        await c._wait_advisor(result.executive_task,timeout_ms=1200.)
        return {'id':req['id'],'ok':True}

    dispatcher = TurnDispatcher(handle,responses.append,timeout_seconds=3.)
    try:
        dispatcher.submit({'id':'ample-probe','method':'process'})
        while not responses:
            await asyncio.sleep(.001)
        assert responses[0]['ok']
        estimate = ledger.estimate(EXECUTIVE,profile)
        assert estimate.ms < 100. and estimate.completed_samples==1
    finally:
        await dispatcher.close()


@pytest.mark.asyncio
async def test_manual_on_work_does_not_claim_or_clear_recovery_history(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE','on')
    age = [0.]
    c,model = make_stage_controller(),BlockingDirector()
    c.director_model = model
    c.hypothalamus.advisor_timing = ledger = AdvisorTiming(clock=lambda:age[0])
    profile = model_profile(model)
    ledger.record(DIRECTOR,profile,2000.,censored=True)
    age[0] = 16.

    async def fast(**kwargs):
        return DirectorAdvice('Manual guidance.',{},.9,0.,'llm')

    model.advise = fast
    result = setup(c)
    assert result.module_gates[DIRECTOR]['reason']=='forced_on'
    await c._wait_advisor(result.director_task,timeout_ms=1600.)
    assert ledger.estimate(DIRECTOR,profile).ms == 2000.
    age[0] = 32.
    assert ledger.recovery_status(DIRECTOR,profile,1600.).allowed
