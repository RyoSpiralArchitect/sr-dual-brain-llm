import asyncio
from dataclasses import replace
import math

import pytest

from core.advisor_timing import AdvisorTiming, model_profile
from core.director_reasoner import DirectorAdvice
from core.thalamic_reticular_nucleus import GateFacts, GateConfig, GateMode, ThalamicReticularNucleus
from core.turn_lifecycle import cancel_task, TurnDispatcher
from test_advisor_timing import BlockingDirector, setup, DIRECTOR, PROFILE
from test_dual_brain import make_stage_controller


def ready(*, confirmed=True, samples=16):
    age = [0.]
    ledger = AdvisorTiming(clock=lambda:age[0],confirmation_enabled=confirmed,max_samples=samples)
    ledger.record(DIRECTOR,PROFILE,2000.,censored=True)
    age[0] = 16.
    return age,ledger


def first_probe(ledger,elapsed=800.):
    token = ledger.start_probe(DIRECTOR,PROFILE,1600.)
    assert token is not None
    return ledger.finish_probe(token,elapsed,usable=True)


def test_two_fresh_successes_and_observed_spread_are_required_before_normal_admission():
    age,ledger = ready()
    assert first_probe(ledger)=='confirmation_pending'
    assert ledger.estimate(DIRECTOR,PROFILE).ms==2000.
    status = ledger.recovery_status(DIRECTOR,PROFILE,1600.)
    assert status.allowed and status.kind=='confirmation' and status.confirmation_required
    assert ledger.start_probe(DIRECTOR,PROFILE,1600.) is None
    token = ledger.start_probe(DIRECTOR,PROFILE,1600.,kind='confirmation')
    assert ledger.start_probe(DIRECTOR,PROFILE,1600.,kind='confirmation') is None
    assert ledger.finish_probe(token,820.,usable=True)=='confirmed'
    estimate = ledger.estimate(DIRECTOR,PROFILE)
    assert (estimate.ms,estimate.kind,estimate.completed_samples,estimate.censored_samples)==(840.,'confirmation_margin',2,0)
    assert not ledger.recovery_status(DIRECTOR,PROFILE,1600.).confirmation_required
    age[0] = 18.
    ledger.record_work(DIRECTOR,PROFILE,1450.,usable=True)
    assert ledger.estimate(DIRECTOR,PROFILE).ms==2100.
    assert not ledger.recovery_status(DIRECTOR,PROFILE,1600.).allowed
    ledger.record_work(DIRECTOR,PROFILE,1.7e308,usable=True)
    assert math.isfinite(ledger.estimate(DIRECTOR,PROFILE).ms)


@pytest.mark.parametrize('elapsed,usable,censored,outcome',[(1400.,True,False,'unstable'),(1600.,False,True,'cancelled'),(20.,False,False,'unusable')])
def test_failed_or_variable_confirmation_is_single_use_and_cannot_prolong_old_backoff(elapsed,usable,censored,outcome):
    age,ledger = ready(samples=2)
    first_probe(ledger)
    age[0] = 20.
    token = ledger.start_probe(DIRECTOR,PROFILE,1600.,kind='confirmation')
    assert ledger.finish_probe(token,elapsed,usable=usable,censored=censored)==outcome
    status = ledger.recovery_status(DIRECTOR,PROFILE,1600.)
    assert not status.allowed and status.confirmation_required
    assert ledger.start_probe(DIRECTOR,PROFILE,1600.,kind='confirmation') is None
    # Even eviction of the old slow sample cannot bypass the probation hold.
    for _ in range(3):
        ledger.record(DIRECTOR,PROFILE,100.)
    status = ledger.recovery_status(DIRECTOR,PROFILE,1600.)
    assert status.confirmation_required and not status.allowed
    age[0] = 60.
    assert not ledger.recovery_status(DIRECTOR,PROFILE,1600.).confirmation_required


def test_failed_probe_allows_original_cold_opportunity_while_v3_keeps_newer_negative_history():
    for confirmation in (False,True):
        age,ledger = ready(confirmed=confirmation)
        token = ledger.start_probe(DIRECTOR,PROFILE,1600.)
        ledger.finish_probe(token,1600.,censored=True)
        age[0] = 61.
        assert (ledger.estimate(DIRECTOR,PROFILE).ms is None)==confirmation
        if confirmation:
            ledger.record(DIRECTOR,PROFILE,2000.,censored=True)
            age[0] = 75.
            assert ledger.recovery_status(DIRECTOR,PROFILE,1600.).reason=='rate_limited'


def test_pending_candidate_expires_and_newer_negative_or_wider_evidence_cannot_be_erased():
    age,ledger = ready()
    first_probe(ledger,900.)
    token = ledger.start_probe(DIRECTOR,PROFILE,1600.,kind='confirmation')
    ledger.record(DIRECTOR,PROFILE,100.)
    assert ledger.finish_probe(token,920.,usable=True)=='unstable'
    assert ledger.recovery_status(DIRECTOR,PROFILE,1600.).confirmation_required
    age[0] = 61.
    assert ledger.recovery_status(DIRECTOR,PROFILE,1600.).kind=='probe'
    assert ledger.start_probe(DIRECTOR,PROFILE,1600.,kind='confirmation') is None


def test_unusable_ordinary_output_after_promotion_is_bounded_without_inventing_a_latency():
    age,ledger = ready()
    first_probe(ledger)
    token = ledger.start_probe(DIRECTOR,PROFILE,1600.,kind='confirmation')
    ledger.finish_probe(token,820.,usable=True)
    age[0] = 18.
    ledger.record_work(DIRECTOR,PROFILE,1.,usable=False)
    assert ledger.estimate(DIRECTOR,PROFILE).ms==840.
    status = ledger.recovery_status(DIRECTOR,PROFILE,1600.)
    assert status.output_blocked and not status.allowed and status.confidence=='suspended'
    age[0] = 76.
    assert not ledger.recovery_status(DIRECTOR,PROFILE,1600.).output_blocked


def test_pure_policy_keeps_manual_modes_and_output_quarantine_separate_from_latency():
    gate = ThalamicReticularNucleus()
    facts = GateFacts(eligible=True,has_input=True,has_draft=True,remaining_budget_ms=1600.,estimated_latency_ms=80.,
                      recovery_confirmation_required=True,recovery_probe_allowed=True,recovery_kind='confirmation')
    decide = lambda mode,f=facts:gate.decide(DIRECTOR,facts=f,config=GateConfig(mode,'explicit',True))
    assert decide(GateMode.AUTO).reason=='bounded_recovery_confirmation'
    assert decide(GateMode.ON).reason=='forced_on' and not decide(GateMode.OFF).enabled
    assert decide(GateMode.AUTO,replace(facts,recovery_probe_allowed=False)).reason=='recovery_unconfirmed'
    assert decide(GateMode.AUTO,replace(facts,recovery_confirmation_required=False,recovery_output_blocked=True,
        recovery_probe_allowed=False)).reason=='unusable_recovery_output'
    assert not decide(GateMode.AUTO,replace(facts,eligible=False)).enabled
    assert not decide(GateMode.AUTO,replace(facts,remaining_budget_ms=50.)).enabled
    with pytest.raises(ValueError):
        AdvisorTiming(max_samples=1)


@pytest.mark.asyncio
async def test_real_task_claim_confirmation_prestart_cancel_and_profile_change_do_not_multiply_work(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE','auto')
    age,elapsed = [0.],[0.]
    monkeypatch.setattr('core.dual_brain.time.perf_counter',lambda:elapsed[0])
    c,model = make_stage_controller(),BlockingDirector()
    c.director_model = model
    c.hypothalamus.advisor_timing = ledger = AdvisorTiming(clock=lambda:age[0])
    profile = model_profile(model)
    ledger.record(DIRECTOR,profile,2000.,censored=True)
    age[0] = 16.

    async def fast(**kwargs):
        elapsed[0] += .1
        return DirectorAdvice('Confirmed guidance.',{},.9,100.,'llm')

    model.advise = fast
    probe = setup(c)
    await c._wait_advisor(probe.director_task,timeout_ms=1600.)
    confirmation = setup(c)
    assert confirmation.module_gates[DIRECTOR]['reason']=='bounded_recovery_confirmation'
    cancel_task(confirmation.director_task)
    await asyncio.gather(confirmation.director_task,return_exceptions=True)
    assert ledger.recovery_status(DIRECTOR,profile,1600.).kind=='confirmation'
    changed = setup(c)
    original_config = model.llm_config
    model.llm_config = replace(model.llm_config,model='new-confirmation-profile')
    await changed.director_task
    assert ledger.recovery_status(DIRECTOR,profile,1600.).allowed
    model.llm_config = original_config
    confirmation = setup(c)
    await c._wait_advisor(confirmation.director_task,timeout_ms=1600.)
    assert ledger.estimate(DIRECTOR,profile).ms < 101.
    terminal = [p for e,p in c.telemetry.events if e=='advisor_work' and p['state']=='completed']
    assert sum(p['recovery_probe_started'] for p in terminal)==1
    assert sum(p['recovery_confirmation_started'] for p in terminal)==1


@pytest.mark.asyncio
async def test_confirmation_cannot_bypass_tight_rpc_or_caller_off(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE','auto')
    age,ledger = ready()
    first_probe(ledger)
    c,model,responses = make_stage_controller(),BlockingDirector(),[]
    c.director_model = model
    # Use the actual model key so this is a real pending confirmation.
    ledger = AdvisorTiming(clock=lambda:age[0])
    age[0] = 0.
    profile = model_profile(model)
    ledger.record(DIRECTOR,profile,2000.,censored=True)
    age[0] = 16.
    lease = ledger.start_probe(DIRECTOR,profile,1600.)
    ledger.finish_probe(lease,800.,usable=True)
    c.hypothalamus.advisor_timing = ledger

    async def handle(req):
        result = setup(c)
        assert result.director_task is None
        assert result.module_gates[DIRECTOR]['reason']=='recovery_unconfirmed'
        return {'id':req['id'],'ok':True}

    dispatcher = TurnDispatcher(handle,responses.append,timeout_seconds=.4)
    try:
        dispatcher.submit({'id':'tight-confirmation','method':'process'})
        while not responses:
            await asyncio.sleep(.001)
        assert responses[0]['ok'] and model.calls==0
    finally:
        await dispatcher.close()
    assert setup(c,executive_observer_mode='off').director_task is None
    assert ledger.recovery_status(DIRECTOR,profile,1600.).allowed


@pytest.mark.asyncio
async def test_confirmation_timeout_drains_the_child_and_preserves_parent_cancellation():
    age = [0.]
    c,model,responses,cleaned = make_stage_controller(),BlockingDirector(),[],asyncio.Event()
    ledger = AdvisorTiming(clock=lambda:age[0])
    c.hypothalamus.advisor_timing = ledger
    profile = model_profile(model)
    ledger.record(DIRECTOR,profile,2000.,censored=True)
    age[0] = 16.
    lease = ledger.start_probe(DIRECTOR,profile,1600.)
    ledger.finish_probe(lease,100.,usable=True)

    async def handle(req):
        task = c._start_advisor(DIRECTOR,model,qid=req['id'],factory=model.advise,
            recovery_probe=True,recovery_kind='confirmation',phase_budget_ms=1600.)
        with pytest.raises(asyncio.TimeoutError):
            await c._wait_advisor(task,timeout_ms=5.)
        assert task.done() and not ledger.recovery_status(DIRECTOR,profile,1600.).allowed
        cleaned.set()
        await asyncio.Event().wait()

    dispatcher = TurnDispatcher(handle,responses.append,timeout_seconds=2.)
    try:
        dispatcher.submit({'id':'cancel-after-confirmation','method':'process'})
        await asyncio.wait_for(cleaned.wait(),1.)
        dispatcher.cancel('cancel-after-confirmation')
        while not responses:
            await asyncio.sleep(.001)
        assert responses[0]['error']['type']=='Cancelled'
        assert not responses[0]['error']['may_have_committed']
    finally:
        await dispatcher.close()
