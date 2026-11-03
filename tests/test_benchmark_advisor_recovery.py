from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sr-dual-brain-llm/scripts'))
import benchmark_advisor_recovery as pilot
from core.advisor_timing import AdvisorTiming
from test_benchmark_module_gating import fake_response


def normalized(reads):
    return sorted(json.dumps(r,sort_keys=True) for r in reads)


def test_complete_fixture_is_independent_with_explicit_task_obligations():
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    pilot.validate(fixture)
    old = pilot.bench.read_json(pilot.sessions.FIXTURE)
    assert not {c['question'] for c in fixture['cases']} & {c['question'] for c in old['cases']}
    # The prior extra-next-check rubric is not reused. New obligations are
    # explicitly requested in each question, before outputs exist.
    assert all('exactly' in c['question'].lower() and 'exactly' in c['criteria']['task'].lower() for c in fixture['cases'])
    assert 'one mechanism' in fixture['cases'][2]['question']
    assert 'one signal' in fixture['cases'][4]['question']
    fixture['cases'].pop()
    with pytest.raises(ValueError):
        pilot.validate(fixture)


@pytest.mark.parametrize('regime', ['recovery','slow'])
@pytest.mark.asyncio
async def test_bounded_recovery_and_persistently_slow_transports_use_real_owned_tasks(tmp_path,regime):
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    config = pilot.bench.LLMConfig('openai','controlled-recovery','OFFLINE_ONLY',auto_continue=False,max_continuations=0)
    journal = pilot.sessions.SessionJournal(config,{},max_calls=96)
    transport = pilot.ControlledTransport(regime)
    journal.transport._post_json = transport.post
    report = await pilot.run(SimpleNamespace(output_dir=tmp_path,judge_model='test'),fixture,config,journal,transport=transport)
    assert report['complete'] and report['live_provider_calls']==0
    rows = {(r['case'],r['arm']):r for r in report['rows']}
    assert len(rows)==10 and all(r['status']=='ok' and r['pending_advisors']==0 for r in rows.values())
    assert all(r['answer']=='Controlled answer.' and r['calls_by_role'].get('right',0)==0 for r in rows.values())
    for module in ('director_advice','executive_metrics_observation'):
        obs = lambda phase,arm='recovery_auto':rows[(phase,arm)]['metrics']['modules']['advisor_timing'][module]
        assert obs('initial')['outcome']=='timed_out'
        assert not obs('opportunity','ttl_only_auto')['admitted']
        assert obs('opportunity')['recovery_probe_started']
        assert sum(bool(obs(p).get('recovery_probe_started')) for p in pilot.PHASES)==1
        if regime=='recovery':
            assert obs('opportunity')['recovery_outcome']=='recovered'
            assert obs('followup')['outcome']=='completed' and not obs('followup')['recovery_probe']
            assert obs('relapse')['outcome']=='timed_out'
        else:
            assert obs('opportunity')['recovery_outcome']=='cancelled'
            assert obs('followup')['outcome']=='not_admitted'
        assert obs('capped')['outcome']=='not_admitted'
        assert obs('capped')['recovery_reason']=='rate_limited'
    delta = report['summaries']['by_arm']['recovery_auto']['usage']['calls']-report['summaries']['by_arm']['ttl_only_auto']['usage']['calls']
    assert delta == (6 if regime=='recovery' else 2)


@pytest.mark.asyncio
async def test_recorded_probe_reads_and_leases_replay_exact_requests_without_transport(tmp_path,monkeypatch):
    import asyncio
    age,elapsed = [0.],[0.]
    monkeypatch.setattr('core.dual_brain.time.perf_counter',lambda:elapsed[0])
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    config = pilot.bench.LLMConfig('openai','test','PRIVATE_KEY',auto_continue=False,max_continuations=0)
    journal = pilot.sessions.SessionJournal(config,{},max_calls=96)

    async def transport(url,payload,headers,**kwargs):
        system = payload['messages'][0]['content']
        initial = fixture['cases'][0]['question'] in payload['messages'][-1]['content']
        metrics = any('Observer report (internal;' in str(m.get('content','')) for m in payload['messages'])
        if initial and ('Director' in system or metrics):
            elapsed[0] += 3.
            raise asyncio.CancelledError
        return await fake_response(url,payload,headers,**kwargs)

    async def idle(seconds):
        assert seconds==16.
        age[0] += seconds

    journal.transport._post_json = transport
    args = SimpleNamespace(output_dir=tmp_path,judge_model='test')
    live = await pilot.run(args,fixture,config,journal,idle=idle,ledger_clock=lambda:age[0])
    assert live['summaries']['jointly_graded_pairs']==5
    for module in ('director_advice','executive_metrics_observation'):
        assert live['summaries']['by_arm']['recovery_auto']['advisors'][module]['probe_starts']==1
    frozen = deepcopy(journal.data)
    replay_journal = pilot.sessions.SessionJournal(config,{},frozen=frozen,max_calls=96)
    replay_journal.transport = None
    replay = await pilot.run(args,fixture,config,replay_journal)
    assert replay['complete'] and replay['live_provider_calls']==0 and replay_journal.counts==journal.counts
    for a,b in zip(live['rows'],replay['rows']):
        assert (a['case'],a['arm'],a['status'],a['answer'],a['draft'],a['memory_sources']) == (
            b['case'],b['arm'],b['status'],b['answer'],b['draft'],b['memory_sources'])
        assert a['metrics']['modules']['gates']==b['metrics']['modules']['gates']
        assert normalized(a['timing_reads'])==normalized(b['timing_reads'])
    assert live['grades']==replay['grades']
    public = json.dumps(pilot.public_summary(live))
    assert 'PRIVATE_KEY' not in public and 'User message:' not in public


def test_missing_or_invalid_replay_resources_latch_failure_even_if_budget_catches_errors():
    observer = pilot.ObservedRecovery(AdvisorTiming())
    observer.begin([])
    try:
        observer.estimate('director_advice','a'*64)
    except ValueError:
        pass
    with pytest.raises(ValueError):
        observer.finish()
    observer.begin([{'method':'claim','module':'director_advice','profile':'a'*64,'result':'PRIVATE'}])
    with pytest.raises(ValueError):
        observer.start_probe('director_advice','a'*64,1600.)
    with pytest.raises(ValueError):
        observer.finish()
