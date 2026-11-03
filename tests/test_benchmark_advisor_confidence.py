from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sr-dual-brain-llm/scripts'))
import benchmark_advisor_confidence as pilot
from test_benchmark_module_gating import fake_response


def normalize(rows):
    return sorted(json.dumps(r,sort_keys=True) for r in rows)


def test_independent_fixture_asks_every_task_obligation_and_private_advice_schema_requires_quotes():
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    pilot.validate(fixture)
    previous = pilot.bench.read_json(pilot.recovery.FIXTURE)
    assert not {c['question'] for c in fixture['cases']} & {c['question'] for c in previous['cases']}
    assert all('exactly' in c['question'].lower() and 'exactly' in c['criteria']['task'].lower() for c in fixture['cases'])
    labels = {'A':{'text':'A concrete task-specific memo.'}}
    grades = {'scores':{'A':{k:{'verdict':'pass','quote':'concrete'} for k in pilot.ADVICE_CRITERIA}}}
    pilot.validate_advice(json.dumps(grades),labels)
    grades['scores']['A']['potential_use']['quote']='not in the output'
    with pytest.raises(ValueError):
        pilot.validate_advice(json.dumps(grades),labels)
    with pytest.raises(ValueError):
        pilot.validate_advice('{"scores":{},"scores":{}}',labels)


@pytest.mark.parametrize('regime',['steady','jitter','lucky','missed','late'])
@pytest.mark.asyncio
async def test_logical_timing_sequences_count_useful_output_failure_and_missed_opportunities(tmp_path,regime):
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    config = pilot.bench.LLMConfig('openai','controlled-confidence','OFFLINE_ONLY',auto_continue=False,max_continuations=0)
    journal = pilot.recovery.sessions.SessionJournal(config,{},max_calls=96)
    transport = pilot.ControlledTransport(regime)
    journal.transport._post_json = transport.post
    report = await pilot.run(SimpleNamespace(output_dir=tmp_path,judge_model='test'),fixture,config,journal,transport=transport)
    assert report['complete'] and report['live_provider_calls']==0
    assert all(r['status']=='ok' and r['pending_advisors']==0 and r['answer']=='Controlled answer.' for r in report['rows'])
    stats = report['summaries']['by_arm']
    module = 'executive_metrics_observation'
    old,new = [stats[arm]['advisors'][module] for arm in pilot.ARMS]
    if regime=='steady':
        assert old['usable_output']==new['usable_output']==4
        assert old['started']==new['started']==5
        assert new['confirmation_starts']==1 and new['probe_outcomes']['confirmed']==1
    elif regime=='jitter':
        assert new['usable_output']==old['usable_output']+1
        assert new['outcomes']['cancelled']==old['outcomes']['cancelled']-1
        assert new['started']==old['started']
    elif regime in {'lucky','late'}:
        assert new['usable_output']==old['usable_output']+1
        assert new['started']==old['started']+1
    else:
        assert new['usable_output']==old['usable_output']-1
        assert new['started']==old['started']-1
    rows = {(r['case'],r['arm']):r for r in report['rows']}
    if regime=='late':
        assert rows[('followup','confirmed_v4')]['metrics']['modules']['advisor_timing'][module]['usable_output']
        assert not rows[('followup','single_success_v3')]['metrics']['modules']['advisor_timing'][module]['admitted']
    assert new['confirmation_starts'] <= 1
    assert all(row['calls_by_role'].get('right',0)==0 for row in report['rows'])


@pytest.mark.asyncio
async def test_advice_content_grading_and_confirmation_leases_replay_without_transport(tmp_path,monkeypatch):
    import asyncio
    age,elapsed = [0.],[0.]
    monkeypatch.setattr('core.dual_brain.time.perf_counter',lambda:elapsed[0])
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    config = pilot.bench.LLMConfig('openai','test','PRIVATE_KEY',auto_continue=False,max_continuations=0)
    journal = pilot.recovery.sessions.SessionJournal(config,{},max_calls=96)

    async def transport(url,payload,headers,**kwargs):
        try:
            packet = json.loads(payload['messages'][-1]['content'])
        except ValueError:
            packet = {}
        if isinstance(packet,dict) and 'outputs' in packet:
            content = {'scores':{label:{k:{'verdict':'pass','quote':output['text'][:1]} for k in pilot.ADVICE_CRITERIA}
                                for label,output in packet['outputs'].items()}}
            return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(content)}}],
                    'model':'test','usage':{'prompt_tokens':10,'completion_tokens':5}}
        system = payload['messages'][0]['content']
        initial = payload['messages'][-1]['content'].startswith('User message:\n'+fixture['cases'][0]['question']+'\n')
        if initial and ('Director' in system or 'Observer report (internal;' in payload['messages'][-1]['content']):
            elapsed[0] += 3.
            raise asyncio.CancelledError
        return await fake_response(url,payload,headers,**kwargs)

    async def idle(seconds):
        age[0] += seconds

    journal.transport._post_json = transport
    args = SimpleNamespace(output_dir=tmp_path,judge_model='test')
    live = await pilot.run(args,fixture,config,journal,idle=idle,ledger_clock=lambda:age[0])
    assert live['summaries']['jointly_graded_pairs']==5
    for module in ('director_advice','executive_metrics_observation'):
        assert live['summaries']['by_arm']['confirmed_v4']['advisors'][module]['confirmation_starts']==1
    frozen = deepcopy(journal.data)
    replay_journal = pilot.recovery.sessions.SessionJournal(config,{},frozen=frozen,max_calls=96)
    replay_journal.transport = None
    replay = await pilot.run(args,fixture,config,replay_journal)
    assert replay['complete'] and replay['live_provider_calls']==0 and replay_journal.counts==journal.counts
    for a,b in zip(live['rows'],replay['rows']):
        assert (a['case'],a['arm'],a['status'],a['answer'],a['draft'],a['advisor_outputs']) == (
            b['case'],b['arm'],b['status'],b['answer'],b['draft'],b['advisor_outputs'])
        assert a['metrics']['modules']['gates']==b['metrics']['modules']['gates']
        assert normalize(a['timing_reads'])==normalize(b['timing_reads'])
    assert live['grades']==replay['grades'] and live['advice_grades']==replay['advice_grades']
    public = json.dumps(pilot.public_summary(live))
    assert 'PRIVATE_KEY' not in public and 'User message:' not in public and 'Advice.' not in public
