"""Prospective selection, exact scoring, ledgers and replay using scripted HTTP."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sr-dual-brain-llm/scripts'))
import benchmark_reasoning_frontier as b
from offline_provider_guard import forbid_provider_network
from test_benchmark_contract_live import config, response


def wrong(case):
    value = deepcopy(case['expected'])
    domain = case['domain']
    if domain == 'logic':
        key = next(iter(value['status']))
        value['status'][key] = 'entailed_false' if value['status'][key] == 'entailed_true' else 'entailed_true'
    elif domain == 'sql': value['rows'][0]['credit_total'] += 1
    elif domain == 'events': value['jobs'][next(iter(value['jobs']))]['epoch'] += 1
    else: value['queries'][next(iter(value['queries']))]['value'] = 'wrong'
    assert b.tasks.grade(case, json.dumps(value))['schema_valid']
    assert not b.tasks.grade(case, json.dumps(value))['exact']
    return value


class Provider:
    def __init__(self, journal, fixture, failure):
        self.journal, self.fixture, self.failure = journal, fixture, failure

    async def _post_json(self, _url, payload, _headers, **_kwargs):
        if self.failure == 'transport' and self.journal.campaign.admissions == 1:
            raise RuntimeError('PRIVATE_TRANSPORT_ERROR')
        role = self.journal.events[-1]['role']
        identity, arm = self.journal.case.rsplit(':', 1)
        case = next(c for c in self.fixture['cases'] if c['id'] == identity)
        content = json.dumps(payload['messages'])
        assert 'PRIVATE_ORACLE_CANARY' not in content
        if role in {'self_critique', 'critic'}:
            if self.failure == 'review_schema' and identity == 'calibration:L1:logic':
                return response('{}')
            data = {'verdict':'ok','issues':[],'fixes':[]}
            for line in payload['messages'][0]['content'].splitlines():
                if line.startswith('{"version": "explicit_task_goals_v1"'):
                    data['goal_findings'] = [{'goal_id':g['goal_id'],'status':'uncertain','quote':'',
                        'reason':'Scripted mechanism observation.'} for g in json.loads(line)['goals']]
            return response(json.dumps(data))
        if self.failure == 'partial' and identity == 'calibration:L1:logic' and role == 'self_initial':
            return response(json.dumps(case['expected']), 'length')
        # These errors intentionally exercise higher/lower/tied classifications.
        # They are scripted test data, never paid answer-quality evidence.
        erroneous = (case['partition'] == 'calibration' and case['level'] == 2
                     and case['domain'] in {'logic','sql'} and role == 'self_audit' and self.failure != 'ceiling')
        erroneous |= (case['partition'] == 'holdout' and (
            arm == 'self_review' and case['domain'] == 'sql' and role == 'self_audit'
            or arm == 'dual_auto' and case['domain'] == 'events'
            or arm == 'dual_on' and case['domain'] == 'logic'))
        return response(json.dumps(wrong(case) if erroneous else case['expected']))


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['none', 'transport', 'partial', 'review_schema'])
async def test_calibration_selects_only_semantic_failures_then_replays_untouched_comparison(tmp_path, failure):
    fixture = b.pilot.bench.read_json(b.FIXTURE)
    fixture['oracle_canary'] = 'PRIVATE_ORACLE_CANARY'
    cfg = replace(config(), model='gpt-6-luna', max_output_tokens=4096,
        timeout_seconds=60,auto_continue=False,max_continuations=0)
    live, repeat = tmp_path/'live', tmp_path/'replay'
    live.mkdir(); repeat.mkdir()
    with forbid_provider_network() as audit:
        result, journals = await b.run(live, fixture, cfg,
            transport_factory=lambda journal: Provider(journal,fixture,failure))
        frozen = json.loads(json.dumps(journals))
        replay, replay_journals = await b.run(repeat, fixture, cfg, frozen=frozen)
    assert b.canonical(result) == b.canonical(replay)
    assert replay['live_provider_calls'] == audit.blocked_requests == 0
    assert result['live_provider_calls'] <= 192
    assert all(len(j['occurrences']) <= 96 for j in journals.values())
    assert set(replay_journals) == set(journals)
    if failure == 'transport':
        assert result['outcome'] == 'transport_blocked' and not result['evaluation']
        assert result['live_provider_calls'] == 1 and result['usage']['unknown_usage_calls'] == 1
        assert result['usage']['input_tokens'] is None and result['selected_level'] is None
    else:
        assert result['selected_level'] == 2
        assert result['outcome'] == 'heldout_comparison_complete'
        assert len(result['calibration']) == 8 and len(result['evaluation']) == 24
        assert {r['level'] for r in result['evaluation']} == {2,3}
        assert all(r['case'].startswith('holdout:') for r in result['evaluation'])
        assert all(r['output_reserved'] <= b.OUTPUT_ALLOWANCE for r in result['calibration']+result['evaluation'])
        assert {p['outcome'] for p in result['summary']['pairs']} == {'dual_higher','dual_lower','same'}
        assert all(r['policy_class']=='RightBrainPolicy' and r['plan_state_cleared']
            for r in result['evaluation'] if r['arm']!='self_review')
        assert all(r['initial_answer_complete'] and 'initial_grade' in r
            for r in result['evaluation'] if r['arm']!='self_review')
        for journal in journals.values():
            for o in journal['occurrences']:
                assert 'reasoning_effort' not in o['request']['payload']
        if failure != 'none':
            row = result['calibration'][0]
            assert not b.eligible_for_calibration(row)
    public = json.dumps(b.public_summary(result))
    assert all(s not in public for s in ('"question":','"answer":','"initial_answer":',
        'PRIVATE_ORACLE_CANARY','PRIVATE_TRANSPORT_ERROR','FAKE_NO_CREDENTIAL'))


@pytest.mark.asyncio
async def test_output_reservations_are_not_refunded_and_global_transport_latch_crosses_phases(tmp_path):
    fixture = b.pilot.bench.read_json(b.FIXTURE)
    cfg = replace(config(), model='gpt-6-luna',max_output_tokens=4096)
    budget = b.CampaignBudget()
    j = b.PhaseJournal(cfg,b.metadata(fixture,cfg,'calibration'),campaign=budget,max_calls=96)
    class Transport:
        async def _post_json(self,*args,**kwargs): return response('x')
    j.transport = Transport()
    j.begin_trial('budget')
    with forbid_provider_network():
        for _ in range(3):
            await j.post('test','https://fake.invalid',{'max_completion_tokens':4096},{})
        with pytest.raises(b.pilot.AdmissionError):
            await j.post('test','https://fake.invalid',{'max_completion_tokens':1},{})
        assert j.reserved == 12288 and j.calls == budget.admissions == 3
        assert j.trial_denied and j.data['denied_admissions'][-1]['reason']=='trial_output_cap'
        j.begin_trial('next')
        class Failing:
            async def _post_json(self,*args,**kwargs): raise RuntimeError('PRIVATE_TRANSPORT_ERROR')
        j.transport = Failing()
        with pytest.raises(RuntimeError):
            await j.post('test','https://fake.invalid',{'max_completion_tokens':2048},{})
        assert j.reserved == 2048 and budget.blocked
        later = b.PhaseJournal(cfg,b.metadata(fixture,cfg,'evaluation'),campaign=budget,max_calls=96)
        later.transport = Transport(); later.begin_trial('later')
        with pytest.raises(b.pilot.AdmissionError):
            await later.post('test','https://fake.invalid',{'max_completion_tokens':2048},{})
        assert later.calls == later.reserved == 0


def test_unavailability_and_schema_failure_cannot_select_a_semantic_frontier():
    row = {'answer_complete':True,'workflow_valid':True,'grade':{'schema_valid':True,'exact':False}}
    assert b.level_break([deepcopy(row) for _ in range(4)])
    for field in ('answer_complete','workflow_valid'):
        rows = [deepcopy(row) for _ in range(4)]
        for r in rows[:2]: r[field] = False
        assert not b.level_break(rows)
    rows = [deepcopy(row) for _ in range(4)]
    for r in rows[:2]: r['grade']['schema_valid'] = False
    assert not b.level_break(rows)


@pytest.mark.asyncio
async def test_all_correct_calibration_keeps_holdouts_unspent_and_claims_no_winner(tmp_path):
    fixture = b.pilot.bench.read_json(b.FIXTURE)
    cfg = replace(config(),model='gpt-6-luna',max_output_tokens=4096,
        timeout_seconds=60,auto_continue=False,max_continuations=0)
    live, repeat = tmp_path/'live',tmp_path/'replay'
    live.mkdir();repeat.mkdir()
    with forbid_provider_network() as audit:
        result, journals = await b.run(live,fixture,cfg,
            transport_factory=lambda j:Provider(j,fixture,'ceiling'))
        replay,_ = await b.run(repeat,fixture,cfg,frozen=json.loads(json.dumps(journals)))
    assert b.canonical(result)==b.canonical(replay)
    assert result['selected_level'] is None and not result['evaluation']
    assert result['outcome']=='no_eligible_frontier_within_frozen_levels'
    assert result['live_provider_calls']==96 and set(journals)=={'calibration'}
    assert all(r['grade']['exact'] and r['workflow_valid'] for r in result['calibration'])
    assert replay['live_provider_calls']==audit.blocked_requests==0
