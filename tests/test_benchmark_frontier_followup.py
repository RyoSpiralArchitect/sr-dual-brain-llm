"""Unspent confirmation, immutable development data and shared paid ceiling."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from test_benchmark_reasoning_frontier import Provider, config, wrong
import benchmark_frontier_followup as f

b = f.base


def calibration(cfg):
    fixture = b.pilot.bench.read_json(b.FIXTURE)
    rows = []
    for case in fixture['cases']:
        if case['partition'] != 'calibration':
            continue
        is_logic = case['domain'] == 'logic'
        bad = is_logic or (case['domain'] == 'sql' and case['level'] == 5)
        text = json.dumps(wrong(case) if bad else case['expected'])
        rows.append({'case': case['id'], 'domain': case['domain'], 'level': case['level'],
            'arm':'self_review','answer':text,'answer_complete':True,'workflow_valid':not is_logic,
            'grade':b.tasks.grade(case,text),
            'usage':{'calls': (2 if case['level'] <= 2 else 1) if is_logic else 4}})
    meta = b.metadata(fixture,cfg,'campaign') | {'code_sha256': f.preserved_code_digest()}
    source = {'schema_version':1,'complete':True,'source_revision':'FROZEN_ORIGINAL_SOURCE',
        'selected_level':None,'outcome':'no_eligible_frontier_within_frozen_levels',
        'metadata':meta,'calibration':rows,'evaluation':[],
        'summary':b.summarize([]),'phase_receipts':{'calibration':{'requests':80}},
        'usage':{'calls':80,'input_tokens':800,'output_tokens':1600,'known_input_tokens':800,
            'known_output_tokens':1600,'unknown_usage_calls':0,'cancelled_calls':0,'failed_calls':0}}
    source['canonical_sha256'] = b.pilot.bench.digest(b.canonical(source))
    manifest = {'schema_version':1,'experiment':f.ORIGIN,'selection_rule':f.RULE,
        'selected_level':5,'global_call_limit':192,'confirmation_call_limit':96,'prior_admissions':80,
        'source_revision':source['source_revision'],'source_report_sha256':'FILLED_WHEN_SERIALIZED',
        'source_canonical_sha256':source['canonical_sha256'],
        'fixture_sha256':meta['fixture_sha256'],'source_code_sha256':meta['code_sha256'],
        'source_protocol_sha256':meta['protocol_sha256']}
    return fixture,source,manifest


def test_selection_does_not_use_ineligible_errors_or_dual_results():
    cfg = replace(config(),model='gpt-6-luna',max_output_tokens=4096)
    fixture,source,manifest = calibration(cfg)
    assert f.select_level(source['calibration']) == 5
    assert f.verify_source(source,manifest,fixture,cfg) == 5
    assert not any(b.level_break([r for r in source['calibration'] if r['level']==level])
        for level in range(1,7))
    rows = deepcopy(source['calibration'])
    row = next(r for r in rows if r['domain']=='sql' and r['level']==5)
    row['workflow_valid'] = False
    assert f.select_level(rows) is None


@pytest.mark.parametrize('change', ['source','answer','fixture','budget','holdout','profile'])
def test_reject_changed_provenance_or_spent_confirmation(tmp_path, change):
    cfg = replace(config(),model='gpt-6-luna',max_output_tokens=4096)
    fixture,source,manifest = calibration(cfg)
    if change == 'source': manifest['source_code_sha256'] = 'CHANGED'
    elif change == 'answer': source['calibration'][0]['answer'] = '{}'
    elif change == 'fixture': fixture['clock'] += 1
    elif change == 'budget': manifest['prior_admissions'] = 0
    elif change == 'holdout': source['evaluation'].append({'arm':'dual_auto'})
    else: cfg = replace(cfg,max_output_tokens=2048)
    with pytest.raises(ValueError): f.verify_source(source,manifest,fixture,cfg)


def test_original_byte_hash_and_unknown_usage_stay_authoritative(tmp_path):
    cfg = replace(config(),model='gpt-6-luna',max_output_tokens=4096)
    fixture,source,manifest = calibration(cfg)
    path = tmp_path/'original.json'
    path.write_text(json.dumps(source))
    manifest['source_report_sha256'] = f.file_hash(path)
    assert f.load_source(path,manifest,fixture,cfg) == source
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError): f.load_source(path,manifest,fixture,cfg)
    other = source['usage'] | {'input_tokens':None,'output_tokens':None,
        'unknown_usage_calls':1,'cancelled_calls':1}
    combined = f.combined_usage(source['usage'],other)
    assert combined['calls']==160 and combined['input_tokens'] is combined['output_tokens'] is None
    assert combined['known_output_tokens']==3200 and combined['unknown_usage_calls']==1


@pytest.mark.asyncio
async def test_confirmation_only_replays_exactly_and_inherits_original_admissions(tmp_path):
    cfg = replace(config(),model='gpt-6-luna',max_output_tokens=4096,
        timeout_seconds=60,auto_continue=False,max_continuations=0)
    fixture,source,manifest = calibration(cfg)
    class ConfirmationProvider(Provider):
        async def _post_json(self,*args,**kwargs):
            assert self.journal.case.startswith(('holdout:L5:','holdout:L6:'))
            return await super()._post_json(*args,**kwargs)
    live,repeat = tmp_path/'live',tmp_path/'replay'
    live.mkdir();repeat.mkdir()
    with b.pilot.forbid_provider_network() as audit:
        result,journal = await f.run(live,fixture,cfg,source,manifest,
            transport_factory=lambda j:ConfirmationProvider(j,fixture,'none'))
        replay,_ = await f.run(repeat,fixture,cfg,source,manifest,frozen=json.loads(json.dumps(journal)))
    assert f.canonical(result)==f.canonical(replay)
    assert result['outcome']=='heldout_followup_complete' and result['selected_level']==5
    assert not result['calibration'] and len(result['evaluation'])==24
    assert {r['level'] for r in result['evaluation']}=={5,6}
    assert {p['outcome'] for p in result['summary']['pairs']}=={'dual_higher','dual_lower','same'}
    assert result['campaign_admissions']==80+result['usage']['calls']<=176
    assert result['campaign_usage']['calls']==result['campaign_admissions']
    assert result['usage']['calls']==result['live_provider_calls']<=96
    assert all(r['output_reserved']<=12288 for r in result['evaluation'])
    assert replay['live_provider_calls']==audit.blocked_requests==0
    public = json.dumps(b.public_summary(result))
    assert all(s not in public for s in ('"question":','"answer":','"initial_answer":','FAKE_NO_CREDENTIAL'))


@pytest.mark.asyncio
async def test_global_limit_counts_original_calls_before_new_transport(tmp_path):
    cfg = replace(config(),model='gpt-6-luna',max_output_tokens=4096)
    fixture,source,manifest = calibration(cfg)
    budget = b.CampaignBudget(); budget.admissions = 192
    journal = b.PhaseJournal(cfg,f.metadata(fixture,cfg,manifest),campaign=budget,max_calls=96)
    journal.begin_trial('holdout:L5:sql:dual_on')
    with b.pilot.forbid_provider_network() as audit:
        with pytest.raises(b.pilot.AdmissionError):
            await journal.post('draft','https://fake.invalid',{'max_completion_tokens':4096},{})
    assert journal.calls==audit.blocked_requests==0 and budget.admissions==192
    assert journal.data['denied_admissions'][-1]['reason']=='global_call_cap'
