"""Normal policy, actual PostgreSQL writes/corrections/restart, and semantic replay."""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sr-dual-brain-llm/scripts'))
import benchmark_natural_memory as b
from test_benchmark_contract_live import config, response
from offline_provider_guard import forbid_provider_network


class Provider:
    def __init__(self, calibration, fail_write=False, bad_judge=False):
        self.expected={v['answer']:v['expected'] for c in calibration['cases'] for v in c['variants'].values()}
        self.fail_write=fail_write;self.bad_judge=bad_judge;self.judges=0

    async def _post_json(self,_url,payload,_headers,**kwargs):
        messages=payload['messages'];system=messages[0]['content'];user=messages[-1]['content']
        try:packet=json.loads(user)
        except ValueError:packet={}
        if 'answers' in packet:
            self.judges+=1
            assert all(t not in user for t in ('PRIVATE_','"expected"','comparison_reference','variants'))
            if self.bad_judge and self.judges==1:return response('{"scores":{}}')
            scores={label:{c:{'verdict':self.expected.get(answer,{}).get(c,'pass'),
                              'quote':answer[:1]} for c in b.assessment.SEMANTIC_CRITERIA}
                    for label,answer in packet['answers'].items()}
            return response(json.dumps({'scores':scores}))
        if 'records' in packet:
            query=packet.get('question','')
            decisions=[]
            for record in packet['records']:
                supported=('保持率' not in query and ('retention' in record['question'].lower()
                    or 'verbatim' in query.lower() or '再掲' in query))
                decisions.append({'id':record['id'],'supported':supported,'quote':record['answer'] if supported else ''})
            return response(json.dumps({'decisions':decisions}))
        if 'rigorous reviewer (critic)' in system:
            return response('{"verdict":"ok","issues":[],"fixes":[],"confidence_r":0.9}')
        statement=re.search(r"Reply with '([^']+)' only",user)
        if statement:
            if self.fail_write and '43 days' in statement[1]:return response('','stop')
            return response(statement[1])
        for line in system.splitlines():
            if line.startswith('{"source_question":'):
                record=json.loads(line)
                if 'recorded_answer' in record:return response(record['recorded_answer'])
        if '"action": "reselect_record"' in system:
            return response('現在の候補から、再掲する記録をもう一度選んでください。' if '再掲' in user
                            else 'Please choose the record again from the current record choices.')
        if '"action": "clarify_record"' in system:
            return response('Please choose which complete record to quote from the current record choices.')
        if '"action": "clarify_parameter"' in system:
            return response('Do you mean the data-retention period or the retention rate?')
        if '"action": "request_current_source"' in system:
            return response('Please provide a current source for that explicitly requested parameter.')
        if '47 days' in system:return response('According to the current record, the period is 47 days.')
        return response('The available context does not establish that value.')


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['none','write','judge'])
async def test_natural_persistent_episode_and_json_roundtrip_replay(tmp_path,failure):
    dsn=os.environ.get('DUALBRAIN_TEST_PG_DSN')
    if not dsn:pytest.skip('isolated PostgreSQL required')
    cal=b.pilot.bench.read_json(b.CALIBRATION);fixture=b.pilot.bench.read_json(b.FIXTURE)
    cfg=config();judge=config()
    live,replay=tmp_path/'live',tmp_path/'replay';live.mkdir();replay.mkdir()
    j=b.pilot.RawJournal(cfg,b.metadata(cal,fixture,cfg,judge),max_calls=b.CALL_LIMIT)
    j.transport=Provider(cal,fail_write=failure=='write',bad_judge=failure=='judge')
    with forbid_provider_network() as audit:a=await b.run(live,cal,fixture,cfg,judge,j,dsn)
    assert a['complete'] and len(a['rows'])==12 and audit.blocked_requests==0
    assert all(r.get('policy_class')=='RightBrainPolicy' for r in a['rows'] if r['status']=='ok')
    if failure!='write':
        assert all(r['status']=='ok' for r in a['rows'])
        assert all(r['write_complete'] for r in a['rows'] if r['write'])
        assert a['rows'][6]['binding_status']=='invalid_selection'
        assert all(r['diagnostics_v3']['exact_match']['verdict']=='pass' for r in a['rows'] if r['id'] in
                   {'selected_replay','relisted_replay'})
        short=next(r for r in a['rows'] if r['id']=='explicit_rate_missing')
        assert short['condition']['action']=='preserve_unknown' and short['condition_provided']
        assert short['available_context']['missing_evidence']==short['condition']
        assert not a['contract_failures'] and a['trajectory_complete']
    else:
        assert a['rows'][0]['status']=='unavailable'
        assert any(r['status']=='not_executed' for r in a['rows'])
        assert not a['trajectory_complete'] and any(f['contract']=='turn_executed' for f in a['contract_failures'])
    assert a['calibration_summary']['failed_packets']==int(failure=='judge')
    if failure!='judge':assert a['calibration_summary']['matches']==24
    frozen=b.pilot.RawJournal(cfg,b.metadata(cal,fixture,cfg,judge),
        frozen=json.loads(json.dumps(j.data)),max_calls=b.CALL_LIMIT)
    with forbid_provider_network():c=await b.run(replay,cal,fixture,cfg,judge,frozen,dsn)
    assert frozen.transport is None and frozen.calls==0 and b.canonical(a)==b.canonical(c)
    assert a['usage']['calls']==sum(r['usage']['calls'] for r in a['rows'])+sum(r['usage']['calls'] for r in a['calibration'])+sum(g.get('usage',{}).get('calls',0) for g in a['grades'])
    public=json.dumps(b.public_summary(a))
    assert all(t not in public for t in ('"answer"','"question"','"quote"','r1.','"record_ref"','FAKE_NO_CREDENTIAL'))


def test_feedback_replay_changes_only_latency_and_rejects_different_controls():
    from core.hypothalamus import Hypothalamus
    class Journal:
        frozen=None;case='turn';data={}
        def checkpoint(self):pass
    j=Journal();tape=b.FeedbackTape(j);owner=Hypothalamus()
    tape.attach(owner,'hypothalamus','update_feedback')
    owner.update_feedback(reward=.75,latency_ms=7500.)
    tape.finish();expected=owner._latency_avg_ms
    other=Journal();other.data={};other.frozen=deepcopy(j.data)
    replay=b.FeedbackTape(other);new=Hypothalamus();replay.attach(new,'hypothalamus','update_feedback')
    new.update_feedback(reward=.75,latency_ms=1.)
    assert replay.finish()==j.data['feedback_sha256'] and new._latency_avg_ms==expected
    bad=Journal();bad.data={};bad.frozen=deepcopy(j.data)
    failed=b.FeedbackTape(bad);actor=Hypothalamus();failed.attach(actor,'hypothalamus','update_feedback')
    with pytest.raises(ValueError):actor.update_feedback(reward=.45,latency_ms=1.)
    with pytest.raises(ValueError):failed.finish()


def test_internal_eligible_records_and_conditions_are_not_backfilled_into_unexposed_semantic_context():
    cal=b.pilot.bench.read_json(b.CALIBRATION)
    condition=cal['cases'][0]['available_context']['missing_evidence']
    row={'available_context':{'reference':'resolved','approved_records':[{'qid':'hidden','answer':'PRIVATE_VALUE'}]},
         'source_qids':['hidden'],'memory_text':'PRIVATE_SOURCE_BLOCK','condition':condition}
    packet={'request':{'role':'draft','payload':{'messages':[{'role':'system','content':'Visible context only'}]}}}
    b.annotate_exposure(row,[packet])
    assert not row['source_exposure_confirmed'] and not row['condition_provided']
    assert row['source_qids'] is None and row['available_context']['approved_records'] is None
    assert row['available_context']['reference'] is None and 'missing_evidence' not in row['available_context']
    semantic=b.assessment.packet({'question':'What period?','requested_operation':'fact_lookup',
        'available_context':row['available_context']},{'A':{'answer':'Please clarify.'}})
    assert 'PRIVATE_' not in json.dumps(semantic) and semantic['available_context']['condition_status']=='unknown'
