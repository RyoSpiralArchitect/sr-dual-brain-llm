import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sr-dual-brain-llm/scripts'))
import benchmark_selective_consultation as bench
from core.llm_client import LLMConfig


def test_frozen_fixture_separates_development_and_preserves_counterfactual_provenance():
    fixture=bench.read_json(bench.FIXTURE)
    bench.validate(fixture)
    evaluation=[c for c in fixture['cases'] if c['split']=='evaluation']
    assert len(evaluation)==8 and len({c['problem_id'] for c in evaluation})==6
    assert sum(c['origin']=='model' for c in evaluation)==4
    assert {c['origin'] for c in evaluation}=={'model','synthetic_clean','synthetic_faulty'}


@pytest.mark.parametrize('missing_usage', [False, True])
def test_real_sessions_frozen_draft_identity_selective_telemetry_and_exact_replay(tmp_path, monkeypatch, missing_usage):
    monkeypatch.delenv('DUALBRAIN_MEMORY_EMBEDDING_MODEL',raising=False)
    monkeypatch.delenv('DUALBRAIN_MEMORY_ANSWERABILITY_MODEL',raising=False)
    fixture=bench.read_json(bench.FIXTURE)
    config=LLMConfig('openai','fake-model','NEVER_SAVE_THIS_KEY',auto_continue=False)
    args=SimpleNamespace(phase='development',output_dir=tmp_path/'live',judge_model='fake-judge')
    args.output_dir.mkdir()
    journal=bench.base.Journal(config,{},max_calls=64)
    async def post(url,payload,headers,**kwargs):
        fmt=payload.get('response_format',{})
        if fmt.get('type')=='json_schema':
            packet=json.loads(payload['messages'][-1]['content'])
            text=json.dumps({'scores':{label:{c:{'verdict':'pass','quote':answer} for c in bench.base.CRITERIA}
                                      for label,answer in packet['answers'].items()}})
        elif fmt:
            packet=json.loads(payload['messages'][-1]['content'])
            text=json.dumps({'decisions':[{'id':r['id'],'supported':True,'quote':r['answer']} for r in packet['records']]})
        elif 'rigorous reviewer' in payload['messages'][0]['content']:
            text=json.dumps({'verdict':'ok','issues':[],'fixes':[]})
        else:
            text='According to the record, retention is 11 days.'
        response={'model':payload['model'],'choices':[{'finish_reason':'stop','message':{'content':text}}]}
        if not missing_usage:
            response['usage']={'prompt_tokens':1,'completion_tokens':1}
        return response
    journal.transport._post_json=post
    report=asyncio.run(bench.run(args,fixture,config,journal))
    assert report['complete'] and report['grader_failures']==0 and len(report['rows'])==8
    assert (report['input_tokens'] is None) is missing_usage
    assert (report['summaries']['all']['selective_auto']['input_tokens'] is None) is missing_usage
    for case in ('dev-lookup','dev-calculation'):
        rows=[r for r in report['rows'] if r['episode']==case and r['arm']!='no_revision']
        assert len({r['draft_sha256'] for r in rows})==1
    lookup=next(r for r in report['rows'] if r['episode']=='dev-lookup' and r['arm']=='selective_auto')
    assert lookup['route']['action']=='skip' and lookup['calls_by_role'].get('right',0)==0
    assert not lookup['metrics']['system2']['enabled']
    assert 'NEVER_SAVE_THIS_KEY' not in json.dumps(journal.data)
    args.output_dir=tmp_path/'replay'
    args.output_dir.mkdir()
    replay=bench.base.Journal(config,{},frozen=journal.data,max_calls=64)
    async def forbidden(*args,**kwargs):
        pytest.fail('Replay attempted a provider call')
    replay.transport._post_json=forbidden
    repeated=asyncio.run(bench.run(args,fixture,config,replay))
    assert repeated['provider_calls']==0 and repeated['grades']==report['grades']
    assert [r['answer'] for r in repeated['rows']]==[r['answer'] for r in report['rows']]


def test_returned_empty_completion_is_not_a_quota_failure():
    blank={'error_type':'ValueError','response':{'choices':[{'finish_reason':'stop','message':{'content':''}}]}}
    assert bench.generation_failure(blank)
    assert not bench.generation_failure({'error_type':'RuntimeError','http_status':429})
    assert not bench.generation_failure({'error_type':'CancelledError'})


def test_continuation_never_retries_failed_or_completed_cells_and_replays_all_requests(tmp_path,monkeypatch):
    fixture=bench.read_json(bench.FIXTURE)
    config=LLMConfig('openai','fake-model','PRIVATE_KEY',auto_continue=False)
    args=SimpleNamespace(phase='development',output_dir=tmp_path/'prefix',judge_model='fake-judge')
    args.output_dir.mkdir()
    metadata={'model':'fake-model','judge_model':'fake-judge'}
    journal=bench.base.Journal(config,metadata,max_calls=64,
        save=lambda d:bench.checkpoint(args.output_dir/'journal.partial.json',d))
    active={}
    original_session=bench.session
    async def track(case,arm,*rest):
        active['cell']=(case['id'],arm)
        return await original_session(case,arm,*rest)
    monkeypatch.setattr(bench,'session',track)
    failed=[]
    calls=[]
    async def post(url,payload,headers,**kwargs):
        calls.append(active['cell'])
        fmt=payload.get('response_format',{})
        if fmt.get('type')=='json_schema':
            packet=json.loads(payload['messages'][-1]['content'])
            text=json.dumps({'scores':{label:{c:{'verdict':'pass','quote':answer} for c in bench.base.CRITERIA}
                                       for label,answer in packet['answers'].items()}})
        elif fmt:
            packet=json.loads(payload['messages'][-1]['content'])
            text=json.dumps({'decisions':[{'id':r['id'],'supported':True,'quote':r['answer']} for r in packet['records']]})
        elif 'rigorous reviewer' in payload['messages'][0]['content']:
            if not failed:
                failed.append(active['cell']); text=''
            else:
                text=json.dumps({'verdict':'ok','issues':[],'fixes':[]})
        else:
            text='According to the record, retention is 11 days.'
        return {'model':payload['model'],'choices':[{'finish_reason':'stop','message':{'content':text}}],
                'usage':{'prompt_tokens':2,'completion_tokens':1}}
    journal.transport._post_json=post
    checkpoint=bench.checkpoint
    def interrupt(path,data):
        checkpoint(path,data)
        if path.name=='report.partial.json' and data['rows'] and data['rows'][-1].get('failure'):
            raise RuntimeError('Simulated older runner interruption')
    monkeypatch.setattr(bench,'checkpoint',interrupt)
    with pytest.raises(RuntimeError,match='Simulated'):
        asyncio.run(bench.run(args,fixture,config,journal))
    prefix=args.output_dir
    # Emulate the older checkpoint, which predated per-cell failure accounting.
    partial=bench.read_json(prefix/'report.partial.json')
    for key in ('usage','calls_by_role','elapsed_ms','route','failure'):
        partial['rows'][-1].pop(key,None)
    checkpoint(prefix/'report.partial.json',partial)
    frozen_prefix=bench.read_json(prefix/'journal.partial.json')
    frozen_prefix['transport_blocked']=True
    checkpoint(prefix/'journal.partial.json',frozen_prefix)
    monkeypatch.setattr(bench,'checkpoint',checkpoint)
    args.output_dir=tmp_path/'continued'; args.output_dir.mkdir(); args.resume=prefix
    continued=bench.base.Journal(config,metadata,max_calls=64)
    continued.transport._post_json=post
    wrong_model=bench.base.Journal(LLMConfig('openai','different-model','PRIVATE_KEY'),metadata,max_calls=64)
    with pytest.raises(ValueError):
        bench.resume_prefix(args,fixture,LLMConfig('openai','different-model','PRIVATE_KEY'),wrong_model)
    with pytest.raises(ValueError):
        bench.resume_prefix(args,fixture,config,bench.base.Journal(config,metadata,max_calls=1))
    calls.clear()
    result=asyncio.run(bench.run(args,fixture,config,continued))
    assert result['complete'] and result['grader_failures']==0
    assert failed[0] not in calls
    unavailable=[r for r in result['rows'] if r['status']!='ok']
    assert len(unavailable)==1 and unavailable[0]['answer'] is None and unavailable[0]['elapsed_ms'] is None
    assert len(result['grades'])==7 and continued.calls>len(frozen_prefix['responses'])
    args.output_dir=tmp_path/'replay'; args.output_dir.mkdir(); args.resume=None
    replay=bench.base.Journal(config,metadata,frozen=continued.data,max_calls=64)
    repeated=asyncio.run(bench.run(args,fixture,config,replay))
    assert repeated['provider_calls']==0 and repeated['grades']==result['grades']
    assert [r['answer'] for r in repeated['rows']]==[r['answer'] for r in result['rows']]
