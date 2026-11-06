from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sr-dual-brain-llm/scripts'))
import answer_assessment_v4 as a
from test_benchmark_contract_live import config, response
import benchmark_natural_memory as b


def calibration():
    return b.pilot.bench.read_json(b.CALIBRATION)


def test_condition_packet_excludes_oracles_expected_labels_and_unobserved_source_data():
    c=deepcopy(calibration()['cases'][0])
    c['available_context']['expected']='PRIVATE_EXPECTED'
    c['available_context']['missing_evidence']['hidden']='PRIVATE_HIDDEN_CONDITION'
    c['available_context']['generation_messages']=[{'stage':'draft','hidden':'PRIVATE_STAGE',
        'messages':[{'role':'user','content':'Actual visible context','expected':'PRIVATE_NESTED'}]}]
    result=a.packet(c,{'A':{'answer':'Please provide a current source.','expected':'PRIVATE_LABEL'}})
    text=json.dumps(result)
    assert all(t not in text for t in ('PRIVATE_','comparison_reference','variants','expected'))
    assert result['available_context']['condition_status']=='present'
    assert result['available_context']['missing_evidence']['parameter']=='data_retention_period'
    assert result['available_context']['generation_messages']==[{'stage':'draft','messages':[
        {'role':'user','content':'Actual visible context'}]}]


@pytest.mark.parametrize('change', ['missing_key','unknown_parameter','malformed_action','missing_condition','unknown_context'])
def test_malformed_or_missing_condition_does_not_become_a_known_parameter_or_absence(change):
    c=deepcopy(calibration()['cases'][0]);ctx=c['available_context']
    if change=='missing_key': ctx['missing_evidence'].pop('reference')
    elif change=='unknown_parameter':ctx['missing_evidence']['parameter']='inferred_from_private_history'
    elif change=='malformed_action':ctx['missing_evidence']['action']={'value':'request_current_source'}
    elif change=='missing_condition':ctx.pop('missing_evidence')
    else:c['available_context']=None
    result=a.packet(c,{'A':{'answer':'Question'}})['available_context']
    assert result is None or result['condition_status']=='unknown' and result['missing_evidence'] is None


@pytest.mark.parametrize('bad', [True, {}, [{'stage':[],'messages':[]}],
    [{'stage':'draft','messages':[{'role':[],'content':'Actual'}]}],
    [{'stage':'draft','messages':[{'role':'system','content':{'expected':'PRIVATE'}}]}]])
def test_bad_generation_context_stays_unknown_without_a_partial_packet(bad):
    c=deepcopy(calibration()['cases'][0]);c['available_context']['generation_messages']=bad
    result=a.packet(c,{'A':{'answer':'Question'}})
    assert result['available_context']['generation_messages'] is None


def test_v3_diagnostics_and_historical_packet_remain_independent_of_v4_semantics():
    c=calibration()['cases'][0]
    old=a.previous.packet(c,{'A':{'answer':'Current source please.'}})
    assert 'missing_evidence' not in old['available_context']
    result=a.previous.diagnostics(c,'Current source please.')
    assert result['source_validity']['verdict']=='fail'
    assert result['information_sufficiency']['verdict']=='unknown'
    assert a.schema({'A':{'answer':'x'}})['json_schema']['name']==a.VERSION


@pytest.mark.asyncio
@pytest.mark.parametrize('text,finish', [('', 'stop'),('Partial','length'),('{"scores":{}}','stop')])
async def test_invalid_judge_response_is_retained_without_repair(text,finish):
    j=b.pilot.RawJournal(config(),{'test':'v4'},max_calls=3)
    class Transport:
        async def _post_json(self,*args,**kwargs):return response(text,finish)
    j.transport=Transport()
    with pytest.raises(ValueError):
        await a.assess(calibration()['cases'][0],{'A':{'answer':'Please provide a current source.'}},config(),j)
    assert j.calls==1 and j.data['occurrences'][0]['response']==response(text,finish)
