"""Real completion validation and worker delivery cannot turn empty detail into success."""
import asyncio
from copy import deepcopy
import json
import os
from unittest.mock import patch

import pytest

from core.detail_contract import detail_observation
from core.llm_client import LLMConfig
from core.models import RightBrainModel
from engine_stdio import EngineSession, _handle_process
from test_dual_brain import AlwaysConsultPolicy


DRAFT = 'The draft already answers the question.'


async def run_detail(tmp_path, *, content='', finish='stop', transport_error=False, delivery_error=False):
    config = LLMConfig('openai', 'detail-fake', 'FAKE_NO_CREDENTIAL', api_base='https://fake.invalid/v1',
                       auto_continue=True, max_continuations=2)
    calls = []
    with patch.dict(os.environ, {}, clear=True):
        session = await EngineSession.create('detail', right_llm_config=config)
        async def draft(*_args, **_kwargs): return DRAFT
        session.left.generate_answer = draft
        session.controller.policy = AlwaysConsultPolicy()
        async def post(_url, payload, _headers, **_kwargs):
            calls.append(deepcopy(payload))
            if transport_error: raise RuntimeError('PRIVATE_CREDENTIAL_MUST_NOT_APPEAR')
            return {'choices':[{'message':{'content':content},'finish_reason':finish}]}
        session.right._llm_client._post_json = post
        if delivery_error:
            async def unavailable(*_args, **_kwargs): return {'error':'unavailable'}
            session.callosum.ask_detail = unavailable
        try:
            result = await _handle_process(session, {'question':'Explain the supplied plan.',
                'system2_mode':'off','leading_brain':'left','return_telemetry':True},
                blob_root=tmp_path,blob_index={},vision_cache={},vision_cache_limit=0)
            return result, calls
        finally:
            await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('content,finish,status', [
    ('','stop','empty_response'), ('   ','stop','empty_response'),
    ('','length','empty_response'), ('Partial advice.','length','incomplete_response'),
    ('A usable additional condition.','stop','ok'),
])
async def test_actual_worker_preserves_completion_status_without_paid_repeat(tmp_path, content, finish, status):
    result, calls = await run_detail(tmp_path, content=content, finish=finish)
    assert len(calls) == 1  # Also no continuation despite an enabled client config.
    detail = result['metrics']['detail']
    assert detail['deliveries'] == [{'source':'callosum','status':status,'kind':'external',
        'usable':status=='ok','external_complete':status=='ok',
        'error_type':None if status=='ok' else 'CompletionValidationError'}]
    assert detail['usable'] is (status=='ok') and not detail['fallback_used']
    if status != 'ok': assert result['answer'] == DRAFT
    assert not result['metrics']['system2']['used']
    assert 'notes_sum' not in json.dumps(detail) and 'Partial advice.' not in json.dumps(detail)


@pytest.mark.asyncio
async def test_provider_failure_is_not_heuristic_success_or_another_provider_attempt(tmp_path):
    result, calls = await run_detail(tmp_path, transport_error=True)
    detail = result['metrics']['detail']
    assert len(calls)==1 and result['answer']==DRAFT and not detail['usable']
    assert detail['deliveries'][0]['status']=='failed' and not detail['fallback_used']
    assert 'PRIVATE_CREDENTIAL' not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize('content,status,usable', [('', 'empty_response', False), ('Extra condition.', 'ok', True)])
async def test_delivery_failure_keeps_one_existing_fallback_and_checks_its_result(tmp_path, content, status, usable):
    result, calls = await run_detail(tmp_path, content=content, delivery_error=True)
    detail = result['metrics']['detail']
    assert len(calls)==1 and detail['fallback_used'] and detail['usable'] is usable
    assert [d['source'] for d in detail['deliveries']]==['callosum','direct_fallback']
    assert [d['status'] for d in detail['deliveries']]==['delivery_error',status]
    if not usable: assert result['answer']==DRAFT


def test_legacy_and_malformed_delivery_never_claim_external_completion():
    assert detail_observation({'notes_sum':'Legacy note.'})['usable']
    assert not detail_observation({'notes_sum':'Legacy note.'})['external_complete']
    for response in ({}, {'notes_sum':' '}, {'notes_sum':True}, {'detail_status':[]},
                     {'detail_kind':{}}, {'detail_kind':'external','notes_sum':'Unvalidated'},
                     {'detail_kind':'heuristic','detail_usable':False,'notes_sum':'Not usable'},
                     {'detail_kind':'unavailable','notes_sum':'Not available'}):
        observation=detail_observation(response)
        assert not observation['usable'] and not observation['external_complete']


@pytest.mark.asyncio
async def test_cancellation_during_provider_call_is_not_converted_to_a_detail_failure():
    model=RightBrainModel(LLMConfig('openai','cancel-detail','FAKE_NO_CREDENTIAL',api_base='https://fake.invalid/v1'))
    async def cancelled(*_args, **_kwargs): raise asyncio.CancelledError()
    model._llm_client._post_json=cancelled
    with pytest.raises(asyncio.CancelledError):
        await model.deepen('cancel','Question','Draft',None)
