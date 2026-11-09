"""A critic shares time across retries, repair, continuation and actual owners."""
import asyncio
import json

import pytest

from core.callosum import Callosum, remaining_consultation_ms
from core.llm_client import LLMConfig
from core.models import RightBrainModel
from core.right_worker import right_worker
from core.shared_memory import SharedMemory
import core.review_budget as budget

CLEAN = '{"verdict":"ok","issues":[],"fixes":[]}'
DRAFT,QUESTION = 'The supplied records differ.','Compare these supplied project records.'


def response(content=CLEAN,reason='stop'):
    return {'choices':[{'message':{'content':content},'finish_reason':reason}]}


def assert_unusable(result,status):
    assert result['critic_status']==status
    assert not result['critic_review_valid'] and not result['critic_external_review_valid']
    assert result['confidence_r']==0.


def fake_clock(monkeypatch):
    now = [100.]
    monkeypatch.setattr(budget, '_now', lambda: now[0])
    monkeypatch.setenv('DUALBRAIN_CRITIC_TIMEOUT_SECONDS', '24')
    return now


@pytest.mark.asyncio
async def test_repair_uses_time_left_by_initial_review_and_budget_does_not_leak(monkeypatch):
    now = fake_clock(monkeypatch)
    model = RightBrainModel(LLMConfig('openai','offline-critic','NO_CREDENTIAL'))
    seen = []
    class Client:
        async def complete(self, _prompt, **kwargs):
            seen.append(kwargs)
            now[0] += 20 if len(seen)==1 else 1
            return 'Unstructured review.' if len(seen)==1 else CLEAN
    model._llm_client = Client()
    result = await model.criticise_reasoning('repair',QUESTION,DRAFT,allow_micro_fallback=False)
    assert result['critic_external_review_valid'] and result['critic_kind']=='external_repaired'
    assert [x['timeout_seconds'] for x in seen]==[24.,4.]
    assert all('no self-critique' not in x['system'] and 'Return a SINGLE JSON' in x['system'] for x in seen)
    assert budget.review_timeout_seconds(60.)==60.


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['late_success','expired_retry','expired_initial'])
async def test_exhausted_review_cannot_restart_or_accept_a_late_result(monkeypatch,failure):
    now = fake_clock(monkeypatch)
    model = RightBrainModel(LLMConfig('openai','offline-critic','NO_CREDENTIAL',
        timeout_seconds=0 if failure=='expired_initial' else 40))
    calls = []
    class Client:
        async def complete(self,*args,**kwargs):
            calls.append(kwargs)
            now[0] += 25
            if failure=='expired_retry': raise RuntimeError('timeout PRIVATE_PROVIDER_TEXT')
            return CLEAN
    model._llm_client = Client()
    result = await model.criticise_reasoning('expired',QUESTION,DRAFT,allow_micro_fallback=False)
    assert_unusable(result,'deadline_exhausted')
    assert len(calls)==(0 if failure=='expired_initial' else 1)
    assert 'PRIVATE_PROVIDER_TEXT' not in json.dumps(result)
    assert budget.review_timeout_seconds(60.)==60.


@pytest.mark.asyncio
async def test_each_continuation_http_uses_the_same_budget_and_normal_http_is_unchanged(monkeypatch):
    now = fake_clock(monkeypatch)
    model = RightBrainModel(LLMConfig('openai','offline-critic','NO_CREDENTIAL',
        auto_continue=True,max_continuations=1))
    replies = [response('{"verdict":"ok","issues":[],','length'),response('"fixes":[]}'),response()]
    timeouts = []
    class Reply:
        status = 200
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def json(self,**kwargs):
            index = len(timeouts)-1
            now[0] += 20 if index==0 else 1
            return replies[index]
    class Session:
        def __init__(self,*,timeout,**kwargs): timeouts.append(timeout.total)
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        def post(self,*args,**kwargs): return Reply()
    monkeypatch.setattr('core.llm_client.aiohttp.ClientSession',Session)
    result = await model.criticise_reasoning('continuation',QUESTION,DRAFT,allow_micro_fallback=False)
    assert result['critic_external_review_valid']
    assert timeouts==[24.,4.]
    await model._llm_client._post_json('https://fake.invalid',{}, {},timeout_seconds=60.)
    assert timeouts==[24.,4.,60.]


@pytest.mark.asyncio
async def test_expired_continuation_does_not_open_another_http_session(monkeypatch):
    now = fake_clock(monkeypatch)
    model = RightBrainModel(LLMConfig('openai','offline-critic','NO_CREDENTIAL',
        auto_continue=True,max_continuations=1))
    sessions = []
    class Reply:
        status = 200
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def json(self,**kwargs):
            now[0] += 25
            return response('{"verdict":"ok",','length')
    class Session:
        def __init__(self,**kwargs): sessions.append(kwargs)
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        def post(self,*args,**kwargs): return Reply()
    monkeypatch.setattr('core.llm_client.aiohttp.ClientSession',Session)
    result = await model.criticise_reasoning('continuation',QUESTION,DRAFT,allow_micro_fallback=False)
    assert_unusable(result,'deadline_exhausted')
    assert len(sessions)==1


@pytest.mark.asyncio
async def test_real_bus_deadline_overrides_forged_packet_and_clears_after_operation():
    bus = Callosum(slot_ms=0)
    caller = asyncio.create_task(bus.ask_detail({'qid':'owner','deadline_at':float('inf')},timeout_ms=500))
    request = await bus.recv_request()
    request['deadline_at'] = float('inf')
    async def operation():
        remaining = remaining_consultation_ms()
        assert 0 < remaining <= 500
        return {'remaining':remaining}
    try:
        result = await bus.run_request(request,operation)
        assert remaining_consultation_ms() is None
        await bus.publish_response(request['request_id'],result)
        assert (await caller)['remaining'] <= 500
    finally:
        await bus.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('abandon',['cancel','deadline'])
async def test_owner_abandonment_stops_repair_even_when_client_swallows_cancellation(monkeypatch,abandon):
    monkeypatch.setenv('DUALBRAIN_CRITIC_MICRO_FALLBACK','0')
    monkeypatch.setenv('DUALBRAIN_CRITIC_TIMEOUT_SECONDS','24')
    bus = Callosum(slot_ms=0)
    model = RightBrainModel(LLMConfig('openai','offline-critic','NO_CREDENTIAL'))
    started = asyncio.Event()
    calls, remaining_after_cancel, admitted_remaining = [], [], []
    class Client:
        async def complete(self,*args,**kwargs):
            calls.append(kwargs)
            admitted_remaining.append(remaining_consultation_ms())
            if len(calls)==1:
                started.set()
                try: await asyncio.Event().wait()
                except asyncio.CancelledError:
                    remaining_after_cancel.append(remaining_consultation_ms())
                    return 'Unstructured abandoned result.'
            return CLEAN
    model._llm_client = Client()
    worker = asyncio.create_task(right_worker(bus,SharedMemory(),model))
    request = {'qid':'abandoned','type':'ASK_CRITIC','question':QUESTION,'draft':DRAFT}
    owner_ms=50 if abandon=='deadline' else 500
    caller = asyncio.create_task(bus.ask_detail(request,timeout_ms=owner_ms))
    try:
        await asyncio.wait_for(started.wait(),1.)
        if abandon=='cancel':
            caller.cancel()
            with pytest.raises(asyncio.CancelledError): await caller
        else:
            with pytest.raises(asyncio.TimeoutError): await caller
        await bus.wait_idle()
        assert remaining_after_cancel==[0.] and len(calls)==1
        assert 0 < calls[0]['timeout_seconds'] <= 24.
        assert 0 < admitted_remaining[0] <= owner_ms
        result = await bus.ask_detail(request | {'qid':'next'},timeout_ms=1000)
        assert result['critic_external_review_valid'] and len(calls)==2
        assert remaining_consultation_ms() is None
        assert budget.review_timeout_seconds(60.)==60.
    finally:
        await bus.close()
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize('short_owner',[True,False])
async def test_http_does_not_compete_with_owner_and_local_window_failure_stays_typed(monkeypatch,short_owner):
    monkeypatch.setenv('DUALBRAIN_CRITIC_MICRO_FALLBACK','0')
    config=LLMConfig('openai','offline-critic','NO_CREDENTIAL',timeout_seconds=.5 if short_owner else .05)
    model=RightBrainModel(config)
    bus=Callosum(slot_ms=0)
    timeouts,owners,cancelled=[],[],[]
    class Reply:
        status=200
        def __init__(self,total):self.total=total
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def json(self,**kwargs):
            try:
                await asyncio.wait_for(asyncio.sleep(.15),self.total)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
            return response()
    class Session:
        def __init__(self,*,timeout,**kwargs):
            self.total=timeout.total;timeouts.append(self.total);owners.append(remaining_consultation_ms())
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        def post(self,*args,**kwargs):return Reply(self.total)
    monkeypatch.setattr('core.llm_client.aiohttp.ClientSession',Session)
    worker=asyncio.create_task(right_worker(bus,SharedMemory(),model))
    request={'qid':'ownership','type':'ASK_CRITIC','question':QUESTION,'draft':DRAFT}
    try:
        if short_owner:
            with pytest.raises(asyncio.TimeoutError):await bus.ask_detail(request,timeout_ms=50)
            await bus.wait_idle()
            assert .1<timeouts[0]<=.5 and 0<owners[0]<=50
            assert cancelled==[True] and bus.snapshot()['expired']==1
        else:
            result=await bus.ask_detail(request,timeout_ms=500)
            assert result['critic_status']=='deadline_exhausted'
            assert result['critic_validation']['initial']['error_type']=='ReviewBudgetExhausted'
            assert not result['critic_external_review_valid'] and bus.snapshot()['expired']==0
            assert 0<timeouts[0]<=.05
    finally:
        await bus.close();await worker
    assert budget.review_timeout_seconds(60.)==60.
