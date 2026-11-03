"""Local consultation timers must preserve later RPC abandonment on Python 3.10+."""
import asyncio

import pytest

from core.callosum import Callosum
from core.turn_lifecycle import TurnDispatcher


async def until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(poll(), 2.)


async def close_dispatcher(dispatcher):
    # A regression must fail rather than leave the test runner hanging. Healthy
    # paths have already drained; bypass the custom guard only for failed cleanup.
    for state in list(dispatcher.requests.values()):
        if state.task is not None and not state.task.done():
            asyncio.Task.cancel(state.task)
    await dispatcher.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('slot_ms', [0, 1000])
@pytest.mark.parametrize('abandon', ['cancel', 'deadline'])
async def test_local_consultation_timeouts_preserve_rpc_cancel_and_deadline(slot_ms, abandon):
    bus = Callosum(slot_ms=slot_ms)
    local_timeouts, responses, calls = asyncio.Event(), [], []

    async def handle(req):
        calls.append(req['id'])
        if req['id'] == 'first':
            for _ in range(2):
                with pytest.raises(asyncio.TimeoutError):
                    await bus.ask_detail({'qid':'local-timeout'}, timeout_ms=5)
            local_timeouts.set()
            await asyncio.Event().wait()
        return {'id':req['id'], 'ok':True}

    dispatcher = TurnDispatcher(handle, responses.append, timeout_seconds=.12)
    try:
        dispatcher.submit({'id':'first'})
        await asyncio.wait_for(local_timeouts.wait(), 1.)
        assert bus.snapshot()['expired'] == 2
        assert bus.snapshot()['pending'] == bus.snapshot()['queued'] == 0
        dispatcher.submit({'id':'second'})
        if abandon == 'cancel':
            assert dispatcher.cancel('first') == {'accepted':True, 'deferred':False}
        await until(lambda:len(responses) == 2)
        assert calls == ['first', 'second']
        assert responses[0]['error']['type'] == ('Cancelled' if abandon == 'cancel' else 'DeadlineExceeded')
        assert not responses[0]['error']['may_have_committed']
        assert responses[1] == {'id':'second', 'ok':True}
        assert dispatcher.snapshot()['pending'] == 0
    finally:
        await close_dispatcher(dispatcher)
        await bus.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('abandon', ['cancel', 'deadline'])
async def test_controller_hard_consultation_timer_drains_child_and_keeps_rpc_cancellable(abandon):
    # Import EngineSession dependencies only for the full Python job; the direct
    # Callosum tests above remain usable with the minimum-version dependencies.
    from test_dual_brain import AlwaysConsultPolicy, make_stage_controller

    started, cleaning, release, local_timeout = [asyncio.Event() for _ in range(4)]
    finished, calls, responses = [], [], []

    class StalledCallosum:
        slot_ms = 123

        async def ask_detail(self, payload, timeout_ms=3000):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                finished.append('cleaned')

    controller = make_stage_controller()
    controller.policy = AlwaysConsultPolicy()
    controller.callosum = channel = StalledCallosum()

    async def consult_stage(**kwargs):
        with pytest.raises(asyncio.TimeoutError):
            await kwargs['ask_callosum_with_timeout']({'qid':'hard-timeout'}, timeout_ms=1)
        assert channel.slot_ms == 123 and finished == ['cleaned']
        local_timeout.set()
        await asyncio.Event().wait()

    controller._run_right_consult_stage = consult_stage

    async def handle(req):
        calls.append(req['id'])
        if req['id'] == 'first':
            await controller.process('Explain bounded queues under burst traffic.',
                leading_brain='left', system2_mode='off', executive_mode='off', executive_observer_mode='off')
        return {'id':req['id'], 'ok':True}

    dispatcher = TurnDispatcher(handle, responses.append, timeout_seconds=.8)
    try:
        dispatcher.submit({'id':'first'})
        await asyncio.wait_for(started.wait(), 1.)
        await asyncio.wait_for(cleaning.wait(), 1.)
        dispatcher.submit({'id':'second'})
        assert not local_timeout.is_set() and calls == ['first'] and not responses
        release.set()
        await asyncio.wait_for(local_timeout.wait(), 1.)
        if abandon == 'cancel':
            assert dispatcher.cancel('first') == {'accepted':True, 'deferred':False}
        await until(lambda:len(responses) == 2)
        assert calls == ['first', 'second'] and finished == ['cleaned']
        assert responses[0]['error']['type'] == ('Cancelled' if abandon == 'cancel' else 'DeadlineExceeded')
        assert not responses[0]['error']['may_have_committed']
        assert responses[1] == {'id':'second', 'ok':True}
        assert channel.slot_ms == 123 and dispatcher.snapshot()['pending'] == 0
    finally:
        release.set()
        await close_dispatcher(dispatcher)
