"""Serial controller cleanup preserves cancellation and minimum-version behavior."""
import asyncio
import os
from pathlib import Path
import subprocess
import sys

import pytest
from core.callosum import drain_consultation
from core.turn_lifecycle import TurnDispatcher


@pytest.mark.asyncio
@pytest.mark.parametrize('cleanup_error', [False, True])
async def test_drain_propagates_repeated_owner_cancel_only_after_cleanup(cleanup_error):
    started, release, finished = [asyncio.Event() for _ in range(3)]
    class Channel:
        async def wait_idle(self):
            started.set()
            await release.wait()
            finished.set()
            if cleanup_error:
                raise RuntimeError('PRIVATE_CLEANUP_FAILURE')
    task = asyncio.create_task(drain_consultation(Channel()))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not finished.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize('abandon', ['cancel', 'deadline'])
@pytest.mark.parametrize('during', [False, True])
async def test_cleanup_and_local_timeout_do_not_spend_rpc_cancellation(abandon, during):
    started, release, done = [asyncio.Event() for _ in range(3)]
    replies = []
    class Channel:
        async def wait_idle(self):
            started.set()
            await release.wait()
            done.set()
    async def handle(req):
        if req['id'] == 'first':
            await drain_consultation(Channel())
            await asyncio.Event().wait()
        return {'id': req['id'], 'ok': True}
    dispatcher = TurnDispatcher(handle, replies.append, timeout_seconds=.12)
    async def until(predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(.001)
        await asyncio.wait_for(wait(), 1.)
    try:
        dispatcher.submit({'id': 'first'})
        first_state = dispatcher.requests['first']
        await asyncio.wait_for(started.wait(), 1.)
        if not during:
            release.set()
            await done.wait()
        if abandon == 'cancel':
            assert dispatcher.cancel('first')['accepted']
        else:
            await until(lambda: first_state.reason is not None)
        if during:
            await asyncio.sleep(0)
            assert not replies
            release.set()
        await until(lambda: len(replies) == 1)
        dispatcher.submit({'id': 'second'})
        await until(lambda: len(replies) == 2)
        assert done.is_set() and not replies[0]['ok']
        assert replies[0]['error']['type'] == ('Cancelled' if abandon == 'cancel' else 'DeadlineExceeded')
        assert replies[1] == {'id': 'second', 'ok': True}
        assert dispatcher.snapshot()['pending'] == 0
    finally:
        release.set()
        await dispatcher.close()


def test_done_operation_callback_cannot_starve_the_idle_wait():
    code = '''
import asyncio
from core.callosum import Callosum
async def main():
    bus = Callosum(slot_ms=0)
    done = asyncio.get_running_loop().create_future()
    done.set_result(None)
    bus._running.add(done)
    asyncio.get_running_loop().call_soon(bus._operation_done, done)
    await bus.wait_idle()
    assert not bus._running
asyncio.run(main())
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm'))
    subprocess.run([sys.executable, '-c', code], env=env, check=True, timeout=5., capture_output=True)
