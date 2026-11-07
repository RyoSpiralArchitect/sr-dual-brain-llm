"""Preview ownership works with the minimum Python dependencies/version."""
import asyncio

import pytest

from core.speculative_preview import OwnedPreview
from core.turn_lifecycle import TurnDispatcher


@pytest.mark.asyncio
@pytest.mark.parametrize('requested,suppressed', [(False, False), (False, True), (True, True)])
async def test_non_admitted_preview_never_calls_generator(requested, suppressed):
    work = OwnedPreview(requested=requested, suppressed=suppressed)
    async def forbidden():
        pytest.fail('Suppressed work must not start')
    work.start(forbidden)
    await work.close()
    assert work.task is None and not work.observation['admitted'] and not work.observation['started']
    assert work.observation['outcome'] == ('suppressed' if requested else 'not_requested')


@pytest.mark.asyncio
async def test_cancellation_before_first_run_keeps_generator_unstarted():
    work = OwnedPreview(requested=True)
    async def forbidden():
        pytest.fail('The task was cancelled before its first run')
    work.start(forbidden)
    await work.close()
    assert work.task.done() and work.task.cancelled()
    assert work.observation['admitted'] and not work.observation['started']
    assert work.observation['outcome'] == 'cancelled_before_start'


@pytest.mark.asyncio
@pytest.mark.parametrize('result,outcome', [('notes', 'completed'), ('', 'empty'), (RuntimeError('PRIVATE'), 'failed')])
async def test_completed_or_failed_preview_is_observed_without_error_text(result, outcome):
    work = OwnedPreview(requested=True)
    async def generate():
        if isinstance(result, BaseException):
            raise result
        return result
    work.start(generate)
    await asyncio.gather(work.task, return_exceptions=True)
    await work.close()
    assert work.observation['outcome'] == outcome
    assert work.observation['available'] is (outcome == 'completed')
    assert 'PRIVATE' not in str(work.observation)


@pytest.mark.asyncio
@pytest.mark.parametrize('abandon', ['cancel', 'deadline'])
async def test_owning_turn_drains_cancelled_preview_before_next_request(abandon):
    entered, cleaning, release = [asyncio.Event() for _ in range(3)]
    calls, responses, observations = [], [], []
    async def generate():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
    async def handle(req):
        calls.append(req['id'])
        if req['id'] == 'first':
            work = OwnedPreview(requested=True)
            work.start(generate)
            try:
                await asyncio.Event().wait()
            finally:
                try:
                    await work.close()
                finally:
                    observations.append(dict(work.observation))
        return {'id': req['id'], 'ok': True}
    dispatcher = TurnDispatcher(handle, responses.append, timeout_seconds=.3)
    try:
        dispatcher.submit({'id': 'first'})
        await asyncio.wait_for(entered.wait(), 1.)
        if abandon == 'cancel':
            assert dispatcher.cancel('first')['accepted']
        await asyncio.wait_for(cleaning.wait(), 1.)
        dispatcher.submit({'id': 'second'})
        assert calls == ['first'] and not responses
        release.set()
        async def finished():
            while len(responses) != 2:
                await asyncio.sleep(.001)
        await asyncio.wait_for(finished(), 1.)
        assert calls == ['first', 'second']
        assert responses[0]['error']['type'] == ('Cancelled' if abandon == 'cancel' else 'DeadlineExceeded')
        assert responses[1] == {'id': 'second', 'ok': True}
        assert observations[0]['outcome'] == 'cancelled_after_start'
        assert dispatcher.snapshot()['pending'] == 0
    finally:
        release.set()
        await dispatcher.close()


@pytest.mark.asyncio
async def test_cancellation_arriving_during_close_preserves_cleanup_and_observation():
    entered, cleaning, release = [asyncio.Event() for _ in range(3)]
    work = OwnedPreview(requested=True)
    async def generate():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
    work.start(generate)
    await entered.wait()
    closer = asyncio.create_task(work.close())
    await cleaning.wait()
    closer.cancel()
    await asyncio.sleep(0)
    assert not closer.done() and not work.task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closer
    assert work.task.done() and work.observation['outcome'] == 'cancelled_after_start'
