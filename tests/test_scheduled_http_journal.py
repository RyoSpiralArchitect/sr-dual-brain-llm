"""Delayed transport, owner cancellation and schedule integrity under no HTTP."""
import asyncio
from copy import deepcopy
import json

import pytest

from test_benchmark_contract_live import config, response
from scheduled_http_journal import ScheduledJournal
from offline_provider_guard import forbid_provider_network


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['completed', 'cancelled', 'error'])
async def test_overlapping_transport_schedule_replays_by_real_owner(outcome):
    preview_entered, release_draft = asyncio.Event(), asyncio.Event()
    class Provider:
        async def _post_json(self, _url, payload, _headers):
            if payload['kind'] == 'draft':
                await release_draft.wait()
                return response('draft')
            preview_entered.set()
            if outcome == 'cancelled':
                release_draft.set()
                await asyncio.Event().wait()
            if outcome == 'error':
                release_draft.set()
                raise OSError('PRIVATE_ERROR')
            release_draft.set()
            return response('preview')
    async def execute(journal):
        draft = asyncio.create_task(journal.post('draft', 'fake', {'kind': 'draft'}, {}))
        await asyncio.sleep(0)
        preview = asyncio.create_task(journal.post('right', 'fake', {'kind': 'preview'}, {}))
        answer = await draft
        if not preview.done():
            preview.cancel()
        results = await asyncio.gather(preview, return_exceptions=True)
        journal.finish()
        return answer, results
    journal = ScheduledJournal(config(), {'test': outcome}, max_calls=4)
    journal.transport = Provider()
    with forbid_provider_network() as audit:
        first = await execute(journal)
        replay = ScheduledJournal(config(), {'test': outcome}, max_calls=4,
                                  frozen=json.loads(json.dumps(journal.data)))
        second = await execute(replay)
    assert replay.transport is None and replay.calls == 0 and audit.blocked_requests == 0
    assert replay.data['transport_events'] == journal.data['transport_events']
    assert replay.events == journal.events
    assert first[0] == second[0]
    assert type(first[1][0]) is type(second[1][0])
    assert len(journal.data['occurrences']) == 2
    usage = journal.data['usage']
    assert usage['unknown_usage_calls'] == int(outcome != 'completed')
    assert usage['cancelled_calls'] == int(outcome == 'cancelled')
    assert 'PRIVATE' not in json.dumps(journal.data)


@pytest.mark.asyncio
async def test_cancelled_before_admission_has_no_transport_events_or_attempts():
    journal = ScheduledJournal(config(), {'test': 'before'})
    task = asyncio.create_task(journal.post('right', 'fake', {}, {}))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    journal.finish()
    assert journal.calls == 0 and journal.data['transport_events'] == []
    replay = ScheduledJournal(config(), {'test': 'before'}, frozen=deepcopy(journal.data))
    replay.finish()


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['digest', 'completion_before_start', 'missing_completion', 'wrong_kind'])
async def test_schedule_tampering_is_rejected_before_replay(change):
    journal = ScheduledJournal(config(), {'test': 'integrity'})
    class Provider:
        async def _post_json(self, *_args, **_kwargs):
            return response()
    journal.transport = Provider()
    await journal.post('draft', 'fake', {}, {})
    journal.finish()
    frozen = deepcopy(journal.data)
    if change == 'digest':
        frozen['transport_events_sha256'] = 'invalid'
    else:
        if change == 'completion_before_start': frozen['transport_events'].reverse()
        elif change == 'missing_completion': frozen['transport_events'].pop()
        else: frozen['transport_events'][-1]['event'] = 'cancelled'
        from benchmark_contract_live import bench
        frozen['transport_events_sha256'] = bench.digest(frozen['transport_events'])
    with pytest.raises(ValueError):
        ScheduledJournal(config(), {'test': 'integrity'}, frozen=frozen)


@pytest.mark.asyncio
async def test_a_recorded_cancellation_is_not_injected_without_actual_owner_cancellation():
    journal = ScheduledJournal(config(), {'test': 'owner'})
    entered = asyncio.Event()
    class Pending:
        async def _post_json(self, *_args, **_kwargs):
            entered.set()
            await asyncio.Event().wait()
    journal.transport = Pending()
    task = asyncio.create_task(journal.post('right', 'fake', {}, {}))
    await entered.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    journal.finish()
    replay = ScheduledJournal(config(), {'test': 'owner'}, frozen=deepcopy(journal.data), replay_wait_seconds=.02)
    from benchmark_contract_live import AdmissionError
    with pytest.raises(AdmissionError):
        await replay.post('right', 'fake', {}, {})
    with pytest.raises(ValueError):
        replay.finish()
