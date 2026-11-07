"""A native consultation timeout must finish provider cleanup before fallback."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
from core.callosum import Callosum
from core.right_worker import right_worker
from offline_provider_guard import forbid_provider_network
from scheduled_http_journal import ScheduledJournal
from test_benchmark_contract_live import config, response
from test_dual_brain import AlwaysConsultPolicy, make_stage_controller


class ProbeFinished(BaseException):
    pass


async def trajectory(journal):
    from test_dual_brain import TrackingTelemetry
    controller = make_stage_controller(telemetry=TrackingTelemetry())
    controller.policy = AlwaysConsultPolicy()
    controller.callosum = bus = Callosum(slot_ms=0)
    started = asyncio.Event()
    class Right:
        async def criticise_reasoning(self, *args, **kwargs):
            started.set()
            return await journal.post('critic', 'fake', {'kind': 'critic'}, {})
    worker = asyncio.create_task(right_worker(bus, None, Right()))
    async def consult(**kwargs):
        kwargs['decision'].slot_ms = 0
        with pytest.raises(asyncio.TimeoutError):
            await kwargs['ask_callosum_with_timeout']({'qid': 'same', 'type': 'ASK_CRITIC',
                'question': 'Explain bounded queues.', 'draft': 'draft'}, timeout_ms=30)
        assert started.is_set()
        assert bus.snapshot()['running'] == 0
        await journal.post('critic', 'fake', {'kind': 'fallback'}, {})
        raise ProbeFinished()
    controller._run_right_consult_stage = consult
    try:
        with pytest.raises(ProbeFinished):
            await controller.process('Explain bounded queues under burst traffic.', leading_brain='left',
                system2_mode='off', executive_mode='off', executive_observer_mode='off')
    finally:
        await bus.close()
        await worker
    journal.finish()
    assert bus.snapshot()['occupied'] == 0


@pytest.mark.asyncio
async def test_native_timeout_cancel_cleanup_then_fallback_preserves_exact_transport_order():
    class Provider:
        async def _post_json(self, _url, payload, _headers):
            if payload['kind'] == 'critic':
                try:
                    await asyncio.Event().wait()
                finally:
                    # Real clients and the scheduled replayer can await cleanup.
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
            return response('fallback')
    live = ScheduledJournal(config(), {'test': 'native-cancel-fallback'}, max_calls=4)
    live.transport = Provider()
    with forbid_provider_network() as audit:
        await trajectory(live)
        frozen = json.loads(json.dumps(live.data))
        replay = ScheduledJournal(config(), frozen['metadata'], frozen=frozen,
                                  max_calls=4, replay_wait_seconds=.5)
        await trajectory(replay)
    assert audit.blocked_requests == replay.calls == 0 and replay.transport is None
    assert replay.events == live.events
    assert live.data['transport_events'] == replay.data['transport_events'] == [
        {'event': 'start', 'occurrence': 0}, {'event': 'cancelled', 'occurrence': 0},
        {'event': 'start', 'occurrence': 1}, {'event': 'response', 'occurrence': 1}]
    assert live.data['usage']['unknown_usage_calls'] == 1
