import asyncio

import pytest

from core.event_bus import MemoryBus


class _Recorder:
    def __init__(self):
        self.events = []

    def log_dict(self, ev):
        self.events.append(ev)


@pytest.mark.asyncio
async def test_memory_bus_delivers_events_and_traces():
    bus = MemoryBus()
    processed = asyncio.Event()
    received: list[dict] = []

    async def handler(ev):
        received.append(ev)
        processed.set()

    recorder = _Recorder()
    bus.attach_tracer(recorder)
    bus.subscribe(handler)

    payload = {"type": "ping", "value": 1}
    await bus.publish(payload)
    await asyncio.wait_for(processed.wait(), timeout=0.5)

    assert received == [payload]
    assert recorder.events == [payload]

    await bus.close()


@pytest.mark.asyncio
async def test_memory_bus_context_manager_closes_automatically():
    processed = asyncio.Event()

    async with MemoryBus() as bus:
        async def handler(ev):
            processed.set()

        bus.subscribe(handler)
        await bus.publish({"type": "lifecycle"})
        await asyncio.wait_for(processed.wait(), timeout=0.5)

    # Exiting the context should have closed the bus so publishing again should fail.
    # We verify by ensuring the background task has stopped.
    assert bus._stopped.is_set()


@pytest.mark.asyncio
async def test_memory_bus_close_is_idempotent():
    bus = MemoryBus()
    await bus.close()
    await bus.close()
    assert bus._closed
