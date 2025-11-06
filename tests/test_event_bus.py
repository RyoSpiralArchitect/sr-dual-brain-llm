import asyncio

import pytest

from core.event_bus import BrainWave, MemoryBus


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

    assert received[0]["type"] == payload["type"]
    assert received[0]["value"] == payload["value"]
    assert "timestamp" in received[0]
    assert recorder.events[0]["type"] == payload["type"]

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


@pytest.mark.asyncio
async def test_memory_bus_normalizes_brainwave_events():
    bus = MemoryBus()
    processed = asyncio.Event()
    received: list[dict] = []

    async def handler(ev):
        received.append(ev)
        processed.set()

    bus.subscribe(handler)

    wave = BrainWave(
        band="gamma",
        frequency=42.0,
        amplitude=0.8,
        phase=0.25,
        coherence=0.9,
        metadata={"region": "prefrontal"},
    )

    await bus.publish(wave)
    await asyncio.wait_for(processed.wait(), timeout=0.5)

    assert received[0]["type"] == "brainwave"
    assert received[0]["band"] == "gamma"
    assert received[0]["metadata"] == {"region": "prefrontal"}
    assert received[0]["energy"] == pytest.approx((wave.amplitude ** 2) * wave.frequency)

    await bus.close()


@pytest.mark.asyncio
async def test_memory_bus_stream_exposes_low_level_feed():
    bus = MemoryBus()

    stream = await bus.open_stream(max_buffer=1)

    async with stream:
        consumer = asyncio.create_task(stream.__anext__())

        await bus.publish({"type": "spike", "potential": 0.2})
        event = await asyncio.wait_for(consumer, timeout=0.5)

    assert event["type"] == "spike"
    assert "timestamp" in event

    await bus.close()
