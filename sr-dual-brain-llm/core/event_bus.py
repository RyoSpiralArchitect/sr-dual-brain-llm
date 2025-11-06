# ============================================================================
#  SpiralReality Proprietary
#  Copyright (c) 2025 SpiralReality. All Rights Reserved.
#
#  NOTICE: This file contains confidential and proprietary information of
#  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
#  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
#  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
#
#  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
#  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
#  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
#  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ============================================================================

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


class EventBus:
    async def publish(self, ev: Dict[str, Any]):
        raise NotImplementedError

    def subscribe(self, coro: Callable[[Dict[str, Any]], Awaitable[None]]):
        raise NotImplementedError

    def attach_tracer(self, tracer):
        self._tracer = tracer


_STOP_SENTINEL: Dict[str, Any] = {"__memory_bus_stop__": True}


@dataclass(frozen=True)
class BrainWave:
    """Representation of a neural oscillation travelling through the system.

    A ``BrainWave`` encapsulates the *band* (alpha, beta, gamma, …), the
    dominant ``frequency`` in hertz, and its measured ``amplitude``.  Optional
    phase and coherence values allow higher level modules to reason about
    synchrony across regions.  Each wave is timestamped so that downstream
    consumers can build temporal relationships between bursts.
    """

    band: str
    frequency: float
    amplitude: float
    phase: float = 0.0
    coherence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.perf_counter())

    def as_event(self) -> Dict[str, Any]:
        """Project the wave into a generic event payload."""

        payload: Dict[str, Any] = {
            "type": "brainwave",
            "band": self.band,
            "frequency": self.frequency,
            "amplitude": self.amplitude,
            "phase": self.phase,
            "timestamp": self.timestamp,
        }

        if self.coherence is not None:
            payload["coherence"] = self.coherence

        if self.metadata:
            payload["metadata"] = dict(self.metadata)

        # For richer low-level consumers we embed an energy estimate – this is
        # intentionally lightweight to avoid additional dependencies.
        payload["energy"] = (self.amplitude ** 2) * self.frequency

        return payload


class _NeuralStream:
    """Async iterator over the raw event feed used for low-level inspection."""

    def __init__(self, bus: "MemoryBus", queue: asyncio.Queue[Any]):
        self._bus = bus
        self._queue = queue
        self._closed = False

    async def __aenter__(self) -> "_NeuralStream":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:
        await self.close()
        return None

    def __aiter__(self) -> "_NeuralStream":
        return self

    async def __anext__(self) -> Dict[str, Any]:
        if self._closed:
            raise StopAsyncIteration

        item = await self._queue.get()
        if item is _STOP_SENTINEL:
            await self._bus._retire_stream(self._queue, send_stop=False)
            self._closed = True
            raise StopAsyncIteration

        return item

    async def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        await self._bus._retire_stream(self._queue, send_stop=True)


class MemoryBus(EventBus):
    """In-memory async event bus used by the dual-brain runtime.

    The original implementation launched a background task in ``__init__``
    without any lifecycle controls.  In longer running processes (or during
    unit tests) this frequently left pending tasks on the event loop and would
    emit ``Task was destroyed but it is pending!`` warnings when the loop
    closed.  The revised version adds explicit shutdown semantics via
    :meth:`close`, implements an async context manager for ergonomics, and
    ensures tracers and subscribers are executed safely before the pump
    terminates.
    """

    def __init__(self):
        self._subs: List[Callable[[Dict[str, Any]], Awaitable[None]]] = []
        self._q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._tracer = None
        self._pump_task: asyncio.Task[None] | None = asyncio.create_task(self._pump())
        self._stopped = asyncio.Event()
        self._closed = False
        self._taps: List[asyncio.Queue[Any]] = []
        self._tap_lock = asyncio.Lock()

    async def __aenter__(self) -> "MemoryBus":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:
        await self.close()
        return None

    async def _pump(self) -> None:
        try:
            while True:
                ev = await self._q.get()
                if ev is _STOP_SENTINEL:
                    await self._propagate_stop()
                    break

                await self._fan_out(ev)
        except asyncio.CancelledError:
            # Cancellation is part of the normal shutdown path – we still need
            # to flush taps to ensure low-level consumers exit cleanly.
            await self._propagate_stop()
            raise
        finally:
            self._stopped.set()

    async def publish(self, ev: Dict[str, Any] | BrainWave) -> None:
        if self._closed:
            raise RuntimeError("MemoryBus is closed")

        await self._q.put(self._normalize_event(ev))

    def subscribe(self, coro: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        self._subs.append(coro)

    async def open_stream(self, *, max_buffer: int = 256) -> _NeuralStream:
        """Expose a low-level stream of raw events.

        The resulting object is both an async iterator and an async context
        manager, mirroring how neuroscientists wire electrodes into a region of
        cortex and then record continuous waveforms.
        """

        if self._closed:
            raise RuntimeError("MemoryBus is closed")

        tap_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max_buffer)
        async with self._tap_lock:
            self._taps.append(tap_queue)

        return _NeuralStream(self, tap_queue)

    async def close(self) -> None:
        if self._closed:
            # Wait for the pump to conclude in case close is called from
            # multiple places.
            await self._stopped.wait()
            return

        self._closed = True
        if self._pump_task is None:
            return

        await self._q.put(_STOP_SENTINEL)
        await self._stopped.wait()
        if not self._pump_task.done():
            self._pump_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._pump_task

    def attach_tracer(self, tracer):
        self._tracer = tracer

    async def _fan_out(self, ev: Dict[str, Any]) -> None:
        tracer = getattr(self, "_tracer", None)
        if tracer is not None:
            try:
                tracer.log_dict(ev)
            except Exception:
                # Tracer failure should not disrupt delivery.
                pass

        for sub in list(self._subs):
            try:
                await sub(ev)
            except Exception:
                # Individual subscriber errors should not prevent
                # others from receiving the event.
                pass

        taps = await self._get_taps_snapshot()
        for tap in taps:
            try:
                tap.put_nowait(ev)
            except asyncio.QueueFull:
                # Drop the oldest sample to keep the most recent energy state.
                try:
                    _ = tap.get_nowait()
                    tap.put_nowait(ev)
                except asyncio.QueueEmpty:
                    pass
                except asyncio.QueueFull:
                    # If the tap is still full we abandon the event for this
                    # consumer to avoid stalling the bus.
                    pass

    async def _get_taps_snapshot(self) -> List[asyncio.Queue[Any]]:
        async with self._tap_lock:
            return list(self._taps)

    async def _retire_stream(self, queue: asyncio.Queue[Any], *, send_stop: bool) -> None:
        if send_stop:
            try:
                queue.put_nowait(_STOP_SENTINEL)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(_STOP_SENTINEL)
                except asyncio.QueueFull:
                    pass

        async with self._tap_lock:
            try:
                self._taps.remove(queue)
            except ValueError:
                pass

    async def _propagate_stop(self) -> None:
        taps = await self._get_taps_snapshot()
        for tap in taps:
            await self._retire_stream(tap, send_stop=True)

    def _normalize_event(self, ev: Dict[str, Any] | BrainWave) -> Dict[str, Any]:
        if isinstance(ev, BrainWave):
            normalized = ev.as_event()
        elif isinstance(ev, dict):
            normalized = dict(ev)
        else:
            raise TypeError("Unsupported event payload")

        normalized.setdefault("timestamp", time.perf_counter())
        return normalized


__all__ = ["EventBus", "MemoryBus", "BrainWave"]
