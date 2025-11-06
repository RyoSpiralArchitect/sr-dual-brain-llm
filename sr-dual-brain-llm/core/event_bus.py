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
from typing import Any, Awaitable, Callable, Dict, List, Optional


class EventBus:
    async def publish(self, ev: Dict[str, Any]):
        raise NotImplementedError

    def subscribe(self, coro: Callable[[Dict[str, Any]], Awaitable[None]]):
        raise NotImplementedError

    def attach_tracer(self, tracer):
        self._tracer = tracer


_STOP_SENTINEL: Dict[str, Any] = {"__memory_bus_stop__": True}


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
                    break

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
        except asyncio.CancelledError:
            # Cancellation is part of the normal shutdown path.
            pass
        finally:
            self._stopped.set()

    async def publish(self, ev: Dict[str, Any]) -> None:
        await self._q.put(ev)

    def subscribe(self, coro: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        self._subs.append(coro)

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
