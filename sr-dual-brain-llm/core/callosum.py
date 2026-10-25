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
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import math
import uuid
from typing import Any, Awaitable, Callable, Dict


class CallosumClosed(RuntimeError):
    """The session's consultation channel has shut down."""


class CallosumOverloaded(RuntimeError):
    """All bounded consultation slots are occupied; no work was admitted."""


class RequestAbandoned(Exception):
    """Normal worker control flow when a caller no longer needs its result."""


@dataclass
class _Pending:
    qid: str
    deadline: float
    future: asyncio.Future
    operation: asyncio.Future | None = None
    claimed: bool = False


class Callosum:
    """Bounded, request-correlated consultations on a single asyncio loop.

    qid identifies a dialogue turn. A fresh request_id identifies each attempt.
    Timeout covers slotting, queue residence and execution together. Workers must
    echo request_id and use run_request to propagate caller cancellation.
    """
    def __init__(self, slot_ms: int = 250, *, max_pending: int = 64):
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        self.slot_ms = slot_ms
        self.max_pending = max_pending
        self._pending: Dict[str, _Pending] = {}
        self._queued: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._available = asyncio.Event()
        self._running: set[asyncio.Future] = set()
        self._closed = False
        self._rejected = self._expired = self._cancelled = self._discarded = 0

    @staticmethod
    def deadline_for(timeout_ms: int, deadline: float | None = None) -> float:
        loop = asyncio.get_running_loop()
        timeout = float(timeout_ms) / 1000.
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive and finite")
        end = loop.time() + timeout
        if deadline is not None:
            if not math.isfinite(deadline):
                raise ValueError("Deadline must be finite loop time")
            end = min(end, deadline)
        if end <= loop.time():
            raise TimeoutError("Consultation deadline elapsed before admission")
        return end

    async def ask_detail(self, payload: Dict[str, Any], timeout_ms: int = 3000,
                         *, deadline: float | None = None) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        end = self.deadline_for(timeout_ms, deadline)
        slot = float(self.slot_ms) / 1000.
        if not math.isfinite(slot) or slot < 0:
            raise ValueError("Slot must be nonnegative and finite")
        if self._closed:
            raise CallosumClosed("Consultation channel is closed")
        if self._occupied() >= self.max_pending:
            self._rejected += 1
            raise CallosumOverloaded("Consultation capacity reached")
        packet = deepcopy(payload)
        qid = packet.get("qid") or str(uuid.uuid4())
        if not isinstance(qid, str):
            raise ValueError("qid must be a string")
        request_id = str(uuid.uuid4())
        packet.update(qid=qid, request_id=request_id, deadline_at=end)
        pending = _Pending(qid, end, loop.create_future())
        self._pending[request_id] = pending
        try:
            async with asyncio.timeout_at(end):
                # Delay dispatch, not just the caller's receipt. Closing the bus
                # can interrupt this slot without waiting for its timer.
                if slot:
                    try:
                        return await asyncio.wait_for(asyncio.shield(pending.future), slot)
                    except TimeoutError:
                        pass
                self._queued[request_id] = packet
                self._available.set()
                return await pending.future
        except TimeoutError:
            self._expired += 1
            raise
        except asyncio.CancelledError:
            self._cancelled += 1
            raise
        finally:
            self._pending.pop(request_id, None)
            self._queued.pop(request_id, None)
            if not self._queued and not self._closed:
                self._available.clear()
            if not pending.future.done():
                pending.future.cancel()
            elif not pending.future.cancelled():
                pending.future.exception()  # consume shutdown exceptions on cancellation races
            if pending.operation is not None:
                self._cancel_operation(pending.operation)

    @staticmethod
    def _cancel_operation(task: asyncio.Future) -> None:
        # A second cancellation can interrupt a client's asynchronous cleanup.
        if not task.done() and (not isinstance(task, asyncio.Task) or not task.cancelling()):
            task.cancel()

    def _active(self, request_id: str) -> _Pending | None:
        pending = self._pending.get(request_id)
        if (self._closed or pending is None or pending.future.done()
                or pending.deadline <= asyncio.get_running_loop().time()):
            return None
        return pending

    def _occupied(self) -> int:
        attached = {p.operation for p in self._pending.values() if p.operation is not None}
        draining = sum(not task.done() and task not in attached for task in self._running)
        # An abandoned client must not free physical capacity before its model
        # has acknowledged cancellation, including with multiple consumers.
        return len(self._pending) + draining

    async def publish_response(self, request_id: str, response: Dict[str, Any]) -> bool:
        """Accept only this live attempt; qid alone can never select a receiver."""
        pending = self._active(request_id)
        if pending is None:
            self._discarded += 1
            return False
        pending.future.set_result({**response, "qid": pending.qid, "request_id": request_id})
        return True

    async def recv_request(self) -> Dict[str, Any]:
        while True:
            if self._closed:
                raise CallosumClosed("Consultation channel is closed")
            while self._queued:
                request_id, packet = self._queued.popitem(last=False)
                if self._active(request_id) is not None:
                    return packet
            self._available.clear()
            await self._available.wait()

    async def run_request(self, request: Dict[str, Any], operation: Callable[[], Awaitable[Any]]) -> Any:
        """Run one worker operation with the caller's lifetime and deadline."""
        request_id = request["request_id"]
        pending = self._active(request_id)
        if pending is None:
            raise RequestAbandoned()
        if pending.claimed:
            raise RuntimeError("Consultation request already claimed")
        pending.claimed = True
        async def execute():
            # Recheck at actual execution, not just when the task was scheduled.
            # Do not even invoke a provider factory for an abandoned request.
            if self._active(request_id) is None:
                raise RequestAbandoned()
            return await operation()
        task = asyncio.create_task(execute())
        pending.operation = task
        self._running.add(task)
        task.add_done_callback(self._operation_done)
        try:
            # ask_detail owns the single deadline timer. Shielding prevents a
            # worker shutdown from injecting a second cancellation into client
            # cleanup already started by caller expiry/close.
            result = await asyncio.shield(task)
            if self._active(request_id) is None:
                raise RequestAbandoned()
            return result
        except asyncio.CancelledError:
            # Per-request cancellation must not kill the worker pump. An actual
            # cancellation of the worker itself must still propagate.
            if asyncio.current_task().cancelling():
                raise
            raise RequestAbandoned() from None
        except TimeoutError:
            if self._active(request_id) is None:
                raise RequestAbandoned() from None
            raise
        finally:
            self._cancel_operation(task)

    def _operation_done(self, task: asyncio.Future) -> None:
        self._running.discard(task)
        if not task.cancelled():
            # The worker may itself have exited while client cleanup was still
            # running. Retrieve late exceptions instead of logging private text
            # as an unhandled task failure. Active awaiters still receive them.
            task.exception()

    async def close(self) -> None:
        self._closed = True
        self._queued.clear()
        self._available.set()
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_exception(CallosumClosed("Consultation channel closed"))
        running = list(self._running)
        for task in running:
            self._cancel_operation(task)
        if running:
            await asyncio.shield(asyncio.gather(*running, return_exceptions=True))

    def snapshot(self) -> Dict[str, Any]:
        """Payload-free lifecycle counters for health checks and diagnostics."""
        return {"closed": self._closed, "capacity": self.max_pending,
                "occupied": self._occupied(),
                "pending": len(self._pending), "queued": len(self._queued),
                "running": sum(not task.done() for task in self._running),
                "overloaded": self._rejected, "expired": self._expired,
                "cancelled": self._cancelled, "discarded_responses": self._discarded}
