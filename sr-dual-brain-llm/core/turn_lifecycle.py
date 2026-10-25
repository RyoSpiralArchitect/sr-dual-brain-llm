"""Serialized RPC work with a responsive cancellation/control plane."""
import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
import math
import time
from weakref import WeakSet


_current = ContextVar("active_rpc_turn", default=None)
_cancelling = WeakSet()


def cancel_task(task):
    if task is not None and not task.done() and task not in _cancelling:
        if not getattr(task, "cancelling", lambda: 0)():
            task.cancel()
        _cancelling.add(task)


class _TurnTask(asyncio.Task):
    # Track cancellation propagated by an awaiting parent too (Python 3.10 has
    # no public Task.cancelling()). Repeated signals must not interrupt cleanup.
    def cancel(self, *args, **kwargs):
        if self in _cancelling:
            return False
        _cancelling.add(self)
        return super().cancel(*args, **kwargs)


def create_turn_task(coroutine):
    task = _TurnTask(coroutine, loop=asyncio.get_running_loop())
    state = _current.get()
    if state is not None:
        state.children.add(task)
        def finished(child):
            state.children.discard(child)
            if not child.cancelled():
                child.exception()
        task.add_done_callback(finished)
    return task


@contextmanager
def detached_turn():
    token = _current.set(None)
    try:
        yield
    finally:
        _current.reset(token)


def begin_turn_commit():
    """Latch protection before the first factual write; no-op outside RPC."""
    state = _current.get()
    if state is not None:
        if state.committing:
            return
        if state.deadline <= asyncio.get_running_loop().time():
            state.reason = state.reason or "DeadlineExceeded"
        if state.reason:
            raise asyncio.CancelledError()
        state.committing = True
        state.may_have_committed = True


@asynccontextmanager
async def cancellation_barrier(*, may_have_committed=True):
    """Drain recovery/reset operations before acknowledging abandonment."""
    state = _current.get()
    if state is not None:
        if state.deadline <= asyncio.get_running_loop().time():
            state.reason = state.reason or "DeadlineExceeded"
        if state.reason:
            raise asyncio.CancelledError()
        state.protected += 1
        state.may_have_committed |= may_have_committed
    try:
        yield
    finally:
        if state is not None:
            state.protected -= 1
            if state.deadline <= asyncio.get_running_loop().time():
                state.reason = state.reason or "DeadlineExceeded"
    if state is not None and state.reason and not state.committing and not state.protected:
        raise asyncio.CancelledError()


def turn_committing():
    state = _current.get()
    return bool(state is not None and state.committing)


def turn_abandoned():
    state = _current.get()
    return bool(state is not None and state.reason)


async def drain_turn_tasks():
    state = _current.get()
    if state is None:
        return
    # From here even a first cancellation must not interrupt owned cleanup.
    state.finishing = True
    while state.children:
        children = list(state.children)
        for child in children:
            cancel_task(child)
        await asyncio.gather(*children, return_exceptions=True)


@dataclass
class _Request:
    envelope: dict
    deadline: float
    task: asyncio.Task | None = None
    timer: object = None
    reason: str | None = None
    protected: int = 0
    committing: bool = False
    may_have_committed: bool = False
    children: set = field(default_factory=set)
    finishing: bool = False


class TurnDispatcher:
    """Keep stateful work serial while the reader processes control messages.

    No session concurrency is introduced. Cancellation keeps its slot until the
    handler and registered children have drained, including protected commits.
    """
    def __init__(self, handler, send, *, capacity=64, timeout_seconds=120.):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("RPC capacity must be a positive integer")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("RPC timeout must be positive and finite")
        self.handler, self.send = handler, send
        self.capacity, self.timeout = capacity, timeout_seconds
        self.requests, self.queued = {}, OrderedDict()
        self.available = asyncio.Event()
        self.closed = False
        self.worker = asyncio.create_task(self._pump())

    def submit(self, envelope):
        rid = str(envelope.get("id") or "")
        if not rid:
            self.send(self._error(rid, "InvalidRequest", "A unique RPC id is required"))
            return
        if rid in self.requests:
            self.cancel(rid, "DuplicateRequestId")
            return
        if self.closed or len(self.requests) >= self.capacity:
            self.send(self._error(rid, "Overloaded", "RPC capacity unavailable"))
            return
        remaining = self.timeout
        if "deadline_unix_ms" in envelope:
            end = float(envelope["deadline_unix_ms"])
            if not math.isfinite(end):
                raise ValueError("RPC deadline must be finite")
            remaining = min(remaining, end/1000.-time.time())
        if remaining <= 0:
            self.send(self._error(rid, "DeadlineExceeded", "Turn deadline elapsed before admission"))
            return
        state = _Request(deepcopy(envelope), asyncio.get_running_loop().time()+remaining)
        self.requests[rid] = state
        self.queued[rid] = state
        state.timer = asyncio.get_running_loop().call_at(state.deadline, self.cancel, rid, "DeadlineExceeded")
        self.available.set()

    @staticmethod
    def _error(rid, kind, message, **extra):
        return {"id":rid, "ok":False, "error":{"type":kind,"message":message,**extra}}

    def cancel(self, rid, reason="Cancelled"):
        state = self.requests.get(rid)
        if state is None:
            return {"accepted":False,"deferred":False}
        state.reason = state.reason or reason
        deferred = bool(state.committing or state.protected or state.finishing)
        if state.task is None:
            self.queued.pop(rid, None)
            self._finish(rid, state, None)
        elif not deferred:
            cancel_task(state.task)
        return {"accepted":True,"deferred":deferred}

    def _finish(self, rid, state, response):
        state.timer.cancel()
        self.requests.pop(rid, None)
        if state.reason:
            response = self._error(rid, state.reason, "Turn abandoned", may_have_committed=state.may_have_committed)
        self.send(response)

    async def _execute(self, rid, state):
        token = _current.set(state)
        try:
            if state.deadline <= asyncio.get_running_loop().time():
                state.reason = "DeadlineExceeded"
                return None
            return await self.handler(state.envelope)
        finally:
            try:
                await drain_turn_tasks()
            finally:
                _current.reset(token)

    async def _pump(self):
        while True:
            while self.queued:
                rid, state = self.queued.popitem(last=False)
                state.task = _TurnTask(self._execute(rid, state), loop=asyncio.get_running_loop())
                try:
                    response = await state.task
                except asyncio.CancelledError:
                    state.reason = state.reason or "Cancelled"
                    response = None
                except Exception as exc:
                    response = self._error(rid, type(exc).__name__, "RPC execution failed")
                self._finish(rid, state, response)
            if self.closed:
                return
            self.available.clear()
            await self.available.wait()

    def snapshot(self):
        return {"capacity":self.capacity,"pending":len(self.requests),"queued":len(self.queued),
                "draining":sum(bool(s.reason) and s.task is not None for s in self.requests.values())}

    async def close(self):
        self.closed = True
        for rid in list(self.requests):
            self.cancel(rid)
        self.available.set()
        await asyncio.shield(self.worker)
