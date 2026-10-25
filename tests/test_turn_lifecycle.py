"""Model-free RPC lifetime and serialization tests, also run on Python 3.10."""
import asyncio
import time

import pytest

from core.turn_lifecycle import (
    TurnDispatcher, begin_turn_commit, cancellation_barrier, create_turn_task,
)


async def until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(poll(), 2.)


def request(rid, **extra):
    return {"id": rid, "method": "process", "params": {"qid": "same"}, **extra}


@pytest.mark.asyncio
async def test_cancel_drains_children_and_keeps_next_turn_serial():
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls, responses = [], []

    async def child():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    async def handle(req):
        calls.append(req["id"])
        if req["id"] == "first":
            # Cancellation propagates through an awaited child, too. Repeated
            # signals must not cancel the child's async finally a second time.
            await create_turn_task(child())
        return {"id": req["id"], "ok": True}

    dispatcher = TurnDispatcher(handle, responses.append, capacity=2)
    try:
        dispatcher.submit(request("first"))
        await started.wait()
        dispatcher.submit(request("second"))
        assert dispatcher.cancel("first")["accepted"]
        await cleaning.wait()
        dispatcher.cancel("first")
        await asyncio.sleep(.01)
        assert calls == ["first"] and not responses
        assert dispatcher.snapshot()["pending"] == 2
        release.set()
        await until(lambda: len(responses) == 2)
        assert calls == ["first", "second"]
        assert responses[0]["error"] == {"type": "Cancelled", "message": "Turn abandoned", "may_have_committed": False}
        assert responses[1]["ok"]
    finally:
        release.set()
        await dispatcher.close()


@pytest.mark.asyncio
async def test_deadline_queue_cancellation_and_overload_never_start_abandoned_work():
    started, release = asyncio.Event(), asyncio.Event()
    calls, responses = [], []

    async def handle(req):
        calls.append(req["id"])
        started.set()
        await release.wait()
        return {"id": req["id"], "ok": True}

    dispatcher = TurnDispatcher(handle, responses.append, capacity=2)
    try:
        dispatcher.submit(request("running"))
        await started.wait()
        dispatcher.submit(request("expired", deadline_unix_ms=(time.time()+.03)*1000))
        dispatcher.submit(request("overflow"))
        await until(lambda: len(responses) == 2)
        assert {r["error"]["type"] for r in responses} == {"Overloaded", "DeadlineExceeded"}
        dispatcher.submit(request("queued"))
        assert dispatcher.cancel("queued")["accepted"]
        assert not dispatcher.cancel("unknown")["accepted"]
        release.set()
        await until(lambda: len(responses) == 4)
        assert calls == ["running"]
    finally:
        release.set()
        await dispatcher.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("protect", ["commit", "recovery"])
async def test_cancellation_waits_for_a_protected_write_and_does_not_start_more_generation(protect):
    started, release = asyncio.Event(), asyncio.Event()
    calls, responses = [], []

    async def write():
        started.set()
        await release.wait()
        calls.append("write-finished")

    async def handle(req):
        if req["id"] == "first":
            if protect == "commit":
                begin_turn_commit()
                await write()
                # Idempotent: a completed controller may reach a second boundary.
                begin_turn_commit()
            else:
                async with cancellation_barrier():
                    await write()
            calls.append("post-write-generation")
        calls.append(req["id"])
        return {"id": req["id"], "ok": True}

    dispatcher = TurnDispatcher(handle, responses.append)
    try:
        dispatcher.submit(request("first"))
        await started.wait()
        dispatcher.submit(request("second"))
        assert dispatcher.cancel("first") == {"accepted": True, "deferred": True}
        await asyncio.sleep(.01)
        assert not calls and not responses
        release.set()
        await until(lambda: len(responses) == 2)
        assert calls[0] == "write-finished" and calls[-1] == "second"
        assert ("post-write-generation" in calls) == (protect == "commit")
        assert responses[0]["error"]["may_have_committed"]
    finally:
        release.set()
        await dispatcher.close()


@pytest.mark.asyncio
async def test_commit_checks_elapsed_deadline_even_before_timer_callback_runs():
    responses, writes = [], []

    async def handle(req):
        # Simulate synchronous CPU work starving the timer callback.
        time.sleep(.025)
        begin_turn_commit()
        writes.append(True)
        return {"id": req["id"], "ok": True}

    dispatcher = TurnDispatcher(handle, responses.append, timeout_seconds=.01)
    try:
        dispatcher.submit(request("late"))
        await until(lambda: bool(responses))
        assert not writes
        assert responses[0]["error"]["type"] == "DeadlineExceeded"
    finally:
        await dispatcher.close()


@pytest.mark.asyncio
async def test_duplicate_inflight_id_abandons_original_without_a_second_execution():
    started, responses, calls = asyncio.Event(), [], []

    async def handle(req):
        calls.append(req["id"])
        started.set()
        await asyncio.Event().wait()

    dispatcher = TurnDispatcher(handle, responses.append)
    try:
        dispatcher.submit(request("duplicate"))
        await started.wait()
        dispatcher.submit(request("duplicate"))
        await until(lambda: bool(responses))
        assert calls == ["duplicate"] and len(responses) == 1
        assert responses[0]["error"]["type"] == "DuplicateRequestId"
    finally:
        await dispatcher.close()


@pytest.mark.asyncio
async def test_close_abandons_queue_but_drains_a_commit_before_returning():
    started, release, responses, calls = asyncio.Event(), asyncio.Event(), [], []

    async def handle(req):
        calls.append(req["id"])
        begin_turn_commit()
        started.set()
        await release.wait()
        return {"id": req["id"], "ok": True}

    dispatcher = TurnDispatcher(handle, responses.append)
    dispatcher.submit(request("committing"))
    await started.wait()
    dispatcher.submit(request("queued"))
    close = asyncio.create_task(dispatcher.close())
    try:
        await asyncio.sleep(.01)
        assert not close.done() and calls == ["committing"]
        release.set()
        await close
        assert len(responses) == 2 and dispatcher.snapshot()["pending"] == 0
    finally:
        release.set()
        await close


@pytest.mark.asyncio
async def test_invalid_deadline_and_already_expired_work_are_not_admitted():
    responses = []
    async def handle(req):
        pytest.fail("invalid work started")
    dispatcher = TurnDispatcher(handle, responses.append)
    try:
        for invalid in (float("nan"), float("inf"), "bad"):
            with pytest.raises(ValueError):
                dispatcher.submit(request("bad", deadline_unix_ms=invalid))
        dispatcher.submit(request("old", deadline_unix_ms=0))
        assert responses[0]["error"]["type"] == "DeadlineExceeded"
        assert dispatcher.snapshot()["pending"] == 0
    finally:
        await dispatcher.close()
