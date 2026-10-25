import asyncio
from asyncio import TimeoutError
import json

import pytest

from core.callosum import Callosum, CallosumClosed, CallosumOverloaded
from core.right_worker import right_worker
from core.enhanced_callosum import EnhancedCallosum


@pytest.mark.asyncio
async def test_same_turn_attempts_route_independently_and_snapshot_the_payload():
    bus = Callosum(slot_ms=0)
    payload = {"qid":"turn", "request_id":"caller-cannot-reuse-this", "context":{"values":[1]}}
    first = asyncio.create_task(bus.ask_detail(payload))
    old = await bus.recv_request()
    payload["context"]["values"].append(2)
    assert old["context"] == {"values":[1]} and payload["request_id"] == "caller-cannot-reuse-this"
    second = asyncio.create_task(bus.ask_detail({"qid":"turn"}))
    new = await bus.recv_request()
    assert old["request_id"] != new["request_id"]
    assert not await bus.publish_response("turn", {"value":"unscoped"})
    assert await bus.publish_response(new["request_id"], {"value":"new", "qid":"wrong"})
    assert await bus.publish_response(old["request_id"], {"value":"old"})
    assert (await second)["value"] == "new" and (await first)["value"] == "old"
    assert (await second)["qid"] == "turn"
    assert not await bus.publish_response(new["request_id"], {"value":"duplicate"})
    assert bus.snapshot()["pending"] == bus.snapshot()["queued"] == 0


@pytest.mark.asyncio
async def test_timeout_removes_queued_work_and_releases_admission_capacity():
    bus = Callosum(slot_ms=0, max_pending=1)
    with pytest.raises(TimeoutError):
        await bus.ask_detail({"qid":"expired"}, timeout_ms=20)
    assert bus.snapshot()["queued"] == bus.snapshot()["pending"] == 0
    assert bus.snapshot()["expired"] == 1
    ask = asyncio.create_task(bus.ask_detail({"qid":"fresh"}))
    request = await bus.recv_request()
    assert request["qid"] == "fresh"
    await bus.publish_response(request["request_id"], {})
    await ask


@pytest.mark.asyncio
async def test_late_response_cannot_complete_a_reused_qid():
    bus = Callosum(slot_ms=0)
    first = asyncio.create_task(bus.ask_detail({"qid":"same"}, timeout_ms=30))
    old = await bus.recv_request()
    with pytest.raises(TimeoutError):
        await first
    second = asyncio.create_task(bus.ask_detail({"qid":"same"}))
    new = await bus.recv_request()
    assert not await bus.publish_response(old["request_id"], {"value":"stale"})
    assert not second.done()
    await bus.publish_response(new["request_id"], {"value":"current"})
    assert (await second)["value"] == "current"


@pytest.mark.asyncio
@pytest.mark.parametrize("slot", [0, 1000])
async def test_cancellation_during_slot_or_queue_wait_leaves_no_pending_work(slot):
    bus = Callosum(slot_ms=slot, max_pending=1)
    ask = asyncio.create_task(bus.ask_detail({"qid":"cancel"}))
    await asyncio.sleep(0)
    assert bus.snapshot()["pending"] == 1
    if slot == 0:
        await bus._available.wait()
        assert bus.snapshot()["queued"] == 1
    ask.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ask
    assert bus.snapshot()["pending"] == bus.snapshot()["queued"] == 0
    assert bus.snapshot()["cancelled"] == 1


@pytest.mark.asyncio
async def test_slot_and_inherited_deadline_share_one_budget_before_dispatch():
    bus = Callosum(slot_ms=1000)
    with pytest.raises(TimeoutError):
        await bus.ask_detail({}, timeout_ms=20)
    with pytest.raises(TimeoutError):
        await bus.ask_detail({}, timeout_ms=5000, deadline=asyncio.get_running_loop().time()+.02)
    assert bus.snapshot()["queued"] == bus.snapshot()["pending"] == 0
    with pytest.raises(TimeoutError):
        await bus.ask_detail({}, deadline=0.)
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            await bus.ask_detail({}, timeout_ms=value)
    with pytest.raises(ValueError):
        await bus.ask_detail({}, deadline=float("nan"))


@pytest.mark.asyncio
async def test_overload_is_immediate_and_does_not_admit_another_job():
    bus = Callosum(slot_ms=0, max_pending=1)
    ask = asyncio.create_task(bus.ask_detail({"qid":"active"}))
    request = await bus.recv_request()
    with pytest.raises(CallosumOverloaded):
        await bus.ask_detail({"qid":"excess"})
    assert bus.snapshot()["pending"] == 1 and bus.snapshot()["overloaded"] == 1
    await bus.publish_response(request["request_id"], {})
    await ask
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            Callosum(max_pending=invalid)


class BlockingRight:
    def __init__(self):
        self.started, self.cancelled = asyncio.Event(), asyncio.Event()
        self.block = True

    async def result(self):
        if self.block:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
        return {"notes_sum":"fresh", "confidence_r":.8, "verdict":"ok", "issues":[]}

    async def deepen(self, *args, **kwargs):
        return await self.result()

    async def criticise_reasoning(self, *args, **kwargs):
        return await self.result()

    async def generate_lead(self, *args, **kwargs):
        await self.result()
        return "fresh"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["ASK_DETAIL", "ASK_CRITIC", "ASK_LEAD"])
@pytest.mark.parametrize("abandon", ["cancel", "timeout"])
async def test_caller_abandonment_cancels_active_model_and_worker_handles_the_next_request(kind, abandon):
    bus, model = Callosum(slot_ms=0), BlockingRight()
    worker = asyncio.create_task(right_worker(bus, None, model))
    ask = asyncio.create_task(bus.ask_detail({"qid":"turn", "type":kind, "question":"first"}, timeout_ms=150))
    try:
        await asyncio.wait_for(model.started.wait(), 1.)
        if abandon == "cancel":
            ask.cancel()
        with pytest.raises(asyncio.CancelledError if abandon=="cancel" else TimeoutError):
            await ask
        await asyncio.wait_for(model.cancelled.wait(), 1.)
        model.block = False
        result = await bus.ask_detail({"qid":"turn", "type":kind, "question":"next"}, timeout_ms=1000)
        assert result["qid"] == "turn" and not worker.done()
        assert bus.snapshot()["pending"] == bus.snapshot()["queued"] == 0
    finally:
        await bus.close()
        await worker


@pytest.mark.asyncio
async def test_worker_cancellation_is_not_mistaken_for_abandoning_one_request():
    bus, model = Callosum(slot_ms=0), BlockingRight()
    worker = asyncio.create_task(right_worker(bus, None, model))
    ask = asyncio.create_task(bus.ask_detail({"type":"ASK_DETAIL", "question":"block"}))
    await model.started.wait()
    worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await worker
    await model.cancelled.wait()
    ask.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ask
    assert bus.snapshot()["pending"] == 0


@pytest.mark.asyncio
async def test_client_returning_a_late_result_after_cancellation_cannot_contaminate_next_attempt():
    class LateRight(BlockingRight):
        async def result(self):
            if self.block:
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    return {"notes_sum":"STALE", "confidence_r":.9}
            return {"notes_sum":"fresh", "confidence_r":.8}
    bus, model = Callosum(slot_ms=0), LateRight()
    worker = asyncio.create_task(right_worker(bus, None, model))
    ask = asyncio.create_task(bus.ask_detail({"qid":"reuse", "type":"ASK_DETAIL", "question":"first"}))
    try:
        await model.started.wait()
        ask.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ask
        await model.cancelled.wait()
        model.block = False
        result = await bus.ask_detail({"qid":"reuse", "type":"ASK_DETAIL", "question":"next"})
        assert result["notes_sum"] == "fresh"
    finally:
        await bus.close()
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize("slot", [0, 1000])
async def test_close_wakes_callers_and_receivers_and_rejects_new_work(slot):
    bus = Callosum(slot_ms=slot)
    ask = asyncio.create_task(bus.ask_detail({"qid":"close"}))
    await asyncio.sleep(0)
    await bus.close()
    with pytest.raises(CallosumClosed):
        await ask
    with pytest.raises(CallosumClosed):
        await bus.recv_request()
    with pytest.raises(CallosumClosed):
        await bus.ask_detail({})
    await bus.close()
    assert bus.snapshot()["pending"] == bus.snapshot()["queued"] == 0


@pytest.mark.asyncio
async def test_close_unblocks_an_idle_receiver_and_an_active_model():
    idle = Callosum(slot_ms=0)
    receiver = asyncio.create_task(idle.recv_request())
    await asyncio.sleep(0)
    await idle.close()
    with pytest.raises(CallosumClosed):
        await receiver
    bus, model = Callosum(slot_ms=0), BlockingRight()
    worker = asyncio.create_task(right_worker(bus, None, model))
    ask = asyncio.create_task(bus.ask_detail({"type":"ASK_DETAIL", "question":"first"}))
    await model.started.wait()
    await bus.close()
    with pytest.raises(CallosumClosed):
        await ask
    await worker
    assert model.cancelled.is_set() and bus.snapshot()["running"] == 0


@pytest.mark.asyncio
async def test_worker_errors_are_sanitized_and_unsupported_requests_fail_without_waiting_for_timeout():
    class Broken:
        async def deepen(self, *args, **kwargs):
            raise RuntimeError("SECRET source text")
    bus = Callosum(slot_ms=0)
    worker = asyncio.create_task(right_worker(bus, None, Broken()))
    try:
        result = await bus.ask_detail({"type":"ASK_DETAIL", "question":"bad"})
        assert result["error"] == "RuntimeError" and "SECRET" not in json.dumps(result)
        result = await bus.ask_detail({"type":"UNKNOWN"})
        assert result["error"] == "UnsupportedRequestType"
    finally:
        await bus.close()
        await worker


@pytest.mark.asyncio
async def test_enhanced_wrapper_forwards_slot_lifetime_and_shutdown_without_filtering_control():
    bus = EnhancedCallosum(base_callosum=Callosum(slot_ms=0, max_pending=1), enable_filtering=False)
    bus.slot_ms = 10
    assert bus.base_callosum.slot_ms == 10
    model = BlockingRight()
    worker = asyncio.create_task(right_worker(bus, None, model))
    ask = asyncio.create_task(bus.ask_detail({"type":"ASK_DETAIL", "question":"first"}))
    await model.started.wait()
    await bus.close()
    with pytest.raises(CallosumClosed):
        await ask
    await worker
    assert model.cancelled.is_set() and bus.snapshot()["closed"]


@pytest.mark.asyncio
async def test_filtered_requests_still_respect_shutdown_and_the_inherited_deadline():
    bus = EnhancedCallosum()
    kwargs = dict(priority=.1, novelty=.05, task_relevance=.1)
    with pytest.raises(TimeoutError):
        await bus.ask_detail({"content":"noise"}, deadline=0., **kwargs)
    await bus.close()
    with pytest.raises(CallosumClosed):
        await bus.ask_detail({"content":"noise"}, **kwargs)


@pytest.mark.asyncio
async def test_an_operation_cannot_claim_the_same_transport_attempt_twice():
    bus = Callosum(slot_ms=0)
    ask = asyncio.create_task(bus.ask_detail({"qid":"once"}))
    request = await bus.recv_request()
    started, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def operation():
        calls.append(True)
        started.set()
        await release.wait()
        return {"done":True}
    running = asyncio.create_task(bus.run_request(request, operation))
    await started.wait()
    with pytest.raises(RuntimeError, match="already claimed"):
        await bus.run_request(request, operation)
    release.set()
    await bus.publish_response(request["request_id"], await running)
    assert (await ask)["done"] and len(calls)==1


@pytest.mark.asyncio
async def test_cancelled_model_holds_capacity_until_it_actually_stops():
    class SlowStop(BlockingRight):
        def __init__(self):
            super().__init__()
            self.release, self.drained = asyncio.Event(), asyncio.Event()
        async def result(self):
            if self.block:
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    await self.release.wait()
                finally:
                    self.drained.set()
            return {"notes_sum":"fresh", "confidence_r":.8}
    bus, model = Callosum(slot_ms=0, max_pending=1), SlowStop()
    worker = asyncio.create_task(right_worker(bus, None, model))
    ask = asyncio.create_task(bus.ask_detail({"type":"ASK_DETAIL", "question":"first"}))
    try:
        await model.started.wait()
        ask.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ask
        await model.cancelled.wait()
        assert bus.snapshot()["pending"] == 0 and bus.snapshot()["occupied"] == 1
        with pytest.raises(CallosumOverloaded):
            await bus.ask_detail({"type":"ASK_DETAIL", "question":"overload"})
        model.block = False
        model.release.set()
        await model.drained.wait()
        result = await bus.ask_detail({"type":"ASK_DETAIL", "question":"next"})
        assert result["notes_sum"] == "fresh" and bus.snapshot()["occupied"] == 0
    finally:
        model.release.set()
        await bus.close()
        await worker


@pytest.mark.asyncio
async def test_engine_session_shutdown_drains_its_real_worker_and_reports_payload_free_health():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"sr-dual-brain-llm/scripts"))
    from engine_stdio import EngineSession, _handle_health
    session = await EngineSession.create("private-session-id")
    model = BlockingRight()
    session.right.deepen = model.deepen
    ask = asyncio.create_task(session.callosum.ask_detail(
        {"qid":"private-qid", "type":"ASK_DETAIL", "question":"private text"}))
    try:
        await model.started.wait()
        health = _handle_health({"private-session-id":session})
        assert health["consultations"]["occupied"] == health["consultations"]["running"] == 1
        assert "private" not in json.dumps(health)
        await session.close()
        with pytest.raises(CallosumClosed):
            await ask
        assert session.worker_task.done() and model.cancelled.is_set()
        assert _handle_health({"private-session-id":session})["consultations"]["occupied"] == 0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_queued_consultation_expiring_behind_busy_work_never_calls_the_model():
    class SerialRight:
        def __init__(self):
            self.started, self.release = asyncio.Event(), asyncio.Event()
            self.questions = []
        async def deepen(self, qid, question, *args, **kwargs):
            self.questions.append(question)
            if question == "slow":
                self.started.set()
                await self.release.wait()
            return {"notes_sum":question,"confidence_r":.8}
    bus, model = Callosum(slot_ms=0), SerialRight()
    worker = asyncio.create_task(right_worker(bus, None, model))
    first = asyncio.create_task(bus.ask_detail({"type":"ASK_DETAIL","question":"slow"}))
    try:
        await model.started.wait()
        with pytest.raises(TimeoutError):
            await bus.ask_detail({"qid":"reuse","type":"ASK_DETAIL","question":"expired"}, timeout_ms=20)
        assert bus.snapshot()["queued"] == 0
        model.release.set()
        await first
        result = await bus.ask_detail({"qid":"reuse","type":"ASK_DETAIL","question":"fresh"})
        assert result["notes_sum"] == "fresh" and model.questions == ["slow", "fresh"]
    finally:
        model.release.set()
        await bus.close()
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize("abandon", ["cancel", "timeout", "close"])
async def test_overlapping_shutdown_does_not_cancel_asynchronous_model_cleanup_twice(abandon):
    class CleanupRight(BlockingRight):
        def __init__(self):
            super().__init__()
            self.release, self.cleaned = asyncio.Event(), asyncio.Event()
        async def result(self):
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
                await self.release.wait()
                self.cleaned.set()
    bus, model = Callosum(slot_ms=0), CleanupRight()
    worker = asyncio.create_task(right_worker(bus, None, model))
    ask = asyncio.create_task(bus.ask_detail({"type":"ASK_DETAIL","question":"cleanup"}, timeout_ms=100))
    await model.started.wait()
    closers = []
    if abandon == "cancel":
        ask.cancel()
    elif abandon == "close":
        closers.append(asyncio.create_task(bus.close()))
    with pytest.raises({"cancel":asyncio.CancelledError,"timeout":TimeoutError,"close":CallosumClosed}[abandon]):
        await ask
    await model.cancelled.wait()
    closers.extend([asyncio.create_task(bus.close()), asyncio.create_task(bus.close())])
    await asyncio.sleep(0)
    model.release.set()
    await asyncio.gather(*closers)
    await worker
    assert model.cleaned.is_set() and bus.snapshot()["occupied"] == 0


@pytest.mark.asyncio
async def test_cancelling_a_shutdown_waiter_keeps_client_cleanup_owned_and_tracked():
    bus = Callosum(slot_ms=0)
    started, cleaning, release, cleaned = [asyncio.Event() for _ in range(4)]
    async def operation():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            cleaned.set()
    ask = asyncio.create_task(bus.ask_detail({}))
    req = await bus.recv_request()
    executing = asyncio.create_task(bus.run_request(req, operation))
    await started.wait()
    closer = asyncio.create_task(bus.close())
    await cleaning.wait()
    closer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closer
    with pytest.raises(CallosumClosed):
        await ask
    assert bus.snapshot()["running"] == 1
    release.set()
    await bus.close()
    from core.callosum import RequestAbandoned
    with pytest.raises(RequestAbandoned):
        await executing
    assert cleaned.is_set() and bus.snapshot()["occupied"] == 0


@pytest.mark.asyncio
async def test_late_client_failure_after_worker_exit_is_not_logged_as_an_unhandled_task():
    import gc
    started, cleaning, release, finished = [asyncio.Event() for _ in range(4)]
    class FailingCleanup:
        async def deepen(self, *args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                finished.set()
                raise RuntimeError("PRIVATE_STALE_ERROR")
    loop = asyncio.get_running_loop()
    original, errors = loop.get_exception_handler(), []
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    bus = Callosum(slot_ms=0)
    worker = asyncio.create_task(right_worker(bus, None, FailingCleanup()))
    ask = asyncio.create_task(bus.ask_detail({"type":"ASK_DETAIL","question":"first"}))
    try:
        await started.wait()
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
        await cleaning.wait()
        ask.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ask
        release.set()
        await finished.wait()
        await asyncio.sleep(0)
        del ask, worker
        gc.collect()
        await asyncio.sleep(0)
        assert not errors and bus.snapshot()["running"] == 0
    finally:
        release.set()
        await bus.close()
        loop.set_exception_handler(original)


@pytest.mark.asyncio
async def test_shutdown_before_a_scheduled_operation_starts_never_invokes_its_factory():
    from core.callosum import RequestAbandoned
    bus = Callosum(slot_ms=0)
    ask = asyncio.create_task(bus.ask_detail({}))
    req = await bus.recv_request()
    calls = []
    async def result():
        return {}
    def factory():
        calls.append(True)
        return result()
    executing = asyncio.create_task(bus.run_request(req, factory))
    await asyncio.sleep(0)
    await bus.close()
    with pytest.raises(CallosumClosed):
        await ask
    with pytest.raises(RequestAbandoned):
        await executing
    assert calls == [] and bus.snapshot()["occupied"] == 0
