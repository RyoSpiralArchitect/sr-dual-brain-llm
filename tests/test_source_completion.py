"""Adversarial final-answer routes use fake provider responses, never model calls."""
import asyncio
from types import SimpleNamespace

import pytest

from core.llm_client import LLMConfig
from core.models import LeftBrainModel, RightBrainModel
from core.source_completion import source_completion_required
from test_conversation_memory import memory, retrieve, write
from test_dual_brain import make_stage_controller


def source_controller():
    m = memory()
    m.track_persistence = True
    write(m, "Alpha", "old", "14 days")
    m.pending_memory_writes.clear()
    left = LeftBrainModel(LLMConfig("openai", "fake", "NOT_A_KEY", auto_continue=False))
    return make_stage_controller(memory=m, left_model=left)


async def correct(controller, **options):
    return await controller.process(
        "Project Alpha deadline is now seven days", qid="new", system2_mode="off",
        memory_context={"id": "Alpha", "aliases": ["Alpha"], "remember": True, "supersedes": ["old"]},
        **options,
    )


def assert_old_source(m):
    assert [(t.qid, t.evidence_status) for t in m.past_qas] == [("old", "unverified")]
    assert not m.pending_memory_writes and not m.pending_memory_updates
    assert [t.answer for t in retrieve(m, "Project Alpha deadline?")] == ["14 days"]


@pytest.mark.parametrize("finish", [None, "length", "error", "stop"])
def test_complete_draft_then_polish_requires_confirmed_final_answer(finish):
    controller = source_controller()
    calls = []

    async def response(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            return "7 days including weekends", "stop"
        if finish == "error":
            raise OSError("fake provider failure")
        return "7 days except holidays", finish

    async def advice(**kwargs):
        return SimpleNamespace(executive_payload={"confidence": 1}, executive_directives={"style": "concise"})

    controller.left_model._llm_client._openai_style_stream_call = response
    controller._await_executive_advice = advice
    if finish == "stop":
        assert asyncio.run(correct(controller, leading_brain="left", executive_mode="polish")) == "7 days except holidays"
        assert [t.answer for t in retrieve(controller.memory, "Project Alpha deadline?")] == ["7 days except holidays"]
    else:
        with pytest.raises(RuntimeError, match="confirmed final completion"):
            asyncio.run(correct(controller, leading_brain="left", executive_mode="polish"))
        assert_old_source(controller.memory)
    assert len(calls) == 2
    assert not source_completion_required()


@pytest.mark.parametrize("finish", [None, "length", "stop"])
def test_retained_right_preview_requires_its_own_terminal_completion(finish):
    controller = source_controller()
    right = RightBrainModel(LLMConfig("openai", "fake", "NOT_A_KEY", auto_continue=False))
    controller.right_model = right

    async def draft(*args, **kwargs):
        # Let both the successful preview and the built-in local failure fallback finish.
        await asyncio.sleep(0.5)
        return "7 days including weekends", "stop"

    async def preview(*args, **kwargs):
        return "7 days except holidays", finish

    controller.left_model._llm_client._openai_style_stream_call = draft
    right._llm_client._openai_style_stream_call = preview
    route = controller._run_routing_stage

    async def skip_consult(**kwargs):
        kwargs["director_control"] = {"consult": "skip"}
        return await route(**kwargs)

    controller._run_routing_stage = skip_consult
    if finish == "stop":
        assert asyncio.run(correct(controller, leading_brain="right")) == "7 days except holidays"
    else:
        with pytest.raises(RuntimeError, match="confirmed final completion"):
            asyncio.run(correct(controller, leading_brain="right"))
        assert_old_source(controller.memory)


def test_verified_draft_remains_eligible_when_integration_falls_back_unchanged():
    controller = source_controller()

    async def draft(*args, **kwargs):
        return "7 days including weekends", "stop"

    controller.left_model._llm_client._openai_style_stream_call = draft
    stage = controller._run_consult_skipped_stage

    async def integrate(**kwargs):
        result = await stage(**kwargs)

        async def fail(*args, **kwargs):
            raise OSError("fake integration outage")

        controller.left_model._llm_client._openai_style_stream_call = fail
        result.final_answer = await controller.left_model.integrate_info_async(
            question=kwargs["question"], draft=result.final_answer, info=result.final_answer)
        return result

    controller._run_consult_skipped_stage = integrate
    assert asyncio.run(correct(controller, leading_brain="left")) == "7 days including weekends"
    assert [t.answer for t in retrieve(controller.memory, "Project Alpha deadline?")] == ["7 days including weekends"]


def test_output_truncation_cannot_promote_a_partial_verified_completion():
    controller = source_controller()

    async def draft(*args, **kwargs):
        return "7 days except holidays", "stop"

    controller.left_model._llm_client._openai_style_stream_call = draft
    controller._apply_director_output_constraints = lambda text, **kwargs: text[:6]
    with pytest.raises(RuntimeError, match="confirmed final completion"):
        asyncio.run(correct(controller, leading_brain="left"))
    assert_old_source(controller.memory)


def test_completion_requirement_does_not_leak_into_concurrent_ordinary_request():
    controller = source_controller()
    client = controller.left_model._llm_client

    async def response(messages, **kwargs):
        await asyncio.sleep(0)
        if messages[-1]["content"] == "ordinary":
            return "ordinary partial", None
        return "7 days including weekends", "stop"

    client._openai_style_stream_call = response

    async def together():
        return await asyncio.gather(correct(controller, leading_brain="left"), client.complete_stream("ordinary"))

    assert asyncio.run(together()) == ["7 days including weekends", "ordinary partial"]
    assert not source_completion_required()
