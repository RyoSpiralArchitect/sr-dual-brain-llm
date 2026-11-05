"""Critic completion and result-shape controls with every HTTP entry blocked."""
import asyncio
import json

import pytest

from core.callosum import Callosum
from core.dual_brain_support import _system2_resolution_from_critique
from core.llm_client import CompletionValidationError, LLMConfig
from core.models import RightBrainModel
from core.right_worker import right_worker
from test_dual_brain import CaptureLeftModel, make_stage_controller


CLEAN = '{"verdict":"ok","issues":[],"fixes":[]}'
ISSUES = '{"verdict":"issues","issues":["A condition is missing."],"fixes":["Keep the condition."]}'
DRAFT = "The supplied records differ."
QUESTION = "Compare these supplied project records."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Unexpected real provider transport")
    monkeypatch.setattr("core.llm_client.aiohttp.ClientSession", forbidden)


def response(content=CLEAN, reason="stop", **message_fields):
    return {"choices": [{"message": {"content": content, **message_fields}, "finish_reason": reason}]}


def scripted_model(responses, *, auto_continue=False, max_continuations=0):
    model = RightBrainModel(LLMConfig("openai", "offline-critic", "NO_CREDENTIAL",
        api_base="https://fake.invalid/v1", auto_continue=auto_continue,
        max_continuations=max_continuations))
    calls = []

    async def post(_url, payload, _headers, **kwargs):
        calls.append({"payload": payload, **kwargs})
        item = responses[len(calls) - 1]
        if isinstance(item, BaseException):
            raise item
        return item

    model._llm_client._post_json = post
    return model, calls


async def review(model, **kwargs):
    return await model.criticise_reasoning("probe", QUESTION, DRAFT, allow_micro_fallback=False, **kwargs)


def assert_unusable(result, status):
    assert result["critic_status"] == status
    assert result["critic_review_valid"] is False
    assert result["critic_external_review_valid"] is False
    assert result["confidence_r"] == 0.0
    resolved, basis = _system2_resolution_from_critique(
        verdict=result["verdict"], issues=result["issues"],
        critic_kind=result["critic_kind"], critic_sum=result["critic_sum"])
    assert resolved is False and basis == "critic_unhealthy"


@pytest.mark.parametrize("reason", [None, "length", "content_filter", "tool_calls", "unknown"])
def test_unsuccessful_critic_completion_never_becomes_a_clean_review(monkeypatch, reason):
    monkeypatch.setenv("DUALBRAIN_CRITIC_JSON_REPAIR_CALL", "1")
    model, calls = scripted_model([response(reason=reason)])
    result = asyncio.run(review(model))
    assert_unusable(result, "incomplete_response")
    assert result["critic_validation"] == {
        "initial": {"status": "incomplete_response", "error_type": "CompletionValidationError"},
        "repair": None,
    }
    assert len(calls) == 1


@pytest.mark.parametrize("content,status", [
    ('{"verdict":"ok"}', "invalid_schema"),
    ('{"verdict":"ok","issues":[]}', "invalid_schema"),
    ('{"verdict":"ok","issues":"The records contradict the draft.","fixes":[]}', "invalid_schema"),
    ('{"verdict":"ok","issues":[],"fixes":{"required":"change answer"}}', "invalid_schema"),
    ('{"verdict":"issues","issues":[true],"fixes":[]}', "invalid_schema"),
    ('{"verdict":"issues","issues":[{"reason":"Contradiction"}],"fixes":[]}', "invalid_schema"),
    ('{"verdict":"issues","issues":[" "],"fixes":[]}', "invalid_schema"),
    ('{"verdict":true,"issues":[],"fixes":[]}', "invalid_schema"),
    ('{"verdict":"ok","issues":["Contradiction"],"fixes":[]}', "inconsistent"),
    ('{"verdict":"issues","issues":[],"fixes":[]}', "inconsistent"),
    ('{"verdict":"issues","issues":["!!!"],"fixes":[]}', "inconsistent"),
    ('{"verdict":"ok","issues":[],"fixes":["Change the answer"]}', "inconsistent"),
    ('{"verdict":"ok","issues":["Contradiction"],"issues":[],"fixes":[]}', "invalid_schema"),
    ("{'verdict':'ok','issues':['Contradiction'],'issues':[],'fixes':[]}", "invalid_schema"),
])
def test_shape_errors_cannot_discard_issues_or_add_a_repair_call(monkeypatch, content, status):
    monkeypatch.setenv("DUALBRAIN_CRITIC_JSON_REPAIR_CALL", "1")
    model, calls = scripted_model([response(content)])
    result = asyncio.run(review(model))
    assert_unusable(result, status)
    assert result["critic_validation"] == {"initial": {"status": status, "error_type": None}, "repair": None}
    assert len(calls) == 1


@pytest.mark.parametrize("packet,status", [
    ({"choices": []}, "invalid_response"),
    ({"choices": [response()["choices"][0], response()["choices"][0]]}, "invalid_response"),
    ({"choices": [{"message": CLEAN, "finish_reason": "stop"}]}, "invalid_response"),
    (response({"verdict": "ok", "issues": [], "fixes": []}), "invalid_response"),
    (response(refusal="PRIVATE_REFUSAL"), "refused_response"),
    (response(tool_calls=[{"id": "tool"}]), "invalid_response"),
])
def test_invalid_response_envelopes_fail_before_parsing_clean_text(packet, status):
    model, calls = scripted_model([packet])
    result = asyncio.run(review(model))
    assert_unusable(result, status)
    assert len(calls) == 1 and result["critic_validation"]["repair"] is None
    assert "PRIVATE_REFUSAL" not in json.dumps(result)


@pytest.mark.parametrize("content", [CLEAN, ISSUES,
    "```json\n" + CLEAN + "\n```",
    "{'verdict': 'ok', 'issues': [], 'fixes': [],}",
])
def test_valid_reviews_preserve_tolerated_syntax(content):
    model, calls = scripted_model([response(content)])
    result = asyncio.run(review(model))
    assert result["critic_status"] == "ok"
    assert result["critic_review_valid"] is result["critic_external_review_valid"] is True
    assert result["critic_kind"] == "external"
    assert result["verdict"] == ("issues" if content == ISSUES else "ok")
    assert len(calls) == 1


@pytest.mark.parametrize("content,reason,status", [
    (CLEAN, "stop", "ok"),
    (CLEAN, "length", "incomplete_response"),
    ("", "stop", "empty_response"),
    ('{"verdict":"ok"}', "stop", "invalid_schema"),
    ('{"verdict":"ok","issues":"Contradiction","fixes":[]}', "stop", "invalid_schema"),
    ('{"verdict":"ok","issues":["Contradiction"],"issues":[],"fixes":[]}', "stop", "invalid_schema"),
    ('{"verdict":"issues","issues":[],"fixes":[]}', "stop", "inconsistent"),
    ("Still unstructured.", "stop", "unstructured"),
])
def test_existing_single_repair_requires_the_same_completion_and_shape(monkeypatch, content, reason, status):
    monkeypatch.setenv("DUALBRAIN_CRITIC_JSON_REPAIR_CALL", "1")
    model, calls = scripted_model([response("I found an issue."), response(content, reason)])
    result = asyncio.run(review(model))
    assert result["critic_validation"]["initial"] == {"status": "unstructured", "error_type": None}
    assert result["critic_validation"]["repair"]["status"] == status
    if status == "ok":
        assert result["critic_external_review_valid"] is True
        assert result["critic_review_valid"] is True
        assert result["critic_kind"] == "external_repaired"
    else:
        assert_unusable(result, status)
    assert len(calls) == 2
    assert calls[0]["timeout_seconds"] == 24.0 and calls[1]["timeout_seconds"] == 18.0


def test_repair_transport_failure_preserves_error_category_without_provider_text(monkeypatch):
    monkeypatch.setenv("DUALBRAIN_CRITIC_JSON_REPAIR_CALL", "1")
    model, calls = scripted_model([response("I found an issue."), OSError("PRIVATE_PROVIDER_TEXT")])
    result = asyncio.run(review(model))
    assert_unusable(result, "provider_error")
    assert result["critic_validation"]["repair"] == {"status": "provider_error", "error_type": "OSError"}
    assert "PRIVATE_PROVIDER_TEXT" not in json.dumps(result)
    assert len(calls) == 2


def test_repair_refusal_cannot_turn_clean_looking_text_into_a_review(monkeypatch):
    monkeypatch.setenv("DUALBRAIN_CRITIC_JSON_REPAIR_CALL", "1")
    model, calls = scripted_model([response("I found an issue."), response(refusal="PRIVATE_REFUSAL")])
    result = asyncio.run(review(model))
    assert_unusable(result, "refused_response")
    assert result["critic_validation"]["initial"]["status"] == "unstructured"
    assert result["critic_validation"]["repair"]["status"] == "refused_response"
    assert "PRIVATE_REFUSAL" not in json.dumps(result) and len(calls) == 2


def test_repair_disabled_does_not_add_any_request(monkeypatch):
    monkeypatch.setenv("DUALBRAIN_CRITIC_JSON_REPAIR_CALL", "0")
    model, calls = scripted_model([response("I found an issue.")])
    assert_unusable(asyncio.run(review(model)), "unstructured")
    assert len(calls) == 1


@pytest.mark.parametrize("final_reason,status", [("stop", "ok"), ("length", "incomplete_response")])
def test_existing_configured_continuation_requires_final_completion(final_reason, status):
    model, calls = scripted_model([
        response('{"verdict":"ok","issues":[],', "length"),
        response('"fixes":[]}', final_reason),
    ], auto_continue=True, max_continuations=1)
    result = asyncio.run(review(model))
    assert len(calls) == 2
    if status == "ok":
        assert result["critic_external_review_valid"] is True and result["critic_status"] == "ok"
    else:
        assert_unusable(result, status)
    assert result["critic_validation"]["repair"] is None


def test_invalid_external_review_keeps_explicit_local_micro_provenance():
    model, calls = scripted_model([OSError("PRIVATE_PROVIDER_TEXT")])
    result = asyncio.run(model.criticise_reasoning("probe", "Compute 2+2.", "2+2=5"))
    assert result["critic_status"] == "provider_error"
    assert result["critic_kind"] == "micro"
    assert result["critic_review_valid"] is True and result["critic_external_review_valid"] is False
    assert result["verdict"] == "issues" and any("4" in issue for issue in result["issues"])
    assert result["critic_validation"]["initial"] == {"status": "provider_error", "error_type": "OSError"}
    assert "PRIVATE_PROVIDER_TEXT" not in json.dumps(result)
    assert len(calls) == 1


@pytest.mark.parametrize("packet,reason", [
    (response(refusal="PRIVATE_REFUSAL"), "refused_response"),
    ({"choices": [response()["choices"][0], response()["choices"][0]]}, "invalid_response"),
])
def test_strict_envelope_validation_is_opt_in(packet, reason):
    model, calls = scripted_model([packet, packet])
    client = model._llm_client
    assert asyncio.run(client.complete("ordinary completion")) == CLEAN
    with pytest.raises(CompletionValidationError, match="terminal completion") as caught:
        asyncio.run(client.complete("strict completion", require_complete=True))
    assert isinstance(caught.value, RuntimeError) and caught.value.reason == reason
    assert "PRIVATE_REFUSAL" not in str(caught.value)
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("packet,status,valid", [
    (response(), "ok", True),
    (response(reason="content_filter"), "incomplete_response", False),
    (response('{"verdict":"ok","issues":"Contradiction","fixes":[]}'), "invalid_schema", False),
    (response(""), "empty_response", False),
    (OSError("private provider failure"), "provider_error", False),
])
async def test_production_worker_keeps_review_validity_separate_from_retained_answer(monkeypatch, packet, status, valid):
    monkeypatch.setenv("DUALBRAIN_CRITIC_JSON_REPAIR_CALL", "0")
    right, calls = scripted_model([packet])
    bus = Callosum(slot_ms=0)
    controller = make_stage_controller(left_model=CaptureLeftModel(draft=DRAFT, confidence=.9))
    controller.callosum, controller.right_model = bus, right
    received = []
    original_publish = bus.publish_response

    async def publish(request_id, result):
        received.append(result)
        return await original_publish(request_id, result)

    bus.publish_response = publish
    worker = asyncio.create_task(right_worker(bus, controller.memory, right))
    try:
        answer = await controller.process(QUESTION, leading_brain="left", system2_mode="on",
            executive_mode="off", executive_observer_mode="off")
        event = [payload for kind, payload in controller.telemetry.events if kind == "system2_refinement"][-1]
        assert event["critic_status"] == status
        assert event["resolved"] is valid and event["critic_unhealthy"] is not valid
        assert received[0]["critic_external_review_valid"] is valid
        assert received[0]["critic_review_valid"] is valid
        assert received[0]["critic_validation"]["initial"]["status"] == status
        assert answer == DRAFT and len(calls) == 1
        assert bus.snapshot()["occupied"] == 0
    finally:
        await bus.close()
        await worker


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["initial", "repair"])
async def test_cancellation_is_not_converted_to_an_invalid_review_or_retried(monkeypatch, stage):
    monkeypatch.setenv("DUALBRAIN_CRITIC_JSON_REPAIR_CALL", "1")
    model, calls = scripted_model([])
    entered, drained = asyncio.Event(), asyncio.Event()

    async def delayed(*_args, **_kwargs):
        calls.append({})
        if stage == "repair" and len(calls) == 1:
            return response("I found an issue.")
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            drained.set()

    model._llm_client._post_json = delayed
    task = asyncio.create_task(review(model))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert drained.is_set() and len(calls) == (1 if stage == "initial" else 2)
