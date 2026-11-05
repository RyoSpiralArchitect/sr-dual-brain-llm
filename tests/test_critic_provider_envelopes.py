"""A provider envelope must preserve text type and one completed answer."""
import asyncio
from copy import deepcopy
import json

import pytest

from core.llm_client import LLMConfig
from core.models import RightBrainModel


CLEAN = '{"verdict":"ok","issues":[],"fixes":[]}'


def anthropic(text=CLEAN, reason="end_turn"):
    return {"content": [{"type": "text", "text": text}], "stop_reason": reason}


def google(text=CLEAN, reason="STOP"):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": reason}]}


def huggingface(text=CLEAN, reason="eos_token"):
    return {"generated_text": text, "details": {"finish_reason": reason}}


@pytest.fixture(autouse=True)
def offline_contract(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Unexpected provider transport")
    monkeypatch.setattr("core.llm_client.aiohttp.ClientSession", forbidden)
    monkeypatch.setenv("DUALBRAIN_CRITIC_JSON_REPAIR_CALL", "1")


def scripted_model(provider, responses, *, auto_continue=False):
    model = RightBrainModel(LLMConfig(provider, "offline-envelope", "NOT_A_KEY",
        api_base="https://fake.invalid", auto_continue=auto_continue, max_continuations=1))
    calls = []

    async def post(_url, payload, _headers, **_kwargs):
        calls.append(deepcopy(payload))
        return deepcopy(responses[len(calls) - 1])

    model._llm_client._post_json = post
    return model, calls


async def review(model):
    return await model.criticise_reasoning("probe", "Compare the supplied records.",
        "The supplied records differ.", allow_micro_fallback=False)


INVALID = [
    pytest.param("huggingface", huggingface(json.loads(CLEAN)), id="hf-dict-text"),
    pytest.param("huggingface", [huggingface(json.loads(CLEAN))], id="hf-list-dict-text"),
    pytest.param("huggingface", huggingface([CLEAN]), id="hf-array-text"),
    pytest.param("huggingface", huggingface(None), id="hf-null-text"),
    pytest.param("huggingface", {"generated_text": CLEAN}, id="hf-missing-details"),
    pytest.param("huggingface", {"generated_text": CLEAN, "details": None}, id="hf-null-details"),
    pytest.param("huggingface", {"generated_text": CLEAN, "details": "eos_token"}, id="hf-string-details"),
    pytest.param("huggingface", [huggingface(), huggingface("conflicting answer")], id="hf-multiple-results"),
    pytest.param("huggingface", [], id="hf-no-result"),
    pytest.param("huggingface", [False], id="hf-invalid-result"),
    pytest.param("anthropic", anthropic(json.loads(CLEAN)), id="anthropic-dict-text"),
    pytest.param("anthropic", {"content": {"type": "text", "text": CLEAN}, "stop_reason": "end_turn"},
                 id="anthropic-content-not-list"),
    pytest.param("anthropic", {"content": [CLEAN], "stop_reason": "end_turn"}, id="anthropic-invalid-block"),
    pytest.param("anthropic", {"content": [{"text": CLEAN}], "stop_reason": "end_turn"},
                 id="anthropic-missing-block-type"),
    pytest.param("anthropic", {"content": [{"type": "tool_use", "text": CLEAN}], "stop_reason": "end_turn"},
                 id="anthropic-tool-disguised-as-text"),
    pytest.param("anthropic", {"content": [*anthropic()["content"], {"type": "tool_use", "name": "lookup"}],
                               "stop_reason": "end_turn"}, id="anthropic-mixed-tool-block"),
    pytest.param("anthropic", [anthropic(), anthropic()], id="anthropic-not-message"),
    pytest.param("google", google(json.loads(CLEAN)), id="google-dict-text"),
    pytest.param("google", {"candidates": google()["candidates"] * 2}, id="google-multiple-candidates"),
    pytest.param("google", {"candidates": []}, id="google-no-candidate"),
    pytest.param("google", {"candidates": [False]}, id="google-invalid-candidate"),
    pytest.param("google", {"candidates": [{"content": None, "finishReason": "STOP"}]},
                 id="google-missing-content"),
    pytest.param("google", {"candidates": [{"content": {"parts": CLEAN}, "finishReason": "STOP"}]},
                 id="google-parts-not-list"),
    pytest.param("google", {"candidates": [{"content": {"parts": [
        {"text": CLEAN, "functionCall": {"name": "lookup"}}]}, "finishReason": "STOP"}]},
                 id="google-tool-disguised-as-text"),
    pytest.param("google", {"candidates": [{"content": {"parts": [
        {"text": CLEAN}, {"inlineData": {"mimeType": "image/png", "data": "fixture"}}]},
        "finishReason": "STOP"}]}, id="google-mixed-nontext-block"),
]


@pytest.mark.parametrize("provider,packet", INVALID)
def test_invalid_provider_envelope_cannot_become_a_clean_critic(provider, packet):
    model, calls = scripted_model(provider, [packet])
    result = asyncio.run(review(model))
    assert result["critic_status"] == "invalid_response"
    assert result["critic_review_valid"] is result["critic_external_review_valid"] is False
    assert result["confidence_r"] == 0.
    assert result["critic_validation"] == {
        "initial": {"status": "invalid_response", "error_type": "CompletionValidationError"},
        "repair": None,
    }
    assert len(calls) == 1  # Envelope failure must not add transport retries or JSON repair.


@pytest.mark.parametrize("provider,packet", [
    ("anthropic", anthropic()), ("anthropic", anthropic(reason="stop_sequence")),
    ("anthropic", {"content": [{"type": "text", "text": '{"verdict":"ok",'},
        {"type": "text", "text": '"issues":[],"fixes":[]}'}], "stop_reason": "end_turn"}),
    ("google", google()),
    ("google", {"candidates": [{"content": {"parts": [{"text": '{"verdict":"ok",'},
        {"text": '"issues":[],"fixes":[]}', "thoughtSignature": "fixture"}]}, "finishReason": "STOP"}]}),
    ("huggingface", huggingface()), ("huggingface", [huggingface(reason="stop_sequence")]),
    ("huggingface", [{"translation_text": CLEAN, "details": {"finish_reason": "eos_token"}}]),
])
def test_supported_text_completions_keep_successful_terminal_reasons(provider, packet):
    model, calls = scripted_model(provider, [packet])
    result = asyncio.run(review(model))
    assert result["critic_status"] == "ok" and result["verdict"] == "ok"
    assert result["critic_review_valid"] is result["critic_external_review_valid"] is True
    assert len(calls) == 1


@pytest.mark.parametrize("provider,packet,status", [
    ("anthropic", anthropic(reason="max_tokens"), "incomplete_response"),
    ("google", google(reason="MAX_TOKENS"), "incomplete_response"),
    ("huggingface", huggingface(reason="length"), "incomplete_response"),
    ("anthropic", anthropic(""), "empty_response"),
    ("google", google(""), "empty_response"),
    ("huggingface", huggingface(""), "empty_response"),
    ("anthropic", {"content": [], "stop_reason": "end_turn"}, "empty_response"),
    ("google", {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}, "empty_response"),
])
def test_valid_envelope_keeps_empty_and_incomplete_completion_distinct(provider, packet, status):
    model, calls = scripted_model(provider, [packet])
    result = asyncio.run(review(model))
    assert result["critic_status"] == status
    assert result["critic_external_review_valid"] is False
    assert result["critic_validation"]["repair"] is None and len(calls) == 1


@pytest.mark.parametrize("provider,packet,ordinary_text", [
    ("huggingface", huggingface(json.loads(CLEAN)), str(json.loads(CLEAN))),
    ("huggingface", [huggingface(), huggingface("another result")], CLEAN),
    ("huggingface", {"generated_text": CLEAN, "details": "malformed"}, CLEAN),
    ("anthropic", {"content": [{"type": "tool_use", "text": CLEAN}], "stop_reason": "end_turn"}, CLEAN),
    ("google", {"candidates": google()["candidates"] * 2}, CLEAN),
    ("google", {"candidates": [{"content": {"parts": [{"text": CLEAN},
        {"functionCall": {"name": "lookup"}}]}, "finishReason": "STOP"}]}, CLEAN),
])
def test_strict_envelope_checks_leave_ordinary_completions_unchanged(provider, packet, ordinary_text):
    model, calls = scripted_model(provider, [packet, packet])
    assert asyncio.run(model._llm_client.complete("ordinary completion")) == ordinary_text
    assert asyncio.run(review(model))["critic_status"] == "invalid_response"
    assert len(calls) == 2


@pytest.mark.parametrize("provider,first,malformed", [
    ("huggingface", huggingface("The attribution needs review."), huggingface(json.loads(CLEAN))),
    ("anthropic", anthropic("The attribution needs review."),
        {"content": [{"type": "tool_use", "text": CLEAN}], "stop_reason": "end_turn"}),
    ("google", google("The attribution needs review."),
        {"candidates": [{"content": {"parts": [{"text": CLEAN, "functionCall": {}}]}, "finishReason": "STOP"}]}),
])
def test_critic_repair_cannot_bypass_the_provider_envelope(provider, first, malformed):
    model, calls = scripted_model(provider, [first, malformed])
    result = asyncio.run(review(model))
    assert result["critic_status"] == "invalid_response"
    assert result["critic_external_review_valid"] is False
    assert result["critic_validation"]["initial"]["status"] == "unstructured"
    assert result["critic_validation"]["repair"] == {
        "status": "invalid_response", "error_type": "CompletionValidationError"}
    assert len(calls) == 2


def test_anthropic_continuation_cannot_hide_a_nontext_completion():
    model, calls = scripted_model("anthropic", [
        anthropic('{"verdict":"ok","issues":[],', reason="max_tokens"),
        {"content": [{"type": "tool_use", "text": '"fixes":[]}'}], "stop_reason": "end_turn"},
    ], auto_continue=True)
    result = asyncio.run(review(model))
    assert result["critic_status"] == "invalid_response"
    assert result["critic_external_review_valid"] is False
    assert result["critic_validation"]["repair"] is None
    assert len(calls) == 2
