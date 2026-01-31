import asyncio

from core.llm_client import LLMConfig, LLMClient, load_llm_config


def test_load_llm_config_prefers_scoped_and_provider_base(monkeypatch):
    monkeypatch.setenv("RIGHT_BRAIN_PROVIDER", "mistral")
    monkeypatch.setenv("RIGHT_BRAIN_MODEL", "mistral-large")
    monkeypatch.setenv("MISTRAL_API_KEY", "abc")
    monkeypatch.setenv("MISTRAL_API_BASE", "https://example.mistral.ai/v1")
    cfg = load_llm_config("RIGHT_BRAIN")
    assert cfg is not None
    assert cfg.provider == "mistral"
    assert cfg.model == "mistral-large"
    assert cfg.api_key == "abc"
    assert cfg.api_base == "https://example.mistral.ai/v1"


def test_load_llm_config_supports_shared_env(monkeypatch):
    monkeypatch.delenv("RIGHT_BRAIN_PROVIDER", raising=False)
    monkeypatch.delenv("RIGHT_BRAIN_MODEL", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL_ID", "claude-3-haiku-20240307")
    monkeypatch.setenv("LLM_API_KEY", "shared-key")
    cfg = load_llm_config("RIGHT_BRAIN")
    assert cfg is not None
    assert cfg.provider == "anthropic"
    assert cfg.api_key == "shared-key"


def test_default_bases_cover_supported_providers():
    providers = {
        "openai": "https://api.openai.com/v1",
        "mistral": "https://api.mistral.ai/v1",
        "xai": "https://api.x.ai/v1",
        "anthropic": "https://api.anthropic.com",
        "google": "https://generativelanguage.googleapis.com/v1beta",
        "huggingface": "https://api-inference.huggingface.co/models",
    }
    for provider, expected_base in providers.items():
        cfg = LLMConfig(provider=provider, model="dummy", api_key="key")
        client = LLMClient(cfg)
        assert client._default_base() == expected_base


def test_openai_style_auto_continue_on_length(monkeypatch):
    cfg = LLMConfig(
        provider="openai",
        model="dummy",
        api_key="key",
        max_output_tokens=5,
        auto_continue=True,
        max_continuations=2,
    )
    client = LLMClient(cfg)
    calls = []

    async def fake_post_json(url, payload, headers):
        calls.append(payload)
        if len(calls) == 1:
            return {
                "choices": [
                    {
                        "message": {"content": "Hello"},
                        "finish_reason": "length",
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {"content": " world"},
                    "finish_reason": "stop",
                }
            ]
        }

    monkeypatch.setattr(client, "_post_json", fake_post_json)

    out = asyncio.run(client.complete("prompt", system="sys", temperature=0.2))
    assert out == "Hello world"
    assert len(calls) == 2
    assert any(msg.get("role") == "assistant" for msg in calls[1].get("messages", []))


def test_openai_style_auto_continue_can_be_disabled(monkeypatch):
    cfg = LLMConfig(
        provider="openai",
        model="dummy",
        api_key="key",
        max_output_tokens=5,
        auto_continue=False,
        max_continuations=2,
    )
    client = LLMClient(cfg)
    calls = []

    async def fake_post_json(url, payload, headers):
        calls.append(payload)
        return {
            "choices": [
                {
                    "message": {"content": "Hello"},
                    "finish_reason": "length",
                }
            ]
        }

    monkeypatch.setattr(client, "_post_json", fake_post_json)

    out = asyncio.run(client.complete("prompt", system="sys", temperature=0.2))
    assert out == "Hello"
    assert len(calls) == 1


def test_iter_sse_data_handles_chunk_boundaries():
    async def fake_chunks():
        yield b"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n"
        yield b"\n"
        yield b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n"
        yield b"data: [DONE]\n\n"

    async def collect():
        out = []
        async for item in LLMClient._iter_sse_data(fake_chunks()):
            out.append(item)
        return out

    items = asyncio.run(collect())
    assert items[:2] == [
        "{\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}",
        "{\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}",
    ]
    assert items[-1] == "[DONE]"


def test_openai_style_consume_stream_calls_callback():
    cfg = LLMConfig(provider="openai", model="dummy", api_key="key")
    client = LLMClient(cfg)
    deltas = []

    async def fake_data():
        yield "{\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}"
        yield "{\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}"
        yield "[DONE]"

    def on_delta(text: str):
        deltas.append(text)

    full, finish = asyncio.run(client._consume_openai_style_stream(fake_data(), on_delta=on_delta))
    assert full == "Hello"
    assert finish == "stop"
    assert deltas == ["Hel", "lo"]


def test_openai_style_complete_stream_auto_continue(monkeypatch):
    cfg = LLMConfig(
        provider="openai",
        model="dummy",
        api_key="key",
        max_output_tokens=5,
        auto_continue=True,
        max_continuations=2,
    )
    client = LLMClient(cfg)
    calls = []
    streamed = []

    async def fake_stream_call(messages, *, temperature, on_delta):
        calls.append(messages)
        if len(calls) == 1:
            if on_delta:
                on_delta("Hello")
            return "Hello", "length"
        if on_delta:
            on_delta(" world")
        return " world", "stop"

    monkeypatch.setattr(client, "_openai_style_stream_call", fake_stream_call)

    out = asyncio.run(client.complete_stream("prompt", system="sys", temperature=0.2, on_delta=streamed.append))
    assert out == "Hello world"
    assert streamed == ["Hello", " world"]
    assert len(calls) == 2
    assert any(msg.get("role") == "assistant" for msg in calls[1])
