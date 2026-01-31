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
