from core.llm_client import LLMConfig, LLMClient, load_llm_config
from core.models import LeftBrainModel, RightBrainModel


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


def test_neuro_system_prompts_are_brain_focused():
    left_prompt = LeftBrainModel._neuro_system_prompt("ctx")
    right_prompt = RightBrainModel._neuro_system_prompt("ctx", include_projection=True, psychoid_projection={"norm": 0.5})
    assert "left hemisphere" in left_prompt.lower()
    assert "right hemisphere" in right_prompt.lower()
    assert "context" in left_prompt.lower() and "context" in right_prompt.lower()
    assert "psychoid" in right_prompt.lower()
