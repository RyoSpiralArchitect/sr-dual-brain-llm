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

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import aiohttp


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    api_base: Optional[str] = None
    organization: Optional[str] = None
    max_output_tokens: int = 512
    timeout_seconds: int = 40
    extra_headers: Dict[str, str] = field(default_factory=dict)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)


def load_llm_config(scope: str = "LLM") -> Optional[LLMConfig]:
    """Load an LLM configuration from environment variables.

    The function is intentionally permissive so that users can quickly try out
    different providers (OpenAI, Google, Anthropic, Mistral, xAI, HuggingFace)
    without needing Kafka/Docker. It checks role-specific overrides first
    (e.g., LEFT_BRAIN_PROVIDER) and then falls back to the shared LLM_* vars.
    """

    prefix = scope.upper()
    provider = _env(f"{prefix}_PROVIDER") or _env("LLM_PROVIDER")
    model = _env(f"{prefix}_MODEL") or _env("LLM_MODEL_ID")

    if not provider or not model:
        return None

    provider_key = f"{provider.upper()}_API_KEY"
    api_key = (
        _env(f"{prefix}_API_KEY")
        or _env(provider_key)
        or _env("LLM_API_KEY")
        or _env("HUGGINGFACE_API_TOKEN")
        or _env("HF_TOKEN")
    )

    if not api_key:
        return None

    provider_base = _env(f"{provider.upper()}_API_BASE")
    api_base = _env(f"{prefix}_API_BASE") or _env("LLM_API_BASE") or provider_base
    organization = _env(f"{prefix}_ORG") or _env("OPENAI_ORGANIZATION")
    max_output_tokens = int(
        _env(f"{prefix}_MAX_TOKENS")
        or _env("LLM_MAX_OUTPUT_TOKENS")
        or 512
    )
    timeout_seconds = int(_env(f"{prefix}_TIMEOUT") or _env("LLM_TIMEOUT") or 40)

    extra_headers: Dict[str, str] = {}
    if provider.lower() == "anthropic":
        extra_headers["anthropic-version"] = _env("ANTHROPIC_VERSION", "2023-06-01")

    return LLMConfig(
        provider=provider.lower(),
        model=model,
        api_key=api_key,
        api_base=api_base,
        organization=organization,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        extra_headers=extra_headers,
    )


class LLMClient:
    """Thin async HTTP client that can hit multiple popular LLM APIs.

    The implementation sticks to provider-compatible HTTP payloads so we don't
    need extra SDK dependencies. Only minimal fields are sent to keep requests
    simple and debuggable.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = config.provider

    async def _post_json(self, url: str, payload: Dict[str, object], headers: Dict[str, str]) -> Dict[str, object]:
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 400:
                    raise RuntimeError(f"{self.config.provider} returned {resp.status}: {data}")
                return data

    async def complete(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.7) -> str:
        provider = self.provider
        if provider in {"openai", "mistral", "xai"}:
            return await self._openai_style(prompt, system=system, temperature=temperature)
        if provider == "anthropic":
            return await self._anthropic(prompt, system=system, temperature=temperature)
        if provider == "google":
            return await self._google(prompt, system=system, temperature=temperature)
        if provider == "huggingface":
            return await self._huggingface(prompt, system=system, temperature=temperature)
        raise ValueError(f"Unsupported provider: {provider}")

    def _default_base(self) -> str:
        provider = self.provider
        if provider == "openai":
            return "https://api.openai.com/v1"
        if provider == "mistral":
            return "https://api.mistral.ai/v1"
        if provider == "xai":
            return "https://api.x.ai/v1"
        if provider == "anthropic":
            return "https://api.anthropic.com"
        if provider == "google":
            return "https://generativelanguage.googleapis.com/v1beta"
        if provider == "huggingface":
            return "https://api-inference.huggingface.co/models"
        raise ValueError(f"Unsupported provider: {provider}")

    async def _openai_style(self, prompt: str, *, system: Optional[str], temperature: float) -> str:
        base = (self.config.api_base or self._default_base()).rstrip("/")
        url = f"{base}/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": self.config.max_output_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        headers.update(self.config.extra_headers)
        data = await self._post_json(url, payload, headers)
        choices = data.get("choices") or []
        if choices and choices[0].get("message"):
            return choices[0]["message"].get("content", "").strip()
        if choices and choices[0].get("delta"):
            return choices[0]["delta"].get("content", "").strip()
        return str(data)

    async def _anthropic(self, prompt: str, *, system: Optional[str], temperature: float) -> str:
        base = (self.config.api_base or self._default_base()).rstrip("/")
        url = f"{base}/v1/messages"
        headers = {
            "x-api-key": self.config.api_key,
            "content-type": "application/json",
            "anthropic-version": self.config.extra_headers.get("anthropic-version", "2023-06-01"),
        }
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_output_tokens,
            "temperature": float(temperature),
        }
        if system:
            payload["system"] = system
        data = await self._post_json(url, payload, headers)
        content = data.get("content") or []
        if content and isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "\n".join([part for part in text_parts if part]).strip()
        return str(data)

    async def _google(self, prompt: str, *, system: Optional[str], temperature: float) -> str:
        base = (self.config.api_base or self._default_base()).rstrip("/")
        url = f"{base}/models/{self.config.model}:generateContent?key={self.config.api_key}"
        headers = {"Content-Type": "application/json"}
        payload: Dict[str, object] = {
            "contents": [
                {
                    "parts": [{"text": prompt}],
                }
            ],
        }
        if system:
            payload["system_instruction"] = {"parts": [{"text": system}]}
        payload["generationConfig"] = {
            "temperature": float(temperature),
            "maxOutputTokens": self.config.max_output_tokens,
        }
        data = await self._post_json(url, payload, headers)
        candidates = data.get("candidates") or []
        if candidates:
            parts = (candidates[0].get("content") or {}).get("parts", [])
            texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
            if texts:
                return "\n".join([t for t in texts if t]).strip()
        return str(data)

    async def _huggingface(self, prompt: str, *, system: Optional[str], temperature: float) -> str:
        base = (self.config.api_base or self._default_base()).rstrip("/")
        url = f"{base}/{self.config.model}"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, object] = {
            "inputs": prompt if not system else f"{system}\n\nUser: {prompt}\nAssistant:",
            "options": {"use_cache": True, "wait_for_model": True},
            "parameters": {
                "temperature": float(temperature),
                "max_new_tokens": self.config.max_output_tokens,
                "return_full_text": False,
            },
        }
        data = await self._post_json(url, payload, headers)
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                text = first.get("generated_text") or first.get("translation_text")
                if text:
                    return str(text).strip()
        if isinstance(data, dict):
            if "generated_text" in data:
                return str(data["generated_text"]).strip()
        return str(data)
