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

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .llm_client import LLMClient, LLMConfig, load_llm_config


_DIRECTOR_GUARDRAILS = (
    "You are the Executive Director module.\n"
    "You do NOT produce the user-facing answer.\n"
    "Your job is to steer memory usage + response discipline to reduce:\n"
    "- topic drift\n"
    "- repeated themes after the user disengages\n"
    "- 'reward-hack' verbosity (unasked tangents)\n"
    "- accidental references to hidden UI/telemetry.\n"
    "\n"
    "No chain-of-thought: output only conclusions as STRICT JSON.\n"
    "Never mention internal orchestration (left/right brain, corpus callosum, telemetry, qid, architecture paths).\n"
    "\n"
    "Return STRICT JSON only (no markdown, no extra text) with this schema:\n"
    "{\n"
    '  "memo": string,  // short internal memo (<= 1200 chars)\n'
    '  "control": {\n'
    '    "consult": "auto" | "force" | "skip",\n'
    '    "temperature": number | null,  // 0..1\n'
    '    "max_chars": number | null,  // clamp final user-facing answer length\n'
    '    "memory": {\n'
    '      "working_memory": "auto" | "keep" | "drop",\n'
    '      "long_term": "auto" | "keep" | "drop"\n'
    "    },\n"
    '    "append_clarifying_question": string | null\n'
    "  },\n"
    '  "confidence": number  // 0..1\n'
    "}\n"
)


def _safe_json_object(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    cleaned: Dict[str, Any] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            k = str(k)
        cleaned[k] = v
    return cleaned


@dataclass(frozen=True)
class DirectorAdvice:
    memo: str
    control: Dict[str, Any]
    confidence: float
    latency_ms: float
    source: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "memo": self.memo,
            "mix_in": "",
            "directives": dict(self.control),
            "confidence": float(self.confidence),
            "latency_ms": float(self.latency_ms),
            "source": self.source,
        }


class DirectorReasonerModel:
    def __init__(self, llm_config: Optional[LLMConfig] = None) -> None:
        self.llm_config = (
            llm_config
            or load_llm_config("DIRECTOR")
            or load_llm_config("EXECUTIVE")
            or load_llm_config("LEFT_BRAIN")
            or load_llm_config()
        )
        self._llm_client = LLMClient(self.llm_config) if self.llm_config else None
        self.uses_external_llm = bool(self._llm_client)

    async def advise(
        self,
        *,
        question: str,
        context: str,
        signals: Optional[Dict[str, Any]] = None,
        temperature: float = 0.15,
    ) -> DirectorAdvice:
        start = time.perf_counter()

        signals_obj = _safe_json_object(signals or {})
        if self._llm_client:
            prompt = (
                f"User message:\n{question}\n\n"
                f"Signals (JSON):\n{json.dumps(signals_obj, ensure_ascii=False)[:1200]}\n\n"
                f"Context (may be empty):\n{(context or '')[:1600]}\n\n"
                "Return JSON now."
            )
            try:
                raw = await self._llm_client.complete(
                    prompt,
                    system=_DIRECTOR_GUARDRAILS,
                    temperature=temperature,
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000.0
                memo = f"Director disabled (LLM error): {exc.__class__.__name__}: {exc}"
                return DirectorAdvice(
                    memo=memo[:1200],
                    control={
                        "consult": "auto",
                        "temperature": None,
                        "max_chars": None,
                        "memory": {"working_memory": "auto", "long_term": "auto"},
                        "append_clarifying_question": None,
                    },
                    confidence=0.0,
                    latency_ms=latency_ms,
                    source="error",
                )

            raw_text = str(raw or "").strip()
            payload: Dict[str, Any] = {}
            try:
                payload = json.loads(raw_text)
            except Exception:
                left = raw_text.find("{")
                right = raw_text.rfind("}")
                if left >= 0 and right > left:
                    try:
                        payload = json.loads(raw_text[left : right + 1])
                    except Exception:
                        payload = {}

            payload = _safe_json_object(payload)
            memo = str(payload.get("memo") or "").strip()
            control = _safe_json_object(payload.get("control"))
            confidence_raw = payload.get("confidence")
            try:
                confidence = float(confidence_raw)
            except Exception:
                confidence = 0.4

            latency_ms = (time.perf_counter() - start) * 1000.0
            return DirectorAdvice(
                memo=(memo[:1200] if memo else "(no memo)"),
                control=control,
                confidence=max(0.0, min(1.0, confidence)),
                latency_ms=latency_ms,
                source="llm",
            )

        # Heuristic fallback (no LLM configured)
        latency_ms = (time.perf_counter() - start) * 1000.0
        memo = (
            "(Director memo / heuristic)\n"
            "- Keep answers short on trivial turns\n"
            "- Avoid telemetry/UI references\n"
            "- Prefer dropping long-term memory if user input is short\n"
        )
        long_term = "drop" if signals_obj.get("is_trivial_chat") else "auto"
        return DirectorAdvice(
            memo=memo[:1200],
            control={
                "consult": "skip" if signals_obj.get("is_trivial_chat") else "auto",
                "temperature": 0.4 if signals_obj.get("is_trivial_chat") else None,
                "max_chars": 280 if signals_obj.get("is_trivial_chat") else None,
                "memory": {"working_memory": "keep", "long_term": long_term},
                "append_clarifying_question": None,
            },
            confidence=0.25,
            latency_ms=latency_ms,
            source="heuristic",
        )
