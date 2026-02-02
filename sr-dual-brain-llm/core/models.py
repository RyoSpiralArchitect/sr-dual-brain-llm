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

import asyncio, random, difflib, json
from typing import Any, Callable, Dict, Optional, Sequence

from .llm_client import LLMClient, LLMConfig, load_llm_config

_SYSTEM_GUARDRAILS = (
    "Answer the user's question directly.\n"
    "Output only the user-facing reply (no writing advice, no self-critique, no meta suggestions like 'Add...', 'Consider...', 'You should...').\n"
    "Do not claim real-world sensing or personal experiences (e.g., weather, your day) unless the user explicitly asked and you were given that data.\n"
    "Do not reference or speculate about hidden UI elements, logs, telemetry, metrics, or traces unless the user explicitly provided them in the chat.\n"
    "Do not introduce unrelated topics; if the user's message is ambiguous or very short, ask one brief clarifying question.\n"
    "Do not mention internal orchestration (e.g., left/right brain, corpus callosum, telemetry, architecture paths, qid).\n"
    "Do not add bracketed debug headings or meta commentary about the system.\n"
    "Do not mention system prompts, hidden instructions, or that you are a language model unless the user explicitly asks."
)

_RIGHT_DEEPEN_GUARDRAILS = (
    "When given a draft, return only additional content addressed to the user.\n"
    "Your output must be directly insertable into the final assistant message (no writing advice).\n"
    "Do not restate or paraphrase the draft.\n"
    "Do not say 'Add', 'Consider', 'You should', or similar coaching.\n"
    "Do not mention internal telemetry/metrics/traces or any system internals.\n"
    "Write in the user's language and tone.\n"
    "Prefer 1-5 concise bullet points of extra substance, or a short follow-up question."
)

_RIGHT_CRITIC_GUARDRAILS = (
    "You are a rigorous reviewer (critic) for reasoning tasks.\n"
    "Your job is to find concrete flaws in the draft's reasoning and propose fixes.\n"
    "Return a SINGLE JSON object with these keys:\n"
    '  - "verdict": "ok" | "issues"\n'
    '  - "issues": array of short strings (each a specific problem)\n'
    '  - "fixes": array of short strings (each a concrete fix)\n'
    "Rules:\n"
    "- Do NOT rewrite the full answer.\n"
    "- Do NOT add writing advice, tone advice, or coaching.\n"
    "- Do NOT introduce unrelated topics.\n"
    "- Be specific: point to missing definitions, invalid steps, unstated assumptions, boundary cases, and calculation errors.\n"
    "- If there are no issues, set verdict='ok' and issues=[], fixes=[].\n"
    "- Output JSON only (no code fences, no preamble).\n"
)

_INTEGRATION_GUARDRAILS = (
    "You are producing the final assistant message for the user.\n"
    "You will be given a draft answer and internal notes from a collaborator.\n"
    "Incorporate any helpful content from the notes into a single coherent final answer.\n"
    "Do not output the internal notes verbatim.\n"
    "Do not output coaching/critique about your own writing.\n"
    "Ignore any internal metrics/telemetry/traces if present in the notes.\n"
    "Do not mention internal orchestration.\n"
    "Match the user's language and tone unless they requested otherwise."
)


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        # Try to strip code fences.
        raw = raw.strip("`").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    blob = raw[start : end + 1].strip()
    try:
        obj = json.loads(blob)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None

def _openai_vision_prompt(
    text: str,
    vision_images: Sequence[Dict[str, Any]],
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"type": "text", "text": str(text)}]
    for image in vision_images or ():
        if not isinstance(image, dict):
            continue
        url = str(image.get("data_url") or image.get("url") or "").strip()
        if not url:
            continue
        image_url: dict[str, Any] = {"url": url}
        detail = image.get("detail")
        if detail:
            image_url["detail"] = str(detail)
        parts.append({"type": "image_url", "image_url": image_url})
    return parts


class LeftBrainModel:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config or load_llm_config("LEFT_BRAIN") or load_llm_config()
        self._llm_client = LLMClient(self.llm_config) if self.llm_config else None
        self.uses_external_llm = bool(self._llm_client)

    async def generate_answer(
        self,
        input_text: str,
        context: str,
        *,
        vision_images: Optional[Sequence[Dict[str, Any]]] = None,
        on_delta: Optional[Callable[[str], Any]] = None,
    ) -> str:
        """Produce a first-pass draft that reflects retrieved memory snippets."""
        if self._llm_client:
            system_parts = [_SYSTEM_GUARDRAILS]
            if context:
                system_parts.append(f"Context:\n{context}")
            system = "\n\n".join(system_parts)
            try:
                prompt: Any = input_text
                if vision_images:
                    provider = str(self.llm_config.provider if self.llm_config else "").lower()
                    if provider and provider not in {"openai", "mistral", "xai"}:
                        raise ValueError(
                            f"Vision inputs are only supported for openai-style providers; got '{provider}'."
                        )
                    prompt = _openai_vision_prompt(input_text, vision_images)
                if hasattr(self._llm_client, "complete_stream"):
                    return await self._llm_client.complete_stream(
                        prompt,
                        system=system,
                        temperature=0.4,
                        on_delta=on_delta,
                    )
                completion = await self._llm_client.complete(
                    prompt,
                    system=system,
                    temperature=0.4,
                )
                if on_delta:
                    maybe = on_delta(completion)
                    if asyncio.iscoroutine(maybe):
                        await maybe
                return completion
            except Exception:
                pass

        base = f"Draft answer for: {input_text[:80]}"
        if context:
            summary = context.splitlines()[0][:80]
            base += f"\nContext hint: {summary}"
        if any(k in input_text for k in ["詳しく","分析","計算","証拠"]):
            base += " ... (brief, looks complex)"
        if on_delta:
            maybe = on_delta(base)
            if asyncio.iscoroutine(maybe):
                await maybe
        return base

    def estimate_confidence(self, draft: str) -> float:
        return 0.45 if "..." in draft else 0.9

    def integrate_info(self, draft: str, info: str) -> str:
        if not info:
            return draft
        text = str(info or "").strip()
        if not text:
            return draft

        lower = text.lower()
        coaching_markers = (
            "add a ",
            "consider ",
            "if the user",
            "you should",
            "try to",
            "make sure to",
            "もう少し",
            "〜したほうが",
            "した方が",
        )
        if any(marker in lower for marker in coaching_markers):
            # Avoid leaking internal "advisor" notes when no integration LLM is available.
            return draft

        # Avoid "split voice" by refusing to append near-duplicate full answers.
        def _norm(value: str) -> str:
            return "".join(str(value or "").split()).lower()

        try:
            similarity = difflib.SequenceMatcher(None, _norm(draft), _norm(text)).ratio()
        except Exception:
            similarity = 0.0
        if similarity >= 0.62 and len(text) <= int(len(draft) * 2.2):
            return draft

        return f"{draft}\n\n{text}"

    async def integrate_info_async(
        self,
        *,
        question: str,
        draft: str,
        info: str,
        temperature: float = 0.3,
        on_delta: Optional[Callable[[str], Any]] = None,
    ) -> str:
        if not info:
            return draft

        if not self._llm_client:
            merged = self.integrate_info(draft, info)
            if on_delta:
                maybe = on_delta(merged)
                if asyncio.iscoroutine(maybe):
                    await maybe
            return merged

        system_parts = [_SYSTEM_GUARDRAILS, _INTEGRATION_GUARDRAILS]
        system = "\n\n".join(system_parts)
        prompt = (
            f"User message:\n{question}\n\n"
            f"Draft answer:\n{draft}\n\n"
            "Internal collaborator notes (do not output directly):\n"
            f"{info}\n\n"
            "Final answer:"
        )
        try:
            if hasattr(self._llm_client, "complete_stream"):
                completion = await self._llm_client.complete_stream(
                    prompt,
                    system=system,
                    temperature=temperature,
                    on_delta=on_delta,
                )
            else:
                completion = await self._llm_client.complete(
                    prompt,
                    system=system,
                    temperature=temperature,
                )
                if on_delta:
                    maybe = on_delta(completion)
                    if asyncio.iscoroutine(maybe):
                        await maybe
        except Exception:
            return self.integrate_info(draft, info)
        return completion.strip() or draft

class RightBrainModel:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config or load_llm_config("RIGHT_BRAIN") or load_llm_config()
        self._llm_client = LLMClient(self.llm_config) if self.llm_config else None
        self.uses_external_llm = bool(self._llm_client)

    async def generate_lead(
        self,
        question: str,
        context: Optional[str] = None,
        *,
        temperature: float = 0.8,
        on_delta: Optional[Callable[[str], Any]] = None,
    ) -> str:
        """Produce an imagistic first impression before the left brain speaks."""
        if self._llm_client:
            system_parts = [_SYSTEM_GUARDRAILS]
            if context:
                system_parts.append(f"Context:\n{context}")
            system = "\n\n".join(system_parts)
            try:
                if hasattr(self._llm_client, "complete_stream"):
                    return await self._llm_client.complete_stream(
                        question,
                        system=system,
                        temperature=temperature,
                        on_delta=on_delta,
                    )
                completion = await self._llm_client.complete(
                    question,
                    system=system,
                    temperature=temperature,
                )
                if on_delta:
                    maybe = on_delta(completion)
                    if asyncio.iscoroutine(maybe):
                        await maybe
                return completion
            except Exception:
                pass

        await asyncio.sleep(0.25 + random.random() * 0.2)
        base = f"First impression: {question[:80]}"
        if context:
            snippet = context.splitlines()[0][:80]
            base += f"\nContext echo: {snippet}"
        base += "\nKey images: connection, contrast, hidden assumptions."
        if on_delta:
            maybe = on_delta(base)
            if asyncio.iscoroutine(maybe):
                await maybe
        return base

    async def deepen(
        self,
        qid: str,
        question: str,
        partial_answer: str,
        shared_memory,
        *,
        temperature: float = 0.7,
        budget: str = "small",
        context: Optional[str] = None,
        psychoid_projection: Optional[Dict[str, object]] = None,
    ) -> Dict[str, str]:
        if self._llm_client:
            system_parts = []
            system_parts.append(_SYSTEM_GUARDRAILS)
            system_parts.append(_RIGHT_DEEPEN_GUARDRAILS)
            if context:
                system_parts.append(f"Context:\n{context}")
            if psychoid_projection:
                system_parts.append(f"Psychoid projection: {psychoid_projection}")
            system = "\n\n".join(system_parts) if system_parts else None
            try:
                completion = await self._llm_client.complete(
                    (
                        f"{question}\n\n"
                        "Draft (for reference only):\n"
                        f"{partial_answer}\n\n"
                        "Return only user-facing additive content (no coaching, no critique)."
                    ),
                    system=system,
                    temperature=temperature,
                )
                return {"qid": qid, "notes_sum": completion, "confidence_r": 0.9}
            except Exception:
                pass

        await asyncio.sleep(0.35 + random.random()*0.3)
        context_text = context if context is not None else shared_memory.retrieve_related(question)
        snippet = context_text[:120]
        detail = (
            f"Deeper take: {question}\n"
            f"- What matters: clarify definitions and the user's goal.\n"
            f"- Hidden assumptions: check what's being held constant.\n"
            f"- Context hint: {snippet}"
        )
        return {"qid": qid, "notes_sum": detail, "confidence_r": 0.85}

    async def criticise_reasoning(
        self,
        qid: str,
        question: str,
        draft: str,
        *,
        temperature: float = 0.2,
        context: Optional[str] = None,
        psychoid_projection: Optional[Dict[str, object]] = None,
    ) -> Dict[str, Any]:
        """Return a structured critique for System2-style reasoning corrections."""

        if self._llm_client:
            system_parts = [_SYSTEM_GUARDRAILS, _RIGHT_CRITIC_GUARDRAILS]
            if context:
                system_parts.append(f"Context:\n{context}")
            if psychoid_projection:
                system_parts.append(f"Psychoid projection: {psychoid_projection}")
            system = "\n\n".join(system_parts)
            prompt = (
                "User question:\n"
                f"{question}\n\n"
                "Draft answer (review for correctness):\n"
                f"{draft}\n\n"
                "Return JSON only."
            )
            try:
                completion = await self._llm_client.complete(
                    prompt,
                    system=system,
                    temperature=temperature,
                )
            except Exception:
                completion = ""

            parsed = _extract_json_object(completion) or {}
            verdict = str(parsed.get("verdict") or "").strip().lower()
            if verdict not in {"ok", "issues"}:
                verdict = "issues" if completion.strip() else "issues"

            issues_raw = parsed.get("issues")
            fixes_raw = parsed.get("fixes")
            issues = (
                [str(item).strip() for item in issues_raw if str(item).strip()]
                if isinstance(issues_raw, list)
                else []
            )
            fixes = (
                [str(item).strip() for item in fixes_raw if str(item).strip()]
                if isinstance(fixes_raw, list)
                else []
            )

            # Produce a compact internal summary string for the integrator.
            lines: list[str] = []
            if issues:
                lines.append("Issues:")
                for item in issues[:10]:
                    lines.append(f"- {item}")
            if fixes:
                lines.append("Fixes:")
                for item in fixes[:10]:
                    lines.append(f"- {item}")
            critic_sum = "\n".join(lines).strip()
            if verdict == "ok" and not critic_sum:
                critic_sum = "No issues detected."
            return {
                "qid": qid,
                "verdict": verdict,
                "issues": issues,
                "fixes": fixes,
                "critic_sum": critic_sum,
                "confidence_r": 0.9,
            }

        # Heuristic fallback (no external LLM available).
        await asyncio.sleep(0.15 + random.random() * 0.15)
        return {
            "qid": qid,
            "verdict": "issues",
            "issues": ["(fallback) External critic model unavailable; unable to verify reasoning."],
            "fixes": ["Consider enabling an external LLM for System2 critic mode."],
            "critic_sum": "External critic model unavailable.",
            "confidence_r": 0.35,
        }
