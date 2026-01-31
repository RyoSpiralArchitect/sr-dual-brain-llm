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

import asyncio, random
from typing import Dict, Optional

from .llm_client import LLMClient, LLMConfig, load_llm_config

_SYSTEM_GUARDRAILS = (
    "Answer the user's question directly.\n"
    "Do not mention internal orchestration (e.g., left/right brain, corpus callosum, telemetry, architecture paths, qid).\n"
    "Do not add bracketed debug headings or meta commentary about the system."
)


class LeftBrainModel:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config or load_llm_config("LEFT_BRAIN") or load_llm_config()
        self._llm_client = LLMClient(self.llm_config) if self.llm_config else None
        self.uses_external_llm = bool(self._llm_client)

    async def generate_answer(self, input_text: str, context: str) -> str:
        """Produce a first-pass draft that reflects retrieved memory snippets."""
        if self._llm_client:
            system_parts = [_SYSTEM_GUARDRAILS]
            if context:
                system_parts.append(f"Context:\n{context}")
            system = "\n\n".join(system_parts)
            try:
                return await self._llm_client.complete(input_text, system=system, temperature=0.4)
            except Exception:
                pass

        base = f"Draft answer for: {input_text[:80]}"
        if context:
            summary = context.splitlines()[0][:80]
            base += f"\nContext hint: {summary}"
        if any(k in input_text for k in ["詳しく","分析","計算","証拠"]):
            base += " ... (brief, looks complex)"
        return base

    def estimate_confidence(self, draft: str) -> float:
        return 0.45 if "..." in draft else 0.9

    def integrate_info(self, draft: str, info: str) -> str:
        if not info:
            return draft
        return f"{draft}\n\n{info}"

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
    ) -> str:
        """Produce an imagistic first impression before the left brain speaks."""
        if self._llm_client:
            system_parts = [_SYSTEM_GUARDRAILS]
            if context:
                system_parts.append(f"Context:\n{context}")
            system = "\n\n".join(system_parts)
            try:
                return await self._llm_client.complete(
                    question,
                    system=system,
                    temperature=temperature,
                )
            except Exception:
                pass

        await asyncio.sleep(0.25 + random.random() * 0.2)
        base = f"First impression: {question[:80]}"
        if context:
            snippet = context.splitlines()[0][:80]
            base += f"\nContext echo: {snippet}"
        base += "\nKey images: connection, contrast, hidden assumptions."
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
            if context:
                system_parts.append(f"Context:\n{context}")
            if psychoid_projection:
                system_parts.append(f"Psychoid projection: {psychoid_projection}")
            system = "\n\n".join(system_parts) if system_parts else None
            try:
                completion = await self._llm_client.complete(
                    f"{question}\n\nDraft summary:\n{partial_answer}",
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
