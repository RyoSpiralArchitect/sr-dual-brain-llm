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

class LeftBrainModel:
    def __init__(self):
        pass

    async def generate_answer(self, input_text: str, context: str) -> str:
        """Produce a first-pass draft that reflects retrieved memory snippets."""
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
        return f"{draft}\n(Reference from RightBrain: {info})"

class RightBrainModel:
    def __init__(self):
        pass

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
    ) -> Dict[str, str]:
        await asyncio.sleep(0.35 + random.random()*0.3)
        context_text = context if context is not None else shared_memory.retrieve_related(question)
        snippet = context_text[:120]
        detail = (
            f"Deep analysis for qid={qid}: insights about '{question[:50]}'"
            f" | temp={temperature:.2f} | budget={budget} | ctx=[{snippet}]"
        )
        return {"qid": qid, "notes_sum": detail, "confidence_r": 0.85}
