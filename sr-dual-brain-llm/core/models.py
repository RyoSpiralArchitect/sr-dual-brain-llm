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
from typing import Dict

class LeftBrainModel:
    def __init__(self):
        pass

    async def generate_answer(self, input_text: str, context: str) -> str:
        base = f"Draft answer for: {input_text[:80]}"
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

    async def deepen(self, qid: str, question: str, partial_answer: str, shared_memory) -> Dict[str, str]:
        await asyncio.sleep(0.5 + random.random()*0.5)
        context = shared_memory.retrieve_related(question)
        detail = f"Deep analysis for qid={qid}: insights about '{question[:50]}' | ctx=[{context[:120]}]"
        return {"qid": qid, "notes_sum": detail, "confidence_r": 0.85}
