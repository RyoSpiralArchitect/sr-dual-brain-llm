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

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.shared_memory import SharedMemory
from core.callosum import Callosum
from core.models import LeftBrainModel, RightBrainModel
from core.policy import RightBrainPolicy
from core.policy_selector import decide_leading_brain
from core.orchestrator import Orchestrator
from core.auditor import Auditor
from core.hypothalamus import Hypothalamus
from core.policy_modes import ReasoningDial
from core.dual_brain import DualBrainController

async def right_worker(callosum, mem, right_model):
    if hasattr(callosum, "recv_request"):
        while True:
            req = await callosum.recv_request()
            if req.get("type") == "ASK_DETAIL":
                qid = req["qid"]
                try:
                    detail = await right_model.deepen(
                        qid,
                        req["question"],
                        req.get("draft_sum", ""),
                        mem,
                        temperature=req.get("temperature", 0.7),
                        budget=req.get("budget", "small"),
                        context=req.get("context"),
                    )
                    await callosum.publish_response(qid, {"qid": qid, "notes_sum": detail["notes_sum"], "confidence_r": detail["confidence_r"]})
                except Exception as e:
                    await callosum.publish_response(qid, {"qid": qid, "error": str(e)})
            elif req.get("type") == "ASK_LEAD":
                qid = req.get("qid")
                try:
                    impression = await right_model.generate_lead(
                        req.get("question", ""),
                        req.get("context", ""),
                        temperature=float(req.get("temperature", 0.85)),
                    )
                except Exception as e:
                    await callosum.publish_response(qid, {"qid": qid, "error": str(e)})
                else:
                    await callosum.publish_response(
                        qid,
                        {
                            "qid": qid,
                            "lead_notes": impression,
                        },
                    )
    else:
        await asyncio.sleep(0.1)

async def main():
    callosum = Callosum()
    mem = SharedMemory()
    left = LeftBrainModel()
    right = RightBrainModel()
    policy = RightBrainPolicy()
    orchestrator = Orchestrator(2)
    auditor = Auditor()
    hypothalamus = Hypothalamus()
    dial_mode = os.environ.get("REASONING_DIAL", "evaluative")
    try:
        dial = ReasoningDial(mode=dial_mode)
    except AssertionError:
        print(f"Unknown reasoning dial '{dial_mode}', falling back to evaluative mode.")
        dial = ReasoningDial(mode="evaluative")
    controller = DualBrainController(
        callosum=callosum,
        memory=mem,
        left_model=left,
        right_model=right,
        policy=policy,
        hypothalamus=hypothalamus,
        reasoning_dial=dial,
        auditor=auditor,
        orchestrator=orchestrator,
    )
    asyncio.create_task(right_worker(callosum, mem, right))
    print("Server ready (callosum=memory). Type questions (stdin). Ctrl-C to quit.")
    loop = asyncio.get_running_loop()
    while True:
        question = await loop.run_in_executor(None, input, "Q> ")
        if not question.strip():
            continue
        context_hints = {
            "novelty": mem.novelty_score(question),
            "last_leading": mem.get_kv("last_leading_brain"),
        }
        leading_brain = decide_leading_brain(question, context_hints)
        mem.put_kv("last_leading_brain", leading_brain)
        ans = await controller.process(question, leading_brain=leading_brain)
        print("A>", ans)

if __name__ == "__main__":
    asyncio.run(main())
