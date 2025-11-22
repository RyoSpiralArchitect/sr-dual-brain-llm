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

import os
import asyncio

from core.shared_memory import SharedMemory
from core.models import LeftBrainModel, RightBrainModel
from core.policy import RightBrainPolicy
from core.orchestrator import Orchestrator
from core.auditor import Auditor
from core.hypothalamus import Hypothalamus
from core.policy_modes import ReasoningDial
from core.dual_brain import DualBrainController
from core.policy_selector import decide_leading_brain

_BACKEND = os.environ.get("CALLOSUM_BACKEND", "memory")

def load_callosum(name: str):
    if name == "memory":
        from core.callosum import Callosum as Impl; return Impl()
    if name == "kafka":
        from core.callosum_kafka import CallosumKafka as Impl; return Impl()
    if name == "mqtt":
        from core.callosum_mqtt import CallosumMQTT as Impl; return Impl()
    raise ValueError(f"Unknown backend: {name}")

async def right_worker(callosum, mem, right_model):
    if hasattr(callosum, "recv_request"):
        while True:
            req = await callosum.recv_request()
            if req.get("type") in {None, "ASK_DETAIL", "RIGHT_LEAD"}:
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
    else:
        await asyncio.sleep(0.1)

async def main():
    callosum = load_callosum(_BACKEND)
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
    if _BACKEND == "memory":
        asyncio.create_task(right_worker(callosum, mem, right))
    print(f"Server ready (backend={_BACKEND}). Type questions (stdin). Ctrl-C to quit.")
    loop = asyncio.get_event_loop()
    while True:
        question = await loop.run_in_executor(None, input, "Q> ")
        if not question.strip():
            continue
        context_snapshot = {
            "memory": mem.get_context(3),
            "novelty": mem.novelty_score(question),
            "recent_lead": mem.get_kv("last_leading_brain"),
        }
        leading_brain = decide_leading_brain(question, context_snapshot)
        ans = await controller.process(
            question,
            leading_brain=leading_brain,
            context_snapshot=context_snapshot,
        )
        print("A>", ans)

if __name__ == "__main__":
    asyncio.run(main())
