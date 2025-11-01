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

import os, asyncio, uuid
from core.shared_memory import SharedMemory
from core.models import LeftBrainModel, RightBrainModel
from core.policy import RightBrainPolicy
from core.orchestrator import Orchestrator
from core.auditor import Auditor

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
            if req.get("type") == "ASK_DETAIL":
                qid = req["qid"]
                try:
                    detail = await right_model.deepen(qid, req["question"], req.get("draft_sum",""), mem)
                    await callosum.publish_response(qid, {"qid": qid, "notes_sum": detail["notes_sum"], "confidence_r": detail["confidence_r"]})
                except Exception as e:
                    await callosum.publish_response(qid, {"qid": qid, "error": str(e)})
    else:
        await asyncio.sleep(0.1)

async def handle_query(question: str, callosum, mem, left, policy, auditor, orchestrator):
    draft = await left.generate_answer(question, mem.get_context())
    conf = left.estimate_confidence(draft)
    state = {"left_conf": conf, "draft_len": len(draft), "novelty":0.0, "q_type": ("hard" if any(k in question for k in ['計算','分析','証拠']) else "medium")}
    action = policy.decide(state)
    qid = str(uuid.uuid4())
    if not orchestrator.register_request(qid):
        return "Loop-killed"
    final_answer = draft
    try:
        if action == 0:
            pass
        else:
            payload = {"type":"ASK_DETAIL","qid": qid, "question":question, "draft_sum": draft if len(draft)<160 else draft[:160]}
            resp = await callosum.ask_detail(payload, timeout_ms=5000)
            if resp.get("error"):
                final_answer = draft + "\n(右脳応答エラー)"
            else:
                final_answer = left.integrate_info(draft, resp["notes_sum"])
    except asyncio.TimeoutError:
        final_answer = draft + "\n(右脳タイムアウト: 後で詳細を返します)"
    finally:
        auditor_res = auditor.check(final_answer)
        if not auditor_res["ok"]:
            final_answer = draft + f"\n(Auditor veto: {auditor_res['reason']})"
        mem.store({"Q":question,"A":final_answer})
        orchestrator.clear(qid)
    return final_answer

async def main():
    callosum = load_callosum(_BACKEND)
    mem = SharedMemory(); left = LeftBrainModel(); right = RightBrainModel()
    policy = RightBrainPolicy(); orchestrator = Orchestrator(2); auditor = Auditor()
    if _BACKEND == "memory":
        asyncio.create_task(right_worker(callosum, mem, right))
    print(f"Server ready (backend={_BACKEND}). Type questions (stdin). Ctrl-C to quit.")
    loop = asyncio.get_event_loop()
    while True:
        question = await loop.run_in_executor(None, input, "Q> ")
        if not question.strip():
            continue
        ans = await handle_query(question, callosum, mem, left, policy, auditor, orchestrator)
        print("A>", ans)

if __name__ == "__main__":
    asyncio.run(main())
