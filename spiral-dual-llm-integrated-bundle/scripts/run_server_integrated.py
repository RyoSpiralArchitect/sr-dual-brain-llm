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

import asyncio, uuid
from typing import Dict, Any
from core.events import new_event, EventTracer
from core.policy_modes import ReasoningDial
from core.hypothalamus import Hypothalamus
from core.amygdala import Amygdala
from core.temporal_hippocampal_indexing import TemporalHippocampalIndexing

class SharedMemory:
    def __init__(self): self.pairs = []
    def get_context(self):
        return "\n".join([f"Q:{q} A:{a}" for q,a in self.pairs][-5:])
    def store(self, q, a): self.pairs.append((q,a))
    def retrieve_related(self, q): return self.get_context()

class Callosum:
    def __init__(self, tracer, episode_id):
        self.q = asyncio.Queue(); self.slot_ms = 250
        self.tracer = tracer; self.episode_id = episode_id; self.step = 0
    async def ask_detail(self, qid: str, payload: Dict[str,Any]):
        self.step += 1
        self.tracer.log(new_event("CALL_SEND", self.episode_id, self.step, qid, payload))
        await self.q.put(payload)
        await asyncio.sleep(self.slot_ms/1000.0)
        ans = await self.q.get()
        self.tracer.log(new_event("CALL_RECV", self.episode_id, self.step, qid, {"detail_len": len(ans.get("notes_sum",""))}))
        return ans

class LeftBrain:
    async def draft(self, q, ctx):
        draft = f"[LeftDraft] {q[:40]} ..."
        conf = 0.6 if ("詳しく" in q or "なぜ" in q) else 0.85
        stats = {"draft_len": len(draft)}
        await asyncio.sleep(0.05); return draft, conf, stats

class RightBrain:
    async def deepen(self, q, draft, mem, budget="small"):
        ctx = mem.retrieve_related(q)
        notes = f"[RightDetail/{budget}] key-points for: {q[:32]}"
        await asyncio.sleep(0.15 if budget=='small' else 0.35)
        return {"notes_sum": notes, "confidence_r": 0.75, "cost": {"tokens": len(notes), "ms": 150 if budget=='small' else 350}}

class Auditor: async def check(self, txt): return {"ok": True}

async def handle_query(qid, question, tracer, dial_mode="evaluative"):
    episode_id = str(uuid.uuid4())
    mem = SharedMemory(); hypo = Hypothalamus(); amg = Amygdala(); thi = TemporalHippocampalIndexing()
    dial = ReasoningDial(mode=dial_mode); callo = Callosum(tracer, episode_id)
    left = LeftBrain(); right = RightBrain(); audit = Auditor()
    epi_ctx = thi.retrieve_summary(question) or mem.get_context()
    draft, conf, stats = await left.draft(question, epi_ctx)
    am = amg.analyze(question + "\n" + draft)
    temp = hypo.recommend_temperature(conf)
    state = {"left_conf": conf, "draft_len": stats["draft_len"], "novelty": 1.0 if not epi_ctx else 0.2, "risk": am.get("risk",0.0), "temp": temp, "q_type": "hard" if ("計算" in question or "なぜ" in question) else "medium"}
    action = 0 if conf >= 0.8 and am.get("risk",0.0) < 0.66 else 1
    action = dial.adjust_decision(state, action)
    temp = dial.scale_temperature(temp); budget = dial.pick_budget()
    callo.slot_ms = hypo.recommend_slot_ms(am.get("risk",0.0))
    tracer.log(new_event("POLICY_DECISION", episode_id, 1, qid, {"state": state, "action": action, "reward": 0.0, "logp": 0.0}))
    detail = {}
    if action != 0:
        detail = await callo.ask_detail(qid, {"type":"ASK_DETAIL","question":question,"draft_sum":draft[:120], "budget": budget})
        # Right brain replies by putting a DETAIL back (simulate immediately)
        await callo.q.put(await right.deepen(question, draft, mem, budget=budget))
    final = draft if action == 0 else f"{draft}\n（参考: {detail.get('notes_sum','')}）"
    await audit.check(final)
    thi.index_episode(qid, question, final); mem.store(question, final)
    hypo.update_feedback(reward=0.7 if action!=0 else 0.5, latency_ms=detail.get("cost",{}).get("ms",0))
    return {"final": final, "state": state, "action": action, "slot_ms": callo.slot_ms}

async def main():
    tracer = EventTracer("traces/session.jsonl")
    qs = [("q1","このデータセットの統計分析結果を詳しく説明してください。"),
          ("q2","パスワードやAPIキーの扱いはどうすべき？"),
          ("q3","なぜ右脳を呼ぶ必要があるの？簡潔に。")]
    for qid, q in qs:
        res = await handle_query(qid, q, tracer, dial_mode="evaluative")
        print(f"[{qid}] slot={res['slot_ms']} action={res['action']}\n{res['final'][:120]}...")
    tracer.close()
if __name__ == "__main__": asyncio.run(main())
