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

import json, time, uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

EVENT_TYPES = ["CALL_SEND","CALL_RECV","ORCH_ASSIGN","ORCH_DONE","POLICY_DECISION","AUDIT_RESULT","ERROR","HEARTBEAT"]

@dataclass
class Event:
    type: str; ts: float; episode_id: str; step: int; qid: str; payload: Dict[str, Any]
    def to_json(self) -> str: return json.dumps(asdict(self), ensure_ascii=False)

class EventTracer:
    def __init__(self, path: str):
        self.path = path; self._fh = open(self.path, "a", encoding="utf-8")
    def log(self, ev: 'Event') -> None:
        self._fh.write(ev.to_json() + "\n"); self._fh.flush()
    def log_dict(self, ev_dict: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(ev_dict, ensure_ascii=False) + "\n"); self._fh.flush()
    def close(self):
        try: self._fh.close()
        except Exception: pass

def new_event(type: str, episode_id: Optional[str] = None, step: int = 0, qid: str = "", payload: Optional[Dict[str,Any]] = None) -> Event:
    if type not in EVENT_TYPES: raise ValueError("Unknown event type")
    return Event(type=type, ts=time.time(), episode_id=episode_id or str(uuid.uuid4()), step=step, qid=qid, payload=payload or {})

class Timeline:
    def __init__(self, path: str):
        self.path = path; self.events = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: self.events.append(json.loads(line))
                except Exception: continue
        self.events.sort(key=lambda e: e.get("ts", 0.0))
    def __iter__(self):
        for ev in self.events: yield ev
    def replay(self, speed: float = 1.0):
        if not self.events: return
        t0 = self.events[0]["ts"]
        import time as _t; start = _t.time()
        for ev in self.events:
            delay = (ev["ts"] - t0) / max(1e-6, speed)
            now = _t.time(); remain = delay - (now - start)
            if remain > 0: _t.sleep(remain)
            yield ev
