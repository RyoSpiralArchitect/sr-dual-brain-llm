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

from typing import Dict


class Orchestrator:
    def __init__(self, max_roundtrips=2):
        self.max_roundtrips = max_roundtrips
        self.active_roundtrips: Dict[str, Dict[str, object]] = {}

    def register_request(self, qid: str, *, leading_brain: str | None = None) -> bool:
        entry = self.active_roundtrips.get(qid)
        if entry is None:
            entry = {"count": 0, "leader": None}
            self.active_roundtrips[qid] = entry
        entry["count"] = int(entry.get("count", 0)) + 1
        if leading_brain and not entry.get("leader"):
            entry["leader"] = leading_brain
        return int(entry["count"]) <= self.max_roundtrips

    def leader(self, qid: str) -> str | None:
        entry = self.active_roundtrips.get(qid)
        if not entry:
            return None
        leader = entry.get("leader")
        return str(leader) if leader else None

    def clear(self, qid: str):
        if qid in self.active_roundtrips:
            del self.active_roundtrips[qid]
