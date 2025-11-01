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

import time, numpy as np
def _tokenize(text: str):
    return [t for t in text.replace("\n"," ").replace("\t"," ").split(" ") if t]
class TemporalHippocampalIndexing:
    def __init__(self, dim: int = 128):
        self.dim = dim; self.episodes = []; self._eps = 1e-8
    def _embed(self, text: str):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text.lower()):
            v[hash(tok) % self.dim] += 1.0
        n = float(np.linalg.norm(v) + self._eps); return v / n
    def index_episode(self, qid: str, question: str, answer: str):
        payload = f"Q: {question}\nA: {answer}"
        vec = self._embed(payload)
        self.episodes.append({"ts": time.time(), "qid": qid, "text": payload, "vec": vec})
    def retrieve(self, query: str, topk: int = 3):
        if not self.episodes: return []
        qv = self._embed(query)
        scored = [(float(np.dot(qv, ep["vec"])), ep) for ep in self.episodes]
        scored.sort(key=lambda x: x[0], reverse=True); return scored[:topk]
    def retrieve_summary(self, query: str, topk: int = 3, max_chars: int = 240) -> str:
        hits = self.retrieve(query, topk=topk)
        if not hits: return ""
        parts = []
        for sim, ep in hits:
            s = f"(sim={sim:.2f}) " + ep["text"].replace("\n"," ")[:max_chars]
            parts.append(s)
        return " | ".join(parts)
