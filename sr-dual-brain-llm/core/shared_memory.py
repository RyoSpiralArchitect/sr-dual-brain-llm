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

from typing import List, Tuple, Dict, Any

def _tokenize(text: str) -> List[str]:
    """Lightweight tokeniser used for novelty scoring."""
    return [tok for tok in text.replace("\n", " ").split(" ") if tok]

class SharedMemory:
    def __init__(self):
        self.past_qas: List[Tuple[str, str]] = []
        self.kv: Dict[str, Any] = {}

    def store(self, qa_pair: Dict[str, str]):
        self.past_qas.append((qa_pair["Q"], qa_pair["A"]))

    def get_context(self, n=5):
        return "\n".join([f"Q:{q} A:{a}" for q,a in self.past_qas[-n:]])

    def retrieve_related(self, question: str, n=3):
        hits = [ (q,a) for (q,a) in self.past_qas if question in q or question in a ]
        if not hits:
            return self.get_context(n)
        return "\n".join([f"Q:{q} A:{a}" for q,a in hits[:n]])

    def put_kv(self, key: str, value: Any):
        self.kv[key] = value

    def get_kv(self, key: str, default=None):
        return self.kv.get(key, default)

    def novelty_score(self, question: str) -> float:
        """Return a score in ``[0, 1]`` indicating how novel a question is."""
        if not self.past_qas:
            return 1.0
        q_tokens = set(_tokenize(question))
        if not q_tokens:
            return 1.0
        highest_overlap = 0.0
        for past_q, _ in self.past_qas:
            past_tokens = set(_tokenize(past_q))
            if not past_tokens:
                continue
            intersection = len(q_tokens & past_tokens)
            union = len(q_tokens | past_tokens)
            if union:
                highest_overlap = max(highest_overlap, intersection / union)
        return max(0.0, 1.0 - highest_overlap)
