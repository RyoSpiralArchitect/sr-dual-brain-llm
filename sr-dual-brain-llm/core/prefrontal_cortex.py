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

"""Simplified dorsolateral prefrontal cortex analogue for control signals."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable


def _tokenize(text: str) -> Iterable[str]:
    return [tok for tok in text.replace("\n", " ").lower().split(" ") if tok]


def _lexical_overlap(a: str, b: str) -> float:
    a_tokens = set(_tokenize(a))
    b_tokens = set(_tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return intersection / union if union else 0.0


class PrefrontalCortex:
    """Track working-memory goals and provide control gating like the DLPFC."""

    def __init__(
        self,
        *,
        working_memory_horizon: int = 6,
        decay: float = 0.82,
        conflict_threshold: float = 0.45,
    ) -> None:
        self.decay = decay
        self.conflict_threshold = conflict_threshold
        self.working_memory: Deque[Dict[str, float]] = deque(maxlen=working_memory_horizon)
        self.control_tone = 0.6

    def update_goal_state(
        self,
        question: str,
        draft: str,
        context: str,
        *,
        amygdala_signal: Dict[str, float] | None = None,
        novelty: float = 0.0,
    ) -> Dict[str, float]:
        """Refresh working memory and return control metrics."""

        amygdala_signal = amygdala_signal or {}
        overlap = _lexical_overlap(question, draft)
        lexical_conflict = max(0.0, 1.0 - overlap)
        risk = float(amygdala_signal.get("risk", 0.0))
        conflict = max(0.0, min(1.0, 0.65 * lexical_conflict + 0.35 * risk))
        arousal = float(amygdala_signal.get("arousal", 0.0))
        goal_focus = max(0.0, min(1.0, 0.4 + 0.3 * novelty + 0.3 * arousal))

        self.working_memory.append(
            {"conflict": conflict, "risk": risk, "focus": goal_focus}
        )
        self.control_tone = self.decay * self.control_tone + (1 - self.decay) * (1.0 - conflict)

        return {
            "goal_focus": goal_focus,
            "conflict": conflict,
            "control_tone": self.control_tone,
        }

    def gate_consult(self, proposed_action: int, state: Dict[str, float]) -> int:
        """Override policy decisions when conflict or risk is high."""

        conflict = float(state.get("conflict_signal", 0.0))
        risk = float(state.get("amygdala_risk", 0.0))
        novelty = float(state.get("novelty", 0.0))
        left_conf = float(state.get("left_conf", 0.5))

        if conflict >= self.conflict_threshold or risk >= 0.5:
            return max(proposed_action, 1)

        if left_conf > 0.85 and novelty < 0.2 and conflict < 0.25:
            return 0

        return proposed_action

    def modulate_temperature(self, temperature: float, state: Dict[str, float]) -> float:
        conflict = float(state.get("conflict_signal", 0.0))
        novelty = float(state.get("novelty", 0.0))
        arousal = float(state.get("amygdala_arousal", 0.0))
        adjustment = 0.15 * conflict + 0.1 * novelty + 0.1 * arousal - 0.1 * self.control_tone
        temp = temperature + adjustment
        return max(0.25, min(0.95, temp))

    def integrate_feedback(self, reward: float, latency_ms: float, *, success: bool) -> None:
        penalty = 0.1 if not success else 0.0
        score = max(0.0, min(1.0, reward - penalty))
        self.control_tone = self.decay * self.control_tone + (1 - self.decay) * score
