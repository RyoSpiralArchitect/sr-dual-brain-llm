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

"""Cerebellum-inspired fast micro-correction layer.

The cerebellum is often modeled as providing fast, learned, corrective updates.
In this codebase we use it as a small, optional layer that converts high-
confidence micro-critic findings into a minimal revision instruction block for
the integrator.
"""

from __future__ import annotations

from .micro_critic import MicroCriticResult


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


class Cerebellum:
    def __init__(self, *, min_confidence: float = 0.88, max_issues: int = 4) -> None:
        self.min_confidence = _clamp01(min_confidence)
        self.max_issues = int(max_issues)

    def should_apply(self, micro: MicroCriticResult) -> bool:
        if micro is None:
            return False
        if str(micro.verdict or "").strip().lower() != "issues":
            return False
        if _clamp01(micro.confidence_r) < float(self.min_confidence):
            return False
        if len(micro.issues or []) > int(self.max_issues):
            return False
        return True

    def build_internal_notes(self, micro: MicroCriticResult) -> str:
        blob = str(micro.critic_sum or "").strip()
        header = (
            "Cerebellar micro-correction (internal; do not output directly). "
            "Fix the high-confidence numeric issue(s) flagged below. "
            "Apply minimal precise edits and preserve the original voice.\n"
        )
        if not blob:
            return header.strip()
        return header + blob

