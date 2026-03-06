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

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .anterior_cingulate import ConflictSignal
from .micro_critic import MicroCriticResult


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


@dataclass(frozen=True)
class CerebellarForecast:
    predicted_gain: float
    residual_risk: float
    recommended_path: str
    confidence: float
    notes: tuple[str, ...] = ()

    def to_payload(self) -> Dict[str, Any]:
        return {
            "predicted_gain": float(_clamp01(self.predicted_gain)),
            "residual_risk": float(_clamp01(self.residual_risk)),
            "recommended_path": str(self.recommended_path or "observe"),
            "confidence": float(_clamp01(self.confidence)),
            "notes": list(self.notes),
        }


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

    def forecast(
        self,
        micro: MicroCriticResult,
        *,
        conflict: Optional[ConflictSignal] = None,
        system2_active: bool = False,
        consult_planned: bool = False,
    ) -> CerebellarForecast:
        issue_count = int(len(micro.issues or []))
        micro_conf = _clamp01(micro.confidence_r)
        conflict_level = _clamp01(
            conflict.conflict_level if conflict is not None else 0.0
        )
        deterministic_bonus = 0.18 if str(micro.domain or "") in {
            "arithmetic",
            "linear_equation",
            "probability",
            "unit_conversion",
        } else 0.0
        predicted_gain = _clamp01(
            0.46 * micro_conf
            + deterministic_bonus
            + 0.14 * max(0.0, 1.0 - min(issue_count, 5) / 5.0)
            - 0.18 * conflict_level
            - (0.12 if system2_active else 0.0)
            - (0.10 if consult_planned else 0.0)
        )
        residual_risk = _clamp01(
            0.48 * conflict_level
            + 0.12 * min(issue_count, 5)
            + 0.12 * max(0.0, 1.0 - micro_conf)
            - 0.35 * predicted_gain
        )
        recommended_path = "observe"
        if system2_active or consult_planned:
            recommended_path = "defer"
        elif predicted_gain >= 0.55 and residual_risk <= 0.55:
            recommended_path = "micro_correct"
        elif residual_risk >= 0.68:
            recommended_path = "consult"
        notes = [f"domain:{micro.domain}", f"issues:{issue_count}"]
        if conflict is not None:
            notes.append(f"conflict:{conflict_level:.2f}")
        return CerebellarForecast(
            predicted_gain=predicted_gain,
            residual_risk=residual_risk,
            recommended_path=recommended_path,
            confidence=micro_conf,
            notes=tuple(notes),
        )

    def build_internal_notes(
        self,
        micro: MicroCriticResult,
        *,
        forecast: Optional[CerebellarForecast] = None,
    ) -> str:
        blob = str(micro.critic_sum or "").strip()
        header = (
            "Cerebellar micro-correction (internal; do not output directly). "
            "Fix the high-confidence numeric issue(s) flagged below. "
            "Apply minimal precise edits and preserve the original voice.\n"
        )
        if forecast is not None:
            header += (
                f"[forward-model] predicted_gain={forecast.predicted_gain:.2f} "
                f"residual_risk={forecast.residual_risk:.2f} "
                f"path={forecast.recommended_path}\n"
            )
        if not blob:
            return header.strip()
        return header + blob
