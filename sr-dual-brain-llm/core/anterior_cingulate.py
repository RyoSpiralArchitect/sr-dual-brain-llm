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

"""Anterior cingulate cortex (ACC) inspired conflict monitoring.

This module provides a lightweight "conflict signal" that can be logged to
telemetry and used for gating additional control (e.g., cerebellar correction
or System2 engagement).

The implementation is intentionally conservative: it relies only on
high-confidence signals (currently the deterministic micro-critic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .micro_critic import MicroCriticResult


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


@dataclass(frozen=True)
class ConflictSignal:
    """Conflict signal produced by the ACC monitor."""

    conflict_level: float
    effort_level: float = 0.0
    uncertainty: float = 0.0
    adaptation_signal: float = 0.0
    recommended_control: str = "monitor"
    micro_domain: Optional[str] = None
    micro_confidence: float = 0.0
    micro_issues: int = 0
    sources: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "conflict_level": float(_clamp01(self.conflict_level)),
            "effort_level": float(_clamp01(self.effort_level)),
            "uncertainty": float(_clamp01(self.uncertainty)),
            "adaptation_signal": float(_clamp01(self.adaptation_signal)),
            "recommended_control": str(self.recommended_control or "monitor"),
            "sources": list(self.sources),
        }
        if self.micro_domain:
            payload["micro_domain"] = str(self.micro_domain)
            payload["micro_confidence"] = float(_clamp01(self.micro_confidence))
            payload["micro_issues"] = int(self.micro_issues)
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


class AnteriorCingulateCortex:
    """Conflict monitoring and control signal generator."""

    def __init__(self, *, overconfidence_threshold: float = 0.85) -> None:
        self.overconfidence_threshold = _clamp01(overconfidence_threshold)

    def monitor(
        self,
        *,
        question: str,
        draft: str,
        left_confidence: float,
        micro: Optional[MicroCriticResult] = None,
    ) -> ConflictSignal:
        """Generate a conflict signal from available error detectors.

        Args:
            question: User input (unused for now; kept for future heuristics).
            draft: Candidate answer text (unused for now; kept for future heuristics).
            left_confidence: Confidence estimate from the draft generator.
            micro: Optional deterministic micro-critic result.
        """

        _ = question

        lc = _clamp01(left_confidence)
        conflict = 0.0
        effort = 0.0
        uncertainty = 0.0
        sources: list[str] = []
        notes: list[str] = []
        q = str(question or "")
        q_len = len(q)
        draft_len = len(str(draft or ""))

        if micro is not None:
            sources.append("micro")
            issue_count = int(len(micro.issues or []))
            micro_conf = _clamp01(micro.confidence_r)
            if micro.verdict == "issues" and issue_count > 0:
                # Micro-critic issues are high-confidence; scale by:
                # - its own confidence,
                # - issue count (bounded),
                # - and a small "overconfidence penalty" to mimic ACC conflict.
                base = 0.3 + 0.45 * micro_conf + 0.08 * min(issue_count, 6)
                conflict = max(conflict, _clamp01(base))
                effort = max(
                    effort,
                    _clamp01(
                        0.18
                        + 0.12 * min(issue_count, 6)
                        + 0.08 * min(1.0, q_len / 220.0)
                        + 0.06 * min(1.0, draft_len / 320.0)
                    ),
                )
                uncertainty = max(
                    uncertainty,
                    _clamp01(max(0.0, lc - micro_conf) + 0.08 * min(issue_count, 4)),
                )
                notes.append(f"micro:{micro.domain}:{issue_count}")
                if lc >= self.overconfidence_threshold:
                    conflict = _clamp01(conflict + 0.08)
                    notes.append("overconfidence")
            elif micro.verdict == "ok":
                # Ongoing monitoring baseline.
                conflict = max(conflict, 0.02)
                effort = max(effort, 0.05)
                uncertainty = max(uncertainty, _clamp01(max(0.0, lc - micro_conf) * 0.4))

        if not sources:
            sources.append("heuristic")
            effort = _clamp01(0.06 + 0.10 * min(1.0, q_len / 260.0))

        adaptation_signal = _clamp01(
            0.58 * conflict + 0.24 * effort + 0.18 * uncertainty
        )
        recommended_control = "monitor"
        if conflict >= 0.72 and (micro is None or len(micro.issues or []) >= 2):
            recommended_control = "system2"
        elif conflict >= 0.52 or adaptation_signal >= 0.58:
            recommended_control = "consult"
        elif conflict <= 0.12 and lc >= 0.75:
            recommended_control = "stabilise"

        return ConflictSignal(
            conflict_level=_clamp01(conflict),
            effort_level=_clamp01(effort),
            uncertainty=_clamp01(uncertainty),
            adaptation_signal=_clamp01(adaptation_signal),
            recommended_control=recommended_control,
            micro_domain=(micro.domain if micro is not None else None),
            micro_confidence=(_clamp01(micro.confidence_r) if micro is not None else 0.0),
            micro_issues=(int(len(micro.issues or [])) if micro is not None else 0),
            sources=tuple(sources),
            notes=tuple(notes),
        )
