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

"""Basal ganglia inspired gating heuristics for cooperative action selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass
class BasalGangliaSignal:
    """Summary of the action selection signal."""

    go_probability: float
    inhibition: float
    dopamine_level: float
    recommended_action: Optional[int] = None
    note: str = ""

    def to_dict(self) -> Dict[str, float | int | str | None]:
        return {
            "go_probability": float(self.go_probability),
            "inhibition": float(self.inhibition),
            "dopamine_level": float(self.dopamine_level),
            "recommended_action": self.recommended_action,
            "note": self.note,
        }


class BasalGanglia:
    """Lightweight dynamical system approximating basal ganglia gating."""

    def __init__(
        self,
        *,
        baseline_dopamine: float = 0.45,
        inertia: float = 0.85,
        novelty_weight: float = 0.22,
        risk_weight: float = 0.27,
    ) -> None:
        self.baseline_dopamine = _clamp(baseline_dopamine)
        self.inertia = _clamp(inertia, 0.4, 0.99)
        self.novelty_weight = novelty_weight
        self.risk_weight = risk_weight
        self.dopamine_level = self.baseline_dopamine
        self.last_signal: Optional[BasalGangliaSignal] = None

    def evaluate(
        self,
        *,
        state: Dict[str, float],
        affect: Dict[str, float],
        focus_metric: float,
    ) -> BasalGangliaSignal:
        """Combine novelty, affect, and focus to shape action tendencies."""

        novelty = float(state.get("novelty", 0.0))
        consult_bias = float(state.get("consult_bias", 0.0))
        risk = float(affect.get("risk", 0.0))
        focus_metric = _clamp(float(focus_metric))

        go_drive = (
            0.35 * self.dopamine_level
            + self.novelty_weight * novelty
            + 0.18 * consult_bias
            + 0.12 * focus_metric
            - self.risk_weight * risk
        )
        inhibition = (
            0.25
            + 0.32 * risk
            - 0.18 * self.dopamine_level
            - 0.15 * focus_metric
        )
        go_probability = _clamp(0.35 + go_drive)
        inhibition = _clamp(inhibition)

        recommended_action: Optional[int] = None
        note = ""
        if go_probability - inhibition >= 0.18 and (novelty + risk) >= 0.8:
            recommended_action = 1
            note = "stimulate_consult"
        elif inhibition - go_probability >= 0.2 and focus_metric >= 0.6 and risk < 0.4:
            recommended_action = 0
            note = "stabilise"

        signal = BasalGangliaSignal(
            go_probability=go_probability,
            inhibition=inhibition,
            dopamine_level=self.dopamine_level,
            recommended_action=recommended_action,
            note=note,
        )
        self.last_signal = signal
        return signal

    def integrate_feedback(self, *, reward: float, latency_ms: float) -> None:
        """Update internal dopamine tone based on reward/latency feedback."""

        reward = _clamp(reward)
        latency_norm = _clamp(latency_ms / 8000.0)
        target = _clamp(0.45 + 0.4 * (reward - 0.5) - 0.2 * latency_norm)
        self.dopamine_level = (
            self.inertia * self.dopamine_level + (1.0 - self.inertia) * target
        )

    def tags(self, signal: BasalGangliaSignal) -> Iterable[str]:
        if signal.go_probability >= 0.6:
            yield "basal_go_high"
        elif signal.go_probability <= 0.3:
            yield "basal_go_low"
        if signal.inhibition >= 0.6:
            yield "basal_inhibit_high"
        if signal.note:
            yield f"basal_{signal.note}"

