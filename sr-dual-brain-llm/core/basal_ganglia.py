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
    direct_pathway: float = 0.0
    indirect_pathway: float = 0.0
    hyperdirect_pathway: float = 0.0
    gating_balance: float = 0.0
    dominant_pathway: str = "balanced"
    recommended_action: Optional[int] = None
    note: str = ""

    def to_dict(self) -> Dict[str, float | int | str | None]:
        return {
            "go_probability": float(self.go_probability),
            "inhibition": float(self.inhibition),
            "dopamine_level": float(self.dopamine_level),
            "direct_pathway": float(self.direct_pathway),
            "indirect_pathway": float(self.indirect_pathway),
            "hyperdirect_pathway": float(self.hyperdirect_pathway),
            "gating_balance": float(self.gating_balance),
            "dominant_pathway": self.dominant_pathway,
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
        salience_weight: float = 0.14,
        conflict_weight: float = 0.24,
    ) -> None:
        self.baseline_dopamine = _clamp(baseline_dopamine)
        self.inertia = _clamp(inertia, 0.4, 0.99)
        self.novelty_weight = novelty_weight
        self.risk_weight = risk_weight
        self.salience_weight = salience_weight
        self.conflict_weight = conflict_weight
        self.dopamine_level = self.baseline_dopamine
        self.last_signal: Optional[BasalGangliaSignal] = None

    def evaluate(
        self,
        *,
        state: Dict[str, float],
        affect: Dict[str, float],
        focus_metric: float,
        conflict_level: float = 0.0,
        salience_level: float = 0.0,
    ) -> BasalGangliaSignal:
        """Combine novelty, affect, and focus to shape action tendencies."""

        novelty = float(state.get("novelty", 0.0))
        consult_bias = float(state.get("consult_bias", 0.0))
        risk = float(affect.get("risk", 0.0))
        focus_metric = _clamp(float(focus_metric))
        conflict_level = _clamp(float(conflict_level))
        salience_level = _clamp(float(salience_level))

        go_drive = (
            0.35 * self.dopamine_level
            + self.novelty_weight * novelty
            + 0.18 * consult_bias
            + 0.12 * focus_metric
            + self.salience_weight * salience_level
            - self.risk_weight * risk
            - 0.10 * conflict_level
        )
        inhibition = (
            0.25
            + 0.32 * risk
            + self.conflict_weight * conflict_level
            - 0.18 * self.dopamine_level
            - 0.15 * focus_metric
            - 0.06 * salience_level
        )
        go_probability = _clamp(0.35 + go_drive)
        inhibition = _clamp(inhibition)
        direct_pathway = _clamp(
            0.48 * go_probability
            + 0.22 * max(0.0, consult_bias)
            + 0.16 * salience_level
            + 0.14 * self.dopamine_level
        )
        indirect_pathway = _clamp(
            0.55 * inhibition
            + 0.18 * risk
            + 0.15 * max(0.0, -consult_bias)
            + 0.12 * max(0.0, 1.0 - focus_metric)
        )
        hyperdirect_pathway = _clamp(
            0.58 * conflict_level
            + 0.18 * risk
            + 0.12 * max(0.0, salience_level - focus_metric)
        )
        gating_balance = _clamp(
            0.5 + 0.5 * (direct_pathway - max(indirect_pathway, hyperdirect_pathway)),
        )

        recommended_action: Optional[int] = None
        note = ""
        dominant_pathway = "direct"
        if hyperdirect_pathway >= max(direct_pathway, indirect_pathway):
            dominant_pathway = "hyperdirect"
        elif indirect_pathway >= direct_pathway:
            dominant_pathway = "indirect"

        if (
            hyperdirect_pathway >= 0.55
            and conflict_level >= 0.5
            and go_probability >= 0.35
        ):
            recommended_action = 1
            note = "hyperdirect_consult"
        elif go_probability - inhibition >= 0.18 and (novelty + risk + salience_level) >= 0.8:
            recommended_action = 1
            note = "direct_go_consult"
        elif inhibition - go_probability >= 0.2 and focus_metric >= 0.6 and risk < 0.4:
            recommended_action = 0
            note = "indirect_hold"

        signal = BasalGangliaSignal(
            go_probability=go_probability,
            inhibition=inhibition,
            dopamine_level=self.dopamine_level,
            direct_pathway=direct_pathway,
            indirect_pathway=indirect_pathway,
            hyperdirect_pathway=hyperdirect_pathway,
            gating_balance=gating_balance,
            dominant_pathway=dominant_pathway,
            recommended_action=recommended_action,
            note=note,
        )
        self.last_signal = signal
        return signal

    def integrate_feedback(
        self,
        *,
        reward: float,
        latency_ms: float,
        conflict_resolved: bool = False,
        system2_used: bool = False,
    ) -> None:
        """Update internal dopamine tone based on reward/latency feedback."""

        reward = _clamp(reward)
        latency_norm = _clamp(latency_ms / 8000.0)
        target = _clamp(
            0.45
            + 0.4 * (reward - 0.5)
            - 0.2 * latency_norm
            + (0.08 if conflict_resolved else 0.0)
            - (0.04 if system2_used and reward < 0.55 else 0.0)
        )
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
