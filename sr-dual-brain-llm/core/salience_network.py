"""Salience-network inspired switching between control regimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .insula import InteroceptiveState


def _clamp01(value: float) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class SalienceSignal:
    level: float
    dominant_network: str
    executive_score: float
    memory_score: float
    language_score: float
    default_mode_score: float
    consult_gain: float
    system2_gate: bool
    memory_gate: str
    suppress_default_mode: bool
    notes: Tuple[str, ...] = ()

    def to_payload(self) -> Dict[str, Any]:
        return {
            "level": float(_clamp01(self.level)),
            "dominant_network": self.dominant_network,
            "executive_score": float(_clamp01(self.executive_score)),
            "memory_score": float(_clamp01(self.memory_score)),
            "language_score": float(_clamp01(self.language_score)),
            "default_mode_score": float(_clamp01(self.default_mode_score)),
            "consult_gain": float(max(-1.0, min(1.0, self.consult_gain))),
            "system2_gate": bool(self.system2_gate),
            "memory_gate": self.memory_gate,
            "suppress_default_mode": bool(self.suppress_default_mode),
            "notes": list(self.notes),
        }


class SalienceNetwork:
    """Route control toward executive, memory, or low-effort language states."""

    def evaluate(
        self,
        *,
        question: str,
        interoception: InteroceptiveState,
        focus_metric: float,
        q_type_hint: str,
        is_trivial_chat: bool,
        has_working_memory: bool,
        has_long_term_memory: bool,
        has_hippocampal_memory: bool,
    ) -> SalienceSignal:
        focus_metric = _clamp01(focus_metric)
        q_type_norm = str(q_type_hint or "easy").strip().lower()
        q_type_score = 0.85 if q_type_norm == "hard" else 0.60 if q_type_norm == "medium" else 0.20

        executive = _clamp01(
            0.42 * interoception.salience
            + 0.25 * interoception.uncertainty
            + 0.20 * q_type_score
            + 0.13 * (1.0 - focus_metric)
        )
        memory = _clamp01(
            0.36 * interoception.load
            + 0.22 * interoception.novelty
            + (0.18 if has_long_term_memory else 0.0)
            + (0.14 if has_hippocampal_memory else 0.0)
            + (0.10 if has_working_memory else 0.0)
        )
        language = _clamp01(
            0.42 * (1.0 - interoception.salience)
            + 0.22 * interoception.stability
            + 0.20 * (1.0 - interoception.load)
            + 0.16 * (1.0 if q_type_norm == "easy" else 0.0)
        )
        low_salience = max(0.0, 0.45 - interoception.salience) / 0.45
        low_uncertainty = max(0.0, 0.40 - interoception.uncertainty) / 0.40
        default_mode = _clamp01(
            0.34 * interoception.stability
            + 0.24 * (1.0 - interoception.urgency)
            + 0.24 * low_salience
            + 0.18 * low_uncertainty
        )

        if is_trivial_chat:
            executive = _clamp01(executive * 0.40)
            memory = _clamp01(memory * 0.35)
            default_mode = _clamp01(default_mode * 0.60)
            language = _clamp01(max(language, 0.65))

        scores = {
            "executive_control": executive,
            "memory_recall": memory,
            "language": language,
            "default_mode": default_mode,
        }
        dominant_network = max(scores, key=scores.get)
        level = scores[dominant_network]

        system2_gate = bool(
            dominant_network in {"executive_control", "memory_recall"}
            and (
                level >= 0.58
                or interoception.urgency >= 0.62
                or q_type_norm == "hard"
            )
        )
        suppress_default_mode = bool(
            dominant_network in {"executive_control", "memory_recall"}
            and (level >= 0.50 or interoception.urgency >= 0.55)
        )

        if dominant_network == "executive_control":
            consult_gain = 0.14 if level >= 0.55 else 0.08
        elif dominant_network == "memory_recall":
            consult_gain = 0.09 if level >= 0.55 else 0.04
        elif is_trivial_chat:
            consult_gain = -0.08
        else:
            consult_gain = -0.03

        if is_trivial_chat:
            memory_gate = "working_only" if has_working_memory else "suppress_long_term"
        elif dominant_network == "memory_recall":
            memory_gate = "episodic"
        elif dominant_network == "executive_control":
            memory_gate = (
                "balanced"
                if (has_long_term_memory or has_hippocampal_memory)
                else "working_only"
            )
        elif (
            dominant_network == "language"
            and level >= 0.55
            and not (has_long_term_memory or has_hippocampal_memory)
        ):
            memory_gate = "working_only"
        else:
            memory_gate = "balanced"

        notes = [f"q_type:{q_type_norm}", f"dominant:{dominant_network}"]
        if system2_gate:
            notes.append("system2_gate")
        if suppress_default_mode:
            notes.append("suppress_default_mode")

        return SalienceSignal(
            level=level,
            dominant_network=dominant_network,
            executive_score=executive,
            memory_score=memory,
            language_score=language,
            default_mode_score=default_mode,
            consult_gain=consult_gain,
            system2_gate=system2_gate,
            memory_gate=memory_gate,
            suppress_default_mode=suppress_default_mode,
            notes=tuple(notes),
        )


__all__ = ["SalienceNetwork", "SalienceSignal"]
