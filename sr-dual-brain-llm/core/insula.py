"""Insula-inspired interoceptive signal synthesis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple


def _clamp01(value: float) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, value))


_URGENCY_PATTERN = re.compile(
    r"(urgent|urgency|outage|incident|deploy|deployment|rollback|p0|sev[ -]?1|"
    r"leak|breach|immediately|now|asap|"
    r"緊急|至急|障害|事故|漏えい|流出|今すぐ|直ちに)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class InteroceptiveState:
    salience: float
    uncertainty: float
    load: float
    urgency: float
    stability: float
    arousal: float
    risk: float
    novelty: float
    sources: Tuple[str, ...] = ()

    def to_payload(self) -> Dict[str, Any]:
        return {
            "salience": float(_clamp01(self.salience)),
            "uncertainty": float(_clamp01(self.uncertainty)),
            "load": float(_clamp01(self.load)),
            "urgency": float(_clamp01(self.urgency)),
            "stability": float(_clamp01(self.stability)),
            "arousal": float(_clamp01(self.arousal)),
            "risk": float(_clamp01(self.risk)),
            "novelty": float(_clamp01(self.novelty)),
            "sources": list(self.sources),
        }


class Insula:
    """Approximate interoceptive integration for routing decisions."""

    def assess(
        self,
        *,
        question: str,
        affect: Dict[str, float],
        novelty: float,
        focus_metric: float,
        context_signal_len: int = 0,
        has_working_memory: bool = False,
        has_long_term_memory: bool = False,
        is_trivial_chat: bool = False,
    ) -> InteroceptiveState:
        q = str(question or "").strip()
        q_len = len(q)
        line_count = q.count("\n") + (1 if q else 0)
        delimiter_hits = len(re.findall(r"[,、，;；:：]", q))
        has_digits = bool(re.search(r"\d", q))
        has_question = ("?" in q) or ("？" in q)
        urgency_marker = 1.0 if _URGENCY_PATTERN.search(q) else 0.0

        arousal = _clamp01(affect.get("arousal", 0.0))
        risk = _clamp01(affect.get("risk", 0.0))
        novelty = _clamp01(novelty)
        focus_metric = _clamp01(focus_metric)

        question_pressure = _clamp01(
            0.18 * max(0, line_count - 1)
            + 0.07 * delimiter_hits
            + (0.12 if has_digits else 0.0)
            + (0.18 if has_question else 0.0)
            + 0.30 * min(1.0, q_len / 220.0)
        )
        memory_pressure = _clamp01(
            0.55 * min(1.0, float(context_signal_len) / 360.0)
            + (0.10 if has_working_memory else 0.0)
            + (0.15 if has_long_term_memory else 0.0)
        )
        load = _clamp01(
            0.50 * question_pressure
            + 0.35 * memory_pressure
            + 0.15 * (1.0 - focus_metric)
        )
        uncertainty = _clamp01(
            0.40 * (1.0 - focus_metric)
            + 0.25 * novelty
            + 0.20 * risk
            + 0.15 * question_pressure
        )
        urgency = _clamp01(
            0.45 * risk + 0.25 * arousal + 0.15 * novelty + 0.15 * urgency_marker
        )
        stability = _clamp01(
            1.0 - (0.45 * risk + 0.25 * arousal + 0.30 * uncertainty)
        )
        salience = _clamp01(
            0.35 * urgency + 0.25 * load + 0.20 * uncertainty + 0.20 * novelty
        )

        if is_trivial_chat:
            load = _clamp01(load * 0.45)
            uncertainty = _clamp01(uncertainty * 0.55)
            urgency = _clamp01(urgency * 0.40)
            salience = _clamp01(salience * 0.40)
            stability = _clamp01(min(1.0, stability + 0.12))

        sources = ["affect", "novelty", "focus", "context"]
        if urgency_marker:
            sources.append("urgency_marker")
        if is_trivial_chat:
            sources.append("trivial_chat")

        return InteroceptiveState(
            salience=salience,
            uncertainty=uncertainty,
            load=load,
            urgency=urgency,
            stability=stability,
            arousal=arousal,
            risk=risk,
            novelty=novelty,
            sources=tuple(sources),
        )


__all__ = ["Insula", "InteroceptiveState"]
