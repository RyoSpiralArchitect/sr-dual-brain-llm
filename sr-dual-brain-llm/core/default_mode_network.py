"""Mind-wandering helper approximating the brain's default mode network."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from .schema import ArchetypeActivation, EmergentIdeaModel, UnconsciousSummaryModel


@dataclass
class DMNTrace:
    """Historical record of unconscious summaries processed by the DMN."""

    archetype: str
    label: str
    intensity: float
    cache_depth: int
    stress_released: float
    emergent_count: int
    timestamp: float = field(default_factory=lambda: time.time())

    def weight(self) -> float:
        decay = max(0.15, 1.0 - 0.12 * max(0.0, time.time() - self.timestamp) / 10.0)
        return self.intensity * decay


@dataclass
class DefaultModeReflection:
    """Reflection synthesised from unconscious cues during rest-like periods."""

    theme: str
    confidence: float
    primary_archetype: str
    supporting_archetypes: List[str]
    stress_released: float
    cache_depth: int
    emergent_count: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "theme": self.theme,
            "confidence": round(self.confidence, 4),
            "primary_archetype": self.primary_archetype,
            "supporting_archetypes": list(self.supporting_archetypes),
            "stress_released": round(self.stress_released, 4),
            "cache_depth": self.cache_depth,
            "emergent_count": self.emergent_count,
        }


class DefaultModeNetwork:
    """Approximate the default mode network surfacing resting-state insights."""

    def __init__(
        self,
        *,
        min_cache_depth: int = 3,
        stress_release_threshold: float = 0.35,
        activation_bias: float = 0.6,
        cooldown_steps: int = 2,
        max_history: int = 12,
        max_reflections: int = 2,
    ) -> None:
        self.min_cache_depth = min_cache_depth
        self.stress_release_threshold = stress_release_threshold
        self.activation_bias = activation_bias
        self.cooldown_steps = cooldown_steps
        self.max_history = max_history
        self.max_reflections = max_reflections
        self._history: List[DMNTrace] = []
        self._cooldown = 0
        self._last_reflections: List[DefaultModeReflection] = []

    def _store_trace(self, summary: UnconsciousSummaryModel) -> DMNTrace:
        scores: List[ArchetypeActivation] = list(summary.archetype_map)
        if not scores:
            raise ValueError("Summary missing archetype_map entries")
        top = scores[0]
        trace = DMNTrace(
            archetype=str(top.id or "unknown"),
            label=str(top.label or top.id or "unknown"),
            intensity=float(top.intensity),
            cache_depth=int(summary.cache_depth),
            stress_released=float(summary.stress_released or 0.0),
            emergent_count=len(summary.emergent_ideas or []),
        )
        self._history.append(trace)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]
        return trace

    def _compose_theme(
        self,
        top: ArchetypeActivation,
        supports: Sequence[ArchetypeActivation],
        emergent: Sequence[EmergentIdeaModel],
    ) -> str:
        label = str(top.label or top.id or "Archetype")
        if emergent:
            primary = emergent[0]
            elabel = primary.label or primary.archetype or "insight"
            return f"{label} reframed by {elabel}"
        if supports:
            mix = ", ".join(str(s.label or s.id) for s in supports[:2])
            return f"{label} braided with {mix}"
        return label

    def _activation_score(self, trace: DMNTrace) -> float:
        stress_factor = min(0.4, trace.stress_released * 0.5)
        cache_factor = min(0.25, max(0.0, trace.cache_depth - self.min_cache_depth) * 0.08)
        emergent_factor = min(0.25, trace.emergent_count * 0.1)
        return trace.weight() + stress_factor + cache_factor + emergent_factor

    def reflect(self, summary: UnconsciousSummaryModel) -> List[DefaultModeReflection]:
        """Produce resting-state reflections given the unconscious summary."""

        if not summary.archetype_map:
            self._last_reflections = []
            return []

        trace = self._store_trace(summary)
        scores: List[ArchetypeActivation] = list(summary.archetype_map)
        if self._cooldown > 0:
            self._cooldown -= 1
            self._last_reflections = []
            return []

        if (
            trace.cache_depth < self.min_cache_depth
            and trace.stress_released < self.stress_release_threshold
        ):
            self._last_reflections = []
            return []

        activation = self._activation_score(trace)
        if activation < self.activation_bias:
            self._last_reflections = []
            return []

        supports: List[ArchetypeActivation] = list(scores[1:])
        emergent: List[EmergentIdeaModel] = list(summary.emergent_ideas or [])
        top_score = scores[0]
        theme = self._compose_theme(top_score, supports, emergent)
        supporting_ids = [
            str(s.id or "unknown")
            for s in supports
            if float(s.intensity) >= 0.05
        ][:3]

        reflections = [
            DefaultModeReflection(
                theme=theme,
                confidence=min(1.0, activation),
                primary_archetype=str(top_score.id or "unknown"),
                supporting_archetypes=supporting_ids,
                stress_released=trace.stress_released,
                cache_depth=trace.cache_depth,
                emergent_count=trace.emergent_count,
            )
        ]

        if len(emergent) > 1 and len(reflections) < self.max_reflections:
            for idea in emergent[1: self.max_reflections]:
                reflections.append(
                    DefaultModeReflection(
                        theme=str(idea.label or idea.archetype or "Insight"),
                        confidence=min(1.0, activation * 0.85),
                        primary_archetype=str(
                            idea.archetype or reflections[0].primary_archetype
                        ),
                        supporting_archetypes=supporting_ids,
                        stress_released=trace.stress_released,
                        cache_depth=trace.cache_depth,
                        emergent_count=trace.emergent_count,
                    )
                )
                if len(reflections) >= self.max_reflections:
                    break

        self._cooldown = self.cooldown_steps
        self._last_reflections = reflections
        return list(reflections)

    @property
    def last_reflections(self) -> List[DefaultModeReflection]:
        return list(self._last_reflections)


__all__ = [
    "DefaultModeNetwork",
    "DefaultModeReflection",
    "DMNTrace",
]
