"""Typed payload helpers shared across the dual-brain pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Union


def _as_float(value: float) -> float:
    """Return ``value`` coerced to ``float``."""

    return float(value)


def _as_int(value: int) -> int:
    """Return ``value`` coerced to ``int``."""

    return int(value)


def _coerce_bias_entries(
    entries: Iterable[Union["AttentionBiasEntry", Dict[str, object]]],
) -> List["AttentionBiasEntry"]:
    coerced: List[AttentionBiasEntry] = []
    for entry in entries:
        if isinstance(entry, AttentionBiasEntry):
            coerced.append(entry)
        else:
            coerced.append(AttentionBiasEntry(**entry))
    return coerced


def _coerce_archetypes(
    entries: Iterable[Union["ArchetypeActivation", Dict[str, object]]],
) -> List["ArchetypeActivation"]:
    coerced: List[ArchetypeActivation] = []
    for entry in entries:
        if isinstance(entry, ArchetypeActivation):
            coerced.append(entry)
        else:
            coerced.append(ArchetypeActivation(**entry))
    return coerced


def _coerce_emergent(
    entries: Iterable[Union["EmergentIdeaModel", Dict[str, object]]],
) -> List["EmergentIdeaModel"]:
    coerced: List[EmergentIdeaModel] = []
    for entry in entries:
        if isinstance(entry, EmergentIdeaModel):
            coerced.append(entry)
        else:
            coerced.append(EmergentIdeaModel(**entry))
    return coerced


@dataclass
class GeometryModel:
    """Polar representation of an archetypal event embedding."""

    r: float
    theta: float
    curvature_proxy: float

    def __post_init__(self) -> None:
        self.r = _as_float(self.r)
        self.theta = _as_float(self.theta)
        self.curvature_proxy = _as_float(self.curvature_proxy)

    def to_payload(self) -> Dict[str, float]:
        return {
            "r": self.r,
            "theta": self.theta,
            "curvature_proxy": self.curvature_proxy,
        }


@dataclass
class ArchetypeActivation:
    """Archetypal activation surfaced by the unconscious field."""

    id: str
    label: str
    intensity: float

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.label = str(self.label)
        self.intensity = max(0.0, _as_float(self.intensity))

    def to_payload(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "intensity": self.intensity,
        }


@dataclass
class EmergentIdeaModel:
    """Incubated seed that resurfaced as a structured insight."""

    archetype: str
    label: str
    intensity: float
    incubation_rounds: int
    trigger_similarity: float
    origin: str

    def __post_init__(self) -> None:
        self.archetype = str(self.archetype)
        self.label = str(self.label)
        self.intensity = _as_float(self.intensity)
        self.incubation_rounds = _as_int(self.incubation_rounds)
        self.trigger_similarity = _as_float(self.trigger_similarity)
        self.origin = str(self.origin)

    def to_payload(self) -> Dict[str, object]:
        return {
            "archetype": self.archetype,
            "label": self.label,
            "intensity": self.intensity,
            "incubation_rounds": self.incubation_rounds,
            "trigger_similarity": self.trigger_similarity,
            "origin": self.origin,
        }


@dataclass
class AttentionBiasEntry:
    """Attention-bias contribution for a single archetype."""

    archetype: str
    label: str
    weight: float
    resonance: float

    def __post_init__(self) -> None:
        self.archetype = str(self.archetype)
        self.label = str(self.label)
        self.weight = _as_float(self.weight)
        self.resonance = _as_float(self.resonance)

    def to_payload(self) -> Dict[str, object]:
        return {
            "archetype": self.archetype,
            "label": self.label,
            "weight": self.weight,
            "resonance": self.resonance,
        }


@dataclass
class PsychoidSignalModel:
    """Structured payload projecting archetypal cues into attention biases."""

    attention_bias: List[AttentionBiasEntry] = field(default_factory=list)
    bias_vector: List[float] = field(default_factory=list)
    psychoid_tension: float = 0.0
    resonance: float = 0.0
    signifier_chain: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.attention_bias = _coerce_bias_entries(self.attention_bias)
        self.bias_vector = [_as_float(value) for value in self.bias_vector]
        self.psychoid_tension = _as_float(self.psychoid_tension)
        self.resonance = _as_float(self.resonance)
        self.signifier_chain = [str(item) for item in self.signifier_chain]

    def to_payload(self) -> Dict[str, object]:
        return {
            "attention_bias": [entry.to_payload() for entry in self.attention_bias],
            "bias_vector": list(self.bias_vector),
            "psychoid_tension": self.psychoid_tension,
            "resonance": self.resonance,
            "signifier_chain": list(self.signifier_chain),
        }


@dataclass
class UnconsciousSummaryModel:
    """Complete summary exported by the unconscious field."""

    top_k: List[str] = field(default_factory=list)
    geometry: GeometryModel = field(default_factory=lambda: GeometryModel(0.0, 0.0, 0.0))
    archetype_map: List[ArchetypeActivation] = field(default_factory=list)
    emergent_ideas: List[EmergentIdeaModel] = field(default_factory=list)
    stress_released: float = 0.0
    cache_depth: int = 0
    psychoid_signal: Optional[PsychoidSignalModel] = None
    motifs: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, GeometryModel):
            self.geometry = GeometryModel(**self.geometry)
        self.top_k = [str(item) for item in self.top_k]
        self.archetype_map = _coerce_archetypes(self.archetype_map)
        self.emergent_ideas = _coerce_emergent(self.emergent_ideas)
        self.stress_released = _as_float(self.stress_released)
        self.cache_depth = _as_int(self.cache_depth)
        if self.psychoid_signal is not None and not isinstance(
            self.psychoid_signal, PsychoidSignalModel
        ):
            self.psychoid_signal = PsychoidSignalModel(**self.psychoid_signal)
        if self.motifs is not None:
            self.motifs = [str(item) for item in self.motifs]

    def to_payload(self) -> Dict[str, object]:
        return {
            "top_k": list(self.top_k),
            "geometry": self.geometry.to_payload(),
            "archetype_map": [score.to_payload() for score in self.archetype_map],
            "emergent_ideas": [idea.to_payload() for idea in self.emergent_ideas],
            "stress_released": self.stress_released,
            "cache_depth": self.cache_depth,
            "psychoid_signal": self.psychoid_signal.to_payload()
            if self.psychoid_signal
            else None,
            "motifs": list(self.motifs) if self.motifs is not None else None,
        }


__all__ = [
    "ArchetypeActivation",
    "AttentionBiasEntry",
    "EmergentIdeaModel",
    "GeometryModel",
    "PsychoidSignalModel",
    "UnconsciousSummaryModel",
]
