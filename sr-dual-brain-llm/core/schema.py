"""Typed payload models shared across the dual-brain pipeline."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class GeometryModel(BaseModel):
    """Polar representation of an archetypal event embedding."""

    r: float
    theta: float
    curvature_proxy: float


class ArchetypeActivation(BaseModel):
    """Archetypal activation surfaced by the unconscious field."""

    id: str
    label: str
    intensity: float

    @validator("intensity")
    def _clamp_intensity(cls, value: float) -> float:
        return max(0.0, float(value))


class EmergentIdeaModel(BaseModel):
    """Incubated seed that resurfaced as a structured insight."""

    archetype: str
    label: str
    intensity: float
    incubation_rounds: int
    trigger_similarity: float
    origin: str

    @validator("intensity", "trigger_similarity")
    def _normalise_float(cls, value: float) -> float:
        return float(value)


class AttentionBiasEntry(BaseModel):
    """Attention-bias contribution for a single archetype."""

    archetype: str
    label: str
    weight: float
    resonance: float

    @validator("weight", "resonance")
    def _normalise_weight(cls, value: float) -> float:
        return float(value)


class PsychoidSignalModel(BaseModel):
    """Structured payload projecting archetypal cues into attention biases."""

    attention_bias: List[AttentionBiasEntry] = Field(default_factory=list)
    bias_vector: List[float] = Field(default_factory=list)
    psychoid_tension: float = 0.0
    resonance: float = 0.0
    signifier_chain: List[str] = Field(default_factory=list)

    @validator("bias_vector", each_item=True)
    def _normalise_bias(cls, value: float) -> float:
        return float(value)

    def to_payload(self) -> Dict[str, object]:
        """Return a JSON-serialisable dictionary."""

        payload: Dict[str, object] = {
            "bias_vector": list(self.bias_vector),
            "psychoid_tension": float(self.psychoid_tension),
            "resonance": float(self.resonance),
            "signifier_chain": list(self.signifier_chain),
            "attention_bias": [entry.dict() for entry in self.attention_bias],
        }
        return payload


class UnconsciousSummaryModel(BaseModel):
    """Complete summary exported by the unconscious field."""

    top_k: List[str] = Field(default_factory=list)
    geometry: GeometryModel
    archetype_map: List[ArchetypeActivation] = Field(default_factory=list)
    emergent_ideas: List[EmergentIdeaModel] = Field(default_factory=list)
    stress_released: float = 0.0
    cache_depth: int = 0
    psychoid_signal: Optional[PsychoidSignalModel] = None
    motifs: Optional[List[str]] = None

    def to_payload(self) -> Dict[str, object]:
        """Return a JSON-serialisable dictionary."""

        payload: Dict[str, object] = {
            "top_k": list(self.top_k),
            "geometry": self.geometry.dict(),
            "archetype_map": [score.dict() for score in self.archetype_map],
            "emergent_ideas": [idea.dict() for idea in self.emergent_ideas],
            "stress_released": float(self.stress_released),
            "cache_depth": int(self.cache_depth),
            "psychoid_signal": self.psychoid_signal.to_payload()
            if self.psychoid_signal
            else None,
            "motifs": list(self.motifs) if self.motifs else None,
        }
        return payload


__all__ = [
    "ArchetypeActivation",
    "AttentionBiasEntry",
    "EmergentIdeaModel",
    "GeometryModel",
    "PsychoidSignalModel",
    "UnconsciousSummaryModel",
]
