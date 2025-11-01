"""Utilities for mapping psychoid archetype signals into attention biases."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Union

try:  # Optional torch support
    import torch  # type: ignore

    HAVE_TORCH = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    HAVE_TORCH = False


@dataclass
class PsychoidAttentionProjection:
    """Container describing an attention-bias matrix derived from psychoid cues."""

    bias_matrix: List[List[float]]
    norm: float
    temperature: float
    metadata: Dict[str, float]

    def to_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "matrix": self.bias_matrix,
            "norm": self.norm,
            "temperature": self.temperature,
        }
        payload.update(self.metadata)
        return payload


from .schema import PsychoidSignalModel


class PsychoidAttentionAdapter:
    """Transform psychoid archetype signals into QKV-friendly bias matrices."""

    def __init__(
        self,
        *,
        base_temperature: float = 1.0,
        clamp: float = 2.4,
        minimum_bias: float = 1e-4,
    ) -> None:
        self.base_temperature = base_temperature
        self.clamp = clamp
        self.minimum_bias = abs(minimum_bias)

    def build_projection(
        self,
        signal: Union[Mapping[str, Any], PsychoidSignalModel],
        *,
        seq_len: int,
        qkv_dim: Optional[int] = None,
    ) -> PsychoidAttentionProjection:
        """Convert a psychoid signal mapping into an attention bias projection."""

        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        if isinstance(signal, PsychoidSignalModel):
            raw_source = signal.bias_vector
        else:
            raw_source = signal.get("bias_vector", [])
        raw_vector = [float(v) for v in raw_source if v is not None]
        if not raw_vector:
            raw_vector = [0.0]

        qkv = max(seq_len, qkv_dim or len(raw_vector) or 1)
        repeats = (qkv + len(raw_vector) - 1) // len(raw_vector)
        padded = (raw_vector * repeats)[:qkv]
        scaled = [self._scale_weight(weight) for weight in padded]

        bias_matrix: List[List[float]] = []
        for row in range(seq_len):
            start = row % len(scaled)
            row_entries = []
            for col in range(seq_len):
                idx = (start + col) % len(scaled)
                row_entries.append(scaled[idx])
            bias_matrix.append(row_entries)

        norm = math.sqrt(sum(value * value for row in bias_matrix for value in row))
        if norm < self.minimum_bias:
            norm = self.minimum_bias

        if isinstance(signal, PsychoidSignalModel):
            resonance = float(signal.resonance)
            tension = float(signal.psychoid_tension)
            chain_length = float(len(signal.signifier_chain))
        else:
            resonance = float(signal.get("resonance", 0.0) or 0.0)
            tension = float(signal.get("psychoid_tension", 0.0) or 0.0)
            chain_length = float(len(signal.get("signifier_chain") or []))

        metadata = {
            "resonance": resonance,
            "psychoid_tension": tension,
            "chain_length": chain_length,
            "clamp": float(self.clamp),
        }

        return PsychoidAttentionProjection(
            bias_matrix=bias_matrix,
            norm=norm,
            temperature=self.base_temperature,
            metadata=metadata,
        )

    def apply_to_scores(
        self,
        scores: "torch.Tensor",
        projection: PsychoidAttentionProjection,
    ) -> "torch.Tensor":  # pragma: no cover - requires torch runtime
        """Inject the psychoid bias matrix into an attention score tensor."""

        if not HAVE_TORCH:
            raise RuntimeError("torch is required to apply psychoid attention biases")

        bias = torch.tensor(projection.bias_matrix, dtype=scores.dtype, device=scores.device)
        while bias.dim() < scores.dim():
            bias = bias.unsqueeze(0)
        return scores + bias

    def _scale_weight(self, weight: float) -> float:
        scaled = math.tanh(weight * self.base_temperature) * self.clamp
        if abs(scaled) < self.minimum_bias:
            return math.copysign(self.minimum_bias, scaled if scaled != 0 else 1.0)
        return scaled


__all__ = [
    "HAVE_TORCH",
    "PsychoidAttentionAdapter",
    "PsychoidAttentionProjection",
]
