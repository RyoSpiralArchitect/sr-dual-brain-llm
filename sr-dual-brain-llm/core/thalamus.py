"""Thalamus-inspired relay gating for context streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .salience_network import SalienceSignal


def _clamp01(value: float) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class ThalamicRelay:
    target_network: str
    relay_gain: float
    keep_working_memory: bool
    keep_long_term_memory: bool
    keep_schema_memory: bool
    keep_pitfall_memory: bool
    keep_hippocampal_memory: bool
    suppress_default_mode: bool
    notes: Tuple[str, ...] = ()

    def to_payload(self) -> Dict[str, Any]:
        return {
            "target_network": self.target_network,
            "relay_gain": float(_clamp01(self.relay_gain)),
            "keep_working_memory": bool(self.keep_working_memory),
            "keep_long_term_memory": bool(self.keep_long_term_memory),
            "keep_schema_memory": bool(self.keep_schema_memory),
            "keep_pitfall_memory": bool(self.keep_pitfall_memory),
            "keep_hippocampal_memory": bool(self.keep_hippocampal_memory),
            "suppress_default_mode": bool(self.suppress_default_mode),
            "notes": list(self.notes),
        }


class Thalamus:
    """Gate memory streams according to the current salience regime."""

    def route(
        self,
        *,
        context_parts: Dict[str, str],
        salience: SalienceSignal,
    ) -> ThalamicRelay:
        memory_gate = str(salience.memory_gate or "balanced").strip().lower()
        has_working = bool(context_parts.get("working_memory"))
        has_memory = bool(context_parts.get("memory"))
        has_schema = bool(context_parts.get("schema"))
        has_pitfalls = bool(context_parts.get("pitfalls"))
        has_hippocampal = bool(context_parts.get("hippocampal"))

        keep_working = has_working
        keep_long_term = has_memory
        keep_schema = has_schema
        keep_pitfalls = has_pitfalls
        keep_hippocampal = has_hippocampal

        if memory_gate == "suppress_long_term":
            keep_long_term = False
            keep_schema = False
            keep_pitfalls = False
            keep_hippocampal = False
        elif memory_gate == "working_only":
            if has_working:
                keep_long_term = False
                keep_schema = False
                keep_pitfalls = False
                keep_hippocampal = False
            else:
                notes = [f"memory_gate:{memory_gate}", f"target:{salience.dominant_network}", "fallback_long_term"]
                return ThalamicRelay(
                    target_network=salience.dominant_network,
                    relay_gain=salience.level,
                    keep_working_memory=keep_working,
                    keep_long_term_memory=keep_long_term,
                    keep_schema_memory=keep_schema,
                    keep_pitfall_memory=keep_pitfalls,
                    keep_hippocampal_memory=keep_hippocampal,
                    suppress_default_mode=salience.suppress_default_mode,
                    notes=tuple(notes),
                )
        elif memory_gate == "episodic":
            keep_working = has_working
            keep_long_term = has_memory
            keep_schema = has_schema
            keep_pitfalls = has_pitfalls
            keep_hippocampal = has_hippocampal
        else:
            keep_working = has_working

        notes = [f"memory_gate:{memory_gate}", f"target:{salience.dominant_network}"]
        return ThalamicRelay(
            target_network=salience.dominant_network,
            relay_gain=salience.level,
            keep_working_memory=keep_working,
            keep_long_term_memory=keep_long_term,
            keep_schema_memory=keep_schema,
            keep_pitfall_memory=keep_pitfalls,
            keep_hippocampal_memory=keep_hippocampal,
            suppress_default_mode=salience.suppress_default_mode,
            notes=tuple(notes),
        )


__all__ = ["ThalamicRelay", "Thalamus"]
