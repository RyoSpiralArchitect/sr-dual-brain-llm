"""Typed request/response payloads for corpus callosum transports.

The original implementation in :mod:`core.callosum` relied on dynamic
``dict`` instances that were passed between the left- and right-brain
components.  That made static analysis difficult and left plenty of room for
runtime ``KeyError``/``AttributeError`` situations once optional fields were
added.  This module provides dataclasses that capture the supported payloads in
one place so that transports, workers, and controllers can speak the same
language.

The dataclasses expose ``to_payload`` / ``from_payload`` helpers so existing
backends that ultimately move JSON across the wire can still operate on plain
Python dictionaries when required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional

try:  # Python 3.11+
    from typing import Literal
except ImportError:  # pragma: no cover - fallback for 3.10
    from typing_extensions import Literal  # type: ignore

_KNOWN_DETAIL_REQUEST_KEYS = {
    "type",
    "qid",
    "question",
    "draft_sum",
    "temperature",
    "budget",
    "context",
    "hemisphere_mode",
    "hemisphere_bias",
}


def _normalise_optional(value: Any) -> Any:
    """Return ``None`` for empty containers so JSON payloads stay compact."""

    if value is None:
        return None
    if value == [] or value == {}:
        return None
    return value


@dataclass(slots=True)
class DetailRequest:
    """Structured payload describing a request for right-brain detail."""

    qid: str
    question: str
    draft_summary: str
    temperature: float
    budget: str
    context: Optional[str] = None
    hemisphere_mode: str = ""
    hemisphere_bias: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)
    type: Literal["ASK_DETAIL"] = field(init=False, default="ASK_DETAIL")

    def __post_init__(self) -> None:
        # Drop any ``None`` entries from the extras dictionary so downstream
        # transports don't have to repeatedly guard against them.
        self.extras = {k: v for k, v in self.extras.items() if v is not None}

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": self.type,
            "qid": self.qid,
            "question": self.question,
            "draft_sum": self.draft_summary,
            "temperature": self.temperature,
            "budget": self.budget,
            "context": self.context,
            "hemisphere_mode": self.hemisphere_mode,
            "hemisphere_bias": self.hemisphere_bias,
        }
        payload.update(self.extras)
        return {key: _normalise_optional(value) for key, value in payload.items() if value is not None}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DetailRequest":
        message_type = payload.get("type") or "ASK_DETAIL"
        if message_type != "ASK_DETAIL":
            raise ValueError(f"Unsupported payload type: {message_type}")

        missing = [key for key in ("qid", "question", "draft_sum", "temperature", "budget") if key not in payload]
        if missing:
            raise ValueError(f"DetailRequest payload missing keys: {missing}")

        extras = {
            key: value
            for key, value in payload.items()
            if key not in _KNOWN_DETAIL_REQUEST_KEYS and value is not None
        }

        return cls(
            qid=str(payload["qid"]),
            question=str(payload["question"]),
            draft_summary=str(payload["draft_sum"]),
            temperature=float(payload["temperature"]),
            budget=str(payload["budget"]),
            context=payload.get("context"),
            hemisphere_mode=str(payload.get("hemisphere_mode", "")),
            hemisphere_bias=float(payload.get("hemisphere_bias", 0.0) or 0.0),
            extras=extras,
        )


@dataclass(slots=True)
class DetailResponse:
    """Structured representation of the right-brain response."""

    qid: str
    notes_summary: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"qid": self.qid}
        if self.notes_summary is not None:
            payload["notes_sum"] = self.notes_summary
        if self.confidence is not None:
            payload["confidence_r"] = self.confidence
        if self.error is not None:
            payload["error"] = self.error
        payload.update(self.extras)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DetailResponse":
        if "qid" not in payload:
            raise ValueError("DetailResponse payload must include 'qid'")

        extras = {
            key: value
            for key, value in payload.items()
            if key not in {"qid", "notes_sum", "confidence_r", "error"}
        }

        confidence = payload.get("confidence_r")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                confidence = None

        return cls(
            qid=str(payload["qid"]),
            notes_summary=payload.get("notes_sum"),
            confidence=confidence,
            error=payload.get("error"),
            extras=extras,
        )

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.notes_summary)


def ensure_mapping(data: DetailRequest | DetailResponse | Mapping[str, Any]) -> Dict[str, Any]:
    """Return a ``dict`` representation of any supported payload variant."""

    if isinstance(data, DetailRequest) or isinstance(data, DetailResponse):
        return data.to_payload()
    if isinstance(data, MutableMapping):
        return dict(data)
    if isinstance(data, Mapping):
        return {key: value for key, value in data.items()}
    raise TypeError(f"Unsupported payload type: {type(data)!r}")


__all__ = [
    "DetailRequest",
    "DetailResponse",
    "ensure_mapping",
]

