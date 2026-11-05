"""Bounded missing-evidence conditions derived from explicit request metadata.

No source text is consulted here. Unsupported question shapes stay unclassified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from .memory_intent import (
    _normalized_literal_request,
    literal_record_replay_requested,
    scoped_record_replay_requested,
)


@dataclass(frozen=True)
class MemoryRequest:
    operation: str
    parameter: str | None
    parameter_status: str


@dataclass(frozen=True)
class MissingEvidenceCondition:
    reference: str
    identity_basis: str
    operation: str
    parameter: str | None
    parameter_status: str
    source_availability: str
    source_approval: str
    action: str

    def to_dict(self):
        return {"version": "missing_evidence_v1", **asdict(self)}


_PARAMETERS = {
    "data retention period": "data_retention_period",
    "data-retention period": "data_retention_period",
    "retention rate": "retention_rate",
    "データ保持期間": "data_retention_period",
    "データ保存期間": "data_retention_period",
    "リテンション率": "retention_rate",
    "保持率": "retention_rate",
}
_UNDERSPECIFIED = {"retention", "リテンション"}
_SLOT = "\x00"


def _parameter_request(text):
    text = _normalized_literal_request(text)
    # Full matches preserve extra clauses and distinguish a quoted term from
    # an explicit request. In particular, "retention" supplies no units/domain.
    english = re.fullmatch(
        r"(?:please )?(?:(?:what is|what's|tell me) )?(?:the |its |that )?"
        r"(data[- ]retention period|retention rate|retention)", text)
    japanese = re.fullmatch(
        r"(?:その)?(データ保持期間|データ保存期間|リテンション率|保持率|リテンション)"
        r"(?:は|を教えて|を教えてください|について教えて|について教えてください)?", text)
    match = english or japanese
    if match is None:
        return None
    term = match.group(1)
    return MemoryRequest("parameter_lookup", _PARAMETERS.get(term),
                         "underspecified" if term in _UNDERSPECIFIED else "explicit")


def _scoped_parameter_request(text):
    # One complete identity in an allowed slot. Alias deletion must never
    # remove another requested clause or manufacture an unambiguous operation.
    patterns = (
        r"(?:project )?\x00(?:['’]s)?(?:\s*:\s*|\s+)(?P<body>.+)",
        r"(?:please )?(?:what is|what's|tell me) (?:project )?\x00['’]s (?P<body>.+)",
        r"(?P<body>.+) (?:for|of) (?:project )?\x00",
        r"(?:プロジェクト\s*)?\x00\s*の(?P<body>.+)",
        r"(?:プロジェクト\s*)?\x00\s*[:：]\s*(?P<body>.+)",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match is not None:
            return _parameter_request(match.group("body"))
    return None


def declared_reference_followup(question):
    """A bounded unqualified/pronominal request can use this turn's declaration."""
    if not isinstance(question, str) or len(question) > 6000 or _SLOT in question:
        return False
    text = _normalized_literal_request(question)
    if _parameter_request(text) is not None or literal_record_replay_requested(text):
        return True
    pronoun = re.fullmatch(r"(?:that|this) project(?:['’]s)? (.+)|そのプロジェクトの(.+)", text)
    return bool(pronoun is not None and _parameter_request(pronoun.group(1) or pronoun.group(2)))


def explicit_memory_request(question, resolver, *, caller_declared=False):
    """Recognize only whole replay and the documented retention question forms.

    An unknown single-word Project name can be a syntactic target, but it never
    becomes a resolved identity. Multiword names require declared alias spans.
    """
    if not isinstance(question, str) or len(question) > 6000 or _SLOT in question:
        return None
    if scoped_record_replay_requested(question, resolver):
        return MemoryRequest("recorded_answer_replay", None, "not_applicable")
    if caller_declared and literal_record_replay_requested(question):
        return MemoryRequest("recorded_answer_replay", None, "not_applicable")

    parts = resolver.single_alias_parts(question, allow_ambiguous=True)
    if parts is not None:
        return _scoped_parameter_request(_normalized_literal_request(parts[0] + _SLOT + parts[1]))
    text = _normalized_literal_request(question)
    if caller_declared:
        request = _parameter_request(text)
        if request is not None:
            return request
        # Caller-declared pronouns carry identity, even without a current alias.
        pronoun = re.fullmatch(r"(?:that|this) project(?:['’]s)? (.+)|そのプロジェクトの(.+)", text)
        if pronoun is not None:
            return _parameter_request(pronoun.group(1) or pronoun.group(2))

    # This branch identifies the missing reference, not a new memory scope.
    markers = list(re.finditer(r"\bproject (?P<name>[\w-]{1,64})(?![\w-])", text))
    if len(markers) == 1:
        marker = markers[0]
        candidate = text[:marker.start("name")] + _SLOT + text[marker.end("name"):]
        return _scoped_parameter_request(candidate)
    return None
