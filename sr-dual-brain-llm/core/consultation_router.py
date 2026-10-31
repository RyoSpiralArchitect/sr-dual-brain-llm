"""Bounded, model-free estimates of review need; never factual approvals."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
import re
import unicodedata

VERSION = "conversation_lookup_v1"
_REASONING = re.compile(
    r"\b(verify|check|review|audit|prove|calculate|compute|sum|total|difference|"
    r"compare|comparison|versus|why|derive|design|plan|debug|evaluate|tradeoff|"
    r"risk|incident|urgent|if|unless)\b|how\s+(would|should|can)|"
    r"検証|確認して|見直|正しい|妥当|計算|合計|差分|何日差|何倍|比較|なぜ|"
    r"どのよう|どうす|設計|手順|理由|影響|見積|リスク|緊急|障害|もし|場合分け",
    re.I,
)
_LOOKUP = re.compile(r"\b(what|which|when|recall|repeat)\b|教えて|何|どの|いつ|再掲|そのまま|は[?？]?$", re.I)
_NUMBERS = re.compile(r"(?<![A-Za-z0-9_.])[+-]?\d+(?:,\d+)*(?:\.\d+)?(?![A-Za-z0-9_])")
_SCALAR = re.compile(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?")


def _numbers(text):
    normalized = unicodedata.normalize("NFKC", text)
    if re.search(r"\d\s*/\s*\d|\d\s+[0-9]|\d\s*[eE]\s*[+-]?\s*\d|\d\.\d+\.\d", normalized):
        raise ValueError('Unsupported numeric format')
    values = _NUMBERS.findall(normalized)
    if any(_SCALAR.fullmatch(value) is None for value in values):
        raise ValueError('Unsupported numeric grouping')
    return {Decimal(value.replace(',', '')) for value in values}


@dataclass(frozen=True)
class ConsultationRoute:
    action: str  # skip, consult, or defer to the existing policy
    reason: str


def route_conversation_lookup(*, question, draft, confidence, risk, q_type,
                              reference, approved_records):
    """A skip is a routing estimate, not proof that a draft is correct.

    Require a single currently approved record and a resolved reference. Missing
    evidence, qualitative-only records and uncertain drafts retain normal routing.
    Positive reasoning/conflict signals can request a critic even at high confidence.
    """
    q = unicodedata.normalize("NFKC", str(question))
    if _REASONING.search(q) or "```" in q or re.search(r"[=<>≠≥≤→⇒]|\d\s*[+*/^%-]\s*\d", q):
        return ConsultationRoute("consult", "reasoning_or_review_request")
    # Existing "medium" also means a question mark after 26 characters; that
    # surface-length signal alone must not outweigh a bounded supported lookup.
    if q_type == "hard":
        return ConsultationRoute("consult", "question_complexity")
    if not math.isfinite(float(confidence)) or not math.isfinite(float(risk)):
        return ConsultationRoute("defer", "unknown_signal")
    if risk > 0 or confidence < .85:
        return ConsultationRoute("defer", "risk_or_uncertainty")
    if reference != "resolved" or len(approved_records) != 1:
        return ConsultationRoute("defer", "no_unique_supported_record")
    if not draft.strip() or len(q) > 180 or "\n" in q or not _LOOKUP.search(q):
        return ConsultationRoute("defer", "not_bounded_lookup")
    source = approved_records[0]
    try:
        source_numbers, draft_values = _numbers(source.answer), _numbers(draft)
        # Declared identity/version digits are not requested factual coverage.
        identity_numbers = _numbers(" ".join(source.scope_aliases))
        allowed = source_numbers | _numbers(source.question) | _numbers(q)
    except ValueError:
        return ConsultationRoute("consult", "unsupported_numeric_format")
    values = source_numbers - identity_numbers
    if draft_values - allowed:
        return ConsultationRoute("consult", "numeric_drift")
    if not values or not values <= draft_values:
        return ConsultationRoute("defer", "incomplete_numeric_coverage")
    return ConsultationRoute("skip", "supported_lookup")
