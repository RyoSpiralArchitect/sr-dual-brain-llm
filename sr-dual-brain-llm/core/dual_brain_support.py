"""Utility helpers and lightweight types shared by the dual-brain controller."""

from __future__ import annotations

import difflib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


_RIGHT_HEMISPHERE_KEYWORDS = {
    "story",
    "stories",
    "poem",
    "poetry",
    "metaphor",
    "imagine",
    "imagination",
    "dream",
    "vision",
    "art",
    "artistic",
    "myth",
    "mythic",
    "lyric",
    "narrative",
    "creative",
    "感情",
    "夢",
    "詩",
    "物語",
    "象徴",
    "直感",
    "比喩",
}

_LEFT_HEMISPHERE_KEYWORDS = {
    "analyze",
    "analysis",
    "explain",
    "detail",
    "formula",
    "proof",
    "derive",
    "compute",
    "calculation",
    "step",
    "structure",
    "data",
    "algorithm",
    "framework",
    "論理",
    "計算",
    "分析",
    "証明",
    "手順",
    "仕組み",
    "数式",
}

_RIGHT_HEMISPHERE_MARKERS = ["夢", "詩", "物語", "感情", "象徴", "幻想", "archetype"]
_LEFT_HEMISPHERE_MARKERS = ["計算", "分析", "証明", "構造", "論理", "手順", "データ"]

_RIGHT_HEMISPHERE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"compose (?:a )?(poem|story)",
        r"sketch .*journey",
        r"imagine(?:\s+or)?\s+",
        r"mythic\s+journey",
        r"write .*lyrics",
    ]
]

_LEFT_HEMISPHERE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"step[- ]by[- ]step",
        r"derive .*formula",
        r"calculate",
        r"provide an? analysis",
    ]
]


def _semantic_tokens(*chunks: str) -> List[str]:
    tokens: List[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        tokens.extend(re.findall(r"[\w']+", chunk.lower()))
    return tokens


def _format_section(title: str, lines: List[str]) -> str:
    body = "\n".join(lines) if lines else ""
    if body:
        return f"[{title}]\n{body}"
    return f"[{title}]"


def _truncate_text(text: Optional[str], limit: int = 160) -> str:
    if not text:
        return ""
    condensed = " ".join(text.split())
    if len(condensed) <= limit:
        return condensed
    return condensed[: max(0, limit - 3)] + "..."


def _normalise_similarity_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip().lower()


def _similarity_ratio(a: str, b: str) -> float:
    na = _normalise_similarity_text(a)
    nb = _normalise_similarity_text(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _strip_issue_prefix(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^[\-\*\u2022\+\s]+", "", cleaned)
    cleaned = re.sub(r"^(?:\(|\[)?\d+(?:\)|\])?[.\-:\s]+", "", cleaned)
    cleaned = re.sub(
        r"^(?:issue|issues|problem|problems|fix|fixes|note|notes)\s*[:\-]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _normalise_issue_text(text: str) -> str:
    cleaned = _strip_issue_prefix(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    cleaned = re.sub(r"[\"'`“”‘’]", "", cleaned)
    cleaned = re.sub(r"[，、。．,;:.!?]+$", "", cleaned)
    return cleaned


def _issue_token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", str(text or "").lower()))


_ISSUE_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "at",
    "in",
    "on",
    "to",
    "of",
    "for",
    "by",
    "from",
    "with",
    "and",
    "or",
    "that",
    "this",
    "it",
    "its",
    "as",
    "into",
    "over",
    "under",
    "than",
    "then",
    "re",
    "check",
    "double",
    "verify",
    "again",
    "please",
    "need",
    "needs",
    "should",
    "must",
}


def _issue_core_tokens(text: str) -> set[str]:
    tokens = _issue_token_set(text)
    if not tokens:
        return set()
    core = {token for token in tokens if token not in _ISSUE_STOPWORDS and len(token) > 1}
    return core or tokens


_ISSUE_HIGH_SIGNAL_RE = re.compile(
    r"\b("
    r"incorrect|wrong|error|bug|invalid|unsafe|security|privacy|pii|"
    r"contradict|violate|fails?|failure|miscalcul|mismatch|"
    r"typeerror|indexerror|zero\s*division|null|none|overflow|underflow|"
    r"off[-\s]?by[-\s]?one|unsound|false"
    r")\b",
    flags=re.IGNORECASE,
)

_ISSUE_MEDIUM_SIGNAL_RE = re.compile(
    r"\b("
    r"assumption|assumes|constraint|edge\s*case|boundary|"
    r"causal|correlation|complexity|proof|logical|consisten|inconsisten"
    r")\b",
    flags=re.IGNORECASE,
)

_ISSUE_LOW_SIGNAL_RE = re.compile(
    r"\b("
    r"could\s+(?:briefly|explicitly|also\s+)?(?:clarify|mention|state|rephrase|improve)|"
    r"preferable|context[-\s]?dependent|overstated|"
    r"does\s+not\s+explicitly|not\s+explicitly|does\s+not\s+mention|"
    r"does\s+not\s+clarify|no\s+boundary\s+case|irrelevant|"
    r"wording|tone|style|concise|brevity|readability|clarity"
    r")\b",
    flags=re.IGNORECASE,
)

_ISSUE_CODE_HINT_RE = re.compile(
    r"\b(typeerror|indexerror|null|none|len|sum|division|complexity|algorithm)\b",
    flags=re.IGNORECASE,
)

_ARITH_TASK_HINT_RE = re.compile(
    r"\b(compute|calculate|solve|equation|arithmetic|math|sum|product|difference)\b",
    flags=re.IGNORECASE,
)
_ARITH_NEGATIVE_ASSERT_RE = re.compile(
    r"\b(incorrect|wrong|error|not|is\s+not|isn't|does\s+not|false)\b",
    flags=re.IGNORECASE,
)
_ARITH_CLAIM_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-−*/x×÷])\s*(-?\d+(?:\.\d+)?)"
    r"(?:\s*(?:=|equals|is|to|should\s+be|should\s+equal|==)\s*|\s+not\s+)?"
    r"(-?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


def _calc_binary_op(left: float, op: str, right: float) -> Optional[float]:
    op_norm = str(op or "").strip()
    if op_norm in {"+",}:
        return left + right
    if op_norm in {"-", "−"}:
        return left - right
    if op_norm in {"*", "x", "×"}:
        return left * right
    if op_norm in {"/", "÷"}:
        if abs(right) < 1e-12:
            return None
        return left / right
    return None


def _question_looks_arithmetic(question: str) -> bool:
    q = str(question or "")
    if not q.strip():
        return False
    if re.search(r"\d", q) and re.search(r"[+\-−*/x×÷=]", q):
        return True
    return bool(_ARITH_TASK_HINT_RE.search(q))


def _issue_arithmetic_contradiction_penalty(issue_text: str, *, question: str) -> float:
    if not _question_looks_arithmetic(question):
        return 0.0
    text = str(issue_text or "")
    if not _ARITH_NEGATIVE_ASSERT_RE.search(text):
        return 0.0

    penalty = 0.0
    for match in _ARITH_CLAIM_RE.finditer(text):
        try:
            left = float(match.group(1))
            op = str(match.group(2) or "")
            right = float(match.group(3))
            claimed = float(match.group(4))
        except Exception:
            continue
        actual = _calc_binary_op(left, op, right)
        if actual is None:
            continue
        if abs(actual - claimed) <= 1e-9:
            penalty = max(penalty, 3.0)
    return penalty


def _issue_overlap_with_previous(issue_text: str, previous_issues: Sequence[str]) -> float:
    norm = _normalise_issue_text(issue_text)
    if not norm:
        return 0.0
    issue_tokens = _issue_core_tokens(norm)
    if not issue_tokens:
        return 0.0

    best = 0.0
    for prev in previous_issues:
        prev_norm = _normalise_issue_text(prev)
        if not prev_norm:
            continue
        prev_tokens = _issue_core_tokens(prev_norm)
        if prev_tokens:
            inter = len(issue_tokens & prev_tokens)
            union = len(issue_tokens | prev_tokens)
            if union > 0:
                best = max(best, inter / float(union))
            best = max(best, inter / float(min(len(issue_tokens), len(prev_tokens))))
    return best


def _issue_signal_score(issue_text: str, *, question: str = "") -> float:
    norm = _normalise_issue_text(issue_text)
    if not norm:
        return -1.0

    score = 1.0
    if _ISSUE_HIGH_SIGNAL_RE.search(norm):
        score += 1.8
    if _ISSUE_MEDIUM_SIGNAL_RE.search(norm):
        score += 0.8
    if _ISSUE_LOW_SIGNAL_RE.search(norm):
        score -= 1.5

    issue_tokens = _issue_core_tokens(norm)
    question_tokens = _issue_core_tokens(question)
    if issue_tokens and question_tokens:
        overlap = len(issue_tokens & question_tokens)
        if overlap >= 2:
            score += 0.6
        elif overlap == 1:
            score += 0.25

    if "```" in str(question or "") and _ISSUE_CODE_HINT_RE.search(norm):
        score += 0.5

    score -= _issue_arithmetic_contradiction_penalty(norm, question=question)

    return score


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = os.environ.get(name)
    try:
        value = int(str(raw).strip()) if raw is not None else int(default)
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    if maximum is not None:
        value = min(int(maximum), value)
    return value


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = os.environ.get(name)
    try:
        value = float(str(raw).strip()) if raw is not None else float(default)
    except Exception:
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    if maximum is not None:
        value = min(float(maximum), value)
    return value


def _normalise_system2_priority(value: Optional[str], default: str) -> str:
    priority = str(value or "").strip().lower()
    if priority in {"precision", "balanced", "latency"}:
        return priority
    return str(default).strip().lower() or "balanced"


def _resolve_system2_priority(system2_mode: str) -> str:
    mode = str(system2_mode or "auto").strip().lower()
    if mode not in {"auto", "on", "off"}:
        mode = "auto"
    mode_default = "precision" if mode == "auto" else "balanced"
    mode_override = os.environ.get(f"DUALBRAIN_SYSTEM2_PRIORITY_{mode.upper()}")
    if mode_override is not None:
        return _normalise_system2_priority(mode_override, mode_default)
    global_override = os.environ.get("DUALBRAIN_SYSTEM2_PRIORITY")
    if global_override is not None:
        return _normalise_system2_priority(global_override, mode_default)
    return mode_default


def _prioritise_issue_list(
    issues: Sequence[str],
    *,
    question: str,
    limit: int = 8,
    min_score: float = 0.6,
    keep_at_least: int = 0,
) -> List[str]:
    scored: List[tuple[float, int, str]] = []
    for idx, item in enumerate(issues):
        text = _strip_issue_prefix(str(item))
        if not text:
            continue
        scored.append((_issue_signal_score(text, question=question), idx, text))

    if not scored:
        return []

    ranked = sorted(scored, key=lambda row: (row[0], -row[1]), reverse=True)
    selected = [row for row in ranked if row[0] >= float(min_score)]
    if not selected and int(keep_at_least) > 0:
        selected = ranked[: int(keep_at_least)]
    if not selected:
        return []
    selected = sorted(selected, key=lambda row: row[1])[: max(1, int(limit))]
    return [row[2] for row in selected]


def _filter_system2_issues(
    issues: Sequence[str],
    *,
    question: str,
    filter_enabled: bool,
    keep_at_least: int = 0,
    filtered_limit: int = 8,
    raw_limit: int = 12,
) -> List[str]:
    base = [str(item) for item in list(issues)[: max(1, int(raw_limit))] if str(item).strip()]
    if not filter_enabled:
        return base
    return _prioritise_issue_list(
        base,
        question=question,
        limit=filtered_limit,
        keep_at_least=keep_at_least,
    )


def _issue_matches_any(
    issue_norm: str,
    previous_norm: Sequence[str],
    *,
    threshold: float = 0.92,
) -> bool:
    if not issue_norm:
        return False
    issue_tokens = _issue_token_set(issue_norm)
    issue_core_tokens = _issue_core_tokens(issue_norm)
    for prior in previous_norm:
        if not prior:
            continue
        if issue_norm == prior:
            return True
        if len(issue_norm) >= 12 and (issue_norm in prior or prior in issue_norm):
            return True
        if _similarity_ratio(issue_norm, prior) >= threshold:
            return True
        prior_tokens = _issue_token_set(prior)
        if issue_tokens and prior_tokens:
            if min(len(issue_tokens), len(prior_tokens)) >= 4 and (
                issue_tokens.issubset(prior_tokens)
                or prior_tokens.issubset(issue_tokens)
            ):
                return True
            overlap = len(issue_tokens & prior_tokens) / float(len(issue_tokens | prior_tokens))
            if overlap >= 0.72:
                return True
        prior_core_tokens = _issue_core_tokens(prior)
        if issue_core_tokens and prior_core_tokens:
            if min(len(issue_core_tokens), len(prior_core_tokens)) >= 3 and (
                issue_core_tokens.issubset(prior_core_tokens)
                or prior_core_tokens.issubset(issue_core_tokens)
            ):
                return True
            core_overlap = len(issue_core_tokens & prior_core_tokens) / float(
                len(issue_core_tokens | prior_core_tokens)
            )
            if core_overlap >= 0.66:
                return True
    return False


def _normalise_issue_list(
    raw: Any,
    *,
    limit: int = 12,
) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen_norm: List[str] = []
    for item in raw:
        text = _strip_issue_prefix(str(item))
        norm = _normalise_issue_text(text)
        if not norm:
            continue
        if _issue_matches_any(norm, seen_norm):
            continue
        out.append(text)
        seen_norm.append(norm)
        if len(out) >= limit:
            break
    return out


def _novel_issue_items(
    current: Sequence[str],
    previous: Sequence[str],
) -> List[str]:
    prev_norm = [
        _normalise_issue_text(item)
        for item in previous
        if _normalise_issue_text(item)
    ]
    seen_norm: List[str] = []
    novel: List[str] = []
    for item in current:
        cleaned = _strip_issue_prefix(str(item))
        norm = _normalise_issue_text(item)
        if not norm:
            continue
        if _issue_matches_any(norm, prev_norm):
            continue
        if _issue_matches_any(norm, seen_norm):
            continue
        seen_norm.append(norm)
        novel.append(cleaned)
    return novel


def _trim_system2_draft(text: str, limit: int = 2400) -> str:
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)].rstrip() + "..."


def _infer_system2_domain(question: str) -> str:
    q = str(question or "")
    q_lower = q.lower()
    if re.search(r"\breview\b", q_lower) and (
        "snippet" in q_lower
        or "python" in q_lower
        or re.search(r"`[^`]{8,}`", q) is not None
    ):
        return "code_review"
    if re.search(
        r"(latency|error rate|after deployment|deployment|postmortem|triage|"
        r"root[- ]?cause|correlation|causation|causal)",
        q_lower,
    ):
        return "causal_triage"
    if re.search(r"(all\s+a\s+are\s+b|some\s+b\s+are\s+c|does\s+it\s+follow)", q_lower):
        return "logic"
    return "general"


def _system2_revision_guidance(domain: str) -> str:
    domain_norm = str(domain or "").strip().lower()
    if domain_norm == "code_review":
        return (
            "Domain hint: code review.\n"
            "- Do not assume the input is a list; treat it as an iterable.\n"
            "- Cover correctness + edge cases: empty input, non-numeric items, and one-pass iterables.\n"
            "- If you propose a fix, state the intended behavior (e.g., raise ValueError on empty).\n"
        )
    if domain_norm == "causal_triage":
        return (
            "Domain hint: incident triage / causal reasoning.\n"
            "- Avoid assuming specific tooling (feature flags, staging repro); phrase as conditional: \"if available\".\n"
            "- Separate observe → hypothesize → test → confirm. Keep steps actionable.\n"
            "- Include how to distinguish new vs existing errors, and what to do with profiling diffs.\n"
        )
    if domain_norm == "logic":
        return (
            "Domain hint: formal logic.\n"
            "- Be precise about quantifiers (all/some) and provide a counterexample if invalid.\n"
        )
    return (
        "Domain hint: general reasoning.\n"
        "- Address each issue explicitly; if information is missing, state what is missing and give a safe conditional next step.\n"
    )


def _format_system2_revision_notes(
    *,
    question: str,
    issues: Sequence[str],
    fixes: Sequence[str],
    critic_sum: str,
) -> str:
    domain = _infer_system2_domain(question)
    guidance = _system2_revision_guidance(domain)
    issue_block = "\n".join(f"- {item}" for item in issues if str(item).strip()) or "- (none)"
    fix_block = "\n".join(f"- {item}" for item in fixes if str(item).strip()) or "- (none)"
    detail = str(critic_sum or "").strip()
    if not detail:
        detail = "(no critic summary)"
    return (
        "Reasoning critic notes (internal; do not output directly).\n"
        "Goal: improve correctness and resolve the issues.\n"
        f"{guidance}\n"
        "Constraints:\n"
        "- Fix EACH issue explicitly, or state what info is missing and give a safe conditional alternative.\n"
        "- Avoid introducing hard assumptions; use conditional language for environment/tooling specifics.\n"
        "- Prefer concise, structured output.\n\n"
        "Issues (must address):\n"
        f"{issue_block}\n\n"
        "Fix suggestions (if helpful):\n"
        f"{fix_block}\n\n"
        "Critic summary:\n"
        f"{detail}"
    )


_SYSTEM2_CRITIC_UNHEALTHY_MARKERS = (
    "external critic model unavailable",
    "external critic model not configured",
    "unable to verify reasoning",
    "external critic response unstructured",
)

_SYSTEM2_CRITIC_UNHEALTHY_KINDS = {
    "disabled",
}


def _detect_system2_critic_unhealthy_reason(
    *,
    critic_kind: Optional[str],
    issues: Sequence[str],
    critic_sum: Optional[str],
) -> Optional[str]:
    kind = str(critic_kind or "").strip().lower()
    if kind in _SYSTEM2_CRITIC_UNHEALTHY_KINDS:
        return f"critic_kind:{kind}"

    text_parts: List[str] = []
    if critic_sum:
        text_parts.append(str(critic_sum))
    text_parts.extend(str(item) for item in issues if str(item).strip())
    if not text_parts:
        return None

    merged = " ".join(text_parts).lower()
    for marker in _SYSTEM2_CRITIC_UNHEALTHY_MARKERS:
        if marker in merged:
            return marker
    return None


def _looks_like_coaching_notes(question: str, text: str) -> bool:
    """Detect "writing coach" style content that should not leak into user replies."""

    q = str(question or "").strip().lower()
    allow_markers = (
        "rewrite",
        "rephrase",
        "proofread",
        "improve this",
        "make it sound",
        "添削",
        "言い換え",
        "言い方",
        "表現",
        "文章",
        "文面",
        "返事",
    )
    if any(marker in q for marker in allow_markers):
        return False

    t = str(text or "").strip()
    if not t:
        return False

    lower = t.lower()
    coaching_markers = (
        "add a ",
        "add an ",
        "consider ",
        "if the user",
        "you should",
        "try to",
        "make sure to",
        "time of day",
        "もう少し",
        "〜したほうが",
        "したほうが",
        "した方が",
        "時間帯",
        "推測",
        "遊び心",
    )
    if any(marker in lower for marker in coaching_markers):
        return True

    for line in t.splitlines():
        line_norm = line.strip().lower()
        if not line_norm:
            continue
        if line_norm.startswith(("- add ", "- consider ", "add ", "consider ")):
            return True
        if line_norm.startswith(("- 「", "-『", '- "', "- '")) and ("例" in line_norm or "もう少し" in line_norm):
            return True
    return False


def _looks_like_internal_debug_line(line: str) -> bool:
    if not line:
        return False
    raw = line.strip()
    if not raw:
        return False
    lower = raw.lower()
    if lower.startswith("qid "):
        return True
    if lower.startswith("architecture path") or lower.startswith("[architecture path"):
        return True
    if lower.startswith("telemetry (raw)") or lower.startswith("[telemetry"):
        return True
    if lower.startswith("[") and any(
        token in lower
        for token in (
            "left brain",
            "right brain",
            "coherence",
            "unconscious",
            "linguistic",
            "cognitive",
            "psychoid",
            "hemisphere",
            "collaboration",
            "architecture",
        )
    ):
        return True
    if "brain timeout" in lower and "draft" in lower:
        return True
    return False


def _looks_like_writing_coach_line(line: str) -> bool:
    raw = line.strip()
    if not raw:
        return False
    lower = raw.lower()

    if "if the user" in lower:
        return True
    if lower.startswith(("- add a ", "- add an ", "add a ", "add an ")):
        return True
    if lower.startswith(("consider noting the time", "- consider noting the time")):
        return True

    if raw.startswith("-") and ("もう少し" in raw or "時間帯" in raw or "推測" in raw or "例：" in raw):
        return True
    return False


def _sanitize_user_answer(answer: str) -> str:
    if not answer:
        return ""
    lines: List[str] = []
    for line in str(answer).splitlines():
        if _looks_like_internal_debug_line(line) or _looks_like_writing_coach_line(line):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    return cleaned or str(answer).strip()


def _detail_notes_redundant(draft: str, detail_notes: str) -> bool:
    if not draft or not detail_notes:
        return False

    if re.search(r"(^|\n)\s*[-•*]\s+\S", detail_notes):
        return False

    ratio = _similarity_ratio(draft, detail_notes)
    if ratio >= 0.62:
        return len(detail_notes) <= len(draft) * 2.2

    if ratio >= 0.55 and len(detail_notes) <= len(draft) * 1.6:
        return True

    return False


def _format_modules(modules: Sequence[str] | None) -> str:
    if not modules:
        return "∅"
    return ", ".join(modules)


def _summarise_architecture_stage(idx: int, stage: Dict[str, Any]) -> str:
    name = stage.get("stage", f"stage_{idx}")
    modules = _format_modules(stage.get("modules"))
    descriptors: List[str] = []

    if name == "perception":
        signals = stage.get("signals", {})
        affect = signals.get("affect", {}) if isinstance(signals, dict) else {}
        valence = float(affect.get("valence", 0.0))
        arousal = float(affect.get("arousal", 0.0))
        risk = float(affect.get("risk", 0.0))
        novelty = float(affect.get("novelty", 0.0))
        descriptors.append(
            "affect v{:+.2f}/a{:+.2f}/r{:+.2f}/n{:.2f}".format(
                valence, arousal, risk, novelty
            )
        )
        focus = stage.get("focus", {}) or {}
        if isinstance(focus, dict) and focus.get("keywords"):
            keywords = focus.get("keywords", [])
            descriptors.append("focus {}".format(", ".join(str(kw) for kw in keywords[:3])))
            if len(keywords) > 3:
                descriptors.append(f"(+{len(keywords) - 3} more)")
        if isinstance(focus, dict):
            if "relevance" in focus:
                descriptors.append(f"rel {float(focus['relevance']):.2f}")
            if "hippocampal_overlap" in focus:
                descriptors.append(f"hip {float(focus['hippocampal_overlap']):.2f}")
        hemisphere = signals.get("hemisphere") if isinstance(signals, dict) else {}
        if isinstance(hemisphere, dict):
            mode = hemisphere.get("mode", "?")
            bias = float(hemisphere.get("bias", 0.0))
            descriptors.append(f"hemisphere {mode}:{bias:.2f}")
        collaboration = signals.get("collaboration") if isinstance(signals, dict) else {}
        if isinstance(collaboration, dict) and collaboration.get("strength") is not None:
            strength = float(collaboration.get("strength", 0.0))
            balance = float(collaboration.get("balance", 0.0))
            descriptors.append(f"collab {strength:.2f}/{balance:.2f}")
        schema_profile = stage.get("schema_profile")
        if isinstance(schema_profile, dict):
            user_schemas = schema_profile.get("user_schemas", [])
            if user_schemas:
                descriptors.append("schemas {}".format(", ".join(user_schemas[:2])))
            user_modes = schema_profile.get("user_modes", [])
            if user_modes:
                descriptors.append("user modes {}".format(", ".join(user_modes[:2])))
            agent_modes = schema_profile.get("agent_modes", [])
            if agent_modes:
                descriptors.append("agent modes {}".format(", ".join(agent_modes[:2])))
            if "confidence" in schema_profile:
                descriptors.append(f"schema conf {float(schema_profile['confidence']):.2f}")
    elif name == "inner_dialogue":
        leading = stage.get("leading", "?")
        descriptors.append(f"leading {leading}")
        if stage.get("collaborative"):
            descriptors.append("braided")
        step_count = int(stage.get("step_count", 0))
        descriptors.append(f"steps {step_count}")
        phases = stage.get("phases") or []
        if phases:
            phase_list = ", ".join(str(phase) for phase in list(phases)[:4])
            descriptors.append(f"phases {phase_list}")
        temperature = stage.get("temperature")
        if temperature is not None:
            descriptors.append(f"temp {float(temperature):.2f}")
        slot_ms = stage.get("slot_ms")
        if slot_ms is not None:
            descriptors.append(f"slot {int(slot_ms)}ms")
    elif name == "predictive_routing":
        dominant_network = stage.get("dominant_network")
        if dominant_network:
            descriptors.append(f"dominant {dominant_network}")
        top_networks = stage.get("top_networks") or []
        if top_networks:
            descriptors.append("top {}".format(", ".join(str(name) for name in top_networks[:3])))
        phase = stage.get("phase")
        if phase:
            descriptors.append(f"phase {phase}")
        system2_pressure = stage.get("system2_pressure")
        if system2_pressure is not None:
            descriptors.append(f"s2 {float(system2_pressure):.2f}")
        prediction_error = stage.get("prediction_error") or {}
        if isinstance(prediction_error, dict):
            dominant_error = prediction_error.get("dominant_channel")
            overall = prediction_error.get("overall")
            if dominant_error:
                descriptors.append(f"err {dominant_error}")
            if overall is not None:
                descriptors.append(f"overall {float(overall):.2f}")
    elif name == "integration":
        descriptors.append("success" if stage.get("success") else "retry")
        coherence = stage.get("coherence")
        if isinstance(coherence, dict):
            if "combined" in coherence:
                descriptors.append(f"coh {float(coherence['combined']):.2f}")
            if "tension" in coherence:
                descriptors.append(f"ten {float(coherence['tension']):.2f}")
        distortion = stage.get("distortion")
        if isinstance(distortion, dict):
            flags = distortion.get("flags", [])
            if flags:
                descriptors.append("distortions {}".format(", ".join(flags[:3])))
            if "score" in distortion:
                descriptors.append(f"distortion {float(distortion['score']):.2f}")
    elif name == "memory":
        tags = stage.get("tags", []) or []
        descriptors.append(f"tags {len(tags)}")
        rollup = stage.get("hippocampal_rollup") or {}
        if isinstance(rollup, dict) and rollup:
            if "avg_strength" in rollup:
                descriptors.append(f"avg {float(rollup['avg_strength']):.2f}")
            if "strength_coverage" in rollup:
                descriptors.append(f"coverage {float(rollup['strength_coverage']):.2f}")
            lead_parts: List[str] = []
            for key, label in (
                ("lead_left", "L"),
                ("lead_right", "R"),
                ("lead_braided", "B"),
            ):
                val = rollup.get(key)
                if val:
                    lead_parts.append(f"{label}{float(val):.2f}")
            if lead_parts:
                descriptors.append("lead mix " + " ".join(lead_parts))

    descriptor_text = "; ".join(descriptors)
    if descriptor_text:
        return f"{idx}. {name}: {modules} | {descriptor_text}"
    return f"{idx}. {name}: {modules}"


def _summarise_architecture_path(path: Sequence[Dict[str, Any]]) -> List[str]:
    return [_summarise_architecture_stage(idx, stage) for idx, stage in enumerate(path, 1)]


@dataclass
class DecisionOutcome:
    """Details about a single orchestration decision."""

    qid: str
    action: int
    temperature: float
    slot_ms: int
    state: Dict[str, Any]


@dataclass
class HemisphericSignal:
    """Raw hemisphere cue scoring derived from the question and focus."""

    mode: str
    bias: float
    right_score: float
    left_score: float
    token_count: int

    def to_payload(self) -> Dict[str, float]:
        return {
            "mode": self.mode,
            "bias": float(self.bias),
            "right_score": float(self.right_score),
            "left_score": float(self.left_score),
            "token_count": float(self.token_count),
        }

    @property
    def total(self) -> float:
        return self.right_score + self.left_score


@dataclass
class CollaborationProfile:
    """Aggregated assessment of how strongly both hemispheres want to co-lead."""

    strength: float
    balance: float
    density: float
    focus_bonus: float
    token_count: int

    def to_payload(self) -> Dict[str, float]:
        return {
            "strength": float(self.strength),
            "balance": float(self.balance),
            "density": float(self.density),
            "focus_bonus": float(self.focus_bonus),
            "token_count": float(self.token_count),
        }


@dataclass
class InnerDialogueStep:
    """Single step captured during an inner dialogue exchange."""

    phase: str
    role: str
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "phase": self.phase,
            "role": self.role,
            "ts": float(self.timestamp),
        }
        if self.content:
            payload["content"] = self.content
        if self.metadata:
            payload["meta"] = dict(self.metadata)
        return payload
