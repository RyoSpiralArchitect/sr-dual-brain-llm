# ============================================================================
#  SpiralReality Proprietary
#  Copyright (c) 2025 SpiralReality. All Rights Reserved.
#
#  NOTICE: This file contains confidential and proprietary information of
#  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
#  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
#  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
#
#  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
#  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
#  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
#  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ============================================================================

"""Simplified executive-control heuristics inspired by the prefrontal cortex."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _tokenise(text: str) -> List[str]:
    return [tok for tok in text.lower().replace("\n", " ").split(" ") if tok]


def _top_keywords(tokens: Sequence[str], *, limit: int = 6) -> Tuple[str, ...]:
    seen = []
    for tok in tokens:
        if len(tok) < 3:
            continue
        if tok in seen:
            continue
        seen.append(tok)
        if len(seen) >= limit:
            break
    return tuple(seen)


def _line_relevance(line: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 0.0
    tokens = set(_tokenise(line))
    if not tokens:
        return 0.0
    overlap = len(tokens & set(keywords))
    if not overlap:
        return 0.0
    return overlap / max(len(tokens), 1)


_USER_SCHEMA_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "abandonment": (
        "abandon",
        "leave",
        "leaves",
        "leaving",
        "alone",
        "lonely",
        "forsaken",
        "reject",
        "everyone",
        "everybody",
        "孤独",
        "見捨",
        "失う",
    ),
    "mistrust": (
        "betray",
        "lie",
        "deceive",
        "trick",
        "裏切",
        "騙",
        "信用",
    ),
    "failure": (
        "fail",
        "failure",
        "useless",
        "worthless",
        "loser",
        "inept",
        "失敗",
        "無価値",
        "役立たず",
    ),
    "perfectionism": (
        "perfect",
        "must",
        "should",
        "always",
        "never",
        "absolutely",
        "ought",
        "完璧",
        "絶対",
        "べき",
    ),
    "vulnerability": (
        "danger",
        "unsafe",
        "harm",
        "threat",
        "scared",
        "afraid",
        "怖",
        "危険",
    ),
    "self_sacrifice": (
        "sacrifice",
        "please",
        "others",
        "serve",
        "give",
        "助け",
        "尽く",
    ),
}


_USER_MODE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "vulnerable_child": (
        "sad",
        "alone",
        "hurt",
        "scared",
        "worried",
        "anxious",
        "lonely",
        "不安",
        "悲しい",
        "怖い",
    ),
    "angry_child": (
        "angry",
        "furious",
        "rage",
        "irritated",
        "mad",
        "怒",
        "苛立",
    ),
    "compliant_surrender": (
        "sorry",
        "obey",
        "should",
        "must",
        "please",
        "従",
        "べき",
        "申し訳",
    ),
    "healthy_adult": (
        "plan",
        "balance",
        "consider",
        "reflect",
        "step",
        "整える",
        "整理",
        "進め",
    ),
}


_AGENT_MODE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "healthy_adult": (
        "consider",
        "together",
        "let's",
        "we",
        "balance",
        "step",
        "plan",
        "進め",
        "一緒",
    ),
    "nurturing_parent": (
        "support",
        "care",
        "understand",
        "comfort",
        "compassion",
        "empath",
        "励ま",
        "寄り添",
    ),
    "scientist": (
        "analysis",
        "evidence",
        "data",
        "logic",
        "derive",
        "解析",
        "論理",
        "証拠",
    ),
    "coach": (
        "goal",
        "practice",
        "exercise",
        "action",
        "habit",
        "課題",
        "目標",
    ),
}


_ABSOLUTE_PATTERN = re.compile(r"\b(always|never|everybody|nobody|必ず|絶対)\b", re.IGNORECASE)
_LABEL_PATTERN = re.compile(
    r"\b(i am|i'm|we are|you are|they are|私は|俺は|私は)\s+(?:a\s+)?(failure|loser|worthless|stupid|useless|無価値|ダメ)\b",
    re.IGNORECASE,
)


@dataclass
class SchemaProfile:
    """Heuristic schema/mode profile for a single dialogue turn."""

    user_schemas: Tuple[str, ...] = tuple()
    user_modes: Tuple[str, ...] = tuple()
    agent_modes: Tuple[str, ...] = tuple()
    schema_scores: Dict[str, float] = field(default_factory=dict)
    mode_scores: Dict[str, float] = field(default_factory=dict)
    agent_mode_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_schemas": list(self.user_schemas),
            "user_modes": list(self.user_modes),
            "agent_modes": list(self.agent_modes),
            "schema_scores": dict(self.schema_scores),
            "mode_scores": dict(self.mode_scores),
            "agent_mode_scores": dict(self.agent_mode_scores),
            "confidence": float(self.confidence),
            "notes": list(self.notes),
        }

    def is_neutral(self) -> bool:
        return not (self.user_schemas or self.user_modes or self.agent_modes)

    def tags(self) -> Iterable[str]:
        tags = {"schema_profile"}
        if self.user_schemas:
            tags.update(f"schema_user_{schema}" for schema in self.user_schemas)
        else:
            tags.add("schema_user_neutral")
        if self.user_modes:
            tags.update(f"mode_user_{mode}" for mode in self.user_modes)
        else:
            tags.add("mode_user_neutral")
        if self.agent_modes:
            tags.update(f"mode_agent_{mode}" for mode in self.agent_modes)
        else:
            tags.add("mode_agent_neutral")
        if self.confidence >= 0.65:
            tags.add("schema_confidence_high")
        elif self.confidence >= 0.4:
            tags.add("schema_confidence_mid")
        else:
            tags.add("schema_confidence_low")
        tags.add(f"schema_conf_{int(self.confidence * 100):02d}")
        return tuple(sorted(tags))


class SchemaProfiler:
    """Infer CBT-style schemas and modes using lightweight keyword heuristics."""

    def __init__(
        self,
        *,
        schema_threshold: float = 0.18,
        mode_threshold: float = 0.22,
        agent_threshold: float = 0.22,
        schema_keywords: Optional[Dict[str, Sequence[str]]] = None,
        mode_keywords: Optional[Dict[str, Sequence[str]]] = None,
        agent_keywords: Optional[Dict[str, Sequence[str]]] = None,
    ) -> None:
        self.schema_threshold = schema_threshold
        self.mode_threshold = mode_threshold
        self.agent_threshold = agent_threshold
        self.schema_keywords = self._normalise_mapping(
            schema_keywords or _USER_SCHEMA_KEYWORDS
        )
        self.mode_keywords = self._normalise_mapping(
            mode_keywords or _USER_MODE_KEYWORDS
        )
        self.agent_keywords = self._normalise_mapping(
            agent_keywords or _AGENT_MODE_KEYWORDS
        )

    @staticmethod
    def _normalise_mapping(mapping: Dict[str, Sequence[str]]) -> Dict[str, Tuple[str, ...]]:
        normalised: Dict[str, Tuple[str, ...]] = {}
        for key, values in mapping.items():
            normalised[key] = tuple(sorted({value.lower() for value in values if value}))
        return normalised

    @staticmethod
    def _score(tokens: Sequence[str], keywords: Tuple[str, ...]) -> float:
        if not keywords:
            return 0.0
        token_set = set(tokens)
        hits = len(token_set & set(keywords))
        if not hits:
            return 0.0
        base = hits / max(len(keywords), 1)
        coverage = hits / max(len(token_set), 1)
        score = 0.65 * base + 0.35 * coverage
        return max(0.0, min(1.0, score))

    def _score_map(
        self, tokens: Sequence[str], mapping: Dict[str, Tuple[str, ...]]
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for label, keywords in mapping.items():
            score = self._score(tokens, keywords)
            if score > 0.0:
                scores[label] = score
        return scores

    def _select_labels(
        self, scores: Dict[str, float], threshold: float
    ) -> Tuple[str, ...]:
        filtered = [(label, score) for label, score in scores.items() if score >= threshold]
        filtered.sort(key=lambda item: item[1], reverse=True)
        return tuple(label for label, _ in filtered)

    def profile_turn(
        self,
        *,
        question: str,
        answer: str,
        focus: Optional["FocusSummary"] = None,
        affect: Optional[Dict[str, float]] = None,
    ) -> SchemaProfile:
        q_tokens = _tokenise(question)
        a_tokens = _tokenise(answer)
        q_lower = question.lower()
        a_lower = answer.lower()
        notes: List[str] = []

        schema_scores = self._score_map(q_tokens, self.schema_keywords)
        if _ABSOLUTE_PATTERN.search(q_lower):
            boost = max(0.28, schema_scores.get("perfectionism", 0.0))
            schema_scores["perfectionism"] = boost
            notes.append("Detected absolute language suggesting perfectionism")
        if _LABEL_PATTERN.search(q_lower):
            boost = max(0.35, schema_scores.get("failure", 0.0))
            schema_scores["failure"] = boost
            notes.append("Detected negative self-labelling")

        mode_scores = self._score_map(q_tokens, self.mode_keywords)
        agent_mode_scores = self._score_map(a_tokens, self.agent_keywords)

        if focus and focus.keywords:
            focus_set = {kw.lower() for kw in focus.keywords}
            for label, keywords in self.schema_keywords.items():
                if focus_set & set(keywords):
                    schema_scores[label] = max(
                        schema_scores.get(label, 0.0),
                        self.schema_threshold + 0.05 * focus.relevance,
                    )
                    notes.append(f"Focus keyword overlap with schema '{label}'")
            for label, keywords in self.mode_keywords.items():
                if focus_set & set(keywords):
                    mode_scores[label] = max(
                        mode_scores.get(label, 0.0),
                        self.mode_threshold + 0.05 * focus.hippocampal_overlap,
                    )

        if affect:
            valence = float(affect.get("valence", 0.0))
            arousal = float(affect.get("arousal", 0.0))
            if valence < -0.25:
                mode_scores["vulnerable_child"] = max(
                    mode_scores.get("vulnerable_child", 0.0), 0.28 - valence * 0.2
                )
                notes.append("Negative valence nudged vulnerable-child mode")
            if arousal > 0.55:
                mode_scores["angry_child"] = max(
                    mode_scores.get("angry_child", 0.0), 0.22 + (arousal - 0.55)
                )
                notes.append("High arousal nudged angry-child mode")

        if "let us" in a_lower or "let's" in a_lower or "we can" in a_lower:
            agent_mode_scores["healthy_adult"] = max(
                agent_mode_scores.get("healthy_adult", 0.0), 0.32
            )
            notes.append("Collaborative language implies healthy adult mode")
        if "i care" in a_lower or "i understand" in a_lower or "お力" in a_lower:
            agent_mode_scores["nurturing_parent"] = max(
                agent_mode_scores.get("nurturing_parent", 0.0), 0.3
            )

        user_schemas = self._select_labels(schema_scores, self.schema_threshold)
        user_modes = self._select_labels(mode_scores, self.mode_threshold)
        agent_modes = self._select_labels(agent_mode_scores, self.agent_threshold)

        confidence = max(
            [0.0]
            + [schema_scores.get(label, 0.0) for label in user_schemas]
            + [mode_scores.get(label, 0.0) for label in user_modes]
            + [agent_mode_scores.get(label, 0.0) for label in agent_modes]
        )

        return SchemaProfile(
            user_schemas=user_schemas,
            user_modes=user_modes,
            agent_modes=agent_modes,
            schema_scores=schema_scores,
            mode_scores=mode_scores,
            agent_mode_scores=agent_mode_scores,
            confidence=confidence,
            notes=notes,
        )
@dataclass
class FocusSummary:
    """Compact representation of the current executive focus."""

    keywords: Tuple[str, ...]
    relevance: float
    hippocampal_overlap: float

    def to_dict(self) -> Dict[str, float | Sequence[str]]:
        return {
            "keywords": list(self.keywords),
            "relevance": float(self.relevance),
            "hippocampal_overlap": float(self.hippocampal_overlap),
        }


class PrefrontalCortex:
    """Approximate the gating and focusing role of the prefrontal cortex."""

    def __init__(
        self,
        *,
        min_keywords: int = 2,
        gating_threshold: float = 0.25,
        schema_profiler: Optional[SchemaProfiler] = None,
    ) -> None:
        self.min_keywords = min_keywords
        self.gating_threshold = gating_threshold
        self.schema_profiler = schema_profiler or SchemaProfiler()

    def synthesise_focus(
        self,
        *,
        question: str,
        memory_context: str,
        hippocampal_context: str = "",
    ) -> FocusSummary:
        q_tokens = _tokenise(question)
        keywords = _top_keywords(q_tokens)

        memory_lines = [ln for ln in memory_context.split("\n") if ln.strip()]
        hippocampal_lines = [ln for ln in hippocampal_context.split("\n") if ln.strip()]

        if not keywords and memory_lines:
            # Derive fallback keywords from memory if question is sparse.
            memory_tokens: List[str] = []
            for line in memory_lines:
                memory_tokens.extend(_tokenise(line))
            keywords = _top_keywords(memory_tokens)

        if len(keywords) < self.min_keywords:
            # Pad with most informative memory tokens when available.
            aggregate_tokens: List[str] = []
            for line in memory_lines:
                aggregate_tokens.extend(_tokenise(line))
            for tok in aggregate_tokens:
                if tok in keywords:
                    continue
                if len(tok) < 3:
                    continue
                keywords = tuple(list(keywords) + [tok])
                if len(keywords) >= self.min_keywords:
                    break

        line_scores = [_line_relevance(line, keywords) for line in memory_lines]
        if line_scores:
            relevance = sum(line_scores) / max(len(memory_lines), 1)
        else:
            relevance = 0.0

        hippocampal_scores = [_line_relevance(line, keywords) for line in hippocampal_lines]
        if hippocampal_scores:
            hippocampal_overlap = sum(hippocampal_scores) / max(len(hippocampal_lines), 1)
        else:
            hippocampal_overlap = 0.0

        # Normalise to [0, 1]
        relevance = max(0.0, min(1.0, relevance))
        hippocampal_overlap = max(0.0, min(1.0, hippocampal_overlap))

        return FocusSummary(keywords=keywords, relevance=relevance, hippocampal_overlap=hippocampal_overlap)

    @staticmethod
    def is_trivial_chat_turn(question: str) -> bool:
        """Return True when the user input is likely lightweight chit-chat.

        This is used as an executive override to avoid unnecessary consults
        (latency/cost) and reduce "split voice" failure modes on greetings.
        """

        q = (question or "").strip()
        if not q:
            return True
        if len(q) > 36:
            return False

        q_lower = q.lower()
        if re.search(r"\d", q_lower):
            return False
        if "http://" in q_lower or "https://" in q_lower:
            return False
        if any(token in q_lower for token in ("analy", "analysis", "explain", "derive", "calculate", "proof", "why", "how")):
            return False
        if any(token in q for token in ("分析", "計算", "証明", "なぜ", "どうやって")):
            return False

        exact = {
            "hi",
            "hey",
            "hello",
            "yo",
            "sup",
            "やあ",
            "こんにちは",
            "こんばんは",
            "おはよう",
            "もしもし",
            "元気",
            "元気？",
            "なに",
            "なに？",
            "何",
            "何？",
        }
        if q_lower in exact or q in exact:
            return True

        if re.fullmatch(r"[!?！？。…]+", q):
            return True

        starters = (
            "やあ",
            "こんにちは",
            "おはよう",
            "こんばんは",
            "もしもし",
            "hi",
            "hello",
            "hey",
        )
        if any(q_lower.startswith(prefix) for prefix in starters) and len(q_lower) <= 20:
            return True

        return False

    def gate_context(self, context: str, focus: FocusSummary) -> str:
        if not context:
            return context
        if not focus.keywords or focus.relevance < self.gating_threshold:
            return context

        lines = [ln for ln in context.split("\n") if ln.strip()]
        if not lines:
            return context

        gated: List[str] = []
        for line in lines:
            if line.startswith("[Hippocampal"):
                gated.append(line)
                continue
            if _line_relevance(line, focus.keywords) >= self.gating_threshold / 2:
                gated.append(line)
        if gated:
            return "\n".join(gated)
        return context

    def adjust_consult_bias(self, current_bias: float, focus: FocusSummary) -> float:
        """Adjust the consult bias based on perceived focus quality."""

        # If focus is diffuse, encourage right-brain consultation slightly more.
        if focus.relevance < 0.2:
            current_bias += 0.1
        # Strong hippocampal overlap implies supportive episodic cues; reduce bias.
        if focus.hippocampal_overlap > 0.45:
            current_bias -= 0.05
        return max(-1.0, min(1.0, current_bias))

    def focus_metric(self, focus: FocusSummary) -> float:
        """Return a scalar coherence metric useful for policy state."""

        combined = 0.6 * focus.relevance + 0.4 * focus.hippocampal_overlap
        return max(0.0, min(1.0, combined))

    def tags(self, focus: FocusSummary) -> Iterable[str]:
        if focus.relevance >= 0.5:
            yield "focus_high"
        elif focus.relevance < 0.2:
            yield "focus_low"
        if focus.hippocampal_overlap >= 0.5:
            yield "hippocampal_supported"
        if focus.keywords:
            for kw in focus.keywords[:3]:
                yield f"focus_{kw}"

    def profile_turn(
        self,
        *,
        question: str,
        answer: str,
        focus: Optional[FocusSummary] = None,
        affect: Optional[Dict[str, float]] = None,
    ) -> SchemaProfile:
        """Delegate schema/mode inference to the configured profiler."""

        return self.schema_profiler.profile_turn(
            question=question,
            answer=answer,
            focus=focus,
            affect=affect,
        )
