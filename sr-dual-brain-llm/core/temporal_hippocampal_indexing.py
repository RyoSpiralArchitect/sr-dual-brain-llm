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

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


EMBEDDING_VERSION = "srdb-blake2b-v1"

_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u3040-\u30ff\u4e00-\u9fff]+", re.UNICODE)
_KANJI_RE = re.compile(r"[\u4e00-\u9fff]")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "he",
    "her",
    "him",
    "his",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}
_INTERNAL_DEBUG_TOKENS = (
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
    "telemetry",
)


def _tokenize(text: str) -> List[str]:
    lowered = (text or "").replace("\n", " ").replace("\t", " ").lower()
    tokens: List[str] = []
    for tok in _WORD_RE.findall(lowered):
        tok = tok.strip()
        if not tok:
            continue
        tokens.append(tok)
        if _KANJI_RE.search(tok) and len(tok) >= 2:
            # Add kanji bigrams to recover overlap signals in CJK text without
            # introducing the noisy hiragana/katakana n-grams that cause
            # unrelated "sticky" recalls.
            for i in range(len(tok) - 1):
                a = tok[i]
                b = tok[i + 1]
                if _KANJI_RE.match(a) and _KANJI_RE.match(b):
                    tokens.append(a + b)
    return tokens


def _normalise_tags(tags: Iterable[str] | None) -> Tuple[str, ...]:
    if not tags:
        return ()
    unique = {tag.strip().lower() for tag in tags if tag and tag.strip()}
    return tuple(sorted(unique))


def _stable_hash_64(token: str) -> int:
    digest = hashlib.blake2b(
        token.encode("utf-8"),
        digest_size=8,
        person=b"srdb.emb.v1",
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def _looks_like_internal_debug_line(line: str) -> bool:
    raw = str(line or "").strip()
    if not raw:
        return False
    lower = raw.lower()
    if lower.startswith("qid "):
        return True
    if lower.startswith("architecture path") or lower.startswith("[architecture path"):
        return True
    if lower.startswith("telemetry (raw)") or lower.startswith("[telemetry"):
        return True
    if lower.startswith("[") and any(token in lower for token in _INTERNAL_DEBUG_TOKENS):
        return True
    if "brain timeout" in lower and "draft" in lower:
        return True
    return False


def _looks_like_writing_coach_line(line: str) -> bool:
    raw = str(line or "").strip()
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


def _sanitize_memory_text(text: str) -> str:
    if not text:
        return ""
    lines: List[str] = []
    for line in str(text).splitlines():
        if _looks_like_internal_debug_line(line) or _looks_like_writing_coach_line(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


@dataclass
class EpisodicTrace:
    """Single hippocampal memory containing hemispheric collaboration context."""

    qid: str
    question: str
    answer: str
    vector: np.ndarray
    embedding_version: str = EMBEDDING_VERSION
    timestamp: float = field(default_factory=lambda: time.time())
    leading: Optional[str] = None
    collaboration_strength: Optional[float] = None
    selection_reason: Optional[str] = None
    tags: Tuple[str, ...] = field(default_factory=tuple)
    annotations: Dict[str, Any] = field(default_factory=dict)
    consolidation_score: float = 0.35
    retrieval_count: int = 0
    replay_count: int = 0
    last_accessed_at: float = field(default_factory=lambda: time.time())
    last_replayed_at: Optional[float] = None

    def payload(self) -> str:
        return f"Q: {self.question}\nA: {self.answer}"

    def summary(
        self,
        *,
        similarity: Optional[float] = None,
        max_chars: int = 240,
        include_meta: bool = True,
    ) -> str:
        parts: List[str] = []
        if include_meta:
            if similarity is not None:
                parts.append(f"sim={similarity:.2f}")
            if self.leading:
                parts.append(f"lead={self.leading}")
            if self.collaboration_strength is not None:
                parts.append(f"collab={self.collaboration_strength:.2f}")
            if self.selection_reason:
                parts.append(self.selection_reason)
        prefix = f"({'; '.join(parts)}) " if parts else ""
        question_clean = _sanitize_memory_text(self.question).replace("\n", " ").strip()
        answer_clean = _sanitize_memory_text(self.answer).replace("\n", " ").strip()
        if not question_clean or not answer_clean:
            return ""
        answer_snippet = answer_clean[:max_chars]
        return f"{prefix}Q: {question_clean[:max_chars]} | A: {answer_snippet}"


class TemporalHippocampalIndexing:
    def __init__(
        self,
        dim: int = 128,
        *,
        min_lexical_overlap: float | None = None,
    ):
        self.dim = dim
        self.episodes: List[EpisodicTrace] = []
        self._eps = 1e-8
        self._min_lexical_overlap = self._resolve_min_lexical_overlap(min_lexical_overlap)

    @staticmethod
    def _resolve_min_lexical_overlap(value: float | None) -> float:
        if value is None:
            raw = str(os.environ.get("DUALBRAIN_HIPPO_MIN_LEXICAL", "0.0") or "0.0")
            try:
                value = float(raw)
            except Exception:
                value = 0.0
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _content_tokens(tokens: List[str]) -> set[str]:
        filtered: set[str] = set()
        for tok in tokens or []:
            tok = str(tok or "").strip().lower()
            if not tok:
                continue
            if tok in _STOPWORDS:
                continue
            # Avoid matching purely on punctuation/very short fragments.
            if len(tok) <= 1:
                continue
            filtered.add(tok)
        return filtered

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text.lower()):
            v[_stable_hash_64(tok) % self.dim] += 1.0
        n = float(np.linalg.norm(v) + self._eps)
        return v / n

    def embed_text(self, text: str) -> np.ndarray:
        return self._embed(text)

    def index_episode(
        self,
        qid: str,
        question: str,
        answer: str,
        *,
        leading: Optional[str] = None,
        collaboration_strength: Optional[float] = None,
        selection_reason: Optional[str] = None,
        tags: Iterable[str] | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = f"Q: {question}\nA: {answer}"
        vec = self._embed(payload)
        norm_tags = _normalise_tags(tags)
        annotations: Dict[str, Any] = dict(metadata or {})
        if "hemisphere_mode" in annotations:
            annotations["hemisphere_mode"] = str(annotations["hemisphere_mode"])
        if "hemisphere_bias" in annotations and annotations["hemisphere_bias"] is not None:
            annotations["hemisphere_bias"] = float(annotations["hemisphere_bias"])
        salience_level = 0.0
        if annotations.get("salience_level") is not None:
            try:
                salience_level = max(0.0, min(1.0, float(annotations["salience_level"])))
            except Exception:
                salience_level = 0.0
        reward = 0.0
        if annotations.get("reward") is not None:
            try:
                reward = max(0.0, min(1.0, float(annotations["reward"])))
            except Exception:
                reward = 0.0
        resolved = 1.0 if annotations.get("conflict_resolved") else 0.0
        base_consolidation = max(
            0.15,
            min(
                1.0,
                0.28
                + 0.22 * float(collaboration_strength or 0.0)
                + 0.20 * salience_level
                + 0.18 * reward
                + 0.10 * resolved,
            ),
        )
        trace = EpisodicTrace(
            qid=qid,
            question=question,
            answer=answer,
            vector=vec,
            embedding_version=EMBEDDING_VERSION,
            leading=(leading or None),
            collaboration_strength=None
            if collaboration_strength is None
            else float(collaboration_strength),
            selection_reason=selection_reason,
            tags=norm_tags,
            annotations=annotations,
            consolidation_score=base_consolidation,
        )
        self.episodes.append(trace)

    def retrieve(self, query: str, topk: int = 3) -> List[Tuple[float, EpisodicTrace]]:
        if not self.episodes:
            return []
        qv = self._embed(query)
        query_tokens = self._content_tokens(_tokenize(query))
        if not query_tokens:
            return []
        scored: List[Tuple[float, float, int, float, EpisodicTrace]] = []
        for idx, trace in enumerate(self.episodes):
            sim = float(np.dot(qv, trace.vector))
            lexical = 0.0
            if query_tokens:
                trace_tokens = self._content_tokens(_tokenize(trace.question))
                overlap = len(query_tokens & trace_tokens)
                lexical = overlap / max(len(query_tokens), 1)

            if lexical <= 0.0:
                # The lightweight embedding uses feature hashing, which can
                # yield false positives via collisions. Require at least some
                # lexical overlap to avoid "sticky" irrelevant recalls.
                continue
            if lexical < self._min_lexical_overlap:
                continue
            combined = sim + 0.35 * lexical
            scored.append((combined, lexical, -idx, sim, trace))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        hits = [(sim, trace) for _, _, _, sim, trace in scored[:topk]]
        now = time.time()
        for sim, trace in hits:
            trace.retrieval_count += 1
            trace.last_accessed_at = now
            trace.consolidation_score = min(
                1.0,
                float(trace.consolidation_score) + 0.03 + 0.04 * max(0.0, sim),
            )
        return hits

    def retrieve_summary(
        self,
        query: str,
        topk: int = 3,
        max_chars: int = 240,
        *,
        include_meta: bool = True,
    ) -> str:
        hits = self.retrieve(query, topk=topk)
        if not hits:
            return ""
        parts = []
        for sim, trace in hits:
            summary = trace.summary(
                similarity=(sim if include_meta else None),
                max_chars=max_chars,
                include_meta=include_meta,
            )
            if summary:
                parts.append(summary)
        return " | ".join(parts)

    def collaboration_rollup(self, window: int = 10) -> Dict[str, float]:
        if not self.episodes:
            return {
                "window": 0.0,
                "avg_strength": 0.0,
                "lead_left": 0.0,
                "lead_right": 0.0,
                "lead_braided": 0.0,
                "strength_coverage": 0.0,
                "avg_consolidation": 0.0,
                "avg_retrievals": 0.0,
                "avg_replays": 0.0,
            }
        window = max(1, min(window, len(self.episodes)))
        subset = self.episodes[-window:]
        strengths = [
            trace.collaboration_strength
            for trace in subset
            if trace.collaboration_strength is not None
        ]
        avg_strength = sum(strengths) / len(strengths) if strengths else 0.0
        avg_consolidation = sum(float(trace.consolidation_score) for trace in subset) / len(subset)
        avg_retrievals = sum(int(trace.retrieval_count) for trace in subset) / len(subset)
        avg_replays = sum(int(trace.replay_count) for trace in subset) / len(subset)
        lead_counts = {
            "left": 0,
            "right": 0,
            "braided": 0,
        }
        for trace in subset:
            if trace.leading in lead_counts:
                lead_counts[trace.leading] += 1
        coverage = len(strengths) / window
        return {
            "window": float(window),
            "avg_strength": float(avg_strength),
            "lead_left": float(lead_counts["left"]) / window,
            "lead_right": float(lead_counts["right"]) / window,
            "lead_braided": float(lead_counts["braided"]) / window,
            "strength_coverage": float(coverage),
            "avg_consolidation": float(avg_consolidation),
            "avg_retrievals": float(avg_retrievals),
            "avg_replays": float(avg_replays),
        }

    def replay_candidates(
        self,
        query: str,
        *,
        topk: int = 2,
    ) -> List[Tuple[float, EpisodicTrace]]:
        hits = self.retrieve(query, topk=max(topk * 3, topk))
        if not hits:
            return []
        now = time.time()
        scored: List[Tuple[float, float, EpisodicTrace]] = []
        for sim, trace in hits:
            age_seconds = max(1.0, now - float(trace.timestamp))
            freshness = max(0.0, 1.0 - min(age_seconds, 86400.0) / 86400.0)
            replay_novelty = 1.0 / (1.0 + float(trace.replay_count))
            score = (
                0.50 * max(0.0, sim)
                + 0.25 * float(trace.consolidation_score)
                + 0.15 * replay_novelty
                + 0.10 * freshness
            )
            scored.append((score, sim, trace))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = scored[:topk]
        if selected:
            replay_now = time.time()
            for _, _, trace in selected:
                trace.replay_count += 1
                trace.last_replayed_at = replay_now
                trace.consolidation_score = min(
                    1.0,
                    float(trace.consolidation_score) + 0.05,
                )
        return [(sim, trace) for _, sim, trace in selected]

    def replay_summary(
        self,
        query: str,
        *,
        topk: int = 2,
        max_chars: int = 180,
        include_meta: bool = True,
    ) -> str:
        hits = self.replay_candidates(query, topk=topk)
        if not hits:
            return ""
        parts: List[str] = []
        for sim, trace in hits:
            summary = trace.summary(
                similarity=(sim if include_meta else None),
                max_chars=max_chars,
                include_meta=include_meta,
            )
            if summary:
                parts.append(summary)
        return " | ".join(parts)

    def consolidate_feedback(
        self,
        *,
        qid: str,
        reward: float,
        success: bool,
        conflict_resolved: bool = False,
        system2_used: bool = False,
        salience_level: float = 0.0,
    ) -> Optional[Dict[str, float]]:
        for trace in reversed(self.episodes):
            if str(trace.qid) != str(qid):
                continue
            reward = max(0.0, min(1.0, float(reward)))
            salience_level = max(0.0, min(1.0, float(salience_level)))
            before = float(trace.consolidation_score)
            delta = (
                0.14 * reward
                + (0.08 if success else -0.03)
                + (0.10 if conflict_resolved else 0.0)
                + (0.04 if system2_used else 0.0)
                + 0.05 * salience_level
            )
            trace.consolidation_score = max(0.05, min(1.0, before + delta))
            trace.annotations["reward"] = reward
            trace.annotations["success"] = bool(success)
            trace.annotations["conflict_resolved"] = bool(conflict_resolved)
            trace.annotations["system2_used"] = bool(system2_used)
            trace.annotations["salience_level"] = salience_level
            trace.last_accessed_at = time.time()
            return {
                "before": float(before),
                "after": float(trace.consolidation_score),
                "reward": float(reward),
                "success": 1.0 if success else 0.0,
                "conflict_resolved": 1.0 if conflict_resolved else 0.0,
                "system2_used": 1.0 if system2_used else 0.0,
                "salience_level": float(salience_level),
            }
        return None

    def forget_stale(
        self,
        *,
        max_episodes: int = 256,
        min_keep: int = 64,
    ) -> Dict[str, float]:
        total_before = len(self.episodes)
        if total_before <= max_episodes:
            return {
                "before": float(total_before),
                "after": float(total_before),
                "forgotten": 0.0,
            }
        min_keep = max(1, min(int(min_keep), total_before))
        protected = self.episodes[-min_keep:]
        candidates = self.episodes[:-min_keep]
        scored: List[Tuple[float, int, EpisodicTrace]] = []
        now = time.time()
        for idx, trace in enumerate(candidates):
            age_seconds = max(1.0, now - float(trace.timestamp))
            age_penalty = min(age_seconds / 86400.0, 30.0)
            score = (
                float(trace.consolidation_score)
                + 0.03 * float(trace.retrieval_count)
                + 0.04 * float(trace.replay_count)
                - 0.02 * age_penalty
            )
            scored.append((score, idx, trace))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        keep_budget = max_episodes - len(protected)
        kept_candidates = [trace for _, _, trace in scored[:keep_budget]]
        self.episodes = kept_candidates + protected
        forgotten = max(0, total_before - len(self.episodes))
        return {
            "before": float(total_before),
            "after": float(len(self.episodes)),
            "forgotten": float(forgotten),
        }
