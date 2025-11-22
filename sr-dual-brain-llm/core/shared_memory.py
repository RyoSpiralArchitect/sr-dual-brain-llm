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

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _tokenize(text: str) -> List[str]:
    """Lightweight tokeniser used for novelty scoring and retrieval."""

    lowered = text.replace("\n", " ").lower()
    return [tok for tok in lowered.split(" ") if tok]


def _normalise_tags(tags: Iterable[str] | None) -> Tuple[str, ...]:
    if not tags:
        return ()
    unique = {tag.strip().lower() for tag in tags if tag and tag.strip()}
    return tuple(sorted(unique))


def _derive_tags(question: str) -> Tuple[str, ...]:
    """Create lightweight tags from a question for later retrieval."""

    tokens = [tok for tok in _tokenize(question) if len(tok) > 3]
    return tuple(sorted(set(tokens[:6])))


@dataclass(frozen=True)
class MemoryTrace:
    """Structured representation of a QA pair stored in memory."""

    question: str
    answer: str
    timestamp: float = field(default_factory=lambda: time.time())
    tags: Tuple[str, ...] = field(default_factory=tuple)
    question_tokens: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.question_tokens:
            object.__setattr__(self, "question_tokens", tuple(_tokenize(self.question)))
        if not self.tags:
            object.__setattr__(self, "tags", _derive_tags(self.question))


class SharedMemory:
    def __init__(self, *, max_items: int = 1024):
        self.max_items = max_items
        self.past_qas: List[MemoryTrace] = []
        self.kv: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Storage helpers
    def store(self, qa_pair: Dict[str, str] | MemoryTrace, *, tags: Iterable[str] | None = None) -> None:
        """Persist a QA pair or trace into the rolling memory buffer."""

        if isinstance(qa_pair, MemoryTrace):
            trace = qa_pair
        else:
            trace = MemoryTrace(
                question=qa_pair["Q"],
                answer=qa_pair["A"],
                tags=_normalise_tags(tags) or _derive_tags(qa_pair["Q"]),
            )

        self.past_qas.append(trace)
        if len(self.past_qas) > self.max_items:
            del self.past_qas[0 : len(self.past_qas) - self.max_items]

    # ------------------------------------------------------------------
    def get_context(self, n: int = 5) -> str:
        return "\n".join(
            [f"Q:{trace.question} A:{trace.answer}" for trace in self.past_qas[-n:]]
        )

    # ------------------------------------------------------------------
    def _score_trace(
        self,
        trace: MemoryTrace,
        q_tokens: Sequence[str],
        q_tags: Sequence[str],
        recency_weight: float,
    ) -> float:
        if not q_tokens:
            return recency_weight

        trace_tokens = set(trace.question_tokens)
        q_tokens_set = set(q_tokens)
        if not trace_tokens:
            lexical_score = 0.0
        else:
            intersection = len(trace_tokens & q_tokens_set)
            union = len(trace_tokens | q_tokens_set)
            lexical_score = intersection / union if union else 0.0

        q_tags_set = set(q_tags)
        tag_bonus = 0.0
        if q_tags_set and trace.tags:
            tag_bonus = len(q_tags_set & set(trace.tags)) / max(len(q_tags_set), 1)

        return 0.7 * lexical_score + 0.2 * tag_bonus + 0.1 * recency_weight

    # ------------------------------------------------------------------
    def retrieve_related(self, question: str, n: int = 3) -> str:
        """Return a formatted slice of relevant memory traces."""

        if not self.past_qas:
            return ""

        q_tokens = _tokenize(question)
        q_tags = _derive_tags(question)
        total = len(self.past_qas)

        scored = []
        for idx, trace in enumerate(self.past_qas):
            recency_weight = (idx + 1) / total
            score = self._score_trace(trace, q_tokens, q_tags, recency_weight)
            if score > 0.0:
                scored.append((score, trace))

        if not scored:
            return self.get_context(n)

        scored.sort(key=lambda item: item[0], reverse=True)
        top = [f"Q:{trace.question} A:{trace.answer}" for _, trace in scored[:n]]
        return "\n".join(top)

    # ------------------------------------------------------------------
    def put_kv(self, key: str, value: Any) -> None:
        self.kv[key] = value

    def get_kv(self, key: str, default: Any = None) -> Any:
        return self.kv.get(key, default)

    def record_leading_brain(self, qid: str, leading_brain: str) -> None:
        brain = str(leading_brain).strip().lower()
        if brain not in {"left", "right"}:
            return
        self.put_kv(f"leading_brain:{qid}", brain)
        self.put_kv("last_leading_brain", brain)

    def get_leading_brain(self, qid: str) -> str | None:
        return self.kv.get(f"leading_brain:{qid}")

    # ------------------------------------------------------------------
    def novelty_score(self, question: str) -> float:
        """Return a score in ``[0, 1]`` indicating how novel a question is."""

        if not self.past_qas:
            return 1.0

        q_tokens = set(_tokenize(question))
        if not q_tokens:
            return 1.0

        highest_overlap = 0.0
        for trace in self.past_qas:
            past_tokens = set(trace.question_tokens)
            if not past_tokens:
                continue
            intersection = len(q_tokens & past_tokens)
            union = len(q_tokens | past_tokens)
            if union:
                highest_overlap = max(highest_overlap, intersection / union)
        return max(0.0, 1.0 - highest_overlap)
