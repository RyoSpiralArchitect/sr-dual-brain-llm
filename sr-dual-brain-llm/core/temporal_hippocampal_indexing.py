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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _tokenize(text: str) -> List[str]:
    return [t for t in text.replace("\n", " ").replace("\t", " ").split(" ") if t]


def _normalise_tags(tags: Iterable[str] | None) -> Tuple[str, ...]:
    if not tags:
        return ()
    unique = {tag.strip().lower() for tag in tags if tag and tag.strip()}
    return tuple(sorted(unique))


@dataclass
class EpisodicTrace:
    """Single hippocampal memory containing hemispheric collaboration context."""

    qid: str
    question: str
    answer: str
    vector: np.ndarray
    timestamp: float = field(default_factory=lambda: time.time())
    leading: Optional[str] = None
    collaboration_strength: Optional[float] = None
    selection_reason: Optional[str] = None
    tags: Tuple[str, ...] = field(default_factory=tuple)
    annotations: Dict[str, Any] = field(default_factory=dict)

    def payload(self) -> str:
        return f"Q: {self.question}\nA: {self.answer}"

    def summary(self, *, similarity: Optional[float] = None, max_chars: int = 240) -> str:
        parts: List[str] = []
        if similarity is not None:
            parts.append(f"sim={similarity:.2f}")
        if self.leading:
            parts.append(f"lead={self.leading}")
        if self.collaboration_strength is not None:
            parts.append(f"collab={self.collaboration_strength:.2f}")
        if self.selection_reason:
            parts.append(self.selection_reason)
        prefix = f"({'; '.join(parts)}) " if parts else ""
        answer_snippet = self.answer.replace("\n", " ")[:max_chars]
        return f"{prefix}Q: {self.question[:max_chars]} | A: {answer_snippet}"


class TemporalHippocampalIndexing:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.episodes: List[EpisodicTrace] = []
        self._eps = 1e-8

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text.lower()):
            v[hash(tok) % self.dim] += 1.0
        n = float(np.linalg.norm(v) + self._eps)
        return v / n

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
        trace = EpisodicTrace(
            qid=qid,
            question=question,
            answer=answer,
            vector=vec,
            leading=(leading or None),
            collaboration_strength=None
            if collaboration_strength is None
            else float(collaboration_strength),
            selection_reason=selection_reason,
            tags=norm_tags,
            annotations=annotations,
        )
        self.episodes.append(trace)

    def retrieve(self, query: str, topk: int = 3) -> List[Tuple[float, EpisodicTrace]]:
        if not self.episodes:
            return []
        qv = self._embed(query)
        scored: List[Tuple[float, EpisodicTrace]] = []
        for trace in self.episodes:
            sim = float(np.dot(qv, trace.vector))
            scored.append((sim, trace))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:topk]

    def retrieve_summary(
        self, query: str, topk: int = 3, max_chars: int = 240
    ) -> str:
        hits = self.retrieve(query, topk=topk)
        if not hits:
            return ""
        parts = [trace.summary(similarity=sim, max_chars=max_chars) for sim, trace in hits]
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
            }
        window = max(1, min(window, len(self.episodes)))
        subset = self.episodes[-window:]
        strengths = [
            trace.collaboration_strength
            for trace in subset
            if trace.collaboration_strength is not None
        ]
        avg_strength = sum(strengths) / len(strengths) if strengths else 0.0
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
        }
