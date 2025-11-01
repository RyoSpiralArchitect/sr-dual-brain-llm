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

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


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

    def __init__(self, *, min_keywords: int = 2, gating_threshold: float = 0.25) -> None:
        self.min_keywords = min_keywords
        self.gating_threshold = gating_threshold

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

