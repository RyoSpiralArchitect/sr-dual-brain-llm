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

"""Utility helpers for deciding which hemisphere should lead an interaction."""

from __future__ import annotations

from typing import Dict, Optional

_RIGHT_CUES = {
    "draw",
    "imagine",
    "imagery",
    "symbol",
    "symbols",
    "dream",
    "dreams",
    "myth",
    "myths",
    "poem",
    "poetry",
    "story",
    "stories",
    "vision",
    "waterfall",
    "waterfalls",
    "archetype",
    "archetypes",
    "fantasy",
    "fantasies",
    "比喩",
    "夢",
    "象徴",
    "直感",
    "詩",
    "想像",
}

_LEFT_CUES = {
    "analyze",
    "analyse",
    "analysis",
    "breakdown",
    "explain",
    "explanation",
    "summarize",
    "summarise",
    "calculate",
    "calculation",
    "derive",
    "proof",
    "proofs",
    "structure",
    "logic",
    "diagram",
    "数式",
    "計算",
    "分析",
    "整理",
    "論理",
    "要約",
    "解析",
}


def _score_cues(text: str, cues: set[str]) -> int:
    lowered = text.lower()
    score = 0
    for cue in cues:
        if cue.lower() in lowered:
            score += 1
    return score


def decide_leading_brain(user_input: str, context: Optional[Dict[str, object]] = None) -> str:
    """Return ``"left"`` or ``"right"`` depending on semantic tilt."""

    context = context or {}
    combined = user_input
    memory_excerpt = context.get("memory")
    if isinstance(memory_excerpt, str):
        combined += " \n" + memory_excerpt
    novelty = float(context.get("novelty", 0.0))
    recent_lead = str(context.get("recent_lead", ""))

    left_score = _score_cues(combined, _LEFT_CUES)
    right_score = _score_cues(combined, _RIGHT_CUES)

    if right_score > left_score:
        return "right"
    if left_score > right_score:
        return "left"

    if novelty >= 0.6:
        return "right"
    if novelty <= 0.2:
        return "left"

    if recent_lead in {"left", "right"}:
        return "right" if recent_lead == "left" else "left"

    # Deterministic fallback that still provides some variation without randomness.
    parity = sum(ord(ch) for ch in user_input) % 2
    return "right" if parity else "left"


__all__ = ["decide_leading_brain"]
