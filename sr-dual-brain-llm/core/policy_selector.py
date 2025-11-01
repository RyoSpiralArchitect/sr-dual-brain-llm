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

"""Lightweight heuristics for selecting which hemisphere leads a dialogue."""

from __future__ import annotations

import random
from typing import Dict, Iterable


_RIGHT_KEYWORDS = {
    "draw",
    "imagine",
    "imagination",
    "dream",
    "symbol",
    "symbolic",
    "vision",
    "fantasy",
    "archetype",
    "myth",
    "poem",
    "lyric",
}

_LEFT_KEYWORDS = {
    "analyze",
    "analysis",
    "summarize",
    "summary",
    "breakdown",
    "explain",
    "derive",
    "calculate",
    "compute",
}


def _contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def decide_leading_brain(user_input: str, context: Dict[str, object] | None = None) -> str:
    """Return ``"left"`` or ``"right"`` depending on the request semantics.

    The selector relies on lightweight heuristics so it can run synchronously
    alongside the interactive loop.  It inspects the input prompt first and then
    looks at optional context hints (novelty scores, prior leading preferences,
    or explicit overrides).
    """

    if context is None:
        context = {}

    # Honour explicit overrides from previous interactions first.
    override = (context.get("override") or "").strip().lower()
    if override in {"left", "right"}:
        return override

    if _contains_keyword(user_input, _RIGHT_KEYWORDS):
        return "right"
    if _contains_keyword(user_input, _LEFT_KEYWORDS):
        return "left"

    # Subtle nudges based on novelty and recent flow.
    novelty = float(context.get("novelty") or 0.0)
    last_leading = (context.get("last_leading") or "").strip().lower()

    if novelty < 0.35 and last_leading == "left":
        # If we are in a familiar territory that the left brain recently led,
        # rotate to the right brain to surface alternative associations.
        return "right"
    if novelty > 0.75 and last_leading == "right":
        # When the previous exchange already leaned right and the question is
        # highly novel, allow the left brain to scaffold structure first.
        return "left"

    if "dream" in user_input.lower() or "symbol" in user_input.lower():
        return "right"

    rng = context.get("rng")
    if rng is not None and hasattr(rng, "choice"):
        return rng.choice(["left", "right"])
    return random.choice(["left", "right"])


__all__ = ["decide_leading_brain"]

