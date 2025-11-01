from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


_TOKEN_RE = re.compile(r"[\w']+")


def _tokenise(text: str) -> List[str]:
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_RE.findall(text.lower()) if tok]


def _coverage_score(tokens: Sequence[str], seeds: Sequence[str]) -> float:
    if not seeds:
        return 0.0
    token_set = set(tokens)
    if not token_set:
        return 0.0
    hits = sum(1 for seed in seeds if seed in token_set)
    return max(0.0, min(1.0, hits / max(len(seeds), 1)))


def _cohesion_score(text: str) -> float:
    if not text:
        return 0.0
    sentences = [seg.strip() for seg in re.split(r"[.!?。！？]", text) if seg.strip()]
    if not sentences:
        return 0.0
    lengths = [len(_tokenise(sentence)) for sentence in sentences]
    if not lengths:
        return 0.0
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.0
    variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)
    # map low variance to high cohesion, high variance to lower cohesion
    return max(0.0, min(1.0, 1.0 / (1.0 + variance / (mean + 1e-6))))


def _resonance_mix(*components: float) -> float:
    filtered = [max(0.0, min(1.0, comp)) for comp in components if comp is not None]
    if not filtered:
        return 0.0
    return sum(filtered) / len(filtered)


@dataclass
class HemisphericCoherence:
    """Snapshot of how coherent a hemisphere's contribution appears."""

    coverage: float
    cohesion: float
    resonance: float
    tokens: int
    highlights: List[str] = field(default_factory=list)

    def score(self) -> float:
        raw = 0.5 * self.coverage + 0.35 * self.cohesion + 0.15 * self.resonance
        return max(0.0, min(1.0, raw))

    def to_payload(self) -> Dict[str, object]:
        return {
            "coverage": float(self.coverage),
            "cohesion": float(self.cohesion),
            "resonance": float(self.resonance),
            "tokens": int(self.tokens),
            "score": float(self.score()),
            "highlights": list(self.highlights),
        }


@dataclass
class CoherenceSignal:
    """Combined coherence profile emitted by the brainstem integrator."""

    left: HemisphericCoherence
    right: Optional[HemisphericCoherence]
    combined_score: float
    tension: float
    contributions: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def right_or_blank(self) -> HemisphericCoherence:
        if self.right is not None:
            return self.right
        return HemisphericCoherence(coverage=0.0, cohesion=0.0, resonance=0.0, tokens=0)

    def to_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "left": self.left.to_payload(),
            "combined": float(self.combined_score),
            "tension": float(self.tension),
            "notes": list(self.notes),
            "contributions": dict(self.contributions),
        }
        if self.right is not None:
            payload["right"] = self.right.to_payload()
        return payload

    def tags(self) -> Iterable[str]:
        tags = {"coherence"}
        if self.combined_score >= 0.66:
            tags.add("coherence_high")
        elif self.combined_score >= 0.45:
            tags.add("coherence_mid")
        else:
            tags.add("coherence_low")
        if self.tension >= 0.55:
            tags.add("coherence_tension_high")
        elif self.tension >= 0.35:
            tags.add("coherence_tension_mid")
        else:
            tags.add("coherence_tension_low")
        if self.right is None or self.right.tokens == 0:
            tags.add("coherence_left_dominant")
        return tags


class CoherenceResonator:
    """Blend left/right coherence signals and surface integrative cues."""

    def __init__(
        self,
        *,
        left_weight: float = 0.55,
        right_weight: float = 0.45,
        tension_sensitivity: float = 1.15,
    ) -> None:
        total = left_weight + right_weight
        if total <= 0:
            raise ValueError("Weights must be positive")
        self.left_weight = left_weight / total
        self.right_weight = right_weight / total
        self.tension_sensitivity = max(0.1, tension_sensitivity)
        self._last_left: Optional[HemisphericCoherence] = None
        self._last_right: Optional[HemisphericCoherence] = None

    # ------------------------------------------------------------------
    def capture_left(
        self,
        *,
        question: str,
        draft: str,
        context: str,
        focus_keywords: Sequence[str] | None = None,
        focus_metric: float = 0.0,
    ) -> HemisphericCoherence:
        tokens = _tokenise(draft)
        if focus_keywords:
            seeds = [kw.lower() for kw in focus_keywords if kw]
        else:
            seeds = _tokenise(question)
        coverage = _coverage_score(tokens, seeds)
        cohesion = _cohesion_score(draft)
        resonance = _resonance_mix(focus_metric, coverage, min(1.0, len(tokens) / 320.0))
        highlights: List[str] = []
        if seeds:
            highlights.append(f"focus matches: {coverage * len(seeds):.1f}/{len(seeds)}")
        if context:
            ctx_tokens = _tokenise(context)
            ctx_overlap = _coverage_score(tokens, ctx_tokens[:12])
            highlights.append(f"context overlap: {ctx_overlap:.2f}")
        profile = HemisphericCoherence(
            coverage=coverage,
            cohesion=cohesion,
            resonance=resonance,
            tokens=len(tokens),
            highlights=highlights,
        )
        self._last_left = profile
        return profile

    # ------------------------------------------------------------------
    def capture_right(
        self,
        *,
        question: str,
        draft: str,
        detail_notes: Optional[str],
        focus_keywords: Sequence[str] | None,
        psychoid_signal: Optional[Mapping[str, object]] = None,
        confidence: float = 0.0,
        source: str = "",
    ) -> HemisphericCoherence:
        text = detail_notes or ""
        tokens = _tokenise(text)
        if focus_keywords:
            seeds = [kw.lower() for kw in focus_keywords if kw]
        else:
            seeds = _tokenise(question)[:8]
        coverage = _coverage_score(tokens, seeds)
        cohesion = _cohesion_score(text)
        psychoid_resonance = float(psychoid_signal.get("resonance", 0.0)) if psychoid_signal else 0.0
        tension = float(psychoid_signal.get("psychoid_tension", 0.0)) if psychoid_signal else 0.0
        resonance = _resonance_mix(cohesion, coverage, psychoid_resonance, confidence, 1.0 - abs(tension))
        highlights = []
        if source:
            highlights.append(f"source:{source}")
        if psychoid_signal:
            chain = psychoid_signal.get("signifier_chain") or []
            if chain:
                highlights.append(f"signifiers:{len(chain)}")
        profile = HemisphericCoherence(
            coverage=coverage,
            cohesion=cohesion,
            resonance=resonance,
            tokens=len(tokens),
            highlights=highlights,
        )
        self._last_right = profile
        return profile

    # ------------------------------------------------------------------
    def integrate(
        self,
        *,
        final_answer: str,
        psychoid_projection: Optional[Mapping[str, object]] = None,
    ) -> Optional[CoherenceSignal]:
        if self._last_left is None:
            return None
        left = self._last_left
        right = self._last_right
        left_score = left.score()
        right_score = right.score() if right is not None else 0.0
        weighted = self.left_weight * left_score + self.right_weight * right_score
        tension = abs(left_score - right_score)
        tension = max(0.0, min(1.0, tension * self.tension_sensitivity))
        synergy = 1.0 - min(1.0, tension)
        length_factor = min(1.0, len(_tokenise(final_answer)) / 640.0)
        combined = max(0.0, min(1.0, 0.6 * weighted + 0.25 * synergy + 0.15 * length_factor))
        contributions = {
            "left": float(left_score),
            "right": float(right_score),
            "synergy": float(synergy),
        }
        notes: List[str] = []
        if tension >= 0.55:
            notes.append("High hemispheric tension detected")
        elif tension >= 0.35:
            notes.append("Moderate hemispheric tension")
        else:
            notes.append("Hemispheres aligned")
        if psychoid_projection is not None:
            norm = float(psychoid_projection.get("norm", 0.0) or 0.0)
            contributions["psychoid_norm"] = norm
            if norm:
                notes.append(f"Attention norm {norm:.2f} influenced coherence")
        signal = CoherenceSignal(
            left=left,
            right=right,
            combined_score=combined,
            tension=tension,
            contributions=contributions,
            notes=notes,
        )
        return signal

    # ------------------------------------------------------------------
    def vectorise_left(self) -> Optional[List[float]]:
        if self._last_left is None:
            return None
        profile = self._last_left
        return [
            float(profile.score()),
            float(profile.coverage),
            float(profile.cohesion),
            float(profile.resonance),
            float(profile.tokens),
        ]

    # ------------------------------------------------------------------
    def annotate_answer(self, answer: str, signal: CoherenceSignal) -> str:
        left_score = signal.left.score()
        block = [
            "[Coherence Integration]",
            "- left  score {score:.2f} coverage {coverage:.2f} cohesion {cohesion:.2f}".format(
                score=left_score,
                coverage=signal.left.coverage,
                cohesion=signal.left.cohesion,
            ),
        ]
        right = signal.right
        if right is not None:
            right_score = right.score()
            block.append(
                "- right score {score:.2f} coverage {coverage:.2f} cohesion {cohesion:.2f}".format(
                    score=right_score,
                    coverage=right.coverage,
                    cohesion=right.cohesion,
                )
            )
        else:
            block.append("- right score 0.00 (no detail response)")
        block.append(f"- combined {signal.combined_score:.2f} | tension {signal.tension:.2f}")
        if signal.notes:
            for note in signal.notes:
                block.append(f"- note: {note}")
        return f"{answer}\n\n" + "\n".join(block)


__all__ = [
    "CoherenceResonator",
    "CoherenceSignal",
    "HemisphericCoherence",
]
