"""Hemispheric coherence scoring utilities."""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


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
    mode: str = "balanced"
    unconscious: Optional["UnconsciousLinguisticWeave"] = None
    linguistic_depth: float = 0.0
    motifs: Optional["LinguisticMotifProfile"] = None

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
            "mode": self.mode,
            "linguistic_depth": float(self.linguistic_depth),
        }
        if self.right is not None:
            payload["right"] = self.right.to_payload()
        if self.unconscious is not None:
            payload["unconscious"] = self.unconscious.to_payload()
        if self.motifs is not None:
            payload["motifs"] = self.motifs.to_payload()
        return payload

    def tags(self) -> Iterable[str]:
        tags = {"coherence"}
        tags.add(f"coherence_mode_{self.mode}")
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
        if self.unconscious is not None:
            tags.add("coherence_unconscious")
            depth = self.unconscious.score()
            if depth >= 0.7:
                tags.add("linguistic_fabric_rich")
            elif depth >= 0.45:
                tags.add("linguistic_fabric_balanced")
            else:
                tags.add("linguistic_fabric_sparse")
        if self.motifs is not None:
            tags.add("coherence_linguistic_motif")
            motif_score = self.motifs.score()
            if motif_score >= 0.7:
                tags.add("linguistic_motif_resonant")
            elif motif_score >= 0.45:
                tags.add("linguistic_motif_balanced")
            else:
                tags.add("linguistic_motif_sparse")
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
        self._default_left_weight = left_weight / total
        self._default_right_weight = right_weight / total
        self.left_weight = self._default_left_weight
        self.right_weight = self._default_right_weight
        self.tension_sensitivity = max(0.1, tension_sensitivity)
        self._base_tension = self.tension_sensitivity
        self._mode = "balanced"
        self._last_left: Optional[HemisphericCoherence] = None
        self._last_right: Optional[HemisphericCoherence] = None
        self._last_signal: Optional[CoherenceSignal] = None
        self._last_unconscious: Optional[UnconsciousLinguisticWeave] = None
        self._last_motifs: Optional[LinguisticMotifProfile] = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear cached hemispheric profiles from the previous interaction."""

        self._last_left = None
        self._last_right = None
        self._last_signal = None
        self._last_unconscious = None
        self._last_motifs = None

    # ------------------------------------------------------------------
    def _apply_weights(self, left_weight: float, right_weight: float) -> None:
        left_weight = max(0.05, float(left_weight))
        right_weight = max(0.05, float(right_weight))
        total = left_weight + right_weight
        self.left_weight = left_weight / total
        self.right_weight = right_weight / total

    # ------------------------------------------------------------------
    def retune(self, mode: str, *, intensity: float = 0.0) -> None:
        """Adapt weighting to emphasise the dominant hemisphere."""

        clamped = max(0.0, min(1.0, float(intensity)))
        boost = 0.15 + 0.25 * clamped
        if mode == "left":
            left = self._default_left_weight + boost
            right = self._default_right_weight - boost
            self.tension_sensitivity = min(
                2.5, self._base_tension * (1.0 + 0.25 * (0.5 + clamped))
            )
        elif mode == "right":
            left = self._default_left_weight - boost
            right = self._default_right_weight + boost
            self.tension_sensitivity = max(
                0.2, self._base_tension * (1.0 - 0.3 * (0.5 + clamped))
            )
        else:
            left = self._default_left_weight
            right = self._default_right_weight
            self.tension_sensitivity = self._base_tension
            mode = "balanced"
        self._apply_weights(left, right)
        self._mode = mode

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
            seeds = sorted({kw.lower() for kw in focus_keywords if kw})
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
        if cohesion:
            highlights.append(f"cohesion: {cohesion:.2f}")
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
            seeds = sorted({kw.lower() for kw in focus_keywords if kw})
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
        if cohesion:
            highlights.append(f"cohesion: {cohesion:.2f}")
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
        unconscious_summary: Optional[Mapping[str, object]] = None,
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
        unconscious_score = self._last_unconscious.score() if self._last_unconscious else 0.0
        motif_score = self._last_motifs.score() if self._last_motifs else 0.0
        combined = max(
            0.0,
            min(
                1.0,
                0.38 * weighted
                + 0.2 * synergy
                + 0.16 * length_factor
                + 0.14 * unconscious_score
                + 0.12 * motif_score,
            ),
        )
        contributions = {
            "left": float(left_score),
            "right": float(right_score),
            "synergy": float(synergy),
        }
        if self._last_unconscious is not None:
            contributions["unconscious"] = float(self._last_unconscious.score())
            contributions["linguistic_density"] = float(
                self._last_unconscious.lexical_density
            )
            contributions["signifier_depth"] = float(
                self._last_unconscious.signifier_depth
            )
        if self._last_motifs is not None:
            contributions["motifs"] = float(self._last_motifs.score())
            contributions["motif_density"] = float(self._last_motifs.motif_density)
            contributions["alliteration"] = float(self._last_motifs.alliteration)
            contributions["cadence"] = float(self._last_motifs.cadence)
        notes: List[str] = []
        if tension >= 0.55:
            notes.append("High hemispheric tension detected")
        elif tension >= 0.35:
            notes.append("Moderate hemispheric tension")
        else:
            notes.append("Hemispheres aligned")
        if right is None:
            notes.append("No right-brain detail captured")
        if psychoid_projection is not None:
            norm = float(psychoid_projection.get("norm", 0.0) or 0.0)
            contributions["psychoid_norm"] = norm
            if norm:
                notes.append(f"Attention norm {norm:.2f} influenced coherence")
        if self._last_unconscious is not None and self._last_unconscious.highlights:
            for entry in self._last_unconscious.highlights:
                notes.append(f"Unconscious: {entry}")
        if unconscious_summary:
            cache_depth = float(unconscious_summary.get("cache_depth", 0.0) or 0.0)
            if cache_depth:
                contributions["unconscious_cache"] = cache_depth
        signal = CoherenceSignal(
            left=left,
            right=right,
            combined_score=combined,
            tension=tension,
            contributions=contributions,
            notes=notes,
            mode=self._mode,
            unconscious=self._last_unconscious,
            linguistic_depth=unconscious_score,
            motifs=self._last_motifs,
        )
        self._last_signal = signal
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
            min(1.0, float(profile.tokens) / 320.0),
        ]

    # ------------------------------------------------------------------
    def vectorise_pair(self) -> Optional[Tuple[List[float], List[float]]]:
        """Return the last left/right vectors for quick feature export."""

        if self._last_left is None or self._last_right is None:
            return None
        left_vec = self.vectorise_left() or []
        right_profile = self._last_right
        right_vec = [
            float(right_profile.score()),
            float(right_profile.coverage),
            float(right_profile.cohesion),
            float(right_profile.resonance),
            min(1.0, float(right_profile.tokens) / 320.0),
        ]
        return left_vec, right_vec

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
        block.append(f"- routing mode: {signal.mode}")
        if signal.notes:
            for note in signal.notes:
                block.append(f"- note: {note}")
        segments = ["\n".join(block)]
        if signal.unconscious is not None:
            weave = signal.unconscious
            weave_block = [
                "[Unconscious Linguistic Fabric]",
                "- fabric score {score:.2f} | lexical density {density:.2f}".format(
                    score=weave.score(), density=weave.lexical_density
                ),
                "- archetype resonance {resonance:.2f} | variation {variation:.2f}".format(
                    resonance=weave.archetype_resonance,
                    variation=weave.variation,
                ),
                "- signifier depth {depth:.2f} | tokens {tokens}".format(
                    depth=weave.signifier_depth,
                    tokens=weave.tokens,
                ),
            ]
            if weave.highlights:
                weave_block.extend(f"- note: {entry}" for entry in weave.highlights)
            segments.append("\n".join(weave_block))
        if signal.motifs is not None:
            motif = signal.motifs
            motif_block = [
                "[Linguistic Motifs]",
                "- motif score {score:.2f} | density {density:.2f}".format(
                    score=motif.score(), density=motif.motif_density
                ),
                "- alliteration {alliteration:.2f} | cadence {cadence:.2f}".format(
                    alliteration=motif.alliteration,
                    cadence=motif.cadence,
                ),
                "- echo {echo:.2f} | loops {loops}".format(
                    echo=motif.echo,
                    loops=motif.repeated_loops,
                ),
            ]
            if motif.highlights:
                motif_block.extend(f"- note: {entry}" for entry in motif.highlights)
            segments.append("\n".join(motif_block))
        return f"{answer}\n\n" + "\n\n".join(segments)

    # ------------------------------------------------------------------
    def last_signal(self) -> Optional[CoherenceSignal]:
        """Expose the most recent integration result for diagnostics."""

        return self._last_signal

    # ------------------------------------------------------------------
    def capture_linguistic_motifs(
        self,
        *,
        question: str,
        draft: str,
        final_answer: str,
        unconscious_summary: Optional[Mapping[str, object]] = None,
    ) -> Optional["LinguisticMotifProfile"]:
        tokens = _tokenise(final_answer)
        if len(tokens) < 4:
            self._last_motifs = None
            return None

        motif_density = _motif_density(tokens)
        alliteration = _alliteration(tokens)
        cadence = _cadence(final_answer)
        echo = _echo_density(tokens)

        highlights: List[str] = []
        if motif_density:
            highlights.append(f"motif density {motif_density:.2f}")
        if alliteration:
            highlights.append(f"alliteration {alliteration:.2f}")
        if echo:
            highlights.append(f"echo {echo:.2f}")

        question_overlap = _coverage_score(tokens, _tokenise(question)[:20])
        draft_overlap = _coverage_score(tokens, _tokenise(draft)[:20])
        overlap = max(question_overlap, draft_overlap)
        if overlap:
            highlights.append(f"overlap {overlap:.2f}")

        loops = 0
        if unconscious_summary is not None:
            motifs = unconscious_summary.get("motifs") or ()
            if isinstance(motifs, Sequence):
                loops = len(motifs)
                if loops:
                    highlights.append(f"unconscious motifs {loops}")

        profile = LinguisticMotifProfile(
            motif_density=motif_density,
            alliteration=alliteration,
            cadence=cadence,
            echo=echo,
            repeated_loops=loops,
            highlights=highlights,
        )
        self._last_motifs = profile
        return profile

    # ------------------------------------------------------------------
    def last_motifs(self) -> Optional["LinguisticMotifProfile"]:
        return self._last_motifs

    # ------------------------------------------------------------------
    def capture_unconscious(
        self,
        *,
        question: str,
        draft: str,
        final_answer: str,
        summary: Optional[Mapping[str, object]] = None,
        psychoid_signal: Optional[Mapping[str, object]] = None,
    ) -> Optional["UnconsciousLinguisticWeave"]:
        tokens = _tokenise(final_answer)
        if not tokens and summary is None and psychoid_signal is None:
            self._last_unconscious = None
            return None

        question_tokens = _tokenise(question)
        draft_tokens = _tokenise(draft)
        lexical_density = _lexical_density(tokens)
        variation = _variation(tokens)
        archetype_resonance = 0.0
        highlights: List[str] = []
        emergent_count = 0
        if summary is not None:
            archetype_entries = summary.get("archetype_map") or []
            intensities: List[float] = []
            for entry in archetype_entries:
                try:
                    intensities.append(float(entry.get("intensity", 0.0)))
                except Exception:  # pragma: no cover - defensive
                    continue
            if intensities:
                archetype_resonance = max(0.0, min(1.0, sum(intensities) / (len(intensities) * 1.5)))
                highlights.append(
                    "archetype avg {:.2f}".format(archetype_resonance)
                )
            emergent = summary.get("emergent_ideas") or []
            emergent_count = len(emergent)
            if emergent_count:
                highlights.append(f"emergent ideas {emergent_count}")
        signifier_depth = 0.0
        if psychoid_signal is None and summary is not None:
            psychoid_signal = summary.get("psychoid_signal")  # type: ignore[assignment]
        if psychoid_signal:
            chain = psychoid_signal.get("signifier_chain") or []
            if isinstance(chain, Sequence):
                signifier_depth = min(1.0, len(chain) / 12.0)
                if chain:
                    highlights.append(f"signifiers {len(chain)}")
            resonance = psychoid_signal.get("resonance")
            if isinstance(resonance, (int, float)):
                archetype_resonance = max(archetype_resonance, max(0.0, min(1.0, float(resonance))))

        # Encourage cross-hemispheric linguistic weaving by tracking overlaps
        question_overlap = _coverage_score(tokens, question_tokens[:16])
        draft_overlap = _coverage_score(tokens, draft_tokens[:16])
        variation = max(variation, 0.5 * (question_overlap + draft_overlap))
        if question_overlap:
            highlights.append(f"question weave {question_overlap:.2f}")
        if draft_overlap:
            highlights.append(f"draft weave {draft_overlap:.2f}")

        weave = UnconsciousLinguisticWeave(
            lexical_density=lexical_density,
            archetype_resonance=archetype_resonance,
            signifier_depth=signifier_depth,
            variation=variation,
            tokens=len(tokens),
            highlights=highlights,
            emergent_count=emergent_count,
        )
        self._last_unconscious = weave
        return weave


def _lexical_density(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return max(0.0, min(1.0, len(set(tokens)) / len(tokens)))


def _variation(tokens: Sequence[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    bigrams = {tuple(tokens[idx : idx + 2]) for idx in range(len(tokens) - 1)}
    variation = len(bigrams) / max(len(tokens) - 1, 1)
    return max(0.0, min(1.0, variation))


def _motif_density(tokens: Sequence[str]) -> float:
    stems = [token[:4] for token in tokens if token]
    if len(stems) < 4:
        return 0.0
    counter = collections.Counter(stems)
    repeats = sum(1 for stem, count in counter.items() if count > 1)
    return max(0.0, min(1.0, repeats / len(counter)))


def _alliteration(tokens: Sequence[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    count = 0
    for first, second in zip(tokens, tokens[1:]):
        if not first or not second:
            continue
        if first[0] == second[0]:
            count += 1
    return max(0.0, min(1.0, count / (len(tokens) - 1)))


def _cadence(text: str) -> float:
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
    return max(0.0, min(1.0, 1.0 / (1.0 + variance / (mean + 1e-6))))


def _echo_density(tokens: Sequence[str]) -> float:
    if len(tokens) < 4:
        return 0.0
    last_index: Dict[str, int] = {}
    echoes = 0
    for idx, token in enumerate(tokens):
        prev = last_index.get(token)
        if prev is not None and idx - prev > 2:
            echoes += 1
        last_index[token] = idx
    return max(0.0, min(1.0, echoes / len(tokens)))


@dataclass
class UnconsciousLinguisticWeave:
    """Linguistic texture derived from unconscious field alignment."""

    lexical_density: float
    archetype_resonance: float
    signifier_depth: float
    variation: float
    tokens: int
    highlights: List[str] = field(default_factory=list)
    emergent_count: int = 0

    def score(self) -> float:
        base = 0.45 * self.lexical_density + 0.35 * self.archetype_resonance + 0.2 * self.variation
        enriched = base + 0.05 * min(1.0, self.signifier_depth)
        return max(0.0, min(1.0, enriched))

    def to_payload(self) -> Dict[str, object]:
        return {
            "lexical_density": float(self.lexical_density),
            "archetype_resonance": float(self.archetype_resonance),
            "signifier_depth": float(self.signifier_depth),
            "variation": float(self.variation),
            "tokens": int(self.tokens),
            "score": float(self.score()),
            "highlights": list(self.highlights),
            "emergent_count": int(self.emergent_count),
        }


@dataclass
class LinguisticMotifProfile:
    """Motif-oriented linguistic structure captured from the full response."""

    motif_density: float
    alliteration: float
    cadence: float
    echo: float
    repeated_loops: int = 0
    highlights: List[str] = field(default_factory=list)

    def score(self) -> float:
        base = 0.35 * self.motif_density + 0.3 * self.cadence + 0.2 * self.alliteration
        enriched = base + 0.15 * max(0.0, min(1.0, self.echo))
        return max(0.0, min(1.0, enriched))

    def to_payload(self) -> Dict[str, object]:
        return {
            "motif_density": float(self.motif_density),
            "alliteration": float(self.alliteration),
            "cadence": float(self.cadence),
            "echo": float(self.echo),
            "score": float(self.score()),
            "repeated_loops": int(self.repeated_loops),
            "highlights": list(self.highlights),
        }


__all__ = [
    "CoherenceResonator",
    "CoherenceSignal",
    "HemisphericCoherence",
    "UnconsciousLinguisticWeave",
    "LinguisticMotifProfile",
]
