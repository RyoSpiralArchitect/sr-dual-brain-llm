import pytest

from core.coherence_resonator import CoherenceResonator
from core.schema import (
    ArchetypeActivation,
    EmergentIdeaModel,
    GeometryModel,
    PsychoidSignalModel,
    UnconsciousSummaryModel,
)


def test_resonator_tracks_left_and_right_profiles():
    resonator = CoherenceResonator()

    draft_text = (
        "This analysis explores mythic journeys and harmonics while keeping "
        "analysis grounded in layered archetypal signals."
    )
    left = resonator.capture_left(
        question="Analyse layered harmonics in mythic journeys",
        draft=draft_text,
        context="Mythic journeys often rely on harmonic layering for meaning.",
        focus_keywords=("analysis", "harmonics", "archetypal"),
        focus_metric=0.58,
    )

    assert 0.0 <= left.coverage <= 1.0
    assert left.cohesion > 0.0
    assert pytest.approx(left.score(), rel=1e-6) == left.to_payload()["score"]

    vector = resonator.vectorise_left()
    assert vector and pytest.approx(vector[0], rel=1e-6) == left.score()
    assert 0.0 <= vector[-1] <= 1.0

    right = resonator.capture_right(
        question="Analyse layered harmonics in mythic journeys",
        draft="",
        detail_notes="Layered harmonics resonate with archetypal tides",
        focus_keywords=("harmonics", "archetypal"),
        psychoid_signal=PsychoidSignalModel(
            resonance=0.75,
            psychoid_tension=0.2,
            signifier_chain=["echo"],
        ),
        confidence=0.72,
        source="callosum",
    )

    assert right.resonance > 0.5
    assert any("source:callosum" in entry for entry in right.highlights)

    summary = UnconsciousSummaryModel(
        top_k=["dream", "myth"],
        geometry=GeometryModel(r=0.0, theta=0.0, curvature_proxy=0.0),
        archetype_map=[
            ArchetypeActivation(id="dream", label="Dream", intensity=0.9),
            ArchetypeActivation(id="myth", label="Myth", intensity=0.6),
        ],
        emergent_ideas=[
            EmergentIdeaModel(
                archetype="dream",
                label="Dream tide",
                intensity=0.55,
                incubation_rounds=1,
                trigger_similarity=0.8,
                origin="test",
            )
        ],
        cache_depth=2,
        psychoid_signal=PsychoidSignalModel(
            signifier_chain=["sea", "tide", "moon"],
            resonance=0.82,
        ),
    )
    resonator.capture_unconscious(
        question="Analyse layered harmonics in mythic journeys",
        draft=draft_text,
        final_answer="Combined narration touches harmonics and archetypes coherently.",
        summary=summary,
    )

    motifs_summary = UnconsciousSummaryModel(
        top_k=[],
        geometry=GeometryModel(r=0.0, theta=0.0, curvature_proxy=0.0),
        archetype_map=[],
        motifs=["chorus", "pulse"],
    )

    motifs = resonator.capture_linguistic_motifs(
        question="Analyse layered harmonics in mythic journeys",
        draft=draft_text,
        final_answer="Combined narration touches harmonics and archetypes coherently.",
        unconscious_summary=motifs_summary,
    )

    assert motifs is not None
    assert motifs.score() >= motifs.motif_density

    signal = resonator.integrate(
        final_answer="Combined narration touches harmonics and archetypes coherently.",
        psychoid_projection={"norm": 0.91},
        unconscious_summary=summary,
    )

    assert signal is not None
    assert signal.right is not None
    assert signal.combined_score >= signal.left.score() * 0.5
    payload = signal.to_payload()
    assert payload["contributions"]["psychoid_norm"] == pytest.approx(0.91)
    assert signal.unconscious is not None
    assert signal.linguistic_depth == pytest.approx(signal.unconscious.score())
    assert payload["unconscious"]["score"] == pytest.approx(signal.unconscious.score())
    assert signal.motifs is not None
    assert payload["motifs"]["score"] == pytest.approx(signal.motifs.score())
    annotated = resonator.annotate_answer("final", signal)
    assert "[Coherence Integration]" in annotated
    assert "[Unconscious Linguistic Fabric]" in annotated
    assert "[Linguistic Motifs]" in annotated
    assert "coherence" in set(signal.tags())


def test_resonator_reset_and_pair_vectorisation():
    resonator = CoherenceResonator()

    resonator.capture_left(
        question="Outline resonance",
        draft="Left brain outlines resonance and rhythm in detail.",
        context="Resonance thrives on rhythm",
        focus_keywords=("resonance", "rhythm"),
        focus_metric=0.5,
    )
    resonator.capture_right(
        question="Outline resonance",
        draft="",
        detail_notes="Right brain expands on resonance with rhythm and flow.",
        focus_keywords=("resonance",),
        psychoid_signal=PsychoidSignalModel(resonance=0.9),
        confidence=0.6,
        source="callosum",
    )

    pair = resonator.vectorise_pair()
    assert pair is not None
    left_vec, right_vec = pair
    assert len(left_vec) == len(right_vec) == 5
    assert right_vec[0] <= 1.0

    signal = resonator.integrate(final_answer="Combined answer")
    assert signal is not None
    assert resonator.last_signal() is signal

    resonator.reset()
    assert resonator.vectorise_left() is None
    assert resonator.vectorise_pair() is None
    assert resonator.last_signal() is None


def test_integrate_handles_missing_right_profile():
    resonator = CoherenceResonator()
    resonator.capture_left(
        question="Summarise themes",
        draft="Themes align closely with the requested summary.",
        context="Themes align",
        focus_metric=0.42,
    )

    signal = resonator.integrate(final_answer="Themes align with summary")
    assert signal is not None
    assert signal.right is None
    assert any("No right-brain" in note for note in signal.notes)


def test_retune_adjusts_weights_and_mode_annotations():
    resonator = CoherenceResonator()
    base_left = resonator.left_weight
    base_right = resonator.right_weight

    resonator.retune("right", intensity=0.7)
    assert resonator.right_weight > base_right
    assert resonator.left_weight < base_left

    resonator.capture_left(
        question="Tell a story",
        draft="A lyrical story emerges from symbols and dreams.",
        context="Dream symbols entwine.",
        focus_keywords=("story", "symbols"),
        focus_metric=0.6,
    )
    resonator.capture_right(
        question="Tell a story",
        draft="",
        detail_notes="Mythic tides colour the narrative with emotive hues.",
        focus_keywords=("mythic", "narrative"),
        psychoid_signal=PsychoidSignalModel(resonance=0.8),
        confidence=0.55,
        source="callosum",
    )

    signal = resonator.integrate(final_answer="Mythic narrative braided together")
    assert signal is not None
    assert signal.mode == "right"
    assert "coherence_mode_right" in set(signal.tags())
    assert "routing mode" in resonator.annotate_answer("ans", signal)

    resonator.retune("balanced", intensity=0.0)
    assert pytest.approx(resonator.left_weight, rel=1e-6) == resonator._default_left_weight
