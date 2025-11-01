import pytest

from core.coherence_resonator import CoherenceResonator


def test_resonator_tracks_left_and_right_profiles():
    resonator = CoherenceResonator()

    left = resonator.capture_left(
        question="Analyse layered harmonics in mythic journeys",
        draft=(
            "This analysis explores mythic journeys and harmonics while keeping "
            "analysis grounded in layered archetypal signals."
        ),
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
        psychoid_signal={"resonance": 0.75, "psychoid_tension": 0.2},
        confidence=0.72,
        source="callosum",
    )

    assert right.resonance > 0.5
    assert any("source:callosum" in entry for entry in right.highlights)

    signal = resonator.integrate(
        final_answer="Combined narration touches harmonics and archetypes coherently.",
        psychoid_projection={"norm": 0.91},
    )

    assert signal is not None
    assert signal.right is not None
    assert signal.combined_score >= signal.left.score() * 0.5
    payload = signal.to_payload()
    assert payload["contributions"]["psychoid_norm"] == pytest.approx(0.91)
    annotated = resonator.annotate_answer("final", signal)
    assert "[Coherence Integration]" in annotated
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
        psychoid_signal={"resonance": 0.9},
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
