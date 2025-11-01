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
