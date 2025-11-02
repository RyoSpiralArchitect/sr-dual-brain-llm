from core.prefrontal_cortex import PrefrontalCortex, FocusSummary, SchemaProfiler


def test_focus_synthesis_and_gating_preserve_hippocampal_hint():
    cortex = PrefrontalCortex(gating_threshold=0.1)
    memory_context = "Q:Explain orbital resonance A:Focus on tidal locking\nQ:Discuss gravity wells A:Talk about stable points"
    hippocampal_context = "Orbital resonance triggered deep memory"

    focus = cortex.synthesise_focus(
        question="How does orbital resonance stabilise moons?",
        memory_context=memory_context,
        hippocampal_context=hippocampal_context,
    )
    assert focus.keywords, "focus should surface keywords"
    assert 0.0 <= focus.relevance <= 1.0
    assert focus.hippocampal_overlap >= 0.0

    gated = cortex.gate_context(
        f"{memory_context}\n[Hippocampal recall] {hippocampal_context}",
        focus,
    )
    assert "[Hippocampal recall]" in gated
    assert "Discuss gravity" not in gated, "Irrelevant line should be gated out"


def test_prefrontal_bias_and_tags():
    cortex = PrefrontalCortex()
    focus = FocusSummary(keywords=("orbital", "resonance"), relevance=0.18, hippocampal_overlap=0.6)

    adjusted = cortex.adjust_consult_bias(0.0, focus)
    assert adjusted <= 0.05 and adjusted > -0.1

    metric = cortex.focus_metric(focus)
    assert 0.0 <= metric <= 1.0

    tags = set(cortex.tags(focus))
    assert "focus_low" in tags
    assert "hippocampal_supported" in tags
    assert any(tag.startswith("focus_") for tag in tags)


def test_schema_profiler_flags_perfectionism_and_modes():
    profiler = SchemaProfiler()
    profile = profiler.profile_turn(
        question="I always fail and everybody leaves me alone.",
        answer="Let's plan balanced steps together so you feel supported.",
    )
    assert "perfectionism" in profile.user_schemas
    assert "abandonment" in profile.user_schemas
    assert "healthy_adult" in profile.agent_modes
    tags = set(profile.tags())
    assert "schema_user_perfectionism" in tags
    assert "mode_agent_healthy_adult" in tags


def test_prefrontal_cortex_profiles_turn_with_affect():
    cortex = PrefrontalCortex()
    focus = FocusSummary(keywords=("fail", "alone"), relevance=0.5, hippocampal_overlap=0.3)
    profile = cortex.profile_turn(
        question="I always fail and everyone leaves me alone.",
        answer="Let's explore calmer options and support you with care.",
        focus=focus,
        affect={"valence": -0.6, "arousal": 0.7},
    )
    assert profile is not None
    assert "perfectionism" in profile.user_schemas
    assert "vulnerable_child" in profile.user_modes
    assert "healthy_adult" in profile.agent_modes
    assert profile.to_dict()["confidence"] >= 0.0
