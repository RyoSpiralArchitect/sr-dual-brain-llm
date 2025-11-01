from core.unconscious_field import UnconsciousField, PipelineConfig, Prototype


def test_unconscious_field_generates_consistent_topk():
    field = UnconsciousField(config=PipelineConfig(dim=32, seed=7))
    mapping = field.analyse(question="Describe a heroic journey", draft="The protagonist faces a dragon.")

    assert mapping.top_k, "Expected at least one archetype"
    summary = field.summary(mapping)
    assert summary.top_k, "Summary should include top archetypes"
    assert len(summary.archetype_map) == 3
    assert summary.archetype_map[0].id == mapping.top_k[0]
    psychoid_signal = summary.psychoid_signal
    assert psychoid_signal, "Psychoid signal should be included in the summary"
    assert psychoid_signal.attention_bias, "Attention bias should not be empty"
    assert len(psychoid_signal.bias_vector) >= 4


def test_unconscious_field_accepts_custom_prototypes():
    prototypes = {
        "dreamer": Prototype(id="dreamer", label="Dreamer", keywords=["dream", "night", "symbol"]),
        "guardian": Prototype(id="guardian", label="Guardian", keywords=["protect", "shield", "safety"]),
    }
    field = UnconsciousField(prototypes=prototypes, config=PipelineConfig(dim=16, seed=3))
    mapping = field.analyse(question="I feel safe under the night sky", draft=None)
    assert {score.id for score in mapping.archetype_map} == set(prototypes.keys())


def test_unconscious_field_incubates_and_releases_emergent_ideas():
    field = UnconsciousField(config=PipelineConfig(dim=24, seed=9))
    question = "A hero cannot yet defeat the dragon guarding the gate."
    draft = "The champion hesitates at the threshold and retreats."
    mapping = field.analyse(question=question, draft=draft)
    outcome = field.integrate_outcome(
        mapping=mapping,
        question=question,
        draft=draft,
        final_answer="",
        success=False,
        decision_state={"novelty": 0.9, "left_conf_raw": 0.3},
    )
    assert outcome["seed_cached"] is True

    # First revisit should continue incubating without immediate emergence.
    revisit = field.analyse(question=question, draft="A hesitant warrior circles the keep.")
    interim_summary = field.summary(revisit)
    assert not interim_summary.emergent_ideas, "Seed should continue incubating"

    # Second revisit should surface the idea.
    second_revisit = field.analyse(
        question=question,
        draft="The warrior recognises the dragon as inner doubt and steps forward.",
    )
    final_summary = field.summary(second_revisit)
    assert final_summary.emergent_ideas, "Incubated seed should surface as an emergent insight"
    assert final_summary.cache_depth >= 0
    psychoid_signal = final_summary.psychoid_signal
    assert psychoid_signal
    assert psychoid_signal.signifier_chain, "Signifier chain should grow across incubations"


def test_unconscious_field_streams_negative_stress():
    field = UnconsciousField(config=PipelineConfig(dim=20, seed=5))
    mapping = field.analyse(question="Describe a failure in detail.", draft="The plan collapsed miserably.")
    field.integrate_outcome(
        mapping=mapping,
        question="Describe a failure in detail.",
        draft="The plan collapsed miserably.",
        final_answer="",
        success=False,
        decision_state={"novelty": 0.1},
        affect={"valence": -0.8, "risk": 0.7},
    )

    recovery_mapping = field.analyse(question="How do we recover?", draft=None)
    recovery_summary = field.summary(recovery_mapping)
    assert recovery_summary.stress_released > 0.0

    # After releasing once, the stress should no longer accumulate.
    follow_up_mapping = field.analyse(question="Plan a new approach", draft=None)
    follow_up_summary = field.summary(follow_up_mapping)
    assert follow_up_summary.stress_released == 0.0
    assert follow_up_summary.psychoid_signal is None or follow_up_summary.psychoid_signal.resonance >= 0.0
