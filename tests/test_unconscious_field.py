from core.unconscious_field import UnconsciousField, PipelineConfig, Prototype


def test_unconscious_field_generates_consistent_topk():
    field = UnconsciousField(config=PipelineConfig(dim=32, seed=7))
    mapping = field.analyse(question="Describe a heroic journey", draft="The protagonist faces a dragon.")

    assert mapping.top_k, "Expected at least one archetype"
    summary = field.summary(mapping)
    assert summary["top_k"], "Summary should include top archetypes"
    assert len(summary["archetype_map"]) == 3
    assert summary["archetype_map"][0]["id"] == mapping.top_k[0]


def test_unconscious_field_accepts_custom_prototypes():
    prototypes = {
        "dreamer": Prototype(id="dreamer", label="Dreamer", keywords=["dream", "night", "symbol"]),
        "guardian": Prototype(id="guardian", label="Guardian", keywords=["protect", "shield", "safety"]),
    }
    field = UnconsciousField(prototypes=prototypes, config=PipelineConfig(dim=16, seed=3))
    mapping = field.analyse(question="I feel safe under the night sky", draft=None)
    assert {score.id for score in mapping.archetype_map} == set(prototypes.keys())
