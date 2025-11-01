from core.default_mode_network import DefaultModeNetwork
from core.schema import (
    ArchetypeActivation,
    EmergentIdeaModel,
    GeometryModel,
    UnconsciousSummaryModel,
)


def _make_summary(
    *,
    archetypes,
    cache_depth: int,
    stress_released: float = 0.0,
    emergent=None,
) -> UnconsciousSummaryModel:
    return UnconsciousSummaryModel(
        top_k=[entry["id"] for entry in archetypes],
        geometry=GeometryModel(r=0.0, theta=0.0, curvature_proxy=0.0),
        archetype_map=[
            ArchetypeActivation(**entry)
            for entry in archetypes
        ],
        cache_depth=cache_depth,
        stress_released=stress_released,
        emergent_ideas=[
            EmergentIdeaModel(
                archetype=item["archetype"],
                label=item["label"],
                intensity=item.get("intensity", 0.0),
                incubation_rounds=item.get("incubation_rounds", 1),
                trigger_similarity=item.get("trigger_similarity", 0.0),
                origin=item.get("origin", "test"),
            )
            for item in (emergent or [])
        ],
    )


def test_default_mode_network_requires_activation():
    network = DefaultModeNetwork(
        min_cache_depth=4,
        stress_release_threshold=0.6,
        activation_bias=0.9,
        cooldown_steps=0,
    )
    summary = _make_summary(
        archetypes=[
            {"id": "shadow", "label": "Shadow", "intensity": 0.35},
            {"id": "persona", "label": "Persona", "intensity": 0.2},
        ],
        cache_depth=1,
        stress_released=0.05,
    )
    reflections = network.reflect(summary)
    assert reflections == []


def test_default_mode_network_surfaces_reflection_when_engaged():
    network = DefaultModeNetwork(
        min_cache_depth=1,
        stress_release_threshold=0.1,
        activation_bias=0.3,
        cooldown_steps=1,
        max_reflections=3,
    )
    summary = _make_summary(
        archetypes=[
            {"id": "hero", "label": "Hero", "intensity": 0.6},
            {"id": "sage", "label": "Sage", "intensity": 0.3},
            {"id": "shadow", "label": "Shadow", "intensity": 0.12},
        ],
        cache_depth=4,
        stress_released=0.45,
        emergent=[
            {"archetype": "hero", "label": "Triumphant return", "intensity": 0.42},
            {"archetype": "sage", "label": "Guiding insight", "intensity": 0.27},
        ],
    )
    reflections = network.reflect(summary)
    assert reflections
    assert reflections[0].primary_archetype == "hero"
    assert "Hero" in reflections[0].theme
    assert reflections[0].confidence >= 0.3
    # Cooldown should suppress immediate subsequent reflections
    assert network.reflect(summary) == []
