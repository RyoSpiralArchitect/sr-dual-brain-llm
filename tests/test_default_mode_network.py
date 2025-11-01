from core.default_mode_network import DefaultModeNetwork


def test_default_mode_network_requires_activation():
    network = DefaultModeNetwork(
        min_cache_depth=4,
        stress_release_threshold=0.6,
        activation_bias=0.9,
        cooldown_steps=0,
    )
    summary = {
        "archetype_map": [
            {"id": "shadow", "label": "Shadow", "intensity": 0.35},
            {"id": "persona", "label": "Persona", "intensity": 0.2},
        ],
        "cache_depth": 1,
        "stress_released": 0.05,
        "emergent_ideas": [],
    }
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
    summary = {
        "archetype_map": [
            {"id": "hero", "label": "Hero", "intensity": 0.6},
            {"id": "sage", "label": "Sage", "intensity": 0.3},
            {"id": "shadow", "label": "Shadow", "intensity": 0.12},
        ],
        "cache_depth": 4,
        "stress_released": 0.45,
        "emergent_ideas": [
            {"archetype": "hero", "label": "Triumphant return", "intensity": 0.42},
            {"archetype": "sage", "label": "Guiding insight", "intensity": 0.27},
        ],
    }
    reflections = network.reflect(summary)
    assert reflections
    assert reflections[0]["primary_archetype"] == "hero"
    assert "Hero" in reflections[0]["theme"]
    assert reflections[0]["confidence"] >= 0.3
    # Cooldown should suppress immediate subsequent reflections
    assert network.reflect(summary) == []
