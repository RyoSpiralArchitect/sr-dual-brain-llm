import json
import math
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "sr-dual-brain-llm" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_unconscious_incubation import (  # noqa: E402
    _near_miss_events_with_turns,
    _filter_sequences,
    _load_sequences,
    _sequence_observation,
    _sequence_lineage,
    _summarise_sequences,
)


def test_load_sequences_accepts_object_wrapper(tmp_path):
    path = tmp_path / "sequences.json"
    path.write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "id": "mirror_return",
                        "tags": ["mirror"],
                        "target_archetypes": ["syzygy"],
                        "turns": [
                            {
                                "role": "seed",
                                "question": "Seed the mirror motif.",
                                "target_archetypes": ["syzygy"],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    sequences = _load_sequences(path)

    assert sequences[0]["id"] == "mirror_return"
    assert sequences[0]["turns"][0]["id"] == "t01"
    assert sequences[0]["turns"][0]["role"] == "seed"
    assert sequences[0]["turns"][0]["target_archetypes"] == ["syzygy"]


def test_filter_sequences_by_id_and_tag():
    sequences = [
        {"id": "a", "tags": ["mirror"], "turns": [{"question": "q"}]},
        {"id": "b", "tags": ["forest"], "turns": [{"question": "q"}]},
    ]

    assert [seq["id"] for seq in _filter_sequences(sequences, only_ids="a", only_tags=None)] == ["a"]
    assert [seq["id"] for seq in _filter_sequences(sequences, only_ids=None, only_tags="forest")] == ["b"]


def test_sequence_observation_marks_target_emergence_and_pressure_delta():
    sequence = {
        "target_archetypes": ["syzygy"],
        "turns": [
            {
                "turn_index": 1,
                "role": "seed",
                "error": None,
                "unconscious": {
                    "emergent_ideas": 0,
                    "emergent_archetypes": [],
                    "incubation_pressure": 0.2,
                    "cache_depth": 0,
                },
                "cached_seed": {
                    "turn_index": 1,
                    "turn_id": "seed",
                    "role": "seed",
                    "archetype": "syzygy",
                    "label": "Syzygy",
                    "origin": "Seed the mirror motif.",
                },
                "unconscious_outcome": {"seed_cached": True},
                "leakage": {"has_internal_leak": False},
            },
            {
                "turn_index": 2,
                "role": "distractor",
                "error": None,
                "unconscious": {
                    "emergent_ideas": 0,
                    "emergent_archetypes": [],
                    "incubation_pressure": 0.5,
                    "cache_depth": 1,
                },
                "unconscious_outcome": {"seed_cached": False},
                "leakage": {"has_internal_leak": False},
            },
            {
                "turn_index": 3,
                "role": "return",
                "error": None,
                "unconscious": {
                    "emergent_ideas": 1,
                    "emergent_details": [
                        {
                            "archetype": "syzygy",
                            "label": "Syzygy",
                            "origin": "Seed the mirror motif.",
                            "incubation_rounds": 3,
                            "trigger_similarity": 0.91,
                        }
                    ],
                    "emergent_archetypes": ["syzygy"],
                    "incubation_pressure": 0.4,
                    "cache_depth": 0,
                },
                "unconscious_outcome": {"seed_cached": True},
                "leakage": {"has_internal_leak": False},
            },
        ],
    }

    obs = _sequence_observation(sequence)

    assert obs["emerged"] is True
    assert obs["first_emergent_turn_index"] == 3
    assert obs["first_emergent_role"] == "return"
    assert obs["target_emergent_hits"] == ["syzygy"]
    assert obs["target_emerged"] is True
    assert obs["seed_to_emergent_transitions"] == ["syzygy->syzygy"]
    assert obs["seed_emergent_same_archetype_rate"] == 1.0
    assert obs["seed_emergent_origin_match_rate"] == 1.0
    assert math.isclose(obs["pressure_delta"], 0.2, rel_tol=1e-9)


def test_sequence_lineage_maps_cached_seed_to_emergent_archetype():
    sequence = {
        "turns": [
            {
                "turn_index": 1,
                "id": "seed",
                "role": "seed",
                "cached_seed": {
                    "turn_index": 1,
                    "turn_id": "seed",
                    "role": "seed",
                    "archetype": "syzygy",
                    "label": "Syzygy",
                    "origin": "A mirror reflects user intent.",
                },
                "unconscious": {"emergent_details": []},
            },
            {
                "turn_index": 3,
                "id": "return",
                "role": "return",
                "cached_seed": {
                    "turn_index": 3,
                    "turn_id": "return",
                    "role": "return",
                    "archetype": "shadow",
                    "label": "Shadow",
                    "origin": "Hidden over-personalization risk.",
                },
                "unconscious": {"emergent_details": []},
            },
            {
                "turn_index": 4,
                "id": "echo",
                "role": "echo_return",
                "unconscious": {
                    "emergent_details": [
                        {
                            "archetype": "syzygy",
                            "label": "Syzygy",
                            "origin": "A mirror reflects user intent.",
                            "incubation_rounds": 3,
                            "trigger_similarity": 0.97,
                        }
                    ]
                },
            },
        ]
    }

    lineage = _sequence_lineage(sequence)

    assert lineage["cached_seed_archetypes"] == ["shadow", "syzygy"]
    assert lineage["emergent_archetypes"] == ["syzygy"]
    assert lineage["archetype_transitions"] == ["syzygy->syzygy"]
    assert lineage["same_archetype_link_rate"] == 1.0
    assert lineage["origin_matched_link_rate"] == 1.0
    assert lineage["links"][0]["match_type"] == "same_archetype_and_origin"


def test_sequence_observation_tracks_echo_near_miss_gap_and_state():
    sequence = {
        "target_archetypes": ["shadow"],
        "turns": [
            {
                "turn_index": 1,
                "id": "seed",
                "role": "seed",
                "cached_seed": {
                    "turn_index": 1,
                    "turn_id": "seed",
                    "role": "seed",
                    "archetype": "shadow",
                    "label": "Shadow",
                    "origin": "An unresolved shadow seed.",
                },
                "unconscious": {
                    "top_k": ["shadow"],
                    "emergent_details": [],
                    "emergent_ideas": 0,
                    "harvest_attempts": [],
                    "incubation_pressure": 0.2,
                    "cache_depth": 1,
                },
                "unconscious_outcome": {"seed_cached": True},
                "leakage": {"has_internal_leak": False},
            },
            {
                "turn_index": 4,
                "id": "echo",
                "role": "echo_return",
                "unconscious": {
                    "top_k": ["shadow"],
                    "emergent_details": [],
                    "emergent_ideas": 0,
                    "emergent_archetypes": [],
                    "harvest_attempts": [
                        {
                            "archetype": "shadow",
                            "label": "Shadow",
                            "intensity": 0.22,
                            "novelty": 0.7,
                            "incubation_rounds": 3,
                            "trigger_similarity": 0.742,
                            "threshold": 0.764,
                            "threshold_gap": 0.022,
                            "threshold_margin": -0.022,
                            "emerged": False,
                            "similarity_pass": False,
                            "incubation_pass": True,
                            "intensity_pass": True,
                            "failure_reasons": ["similarity_below_threshold"],
                            "status": "survived",
                            "origin": "An unresolved shadow seed.",
                        }
                    ],
                    "incubation_pressure": 0.5,
                    "cache_depth": 1,
                },
                "unconscious_outcome": {"seed_cached": False},
                "leakage": {"has_internal_leak": False},
            },
        ],
    }

    near_misses = _near_miss_events_with_turns(sequence["turns"])
    obs = _sequence_observation(sequence)

    assert near_misses[0]["state"] == "near_surface"
    assert obs["echo_near_miss_attempts"] == 1
    assert obs["closest_echo_near_miss"]["archetype"] == "shadow"
    assert math.isclose(obs["closest_echo_near_miss_gap"], 0.022, rel_tol=1e-9)


def test_summarise_sequences_groups_turn_roles():
    sequence = {
        "id": "mirror",
        "observation": {
            "turns": 3,
            "ok_turns": 3,
            "error_turns": 0,
            "leak_turns": 0,
            "seed_cached_turns": 2,
            "emerged": True,
            "first_emergent_turn_index": 3,
            "target_emerged": True,
            "lineage": {
                "links": [
                    {
                        "archetype_transition": "syzygy->syzygy",
                        "same_archetype": True,
                        "origin_match": True,
                    }
                ]
            },
            "seed_emergent_same_archetype_rate": 1.0,
            "seed_emergent_origin_match_rate": 1.0,
            "near_miss_attempts": 1,
            "echo_near_miss_attempts": 1,
            "closest_near_miss_gap": 0.04,
            "closest_echo_near_miss_gap": 0.04,
            "peak_incubation_pressure": 0.6,
            "pressure_delta": 0.3,
            "final_cache_depth": 1,
            "peak_cache_depth": 2,
        },
        "turns": [
            {
                "role": "seed",
                "error": None,
                "unconscious": {
                    "emergent_ideas": 0,
                    "harvest_attempt_count": 0,
                    "harvest_near_miss_count": 0,
                    "incubation_pressure": 0.2,
                    "cache_depth": 0,
                },
                "unconscious_outcome": {"seed_cached": True},
                "archetype_trace": {"cue_alignment": "top_k_aligned"},
                "leakage": {"has_internal_leak": False},
            },
            {
                "role": "distractor",
                "error": None,
                "unconscious": {
                    "emergent_ideas": 0,
                    "harvest_attempt_count": 1,
                    "harvest_near_miss_count": 1,
                    "closest_near_miss": {
                        "archetype": "syzygy",
                        "threshold_gap": 0.04,
                    },
                    "incubation_pressure": 0.4,
                    "cache_depth": 1,
                },
                "unconscious_outcome": {"seed_cached": False},
                "archetype_trace": {"cue_alignment": "unlabeled"},
                "leakage": {"has_internal_leak": False},
            },
            {
                "role": "return",
                "error": None,
                "unconscious": {
                    "emergent_ideas": 1,
                    "harvest_attempt_count": 0,
                    "harvest_near_miss_count": 0,
                    "incubation_pressure": 0.6,
                    "cache_depth": 1,
                },
                "unconscious_outcome": {"seed_cached": True},
                "archetype_trace": {"cue_alignment": "motif_only"},
                "leakage": {"has_internal_leak": False},
            },
            {
                "role": "echo_return",
                "error": None,
                "unconscious": {
                    "emergent_ideas": 0,
                    "emergent_archetypes": [],
                    "top_k": ["syzygy"],
                    "harvest_attempt_count": 1,
                    "harvest_near_miss_count": 1,
                    "harvest_attempts": [
                        {
                            "archetype": "syzygy",
                            "label": "Syzygy",
                            "threshold_gap": 0.04,
                            "trigger_similarity": 0.73,
                            "emerged": False,
                            "similarity_pass": False,
                            "incubation_pass": True,
                            "intensity_pass": True,
                        }
                    ],
                    "closest_near_miss": {
                        "archetype": "syzygy",
                        "threshold_gap": 0.04,
                    },
                    "incubation_pressure": 0.55,
                    "cache_depth": 1,
                },
                "unconscious_outcome": {"seed_cached": False},
                "archetype_trace": {"cue_alignment": "top_k_aligned"},
                "leakage": {"has_internal_leak": False},
            },
        ],
    }

    summary = _summarise_sequences([sequence])

    assert summary["total_sequences"] == 1
    assert summary["emergent_sequence_rate"] == 1.0
    assert summary["target_emergent_sequence_rate"] == 1.0
    assert summary["seed_cached_sequence_rate"] == 1.0
    assert summary["seed_to_emergent_links"] == 1
    assert summary["seed_to_emergent_same_archetype_rate"] == 1.0
    assert summary["seed_to_emergent_origin_match_rate"] == 1.0
    assert summary["seed_to_emergent_transition_counts"] == {"syzygy->syzygy": 1}
    assert summary["max_peak_cache_depth"] == 2
    assert summary["near_miss_attempts"] == 1
    assert summary["echo_near_miss_attempts"] == 1
    assert summary["sequences_with_echo_near_miss"] == 1
    assert summary["echo_near_miss_state_counts"] == {"mid_depth": 1}
    assert math.isclose(summary["avg_closest_echo_near_miss_gap"], 0.04, rel_tol=1e-9)
    assert summary["turns_by_role"]["seed"]["seed_cached_rate"] == 1.0
    assert summary["turns_by_role"]["return"]["emergent_rate"] == 1.0
    assert summary["turns_by_role"]["return"]["cue_motif_only_rate"] == 1.0
    assert summary["turns_by_role"]["echo_return"]["near_miss_turn_rate"] == 1.0
