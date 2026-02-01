from core.shared_memory import MemoryTrace, SharedMemory


def test_novelty_score_detects_similarity():
    memory = SharedMemory()
    memory.store({"Q": "solar temperature analysis", "A": "about 6000K"})
    completely_new = memory.novelty_score("explain lunar eclipse mechanics")
    similar = memory.novelty_score("solar temperature analysis details")

    assert 0.0 <= completely_new <= 1.0
    assert 0.0 <= similar <= 1.0
    assert similar < completely_new


def test_retrieve_related_uses_similarity_and_tags():
    memory = SharedMemory(max_items=4)
    memory.store({"Q": "Explain diffusion in solids", "A": "Use Fick's law."}, tags={"physics"})
    memory.store({"Q": "Describe osmosis", "A": "Semi-permeable membranes."}, tags={"biology"})
    # Insert a trace directly to verify MemoryTrace support and recency weighting
    memory.store(
        MemoryTrace(question="Outline decoherence in quantum systems", answer="Phase information dissipates."),
    )

    related = memory.retrieve_related("quantum decoherence with phase info", n=2)
    assert "decoherence" in related
    assert "diffusion" not in related

    # Ensure buffer trims older entries once capacity exceeded
    memory.store({"Q": "Detail thermodynamic entropy", "A": "S = k ln W."})
    memory.store({"Q": "Summarise entropy increase", "A": "Second law"})
    assert len(memory.past_qas) <= memory.max_items


def test_record_dialogue_flow_tracks_architecture():
    memory = SharedMemory()
    steps = [{"phase": "left_draft", "role": "left"}]
    architecture = [
        {"stage": "perception", "modules": ["Amygdala"], "signals": {"novelty": 0.2}},
        {"stage": "inner_dialogue", "modules": ["LeftBrainModel"], "step_count": 1},
    ]

    memory.record_dialogue_flow(
        "qid-1",
        leading_brain="left",
        follow_brain="right",
        preview="hello",
        steps=steps,
        architecture=architecture,
    )

    record = memory.dialogue_flow("qid-1")
    assert record is not None
    assert record["step_count"] == 1
    assert record["architecture_count"] == 2
    assert record["architecture"][0]["stage"] == "perception"


def test_memory_retrieval_sanitises_internal_and_coaching_lines():
    memory = SharedMemory()
    memory.store(
        {
            "Q": "hello",
            "A": "こんにちは！\nqid 123\n- Add a warm, informal tone.\n[Hemisphere Routing]\nmode: balanced",
        }
    )

    related = memory.retrieve_related("hello", n=1)
    assert "こんにちは" in related
    assert "qid 123" not in related
    assert "Add a warm" not in related
    assert "Hemisphere" not in related


def test_schema_memory_retrieval_sanitises_internal_lines():
    memory = SharedMemory()
    memory.put_kv(
        "schema_memories",
        [
            {
                "summary": "[Architecture Path]\n1. perception...\nUser likes cats.",
                "tags": ["cats"],
            }
        ],
    )

    out = memory.retrieve_schema_related("cats", n=1)
    assert "User likes cats" in out
    assert "Architecture Path" not in out
