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
