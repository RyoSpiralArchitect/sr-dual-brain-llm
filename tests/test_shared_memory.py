from core.shared_memory import SharedMemory


def test_novelty_score_detects_similarity():
    memory = SharedMemory()
    memory.store({"Q": "solar temperature analysis", "A": "about 6000K"})
    completely_new = memory.novelty_score("explain lunar eclipse mechanics")
    similar = memory.novelty_score("solar temperature analysis details")

    assert 0.0 <= completely_new <= 1.0
    assert 0.0 <= similar <= 1.0
    assert similar < completely_new
