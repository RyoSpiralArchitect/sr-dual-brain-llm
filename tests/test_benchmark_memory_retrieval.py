import asyncio
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))
import benchmark_memory_retrieval as bench  # noqa: E402
from core.memory_retrieval import QuestionSemanticIndex  # noqa: E402


class FakeBackend:
    identity = "test-only-semantic-v1"

    def __init__(self):
        self.calls = 0
        self.questions = []

    async def embed(self, questions):
        self.calls += 1
        self.questions.extend(questions)
        return [tuple(hashlib.sha256(q.encode()).digest()[:8]) for q in questions]


def test_offline_competition_keeps_all_queries_and_separates_metadata_filters():
    fixture = bench.read_json(bench.DEFAULT_FIXTURE)
    report = asyncio.run(bench.run(fixture))
    assert report["complete"] and not report["answer_quality_evaluated"]
    assert report["semantic"] is None
    assert len(report["rows"]) == 2 * len(fixture["queries"])
    old = next(row for row in report["rows"] if row["method"] == "legacy_lexical" and row["id"] == "aster-echo")
    new = next(row for row in report["rows"] if row["method"] == "question_lexical" and row["id"] == "aster-echo")
    assert old["selected_qids"][0] == "aster-old"
    assert new["selected_qids"][0] == "aster-current"
    assert "aster-old" not in new["selected_qids"]
    for row in report["recall_ablation"]["rows"]:
        if row["method"] == "question" and row["variant"] == "echo":
            assert row["recovered"]
    assert all(row["abstained"] for row in report["rows"] if row["method"] == "question_lexical" and row["kind"] == "unrelated")


def test_calibration_does_not_consult_evaluation_labels_and_embedding_calls_are_batched():
    fixture = bench.read_json(bench.DEFAULT_FIXTURE)
    backend = FakeBackend()
    checkpoints = []
    report = asyncio.run(bench.run(fixture, backend=backend, on_embeddings=checkpoints.append))
    assert backend.calls == 1 and checkpoints
    assert report["complete"] and len(report["rows"]) == 3 * len(fixture["queries"])
    assert not report["calibration"]["evaluation_used_for_selection"]
    assert not any(m["answer"] in backend.questions for m in fixture["memories"])
    replay = asyncio.run(bench.run(fixture, embedding_cache=checkpoints[-1]))
    assert replay["summaries"] == report["summaries"]
    assert replay["semantic"]["provider_calls"] == 0
    assert replay["semantic"]["frozen_replay"]
    bad = deepcopy(checkpoints[-1])
    bad["vectors"].pop(next(iter(bad["vectors"])))
    with pytest.raises(ValueError, match="complete fixture"):
        asyncio.run(bench.run(fixture, embedding_cache=bad))
    index = QuestionSemanticIndex(FakeBackend())
    asyncio.run(index.prepare(backend.questions))
    expected = bench.calibrate(fixture, index)
    mutated = deepcopy(fixture)
    for query in mutated["queries"]:
        if query["split"] == "evaluation":
            query["relevant_qids"] = []
    assert bench.calibrate(mutated, index) == expected


def test_retrieval_fixture_rejects_scene_or_source_leakage():
    fixture = bench.read_json(bench.DEFAULT_FIXTURE)
    fixture["queries"][0]["split"] = "evaluation"
    with pytest.raises(ValueError, match="leakage"):
        bench.validate_fixture(fixture)
    fixture = bench.read_json(bench.DEFAULT_FIXTURE)
    query = deepcopy(fixture["queries"][0])
    query.update(id="renamed", scene="different-name", split="evaluation")
    fixture["queries"].append(query)
    with pytest.raises(ValueError, match="source leakage"):
        bench.validate_fixture(fixture)
