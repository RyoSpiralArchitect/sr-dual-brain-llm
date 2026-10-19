import asyncio
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))
import benchmark_scoped_memory as bench
from test_benchmark_memory_retrieval import FakeBackend


def test_scoped_fixture_is_disjoint_and_retains_all_six_comparators_and_rows():
    fixture = bench.read_json(bench.DEFAULT_FIXTURE)
    backend = FakeBackend()
    caches = []
    report = asyncio.run(bench.run(fixture, backend=backend, on_embeddings=caches.append))
    assert report["complete"] and not report["answer_quality_evaluated"]
    assert len(report["rows"]) == 6 * len(fixture["queries"])
    assert not any(m["answer"] in backend.questions for m in fixture["memories"])
    replay = asyncio.run(bench.run(fixture, embedding_cache=caches[-1]))
    assert replay["summaries"] == report["summaries"]
    assert replay["semantic"]["provider_calls"] == 0
    for result in report["calibration"].values():
        assert not result["evaluation_used_for_selection"]
    bad = deepcopy(caches[-1])
    bad["vectors"].pop(next(iter(bad["vectors"])))
    with pytest.raises(ValueError, match="complete fixture"):
        asyncio.run(bench.run(fixture, embedding_cache=bad))
    mutated = deepcopy(fixture)
    for query in mutated["queries"]:
        if query["split"] == "evaluation":
            query["relevant_qids"] = []
    second = asyncio.run(bench.run(mutated, backend=FakeBackend()))
    assert second["calibration"] == report["calibration"]
