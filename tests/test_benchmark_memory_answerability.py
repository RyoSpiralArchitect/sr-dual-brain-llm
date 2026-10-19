import asyncio
from copy import deepcopy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))
import benchmark_memory_answerability as bench
from test_benchmark_memory_precision import small_fixture
from test_benchmark_memory_retrieval import FakeBackend
from test_memory_answerability import Checks


def test_response_replay_preserves_decisions_and_rejects_missing_or_mismatched_inputs():
    fixture = small_fixture()
    index = asyncio.run(bench.prepare_vectors(fixture, backend=FakeBackend()))
    metadata = {"fixture_sha256": bench.digest(fixture), "phase": "development"}
    live = bench.RecordedChecks(Checks(), metadata)
    first = asyncio.run(bench.run(fixture, index, live, phase="development"))
    frozen = deepcopy(live.journal)
    replay = bench.RecordedChecks(None, metadata, frozen=frozen)
    second = asyncio.run(bench.run(fixture, index, replay, phase="development"))
    assert second["provider_calls"] == 0 and second["complete"]
    assert first["summaries"] == second["summaries"]
    assert [{k: v for k, v in r.items() if k != "latency_ms"} for r in first["rows"]] == [
        {k: v for k, v in r.items() if k != "latency_ms"} for r in second["rows"]]
    assert all(q["split"] == "calibration" for q in second["rows"])
    incomplete = deepcopy(frozen)
    incomplete["responses"].pop(next(iter(incomplete["responses"])))
    with pytest.raises(ValueError, match="coverage"):
        asyncio.run(bench.run(fixture, index, bench.RecordedChecks(None, metadata, frozen=incomplete), phase="development"))
    corrupt = deepcopy(frozen)
    next(iter(corrupt["responses"].values()))["request"]["question"] = "Changed request"
    with pytest.raises(ValueError, match="coverage"):
        asyncio.run(bench.run(fixture, index, bench.RecordedChecks(None, metadata, frozen=corrupt), phase="development"))
    with pytest.raises(ValueError, match="differs"):
        bench.RecordedChecks(None, dict(metadata, phase="evaluation"), frozen=frozen)
    incomplete["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        bench.RecordedChecks(None, metadata, frozen=incomplete)


def test_fixture_keeps_known_scopes_topic_groups_and_exact_cues_disjoint():
    fixture = bench.read_json(bench.DEFAULT_FIXTURE)
    bench.validate_fixture(fixture)
    resolver = bench.MemoryScopeResolver([bench.MemoryTrace(**m) for m in fixture["memories"]])
    groups, cues = {}, {}
    for split in ("calibration", "evaluation"):
        queries = [q for q in fixture["queries"] if q["split"] == split]
        groups[split] = {q["topic_group"] for q in queries}
        cues[split] = {resolver.query(q["question"]).content for q in queries}
        assert all(resolver.query(q["question"]).scopes == {q["scene"]} for q in queries)
    assert not groups["calibration"] & groups["evaluation"]
    assert not cues["calibration"] & cues["evaluation"]
