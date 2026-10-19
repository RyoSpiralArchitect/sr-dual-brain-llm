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


def test_support_replay_can_generate_live_vectors_without_an_environment_model(monkeypatch, tmp_path):
    from types import SimpleNamespace
    fixture = small_fixture()
    index = asyncio.run(bench.prepare_vectors(fixture, backend=FakeBackend()))
    checks = bench.RecordedChecks(Checks(), {"fixture_sha256": bench.digest(fixture), "phase": "development"})
    asyncio.run(bench.run(fixture, index, checks, phase="development"))
    fixture_path, cache_path = tmp_path / "fixture.json", tmp_path / "support.json"
    bench.write_private(fixture_path, fixture)
    bench.write_private(cache_path, checks.journal)
    class Embeddings(FakeBackend):
        input_tokens = 0
        served_model = "new-vector"
        def __init__(self, config, model):
            super().__init__()
            assert config is not None and config.model == model == "new-vector"
            self.config = SimpleNamespace(model=model)
    monkeypatch.setattr(bench, "OpenAIQuestionEmbeddings", Embeddings)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "NOT_A_KEY")
    monkeypatch.delenv("LLM_MODEL_ID", raising=False)
    monkeypatch.delenv("MEMORY_ANSWERABILITY_MODEL", raising=False)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--fixture", str(fixture_path), "--phase", "development",
        "--support-cache", str(cache_path), "--embedding-model", "new-vector", "--output-dir", str(tmp_path / "output")])
    bench.main()
    report = bench.read_json(tmp_path / "output/report.json")
    assert report["complete"] and report["provider_calls"] == 0
    assert report["embedding_usage"]["provider_calls"] > 0


def test_recorder_and_frozen_replay_propagate_backend_identity_invalidation():
    class Drift(Checks):
        async def evaluate(self, request):
            if self.calls == 1:
                self.calls += 1
                self.identity = "changed-model"
                raise ValueError("Model drift")
            return await super().evaluate(request)
    source = {"qid": "q", "question": "Q", "answer": "A", "status": "unverified", "recorded_at": 1}
    metadata = {"fixture_sha256": "probe", "phase": "development"}
    live = bench.RecordedChecks(Drift(), metadata)
    gate = bench.AnswerabilityGate(live)
    asyncio.run(gate.prepare("first", [source]))
    assert gate.allows("first", source)
    asyncio.run(gate.prepare("second", [source]))
    assert not gate.allows("first", source)
    live.finish()
    replay = bench.RecordedChecks(None, metadata, frozen=deepcopy(live.journal))
    gate = bench.AnswerabilityGate(replay)
    asyncio.run(gate.prepare("first", [source]))
    assert gate.allows("first", source)
    asyncio.run(gate.prepare("second", [source]))
    assert not gate.allows("first", source)
    replay.finish()
