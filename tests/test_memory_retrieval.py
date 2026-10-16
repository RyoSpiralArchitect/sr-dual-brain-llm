import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from core.llm_client import LLMConfig
from core.memory_evidence import bounded_evidence, request_memory_context
from core.memory_retrieval import OpenAIQuestionEmbeddings, QuestionSemanticIndex
from core.shared_memory import MemoryTrace, SharedMemory
from core.unconscious_field import UnconsciousField, PipelineConfig


class SemanticBackend:
    identity = "fixture-semantic-v1"

    def __init__(self):
        self.calls = []

    async def embed(self, questions):
        self.calls.append(list(questions))
        return [(1, 0) if q in {"How should I log errors?", "障害の記録はどう残す？"} else (0, 1) for q in questions]


def test_semantic_questions_cross_language_cache_and_no_answer_embedding():
    backend = SemanticBackend()
    index = QuestionSemanticIndex(backend)
    memory = SharedMemory(semantic_index=index)
    memory.store(MemoryTrace("How should I log errors?", "PRIVATE ANSWER", qid="source", seed_id="seed"))
    asyncio.run(memory.prepare_retrieval("障害の記録はどう残す？"))
    assert memory.retrieve_related_traces("障害の記録はどう残す？")[0].qid == "source"
    assert "PRIVATE ANSWER" not in json.dumps(backend.calls)
    assert index.similarity("missing", "How should I log errors?") is None
    asyncio.run(memory.prepare_retrieval("障害の記録はどう残す？"))
    assert len(backend.calls) == 1
    assert not memory.retrieve_related_traces("Completely unrelated question")


def test_embedding_response_coverage_normalization_and_private_identity(monkeypatch):
    backend = OpenAIQuestionEmbeddings(LLMConfig("openai", "chat", "PRIVATE-KEY",
        api_base="https://user:password@example.test/v1/"), "embedding-model")
    observed = []

    async def post(url, payload, headers, **kwargs):
        observed.append((url, payload, headers, kwargs))
        return {"model": "embedding-model", "data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 0]}],
                "usage": {"total_tokens": 9}}

    monkeypatch.setattr(backend.client, "_post_json", post)
    assert asyncio.run(backend.embed(["one", "two"])) == [(1, 0), (0, 1)]
    assert backend.input_tokens == 9 and backend.calls == 1
    assert observed[0][1] == {"model": "embedding-model", "input": ["one", "two"]}
    assert observed[0][2]["Authorization"] == "Bearer PRIVATE-KEY"
    assert "password" not in backend.identity


@pytest.mark.parametrize("vectors", [[[1, 0]], [[1, 0], [1, 0, 0]], [[0, 0], [1, 0]], [[float('nan'), 0], [1, 0]]])
def test_invalid_embedding_batches_never_partially_update_cache(vectors):
    class BadBackend:
        identity = "bad"

        async def embed(self, _questions):
            return vectors

    index = QuestionSemanticIndex(BadBackend())
    with pytest.raises(ValueError):
        asyncio.run(index.prepare(["one", "two"]))
    assert not index.vectors and index.dimension is None


def test_cache_bound_and_dimension_or_backend_change_are_rejected():
    backend = SemanticBackend()
    index = QuestionSemanticIndex(backend, max_items=2)
    asyncio.run(index.prepare(["one", "two", "three"]))
    assert len(index.vectors) == 2 and "one" in index.vectors
    before = deepcopy(index.vectors)

    async def different(_questions):
        return [(1, 0, 0)]

    backend.embed = different
    with pytest.raises(ValueError, match="dimension changed"):
        asyncio.run(index.prepare(["missing"]))
    assert index.vectors == before
    backend.identity = "new-version"
    assert index.similarity("one", "two") is None
    with pytest.raises(ValueError, match="identity changed"):
        asyncio.run(index.prepare(["one"]))


def test_utf8_limits_bound_requests_and_skip_oversized_questions():
    backend = SemanticBackend()
    index = QuestionSemanticIndex(backend)
    questions = ["日" * 1500 + str(i) for i in range(15)]
    result = asyncio.run(index.prepare(["日" * 2001, *questions]))
    assert sum(len(q.encode()) for q in backend.calls[0]) <= 48000
    assert all(len(q.encode()) <= 6000 for q in backend.calls[0])
    assert result["unprepared_questions"] > 0
    assert "日" * 2001 not in index.vectors


def seed_field(*, mode="question", semantic_index=None):
    field = UnconsciousField(config=PipelineConfig(seed=17), recall_mode=mode, semantic_index=semantic_index)
    question = "Design a mirror metaphor for a settings screen that reflects user intent without trapping users in past behavior. Give one design rule and one measurement."
    draft = "Reflect current intent, allow a fresh start, and measure how often users successfully change their choices."
    mapping = field.analyse(question=question, draft=draft)
    outcome = field.integrate_outcome(mapping=mapping, question=question, draft=draft, final_answer="FINAL QA",
        success=False, decision_state={"source_qid": "source", "novelty": .8})
    assert outcome["seed_cached"]
    field.analyse(question="Common exposure before independent probes", draft="")
    return field, question, draft


def test_new_seed_recall_uses_question_and_preserves_symbolic_draft_mapping():
    field, question, original_draft = seed_field()
    initial = field.memory_state()
    mappings = []
    for draft in (original_draft, "Entirely new wording", ""):
        branch = field.fork()
        mapping = branch.analyse(question=question, draft=draft)
        mappings.append(mapping.text)
        summary = branch.summary(mapping)
        assert summary.emergent_ideas[0].source_qid == "source"
        assert summary.harvest_attempts[0].similarity_source == "question_hash"
        assert summary.harvest_attempts[0].trigger_similarity == 1
    assert len(set(mappings)) == 3  # Symbolic mapping still receives the draft.
    assert field.memory_state() == initial


def test_semantic_seed_recall_is_separate_from_archetype_encoder_and_forks():
    field, question, _draft = seed_field()
    backend = SemanticBackend()
    index = QuestionSemanticIndex(backend)
    asyncio.run(index.prepare([question, "Remember this intent in another language"]))
    field.semantic_index = index
    branch = field.fork()
    mapping = branch.analyse(question="Remember this intent in another language", draft="not matching words")
    summary = branch.summary(mapping)
    assert summary.emergent_ideas
    assert summary.harvest_attempts[0].similarity_source == "question_semantic"
    assert field.pipeline.embedder.__class__.__name__ == "HashEmbedder"
    branch.semantic_index.vectors.clear()
    assert field.semantic_index.vectors


def test_question_encoding_cost_is_independent_of_seed_count(monkeypatch):
    field, question, draft = seed_field()
    for i in range(2):
        seed = deepcopy(field._seed_cache[0])
        seed.seed_id = "another-" + str(i)
        field._seed_cache.append(seed)
    calls = []
    encode = field._question_embedder.encode

    def capture(texts):
        calls.append(texts)
        return encode(texts)

    monkeypatch.setattr(field._question_embedder, "encode", capture)
    mapping = field.analyse(question=question, draft=draft)
    assert len(field.summary(mapping).emergent_ideas) == 3
    assert calls == [[question]]
    assert field._question_embedder is not field.pipeline.embedder


def test_evidence_resolves_final_qa_excludes_expired_superseded_or_ambiguous():
    memory = SharedMemory()
    memory.store(MemoryTrace("Original question", "Final QA answer", qid="source", seed_id="seed"))
    memory.store(MemoryTrace("expired", "never show", qid="old", seed_id="expired", valid_until=0))
    memory.store(MemoryTrace("superseded", "never show", qid="old", seed_id="superseded", evidence_status="superseded"))
    evidence = memory.evidence_for_seeds(["seed", "expired", "superseded", "missing"])
    assert len(evidence) == 1 and evidence[0]["answer"] == "Final QA answer"
    assert evidence[0]["status"] == "unverified"
    context = request_memory_context({"context": "Existing context", "memory_evidence": evidence})
    assert "Final QA answer" in context and '"source_qid": "source"' in context
    assert "not confirmed facts" in context
    assert request_memory_context({"context": context, "memory_evidence": evidence}) == context
    memory.store(MemoryTrace("Duplicate", "conflicting", qid="other", seed_id="seed"))
    assert not memory.evidence_for_seeds(["seed"])
    assert not memory.retrieve_related_traces("expired superseded")
    assert "never show" not in memory.get_context(10)
    assert not memory.get_context(0) and not memory.retrieve_related_traces("Original", -1)
    ambiguous_qid = SharedMemory()
    ambiguous_qid.store(MemoryTrace("q", "a", qid="same", seed_id="unique"))
    ambiguous_qid.store(MemoryTrace("q2", "a2", qid="same", seed_id="another"))
    assert not ambiguous_qid.evidence_for_seeds(["unique"])


def test_evidence_caps_and_invalid_statuses():
    entries = [{"seed_id": str(i), "source_qid": "source", "recorded_at": 10,
                "status": "unverified", "question": "Q" * 2000, "answer": "A" * 10000} for i in range(20)]
    evidence = bounded_evidence(entries)
    assert len(evidence) <= 3 and len(json.dumps(evidence, ensure_ascii=False)) <= 2800
    assert bounded_evidence([{**entries[0], "status": "superseded"}]) == []
    with pytest.raises(ValueError):
        MemoryTrace("q", "a", evidence_status="model_says_verified")
