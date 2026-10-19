import asyncio
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from core.memory_retrieval import QuestionSemanticIndex
from core.shared_memory import MemoryTrace, SharedMemory
from test_memory_retrieval import seed_field


class TopicBackend:
    identity = "test-topic-space"

    def __init__(self):
        self.questions = []

    async def embed(self, questions):
        self.questions.extend(questions)
        return [(1, 0, 0) if "audit" in q or "監査" in q else
                (0, 1, 0) if "backup" in q or "バックアップ" in q else (0, 0, 1) for q in questions]


def trace(scope="helix", aliases=("Helix", "ヘリックス"), *, question=None, **kwargs):
    return MemoryTrace(question or f"How long does Project {scope.title()} retain audit logs?", "PRIVATE FINAL ANSWER",
                       retrieval_scope=scope, scope_aliases=aliases, **kwargs)


def test_scope_and_topic_are_independent_cross_language_signals():
    backend = TopicBackend()
    index = QuestionSemanticIndex(backend)
    memory = SharedMemory(semantic_index=index)
    memory.store(trace(qid="helix-logs", seed_id="seed-logs"))
    memory.store(trace(question="How often does Project Helix create database backups?", qid="helix-backup"))
    memory.store(trace("cedar", ("Cedar", "シダー"), qid="cedar-logs"))
    query = "ヘリックスの監査記録はいつまで保存する？"
    asyncio.run(memory.prepare_retrieval(query))
    assert [t.qid for t in memory.retrieve_related_traces(query)] == ["helix-logs"]
    assert [e["source_qid"] for e in memory.evidence_for_seeds(["seed-logs"], question=query)] == ["helix-logs"]
    assert not any("ヘリックス" in q or "helix" in q or "PRIVATE FINAL ANSWER" in q for q in backend.questions)
    assert not memory.evidence_for_seeds(["seed-logs"], question="How long does Project Cedar retain audit logs?")
    menu = "What is Project Helix's lunch menu?"
    asyncio.run(memory.prepare_retrieval(menu))
    assert not memory.retrieve_related_traces(menu)
    assert not memory.evidence_for_seeds(["seed-logs"], question=menu)


def test_local_scope_guard_abstains_on_entity_only_unknown_and_missing_scopes():
    memory = SharedMemory()
    memory.store(trace(qid="helix"))
    memory.store(trace("cedar", ("Cedar", "シダー"), qid="cedar"))
    assert [t.qid for t in memory.retrieve_related_traces("How long does Project Helix retain audit logs?")] == ["helix"]
    for query in ("Project Helix", "What is Project Helix’s cafeteria menu?",
                  "How long does Project Unknown retain audit logs?", "How long are audit logs retained?"):
        assert not memory.retrieve_related_traces(query)
    memory.store(MemoryTrace("How do I record errors?", "Preserved global guidance", qid="global"))
    assert memory.retrieve_related_traces("How do I record errors?")[0].qid == "global"


def test_original_utf8_budget_still_applies_after_alias_removal():
    index = QuestionSemanticIndex(TopicBackend())
    memory = SharedMemory(semantic_index=index)
    source = trace()
    memory.store(source)
    short = "Helix audit logs"
    asyncio.run(memory.prepare_retrieval(short))
    oversized = "Helix " * 1100 + "audit logs"
    assert not index.embedding_questions([oversized])
    assert index.similarity(oversized, source.question) is None


def test_aliases_normalize_width_case_and_do_not_match_neighboring_names():
    memory = SharedMemory()
    memory.store(trace())
    resolver = memory.scope_resolver()
    assert resolver.query("Ｐｒｏｊｅｃｔ　ＨｅＬｉＸ の監査記録").scopes == {"helix"}
    assert resolver.query("ヘリックスの監査記録").scopes == {"helix"}
    assert not resolver.query("Project Helixation audit logs").scopes
    assert not resolver.query("スーパーヘリックスの監査記録").scopes


@pytest.mark.parametrize("alias,query,expected", [
    ("Café", "Project Caféine audit logs", False),
    ("Élan", "Project PréÉlan audit logs", False),
    ("Café", "このCafe\u0301の監査記録", True),
    ("Café", "Project Café audit logs", True),
    ("Café", "Ｐｒｏｊｅｃｔ ＣＡＦÉ audit logs", True),
    ("Мир", "Project Мирный audit logs", False),
    ("١٢", "Project ١٢٣ audit logs", False),
    ("Café", "Project Café\u0301 audit logs", False),
    ("Café", "Project Café\u3099 audit logs", False),
    ("Helix", "Project Helix\u200dation audit logs", False),
    ("ヘリックス", "1ヘリックスの監査記録", False),
    ("Helix", "Project ΩHelix audit logs", False),
])
def test_unicode_name_boundaries_prevent_wrong_scope_assignment(alias, query, expected):
    memory = SharedMemory()
    memory.store(trace("scope", (alias,), question=f"How long does Project {alias} retain audit logs?"))
    cue = memory.scope_resolver().query(query)
    assert bool(cue.scopes) is expected
    if not expected:
        assert not memory.retrieve_related_traces(query)


@pytest.mark.parametrize("query,expected", [
    ("Foo Bars audit logs", {"short"}),
    ("Foo Bar audit logs", {"long"}),
    ("XFoo Bars audit logs", set()),
    ("FooBar audit logs", set()),
])
def test_longest_valid_alias_retries_shorter_candidate_after_boundary_rejection(query, expected):
    memory = SharedMemory()
    memory.store(trace("short", ("Foo",), question="How long does Project Foo retain audit logs?", qid="short"))
    memory.store(trace("long", ("Foo Bar",), question="How long does Project Foo Bar retain audit logs?", qid="long"))
    assert memory.scope_resolver().query(query).scopes == expected
    assert {t.qid for t in memory.retrieve_related_traces(query)} == expected


def test_ambiguous_longest_valid_alias_cannot_fall_back_to_unique_shorter_name():
    memory = SharedMemory()
    memory.store(trace("short", ("Foo",)))
    memory.store(trace("long", ("Foo Bar",)))
    memory.store(trace("other", ("Foo Bar",)))
    cue = memory.scope_resolver().query("Foo Bar audit logs")
    assert cue.ambiguous and cue.scopes == {"long", "other"}
    assert not memory.retrieve_related_traces("Foo Bar audit logs")


@pytest.mark.parametrize("short,long,query", [
    ("Café", "Café Lab", "Café Labs audit logs"),
    ("ヘリックス", "ヘリックス ラボ", "ヘリックス ラボズの監査記録"),
])
def test_overlapping_unicode_aliases_use_the_longest_boundary_valid_match(short, long, query):
    memory = SharedMemory()
    memory.store(trace("short", (short,)))
    memory.store(trace("long", (long,)))
    assert memory.scope_resolver().query(query).scopes == {"short"}


def test_rejected_name_fragments_remain_in_content_when_a_later_alias_matches():
    memory = SharedMemory()
    memory.store(trace("cafe", ("Café",)))
    cue = memory.scope_resolver().query("Caféine and Café audit logs")
    assert cue.scopes == {"cafe"} and "caféine" in cue.content
    assert "and café " not in cue.content


def test_ambiguous_alias_cannot_choose_a_scope_and_registry_tracks_lifecycle():
    memory = SharedMemory()
    first = trace(aliases=("Helix", "North"), qid="one")
    second = trace("cedar", ("Cedar", "North"), qid="two")
    memory.store(first)
    memory.store(second)
    assert not memory.retrieve_related_traces("How long does Project North retain audit logs?")
    memory.past_qas[1] = replace(second, evidence_status="superseded")
    assert memory.scope_resolver().query("Project North").scopes == {"helix"}
    assert memory.retrieve_related_traces("How long does Project North retain audit logs?")[0].qid == "one"
    memory.past_qas[0] = replace(first, valid_until=0)
    assert not memory.scope_resolver().query("Project North").scopes


def test_explicit_multi_scope_query_and_conflicting_records_keep_all_sources():
    memory = SharedMemory()
    memory.store(trace(qid="one"))
    memory.store(trace(qid="two"))
    memory.store(trace("cedar", ("Cedar",), qid="other"))
    selected = memory.retrieve_related_traces("Compare Helix and Cedar audit log retention.")
    assert {t.qid for t in selected} == {"one", "two", "other"}


@pytest.mark.parametrize("kwargs", [
    {"retrieval_scope": "", "scope_aliases": ("name",)},
    {"retrieval_scope": "scope"},
    {"scope_aliases": ("name",)},
    {"retrieval_scope": "scope", "scope_aliases": (" ",)},
    {"retrieval_scope": "scope", "scope_aliases": ("name",) * 17},
])
def test_invalid_scope_metadata_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MemoryTrace("Q", "A", **kwargs)


def test_scope_guard_also_blocks_a_latent_seed_even_with_an_identical_question_hash():
    field, question, _ = seed_field()
    memory = SharedMemory()
    memory.store(trace(question=question, qid="source", seed_id=field._seed_cache[0].seed_id))
    field.recall_scope_resolver = memory.scope_resolver()
    mapping = field.analyse(question=question, draft="different")
    summary = field.summary(mapping)
    assert not summary.emergent_ideas
    assert "scope_mismatch_or_ambiguity" in summary.harvest_attempts[0].failure_reasons
    clone = field.fork()
    assert clone.memory_state() == field.memory_state()
    assert "PRIVATE FINAL ANSWER" not in json.dumps(field.memory_state())


@pytest.mark.parametrize("name,expected", [("Helix", "source"), ("Cedar", "cedar-source")])
def test_latent_recall_resolves_identical_source_questions_by_seed_and_qa_identity(name, expected):
    field, question, _ = seed_field()
    other = deepcopy(field._seed_cache[0])
    other.seed_id = "cedar-seed"
    other.source_qid = "cedar-source"
    field._seed_cache.append(other)
    memory = SharedMemory(semantic_index=QuestionSemanticIndex(TopicBackend()))
    memory.store(trace(question=question, qid="source", seed_id=field._seed_cache[0].seed_id))
    memory.store(trace("cedar", ("Cedar",), question=question, qid="cedar-source", seed_id="cedar-seed"))
    resolver = memory.scope_resolver()
    assert resolver.source(question).ambiguous  # Text alone cannot distinguish them.
    query = f"Project {name}: {question}"
    asyncio.run(memory.prepare_retrieval(query, seed_questions=[question]))
    field.recall_scope_resolver = resolver
    field.semantic_index = memory.semantic_index
    mapping = field.analyse(question=query, draft="New draft")
    ideas = field.summary(mapping).emergent_ideas
    assert [idea.source_qid for idea in ideas] == [expected]
    assert memory.retrieve_related_traces(query)[0].qid == expected
    evidence = memory.evidence_for_seeds([idea.seed_id for idea in ideas], question=query)
    assert [item["source_qid"] for item in evidence] == [expected]


def test_ambiguous_identity_bindings_cannot_assign_a_latent_scope():
    memory = SharedMemory()
    memory.store(trace(question="Same question", qid="duplicate", seed_id="one"))
    memory.store(trace("cedar", ("Cedar",), question="Same question", qid="duplicate", seed_id="two"))
    resolver = memory.scope_resolver()
    assert resolver.source_for_seed("Same question", "one", "duplicate").ambiguous
    assert resolver.source_for_seed("Changed question", "one", "duplicate").ambiguous
