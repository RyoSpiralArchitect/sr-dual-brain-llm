import asyncio
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))
import benchmark_memory_precision as bench
from test_benchmark_memory_retrieval import FakeBackend


def small_fixture():
    fixture = {"schema_version": 1, "origin": "synthetic_competing_memory_questions",
               "study": "scoped_same_topic_precision_v1", "memories": [], "queries": []}
    for split, scope, word in (("calibration", "Alpha", "apples"), ("evaluation", "Beta", "bananas")):
        fixture["memories"].append({"qid": scope, "question": f"Project {scope} {word}", "answer": "test fact",
                                    "retrieval_scope": scope, "scope_aliases": [scope]})
        negative = "bicycles" if split == "calibration" else "aircraft"
        for kind, topic, gold in (("echo", word, [scope]), ("near_miss", negative, [])):
            fixture["queries"].append({"id": f"{scope}-{kind}", "scene": scope, "topic_group": scope,
                                       "split": split, "kind": kind, "question": f"Project {scope} {topic}",
                                       "relevant_qids": gold, "no_answer_reason": "No aircraft fact"})
    return fixture


def candidate(threshold, f1, false):
    return {"threshold": threshold, "mean_set_f1": f1, "no_answer_queries": 2,
            "false_retrieval_queries": false}


def test_precision_constraint_precedes_f1_and_infeasibility_never_recommends_a_threshold():
    result = bench.choose_precision_threshold([candidate(.25, 1., 1), candidate(.7, .8, 0), candidate(.8, .8, 0)])
    assert result["feasible"] and result["chosen_threshold"] == .8
    impossible = bench.choose_precision_threshold([candidate(.5, .9, 2), candidate(.9, .7, 1), candidate(1., .6, 1)])
    assert not impossible["feasible"] and impossible["chosen_threshold"] is None
    assert impossible["best_effort_diagnostic"]["threshold"] == .9
    with pytest.raises(ValueError, match="no-answer"):
        bench.choose_precision_threshold([])


def test_precision_run_replays_all_rows_and_never_embeds_answers_or_rationales():
    fixture = bench.read_json(bench.DEFAULT_FIXTURE)
    backend, caches = FakeBackend(), []
    report = asyncio.run(bench.run(fixture, backend=backend, on_embeddings=caches.append))
    replay = asyncio.run(bench.run(fixture, embedding_cache=caches[-1]))
    assert report["complete"] and not report["runtime_defaults_changed"]
    assert not report["answer_quality_evaluated"]
    assert replay["semantic"]["provider_calls"] == 0
    assert report["summaries"] == replay["summaries"]
    assert report["precision_calibration"] == replay["precision_calibration"]
    assert report["diagnostics"] == replay["diagnostics"]
    assert [(r["id"], r["selected_qids"]) for r in report["rows"]] == [
        (r["id"], r["selected_qids"]) for r in replay["rows"]]
    assert all(m["answer"] not in backend.questions for m in fixture["memories"])
    assert all(q.get("no_answer_reason") not in backend.questions for q in fixture["queries"])
    assert len(report["rows"]) in {7 * 90, 8 * 90}


def test_feasible_calibration_emits_a_selector_and_ignores_evaluation_labels():
    fixture = small_fixture()
    report = asyncio.run(bench.run(fixture, backend=FakeBackend()))
    assert report["precision_calibration"]["feasible"]
    assert "scoped_precision_calibrated" in report["summaries"]
    mutated = deepcopy(fixture)
    mutated["queries"][-2]["relevant_qids"] = []
    mutated["queries"][-1]["relevant_qids"] = ["Beta"]
    second = asyncio.run(bench.run(mutated, backend=FakeBackend()))
    assert second["precision_calibration"] == report["precision_calibration"]
    assert second["calibration"] == report["calibration"]


def test_infeasible_run_does_not_emit_a_fake_safe_selector():
    class ConstantBackend(FakeBackend):
        async def embed(self, questions):
            return [(1., 0.) for _ in questions]

    report = asyncio.run(bench.run(small_fixture(), backend=ConstantBackend()))
    assert not report["precision_calibration"]["feasible"]
    assert report["precision_calibration"]["chosen_threshold"] is None
    assert "scoped_precision_calibrated" not in report["summaries"]
    assert not any(r["method"] == "scoped_precision_calibrated" for r in report["rows"])


@pytest.mark.parametrize("mutation,match", [
    ("group", "Topic group leakage"), ("cue", "content cue leakage"),
    ("unknown_scope", "known scope"), ("rationale", "rationale"),
])
def test_precision_fixture_rejects_invalid_stress_conditions(mutation, match):
    fixture = small_fixture()
    if mutation == "group":
        for q in fixture["queries"]:
            q["topic_group"] = "shared"
    elif mutation == "cue":
        fixture["queries"][-2]["question"] = "Project Beta apples"
    elif mutation == "unknown_scope":
        fixture["queries"][-1]["question"] = "Project Unknown aircraft"
    else:
        fixture["queries"][-1].pop("no_answer_reason")
    with pytest.raises(ValueError, match=match):
        bench.validate_precision_fixture(fixture)


def test_diagnostics_detect_lexical_override_and_do_not_count_conflict_as_competitor():
    fixture = small_fixture()
    original = fixture["memories"][0]
    common = "Project Alpha red green blue orange yellow indigo scarlet azure amber coral purple"
    original["question"] = common + " violet"
    fixture["memories"].insert(1, dict(original, qid="Alpha-conflict", answer="conflicting fact"))
    fixture["memories"].insert(2, dict(original, qid="Alpha-lamps", question="Project Alpha lamps"))
    query = fixture["queries"][0]
    query["question"] = common + " silver"
    fixture["queries"] = [query]
    index = bench.QuestionSemanticIndex(FakeBackend())
    index.scope_resolver = bench.MemoryScopeResolver([bench.MemoryTrace(**m) for m in fixture["memories"]])
    index.vectors.update({index.scope_resolver.query(query["question"]).content: (1., 0.),
                          index.scope_resolver.query(original["question"]).content: (0., 1.), "lamps": (-1., 0.)})
    rows = bench.retrieval_rows(fixture, index, method="test")
    assert set(rows[0]["selected_qids"]) == {"Alpha", "Alpha-conflict"}
    diagnostics = bench.candidate_diagnostics(fixture, index, rows, method="test")[0]
    assert diagnostics["top_distinct_question_gap"] == 1.
    overrides = [c["qid"] for c in diagnostics["candidates"] if c["below_threshold_lexical_override"]]
    assert set(overrides) == {"Alpha", "Alpha-conflict"}
