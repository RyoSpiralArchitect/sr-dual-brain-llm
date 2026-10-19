#!/usr/bin/env python3
"""Same-scope adjacent-facet stress test and constrained calibration; no answer grading."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import json
from pathlib import Path

import benchmark_scoped_memory as scoped
from benchmark_frozen_drafts import read_json, write_private
from benchmark_controlled_memory import checkpoint
from benchmark_memory_retrieval import FrozenEmbeddings, retrieval_rows, summarize, validate_fixture
from core.llm_client import load_llm_config
from core.memory_retrieval import OpenAIQuestionEmbeddings, QuestionSemanticIndex, normalized_vector
from core.memory_scope import MemoryScopeResolver
from core.shared_memory import MemoryTrace, SharedMemory, _derive_tags, _tokenize

DEFAULT_FIXTURE = scoped.PROJECT / "examples/memory_precision_benchmark.json"
PRECISION_THRESHOLDS = tuple(round(.20 + i * .05, 2) for i in range(17))


def validate_precision_fixture(fixture):
    validate_fixture(fixture)
    if fixture.get("study") != "scoped_same_topic_precision_v1":
        raise ValueError("Expected same-topic precision study")
    memories = [MemoryTrace(**item) for item in fixture["memories"]]
    resolver = MemoryScopeResolver(memories)
    by_id = {trace.qid: trace for trace in memories}
    groups, cues = defaultdict(set), defaultdict(set)
    scene_groups = defaultdict(set)
    for query in fixture["queries"]:
        group = query.get("topic_group")
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Declared topic groups required")
        groups[query["split"]].add(group)
        scene_groups[query["scene"]].add(group)
        cue = resolver.query(query["question"])
        if cue.ambiguous or cue.scopes != {query["scene"]}:
            raise ValueError("Every query must identify its declared known scope")
        cues[query["split"]].add(cue.content)
        if any(by_id[qid].retrieval_scope != query["scene"] for qid in query["relevant_qids"]):
            raise ValueError("Relevant sources must belong to the query scope")
        if not query["relevant_qids"] and not str(query.get("no_answer_reason", "")).strip():
            raise ValueError("No-answer rationale required")
    if any(len(values) != 1 for values in scene_groups.values()):
        raise ValueError("A scene must have one declared topic group")
    if groups["calibration"] & groups["evaluation"]:
        raise ValueError("Topic group leakage between splits")
    if cues["calibration"] & cues["evaluation"]:
        raise ValueError("Exact content cue leakage between splits")
    for split in ("calibration", "evaluation"):
        queries = [q for q in fixture["queries"] if q["split"] == split]
        if not any(q["relevant_qids"] for q in queries) or not any(not q["relevant_qids"] for q in queries):
            raise ValueError("Positive and no-answer queries required in both splits")
    return resolver


def choose_precision_threshold(candidates):
    """Zero calibration false-retrieval queries is a constraint, not a tie breaker."""
    if not candidates or any(item["no_answer_queries"] <= 0 for item in candidates):
        raise ValueError("Calibration requires no-answer queries")
    feasible = [item for item in candidates if item["false_retrieval_queries"] == 0]
    best = max(feasible, key=lambda item: (item["mean_set_f1"], item["threshold"])) if feasible else None
    diagnostic = max(candidates, key=lambda item: (-item["false_retrieval_queries"],
                                                  item["mean_set_f1"], item["threshold"]))
    return {"feasible": best is not None, "chosen_threshold": best["threshold"] if best else None,
            "max_calibration_false_retrieval_rate": 0.0,
            "objective": "calibration macro source-set F1 subject to zero no-answer false retrievals; stricter ties",
            "evaluation_used_for_selection": False,
            "best_effort_diagnostic": diagnostic,
            "candidates": candidates}


def calibrate_precision(fixture, index):
    candidates = []
    for threshold in PRECISION_THRESHOLDS:
        index.scope_min_similarity = threshold
        summary = summarize(retrieval_rows(fixture, index, split="calibration"))
        candidates.append({"threshold": threshold, **summary,
                           "false_retrieval_queries": summary["no_answer_queries"] - summary["correct_abstentions"]})
    result = choose_precision_threshold(candidates)
    if result["feasible"]:
        index.scope_min_similarity = result["chosen_threshold"]
    return result


def candidate_diagnostics(fixture, index, rows, *, method):
    memory = SharedMemory(semantic_index=index)
    for item in fixture["memories"]:
        memory.store(MemoryTrace(**item))
    resolver = memory.scope_resolver()
    selected = {row["id"]: row["selected_qids"] for row in rows if row["method"] == method}
    results = []
    for query in fixture["queries"]:
        cue = resolver.query(query["question"])
        candidates, distinct = [], {}
        for position, trace in enumerate(memory.past_qas):
            if not resolver.allows_trace(cue, trace):
                continue
            content = resolver.source(trace.question, trace.retrieval_scope).content
            similarity = index.similarity(query["question"], trace.question, source_scope=trace.retrieval_scope)
            threshold = index.threshold_for(query["question"], trace.question, source_scope=trace.retrieval_scope)
            lexical = memory._score_trace(trace, _tokenize(cue.content), _derive_tags(cue.content),
                                          (position + 1) / len(memory.past_qas), trace_question=content)
            retained = trace.qid in selected[query["id"]]
            candidates.append({"qid": trace.qid, "cosine": similarity, "lexical_score": lexical,
                               "threshold": threshold, "selected": retained,
                               "below_threshold_lexical_override": bool(retained and similarity is not None
                                                                        and similarity < threshold and lexical >= .75)})
            if similarity is not None:
                distinct[content] = similarity
        scores = sorted(distinct.values(), reverse=True)
        results.append({"id": query["id"], "method": method, "split": query["split"], "kind": query["kind"],
                        "top_distinct_question_gap": scores[0] - scores[1] if len(scores) >= 2 else None,
                        "candidates": candidates})
    return results


async def run(fixture, *, backend=None, embedding_cache=None, on_embeddings=None):
    resolver = validate_precision_fixture(fixture)
    cache = embedding_cache

    def retain_cache(data):
        nonlocal cache
        cache = data
        if on_embeddings:
            on_embeddings(data)

    report = await scoped.run(fixture, backend=backend, embedding_cache=embedding_cache, on_embeddings=retain_cache)
    report.update(study=fixture["study"], precision_calibration=None, diagnostics=[],
                  runtime_defaults_changed=False)
    if cache is not None:
        index = QuestionSemanticIndex(FrozenEmbeddings(cache))
        index.scope_resolver = resolver
        index.vectors.update({q: normalized_vector(v) for q, v in cache["vectors"].items()})
        index.dimension = cache["dimension"]
        index.scope_min_similarity = .25
        report["rows"].extend(retrieval_rows(fixture, index, method="scoped_prior_calibrated_025"))
        for method, threshold in (("scoped_hybrid_default", .50), ("scoped_prior_calibrated_025", .25)):
            index.scope_min_similarity = threshold
            report["diagnostics"].extend(candidate_diagnostics(fixture, index, report["rows"], method=method))
        result = calibrate_precision(fixture, index)
        report["precision_calibration"] = result
        if result["feasible"]:
            report["rows"].extend(retrieval_rows(fixture, index, method="scoped_precision_calibrated"))
        # An infeasible constraint deliberately produces no recommended selector.
    report["summaries"] = {method: scoped.summaries([r for r in report["rows"] if r["method"] == method])
                           for method in sorted({r["method"] for r in report["rows"]})}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--semantic-model", help="Explicitly authorizes embedding API calls")
    mode.add_argument("--embedding-cache", type=Path, help="Complete frozen checkpoint; zero provider calls")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    backend = None
    if args.semantic_model:
        config = load_llm_config("MEMORY", default_model=args.semantic_model)
        if config is None:
            raise SystemExit("Configured provider credentials required")
        backend = OpenAIQuestionEmbeddings(config, args.semantic_model)
    try:
        report = asyncio.run(run(read_json(args.fixture), backend=backend,
                                 embedding_cache=read_json(args.embedding_cache) if args.embedding_cache else None,
                                 on_embeddings=lambda data: checkpoint(args.output_dir / "embeddings.partial.json", data)))
    except Exception as exc:
        write_private(args.output_dir / "failure.json", {"complete": False, "error_type": type(exc).__name__,
                                                        "provider_calls": getattr(backend, "calls", 0),
                                                        "input_tokens": getattr(backend, "input_tokens", None)})
        raise SystemExit("Benchmark failed: " + type(exc).__name__ + "; private checkpoint retained") from None
    write_private(args.output_dir / "report.json", report)
    print(json.dumps({"evaluation": {name: value["evaluation"]["all"] for name, value in report["summaries"].items()},
                      "precision_calibration": {key: report["precision_calibration"][key]
                                                for key in ("feasible", "chosen_threshold")}
                      if report["precision_calibration"] else None}, indent=2))


if __name__ == "__main__":
    main()
