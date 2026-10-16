#!/usr/bin/env python3
"""Competing-memory retrieval and frozen-state draft ablation; not answer grading."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from core.llm_client import load_llm_config  # noqa: E402
from core.memory_retrieval import OpenAIQuestionEmbeddings, QuestionSemanticIndex, normalized_vector  # noqa: E402
from core.shared_memory import MemoryTrace, SharedMemory, _derive_tags, _tokenize  # noqa: E402
from benchmark_controlled_memory import DEFAULT_FIXTURE as RECALL_FIXTURE, make_seed  # noqa: E402
from benchmark_frozen_drafts import digest, read_json, write_private  # noqa: E402
from benchmark_system2_ab import _source_revision  # noqa: E402
from core.unconscious_field import PipelineConfig  # noqa: E402

DEFAULT_FIXTURE = PROJECT / "examples/memory_retrieval_benchmark.json"
REWRITTEN_DRAFTS = {
    "mirror_revisit": "Let people revise preferences freely and track successful updates.",
    "forest_revisit": "Rank upcoming work by value and available resources; change direction when new observations arrive.",
    "mask_revisit": "Keep choices understandable and let people access deeper controls when needed.",
}


def validate_fixture(fixture):
    if fixture.get("schema_version") != 1 or fixture.get("origin") != "synthetic_competing_memory_questions":
        raise ValueError("Expected competing-memory fixture")
    memories, queries = fixture["memories"], fixture["queries"]
    ids = [m["qid"] for m in memories]
    if not memories or not queries or len(set(ids)) != len(ids) or len({q["id"] for q in queries}) != len(queries):
        raise ValueError("Unique memory and query IDs required")
    scenes = {}
    source_splits = {}
    for query in queries:
        if query["split"] not in {"calibration", "evaluation"} or not query["question"].strip():
            raise ValueError("Split and nonempty query required")
        if not set(query["relevant_qids"]).issubset(ids):
            raise ValueError("Unknown relevant source")
        if query["scene"] in scenes and scenes[query["scene"]] != query["split"]:
            raise ValueError("Scene leakage between calibration and evaluation")
        scenes[query["scene"]] = query["split"]
        for source in query["relevant_qids"]:
            if source in source_splits and source_splits[source] != query["split"]:
                raise ValueError("Relevant source leakage between splits")
            source_splits[source] = query["split"]
    if set(scenes.values()) != {"calibration", "evaluation"}:
        raise ValueError("Both disjoint splits required")
    for item in memories:
        MemoryTrace(**item)


def legacy_retrieve(memory, question, n=3):
    """Pre-change lexical selector, including no evidence-status/expiry filter."""
    tokens, tags = set(_tokenize(question)), set(_derive_tags(question))
    scored = []
    for i, trace in enumerate(memory.past_qas):
        trace_tokens = set(trace.question_tokens)
        lexical = len(tokens & trace_tokens) / len(tokens | trace_tokens) if tokens | trace_tokens else 0
        bonus = len(tags & set(trace.tags)) / max(len(tags), 1)
        if lexical or bonus:
            scored.append((.75 * lexical + .2 * bonus + .05 * (i + 1) / len(memory.past_qas), trace))
    return [t for _score, t in sorted(scored, key=lambda pair: pair[0], reverse=True)[:n]]


def retrieval_rows(fixture, index=None, *, method="question_lexical", split=None):
    memory = SharedMemory(semantic_index=index)
    for item in fixture["memories"]:
        memory.store(MemoryTrace(**item))
    rows = []
    for query in fixture["queries"]:
        if split is not None and query["split"] != split:
            continue
        started = time.perf_counter()
        traces = (legacy_retrieve(memory, query["question"]) if method == "legacy_lexical"
                  else memory.retrieve_related_traces(query["question"]))
        selected = [trace.qid for trace in traces]
        gold = set(query["relevant_qids"])
        overlap = len(gold & set(selected))
        rows.append({"id": query["id"], "scene": query["scene"], "split": query["split"], "kind": query["kind"],
                     "method": method, "selected_qids": selected, "relevant_qids": sorted(gold),
                     "hit_at_1": bool(selected and selected[0] in gold),
                     "recall_at_3": overlap / len(gold) if gold else None,
                     "precision_at_3": overlap / len(selected) if selected else None,
                     "set_f1": 2 * overlap / (len(gold) + len(selected)) if gold or selected else 1,
                     "abstained": not selected, "unrelated_slots": len(set(selected) - gold),
                     "latency_ms": (time.perf_counter() - started) * 1000})
    return rows


def summarize(rows):
    positive = [r for r in rows if r["relevant_qids"]]
    negative = [r for r in rows if not r["relevant_qids"]]
    return {"queries": len(rows), "positive_queries": len(positive), "no_answer_queries": len(negative),
            "hits_at_1": sum(r["hit_at_1"] for r in positive),
            "hits_at_3": sum(r["recall_at_3"] == 1 for r in positive),
            "correct_abstentions": sum(r["abstained"] for r in negative),
            "unrelated_selected_slots": sum(r["unrelated_slots"] for r in rows),
            "selected_slots": sum(len(r["selected_qids"]) for r in rows),
            "mean_set_f1": sum(r["set_f1"] for r in rows) / len(rows) if rows else None}


def calibrate(fixture, index):
    candidates = []
    for threshold in (.40, .45, .50, .55, .60, .65, .70, .75, .80, .85):
        index.min_similarity = threshold
        summary = summarize(retrieval_rows(fixture, index, method="hybrid", split="calibration"))
        candidates.append({"threshold": threshold, **summary})
    # Frozen objective: macro set F1, then the stricter threshold on ties.
    chosen = max(candidates, key=lambda item: (item["mean_set_f1"], item["threshold"]))
    index.min_similarity = chosen["threshold"]
    return {"objective": "calibration macro set-F1; stricter threshold on ties", "chosen_threshold": chosen["threshold"],
            "candidates": candidates, "evaluation_used_for_selection": False}


def recall_ablation(index=None):
    rows = []
    for case in read_json(RECALL_FIXTURE)["retrieval_cases"]:
        base, outcome = make_seed(case, PipelineConfig(seed=17))
        base.analyse(question="Prepare the independent test branches.", draft="")
        initial = digest(base.memory_state())
        for mode in ("legacy_payload", "question", "hybrid"):
            if mode == "hybrid" and index is None:
                continue
            for draft_mode, draft in (("same", case["probe_draft"]), ("none", ""),
                                       ("rewritten", REWRITTEN_DRAFTS[case["id"]])):
                for variant, question in case["variants"].items():
                    branch = base.fork()
                    branch.recall_mode = "legacy_payload" if mode == "legacy_payload" else "question"
                    branch.semantic_index = deepcopy(index) if mode == "hybrid" else None
                    mapping = branch.analyse(question=question, draft=draft)
                    summary = branch.summary(mapping)
                    attempt = summary.harvest_attempts[0].to_payload()
                    rows.append({"case": case["id"], "method": mode, "draft": draft_mode, "variant": variant,
                                 "seed_id": outcome["seed_id"], "initial_seed_state_sha256": initial,
                                 "recovered": any(idea.seed_id == outcome["seed_id"] for idea in summary.emergent_ideas),
                                 "attempt": attempt})
        assert digest(base.memory_state()) == initial
    return {"scope": "three field-only positive scenes; rewritten drafts not independently validated",
            "quality_evaluated": False, "rows": rows}


class FrozenEmbeddings:
    def __init__(self, cache):
        self.identity = cache["embedding_id"]
        self.config = SimpleNamespace(model=cache.get("requested_model"))
        self.served_model = cache.get("served_model")
        self.calls = 0
        self.input_tokens = 0

    async def embed(self, _questions):
        raise ValueError("Incomplete frozen cache; offline replay cannot make provider calls")


async def run(fixture, *, backend=None, on_embeddings=None, embedding_cache=None):
    validate_fixture(fixture)
    if embedding_cache is not None:
        if backend is not None:
            raise ValueError("Choose live embeddings or frozen replay")
        backend = FrozenEmbeddings(embedding_cache)
    report = {"schema_version": 1, "fixture_sha256": digest(fixture), "source_revision": _source_revision(),
              "complete": False, "answer_quality_evaluated": False, "semantic_enabled": backend is not None,
              "rows": [], "semantic": None}
    index = QuestionSemanticIndex(backend) if backend is not None else None
    if index is not None:
        questions = [m["question"] for m in fixture["memories"]] + [q["question"] for q in fixture["queries"]]
        for case in read_json(RECALL_FIXTURE)["retrieval_cases"]:
            questions.extend([case["seed_question"], *case["variants"].values()])
        questions = list(dict.fromkeys(questions))
        if len(questions) > index.max_items or any(not q.strip() or len(q.encode()) > 6000 for q in questions):
            raise ValueError("Benchmark question corpus exceeds the bounded embedding cache")
        started = time.perf_counter()
        if embedding_cache is not None:
            if (embedding_cache["fixture_sha256"] != digest(fixture)
                    or embedding_cache["recall_fixture_sha256"] != digest(read_json(RECALL_FIXTURE))
                    or embedding_cache["question_set_sha256"] != digest(questions)
                    or set(embedding_cache["vectors"]) != set(questions)):
                raise ValueError("Frozen embedding inputs differ from the complete fixture")
            vectors = {q: normalized_vector(v) for q, v in embedding_cache["vectors"].items()}
            if len({len(v) for v in vectors.values()}) != 1:
                raise ValueError("Frozen embedding dimensions differ")
            index.vectors.update(vectors)
            index.dimension = len(next(iter(vectors.values())))
        while any(q not in index.vectors for q in questions):
            previous_count = len(index.vectors)
            await index.prepare(questions)
            if len(index.vectors) <= previous_count:
                raise ValueError("Embedding preparation did not advance")
            if on_embeddings:
                on_embeddings({"embedding_id": index.identity, "vectors": dict(index.vectors),
                               "dimension": index.dimension,
                               "requested_model": getattr(getattr(backend, "config", None), "model", None),
                               "served_model": getattr(backend, "served_model", None),
                               "fixture_sha256": digest(fixture), "recall_fixture_sha256": digest(read_json(RECALL_FIXTURE)),
                               "question_set_sha256": digest(questions)})
        report["semantic"] = {"embedding_id": index.identity, "requested_model": getattr(getattr(backend, "config", None), "model", None),
                              "served_model": getattr(backend, "served_model", None),
                              "frozen_replay": embedding_cache is not None,
                              "dimension": index.dimension,
                              "relative_cutoff": index.relative_cutoff, "strong_lexical_override": 0.75,
                              "preparation_latency_ms": (time.perf_counter() - started) * 1000,
                              "provider_calls": getattr(backend, "calls", None), "input_tokens": getattr(backend, "input_tokens", None)}
        report["runtime_default_semantic"] = {
            "min_similarity": index.min_similarity,
            "summaries": {split: summarize(retrieval_rows(fixture, index, method="hybrid", split=split))
                          for split in ("calibration", "evaluation")},
        }
        report["calibration"] = calibrate(fixture, index)
    for method in ("legacy_lexical", "question_lexical", "hybrid"):
        if method == "hybrid" and index is None:
            continue
        report["rows"].extend(retrieval_rows(fixture, index if method == "hybrid" else None, method=method))
    report["summaries"] = {method: {split: summarize([r for r in report["rows"] if r["method"] == method and r["split"] == split])
                                   for split in ("calibration", "evaluation")}
                           for method in {r["method"] for r in report["rows"]}}
    report["recall_ablation"] = recall_ablation(index)
    report["complete"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--semantic-model", help="Explicitly authorizes embedding API calls; omitted means offline")
    mode.add_argument("--embedding-cache", type=Path, help="Replay a complete saved vector cache without provider calls")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    backend = None
    if args.semantic_model:
        config = load_llm_config("MEMORY", default_model=args.semantic_model)
        if config is None:
            raise SystemExit("Configured provider credentials required for semantic mode")
        backend = OpenAIQuestionEmbeddings(config, args.semantic_model)

    def checkpoint(value):
        from benchmark_controlled_memory import checkpoint as save
        save(args.output_dir / "embeddings.partial.json", value)

    try:
        report = asyncio.run(run(read_json(args.fixture), backend=backend, on_embeddings=checkpoint,
                                 embedding_cache=read_json(args.embedding_cache) if args.embedding_cache else None))
    except Exception as exc:
        write_private(args.output_dir / "failure.json", {"complete": False, "error_type": type(exc).__name__,
                                                        "embedding_calls": getattr(backend, "calls", 0),
                                                        "input_tokens": getattr(backend, "input_tokens", None)})
        raise SystemExit("Benchmark failed: " + type(exc).__name__ + "; private checkpoint retained") from None
    write_private(args.output_dir / "report.json", report)
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
