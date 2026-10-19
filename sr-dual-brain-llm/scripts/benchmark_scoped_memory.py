#!/usr/bin/env python3
"""Frozen scope/content retrieval comparison; no answer-quality grading."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from core.memory_retrieval import OpenAIQuestionEmbeddings, QuestionSemanticIndex, normalized_vector  # noqa: E402
from core.llm_client import load_llm_config  # noqa: E402
from core.memory_scope import MemoryScopeResolver  # noqa: E402
from core.shared_memory import MemoryTrace  # noqa: E402
from benchmark_memory_retrieval import FrozenEmbeddings, retrieval_rows, summarize, validate_fixture  # noqa: E402
from benchmark_frozen_drafts import digest, read_json, write_private  # noqa: E402
from benchmark_controlled_memory import checkpoint  # noqa: E402
from benchmark_system2_ab import _source_revision  # noqa: E402

DEFAULT_FIXTURE = PROJECT / "examples/scoped_memory_retrieval_benchmark.json"
THRESHOLDS = tuple(round(.20 + i * .05, 2) for i in range(13))


def without_scopes(fixture):
    """Ablation of the new annotations, using the previous selector unchanged."""
    result = deepcopy(fixture)
    for memory in result["memories"]:
        memory.pop("retrieval_scope", None)
        memory.pop("scope_aliases", None)
    return result


def embedding_inputs(fixture, scoped_index):
    raw = list(dict.fromkeys([m["question"] for m in fixture["memories"]]
                            + [q["question"] for q in fixture["queries"]]))
    return list(dict.fromkeys(raw + scoped_index.embedding_questions(raw)))


def calibrate_selector(fixture, index, *, scoped):
    candidates = []
    for threshold in THRESHOLDS:
        if scoped:
            index.scope_min_similarity = threshold
        else:
            index.min_similarity = threshold
        candidates.append({"threshold": threshold,
                           **summarize(retrieval_rows(fixture, index, split="calibration"))})
    chosen = max(candidates, key=lambda item: (item["mean_set_f1"], item["threshold"]))
    if scoped:
        index.scope_min_similarity = chosen["threshold"]
    else:
        index.min_similarity = chosen["threshold"]
    return {"chosen_threshold": chosen["threshold"], "candidates": candidates,
            "objective": "calibration macro source-set F1; stricter threshold on ties",
            "evaluation_used_for_selection": False}


def summaries(rows):
    return {split: {"all": summarize([r for r in rows if r["split"] == split]),
                    "by_kind": {kind: summarize([r for r in rows if r["split"] == split and r["kind"] == kind])
                                for kind in sorted({r["kind"] for r in rows if r["split"] == split})}}
            for split in ("calibration", "evaluation")}


async def run(fixture, *, backend=None, embedding_cache=None, on_embeddings=None):
    validate_fixture(fixture)
    if embedding_cache is not None:
        if backend is not None:
            raise ValueError("Choose live embeddings or frozen replay")
        backend = FrozenEmbeddings(embedding_cache)
    baseline = without_scopes(fixture)
    report = {"schema_version": 1, "source_revision": _source_revision(), "fixture_sha256": digest(fixture),
              "complete": False, "answer_quality_evaluated": False, "semantic": None,
              "operating_points": {"raw_default": .70, "content_default": .50,
                                   "relative_cutoff": .8, "strong_lexical_override": .75},
              "calibration": {}, "rows": [], "summaries": {}}
    raw = QuestionSemanticIndex(backend) if backend is not None else None
    scoped = QuestionSemanticIndex(backend) if backend is not None else None
    if scoped is not None:
        scoped.scope_resolver = MemoryScopeResolver([MemoryTrace(**m) for m in fixture["memories"]])
        questions = embedding_inputs(fixture, scoped)
        if (len(questions) > raw.max_items
                or any(not q.strip() or len(q.encode()) > 6000 for q in questions)):
            raise ValueError("Benchmark question views exceed the bounded embedding cache")
        started = time.perf_counter()
        if embedding_cache is not None:
            if (embedding_cache["fixture_sha256"] != digest(fixture)
                    or embedding_cache["question_set_sha256"] != digest(questions)
                    or set(embedding_cache["vectors"]) != set(questions)):
                raise ValueError("Frozen inputs differ from the complete fixture and question views")
            vectors = {q: normalized_vector(v) for q, v in embedding_cache["vectors"].items()}
            if len({len(v) for v in vectors.values()}) != 1:
                raise ValueError("Frozen embedding dimensions differ")
            raw.vectors.update(vectors)
            raw.dimension = len(next(iter(vectors.values())))
        while any(q not in raw.vectors for q in questions):
            before = len(raw.vectors)
            await raw.prepare(questions)
            if len(raw.vectors) <= before:
                raise ValueError("Embedding preparation did not advance")
            if on_embeddings:
                on_embeddings({"fixture_sha256": digest(fixture), "question_set_sha256": digest(questions),
                               "embedding_id": raw.identity, "vectors": dict(raw.vectors), "dimension": raw.dimension,
                               "requested_model": getattr(getattr(backend, "config", None), "model", None),
                               "served_model": getattr(backend, "served_model", None)})
        scoped.vectors = raw.vectors.copy()
        scoped.dimension = raw.dimension
        report["semantic"] = {"embedding_id": raw.identity, "dimension": raw.dimension,
                              "requested_model": getattr(getattr(backend, "config", None), "model", None),
                              "served_model": getattr(backend, "served_model", None),
                              "provider_calls": getattr(backend, "calls", None),
                              "input_tokens": getattr(backend, "input_tokens", None),
                              "frozen_replay": embedding_cache is not None,
                              "question_views": len(questions),
                              "preparation_latency_ms": (time.perf_counter() - started) * 1000,
                              "note": "Includes raw baseline and content views; not a production per-turn cost"}
    selectors = [("previous_lexical", baseline, None), ("scoped_lexical", fixture, None)]
    if raw is not None:
        selectors += [("previous_hybrid_default", baseline, raw), ("scoped_hybrid_default", fixture, scoped)]
    for name, data, index in selectors:
        report["rows"].extend(retrieval_rows(data, index, method=name))
    if raw is not None:
        for name, data, index, is_scoped in (("previous_hybrid_calibrated", baseline, raw, False),
                                             ("scoped_hybrid_calibrated", fixture, scoped, True)):
            report["calibration"][name] = calibrate_selector(data, index, scoped=is_scoped)
            report["rows"].extend(retrieval_rows(data, index, method=name))
    report["summaries"] = {name: summaries([r for r in report["rows"] if r["method"] == name])
                           for name in sorted({r["method"] for r in report["rows"]})}
    report["complete"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--semantic-model", help="Explicitly authorizes embedding API calls")
    mode.add_argument("--embedding-cache", type=Path, help="Complete frozen vector checkpoint; zero provider calls")
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
    print(json.dumps({name: value["evaluation"]["all"] for name, value in report["summaries"].items()}, indent=2))


if __name__ == "__main__":
    main()
