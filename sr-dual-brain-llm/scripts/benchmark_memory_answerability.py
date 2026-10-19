#!/usr/bin/env python3
"""Frozen question/evidence support experiment; development and evaluation run separately."""
from __future__ import annotations

import argparse
import asyncio
import builtins
import json
from pathlib import Path
import time

import benchmark_scoped_memory as scoped
from benchmark_controlled_memory import checkpoint
from benchmark_frozen_drafts import digest, read_json, write_private
from benchmark_memory_retrieval import FrozenEmbeddings, retrieval_rows, summarize, validate_fixture
from benchmark_system2_ab import _source_revision
from core.llm_client import load_llm_config
from core.memory_answerability import AnswerabilityGate, OpenAIAnswerability, POLICY, request_identity
from core.memory_retrieval import OpenAIQuestionEmbeddings, QuestionSemanticIndex, normalized_vector
from core.memory_scope import MemoryScopeResolver
from core.shared_memory import MemoryTrace, SharedMemory

DEFAULT_FIXTURE = scoped.PROJECT / "examples/memory_answerability_benchmark.json"


class RecordedChecks:
    def __init__(self, backend, metadata, *, frozen=None, save=None):
        self.backend, self.frozen, self.save = backend, frozen, save
        self.identity = frozen["backend_identity"] if frozen else backend.identity
        self.calls = self.input_tokens = self.output_tokens = 0
        self.used, self.missing = set(), set()
        self.journal = dict(metadata, backend_identity=self.identity, policy_sha256=digest(POLICY),
                            complete=False, responses={})
        if frozen and (not frozen.get("complete") or frozen["policy_sha256"] != digest(POLICY)
                       or any(frozen.get(k) != v for k, v in metadata.items())):
            raise ValueError("Frozen support checkpoint differs or is incomplete")

    async def evaluate(self, request):
        key = request_identity(self.identity, request)
        self.used.add(key)
        if self.frozen is not None:
            if key not in self.frozen["responses"]:
                self.missing.add(key)
                raise ValueError("Incomplete frozen support coverage")
            row = self.frozen["responses"][key]
            if row["request"] != request:
                self.missing.add(key)
                raise ValueError("Frozen support input mismatch")
            if row.get("error_type"):
                kind = getattr(builtins, row["error_type"], RuntimeError)
                raise (kind if isinstance(kind, type) and issubclass(kind, Exception) else RuntimeError)("Frozen provider failure")
            return row["response"]
        if self.calls >= 150:
            raise ValueError("Support benchmark call limit exceeded")
        self.calls += 1
        started = time.perf_counter()
        before_in, before_out = self.backend.input_tokens, self.backend.output_tokens
        row = {"request": request}
        try:
            row["response"] = await self.backend.evaluate(request)
            return row["response"]
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            raise
        finally:
            self.input_tokens, self.output_tokens = self.backend.input_tokens, self.backend.output_tokens
            row.update(latency_ms=(time.perf_counter() - started) * 1000,
                       input_tokens=self.input_tokens - before_in if type(self.input_tokens) is int and type(before_in) is int else None,
                       output_tokens=self.output_tokens - before_out if type(self.output_tokens) is int and type(before_out) is int else None)
            self.journal["responses"][key] = row
            if self.save:
                self.save(self.journal)

    def finish(self):
        if self.frozen is not None:
            if self.missing or self.used != set(self.frozen["responses"]):
                raise ValueError("Frozen support coverage differs from the run")
        else:
            self.journal.update(complete=True, served_model=self.backend.served_model,
                                provider_calls=self.calls, input_tokens=self.input_tokens, output_tokens=self.output_tokens)
            if self.save:
                self.save(self.journal)


async def prepare_vectors(fixture, *, backend=None, frozen=None, save=None):
    index = QuestionSemanticIndex(backend or FrozenEmbeddings(frozen), scope_min_similarity=.25)
    index.scope_resolver = MemoryScopeResolver([MemoryTrace(**m) for m in fixture["memories"]])
    questions = scoped.embedding_inputs(fixture, index)
    if len(questions) > index.max_items or any(len(q.encode()) > 6000 or not q.strip() for q in questions):
        raise ValueError("Question view limits exceeded")
    if frozen is not None:
        if (frozen["fixture_sha256"] != digest(fixture) or frozen["question_set_sha256"] != digest(questions)
                or set(frozen["vectors"]) != set(questions)):
            raise ValueError("Frozen embedding inputs differ")
        index.vectors.update({q: normalized_vector(v) for q, v in frozen["vectors"].items()})
        if len({len(v) for v in index.vectors.values()}) != 1:
            raise ValueError("Frozen embedding dimensions differ")
        index.dimension = len(next(iter(index.vectors.values())))
    while any(q not in index.vectors for q in questions):
        prepared = await index.prepare(questions)
        if not prepared["embedded_questions"]:
            raise ValueError("Embedding preparation did not advance")
        if save:
            save({"fixture_sha256": digest(fixture), "question_set_sha256": digest(questions),
                  "embedding_id": index.identity, "dimension": index.dimension, "vectors": dict(index.vectors),
                  "requested_model": backend.config.model, "served_model": backend.served_model})
    return index


def selected_row(query, selected, method):
    gold, ids = set(query["relevant_qids"]), [t.qid for t in selected]
    overlap = len(gold & set(ids))
    return {"id": query["id"], "split": query["split"], "scene": query["scene"], "kind": query["kind"],
            "method": method, "selected_qids": ids, "relevant_qids": sorted(gold),
            "hit_at_1": bool(ids and ids[0] in gold), "recall_at_3": overlap / len(gold) if gold else None,
            "abstained": not ids, "unrelated_slots": len(set(ids) - gold),
            "set_f1": 2 * overlap / (len(gold) + len(ids)) if gold or ids else 1}


async def run(fixture, index, checks, *, phase, candidate_mode="scope", save=None):
    validate_fixture(fixture)
    split = "calibration" if phase == "development" else "evaluation"
    memory = SharedMemory(semantic_index=index, answerability_gate=AnswerabilityGate(checks),
                          answerability_scope_expansion=candidate_mode == "scope")
    for item in fixture["memories"]:
        memory.store(MemoryTrace(**item))
    rows = retrieval_rows(fixture, index, method="scoped_025", split=split)
    report = {"schema_version": 1, "source_revision": _source_revision(), "fixture_sha256": digest(fixture),
              "policy_sha256": digest(POLICY), "phase": phase, "candidate_mode": candidate_mode,
              "complete": False, "rows": rows,
              "answer_quality_evaluated": False, "checks": [], "summaries": {}}
    for query in fixture["queries"]:
        if query["split"] != split:
            continue
        candidates = memory.answerability_candidates(query["question"])
        started = time.perf_counter()
        result = await memory.prepare_retrieval(query["question"])
        selected = memory.retrieve_related_traces(query["question"])
        row = selected_row(query, selected, "answerability")
        report["rows"].append(row)
        gold = set(query["relevant_qids"])
        report["checks"].append({"id": query["id"], "candidate_qids": [t.qid for t in candidates],
                                 "full_gold_in_candidates": gold <= {t.qid for t in candidates} if gold else None,
                                 "latency_ms": (time.perf_counter() - started) * 1000,
                                 **result["answerability"]})
        if save:
            save(report)
    checks.finish()
    report["summaries"] = {method: {"all": summarize([r for r in report["rows"] if r["method"] == method]),
                                    "by_kind": {kind: summarize([r for r in report["rows"] if r["method"] == method and r["kind"] == kind])
                                                for kind in sorted({r["kind"] for r in report["rows"]})}}
                           for method in ("scoped_025", "answerability")}
    report.update(complete=True, provider_calls=checks.calls, input_tokens=checks.input_tokens,
                  output_tokens=checks.output_tokens, frozen_replay=checks.frozen is not None)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--phase", choices=("development", "evaluation"), required=True)
    p.add_argument("--candidate-mode", choices=("shortlist", "scope"), default="scope")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--support-model", help="Explicitly authorizes bounded chat checks")
    mode.add_argument("--support-cache", type=Path, help="Complete frozen responses; zero chat calls")
    embed = p.add_mutually_exclusive_group(required=True)
    embed.add_argument("--embedding-model", help="Explicitly authorizes embedding calls")
    embed.add_argument("--embedding-cache", type=Path)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        fixture = read_json(args.fixture)
        validate_fixture(fixture)
        config = load_llm_config("MEMORY_ANSWERABILITY", default_model=args.support_model)
        backend = OpenAIAnswerability(config, args.support_model) if args.support_model and config else None
        if args.support_model and backend is None:
            raise ValueError("Provider credentials required")
        embedding = OpenAIQuestionEmbeddings(config, args.embedding_model) if args.embedding_model else None
        index = asyncio.run(prepare_vectors(fixture, backend=embedding,
                    frozen=read_json(args.embedding_cache) if args.embedding_cache else None,
                    save=lambda d: checkpoint(args.output_dir / "embeddings.partial.json", d)))
        checks = RecordedChecks(backend, {"fixture_sha256": digest(fixture), "phase": args.phase},
                    frozen=read_json(args.support_cache) if args.support_cache else None,
                    save=lambda d: checkpoint(args.output_dir / "support.partial.json", d))
        report = asyncio.run(run(fixture, index, checks, phase=args.phase, candidate_mode=args.candidate_mode,
                                 save=lambda d: checkpoint(args.output_dir / "report.partial.json", d)))
        report["embedding_usage"] = {"provider_calls": getattr(embedding, "calls", 0),
                                      "input_tokens": getattr(embedding, "input_tokens", 0),
                                      "embedding_id": index.identity, "dimension": index.dimension}
        write_private(args.output_dir / "report.json", report)
        print(json.dumps({"summaries": report["summaries"], "calls": checks.calls,
                          "input_tokens": checks.input_tokens, "output_tokens": checks.output_tokens}, indent=2))
    except Exception as exc:
        write_private(args.output_dir / "failure.json", {"complete": False, "error_type": type(exc).__name__})
        raise SystemExit("Support benchmark failed: " + type(exc).__name__ + "; private checkpoints retained") from None


if __name__ == "__main__":
    main()
