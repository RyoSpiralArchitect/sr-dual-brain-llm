#!/usr/bin/env python3
"""Model-free request-volume experiment for growing/changing memory candidates."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.memory_answerability import AnswerabilityGate, POLICY, digest
from benchmark_support_reference import WholeBatchReference
from benchmark_system2_ab import _source_revision


def fixture():
    long_policy = ("通常の保管規約と地域別の条件を確認し、例外も含めて適用します。" * 30
                   + "ただし欧州では保管期間は7日です。")

    def source(qid, answer):
        return {"qid": qid, "seed_id": "seed-" + qid, "question": "Project Alpha policy memo",
                "answer": answer, "status": "unverified", "truncated": False,
                "recorded_at": 1000., "valid_until": None, "retrieval_scope": "alpha"}

    deadline = source("deadline", "The deadline is fourteen days.")
    fee = source("fee", "Fees are outside the supplied policy.")
    region = source("region", long_policy)
    other = source("other", "The conference room is blue.")
    extra = source("extra", "The archive is read-only after the deadline.")
    revised = dict(deadline, answer="The deadline is seven days.", recorded_at=1001.)
    verified = dict(revised, status="verified")
    question = "What are Alpha's deadline and European retention conditions?"
    second = "Repeat Alpha's deadline and European retention conditions."
    stages = [
        ("initial", question, [deadline, fee, region, other]),
        ("growth", question, [extra, deadline, fee, region, other]),
        ("reorder", question, [other, region, fee, deadline, extra]),
        ("correction", question, [revised, extra, fee, region, other]),
        ("status_change", question, [verified, extra, fee, region, other]),
        ("removal", question, [verified, region, fee]),
        ("different_question", second, [verified, region, fee]),
        ("exact_repeat", second, [verified, region, fee]),
    ]
    # Authored decisions simulate independent checks, not language understanding.
    supported = {s["answer"]: s["qid"] in {"deadline", "region", "extra"}
                 for _, _, sources in stages for s in sources}
    return stages, supported


class SimulatedChecks:
    identity = "authored-incremental-support-v1"

    def __init__(self, supported):
        self.supported = supported
        self.requests = []

    async def evaluate(self, request):
        self.requests.append(deepcopy(request))
        return json.dumps({"decisions": [
            {"id": r["id"], "supported": self.supported[r["answer"]],
             "quote": r["answer"] if self.supported[r["answer"]] else ""}
            for r in request["records"]]}, ensure_ascii=False)


async def run():
    stages, supported = fixture()
    arms = {}
    for name, gate_type in (("whole_batch_reference", WholeBatchReference),
                            ("incremental", AnswerabilityGate)):
        backend = SimulatedChecks(supported)
        gate = gate_type(backend, max_question_chars=2048, max_answer_chars=2048)
        rows = []
        for stage, question, sources in stages:
            before = len(backend.requests)
            result = await gate.prepare(question, sources)
            if result["status"] not in {"ready", "cached"}:
                raise RuntimeError("Simulated support preparation failed")
            rows.append({"stage": stage, **result,
                         "checker_batches": len(backend.requests) - before,
                         "approved_qids": [s["qid"] for s in sources if gate.allows(question, s)]})
        arms[name] = {
            "checker_batches": len(backend.requests),
            "records_sent": sum(len(r["records"]) for r in backend.requests),
            "serialized_policy_and_request_utf8_bytes": sum(
                len(json.dumps({"policy": POLICY, "request": r}, ensure_ascii=False,
                               separators=(",", ":")).encode()) for r in backend.requests),
            "rows": rows,
        }
    if ([r["approved_qids"] for r in arms["whole_batch_reference"]["rows"]]
            != [r["approved_qids"] for r in arms["incremental"]["rows"]]):
        raise RuntimeError("Reference and incremental approvals differ")
    return {"schema_version": 1, "experiment": "model_free_incremental_support",
            "provider_calls": 0, "answer_quality_evaluated": False,
            "source_revision": _source_revision(),
            "fixture_sha256": digest(stages), "policy_sha256": digest(POLICY),
            "approvals_identical": True, "arms": arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = asyncio.run(run())
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in arm.items() if k != "rows"}
                      for name, arm in report["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
