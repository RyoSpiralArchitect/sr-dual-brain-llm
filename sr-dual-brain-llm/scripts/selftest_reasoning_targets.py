#!/usr/bin/env python3
"""Lightweight self-tests for reasoning-target utilities.

Run:
  PYTHONPATH=sr-dual-brain-llm python3 -S sr-dual-brain-llm/scripts/selftest_reasoning_targets.py

If you run Python with `-S`, you may need to add your `site-packages` to `PYTHONPATH`
so imports like `numpy` (pulled in by some core modules) are available.
"""

from __future__ import annotations

from core.dual_brain import DualBrainController
from core.micro_critic import micro_criticise_reasoning


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_infer_question_type() -> None:
    code_review_q = "Review this Python snippet for correctness and edge cases: `def avg(nums): return sum(nums)/len(nums)`."
    causal_q = "Website latency and error rate both increased after deployment. Give a step-by-step causal triage plan to avoid confusing correlation with causation."
    logic_q = "If all A are B, and some B are C, does it follow that some A are C? Give a precise logical explanation."

    _assert(DualBrainController._infer_question_type(code_review_q) in {"medium", "hard"}, "code_review should be medium+")
    _assert(DualBrainController._infer_question_type(causal_q) in {"medium", "hard"}, "causal triage should be medium+")
    _assert(DualBrainController._infer_question_type(logic_q) in {"medium", "hard"}, "logic should be medium+")


def test_micro_code_review_avg() -> None:
    q = "Review this Python snippet for correctness and edge cases: `def avg(nums): return sum(nums)/len(nums)`."

    # Missing both empty + iterable contract.
    draft = "It computes the average of numbers."
    res = micro_criticise_reasoning(q, draft)
    _assert(res is not None and res.domain == "code_review", "expected code_review micro critic")
    _assert(res.verdict == "issues", "expected issues for weak draft")
    _assert(any("ZeroDivisionError" in issue for issue in res.issues), "should flag empty division")
    _assert(any("iterators" in issue or "generators" in issue for issue in res.issues), "should flag iterator/generator contract")

    # Mentions empty list but not iterables/generators.
    draft2 = "Edge case: empty list causes ZeroDivisionError; handle len(nums)==0."
    res2 = micro_criticise_reasoning(q, draft2)
    _assert(res2 is not None and res2.domain == "code_review", "expected code_review micro critic")
    _assert(res2.verdict == "issues", "should still flag iterable contract when missing")

    # Mentions both empty list and iterator/generator contract.
    draft3 = "Handle empty input to avoid ZeroDivisionError, and clarify that nums must be a sized sequence (generators need list(nums) first)."
    res3 = micro_criticise_reasoning(q, draft3)
    _assert(res3 is not None and res3.domain == "code_review", "expected code_review micro critic")
    _assert(res3.verdict == "ok", "expected ok when key edge cases are covered")


def test_micro_causal_triage() -> None:
    q = "Website latency and error rate both increased after deployment. Give a step-by-step causal triage plan to avoid confusing correlation with causation."

    weak = "Check the deploy and see what changed."
    res = micro_criticise_reasoning(q, weak)
    _assert(res is not None and res.domain == "causal_triage", "expected causal_triage micro critic")
    _assert(res.verdict == "issues", "expected issues for weak triage plan")

    ok = (
        "1) Compare before/after baseline metrics (p50/p99 latency + error signatures).\n"
        "2) Isolate the change via rollback/canary/bisect (if available) and test one factor at a time.\n"
        "3) Check confounders: traffic mix/load, upstream dependencies, and network/infrastructure incidents.\n"
    )
    res2 = micro_criticise_reasoning(q, ok)
    _assert(res2 is not None and res2.domain == "causal_triage", "expected causal_triage micro critic")
    _assert(res2.verdict == "ok", "expected ok when core triage elements are present")


def main() -> None:
    test_infer_question_type()
    test_micro_code_review_avg()
    test_micro_causal_triage()
    print("ok")


if __name__ == "__main__":
    main()
