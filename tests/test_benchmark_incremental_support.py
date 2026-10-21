import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))
import benchmark_incremental_support as bench


def test_changing_candidate_experiment_compares_resends_without_model_calls():
    report = asyncio.run(bench.run())
    assert report["provider_calls"] == 0 and not report["answer_quality_evaluated"]
    assert report["approvals_identical"]
    before, after = report["arms"]["whole_batch_reference"], report["arms"]["incremental"]
    assert before["checker_batches"] == after["checker_batches"] == 5
    assert before["records_sent"] == 22 and after["records_sent"] == 10
    assert after["serialized_policy_and_request_utf8_bytes"] < before["serialized_policy_and_request_utf8_bytes"]
    stages = {r["stage"]: r for r in after["rows"]}
    assert stages["growth"]["checked_candidates"] == 1
    assert stages["correction"]["checked_candidates"] == 1
    assert stages["status_change"]["checked_candidates"] == 1
    assert stages["different_question"]["checked_candidates"] == 3
    assert stages["reorder"]["checker_batches"] == stages["removal"]["checker_batches"] == 0
    assert stages["exact_repeat"]["checker_batches"] == 0
    assert stages["initial"]["approved_qids"] == ["deadline", "region"]
    assert stages["growth"]["approved_qids"] == ["extra", "deadline", "region"]
