import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"
sys.path.insert(0, str(SCRIPTS))
import benchmark_frozen_drafts as bench  # noqa: E402
from core.llm_client import LLMClient, LLMConfig  # noqa: E402


def inputs():
    return (bench.read_json(bench.DEFAULT_RUBRIC),
            bench.read_json(bench.PROJECT_ROOT / "examples/system2_frozen_drafts.synthetic.json"))


class FakeClient:
    calls = []
    mode = "ok"

    def __init__(self, config):
        self.config = config
        self.receipts = []

    async def complete(self, prompt, **kwargs):
        self.calls.append((self.config, json.loads(prompt), kwargs))
        if self.mode == "error":
            raise RuntimeError("DO_NOT_STORE credential or private provider error")
        self.receipts.append({"input_tokens": 23, "output_tokens": 11,
                              "finish_reason": "length" if self.mode == "truncated" else "stop"})
        if kwargs["system"] == bench.REVIEW_SYSTEM:
            if self.mode == "invalid_review":
                return '{"verdict":"pass","issues":["contradiction"]}'
            return '{"verdict":"issues","issues":["Fix a specific defect."]}'
        return "A revised fixture answer."


def live_report(monkeypatch, mode="ok", repetitions=1):
    monkeypatch.setattr(FakeClient, "calls", [])
    monkeypatch.setattr(FakeClient, "mode", mode)
    rubric, corpus = inputs()
    left = LLMConfig("openai", "left-fixture", "DO_NOT_STORE", auto_continue=True)
    right = LLMConfig("mistral", "right-fixture", "DO_NOT_STORE", max_output_tokens=9000)
    report = asyncio.run(bench.run_comparison(
        rubric, corpus, live=True, left=left, right=right, repetitions=repetitions,
        max_tokens=256, timeout=12, client_factory=FakeClient,
    ))
    return report, rubric


def fill_ratings(blind, key, ratings):
    ratings["reviewer"] = "independent-fixture-reviewer"
    assignments = {a["packet_id"]: a for a in key["assignments"]}
    for rating in ratings["ratings"]:
        labels = assignments[rating["packet_id"]]["labels"]
        for label, criteria in rating["scores"].items():
            for index, item in enumerate(criteria.values()):
                if item["verdict"] == "unavailable":
                    continue
                arm = labels[label]["arm"]
                # Baseline misses first criterion; self review repairs it but breaks the third.
                item.update(verdict="fail" if index == 0 and arm == "no_revision" else "pass",
                            evidence="Explicit fixture evidence for this criterion.")
                if arm == "self_review" and index == 2:
                    item["verdict"] = "fail"


def test_rubric_freezes_four_existing_questions_and_unique_criteria():
    rubric, corpus = inputs()
    questions = {q["id"]: q["question"] for q in bench.read_json(
        bench.PROJECT_ROOT / "examples/system2_hard_benchmark_questions_en.json")}
    assert len(bench.load_inputs(rubric, corpus)) == 4
    assert sum(len(case["criteria"]) for case in rubric["cases"]) == 20
    for case in rubric["cases"]:
        assert case["question"] == questions[case["id"]]
    corpus["drafts"].append(corpus["drafts"][0])
    with pytest.raises(ValueError, match="unique frozen draft"):
        bench.load_inputs(rubric, corpus)


def test_dry_run_never_calls_clients_or_masquerades_as_quality_evidence():
    rubric, corpus = inputs()
    def forbidden(_config):
        pytest.fail("A dry run must not create a client")
    report = asyncio.run(bench.run_comparison(rubric, corpus, client_factory=forbidden))
    assert len(report["rows"]) == 12
    assert report["evaluation"] == "unscored"
    assert {r["status"] for r in report["rows"]} == {"ok", "not_run"}
    with pytest.raises(ValueError, match="dry runs cannot be scored"):
        bench.prepare_packets(report, rubric)
    with pytest.raises(ValueError, match="requires configured"):
        asyncio.run(bench.run_comparison(rubric, corpus, live=True))


def test_identical_inputs_caps_usage_and_balanced_schedule(monkeypatch):
    report, rubric = live_report(monkeypatch, repetitions=6)
    assert len(FakeClient.calls) == 96
    assert "DO_NOT_STORE" not in json.dumps(report)
    assert report["config"]["roles"]["left"]["auto_continue"] is False
    for config, payload, kwargs in FakeClient.calls:
        assert config.max_output_tokens == kwargs["max_output_tokens"] == 256
        assert config.timeout_seconds == kwargs["timeout_seconds"] == 12
        assert config.auto_continue is False and config.max_continuations == 0
        assert kwargs["temperature"] == 0.2
        frozen = next(item for item in report["inputs"] if item["question"] == payload["question"])
        assert payload["draft"] == frozen["draft"] and payload["context"] == frozen["context"]
        if "review" in payload:
            assert config.model == "left-fixture"
    for row in report["rows"]:
        assert row["status"] == "ok"
        assert row["attempted_calls"] == row["call_budget"]
        assert row["input_tokens"] == (0 if row["arm"] == "no_revision" else 46)
        assert row["output_tokens"] == (0 if row["arm"] == "no_revision" else 22)
    for case in rubric["cases"]:
        orders = [r["order"] for r in report["rows"] if r["id"] == case["id"]
                  and r["arm"] == "no_revision"]
        assert len({tuple(order) for order in orders}) == 6
        for arm in bench.ARMS:
            assert [sum(o[pos] == arm for o in orders) for pos in range(3)] == [2, 2, 2]


@pytest.mark.parametrize("mode", ["invalid_review", "truncated", "error"])
def test_failures_are_retained_and_never_count_as_valid_revisions(monkeypatch, mode):
    report, rubric = live_report(monkeypatch, mode)
    assert len(FakeClient.calls) == 8  # Failed review prevents the second call.
    for row in report["rows"]:
        if row["arm"] != "no_revision":
            assert row["status"] == mode
            assert row["answer"] is None
            assert row["attempted_calls"] == 1 and row["call_budget"] == 2
            if mode == "error":
                assert row["input_tokens"] is None
    assert "DO_NOT_STORE" not in json.dumps(report)
    blind, key, ratings = bench.prepare_packets(report, rubric)
    fill_ratings(blind, key, ratings)
    result = bench.score_packets(blind, key, ratings)
    assert result["arms"]["self_review"]["unavailable"] == 20
    assert result["arms"]["self_review"]["failed_answers"] == 4
    assert result["arms"]["self_review"]["repair_rate"] is None


def test_capture_real_client_usage_and_missing_usage(monkeypatch):
    async def response(_self, *_args, **_kwargs):
        return {"choices": [{"finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    monkeypatch.setattr(LLMClient, "_post_json", response)
    client = bench.MeasuredClient(LLMConfig("openai", "fixture", "secret"))
    asyncio.run(client._post_json("url", {}, {}))
    assert client.receipts == [{"input_tokens": 7, "output_tokens": 3, "finish_reason": "stop"}]
    assert bench.sum_usage([{"input_tokens": 0}, {"input_tokens": None}], "input_tokens") is None


def test_public_config_records_reasoning_model_request_options_without_secrets():
    config = LLMConfig("openai", "gpt-6-luna", "SECRET", max_output_tokens=256)
    public = bench.public_config(config)
    assert public["request_options"] == {"model": "gpt-6-luna", "max_completion_tokens": 256}
    assert "SECRET" not in json.dumps(public)
    config.model = "ordinary-fixture"
    assert bench.public_config(config)["request_options"]["temperature"] == 0.2


def test_blind_scoring_records_repairs_regressions_and_uncertainty(monkeypatch):
    report, rubric = live_report(monkeypatch)
    blind, key, ratings = bench.prepare_packets(report, rubric)
    assert not any(arm in json.dumps(blind) for arm in bench.ARMS)
    fill_ratings(blind, key, ratings)
    result = bench.score_packets(blind, key, ratings)
    assert result["unique_cases"] == result["packets"] == 4
    assert result["arms"]["self_review"]["repaired_defects"] == 4
    assert result["arms"]["self_review"]["introduced_defects"] == 4
    assert result["arms"]["self_review"]["critical_introduced_defects"] == 3
    assert result["arms"]["self_review"]["repair_rate"] == 1.0
    assert result["arms"]["self_review"]["regression_rate"] == 0.25
    assert result["arms"]["right_critic"]["introduced_defects"] == 0
    assignment = key["assignments"][0]
    label = next(l for l, info in assignment["labels"].items() if info["arm"] == "self_review")
    first_criterion = next(iter(ratings["ratings"][0]["scores"][label]))
    ratings["ratings"][0]["scores"][label][first_criterion].update(
        verdict="uncertain", evidence="Ambiguous fixture evidence.")
    result = bench.score_packets(blind, key, ratings)
    assert result["arms"]["self_review"]["uncertain"] == 1
    assert result["arms"]["self_review"]["eligible_repairs"] == 3
    assert result["arms"]["self_review"]["criterion_units"] == 20


def test_scoring_rejects_unreviewed_missing_duplicate_and_changed_data(monkeypatch):
    report, rubric = live_report(monkeypatch)
    blind, key, ratings = bench.prepare_packets(report, rubric)
    with pytest.raises(ValueError, match="reviewer identity"):
        bench.score_packets(blind, key, ratings)
    ratings["reviewer"] = "reviewer"
    with pytest.raises(ValueError, match="verdict and evidence"):
        bench.score_packets(blind, key, ratings)
    fill_ratings(blind, key, ratings)
    bad = deepcopy(ratings)
    bad["ratings"].append(bad["ratings"][0])
    with pytest.raises(ValueError, match="exactly once"):
        bench.score_packets(blind, key, bad)
    bad = deepcopy(ratings)
    bad["ratings"][0]["scores"]["A"].popitem()
    with pytest.raises(ValueError, match="criterion"):
        bench.score_packets(blind, key, bad)
    blind["packets"][0]["question"] += "Changed"
    with pytest.raises(ValueError, match="digest mismatch"):
        bench.score_packets(blind, key, ratings)


def test_prepare_rejects_changed_drafts_missing_arms_and_rubric(monkeypatch):
    report, rubric = live_report(monkeypatch)
    bad = deepcopy(report)
    bad["inputs"][0]["draft"] += "Changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        bench.prepare_packets(bad, rubric)
    bad = deepcopy(report)
    bad["rows"].pop()
    with pytest.raises(ValueError, match="all three arms"):
        bench.prepare_packets(bad, rubric)
    rubric["cases"][0]["criteria"][0]["requirement"] += "Changed"
    with pytest.raises(ValueError, match="unchanged rubric"):
        bench.prepare_packets(report, rubric)


def test_private_outputs_cannot_be_overwritten(tmp_path):
    output = tmp_path / "report.json"
    bench.write_private(output, {"fixture": True})
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        bench.write_private(output, {})


def test_cli_run_prepare_score_round_trip_with_mock_provider(tmp_path, monkeypatch):
    rubric, corpus = inputs()
    corpus_path = tmp_path / "drafts.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    monkeypatch.setattr(bench, "load_llm_config", lambda _scope: LLMConfig("openai", "fixture", "SECRET"))

    async def complete(self, _prompt, **kwargs):
        self.receipts.append({"input_tokens": 5, "output_tokens": 4, "finish_reason": "stop"})
        if kwargs["system"] == bench.REVIEW_SYSTEM:
            return '{"verdict":"pass","issues":[]}'
        return "Fixture revision"

    monkeypatch.setattr(bench.MeasuredClient, "complete", complete)
    run_dir, review_dir = tmp_path / "run", tmp_path / "review"
    monkeypatch.setattr(sys, "argv", ["benchmark", "run", "--drafts", str(corpus_path),
                                      "--live", "--output-dir", str(run_dir)])
    bench.main()
    report = bench.read_json(run_dir / "report.json")
    assert "SECRET" not in json.dumps(report)
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--report", str(run_dir / "report.json"),
                                      "--output-dir", str(review_dir)])
    bench.main()
    blind = bench.read_json(review_dir / "blind.json")
    key = bench.read_json(review_dir / "reveal_key.json")
    ratings = bench.read_json(review_dir / "ratings.json")
    fill_ratings(blind, key, ratings)
    (review_dir / "ratings.json").write_text(json.dumps(ratings), encoding="utf-8")
    score_path = tmp_path / "scores.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "score", "--blind", str(review_dir / "blind.json"),
                                      "--key", str(review_dir / "reveal_key.json"), "--ratings",
                                      str(review_dir / "ratings.json"), "--output", str(score_path)])
    bench.main()
    assert bench.read_json(score_path)["arms"]["self_review"]["repaired_defects"] == 4
