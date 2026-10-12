import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"
sys.path.insert(0, str(SCRIPTS))
import benchmark_controlled_memory as bench  # noqa: E402
from core.llm_client import LLMConfig  # noqa: E402
from core.unconscious_field import PipelineConfig  # noqa: E402
from benchmark_unconscious_incubation import _match_emergent_to_seed  # noqa: E402
from run_offline_checks import offline_environment  # noqa: E402


class FakeClient:
    calls = []
    reason = "stop"

    def __init__(self, config):
        self.config = config
        self.receipts = []

    async def complete(self, prompt, **kwargs):
        self.calls.append((self.config, json.loads(prompt), kwargs))
        self.receipts.append({"finish_reason": self.reason, "input_tokens": 20, "output_tokens": 10})
        return "Fixture answer with available facts and explicit uncertainty."


def live_report(monkeypatch, reason="stop", repetitions=1):
    monkeypatch.setattr(FakeClient, "calls", [])
    monkeypatch.setattr(FakeClient, "reason", reason)
    fixture = bench.read_json(bench.DEFAULT_FIXTURE)
    report = asyncio.run(bench.run_benchmark(
        fixture, live=True, config=LLMConfig("openai", "fixture", "NOT_STORE"),
        repetitions=repetitions, max_tokens=256, timeout=12, client_factory=FakeClient))
    return report, fixture


def fill_ratings(key, ratings):
    ratings["reviewer"] = "independent-fixture-reviewer"
    by_id = {a["packet_id"]: a for a in key["assignments"]}
    for rating in ratings["ratings"]:
        labels = by_id[rating["packet_id"]]["labels"]
        for label, criteria in rating["scores"].items():
            for criterion, item in criteria.items():
                if item["verdict"] == "unavailable":
                    continue
                arm = labels[label]["arm"]
                passed = arm == "correct" or (criterion == "grounding" and arm != "stale")
                item.update(verdict="pass" if passed else "fail", evidence="Quoted fixture evidence.")


def test_fork_preserves_exact_memory_state_and_isolates_pipeline_sampler_and_seeds():
    case = bench.read_json(bench.DEFAULT_FIXTURE)["retrieval_cases"][0]
    field, outcome = bench.make_seed(case, PipelineConfig(seed=17))
    field.analyse(question="Common exposure", draft="")
    original = field.memory_state()
    branch = field.fork()
    assert branch.memory_state() == original
    assert branch._seed_cache[0].seed_id == outcome["seed_id"]
    mapping = branch.analyse(question=case["seed_question"], draft=case["seed_draft"])
    ideas = branch.summary(mapping).emergent_ideas
    assert len(ideas) == 1 and ideas[0].seed_id == outcome["seed_id"]
    assert ideas[0].source_qid == "seed:" + case["id"]
    assert branch._last_harvest_attempts[0].seed_id == outcome["seed_id"]
    assert field.memory_state() == original  # Removal and sampler updates do not leak.
    branch.registry.prototypes["self"].keywords.append("only-in-branch")
    assert "only-in-branch" not in field.registry.prototypes["self"].keywords


def test_same_origin_does_not_substitute_for_a_mismatched_seed_id():
    seeds = [{"seed_id": "first", "turn_index": 1, "archetype": "hero", "origin": "same"},
             {"seed_id": "second", "turn_index": 2, "archetype": "hero", "origin": "same"}]
    match = _match_emergent_to_seed({"seed_id": "first", "turn_index": 3, "origin": "same", "archetype": "hero"}, seeds)
    assert match["match_type"] == "seed_id" and match["seed"] == seeds[0]
    match = _match_emergent_to_seed({"seed_id": "missing", "turn_index": 3, "origin": "same", "archetype": "hero"}, seeds)
    assert match["match_type"] == "unmatched_seed_id" and match["seed"] is None


def test_revisit_branches_share_state_draft_and_incubation_without_cross_branch_history():
    fixture = bench.read_json(bench.DEFAULT_FIXTURE)
    report = bench.retrieval_diagnostics(fixture, repetitions=2)
    assert len(report["rows"]) == 18
    assert report["quality_evaluated"] is False and report["scope"] == "unconscious_field_only"
    for case in fixture["retrieval_cases"]:
        rows = [row for row in report["rows"] if row["id"] == case["id"]]
        assert len({row["initial_memory_sha256"] for row in rows}) == 1
        assert len({row["probe_draft"] for row in rows}) == 1
        assert len({row["seed_id"] for row in rows}) == 1
        assert all(row["harvest_attempts"][0]["incubation_rounds"] == 2 for row in rows)
        assert all(row["recovered_seed_ids"] == [row["seed_id"]] for row in rows if row["variant"] == "echo")


def test_live_inputs_budgets_four_orders_and_trace_provenance(monkeypatch):
    report, fixture = live_report(monkeypatch, repetitions=4)
    assert len(FakeClient.calls) == len(report["rows"]) == 64
    assert "NOT_STORE" not in json.dumps(report)
    assert report["quality_evaluated"] is False
    for config, payload, options in FakeClient.calls:
        assert config.max_output_tokens == options["max_output_tokens"] == 256
        assert config.timeout_seconds == options["timeout_seconds"] == 12
        assert config.auto_continue is False and config.max_continuations == 0
        assert options["system"] == bench.SYSTEM
        assert set(payload) == {"question", "historical_context"}
    for case in fixture["quality_cases"]:
        rows = [row for row in report["rows"] if row["id"] == case["id"]]
        assert {row["input"]["question"] for row in rows} == {case["question"]}
        assert report["memory_snapshots"][case["id"]]["base_memory"] == []
        for arm in bench.ARMS:
            arm_rows = [row for row in rows if row["arm"] == arm]
            assert len({row["memory_sha256"] for row in arm_rows}) == 1
            assert sorted(row["order"].index(arm) for row in arm_rows) == [0, 1, 2, 3]
            if arm == "no_memory":
                assert all(not row["input"]["historical_context"] and not row["injected_seed_ids"] for row in arm_rows)
            else:
                assert all(case[arm] in row["input"]["historical_context"] for row in arm_rows)
                assert all(row["events"][0]["seed_id"] == row["injected_seed_ids"][0] for row in arm_rows)
                assert all(row["events"][-1]["seed_ids"] == row["injected_seed_ids"] for row in arm_rows)


def test_dry_run_is_not_quality_evidence_and_never_creates_a_client():
    fixture = bench.read_json(bench.DEFAULT_FIXTURE)
    def forbidden(_cfg):
        pytest.fail("No provider calls in dry-run")
    report = asyncio.run(bench.run_benchmark(fixture, client_factory=forbidden))
    assert all(row["status"] == "not_run" and row["answer"] is None for row in report["rows"])
    with pytest.raises(ValueError, match="dry runs"):
        bench.prepare_packets(report, fixture)


@pytest.mark.parametrize("reason", ["length", "content_filter", "tool_calls", None])
def test_invalid_completions_remain_unavailable_in_scoring(monkeypatch, reason):
    report, fixture = live_report(monkeypatch, reason)
    blind, key, ratings = bench.prepare_packets(report, fixture)
    fill_ratings(key, ratings)
    scores = bench.score_packets(blind, key, ratings)
    for arm in bench.ARMS:
        assert scores["arms"][arm]["failed_answers"] == 4
        assert scores["arms"][arm]["unavailable"] == 16
        assert scores["arms"][arm]["eligible_repairs"] == 0


def test_independent_scores_separate_coverage_grounding_and_regressions(monkeypatch):
    report, fixture = live_report(monkeypatch)
    blind, key, ratings = bench.prepare_packets(report, fixture)
    assert not any('"arm"' in json.dumps(packet) for packet in blind["packets"])
    fill_ratings(key, ratings)
    score = bench.score_packets(blind, key, ratings)
    assert score["unique_cases"] == score["packets"] == 4
    assert score["arms"]["correct"]["repaired_coverage_or_defects"] == 12
    assert score["arms"]["stale"]["introduced_failures"] == 4
    assert score["arms"]["stale"]["grounding_failures"] == 4
    assert score["arms"]["no_memory"]["grounding_failures"] == 0
    bad = deepcopy(ratings)
    bad["ratings"].pop()
    with pytest.raises(ValueError, match="exactly once"):
        bench.score_packets(blind, key, bad)
    bad = deepcopy(ratings)
    bad["ratings"][0]["scores"]["A"].popitem()
    with pytest.raises(ValueError, match="coverage"):
        bench.score_packets(blind, key, bad)
    blind["packets"][0]["reference"] += " changed"
    with pytest.raises(ValueError, match="digest"):
        bench.score_packets(blind, key, ratings)


def test_import_checks_fixture_state_ids_complete_arms_and_bound_answer(monkeypatch):
    report, fixture = live_report(monkeypatch)
    bad = deepcopy(report)
    bad["rows"].pop()
    with pytest.raises(ValueError, match="four arms"):
        bench.prepare_packets(bad, fixture)
    bad = deepcopy(report)
    bad["rows"][0]["input"]["question"] += " changed"
    with pytest.raises(ValueError, match="provenance"):
        bench.prepare_packets(bad, fixture)
    bad = deepcopy(report)
    row = next(row for row in bad["rows"] if row["arm"] == "correct")
    bad["memory_snapshots"][row["id"]]["interventions"]["correct"][0]["answer"] += " changed"
    with pytest.raises(ValueError, match="frozen fixture"):
        bench.prepare_packets(bad, fixture)
    report["rows"][0]["answer"] = "Unrelated replacement text"
    blind, key, ratings = bench.prepare_packets(report, fixture)
    assert sum(not info["available"] for a in key["assignments"] for info in a["labels"].values()) == 1


def test_fixture_rejects_named_motif_in_motifless_probe():
    fixture = bench.read_json(bench.DEFAULT_FIXTURE)
    fixture["retrieval_cases"][0]["variants"]["motifless"] += " mirror"
    with pytest.raises(ValueError, match="still names"):
        bench.validate_fixture(fixture)


def test_offline_checks_remove_provider_credentials_but_preserve_network_and_github():
    original = {"LLM_API_KEY": "fixture-secret", "LEFT_BRAIN_MODEL": "fixture", "OPENAI_API_KEY": "fixture",
                "HTTPS_PROXY": "proxy", "SSL_CERT_FILE": "cert", "GH_TOKEN": "git-auth", "PATH": "tools"}
    clean = offline_environment(original)
    assert clean == {"HTTPS_PROXY": "proxy", "SSL_CERT_FILE": "cert", "GH_TOKEN": "git-auth", "PATH": "tools"}
    assert original["LLM_API_KEY"] == "fixture-secret"


def test_model_judge_is_distinct_and_ratings_are_bound_to_its_complete_output(monkeypatch):
    report, fixture = live_report(monkeypatch)
    blind, key, _template = bench.prepare_packets(report, fixture)

    class Judge:
        def __init__(self, _config):
            self.receipts = []

        async def complete(self, prompt, **_kwargs):
            packet = json.loads(prompt)["packet"]
            self.receipts.append({"finish_reason": "stop", "input_tokens": 20, "output_tokens": 10})
            return json.dumps({"packet_id": packet["packet_id"], "scores": {
                label: {c["id"]: {"verdict": "pass", "evidence": "Fixture quoted evidence."} for c in packet["criteria"]}
                for label in packet["answers"]}})

    cfg = LLMConfig("openai", "fixture", "NOT_STORE")
    with pytest.raises(ValueError, match="must differ"):
        asyncio.run(bench.judge_packets(blind, key, cfg, model="fixture", client_factory=Judge))
    ratings = asyncio.run(bench.judge_packets(blind, key, cfg, model="judge-fixture", client_factory=Judge))
    assert ratings["complete"] and len(ratings["ratings"]) == 4
    assert "NOT_STORE" not in json.dumps(ratings)
    assert bench.score_packets(blind, key, ratings)["arms"]["correct"]["pass"] == 16
    ratings["ratings"][0]["scores"]["A"][next(iter(ratings["ratings"][0]["scores"]["A"]))]["verdict"] = "fail"
    with pytest.raises(ValueError, match="validated judge output"):
        bench.score_packets(blind, key, ratings)


def test_partial_report_cannot_be_quality_scored_and_checkpoints_are_private(monkeypatch, tmp_path):
    report, fixture = live_report(monkeypatch)
    report["complete"] = False
    with pytest.raises(ValueError, match="Live report"):
        bench.prepare_packets(report, fixture)
    path = tmp_path / "partial.json"
    bench.checkpoint(path, {"rows": 1})
    bench.checkpoint(path, {"rows": 2})
    assert path.stat().st_mode & 0o777 == 0o600
    assert bench.read_json(path) == {"rows": 2}


def test_judge_retry_only_calls_invalid_packets_preserves_failures_and_checks_identity(monkeypatch):
    report, fixture = live_report(monkeypatch)
    blind, key, _template = bench.prepare_packets(report, fixture)

    class Judge:
        calls = []
        broken = True

        def __init__(self, _config):
            self.receipts = []

        async def complete(self, prompt, **kwargs):
            packet = json.loads(prompt)["packet"]
            self.calls.append((packet["packet_id"], kwargs["system"]))
            self.receipts.append({"finish_reason": "stop", "input_tokens": 20, "output_tokens": 10})
            if self.broken and packet["packet_id"] == "m0002":
                return '{"packet_id": "m0002", "scores": broken JSON}'
            return json.dumps({"packet_id": packet["packet_id"], "scores": {
                label: {c["id"]: {"verdict": "pass", "evidence": "Fixture evidence."} for c in packet["criteria"]}
                for label in packet["answers"]}})

    cfg = LLMConfig("openai", "fixture", "NOT_STORE")
    initial = asyncio.run(bench.judge_packets(blind, key, cfg, model="judge-fixture", client_factory=Judge))
    unchanged = deepcopy(initial)
    assert len(Judge.calls) == 4 and not initial["complete"]
    with pytest.raises(ValueError, match="Incomplete model judging"):
        bench.score_packets(blind, key, initial)
    Judge.broken = False
    retried = asyncio.run(bench.judge_packets(
        blind, key, cfg, model="judge-fixture", client_factory=Judge, retry_from=initial))
    assert initial == unchanged
    assert retried["complete"] and retried["new_judge_calls"] == 1 and len(Judge.calls) == 5
    assert Judge.calls[-1][0] == "m0002" and "escape quotation marks" in Judge.calls[-1][1]
    assert sum(item["reused"] for item in retried["judge_stages"]) == 3
    assert retried["prior_failed_attempts"] == [initial["judge_stages"][1]]
    assert retried["retry_source_sha256"] == bench.digest(initial)
    assert bench.score_packets(blind, key, retried)["packets"] == 4
    again = asyncio.run(bench.judge_packets(
        blind, key, cfg, model="judge-fixture", client_factory=Judge, retry_from=retried))
    assert again["complete"] and again["new_judge_calls"] == 0 and len(Judge.calls) == 5
    assert again["prior_failed_attempts"] == retried["prior_failed_attempts"]
    bad = deepcopy(initial)
    bad["packet_sha256"] = "different"
    with pytest.raises(ValueError, match="packets or judge identity"):
        asyncio.run(bench.judge_packets(blind, key, cfg, model="judge-fixture", client_factory=Judge, retry_from=bad))
    with pytest.raises(ValueError, match="packets or judge identity"):
        asyncio.run(bench.judge_packets(blind, key, cfg, model="another-judge", client_factory=Judge, retry_from=initial))
    for changed in ({"max_tokens": 1024}, {"timeout": 120}):
        with pytest.raises(ValueError, match="judge identity/configuration"):
            asyncio.run(bench.judge_packets(blind, key, cfg, model="judge-fixture",
                                           client_factory=Judge, retry_from=initial, **changed))
    bad = deepcopy(initial)
    bad["judge_stages"].append(deepcopy(bad["judge_stages"][0]))
    with pytest.raises(ValueError, match="duplicate/unknown"):
        asyncio.run(bench.judge_packets(blind, key, cfg, model="judge-fixture", client_factory=Judge, retry_from=bad))
    assert len(Judge.calls) == 5
