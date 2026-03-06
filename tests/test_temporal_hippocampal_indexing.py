import math
import json
import os
import subprocess
import sys
from pathlib import Path

from core.temporal_hippocampal_indexing import TemporalHippocampalIndexing


def test_index_episode_tracks_collaboration_metadata():
    hippocampus = TemporalHippocampalIndexing(dim=16)

    hippocampus.index_episode(
        "q1",
        "Describe the symbolic river.",
        "It meanders through intuition.",
        leading="right",
        collaboration_strength=0.8,
        selection_reason="balanced",
        tags={"myth", "intuition"},
        metadata={"hemisphere_mode": "right", "hemisphere_bias": 0.42},
    )
    hippocampus.index_episode(
        "q2",
        "Provide the algorithm.",
        "Enumerate the deterministic steps.",
        leading="left",
        collaboration_strength=0.35,
        selection_reason="analysis",
        metadata={"hemisphere_mode": "left", "hemisphere_bias": 0.2},
    )

    assert len(hippocampus.episodes) == 2
    first = hippocampus.episodes[0]
    assert first.leading == "right"
    assert math.isclose(first.collaboration_strength or 0.0, 0.8, rel_tol=1e-5)
    assert first.annotations["hemisphere_mode"] == "right"
    assert first.annotations["hemisphere_bias"] == 0.42
    assert "intuition" in first.tags

    rollup = hippocampus.collaboration_rollup(window=2)
    assert math.isclose(rollup["avg_strength"], (0.8 + 0.35) / 2, rel_tol=1e-5)
    assert math.isclose(rollup["lead_right"], 0.5, rel_tol=1e-5)
    assert rollup["lead_left"] > 0
    assert rollup["strength_coverage"] == 1.0

    summary = hippocampus.retrieve_summary("symbolic river", topk=1)
    assert "lead=right" in summary
    assert "collab=0.80" in summary

    clean = hippocampus.retrieve_summary("symbolic river", topk=1, include_meta=False)
    assert "lead=right" not in clean
    assert "collab=" not in clean
    assert "sim=" not in clean


def test_embedding_is_deterministic_across_hash_seeds():
    project_root = Path(__file__).resolve().parents[1] / "sr-dual-brain-llm"
    env = os.environ.copy()
    # Run the subprocess with `-S` (no site initialization) to avoid flaky
    # third-party startup hooks on some environments. We explicitly pass through
    # any site-packages already on *this* interpreter's path so dependencies like
    # numpy remain importable.
    site_paths = [p for p in sys.path if p and "site-packages" in p]
    env_paths = [str(project_root), *site_paths, env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join([p for p in env_paths if p])

    script = (
        "import json; "
        "from core.temporal_hippocampal_indexing import TemporalHippocampalIndexing; "
        "idx=TemporalHippocampalIndexing(dim=32); "
        "print('EMBED=' + json.dumps(idx.embed_text('hello world').tolist()))"
    )

    def run(seed: str) -> list[float]:
        run_env = dict(env)
        run_env["PYTHONHASHSEED"] = seed
        out = subprocess.check_output(
            [sys.executable, "-S", "-c", script],
            env=run_env,
            text=True,
        )
        marker = "EMBED="
        for line in reversed(out.splitlines()):
            if line.startswith(marker):
                return list(json.loads(line[len(marker):]))
        raise AssertionError(f"Expected {marker} in subprocess output")

    a = run("1")
    b = run("2")
    assert len(a) == len(b)
    assert all(abs(x - y) < 1e-10 for x, y in zip(a, b))


def test_retrieve_filters_false_positives_without_lexical_overlap():
    hippocampus = TemporalHippocampalIndexing(dim=128)
    hippocampus.index_episode("q1", "hello there", "general greeting")
    hippocampus.index_episode("q2", "pascal wager denial", "discussion about the wager")

    # With feature-hashed embeddings, unrelated queries can collide in-vector space.
    # We require lexical overlap to avoid "sticky" irrelevant recalls.
    hits = hippocampus.retrieve("random totally unrelated words", topk=3)
    assert hits == []


def test_retrieve_summary_sanitises_internal_and_coaching_lines():
    hippocampus = TemporalHippocampalIndexing(dim=32)
    hippocampus.index_episode(
        "q1",
        "hello there",
        "こんにちは！\nqid 123\n- Add a warm, informal tone.\n[Hemisphere Routing]\nmode: balanced",
        leading="left",
        selection_reason="test",
    )

    summary = hippocampus.retrieve_summary("hello", topk=1)
    assert "こんにちは" in summary
    assert "qid 123" not in summary
    assert "Add a warm" not in summary
    assert "Hemisphere" not in summary


def test_replay_summary_updates_replay_counters():
    hippocampus = TemporalHippocampalIndexing(dim=32)
    hippocampus.index_episode(
        "q1",
        "incident review for api outage",
        "stabilise the rollout and inspect auth latency",
        leading="left",
        collaboration_strength=0.55,
        metadata={"salience_level": 0.7, "reward": 0.8},
    )

    replay = hippocampus.replay_summary("api outage review", topk=1, include_meta=False)

    assert "stabilise the rollout" in replay
    trace = hippocampus.episodes[0]
    assert trace.replay_count >= 1
    assert trace.retrieval_count >= 1
    assert trace.last_replayed_at is not None


def test_consolidate_feedback_strengthens_trace():
    hippocampus = TemporalHippocampalIndexing(dim=16)
    hippocampus.index_episode(
        "q1",
        "outline the rollback",
        "rollback the deployment",
        metadata={"reward": 0.4},
    )
    before = hippocampus.episodes[0].consolidation_score

    lifecycle = hippocampus.consolidate_feedback(
        qid="q1",
        reward=0.9,
        success=True,
        conflict_resolved=True,
        system2_used=True,
        salience_level=0.8,
    )

    assert lifecycle is not None
    assert lifecycle["after"] > before
    assert hippocampus.episodes[0].annotations["conflict_resolved"] is True


def test_forget_stale_prunes_low_consolidation_memories():
    hippocampus = TemporalHippocampalIndexing(dim=16)
    for idx in range(6):
        hippocampus.index_episode(
            f"q{idx}",
            f"question {idx}",
            f"answer {idx}",
            metadata={"reward": 0.1 if idx < 4 else 0.9},
        )
        hippocampus.episodes[-1].consolidation_score = 0.08 if idx < 4 else 0.9

    forgetting = hippocampus.forget_stale(max_episodes=4, min_keep=2)

    assert forgetting["forgotten"] == 2.0
    assert len(hippocampus.episodes) == 4
    assert sum(1 for trace in hippocampus.episodes if trace.consolidation_score >= 0.9) >= 2
