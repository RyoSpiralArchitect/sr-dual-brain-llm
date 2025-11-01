import math

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
