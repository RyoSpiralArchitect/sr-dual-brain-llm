import pytest

from core.psychoid_attention import (
    HAVE_TORCH,
    PsychoidAttentionAdapter,
)


def test_projection_shapes_and_metadata():
    adapter = PsychoidAttentionAdapter(base_temperature=1.2, clamp=1.6)
    signal = {
        "bias_vector": [0.2, 0.5, -0.1],
        "resonance": 0.42,
        "psychoid_tension": 0.35,
        "signifier_chain": ["shadow:gate", "anima:mirror"],
    }
    projection = adapter.build_projection(signal, seq_len=4, qkv_dim=6)
    assert len(projection.bias_matrix) == 4
    assert all(len(row) == 4 for row in projection.bias_matrix)
    assert projection.norm >= adapter.minimum_bias
    payload = projection.to_payload()
    assert payload["temperature"] == pytest.approx(1.2)
    assert payload["clamp"] == pytest.approx(1.6)
    assert payload["chain_length"] == pytest.approx(2.0)
    assert payload["psychoid_tension"] == pytest.approx(0.35)


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not available")
def test_projection_can_bias_attention_scores():
    import torch

    adapter = PsychoidAttentionAdapter()
    signal = {"bias_vector": [0.4, 0.0, -0.2]}
    projection = adapter.build_projection(signal, seq_len=3)
    scores = torch.zeros(2, 3, 3)
    biased = adapter.apply_to_scores(scores, projection)
    expected = torch.tensor(
        projection.bias_matrix,
        dtype=biased.dtype,
        device=biased.device,
    )
    assert torch.allclose(
        biased[0],
        expected,
    )
