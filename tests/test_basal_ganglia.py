from core.basal_ganglia import BasalGanglia


def test_basal_ganglia_recommends_consult_under_high_pressure():
    bg = BasalGanglia(baseline_dopamine=0.5)
    state = {"novelty": 0.7, "consult_bias": 0.2}
    affect = {"risk": 0.4}
    signal = bg.evaluate(state=state, affect=affect, focus_metric=0.3)

    assert signal.recommended_action == 1
    assert signal.go_probability > signal.inhibition
    assert "stimulate_consult" in signal.note


def test_basal_ganglia_dopamine_updates_from_feedback():
    bg = BasalGanglia(baseline_dopamine=0.4, inertia=0.8)
    original = bg.dopamine_level

    bg.integrate_feedback(reward=0.9, latency_ms=1200)
    assert bg.dopamine_level > original

    bg.integrate_feedback(reward=0.2, latency_ms=7000)
    assert bg.dopamine_level < 0.9  # ensure bounded
