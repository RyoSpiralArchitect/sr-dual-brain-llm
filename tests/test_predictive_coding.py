from core.insula import Insula
from core.predictive_coding import PredictiveCodingController
from core.salience_network import SalienceNetwork
from core.thalamus import Thalamus


def test_predictive_coding_elevates_attention_for_symbolic_review():
    question = "Review `sum(xs)/len(xs)` for empty input correctness and zero-length handling."
    insula = Insula().assess(
        question=question,
        affect={"arousal": 0.62, "risk": 0.74},
        novelty=0.72,
        focus_metric=0.56,
        context_signal_len=180,
        has_working_memory=True,
        has_long_term_memory=True,
        is_trivial_chat=False,
    )
    salience = SalienceNetwork().evaluate(
        question=question,
        interoception=insula,
        focus_metric=0.56,
        q_type_hint="easy",
        is_trivial_chat=False,
        has_working_memory=True,
        has_long_term_memory=True,
        has_hippocampal_memory=True,
    )
    relay = Thalamus().route(
        context_parts={
            "working_memory": "recent snippet",
            "memory": "past bug notes",
            "schema": "review rigor",
            "pitfalls": "",
            "hippocampal": "last failure trace",
        },
        salience=salience,
    )

    frame = PredictiveCodingController().evaluate(
        question=question,
        q_type_hint="easy",
        precision_priority=False,
        focus_metric=0.56,
        affect={"arousal": 0.62, "risk": 0.74},
        novelty=0.72,
        hemisphere_mode="balanced",
        hemisphere_bias=0.10,
        collaboration_strength=0.38,
        interoception=insula,
        salience_signal=salience,
        thalamic_relay=relay,
        context_signal_len=180,
        has_working_memory=True,
        has_long_term_memory=True,
        has_hippocampal_memory=True,
        is_trivial_chat=False,
    )

    assert frame.system2_ready is True
    assert frame.system2_pressure >= 0.57
    assert frame.networks.task_positive_mode in {"attention", "executive_control"}
    assert frame.networks.task_positive_load >= 0.55
    assert frame.prediction_error.dominant_channel in {
        "attention",
        "executive_control",
        "memory",
    }


def test_predictive_coding_preserves_reflective_state_when_salience_is_low():
    question = "Imagine a quiet forest as a metaphor for memory and reflection."
    insula = Insula().assess(
        question=question,
        affect={"arousal": 0.08, "risk": 0.04},
        novelty=0.24,
        focus_metric=0.22,
        context_signal_len=0,
        has_working_memory=False,
        has_long_term_memory=False,
        is_trivial_chat=False,
    )
    salience = SalienceNetwork().evaluate(
        question=question,
        interoception=insula,
        focus_metric=0.22,
        q_type_hint="easy",
        is_trivial_chat=False,
        has_working_memory=False,
        has_long_term_memory=False,
        has_hippocampal_memory=False,
    )

    frame = PredictiveCodingController().evaluate(
        question=question,
        q_type_hint="easy",
        precision_priority=False,
        focus_metric=0.22,
        affect={"arousal": 0.08, "risk": 0.04},
        novelty=0.24,
        hemisphere_mode="right",
        hemisphere_bias=0.44,
        collaboration_strength=0.52,
        interoception=insula,
        salience_signal=salience,
        thalamic_relay=None,
        context_signal_len=0,
        has_working_memory=False,
        has_long_term_memory=False,
        has_hippocampal_memory=False,
        is_trivial_chat=False,
    )

    assert frame.networks.dominant_network in {"default_mode", "language"}
    assert frame.networks.phase_state == "reflective"
    assert frame.networks.suppress_default_mode is False
    assert frame.system2_ready is False
