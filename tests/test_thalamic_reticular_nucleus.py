from dataclasses import FrozenInstanceError, replace
import json

import pytest

from core.thalamic_reticular_nucleus import (
    ACC_MONITOR, CEREBELLAR_CORRECTION, CONVERSATION_SELECTIVE_CONSULTATION,
    DEFAULT_REGISTRY, GateConfig, GateFacts, GateMode, ThalamicReticularNucleus,
)


@pytest.fixture
def nucleus():
    return ThalamicReticularNucleus()


def selective_facts(**changes):
    return replace(GateFacts(
        eligible=True, has_draft=True, confidence=0.85, risk=0.0,
        approved_records=1, reference_resolved=True,
        literal_replay_requested=True, full_source_replayed=True,
    ), **changes)


def decide_auto(nucleus, module, facts):
    return nucleus.decide(module, facts=facts, config=GateConfig(GateMode.AUTO, "explicit", True))


@pytest.mark.parametrize("module,mode", [
    (CONVERSATION_SELECTIVE_CONSULTATION, GateMode.OFF),
    (ACC_MONITOR, GateMode.ON), (CEREBELLAR_CORRECTION, GateMode.ON),
])
def test_defaults_preserve_existing_enablement(nucleus, module, mode):
    assert nucleus.configure(module, {}) == GateConfig(mode, "default", True)


@pytest.mark.parametrize("value,mode", [
    ("on", GateMode.ON), ("1", GateMode.ON), ("TRUE", GateMode.ON), (" yes ", GateMode.ON),
    ("off", GateMode.OFF), ("0", GateMode.OFF), ("False", GateMode.OFF), ("no", GateMode.OFF),
])
@pytest.mark.parametrize("module", list(DEFAULT_REGISTRY))
def test_legacy_boolean_spellings_remain_compatible(nucleus, value, mode, module):
    spec = nucleus.registry[module]
    assert nucleus.configure(module, {spec.legacy_env: value}) == GateConfig(mode, "legacy", True)
    assert nucleus.configure(module, {spec.mode_env: value}) == GateConfig(mode, "canonical", True)


def test_canonical_mode_takes_precedence_even_when_invalid(nucleus):
    module = CONVERSATION_SELECTIVE_CONSULTATION
    spec = nucleus.registry[module]
    assert nucleus.configure(module, {spec.mode_env: "AUTO", spec.legacy_env: "0"}) == GateConfig(
        GateMode.AUTO, "canonical", True)
    config = nucleus.configure(module, {spec.mode_env: "private-value", spec.legacy_env: "1"})
    decision = nucleus.decide(module, facts=selective_facts(), config=config)
    assert not config.valid and not decision.enabled
    assert decision.reason == "invalid_configuration"
    assert "private-value" not in json.dumps(decision.to_payload())


@pytest.mark.parametrize("value", ["", "auto", "maybe", "private-value", None, True])
def test_invalid_legacy_value_disables_instead_of_falling_back(nucleus, value):
    spec = nucleus.registry[ACC_MONITOR]
    config = nucleus.configure(ACC_MONITOR, {spec.legacy_env: value})
    decision = nucleus.decide(ACC_MONITOR, facts=GateFacts(eligible=True), config=config)
    assert not config.valid and not decision.enabled
    assert decision.reason == "invalid_configuration"


def test_configure_reads_current_environment_without_caching(nucleus, monkeypatch):
    spec = nucleus.registry[ACC_MONITOR]
    monkeypatch.setenv(spec.mode_env, "off")
    assert nucleus.configure(ACC_MONITOR).mode is GateMode.OFF
    monkeypatch.setenv(spec.mode_env, "auto")
    assert nucleus.configure(ACC_MONITOR).mode is GateMode.AUTO
    assert nucleus.configure(ACC_MONITOR, {}).mode is GateMode.ON


@pytest.mark.parametrize("module", list(DEFAULT_REGISTRY))
def test_explicit_on_never_bypasses_hard_eligibility(nucleus, module):
    decision = nucleus.decide(module,
        facts=GateFacts(eligible=False, blocked_reason="primary_write_pending"),
        config=GateConfig(GateMode.ON, "explicit", True))
    assert not decision.enabled and not decision.eligible
    assert decision.reason == "primary_write_pending"
    allowed = nucleus.decide(module, facts=GateFacts(eligible=True),
        config=GateConfig(GateMode.ON, "explicit", True))
    assert allowed.enabled and allowed.reason == "forced_on"


def test_off_and_invalid_configuration_have_fixed_precedence(nucleus):
    facts = GateFacts(eligible=False, blocked_reason="primary_write_pending")
    disabled = nucleus.decide(ACC_MONITOR, facts=facts,
        config=GateConfig(GateMode.OFF, "explicit", True))
    assert disabled.reason == "forced_off"
    invalid = nucleus.decide(ACC_MONITOR, facts=facts,
        config=GateConfig(GateMode.OFF, "canonical", False))
    assert invalid.reason == "invalid_configuration"


def test_missing_micro_eligibility_has_its_own_bounded_reason(nucleus):
    decision = nucleus.decide(CEREBELLAR_CORRECTION,
        facts=GateFacts(eligible=False, blocked_reason="micro_unavailable"),
        config=GateConfig(GateMode.ON, "explicit", True))
    assert not decision.enabled and decision.reason == "micro_unavailable"


def test_auto_selective_enables_only_complete_literal_replay(nucleus):
    decision = decide_auto(nucleus, CONVERSATION_SELECTIVE_CONSULTATION, selective_facts())
    assert decision.enabled and decision.reason == "supported_literal_replay"


@pytest.mark.parametrize("changes,reason", [
    ({"has_draft": False}, "no_draft"), ({"is_trivial": True}, "trivial_turn"),
    ({"confidence": .849}, "risk_or_uncertainty"), ({"risk": .01}, "risk_or_uncertainty"),
    ({"risk": -.01}, "risk_or_uncertainty"),
    ({"literal_replay_requested": False}, "literal_replay_not_requested"),
    ({"full_source_replayed": False}, "source_replay_incomplete"),
    ({"approved_records": 0}, "unsupported_record_count"),
    ({"approved_records": 2}, "unsupported_record_count"),
    ({"approved_records": True}, "unsupported_record_count"),
    ({"reference_resolved": False}, "reference_unresolved"),
])
def test_auto_selective_withholds_for_each_missing_requirement(nucleus, changes, reason):
    decision = decide_auto(nucleus, CONVERSATION_SELECTIVE_CONSULTATION, selective_facts(**changes))
    assert not decision.enabled and decision.reason == reason


@pytest.mark.parametrize("field", ["confidence", "risk"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf"), "0.9", True])
def test_auto_selective_rejects_unknown_or_nonfinite_numeric_signals(nucleus, field, value):
    decision = decide_auto(nucleus, CONVERSATION_SELECTIVE_CONSULTATION,
        selective_facts(**{field: value}))
    assert not decision.enabled and decision.reason == "unknown_signal"


@pytest.mark.parametrize("changes", [
    {"micro_known": True}, {"risk": .1}, {"confidence": .84},
])
def test_auto_acc_runs_when_any_known_signal_requests_observation(nucleus, changes):
    facts = replace(GateFacts(eligible=True, has_draft=True), **changes)
    decision = decide_auto(nucleus, ACC_MONITOR, facts)
    assert decision.enabled and decision.reason == "observation_needed"


@pytest.mark.parametrize("changes,reason", [
    ({"has_draft": False, "micro_known": True}, "no_draft"),
    ({"is_trivial": True, "micro_known": True}, "trivial_turn"),
    ({"confidence": .85, "risk": 0.0}, "observation_not_needed"),
    ({"confidence": None, "risk": None}, "unknown_signal"),
    ({"confidence": float("nan"), "risk": float("inf")}, "unknown_signal"),
    ({"confidence": .9, "risk": None}, "unknown_signal"),
])
def test_auto_acc_does_not_treat_unknown_signals_as_low_confidence(nucleus, changes, reason):
    facts = replace(GateFacts(eligible=True, has_draft=True), **changes)
    decision = decide_auto(nucleus, ACC_MONITOR, facts)
    assert not decision.enabled and decision.reason == reason


@pytest.mark.parametrize("issues", [1, 4])
def test_auto_cerebellar_accepts_bounded_reliable_micro_issues(nucleus, issues):
    facts = GateFacts(eligible=True, micro_known=True, micro_issues=issues, micro_confidence=.88)
    decision = decide_auto(nucleus, CEREBELLAR_CORRECTION, facts)
    assert decision.enabled and decision.reason == "micro_correction_candidate"


@pytest.mark.parametrize("changes,reason", [
    ({"micro_known": False}, "micro_unknown"),
    ({"micro_issues": 0}, "no_micro_issues"), ({"micro_issues": -1}, "no_micro_issues"),
    ({"micro_issues": 5}, "micro_issue_limit"), ({"micro_issues": True}, "unknown_signal"),
    ({"micro_issues": 1.5}, "unknown_signal"),
    ({"micro_confidence": .879}, "micro_confidence_low"),
    ({"micro_confidence": None}, "unknown_signal"),
    ({"micro_confidence": float("nan")}, "unknown_signal"),
    ({"micro_confidence": float("inf")}, "unknown_signal"),
    ({"micro_confidence": -float("inf")}, "unknown_signal"),
])
def test_auto_cerebellar_rejects_unreliable_or_unbounded_micro_work(nucleus, changes, reason):
    facts = replace(GateFacts(eligible=True, micro_known=True, micro_issues=1, micro_confidence=.9), **changes)
    decision = decide_auto(nucleus, CEREBELLAR_CORRECTION, facts)
    assert not decision.enabled and decision.reason == reason


def test_registry_and_specs_are_immutable_and_custom_mapping_is_copied(nucleus):
    with pytest.raises(TypeError):
        nucleus.registry["new"] = nucleus.registry[ACC_MONITOR]
    with pytest.raises(TypeError):
        DEFAULT_REGISTRY["new"] = nucleus.registry[ACC_MONITOR]
    with pytest.raises(FrozenInstanceError):
        nucleus.registry[ACC_MONITOR].default_mode = GateMode.OFF
    with pytest.raises(AttributeError):
        nucleus.registry = {}
    mapping = {ACC_MONITOR: nucleus.registry[ACC_MONITOR]}
    explicit = ThalamicReticularNucleus(mapping)
    mapping.clear()
    assert list(explicit.registry) == [ACC_MONITOR]
    assert not hasattr(explicit, "register")


def test_unknown_module_never_exposes_caller_text_or_executes_policy(nucleus):
    config = nucleus.configure("private-source-identifier", {})
    decision = nucleus.decide("private-source-identifier", facts=GateFacts(eligible=True), config=config)
    assert not decision.enabled and decision.reason == "unknown_module"
    assert decision.module == "unknown"
    assert "private-source-identifier" not in json.dumps(decision.to_payload())


def broken_policy(_facts):
    raise RuntimeError("private-source-text")


@pytest.mark.parametrize("policy", [
    broken_policy,
    lambda _: (True, "private-source-text"),
    lambda _: ("yes", "observation_needed"),
    lambda _: None,
])
def test_policy_failures_disable_gate_without_leaking_details(nucleus, policy):
    spec = replace(nucleus.registry[ACC_MONITOR], auto_policy=policy)
    explicit = ThalamicReticularNucleus({ACC_MONITOR: spec})
    decision = decide_auto(explicit, ACC_MONITOR, GateFacts(eligible=True))
    assert not decision.enabled and decision.reason == "policy_error"
    assert "private-source-text" not in json.dumps(decision.to_payload())


def test_payload_is_bounded_and_records_are_frozen(nucleus):
    facts = GateFacts(eligible=False, blocked_reason="private-source-text")
    config = GateConfig(GateMode.ON, "private-config-value", True)
    decision = nucleus.decide(ACC_MONITOR, facts=facts, config=config)
    assert decision.reason == "outside_scope" and decision.source == "explicit"
    assert decision.to_payload() == {
        "module": "acc_monitor", "phase": "conflict_regulation", "mode": "on",
        "source": "explicit", "eligible": False, "enabled": False,
        "reason": "outside_scope", "policy_version": "acc_observation_v1",
    }
    for record, field, value in ((facts, "eligible", True), (config, "valid", False),
                                 (decision, "enabled", True)):
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, value)


@pytest.mark.parametrize("source", [None, [], {}, ["private-source-text"], 42, True])
def test_malformed_config_source_is_bounded_without_type_errors(nucleus, source):
    decision = nucleus.decide(ACC_MONITOR, facts=GateFacts(eligible=True),
        config=GateConfig(GateMode.ON, source, True))
    assert decision.enabled and decision.source == "explicit"
    assert "private-source-text" not in json.dumps(decision.to_payload())


def test_registry_rejects_key_and_spec_id_mismatch(nucleus):
    with pytest.raises(ValueError, match="keys must match"):
        ThalamicReticularNucleus({CEREBELLAR_CORRECTION: nucleus.registry[ACC_MONITOR]})


@pytest.mark.parametrize("module_id", [
    "", "Acc_monitor", "acc-monitor", " acc_monitor", "acc_monitor ",
    "acc__monitor", "acc_monitor_", "_acc_monitor", "1_acc_monitor", "a" * 65,
    "private\nsource", "acc.monıtor", None, 42,
])
def test_registry_rejects_invalid_functional_ids(nucleus, module_id):
    spec = replace(nucleus.registry[ACC_MONITOR], module_id=module_id)
    with pytest.raises(ValueError, match="Module IDs"):
        ThalamicReticularNucleus({module_id: spec})


@pytest.mark.parametrize("module_id", ["thalamic_reticular_nucleus", "module_gate"])
def test_gate_cannot_be_registered_as_its_own_optional_module(nucleus, module_id):
    spec = replace(nucleus.registry[ACC_MONITOR], module_id=module_id,
        mode_env=f"DUALBRAIN_MODULE_{module_id.upper()}_MODE")
    with pytest.raises(ValueError, match="cannot register itself"):
        ThalamicReticularNucleus({module_id: spec})


@pytest.mark.parametrize("changes,match", [
    ({"phase": ""}, "phases"), ({"phase": "a" * 33}, "phases"),
    ({"phase": "private source"}, "phases"), ({"phase": []}, "phases"),
    ({"policy_version": ""}, "versions"), ({"policy_version": "a" * 65}, "versions"),
    ({"policy_version": "private\nsource"}, "versions"), ({"policy_version": None}, "versions"),
    ({"mode_env": "DUALBRAIN_MODULE_WRONG_MODE"}, "Canonical"),
    ({"mode_env": None}, "Canonical"),
    ({"default_mode": "on"}, "GateMode"), ({"default_mode": None}, "GateMode"),
    ({"auto_policy": None}, "callable"), ({"auto_policy": "policy_name"}, "callable"),
    ({"legacy_env": "private source"}, "Environment names"),
    ({"legacy_env": "A" * 129}, "Environment names"),
    ({"legacy_env": []}, "Environment names"),
])
def test_registry_validates_metadata_configuration_and_explicit_callbacks(nucleus, changes, match):
    spec = replace(nucleus.registry[ACC_MONITOR], **changes)
    with pytest.raises(ValueError, match=match):
        ThalamicReticularNucleus({ACC_MONITOR: spec})


def test_registry_accepts_bounded_custom_spec_without_discovery(nucleus):
    module_id = "a" * 64
    spec = replace(nucleus.registry[ACC_MONITOR], module_id=module_id,
        phase="p" * 32, policy_version="v" * 64,
        mode_env=f"DUALBRAIN_MODULE_{module_id.upper()}_MODE", legacy_env=None)
    gate = ThalamicReticularNucleus({module_id: spec})
    assert gate.registry[module_id] == spec
    assert gate.configure(module_id, {}).mode is GateMode.ON
    assert ACC_MONITOR not in gate.registry


@pytest.mark.parametrize("collision", ["shared_legacy", "canonical_as_legacy", "same_spec"])
def test_registry_rejects_environment_name_collisions(nucleus, collision):
    first = nucleus.registry[ACC_MONITOR]
    second = nucleus.registry[CEREBELLAR_CORRECTION]
    if collision == "shared_legacy":
        second = replace(second, legacy_env=first.legacy_env)
    elif collision == "canonical_as_legacy":
        first = replace(first, legacy_env=second.mode_env)
    else:
        first = replace(first, legacy_env=first.mode_env)
    with pytest.raises(ValueError, match="must be unique"):
        ThalamicReticularNucleus({ACC_MONITOR: first, CEREBELLAR_CORRECTION: second})
