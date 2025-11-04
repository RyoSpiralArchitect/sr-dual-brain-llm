"""Tests for neurotransmitter modulator system."""

import pytest
from core.neurotransmitter_modulator import (
    GABAModulator,
    GlutamateModulator,
    DopamineModulator,
    SerotoninModulator,
    NeurotransmitterModulator,
    ModulationEffect,
    NeurotransmitterPulse,
    InformationFilterResult,
)


def test_gaba_modulator_filters_noise():
    """Test GABA filters out low-priority noise."""
    gaba = GABAModulator()
    
    # Low priority, low novelty = noise
    result = gaba.filter_information(
        priority=0.1,
        novelty=0.05,
        task_relevance=0.1,
    )
    
    assert not result.should_transmit
    assert result.reason == "noise_filtered"
    assert result.suppression_strength > 0.5


def test_gaba_modulator_allows_important_information():
    """Test GABA allows high-priority information through."""
    gaba = GABAModulator()
    
    # High priority information should pass
    result = gaba.filter_information(
        priority=0.8,
        novelty=0.6,
        task_relevance=0.9,
    )
    
    assert result.should_transmit
    assert result.reason == "transmitted"
    assert result.filtered_priority > 0.5


def test_gaba_modulator_filters_low_priority():
    """Test GABA filters low-priority, low-relevance information."""
    gaba = GABAModulator()
    
    result = gaba.filter_information(
        priority=0.2,
        novelty=0.5,
        task_relevance=0.3,
    )
    
    assert not result.should_transmit
    assert result.reason == "low_priority"


def test_gaba_modulator_controls_overactivation():
    """Test GABA prevents over-activation by forcing transitions."""
    gaba = GABAModulator()
    
    # High activation should trigger transition
    allow, pulse = gaba.control_state_transition(
        source_agent="right_brain",
        target_agent="left_brain",
        source_activation=0.9,
    )
    
    assert allow
    assert pulse.transmitter_type == "GABA"
    assert pulse.effect == ModulationEffect.INHIBIT
    assert pulse.intensity > 0.3  # Strong inhibition for over-activation
    assert pulse.target == "right_brain"


def test_gaba_modulator_normal_transition():
    """Test GABA allows smooth transitions at normal activation."""
    gaba = GABAModulator()
    
    allow, pulse = gaba.control_state_transition(
        source_agent="left_brain",
        target_agent="right_brain",
        source_activation=0.5,
    )
    
    assert allow
    assert pulse.transmitter_type == "GABA"
    assert pulse.intensity <= 0.5  # Moderate inhibition


def test_gaba_modulator_focuses_attention():
    """Test GABA generates inhibitory pulses for non-focus regions."""
    gaba = GABAModulator()
    
    # Track some regions
    gaba.activation_levels = {
        "region_a": 0.5,
        "region_b": 0.6,
        "region_c": 0.4,
    }
    
    pulses = gaba.focus_attention(
        focus_target="region_a",
        distractor_strength=0.7,
    )
    
    # Should generate pulses for all non-focus regions
    assert len(pulses) == 2  # region_b and region_c
    
    for pulse in pulses:
        assert pulse.transmitter_type == "GABA"
        assert pulse.effect == ModulationEffect.INHIBIT
        assert pulse.target != "region_a"
        assert pulse.intensity > 0.5  # Strong inhibition for distractors


def test_glutamate_modulator_initiates_task():
    """Test glutamate generates excitatory pulses for task initiation."""
    glutamate = GlutamateModulator()
    
    pulse = glutamate.initiate_task(
        task_complexity=0.8,
        urgency=0.7,
        target_agent="right_brain",
    )
    
    assert pulse.transmitter_type == "Glutamate"
    assert pulse.effect == ModulationEffect.EXCITE
    assert pulse.target == "right_brain"
    assert pulse.intensity > 0.5  # High complexity + urgency


def test_glutamate_modulator_activates_agent():
    """Test glutamate activates specific agents."""
    glutamate = GlutamateModulator()
    
    pulse = glutamate.activate_agent(
        agent_id="left_brain",
        activation_strength=0.6,
    )
    
    assert pulse.transmitter_type == "Glutamate"
    assert pulse.effect == ModulationEffect.EXCITE
    assert pulse.target == "left_brain"
    assert pulse.intensity == 0.6


def test_dopamine_modulator_signals_success_reward():
    """Test dopamine generates reward pulses for successful tasks."""
    dopamine = DopamineModulator()
    
    pulse, ppo_reward = dopamine.signal_reward(
        task_success=True,
        quality_score=0.8,
        efficiency=1.2,
    )
    
    assert pulse.transmitter_type == "Dopamine"
    assert pulse.effect == ModulationEffect.REWARD
    assert ppo_reward > 0  # Positive reward for success
    assert pulse.intensity > 0


def test_dopamine_modulator_signals_failure_penalty():
    """Test dopamine generates penalty pulses for failed tasks."""
    dopamine = DopamineModulator()
    
    pulse, ppo_reward = dopamine.signal_reward(
        task_success=False,
        quality_score=0.3,
        efficiency=0.5,
    )
    
    assert pulse.transmitter_type == "Dopamine"
    assert pulse.effect == ModulationEffect.PUNISH
    assert ppo_reward < 0  # Negative reward for failure


def test_dopamine_modulator_tracks_reward_history():
    """Test dopamine tracks reward history for PPO."""
    dopamine = DopamineModulator()
    
    # Generate several rewards
    for i in range(5):
        dopamine.signal_reward(
            task_success=True,
            quality_score=0.7,
            efficiency=1.0,
        )
    
    assert len(dopamine.reward_history) == 5
    avg_reward = dopamine.get_average_reward()
    assert avg_reward > 0


def test_serotonin_modulator_adjusts_stability():
    """Test serotonin adjusts system stability."""
    serotonin = SerotoninModulator()
    
    # High volatility, low cooperation
    pulse = serotonin.adjust_stability(
        current_volatility=0.8,
        cooperation_score=0.3,
    )
    
    assert pulse.transmitter_type == "Serotonin"
    assert pulse.effect == ModulationEffect.STABILIZE
    assert pulse.intensity > 0.5  # High volatility needs stabilization


def test_serotonin_modulator_low_volatility():
    """Test serotonin with low volatility and good cooperation."""
    serotonin = SerotoninModulator()
    
    pulse = serotonin.adjust_stability(
        current_volatility=0.2,
        cooperation_score=0.9,
    )
    
    assert pulse.transmitter_type == "Serotonin"
    assert pulse.effect == ModulationEffect.STABILIZE
    # Lower intensity for stable, cooperative state
    assert pulse.intensity < 0.8


def test_serotonin_modulator_promotes_cooperation():
    """Test serotonin promotes cooperative state among agents."""
    serotonin = SerotoninModulator()
    
    agents = ["left_brain", "right_brain", "prefrontal"]
    pulses = serotonin.promote_cooperation(
        agents=agents,
        conflict_level=0.6,
    )
    
    assert len(pulses) == 3
    for pulse in pulses:
        assert pulse.transmitter_type == "Serotonin"
        assert pulse.effect == ModulationEffect.STABILIZE
        assert pulse.target in agents
        assert pulse.intensity > 0.5  # Moderate conflict


def test_serotonin_modulator_stability_trend():
    """Test serotonin tracks stability trends."""
    serotonin = SerotoninModulator()
    
    # Simulate improving stability
    for volatility in [0.8, 0.6, 0.4, 0.2]:
        serotonin.adjust_stability(
            current_volatility=volatility,
            cooperation_score=0.5,
        )
    
    trend = serotonin.get_stability_trend()
    assert trend > 0  # Stability improving


def test_neurotransmitter_pulse_payload():
    """Test neurotransmitter pulse exports payload correctly."""
    pulse = NeurotransmitterPulse(
        transmitter_type="GABA",
        effect=ModulationEffect.INHIBIT,
        intensity=0.7,
        target="test_region",
        metadata={"test_key": 1.5},
    )
    
    payload = pulse.to_payload()
    assert payload["transmitter"] == "GABA"
    assert payload["effect"] == "inhibit"
    assert payload["intensity"] == 0.7
    assert payload["target"] == "test_region"
    assert payload["test_key"] == 1.5


def test_integrated_modulator_information_filtering():
    """Test integrated modulator filters information."""
    modulator = NeurotransmitterModulator()
    
    # Filter noise
    result, pulses = modulator.process_information_transfer(
        priority=0.1,
        novelty=0.05,
        task_relevance=0.1,
    )
    
    assert not result.should_transmit
    assert len(pulses) > 0
    assert pulses[0].transmitter_type == "GABA"


def test_integrated_modulator_task_initiation():
    """Test integrated modulator initiates tasks with multiple neurotransmitters."""
    modulator = NeurotransmitterModulator()
    
    pulses = modulator.initiate_task_with_modulation(
        task_complexity=0.7,
        urgency=0.8,
        target_agent="right_brain",
    )
    
    # Should have glutamate (excitation) and serotonin (stability)
    assert len(pulses) >= 2
    
    transmitter_types = [p.transmitter_type for p in pulses]
    assert "Glutamate" in transmitter_types
    assert "Serotonin" in transmitter_types


def test_integrated_modulator_task_completion():
    """Test integrated modulator handles task completion with dopamine."""
    modulator = NeurotransmitterModulator()
    
    ppo_reward, pulses = modulator.complete_task_with_feedback(
        success=True,
        quality=0.85,
        efficiency=1.1,
    )
    
    assert ppo_reward > 0  # Successful task
    assert len(pulses) > 0
    assert pulses[0].transmitter_type == "Dopamine"
    assert pulses[0].effect == ModulationEffect.REWARD


def test_integrated_modulator_payload_export():
    """Test integrated modulator exports complete state."""
    modulator = NeurotransmitterModulator()
    
    # Generate some activity
    modulator.process_information_transfer(
        priority=0.5, novelty=0.4, task_relevance=0.6
    )
    modulator.initiate_task_with_modulation(
        task_complexity=0.6, urgency=0.5, target_agent="test"
    )
    
    payload = modulator.to_payload()
    
    assert "gaba" in payload
    assert "glutamate" in payload
    assert "dopamine" in payload
    assert "serotonin" in payload
    assert payload["total_pulses"] > 0


def test_gaba_modulator_payload_export():
    """Test GABA modulator exports state correctly."""
    gaba = GABAModulator()
    
    gaba.activation_levels = {"region_a": 0.5, "region_b": 0.7}
    
    payload = gaba.to_payload()
    assert payload["baseline_inhibition"] > 0
    assert payload["tracked_regions"] == 2
    assert "region_a" in payload["activation_levels"]


def test_dopamine_modulator_payload_export():
    """Test dopamine modulator exports state correctly."""
    dopamine = DopamineModulator()
    
    dopamine.signal_reward(task_success=True, quality_score=0.8, efficiency=1.0)
    
    payload = dopamine.to_payload()
    assert payload["total_rewards"] == 1
    assert payload["avg_reward"] > 0


def test_serotonin_modulator_payload_export():
    """Test serotonin modulator exports state correctly."""
    serotonin = SerotoninModulator()
    
    serotonin.adjust_stability(current_volatility=0.5, cooperation_score=0.7)
    
    payload = serotonin.to_payload()
    assert payload["stability_samples"] == 1
    assert "stability_trend" in payload


def test_filter_result_payload_export():
    """Test information filter result exports payload correctly."""
    result = InformationFilterResult(
        should_transmit=True,
        filtered_priority=0.7,
        suppression_strength=0.2,
        reason="transmitted",
    )
    
    payload = result.to_payload()
    assert payload["should_transmit"] is True
    assert payload["filtered_priority"] == 0.7
    assert payload["reason"] == "transmitted"
