#!/usr/bin/env python3
"""Demo script showing neurotransmitter modulation in action.

This script demonstrates:
1. GABA-based information filtering
2. Glutamate task initiation
3. Dopamine reward signaling
4. Serotonin stability adjustment
"""

import asyncio
from core.neurotransmitter_modulator import NeurotransmitterModulator
from core.enhanced_callosum import EnhancedCallosum


async def demo_information_filtering():
    """Demonstrate GABA-based information filtering."""
    print("\n" + "="*80)
    print("DEMO 1: GABA-based Information Filtering")
    print("="*80)
    
    callosum = EnhancedCallosum()
    
    # Test 1: Low-priority noise - should be filtered
    print("\nTest 1: Low-priority noise")
    response = await callosum.ask_detail(
        {"content": "random noise", "qid": "demo1"},
        priority=0.1,
        novelty=0.05,
        task_relevance=0.1,
    )
    print(f"  Filtered: {response.get('filtered', False)}")
    print(f"  Reason: {response.get('reason', 'N/A')}")
    
    # Test 2: Important information - should pass
    print("\nTest 2: Important information")
    
    async def responder():
        request = await callosum.recv_request()
        await callosum.publish_response(
            request["qid"],
            {"status": "processed", "result": "analysis complete"}
        )
    
    responder_task = asyncio.create_task(responder())
    
    response = await callosum.ask_detail(
        {"content": "CRITICAL system analysis required", "qid": "demo2"},
        timeout_ms=1000,
    )
    print(f"  Filtered: {response.get('filtered', False)}")
    print(f"  Status: {response.get('status', 'N/A')}")
    
    responder_task.cancel()
    try:
        await responder_task
    except asyncio.CancelledError:
        pass
    
    # Show statistics
    stats = callosum.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Filtered: {stats['filtered_requests']}")
    print(f"  Transmitted: {stats['transmitted_requests']}")
    print(f"  Filter rate: {stats['filter_rate']:.2%}")


def demo_task_lifecycle():
    """Demonstrate complete task lifecycle with neurotransmitter modulation."""
    print("\n" + "="*80)
    print("DEMO 2: Complete Task Lifecycle with Neurotransmitter Modulation")
    print("="*80)
    
    modulator = NeurotransmitterModulator()
    
    # 1. Task Initiation with Glutamate
    print("\nPhase 1: Task Initiation (Glutamate)")
    pulses = modulator.initiate_task_with_modulation(
        task_complexity=0.7,
        urgency=0.8,
        target_agent="right_brain",
    )
    print(f"  Generated {len(pulses)} neurotransmitter pulses:")
    for pulse in pulses:
        print(f"    - {pulse.transmitter_type}: {pulse.effect.value} "
              f"(intensity: {pulse.intensity:.2f})")
    
    # 2. Process with GABA filtering
    print("\nPhase 2: Information Processing (GABA Filtering)")
    filter_result, filter_pulses = modulator.process_information_transfer(
        priority=0.6,
        novelty=0.7,
        task_relevance=0.9,
    )
    print(f"  Should transmit: {filter_result.should_transmit}")
    print(f"  Filtered priority: {filter_result.filtered_priority:.2f}")
    print(f"  Suppression: {filter_result.suppression_strength:.2f}")
    
    # 3. Task Completion with Dopamine
    print("\nPhase 3: Task Completion (Dopamine Reward)")
    ppo_reward, reward_pulses = modulator.complete_task_with_feedback(
        success=True,
        quality=0.85,
        efficiency=1.2,
    )
    print(f"  Task successful!")
    print(f"  PPO Reward: {ppo_reward:.3f}")
    print(f"  Dopamine effect: {reward_pulses[0].effect.value}")
    print(f"  Dopamine intensity: {reward_pulses[0].intensity:.2f}")
    
    # 4. System Stability with Serotonin
    print("\nPhase 4: System Stabilization (Serotonin)")
    stability_pulse = modulator.serotonin.adjust_stability(
        current_volatility=0.3,
        cooperation_score=0.8,
    )
    print(f"  Stability pulse intensity: {stability_pulse.intensity:.2f}")
    print(f"  Effect: {stability_pulse.effect.value}")


def demo_gaba_state_control():
    """Demonstrate GABA-based state transition control."""
    print("\n" + "="*80)
    print("DEMO 3: GABA State Transition Control")
    print("="*80)
    
    modulator = NeurotransmitterModulator()
    
    # Test 1: Normal activation
    print("\nTest 1: Normal activation level (0.5)")
    allow, pulse = modulator.gaba.control_state_transition(
        source_agent="right_brain",
        target_agent="left_brain",
        source_activation=0.5,
    )
    print(f"  Transition allowed: {allow}")
    print(f"  Inhibition intensity: {pulse.intensity:.2f}")
    print(f"  Control: Normal transition")
    
    # Test 2: Over-activation
    print("\nTest 2: Over-activation (0.9)")
    allow, pulse = modulator.gaba.control_state_transition(
        source_agent="right_brain",
        target_agent="left_brain",
        source_activation=0.9,
    )
    print(f"  Transition allowed: {allow}")
    print(f"  Inhibition intensity: {pulse.intensity:.2f}")
    print(f"  Control: Strong inhibition to force transition")


def demo_attention_focusing():
    """Demonstrate GABA-based attention focusing."""
    print("\n" + "="*80)
    print("DEMO 4: GABA Attention Focusing")
    print("="*80)
    
    modulator = NeurotransmitterModulator()
    
    # Set up multiple brain regions
    modulator.gaba.activation_levels = {
        "left_brain": 0.6,
        "right_brain": 0.5,
        "prefrontal": 0.4,
        "amygdala": 0.7,
    }
    
    print("\nCurrent activation levels:")
    for region, level in modulator.gaba.activation_levels.items():
        print(f"  {region}: {level:.2f}")
    
    print("\nFocusing attention on 'left_brain' (distractor strength: 0.7)")
    focus_pulses = modulator.gaba.focus_attention(
        focus_target="left_brain",
        distractor_strength=0.7,
    )
    
    print(f"\nGenerated {len(focus_pulses)} inhibitory pulses:")
    for pulse in focus_pulses:
        print(f"  Target: {pulse.target}")
        print(f"    Inhibition: {pulse.intensity:.2f}")


def demo_dopamine_learning():
    """Demonstrate Dopamine-based reward learning."""
    print("\n" + "="*80)
    print("DEMO 5: Dopamine Reward Learning for PPO")
    print("="*80)
    
    modulator = NeurotransmitterModulator()
    
    print("\nSimulating 5 task completions:")
    outcomes = [
        (True, 0.9, 1.2, "Excellent task"),
        (True, 0.7, 1.0, "Good task"),
        (False, 0.3, 0.6, "Failed task"),
        (True, 0.8, 1.1, "Good task"),
        (True, 0.95, 1.3, "Excellent task"),
    ]
    
    for i, (success, quality, efficiency, desc) in enumerate(outcomes, 1):
        pulse, reward = modulator.dopamine.signal_reward(
            task_success=success,
            quality_score=quality,
            efficiency=efficiency,
        )
        print(f"\n  Task {i}: {desc}")
        print(f"    Success: {success}")
        print(f"    Quality: {quality:.2f}, Efficiency: {efficiency:.2f}")
        print(f"    PPO Reward: {reward:+.3f}")
        print(f"    Dopamine: {pulse.effect.value} (intensity: {pulse.intensity:.2f})")
    
    print(f"\nAverage reward (recent 5): {modulator.dopamine.get_average_reward(5):+.3f}")


def demo_serotonin_cooperation():
    """Demonstrate Serotonin-based cooperation promotion."""
    print("\n" + "="*80)
    print("DEMO 6: Serotonin Cooperation Promotion")
    print("="*80)
    
    modulator = NeurotransmitterModulator()
    
    agents = ["left_brain", "right_brain", "prefrontal"]
    
    print(f"\nPromoting cooperation among: {', '.join(agents)}")
    
    # Test 1: Low conflict
    print("\nScenario 1: Low conflict (0.2)")
    pulses = modulator.serotonin.promote_cooperation(
        agents=agents,
        conflict_level=0.2,
    )
    print(f"  Generated {len(pulses)} stabilization pulses")
    print(f"  Average intensity: {sum(p.intensity for p in pulses) / len(pulses):.2f}")
    
    # Test 2: High conflict
    print("\nScenario 2: High conflict (0.8)")
    pulses = modulator.serotonin.promote_cooperation(
        agents=agents,
        conflict_level=0.8,
    )
    print(f"  Generated {len(pulses)} stabilization pulses")
    print(f"  Average intensity: {sum(p.intensity for p in pulses) / len(pulses):.2f}")
    print(f"  Note: Higher conflict → stronger stabilization")


async def main():
    """Run all demos."""
    print("\n" + "="*80)
    print(" NEUROTRANSMITTER MODULATION SYSTEM DEMONSTRATION")
    print("="*80)
    
    await demo_information_filtering()
    demo_task_lifecycle()
    demo_gaba_state_control()
    demo_attention_focusing()
    demo_dopamine_learning()
    demo_serotonin_cooperation()
    
    print("\n" + "="*80)
    print(" DEMO COMPLETE")
    print("="*80)
    print("\nThe neurotransmitter modulation system provides:")
    print("  ✓ GABA: Information filtering, state control, attention focusing")
    print("  ✓ Glutamate: Task initiation and agent activation")
    print("  ✓ Dopamine: Reward signaling and PPO integration")
    print("  ✓ Serotonin: System stability and cooperation")
    print()


if __name__ == "__main__":
    asyncio.run(main())
