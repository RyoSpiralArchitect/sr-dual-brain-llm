# Neurotransmitter Modulation System

## Overview

The neurotransmitter modulation system provides biologically-inspired control mechanisms for the dual-brain LLM architecture. It implements four key neurotransmitter types (GABA, Glutamate, Dopamine, Serotonin) that regulate information flow, agent activation, reward signaling, and system stability.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│         NeurotransmitterModulator (Coordinator)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ GABAModulator│  │  Glutamate   │  │  Dopamine    │     │
│  │              │  │  Modulator   │  │  Modulator   │     │
│  │ • Filter     │  │              │  │              │     │
│  │ • Control    │  │ • Initiate   │  │ • Reward     │     │
│  │ • Focus      │  │ • Activate   │  │ • PPO Feed   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
│  ┌──────────────┐                                          │
│  │  Serotonin   │                                          │
│  │  Modulator   │                                          │
│  │              │                                          │
│  │ • Stabilize  │                                          │
│  │ • Cooperate  │                                          │
│  └──────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              EnhancedCallosum (Integration)                 │
│                                                             │
│  Standard Callosum + GABA Filtering                        │
│  • Auto priority/novelty detection                         │
│  • Neurotransmitter metadata                               │
│  • Transmission statistics                                 │
└─────────────────────────────────────────────────────────────┘
```

## Neurotransmitter Functions

### GABA (γ-aminobutyric acid) - Inhibitory Control

**Primary Role**: Suppress unwanted activity and maintain balance

#### 1. Information Filtering
- **Purpose**: Filter noise and low-priority information before transmission
- **Metrics**:
  - Priority score (0.0-1.0)
  - Novelty score (0.0-1.0)
  - Task relevance (0.0-1.0)
- **Thresholds**:
  - `MIN_PRIORITY_THRESHOLD = 0.3`
  - `NOISE_DETECTION_THRESHOLD = 0.2`
- **Algorithm**:
  ```python
  importance = priority * 0.5 + novelty * 0.3 + task_relevance * 0.2
  
  if importance < NOISE_DETECTION_THRESHOLD:
      return FILTER  # Complete suppression
  elif priority < MIN_PRIORITY_THRESHOLD and task_relevance < 0.5:
      return FILTER  # Low priority suppression
  else:
      suppression = baseline_inhibition * (1.0 - importance)
      return TRANSMIT with suppression
  ```

#### 2. State Transition Control
- **Purpose**: Prevent over-activation of agents
- **Threshold**: `OVERACTIVATION_THRESHOLD = 0.8`
- **Behavior**:
  - Activation < 0.8: Normal baseline inhibition (0.2)
  - Activation > 0.8: Strong inhibition (activation - 0.5) to force transition

#### 3. Attention Focusing
- **Purpose**: Suppress distractors to enhance focus
- **Algorithm**:
  ```python
  inhibition_strength = baseline + distractor_strength * 0.6
  # Applied to all regions except focus target
  ```

### Glutamate - Excitatory Activation

**Primary Role**: Initiate activity and activate agents

#### 1. Task Initiation
- **Factors**:
  - Task complexity (0.0-1.0)
  - Urgency (0.0-1.0)
- **Formula**:
  ```python
  intensity = baseline + complexity * 0.3 + urgency * 0.2
  # Capped at max_excitation (default 1.0)
  ```

#### 2. Agent Activation
- **Direct activation with specified strength**
- **Used for**: Explicit agent invocation

### Dopamine - Reward & Motivation

**Primary Role**: Signal success/failure and drive learning

#### Reward Calculation
```python
base_reward = 1.0 if success else -0.5
quality_bonus = quality_score * 0.5
efficiency_bonus = (efficiency - 0.5) * 0.3

ppo_reward = (base_reward + quality_bonus + efficiency_bonus) * scale
```

**Example Rewards**:
- Excellent (success, quality=0.9, eff=1.2): +1.66
- Good (success, quality=0.7, eff=1.0): +1.50
- Poor success (success, quality=0.5, eff=0.8): +1.09
- Failure (fail, quality=0.3, eff=0.6): -0.32

**PPO Integration**:
- Rewards stored in history buffer
- Windowed averaging (default window=10)
- Direct feedback to policy learning

### Serotonin - Stability & Cooperation

**Primary Role**: Maintain system stability and promote cooperation

#### 1. Stability Adjustment
- **Inputs**:
  - Current volatility (0.0-1.0)
  - Cooperation score (0.0-1.0)
- **Algorithm**:
  ```python
  current_stability = 1.0 - volatility
  stability_deficit = max(0.0, target_stability - current_stability)
  adjusted_need = stability_deficit * (1.0 - cooperation * 0.3)
  intensity = adjusted_need + baseline_stability
  ```
- **Target Stability**: 0.7 (configurable)

#### 2. Cooperation Promotion
- **Triggered by**: High conflict levels
- **Intensity**: `baseline + conflict_level * 0.4`
- **Applied to**: All agents in the system

## EnhancedCallosum Integration

### Auto-Detection Features

#### Priority Estimation
- **Explicit priority**: Uses `payload["priority"]` if present
- **Critical keywords**: `error, critical, urgent, important, 必須, 緊急` (+0.2)
- **User-facing**: `payload["user_facing"] = True` (+0.2)
- **Base priority**: 0.5

#### Novelty Estimation
- **Explicit novelty**: Uses `payload["novelty"]` if present
- **Novelty keywords**: `new, novel, unprecedented, unexpected, 新しい, 初めて` (+0.15)
- **Base novelty**: 0.5

### Transmission Statistics

The EnhancedCallosum tracks:
- `total_requests`: All ask_detail calls
- `filtered_requests`: Requests blocked by GABA
- `transmitted_requests`: Requests that passed through
- `filter_rate`: filtered / total

## Usage Examples

### Basic Information Filtering

```python
from core.enhanced_callosum import EnhancedCallosum

callosum = EnhancedCallosum()

# This will be filtered (low priority)
response = await callosum.ask_detail(
    {"content": "random noise", "qid": "1"},
    priority=0.1,
    novelty=0.05,
    task_relevance=0.1,
)
# response["filtered"] == True

# This will pass through (high priority)
response = await callosum.ask_detail(
    {"content": "critical error", "qid": "2"},
    # Auto-detects priority from "critical"
)
# response["filtered"] == False
```

### Complete Task Lifecycle

```python
from core.neurotransmitter_modulator import NeurotransmitterModulator

modulator = NeurotransmitterModulator()

# 1. Initiate task with Glutamate + Serotonin
pulses = modulator.initiate_task_with_modulation(
    task_complexity=0.7,
    urgency=0.8,
    target_agent="right_brain",
)
# Returns [GlutamatePulse, SerotoninPulse]

# 2. Process information with GABA filtering
filter_result, pulses = modulator.process_information_transfer(
    priority=0.6,
    novelty=0.7,
    task_relevance=0.9,
)
# filter_result.should_transmit == True

# 3. Complete with Dopamine reward
ppo_reward, pulses = modulator.complete_task_with_feedback(
    success=True,
    quality=0.85,
    efficiency=1.2,
)
# ppo_reward ≈ +1.635 for PPO learning
```

### State Transition Control

```python
gaba = modulator.gaba

# Normal transition
allow, pulse = gaba.control_state_transition(
    source_agent="right_brain",
    target_agent="left_brain",
    source_activation=0.5,
)
# pulse.intensity ≈ 0.2 (baseline)

# Over-activation handling
allow, pulse = gaba.control_state_transition(
    source_agent="right_brain",
    target_agent="left_brain",
    source_activation=0.9,
)
# pulse.intensity ≈ 0.4 (strong inhibition)
```

### Attention Focusing

```python
gaba.activation_levels = {
    "left_brain": 0.6,
    "right_brain": 0.5,
    "prefrontal": 0.4,
}

pulses = gaba.focus_attention(
    focus_target="left_brain",
    distractor_strength=0.7,
)
# Returns inhibitory pulses for right_brain and prefrontal
# Each with intensity ≈ 0.62
```

## Configuration Parameters

### GABAModulator
- `baseline_inhibition`: 0.2 (default)
- `max_inhibition`: 0.9 (default)
- `MIN_PRIORITY_THRESHOLD`: 0.3
- `NOISE_DETECTION_THRESHOLD`: 0.2
- `OVERACTIVATION_THRESHOLD`: 0.8

### GlutamateModulator
- `baseline_excitation`: 0.5 (default)
- `max_excitation`: 1.0 (default)

### DopamineModulator
- `baseline_reward`: 0.0 (default)
- `reward_scale`: 1.0 (default)

### SerotoninModulator
- `baseline_stability`: 0.5 (default)
- `target_stability`: 0.7 (default)

### EnhancedCallosum
- `enable_filtering`: True (default)
- `slot_ms`: 250 (default, from base Callosum)

## Performance Characteristics

### Memory Usage
- **NeurotransmitterModulator**: ~2KB baseline
- **Pulse History**: ~100 bytes per pulse
- **EnhancedCallosum**: ~1KB + base Callosum

### Computational Cost
- **Information Filtering**: O(1)
- **State Transition**: O(1)
- **Attention Focusing**: O(n) where n = number of tracked regions
- **Reward Calculation**: O(1)
- **Stability Adjustment**: O(1)

### Typical Latencies
- GABA filtering: <0.1ms
- Glutamate initiation: <0.1ms
- Dopamine reward: <0.1ms
- Serotonin adjustment: <0.1ms

All operations are synchronous except EnhancedCallosum which uses async/await for Callosum operations.

## Testing

### Test Coverage
- 24 neurotransmitter modulator tests
- 11 enhanced callosum tests
- Coverage: ~95% of code paths

### Key Test Scenarios
1. GABA filters noise correctly
2. GABA allows important information
3. GABA controls over-activation
4. GABA focuses attention
5. Glutamate initiates tasks
6. Dopamine signals rewards/penalties
7. Dopamine tracks history
8. Serotonin adjusts stability
9. Serotonin promotes cooperation
10. EnhancedCallosum integrates filtering
11. Auto-detection works correctly

## Future Enhancements

Potential areas for extension:

1. **Adaptive Thresholds**: Learn optimal thresholds from usage patterns
2. **Multi-Agent Coordination**: Cross-agent neurotransmitter synchronization
3. **Temporal Dynamics**: Time-dependent modulation effects
4. **Plasticity**: Hebbian-like adaptation of modulation strengths
5. **Acetylcholine Integration**: Attention modulation system
6. **Norepinephrine**: Alertness and arousal control
7. **Circadian Rhythms**: Time-of-day dependent modulation

## References

- Hodgkin-Huxley Model: Foundation for neural impulse dynamics
- PPO (Proximal Policy Optimization): Reward signal integration
- Basal Ganglia: Go/No-Go decision making
- Prefrontal Cortex: Executive control and attention

## See Also

- `neural_impulse.py`: Low-level neural dynamics
- `dual_brain.py`: Integration with dual-brain system
- `policy_ppo.py`: PPO policy learning
- `README.md`: System overview
- `demo_neurotransmitters.py`: Interactive demonstrations
