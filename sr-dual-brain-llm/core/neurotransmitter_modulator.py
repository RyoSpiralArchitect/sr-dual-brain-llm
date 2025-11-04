# ============================================================================
#  SpiralReality Proprietary
#  Copyright (c) 2025 SpiralReality. All Rights Reserved.
#
#  NOTICE: This file contains confidential and proprietary information of
#  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
#  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
#  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
#
#  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
#  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
#  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
#  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ============================================================================

"""Neurotransmitter modulator for dual-brain system.

This module implements advanced neurotransmitter-based control mechanisms:
- GABA: Information filtering, noise removal, state transition control
- Glutamate: Task initiation and agent activation
- Dopamine: Reward signaling and PPO feedback
- Serotonin: System stability and cooperative state adjustment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ModulationEffect(Enum):
    """Types of modulation effects on neural processing."""
    INHIBIT = "inhibit"           # Suppress activity
    EXCITE = "excite"             # Enhance activity
    STABILIZE = "stabilize"       # Promote stability
    REWARD = "reward"             # Signal success/reward
    PUNISH = "punish"             # Signal failure/penalty


@dataclass
class NeurotransmitterPulse:
    """Represents a neurotransmitter modulation pulse.
    
    Pulses encode specific control signals that modulate neural activity
    and information flow through the corpus callosum.
    """
    
    transmitter_type: str  # GABA, Glutamate, Dopamine, Serotonin
    effect: ModulationEffect
    intensity: float  # 0.0 to 1.0
    target: Optional[str] = None  # Specific brain region/agent
    metadata: Dict[str, float] = field(default_factory=dict)
    
    def to_payload(self) -> Dict[str, object]:
        """Export pulse data for telemetry."""
        return {
            "transmitter": self.transmitter_type,
            "effect": self.effect.value,
            "intensity": self.intensity,
            "target": self.target or "global",
            **self.metadata,
        }


@dataclass
class InformationFilterResult:
    """Result of GABA-based information filtering."""
    
    should_transmit: bool
    filtered_priority: float
    suppression_strength: float
    reason: str
    
    def to_payload(self) -> Dict[str, object]:
        """Export filter result for telemetry."""
        return {
            "should_transmit": self.should_transmit,
            "filtered_priority": self.filtered_priority,
            "suppression_strength": self.suppression_strength,
            "reason": self.reason,
        }


class GABAModulator:
    """GABA-based inhibitory control system.
    
    Implements:
    - Information filtering and noise removal
    - State transition control
    - Attention focusing through selective inhibition
    """
    
    # Thresholds for information filtering
    MIN_PRIORITY_THRESHOLD = 0.3
    NOISE_DETECTION_THRESHOLD = 0.2
    OVERACTIVATION_THRESHOLD = 0.8
    
    def __init__(
        self,
        *,
        baseline_inhibition: float = 0.2,
        max_inhibition: float = 0.9,
    ) -> None:
        self.baseline_inhibition = baseline_inhibition
        self.max_inhibition = max_inhibition
        
        # Track activation levels per region
        self.activation_levels: Dict[str, float] = {}
        self.recent_transmissions: List[Dict[str, object]] = []
    
    def filter_information(
        self,
        *,
        priority: float,
        novelty: float,
        current_focus: Optional[str] = None,
        task_relevance: float = 1.0,
    ) -> InformationFilterResult:
        """Apply GABA-based filtering to determine if information should be transmitted.
        
        Args:
            priority: Priority score of the information (0.0 to 1.0)
            novelty: Novelty score (0.0 to 1.0)
            current_focus: Current focus area (if any)
            task_relevance: Relevance to current task (0.0 to 1.0)
            
        Returns:
            InformationFilterResult indicating whether to transmit
        """
        # Calculate composite importance score
        importance = (priority * 0.5 + novelty * 0.3 + task_relevance * 0.2)
        
        # Detect noise (low importance across all dimensions)
        if importance < self.NOISE_DETECTION_THRESHOLD:
            return InformationFilterResult(
                should_transmit=False,
                filtered_priority=0.0,
                suppression_strength=self.max_inhibition,
                reason="noise_filtered",
            )
        
        # Check minimum priority threshold
        if priority < self.MIN_PRIORITY_THRESHOLD and task_relevance < 0.5:
            return InformationFilterResult(
                should_transmit=False,
                filtered_priority=priority,
                suppression_strength=0.7,
                reason="low_priority",
            )
        
        # Allow transmission with calculated suppression strength
        suppression = max(0.0, self.baseline_inhibition * (1.0 - importance))
        
        return InformationFilterResult(
            should_transmit=True,
            filtered_priority=importance,
            suppression_strength=suppression,
            reason="transmitted",
        )
    
    def control_state_transition(
        self,
        *,
        source_agent: str,
        target_agent: str,
        source_activation: float,
    ) -> Tuple[bool, NeurotransmitterPulse]:
        """Control state transitions to prevent over-activation.
        
        Args:
            source_agent: Currently active agent
            target_agent: Agent to transition to
            source_activation: Activation level of source (0.0 to 1.0)
            
        Returns:
            Tuple of (allow_transition, inhibitory_pulse)
        """
        # Update activation tracking
        self.activation_levels[source_agent] = source_activation
        
        # Check for over-activation
        if source_activation > self.OVERACTIVATION_THRESHOLD:
            # Generate strong inhibitory pulse to force transition
            pulse = NeurotransmitterPulse(
                transmitter_type="GABA",
                effect=ModulationEffect.INHIBIT,
                intensity=min(1.0, source_activation - 0.5),
                target=source_agent,
                metadata={
                    "activation_level": source_activation,
                    "transition_to": target_agent,
                },
            )
            return True, pulse
        
        # Allow smooth transition with moderate inhibition
        pulse = NeurotransmitterPulse(
            transmitter_type="GABA",
            effect=ModulationEffect.INHIBIT,
            intensity=self.baseline_inhibition,
            target=source_agent,
            metadata={
                "activation_level": source_activation,
                "transition_to": target_agent,
            },
        )
        return True, pulse
    
    def focus_attention(
        self,
        *,
        focus_target: str,
        distractor_strength: float,
    ) -> List[NeurotransmitterPulse]:
        """Generate inhibitory pulses to suppress distractors and focus attention.
        
        Args:
            focus_target: The target to focus on
            distractor_strength: Strength of competing distractors (0.0 to 1.0)
            
        Returns:
            List of inhibitory pulses for non-target regions
        """
        pulses = []
        
        # Strong inhibition for high distractor strength
        inhibition_strength = min(self.max_inhibition, 
                                  self.baseline_inhibition + distractor_strength * 0.6)
        
        # Create pulses for all tracked regions except focus target
        for region in self.activation_levels:
            if region != focus_target:
                pulses.append(NeurotransmitterPulse(
                    transmitter_type="GABA",
                    effect=ModulationEffect.INHIBIT,
                    intensity=inhibition_strength,
                    target=region,
                    metadata={
                        "focus_target": focus_target,
                        "distractor_strength": distractor_strength,
                    },
                ))
        
        return pulses
    
    def to_payload(self) -> Dict[str, object]:
        """Export modulator state for telemetry."""
        return {
            "baseline_inhibition": self.baseline_inhibition,
            "max_inhibition": self.max_inhibition,
            "tracked_regions": len(self.activation_levels),
            "activation_levels": dict(self.activation_levels),
        }


class GlutamateModulator:
    """Glutamate-based excitatory control system.
    
    Implements task initiation and agent activation pulses.
    """
    
    def __init__(
        self,
        *,
        baseline_excitation: float = 0.5,
        max_excitation: float = 1.0,
    ) -> None:
        self.baseline_excitation = baseline_excitation
        self.max_excitation = max_excitation
    
    def initiate_task(
        self,
        *,
        task_complexity: float,
        urgency: float,
        target_agent: str,
    ) -> NeurotransmitterPulse:
        """Generate excitatory pulse to initiate task processing.
        
        Args:
            task_complexity: Complexity of the task (0.0 to 1.0)
            urgency: Task urgency (0.0 to 1.0)
            target_agent: Target agent to activate
            
        Returns:
            Excitatory pulse for task initiation
        """
        # Higher complexity and urgency increase excitation
        intensity = min(
            self.max_excitation,
            self.baseline_excitation + (task_complexity * 0.3 + urgency * 0.2)
        )
        
        return NeurotransmitterPulse(
            transmitter_type="Glutamate",
            effect=ModulationEffect.EXCITE,
            intensity=intensity,
            target=target_agent,
            metadata={
                "task_complexity": task_complexity,
                "urgency": urgency,
            },
        )
    
    def activate_agent(
        self,
        *,
        agent_id: str,
        activation_strength: float,
    ) -> NeurotransmitterPulse:
        """Generate excitatory pulse to activate an agent.
        
        Args:
            agent_id: Agent to activate
            activation_strength: Desired activation strength (0.0 to 1.0)
            
        Returns:
            Excitatory pulse for agent activation
        """
        return NeurotransmitterPulse(
            transmitter_type="Glutamate",
            effect=ModulationEffect.EXCITE,
            intensity=min(self.max_excitation, activation_strength),
            target=agent_id,
            metadata={
                "activation_type": "direct",
            },
        )


class DopamineModulator:
    """Dopamine-based reward system.
    
    Implements reward signaling for PPO policy learning.
    """
    
    def __init__(
        self,
        *,
        baseline_reward: float = 0.0,
        reward_scale: float = 1.0,
    ) -> None:
        self.baseline_reward = baseline_reward
        self.reward_scale = reward_scale
        
        # Track reward history for PPO
        self.reward_history: List[float] = []
    
    def signal_reward(
        self,
        *,
        task_success: bool,
        quality_score: float,
        efficiency: float = 1.0,
    ) -> Tuple[NeurotransmitterPulse, float]:
        """Generate dopamine reward pulse based on task outcome.
        
        Args:
            task_success: Whether task completed successfully
            quality_score: Quality of the output (0.0 to 1.0)
            efficiency: Efficiency metric (0.0 to 1.0+)
            
        Returns:
            Tuple of (reward_pulse, ppo_reward_value)
        """
        # Calculate reward value for PPO
        base_reward = 1.0 if task_success else -0.5
        quality_bonus = quality_score * 0.5
        efficiency_bonus = (efficiency - 0.5) * 0.3
        
        ppo_reward = (base_reward + quality_bonus + efficiency_bonus) * self.reward_scale
        
        # Track for learning
        self.reward_history.append(ppo_reward)
        
        # Generate dopamine pulse
        effect = ModulationEffect.REWARD if ppo_reward > 0 else ModulationEffect.PUNISH
        intensity = min(1.0, abs(ppo_reward))
        
        pulse = NeurotransmitterPulse(
            transmitter_type="Dopamine",
            effect=effect,
            intensity=intensity,
            metadata={
                "task_success": 1.0 if task_success else 0.0,
                "quality_score": quality_score,
                "efficiency": efficiency,
                "ppo_reward": ppo_reward,
            },
        )
        
        return pulse, ppo_reward
    
    def get_average_reward(self, window: int = 10) -> float:
        """Get average reward over recent window for policy updates."""
        if not self.reward_history:
            return 0.0
        recent = self.reward_history[-window:]
        return sum(recent) / len(recent)
    
    def to_payload(self) -> Dict[str, object]:
        """Export modulator state for telemetry."""
        return {
            "baseline_reward": self.baseline_reward,
            "reward_scale": self.reward_scale,
            "total_rewards": len(self.reward_history),
            "avg_reward": self.get_average_reward(),
        }


class SerotoninModulator:
    """Serotonin-based stability system.
    
    Implements system stability and cooperative state adjustment.
    """
    
    def __init__(
        self,
        *,
        baseline_stability: float = 0.5,
        target_stability: float = 0.7,
    ) -> None:
        self.baseline_stability = baseline_stability
        self.target_stability = target_stability
        
        # Track system stability metrics
        self.stability_history: List[float] = []
    
    def adjust_stability(
        self,
        *,
        current_volatility: float,
        cooperation_score: float,
    ) -> NeurotransmitterPulse:
        """Generate serotonin pulse to adjust system stability.
        
        Args:
            current_volatility: Current system volatility (0.0 to 1.0)
            cooperation_score: Inter-agent cooperation score (0.0 to 1.0)
            
        Returns:
            Stabilizing pulse
        """
        # Higher volatility needs more stabilization
        # Higher cooperation is already stable
        current_stability = 1.0 - current_volatility
        stability_deficit = max(0.0, self.target_stability - current_stability)
        
        # Cooperation bonus reduces need for stabilization
        adjusted_need = stability_deficit * (1.0 - cooperation_score * 0.3)
        
        intensity = min(1.0, adjusted_need + self.baseline_stability)
        
        self.stability_history.append(current_stability)
        
        return NeurotransmitterPulse(
            transmitter_type="Serotonin",
            effect=ModulationEffect.STABILIZE,
            intensity=intensity,
            metadata={
                "volatility": current_volatility,
                "cooperation": cooperation_score,
                "stability_deficit": stability_deficit,
            },
        )
    
    def promote_cooperation(
        self,
        *,
        agents: List[str],
        conflict_level: float,
    ) -> List[NeurotransmitterPulse]:
        """Generate serotonin pulses to promote cooperative state.
        
        Args:
            agents: List of agent IDs
            conflict_level: Current conflict level (0.0 to 1.0)
            
        Returns:
            List of stabilizing pulses for each agent
        """
        # Higher conflict needs stronger stabilization
        intensity = min(1.0, self.baseline_stability + conflict_level * 0.4)
        
        pulses = []
        for agent in agents:
            pulses.append(NeurotransmitterPulse(
                transmitter_type="Serotonin",
                effect=ModulationEffect.STABILIZE,
                intensity=intensity,
                target=agent,
                metadata={
                    "conflict_level": conflict_level,
                    "cooperation_mode": True,
                },
            ))
        
        return pulses
    
    def get_stability_trend(self, window: int = 10) -> float:
        """Calculate stability trend over recent window.
        
        Returns:
            Positive if stability improving, negative if declining
        """
        if len(self.stability_history) < 2:
            return 0.0
        
        recent = self.stability_history[-window:]
        if len(recent) < 2:
            return 0.0
        
        # Simple linear trend
        return recent[-1] - recent[0]
    
    def to_payload(self) -> Dict[str, object]:
        """Export modulator state for telemetry."""
        return {
            "baseline_stability": self.baseline_stability,
            "target_stability": self.target_stability,
            "stability_samples": len(self.stability_history),
            "stability_trend": self.get_stability_trend(),
        }


class NeurotransmitterModulator:
    """Coordinated neurotransmitter modulation system.
    
    Integrates GABA, Glutamate, Dopamine, and Serotonin modulators
    to provide comprehensive neural control.
    """
    
    def __init__(self) -> None:
        self.gaba = GABAModulator()
        self.glutamate = GlutamateModulator()
        self.dopamine = DopamineModulator()
        self.serotonin = SerotoninModulator()
        
        # Track all pulses for telemetry
        self.pulse_history: List[NeurotransmitterPulse] = []
    
    def process_information_transfer(
        self,
        *,
        priority: float,
        novelty: float,
        task_relevance: float,
        current_focus: Optional[str] = None,
    ) -> Tuple[InformationFilterResult, List[NeurotransmitterPulse]]:
        """Process information transfer with GABA filtering.
        
        Returns:
            Tuple of (filter_result, modulation_pulses)
        """
        filter_result = self.gaba.filter_information(
            priority=priority,
            novelty=novelty,
            current_focus=current_focus,
            task_relevance=task_relevance,
        )
        
        pulses = []
        if not filter_result.should_transmit:
            # Add inhibitory pulse for filtered information
            pulses.append(NeurotransmitterPulse(
                transmitter_type="GABA",
                effect=ModulationEffect.INHIBIT,
                intensity=filter_result.suppression_strength,
                metadata={"reason": filter_result.reason},
            ))
        
        self.pulse_history.extend(pulses)
        return filter_result, pulses
    
    def initiate_task_with_modulation(
        self,
        *,
        task_complexity: float,
        urgency: float,
        target_agent: str,
        expected_quality: float = 0.7,
    ) -> List[NeurotransmitterPulse]:
        """Initiate task with coordinated neurotransmitter modulation.
        
        Returns:
            List of modulation pulses
        """
        pulses = []
        
        # Glutamate for task initiation
        excitatory_pulse = self.glutamate.initiate_task(
            task_complexity=task_complexity,
            urgency=urgency,
            target_agent=target_agent,
        )
        pulses.append(excitatory_pulse)
        
        # Serotonin for stability during task
        stability_pulse = self.serotonin.adjust_stability(
            current_volatility=urgency * 0.5,  # Urgency creates volatility
            cooperation_score=0.5,  # Neutral cooperation at start
        )
        pulses.append(stability_pulse)
        
        self.pulse_history.extend(pulses)
        return pulses
    
    def complete_task_with_feedback(
        self,
        *,
        success: bool,
        quality: float,
        efficiency: float,
    ) -> Tuple[float, List[NeurotransmitterPulse]]:
        """Complete task with dopamine reward feedback.
        
        Returns:
            Tuple of (ppo_reward, modulation_pulses)
        """
        pulses = []
        
        # Dopamine reward signal
        reward_pulse, ppo_reward = self.dopamine.signal_reward(
            task_success=success,
            quality_score=quality,
            efficiency=efficiency,
        )
        pulses.append(reward_pulse)
        
        self.pulse_history.extend(pulses)
        return ppo_reward, pulses
    
    def to_payload(self) -> Dict[str, object]:
        """Export complete modulator state for telemetry."""
        return {
            "gaba": self.gaba.to_payload(),
            "glutamate": {
                "baseline_excitation": self.glutamate.baseline_excitation,
                "max_excitation": self.glutamate.max_excitation,
            },
            "dopamine": self.dopamine.to_payload(),
            "serotonin": self.serotonin.to_payload(),
            "total_pulses": len(self.pulse_history),
        }
