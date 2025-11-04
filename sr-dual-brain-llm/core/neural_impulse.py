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

"""Neural impulse mechanisms based on biological action potential dynamics.

This module implements a faithful model of neural impulse transmission including:
- Membrane potential dynamics with resting and threshold values
- All-or-nothing action potential generation
- Absolute and relative refractory periods
- Synaptic transmission with neurotransmitter release
- Temporal and spatial summation of inputs
- Signal propagation with realistic timing
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class NeurotransmitterType(Enum):
    """Types of neurotransmitters affecting signal transmission."""
    GLUTAMATE = "glutamate"      # Excitatory - main excitatory neurotransmitter
    GABA = "gaba"                # Inhibitory - main inhibitory neurotransmitter
    DOPAMINE = "dopamine"        # Modulatory - reward, motivation
    SEROTONIN = "serotonin"      # Modulatory - mood, arousal
    ACETYLCHOLINE = "acetylcholine"  # Modulatory - attention, learning


class NeuronState(Enum):
    """States a neuron can be in regarding its ability to fire."""
    RESTING = "resting"                    # Ready to fire
    DEPOLARIZING = "depolarizing"          # Approaching threshold
    FIRING = "firing"                      # Generating action potential
    ABSOLUTE_REFRACTORY = "absolute_refractory"  # Cannot fire at all
    RELATIVE_REFRACTORY = "relative_refractory"  # Can fire with stronger stimulus


@dataclass
class NeuralImpulse:
    """Represents a single action potential (spike) propagating through the system.
    
    Based on the Hodgkin-Huxley model of action potentials, this models the
    all-or-nothing electrical signal that neurons use to communicate.
    """
    
    neuron_id: str
    timestamp: float
    amplitude: float = 100.0  # mV - action potentials are ~100mV amplitude
    duration: float = 0.001   # seconds - typical action potential is ~1ms
    propagation_velocity: float = 100.0  # m/s - typical for myelinated axons
    neurotransmitter: NeurotransmitterType = NeurotransmitterType.GLUTAMATE
    metadata: Dict[str, float] = field(default_factory=dict)
    
    def is_active(self, current_time: float) -> bool:
        """Check if this impulse is still actively propagating."""
        return current_time < (self.timestamp + self.duration)
    
    def to_payload(self) -> Dict[str, float | str]:
        """Export impulse data for telemetry."""
        return {
            "neuron_id": self.neuron_id,
            "timestamp": self.timestamp,
            "amplitude": self.amplitude,
            "duration": self.duration,
            "propagation_velocity": self.propagation_velocity,
            "neurotransmitter": self.neurotransmitter.value,
            **self.metadata,
        }


@dataclass
class SynapticInput:
    """Represents input from a presynaptic neuron to a postsynaptic neuron."""
    
    source_neuron_id: str
    weight: float  # Synaptic strength (0.0 to 1.0)
    neurotransmitter: NeurotransmitterType
    timestamp: float
    
    def get_epsp(self) -> float:
        """Calculate excitatory postsynaptic potential (EPSP) contribution.
        
        Returns positive value for excitatory inputs (EPSP) and negative for
        inhibitory inputs (IPSP).
        """
        base_potential = 15.0  # mV - typical EPSP is 0.5-15mV
        if self.neurotransmitter == NeurotransmitterType.GABA:
            return -base_potential * self.weight  # Inhibitory
        elif self.neurotransmitter == NeurotransmitterType.GLUTAMATE:
            return base_potential * self.weight   # Excitatory
        else:
            # Modulatory neurotransmitters have smaller direct effects
            return base_potential * self.weight * 0.3


class Neuron:
    """Models a single neuron with realistic membrane potential dynamics.
    
    Implements the integrate-and-fire model with refractory periods, which
    captures the essential dynamics of biological neurons while remaining
    computationally tractable.
    """
    
    # Biological constants (in mV)
    RESTING_POTENTIAL = -70.0   # Typical resting membrane potential
    THRESHOLD_POTENTIAL = -55.0  # Threshold for action potential
    PEAK_POTENTIAL = 40.0        # Peak of action potential
    HYPERPOLARIZATION = -75.0    # After-hyperpolarization
    
    # Time constants (in seconds)
    ABSOLUTE_REFRACTORY_PERIOD = 0.001  # 1ms - cannot fire
    RELATIVE_REFRACTORY_PERIOD = 0.003  # 3ms - harder to fire
    MEMBRANE_TIME_CONSTANT = 0.010      # 10ms - membrane integration time
    
    def __init__(
        self,
        neuron_id: str,
        *,
        resting_potential: float | None = None,
        threshold_potential: float | None = None,
        base_neurotransmitter: NeurotransmitterType = NeurotransmitterType.GLUTAMATE,
    ) -> None:
        self.neuron_id = neuron_id
        self.resting_potential = resting_potential or self.RESTING_POTENTIAL
        self.threshold_potential = threshold_potential or self.THRESHOLD_POTENTIAL
        self.base_neurotransmitter = base_neurotransmitter
        
        # Dynamic state
        self.membrane_potential = self.resting_potential
        self.state = NeuronState.RESTING
        self.last_spike_time: Optional[float] = None
        self.spike_count = 0
        
        # Integration window for temporal summation
        self.recent_inputs: List[SynapticInput] = []
        
    def _decay_to_resting(self, dt: float) -> None:
        """Exponential decay of membrane potential back to resting state."""
        decay_factor = 1.0 - (dt / self.MEMBRANE_TIME_CONSTANT)
        decay_factor = max(0.0, min(1.0, decay_factor))
        
        potential_diff = self.membrane_potential - self.resting_potential
        self.membrane_potential -= potential_diff * (1.0 - decay_factor)
    
    def _update_refractory_state(self, current_time: float) -> None:
        """Update neuron state based on time since last spike."""
        if self.last_spike_time is None:
            self.state = NeuronState.RESTING
            return
        
        time_since_spike = current_time - self.last_spike_time
        
        if time_since_spike < self.ABSOLUTE_REFRACTORY_PERIOD:
            self.state = NeuronState.ABSOLUTE_REFRACTORY
            # Force hyperpolarization during absolute refractory period
            self.membrane_potential = self.HYPERPOLARIZATION
        elif time_since_spike < (self.ABSOLUTE_REFRACTORY_PERIOD + self.RELATIVE_REFRACTORY_PERIOD):
            self.state = NeuronState.RELATIVE_REFRACTORY
        else:
            self.state = NeuronState.RESTING
    
    def receive_input(self, synaptic_input: SynapticInput) -> None:
        """Receive synaptic input from presynaptic neuron."""
        self.recent_inputs.append(synaptic_input)
    
    def _clean_old_inputs(self, current_time: float, window: float = 0.020) -> None:
        """Remove inputs older than the integration window (default 20ms)."""
        cutoff_time = current_time - window
        self.recent_inputs = [
            inp for inp in self.recent_inputs
            if inp.timestamp > cutoff_time
        ]
    
    def _integrate_inputs(self, current_time: float) -> float:
        """Perform temporal and spatial summation of recent inputs.
        
        This implements the biological process where multiple synaptic inputs
        combine to influence the membrane potential.
        """
        self._clean_old_inputs(current_time)
        
        if not self.recent_inputs:
            return 0.0
        
        # Spatial summation: sum all concurrent inputs
        total_epsp = sum(inp.get_epsp() for inp in self.recent_inputs)
        
        # Temporal summation: weight recent inputs more heavily
        weighted_sum = 0.0
        for inp in self.recent_inputs:
            time_diff = current_time - inp.timestamp
            temporal_weight = max(0.0, 1.0 - (time_diff / 0.020))  # 20ms window
            weighted_sum += inp.get_epsp() * temporal_weight
        
        # Average spatial and temporal summation
        return (total_epsp + weighted_sum) / 2.0
    
    def update(self, current_time: float, dt: float = 0.001) -> Optional[NeuralImpulse]:
        """Update neuron state and potentially generate action potential.
        
        Args:
            current_time: Current simulation time
            dt: Time step for integration
            
        Returns:
            NeuralImpulse if neuron fires, None otherwise
        """
        # Update refractory state
        self._update_refractory_state(current_time)
        
        # Cannot fire during absolute refractory period
        if self.state == NeuronState.ABSOLUTE_REFRACTORY:
            return None
        
        # Integrate synaptic inputs
        input_potential = self._integrate_inputs(current_time)
        
        # Update membrane potential
        self.membrane_potential += input_potential
        
        # Check for threshold crossing BEFORE decay
        effective_threshold = self.threshold_potential
        if self.state == NeuronState.RELATIVE_REFRACTORY:
            # Higher threshold during relative refractory period
            effective_threshold = self.threshold_potential + 5.0
        
        # All-or-nothing firing
        if self.membrane_potential >= effective_threshold:
            self.state = NeuronState.FIRING
            self.last_spike_time = current_time
            self.spike_count += 1
            
            # Generate action potential
            impulse = NeuralImpulse(
                neuron_id=self.neuron_id,
                timestamp=current_time,
                neurotransmitter=self.base_neurotransmitter,
                metadata={
                    "spike_count": float(self.spike_count),
                    "membrane_potential": self.membrane_potential,
                    "input_count": float(len(self.recent_inputs)),
                },
            )
            
            # Reset to peak and begin repolarization
            self.membrane_potential = self.PEAK_POTENTIAL
            
            return impulse
        
        # Apply decay toward resting potential if not firing
        self._decay_to_resting(dt)
        
        # Update state based on membrane potential
        if self.membrane_potential > self.resting_potential + 5.0:
            if self.state == NeuronState.RESTING:
                self.state = NeuronState.DEPOLARIZING
        
        return None
    
    def get_firing_rate(self, time_window: float = 1.0) -> float:
        """Calculate instantaneous firing rate (spikes per second)."""
        if self.last_spike_time is None:
            return 0.0
        current_time = time.time()
        if (current_time - self.last_spike_time) > time_window:
            return 0.0
        return self.spike_count / max(time_window, current_time - (self.last_spike_time or 0))
    
    def to_payload(self) -> Dict[str, float | str | int]:
        """Export neuron state for telemetry."""
        return {
            "neuron_id": self.neuron_id,
            "membrane_potential": self.membrane_potential,
            "state": self.state.value,
            "spike_count": self.spike_count,
            "recent_inputs": len(self.recent_inputs),
            "last_spike_time": self.last_spike_time or 0.0,
        }


class Synapse:
    """Models synaptic transmission between neurons.
    
    Implements chemical synaptic transmission with neurotransmitter release,
    synaptic delay, and plasticity mechanisms.
    """
    
    def __init__(
        self,
        presynaptic_id: str,
        postsynaptic_id: str,
        *,
        weight: float = 0.5,
        neurotransmitter: NeurotransmitterType = NeurotransmitterType.GLUTAMATE,
        synaptic_delay: float = 0.0005,  # 0.5ms typical synaptic delay
        plasticity_rate: float = 0.01,
    ) -> None:
        self.presynaptic_id = presynaptic_id
        self.postsynaptic_id = postsynaptic_id
        self.weight = max(0.0, min(1.0, weight))
        self.neurotransmitter = neurotransmitter
        self.synaptic_delay = synaptic_delay
        self.plasticity_rate = plasticity_rate
        
        self.transmission_count = 0
        self.last_transmission_time: Optional[float] = None
    
    def transmit(
        self,
        impulse: NeuralImpulse,
        current_time: float,
    ) -> Optional[SynapticInput]:
        """Convert presynaptic action potential to postsynaptic input.
        
        This models the process of neurotransmitter release and binding to
        postsynaptic receptors.
        """
        # Synaptic delay before transmission
        transmission_time = current_time + self.synaptic_delay
        
        # Create postsynaptic input
        synaptic_input = SynapticInput(
            source_neuron_id=self.presynaptic_id,
            weight=self.weight,
            neurotransmitter=self.neurotransmitter,
            timestamp=transmission_time,
        )
        
        self.transmission_count += 1
        self.last_transmission_time = transmission_time
        
        return synaptic_input
    
    def apply_hebbian_plasticity(self, pre_active: bool, post_active: bool) -> None:
        """Apply Hebbian learning: neurons that fire together, wire together.
        
        Strengthens synapse when pre and post neurons fire together,
        weakens when they fire out of sync.
        """
        if pre_active and post_active:
            # Long-term potentiation (LTP)
            self.weight = min(1.0, self.weight + self.plasticity_rate)
        elif pre_active and not post_active:
            # Long-term depression (LTD)
            self.weight = max(0.0, self.weight - self.plasticity_rate * 0.5)
    
    def to_payload(self) -> Dict[str, float | str | int]:
        """Export synapse state for telemetry."""
        return {
            "presynaptic_id": self.presynaptic_id,
            "postsynaptic_id": self.postsynaptic_id,
            "weight": self.weight,
            "neurotransmitter": self.neurotransmitter.value,
            "transmission_count": self.transmission_count,
            "synaptic_delay": self.synaptic_delay,
        }


class NeuralIntegrator:
    """Coordinates neural impulse propagation across a network of neurons.
    
    This manages the simulation of action potentials flowing through a network,
    handling timing, synaptic transmission, and network-level dynamics.
    """
    
    def __init__(self) -> None:
        self.neurons: Dict[str, Neuron] = {}
        self.synapses: List[Synapse] = []
        self.active_impulses: List[NeuralImpulse] = []
        self.simulation_time = 0.0
        self.impulse_history: List[NeuralImpulse] = []
    
    def add_neuron(self, neuron: Neuron) -> None:
        """Register a neuron in the network."""
        self.neurons[neuron.neuron_id] = neuron
    
    def add_synapse(self, synapse: Synapse) -> None:
        """Register a synapse connecting two neurons."""
        self.synapses.append(synapse)
    
    def create_simple_pathway(
        self,
        pathway_id: str,
        num_neurons: int = 3,
        neurotransmitter: NeurotransmitterType = NeurotransmitterType.GLUTAMATE,
    ) -> List[str]:
        """Create a simple feed-forward pathway of neurons.
        
        This creates a chain of neurons connected by synapses, useful for
        modeling signal propagation through processing stages.
        """
        neuron_ids = []
        for i in range(num_neurons):
            neuron_id = f"{pathway_id}_neuron_{i}"
            neuron = Neuron(
                neuron_id,
                base_neurotransmitter=neurotransmitter,
            )
            self.add_neuron(neuron)
            neuron_ids.append(neuron_id)
            
            # Connect to previous neuron
            if i > 0:
                synapse = Synapse(
                    presynaptic_id=neuron_ids[i-1],
                    postsynaptic_id=neuron_id,
                    weight=0.7,
                    neurotransmitter=neurotransmitter,
                )
                self.add_synapse(synapse)
        
        return neuron_ids
    
    def inject_stimulus(
        self,
        neuron_id: str,
        strength: float = 1.0,
        neurotransmitter: NeurotransmitterType = NeurotransmitterType.GLUTAMATE,
    ) -> None:
        """Inject external stimulus into a neuron.
        
        This simulates sensory input or other external signals entering
        the neural network.
        """
        if neuron_id not in self.neurons:
            return
        
        # Create strong synaptic input to trigger action potential
        stimulus = SynapticInput(
            source_neuron_id="external",
            weight=min(1.0, strength),
            neurotransmitter=neurotransmitter,
            timestamp=self.simulation_time,
        )
        
        self.neurons[neuron_id].receive_input(stimulus)
    
    def step(self, dt: float = 0.001) -> List[NeuralImpulse]:
        """Advance simulation by one time step.
        
        Args:
            dt: Time step in seconds (default 1ms)
            
        Returns:
            List of newly generated action potentials
        """
        self.simulation_time += dt
        new_impulses: List[NeuralImpulse] = []
        
        # Update all neurons
        for neuron in self.neurons.values():
            impulse = neuron.update(self.simulation_time, dt)
            if impulse is not None:
                new_impulses.append(impulse)
                self.active_impulses.append(impulse)
                self.impulse_history.append(impulse)
        
        # Propagate impulses through synapses
        for impulse in self.active_impulses:
            if not impulse.is_active(self.simulation_time):
                continue
                
            # Find synapses from this neuron
            for synapse in self.synapses:
                if synapse.presynaptic_id == impulse.neuron_id:
                    synaptic_input = synapse.transmit(impulse, self.simulation_time)
                    if synaptic_input and synapse.postsynaptic_id in self.neurons:
                        self.neurons[synapse.postsynaptic_id].receive_input(synaptic_input)
        
        # Clean up old impulses
        self.active_impulses = [
            imp for imp in self.active_impulses
            if imp.is_active(self.simulation_time)
        ]
        
        return new_impulses
    
    def get_network_activity(self) -> Dict[str, float]:
        """Calculate network-wide activity metrics."""
        total_spikes = sum(n.spike_count for n in self.neurons.values())
        active_neurons = sum(
            1 for n in self.neurons.values()
            if n.state in (NeuronState.FIRING, NeuronState.DEPOLARIZING)
        )
        
        avg_potential = sum(
            n.membrane_potential for n in self.neurons.values()
        ) / max(1, len(self.neurons))
        
        return {
            "total_spikes": float(total_spikes),
            "active_neurons": float(active_neurons),
            "network_size": float(len(self.neurons)),
            "active_ratio": active_neurons / max(1, len(self.neurons)),
            "avg_membrane_potential": avg_potential,
            "simulation_time": self.simulation_time,
        }
    
    def to_payload(self) -> Dict[str, object]:
        """Export integrator state for telemetry."""
        return {
            "simulation_time": self.simulation_time,
            "neuron_count": len(self.neurons),
            "synapse_count": len(self.synapses),
            "active_impulses": len(self.active_impulses),
            "network_activity": self.get_network_activity(),
            "neurons": {nid: n.to_payload() for nid, n in self.neurons.items()},
        }
