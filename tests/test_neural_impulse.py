"""Tests for neural impulse mechanisms."""

import pytest
import time
from core.neural_impulse import (
    NeuralImpulse,
    NeurotransmitterType,
    NeuronState,
    SynapticInput,
    Neuron,
    Synapse,
    NeuralIntegrator,
)


def test_neural_impulse_creation_and_active_state():
    """Test that neural impulses are created correctly and track active state."""
    current_time = time.time()
    impulse = NeuralImpulse(
        neuron_id="test_neuron",
        timestamp=current_time,
        amplitude=100.0,
        duration=0.001,
    )
    
    assert impulse.neuron_id == "test_neuron"
    assert impulse.is_active(current_time)
    assert impulse.is_active(current_time + 0.0005)
    assert not impulse.is_active(current_time + 0.002)
    
    payload = impulse.to_payload()
    assert payload["neuron_id"] == "test_neuron"
    assert payload["amplitude"] == 100.0


def test_synaptic_input_epsp_calculation():
    """Test excitatory and inhibitory postsynaptic potential calculations."""
    current_time = time.time()
    
    # Excitatory input (EPSP)
    excitatory = SynapticInput(
        source_neuron_id="pre",
        weight=0.5,
        neurotransmitter=NeurotransmitterType.GLUTAMATE,
        timestamp=current_time,
    )
    epsp = excitatory.get_epsp()
    assert epsp > 0, "Glutamate should produce positive EPSP"
    
    # Inhibitory input (IPSP)
    inhibitory = SynapticInput(
        source_neuron_id="pre",
        weight=0.5,
        neurotransmitter=NeurotransmitterType.GABA,
        timestamp=current_time,
    )
    ipsp = inhibitory.get_epsp()
    assert ipsp < 0, "GABA should produce negative IPSP"
    
    # Stronger input should produce larger potential
    strong_input = SynapticInput(
        source_neuron_id="pre",
        weight=1.0,
        neurotransmitter=NeurotransmitterType.GLUTAMATE,
        timestamp=current_time,
    )
    assert strong_input.get_epsp() > epsp


def test_neuron_resting_state_and_membrane_potential():
    """Test neuron starts at resting potential and maintains it."""
    neuron = Neuron("test_neuron")
    
    assert neuron.state == NeuronState.RESTING
    assert neuron.membrane_potential == Neuron.RESTING_POTENTIAL
    assert neuron.spike_count == 0
    
    # Update without input should decay toward resting
    neuron.membrane_potential = -65.0  # Slightly depolarized
    current_time = time.time()
    neuron.update(current_time, dt=0.001)
    
    # Should decay back toward resting potential
    assert neuron.membrane_potential < -65.0
    assert neuron.membrane_potential > Neuron.RESTING_POTENTIAL


def test_neuron_fires_when_threshold_reached():
    """Test neuron generates action potential when threshold is crossed."""
    neuron = Neuron("firing_neuron")
    current_time = time.time()
    
    # Strong excitatory input
    strong_input = SynapticInput(
        source_neuron_id="external",
        weight=1.0,
        neurotransmitter=NeurotransmitterType.GLUTAMATE,
        timestamp=current_time,
    )
    
    neuron.receive_input(strong_input)
    
    # Update should trigger action potential
    impulse = neuron.update(current_time, dt=0.001)
    
    assert impulse is not None, "Neuron should fire with strong input"
    assert impulse.neuron_id == "firing_neuron"
    assert neuron.spike_count == 1
    assert neuron.state == NeuronState.FIRING
    assert neuron.last_spike_time == current_time


def test_neuron_refractory_period():
    """Test neuron cannot fire during absolute refractory period."""
    neuron = Neuron("refractory_neuron")
    current_time = time.time()
    
    # Trigger first action potential
    strong_input = SynapticInput(
        source_neuron_id="external",
        weight=1.0,
        neurotransmitter=NeurotransmitterType.GLUTAMATE,
        timestamp=current_time,
    )
    neuron.receive_input(strong_input)
    first_impulse = neuron.update(current_time, dt=0.001)
    assert first_impulse is not None
    
    # Try to fire again immediately (should fail due to absolute refractory)
    current_time += 0.0005  # 0.5ms later
    neuron.receive_input(strong_input)
    second_impulse = neuron.update(current_time, dt=0.001)
    assert second_impulse is None, "Should not fire during absolute refractory period"
    assert neuron.state == NeuronState.ABSOLUTE_REFRACTORY
    
    # After absolute + relative refractory period, should be able to fire
    current_time += 0.005  # 5ms later
    neuron.receive_input(strong_input)
    third_impulse = neuron.update(current_time, dt=0.001)
    assert third_impulse is not None, "Should fire after refractory period"


def test_neuron_temporal_summation():
    """Test neuron integrates multiple inputs over time."""
    neuron = Neuron("integrating_neuron")
    current_time = time.time()
    
    # Multiple weak inputs that individually wouldn't trigger firing
    weak_input = SynapticInput(
        source_neuron_id="external",
        weight=0.3,
        neurotransmitter=NeurotransmitterType.GLUTAMATE,
        timestamp=current_time,
    )
    
    # Send multiple weak inputs
    for i in range(5):
        neuron.receive_input(SynapticInput(
            source_neuron_id=f"external_{i}",
            weight=0.3,
            neurotransmitter=NeurotransmitterType.GLUTAMATE,
            timestamp=current_time + i * 0.002,
        ))
    
    # Update with all inputs present
    current_time += 0.008
    impulse = neuron.update(current_time, dt=0.001)
    
    # Temporal summation should allow firing
    assert impulse is not None, "Temporal summation should trigger action potential"


def test_neuron_inhibitory_suppression():
    """Test inhibitory inputs can prevent firing."""
    neuron = Neuron("inhibited_neuron")
    current_time = time.time()
    
    # Strong excitatory input
    excitatory = SynapticInput(
        source_neuron_id="exc",
        weight=0.8,
        neurotransmitter=NeurotransmitterType.GLUTAMATE,
        timestamp=current_time,
    )
    
    # Strong inhibitory input
    inhibitory = SynapticInput(
        source_neuron_id="inh",
        weight=0.8,
        neurotransmitter=NeurotransmitterType.GABA,
        timestamp=current_time,
    )
    
    neuron.receive_input(excitatory)
    neuron.receive_input(inhibitory)
    
    impulse = neuron.update(current_time, dt=0.001)
    
    # Should not fire due to inhibition
    assert impulse is None, "Inhibition should prevent firing"


def test_synapse_transmission_with_delay():
    """Test synapse transmits impulses with appropriate delay."""
    synapse = Synapse(
        presynaptic_id="pre",
        postsynaptic_id="post",
        weight=0.5,
        synaptic_delay=0.0005,
    )
    
    current_time = time.time()
    impulse = NeuralImpulse(
        neuron_id="pre",
        timestamp=current_time,
    )
    
    synaptic_input = synapse.transmit(impulse, current_time)
    
    assert synaptic_input is not None
    assert synaptic_input.source_neuron_id == "pre"
    assert synaptic_input.timestamp == current_time + 0.0005
    assert synapse.transmission_count == 1


def test_synapse_hebbian_plasticity():
    """Test Hebbian learning strengthens/weakens synapses appropriately."""
    synapse = Synapse(
        presynaptic_id="pre",
        postsynaptic_id="post",
        weight=0.5,
        plasticity_rate=0.1,
    )
    
    initial_weight = synapse.weight
    
    # Correlated activity strengthens synapse (LTP)
    synapse.apply_hebbian_plasticity(pre_active=True, post_active=True)
    assert synapse.weight > initial_weight, "Correlated activity should strengthen synapse"
    
    # Uncorrelated activity weakens synapse (LTD)
    current_weight = synapse.weight
    synapse.apply_hebbian_plasticity(pre_active=True, post_active=False)
    assert synapse.weight < current_weight, "Uncorrelated activity should weaken synapse"


def test_neural_integrator_network_creation():
    """Test neural integrator can create and manage a network."""
    integrator = NeuralIntegrator()
    
    # Create neurons
    neuron1 = Neuron("n1")
    neuron2 = Neuron("n2")
    integrator.add_neuron(neuron1)
    integrator.add_neuron(neuron2)
    
    # Create synapse
    synapse = Synapse("n1", "n2", weight=0.7)
    integrator.add_synapse(synapse)
    
    assert len(integrator.neurons) == 2
    assert len(integrator.synapses) == 1


def test_neural_integrator_signal_propagation():
    """Test action potentials propagate through connected neurons."""
    integrator = NeuralIntegrator()
    
    # Create simple pathway
    pathway = integrator.create_simple_pathway("test", num_neurons=3)
    assert len(pathway) == 3
    
    # Inject strong stimulus into first neuron
    integrator.inject_stimulus(pathway[0], strength=1.0)
    
    # Run simulation
    impulses_per_step = []
    for _ in range(10):
        new_impulses = integrator.step(dt=0.001)
        impulses_per_step.append(len(new_impulses))
    
    # Should see signal propagate through pathway
    total_impulses = sum(impulses_per_step)
    assert total_impulses > 0, "Signal should propagate through pathway"
    
    # Check network activity
    activity = integrator.get_network_activity()
    assert activity["total_spikes"] > 0
    assert activity["network_size"] == 3


def test_neural_integrator_payload_export():
    """Test integrator exports state correctly for telemetry."""
    integrator = NeuralIntegrator()
    pathway = integrator.create_simple_pathway("test", num_neurons=2)
    
    integrator.inject_stimulus(pathway[0], strength=1.0)
    integrator.step(dt=0.001)
    
    payload = integrator.to_payload()
    assert "simulation_time" in payload
    assert "neuron_count" in payload
    assert payload["neuron_count"] == 2
    assert "network_activity" in payload


def test_neuron_different_neurotransmitters():
    """Test neurons with different neurotransmitter types."""
    # Excitatory neuron
    exc_neuron = Neuron(
        "excitatory",
        base_neurotransmitter=NeurotransmitterType.GLUTAMATE,
    )
    
    # Inhibitory neuron
    inh_neuron = Neuron(
        "inhibitory",
        base_neurotransmitter=NeurotransmitterType.GABA,
    )
    
    # Modulatory neuron
    mod_neuron = Neuron(
        "modulatory",
        base_neurotransmitter=NeurotransmitterType.DOPAMINE,
    )
    
    current_time = time.time()
    
    # Trigger all neurons
    strong_input = SynapticInput(
        source_neuron_id="external",
        weight=1.0,
        neurotransmitter=NeurotransmitterType.GLUTAMATE,
        timestamp=current_time,
    )
    
    exc_neuron.receive_input(strong_input)
    inh_neuron.receive_input(strong_input)
    mod_neuron.receive_input(strong_input)
    
    exc_impulse = exc_neuron.update(current_time, dt=0.001)
    inh_impulse = inh_neuron.update(current_time, dt=0.001)
    mod_impulse = mod_neuron.update(current_time, dt=0.001)
    
    assert exc_impulse.neurotransmitter == NeurotransmitterType.GLUTAMATE
    assert inh_impulse.neurotransmitter == NeurotransmitterType.GABA
    assert mod_impulse.neurotransmitter == NeurotransmitterType.DOPAMINE


def test_neuron_firing_rate_calculation():
    """Test neuron calculates firing rate correctly."""
    neuron = Neuron("rate_neuron")
    current_time = time.time()
    
    # Initially no firing
    assert neuron.get_firing_rate() == 0.0
    
    # Trigger multiple spikes
    for i in range(5):
        strong_input = SynapticInput(
            source_neuron_id="external",
            weight=1.0,
            neurotransmitter=NeurotransmitterType.GLUTAMATE,
            timestamp=current_time + i * 0.01,
        )
        neuron.receive_input(strong_input)
        impulse = neuron.update(current_time + i * 0.01, dt=0.001)
        if impulse:
            # Wait for refractory period
            current_time += 0.005
    
    assert neuron.spike_count > 0


def test_neural_integrator_excitatory_inhibitory_balance():
    """Test network with both excitatory and inhibitory neurons."""
    integrator = NeuralIntegrator()
    
    # Create excitatory and inhibitory neurons
    exc_neuron = Neuron("exc", base_neurotransmitter=NeurotransmitterType.GLUTAMATE)
    inh_neuron = Neuron("inh", base_neurotransmitter=NeurotransmitterType.GABA)
    target_neuron = Neuron("target")
    
    integrator.add_neuron(exc_neuron)
    integrator.add_neuron(inh_neuron)
    integrator.add_neuron(target_neuron)
    
    # Connect both to target
    exc_synapse = Synapse("exc", "target", weight=0.7, neurotransmitter=NeurotransmitterType.GLUTAMATE)
    inh_synapse = Synapse("inh", "target", weight=0.7, neurotransmitter=NeurotransmitterType.GABA)
    
    integrator.add_synapse(exc_synapse)
    integrator.add_synapse(inh_synapse)
    
    # Stimulate both
    integrator.inject_stimulus("exc", strength=1.0)
    integrator.inject_stimulus("inh", strength=1.0)
    
    # Run simulation
    for _ in range(20):
        integrator.step(dt=0.001)
    
    # Network should show balanced activity
    activity = integrator.get_network_activity()
    assert activity["total_spikes"] > 0
