# Spiral Dual Brain LLM

The **Spiral Dual Brain LLM** experiment explores how a pair of cooperative language-model agents can solve complex tasks by emulating the interaction between the human left and right hemispheres. A lightweight orchestrator ("left brain") owns the primary conversation with a user, while a specialist agent ("right brain") can be summoned on demand through a configurable communication layer called the *corpus callosum*. Shared episodic memory keeps the context synchronized so both agents work from the same knowledge.

## Key Ideas
- **Dual-agent workflow** – the left brain triages requests and only escalates to the right brain when a domain-specific deep dive is warranted.
- **Pluggable corpus callosum** – switch between in-memory, Kafka, or MQTT transports without touching the business logic.
- **Trainable policy** – a simple PPO policy chooses when to invoke the right brain, enabling experimentation with reinforcement-learning driven cooperation.
- **Adaptive memory + telemetry** – enriched shared memory ranks past traces by similarity/tags and the controller emits telemetry hooks for analytics.
- **Executive focus + unconscious cues** – a prefrontal-inspired module gates context while an unconscious field surfaces archetypal signals for tagging and auditing.
- **Emergent unconscious incubation** – unresolved material is cached in the unconscious, resurfacing as archetypal insight hints while stress from noisy inputs is discharged for the next reasoning loop.
- **Basal ganglia gating** – action selection receives striatal-like go/inhibit signals tuned by novelty, affect, and focus feedback.

## Repository Layout
```
└── sr-dual-brain-llm/
    ├── core/
    │   ├── __init__.py
    │   ├── callosum.py             # Default asyncio in-memory transport
    │   ├── callosum_kafka.py       # Kafka-based transport skeleton
    │   ├── callosum_mqtt.py        # MQTT-based transport skeleton
    │   ├── orchestrator.py         # Left-brain coordinator
    │   ├── prefrontal_cortex.py    # Executive-focus heuristics and gating
    │   ├── basal_ganglia.py        # Striatal go/no-go heuristics for consult control
    │   ├── policy_ppo.py           # PPO policy over discrete actions
    │   ├── shared_memory.py        # Persistent memory helpers
    │   └── unconscious_field.py    # Archetypal unconscious field integration
    ├── scripts/
    │   ├── right_worker_broker.py  # Broker worker that runs the right brain
    │   ├── run_server.py           # Entry point using the in-memory backend
    │   └── train_policy.py         # Minimal PPO training loop
    └── samples/
        └── prompts/                # Example questions and prompts
```

Supporting assets live at the repository root:
- `LICENSE_NOTICE.txt`
- `docker-compose.yml`
- `requirements.txt`

## Prerequisites
- Python 3.10+
- Optional: Docker (for Kafka/Mosquitto brokers)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Orchestrator (In-memory Backend)
```bash
python sr-dual-brain-llm/scripts/run_server.py
```
This launches the left-brain orchestrator and an in-process right-brain specialist connected via the asyncio corpus callosum.

## Switching Communication Backends
Set the `CALLOSUM_BACKEND` environment variable before starting any script. Supported values are:
- `memory` *(default)* – single-process asyncio messaging.
- `kafka` – publishes messages via Kafka topics.
- `mqtt` – routes requests with an MQTT broker (e.g., Mosquitto).

### Kafka Quick Start
```bash
docker-compose up -d            # starts Kafka and Mosquitto services
CALLOSUM_BACKEND=kafka python sr-dual-brain-llm/scripts/run_server.py
# In another terminal
CALLOSUM_BACKEND=kafka python sr-dual-brain-llm/scripts/right_worker_broker.py
```

### MQTT Quick Start
```bash
docker-compose up -d
CALLOSUM_BACKEND=mqtt python sr-dual-brain-llm/scripts/run_server.py
CALLOSUM_BACKEND=mqtt python sr-dual-brain-llm/scripts/right_worker_broker.py
```

## Training the PPO Policy
The bundled script demonstrates how to train the policy that decides when to escalate to the right brain.
```bash
python sr-dual-brain-llm/scripts/train_policy.py --algo ppo --epochs 200
```

## Example Interaction
```
User> Provide a detailed statistical analysis of this dataset.
Left Brain> Drafting an overview and consulting the right brain for deeper insights...
Right Brain> Returning detailed regression diagnostics and visual summaries.
Left Brain> Combined response with right-brain references.
```

## Contributing
1. Fork and clone the repository.
2. Create a new branch for your change.
3. Run `ruff` or your preferred linter before submitting a pull request.
4. Describe the backend configuration you tested in the PR body.

## License
See `LICENSE_NOTICE.txt` for third-party notices and licensing information.
