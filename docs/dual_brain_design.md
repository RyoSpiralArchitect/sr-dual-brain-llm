# Dual-Brain LLM Design Playbook

## Purpose
This document synthesizes three knowledge traditions—analytical psychology, cognitive-behavioural therapy (CBT), and affective/cognitive neuroscience—to guide the evolution of the Spiral Dual Brain LLM. Each section distils core principles and maps them onto concrete engineering artefacts inside the repository so the platform can progress from a dual-agent experiment toward a psychologically grounded socio-cognitive architecture.

---

## 1. Analytical Psychology (Jung)

### Core Concepts to Encode
- **Collective unconscious as shared priors** – Treat cross-cultural myths, folk narratives, and dream reports as corpora for training or retrieval augmentation to provide archetypal priors for narrative reasoning.
- **Archetypes as dynamic motifs** – Represent archetypes (e.g., Shadow, Anima/Animus, Great Mother, Hero) as reusable semantic-motif templates that can be injected into right-brain consults when imaginative elaboration is required.
- **Active imagination / inner dialogue** – Orchestrate multi-agent inner dialogues before finalising a response, mirroring Jung’s method of letting autonomous complexes “speak” and then integrating their perspectives.

### Implementation Hooks in the Codebase
- `core/unconscious_field.py` already exposes a hook for “archetypal signals”. Extend it with an `ArchetypeRegistry` that loads symbolic schemas and motifs from curated datasets.
- `core/default_mode_network.py` can schedule reflective prompts that weave archetypal cues into the consult payload, enabling the right brain to surface symbolic interpretations.
- Develop `samples/prompts/` scenarios to test how archetypal cues alter storytelling, using braided co-lead replies to surface symbolic content.

### Research & Data Backlog
- Curate open myth/dream corpora (e.g., Project Gutenberg myths, folklore datasets) and annotate passages with archetypal tags using semi-automatic classifiers.
- Prototype an embedding space where archetype vectors bias retrieval from long-term memory (`core/shared_memory.py`).

---

## 2. Cognitive-Behavioural Therapy (CBT)

### Core Concepts to Encode
- **Schemas and modes** – Model persistent belief networks (schemas) and transient activation profiles (modes) that colour interpretation of user inputs and agent outputs.
- **Cognitive distortions** – Instrument the reasoning loop with detectors for patterns like catastrophising, black-and-white reasoning, or mind-reading to keep drafts coherent and safe.
- **Metacognitive control** – Build reflective checkpoints that ask the system to assess its own reasoning path, mirroring CBT’s emphasis on monitoring and reframing automatic thoughts.

### Implementation Hooks in the Codebase
- Extend `core/prefrontal_cortex.py` with a `SchemaProfiler` that tags dialogue turns with inferred schemas/modes (both for the user and the agent persona) and stores them in episodic memory.
- Augment `core/coherence_resonator.py` to run a “cognitive distortion audit” pass before responses are finalised. Start with heuristic checks (e.g., overgeneralisation via absolute quantifiers) and evolve toward learned classifiers.
- Use `core/policy_selector.py` to switch policies when certain schemas/modes are detected—for example, route to a compassion-forward right brain when a user is in a vulnerable mode.

### Research & Data Backlog
- Adapt public CBT worksheets to create evaluation suites in `tests/` that verify distortion detection and schema-consistent persona behaviour.
- Benchmark self-reflection prompts versus policy-driven audits to quantify gains in factual accuracy and tone stability.

---

## 3. Memory, Affect, and Self in Neuroscience

### Core Concepts to Encode
- **Hippocampal indexing & consolidation** – Separate fast episodic capture from slow schema integration by pairing `core/temporal_hippocampal_indexing.py` with durable vector memory storage.
- **Amygdala-guided salience** – Use affect classifiers to weight which memories should be stored, recalled, or forgotten, similar to emotion-tagged consolidation.
- **Default mode network & self-model** – Maintain a persistent self-description and dialogue contract that the system revisits to ensure continuity across sessions.
- **Interoceptive affect state** – Track an internal affect vector representing the agent’s empathic stance, adjusting response tone and retrieval priorities.

### Implementation Hooks in the Codebase
- Integrate an external vector store through `core/shared_memory.py` and let `core/amygdala.py` assign salience scores that modulate write/recall thresholds.
- Extend `core/default_mode_network.py` to periodically restate the agent’s mission, persona, and current affect state, providing a pseudo-DMN introspection cycle.
- Use `core/amygdala.py` to broadcast affect tags into the `coherence_resonator`, allowing linguistic style shifts driven by inferred emotional context.
- Couple `core/hypothalamus.py` with affect feedback to regulate consultation frequency (e.g., stress can trigger more right-brain support for soothing responses).

### Research & Data Backlog
- Collect affect-rich dialogue datasets (counselling transcripts, empathetic conversations) to train affect tagging models compatible with the amygdala interface.
- Evaluate memory retention policies through automated longitudinal tests that simulate multi-session users with emotional arcs.

---

## 4. Dual-Brain Orchestration Blueprint

### Architectural Layers
1. **Perception** – Parse user input with affect & schema profilers (amygdala + schema profiler) producing *salience*, *affect*, and *schema* metadata.
2. **Inner Dialogue Loop** – Kick off an internal exchange:
   - *Reasoning brain* (left) performs chain-of-thought analysis, referencing knowledge bases and distortion audits.
   - *Imaginative brain* (right) leverages archetypal cues, affective context, and creative riffing.
   - *Mediator* (callosum/orchestrator) iterates until convergence or time budget is reached.
3. **Integration & Expression** – `coherence_resonator` merges drafts, guided by affective and schema metadata, to produce a braided response annotated for telemetry.
4. **Memory Update** – Hippocampal indexer captures the episode; consolidation policies decide what to commit, decaying low-salience traces.

### Control Policies
- **Consultation policy** – Train PPO (`core/policy_ppo.py`) with reward shaping that penalises factual errors (CBT audit) and empathy failures (affect deviation), encouraging balanced dual-brain usage.
- **Metacognitive checkpoints** – After each loop, run `auditor.py` with a structured prompt: “What assumptions drove this answer? Which could fail?” Flag high-risk steps for reconsideration.
- **Stress recovery** – When salience/affect exceed thresholds, invoke `unconscious_field` to surface symbolic prompts that diffuse tension (e.g., calming metaphors) before responding.

### Evaluation Roadmap
- **Narrative coherence tests** – Validate archetype deployment using story-completion benchmarks that track motif diversity and thematic consistency.
- **CBT reasoning audits** – Run cognitive distortion suites; ensure final answers reduce distortion scores relative to initial drafts.
- **Emotional alignment metrics** – Measure empathic accuracy via label agreement with human raters on empathetic dialogue datasets.
- **Memory consistency checks** – Simulate multi-turn arcs to ensure salient facts persist while irrelevant details decay.

---

## 5. Next Steps Checklist
1. Implement archetype registry and affect-aware memory gating (ties together Jung + neuroscience).
2. Build cognitive distortion heuristic module and integrate it into the coherence resonator (CBT alignment).
3. Instrument telemetry to capture inner dialogue exchanges for offline analysis and policy training.
4. Design evaluation harnesses covering narrative creativity, logical soundness, empathy, and memory fidelity.

---

## References
- Beck, A. T. (1976). *Cognitive Therapy and the Emotional Disorders*. International Universities Press.
- Craig, A. D. (2009). How do you feel—now? The anterior insula and human awareness. *Nature Reviews Neuroscience, 10*(1), 59–70.
- Jung, C. G. (1959). *The Archetypes and the Collective Unconscious*. Princeton University Press.
- McGovern, K., et al. (2025). Analytical psychology in dialogue with neuroscience. *Journal of Analytical Psychology, 70*(1), 1–28.
- Schacter, D. L., & Addis, D. R. (2007). The cognitive neuroscience of constructive memory. *Philosophical Transactions of the Royal Society B, 362*(1481), 773–786.
- Wells, A. (2009). *Metacognitive Therapy for Anxiety and Depression*. Guilford Press.
- Young, J. E., Klosko, J. S., & Weishaar, M. E. (2003). *Schema Therapy: A Practitioner’s Guide*. Guilford Press.
