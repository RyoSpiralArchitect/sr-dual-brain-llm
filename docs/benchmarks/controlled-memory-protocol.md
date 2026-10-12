# Controlled memory experiments

This workflow separates two experiments. Recall diagnostics fork an identical
`UnconsciousField`, including its pipeline and sampler state. Answer trials test
the actual `SharedMemory` selection/formatting path followed by one isolated model
call. They do **not** clone the complete controller, test its routing/adaptive
feedback, or prove which memory a model internally used.

## Recall from the same state

The fixture has three synthetic scenes (mirror, forest and mask). Each creates a
seed once, makes exactly one common exposure, then forks before every probe.
All variants therefore start with the same seed ID, cache, intensity, exposure
count, stress and sampler history. Probe draft text is fixed. Variants change only
the question: exact echo, paraphrase, or a question without the motif name.
The intended requirements are written into the fixture before execution; their
semantic equivalence still needs human review. Motifless text is checked for the
listed motif words, which is not a complete semantic test.

`fork()` makes a deep in-memory copy of the field/pipeline/sampler. Custom
embedders must support `deepcopy`. `memory_state()` exposes cache/sampler state
for audit; its digest is **not** a fingerprint of model weights or the whole
controller. The default deterministic hash embedder measures lexical signals,
not learned semantic understanding. Repeated probes are not independent scenes.
Recovery requires the original `seed_id`; label/text similarity is not lineage
evidence. These diagnostics never produce answer-quality scores.

## Four controlled memory conditions

Four fictional project cases cover logging, cache concurrency, payments and
incident diagnosis. Before any model request, fixed synthetic intervention text
specifies correct, dated erroneous, and unrelated information. Each condition
starts from a copy of an empty `SharedMemory`, and adds exactly one QA trace
(except `no_memory`). Selected context comes from `retrieve_related()`; provenance
comes from the same ranking/filtering via `retrieve_related_traces()`.

| Condition | Stored history | Calls per trial |
| --- | --- | --- |
| `no_memory` | Empty | 1 |
| `correct` | Verified current project facts | 1 |
| `stale` | Dated erroneous/unsafe project facts | 1 |
| `noise` | Project-related header, unrelated content | 1 |

The final question and model instructions are identical across conditions for
each case. Each call uses a new client with no conversational history. Output
caps/timeouts match, continuation is disabled, and actual usage/finish reasons
are retained. The configured temperature is 0.2; GPT-5/6 requests omit temperature
and use reasoning-inclusive `max_completion_tokens`. Four repetitions rotate
each condition through every position once per case. This is 64 generation calls,
16 packets, and **four unique questions**, not 64 independent questions.

Correct history supplies facts withheld from the no-memory condition. Consequently,
coverage improvements can mean filling missing information rather than fixing a
false answer. Appropriate uncertainty can fail factual coverage while passing
grounding. The `grounding` criterion separately tests qualification, evidence
dates and unsupported project-specific claims. Generic advice is not evidence of
current project policy. These are planted interventions, not naturally accumulated
production memories; results cannot establish the full system's benefit.

## Trace semantics

Every newly cached latent seed gets a unique `seed_id` and optional `source_qid`.
Harvest attempts and emergent ideas preserve those IDs. A unique matching QA trace
can be linked to its seed without changing stored text. Links are in-memory;
PostgreSQL persistence of the new metadata is not implemented here.

- `memory_seed_linked`: QA/latent-seed identity link.
- `memory_seed_injected`: an unchanged selected memory block was provided to the
  controller's draft stage; this does not prove a model internally relied on it.
- `unconscious_seed_routed`: IDs were routed into an attempted right-consult
  request. Downstream workers need not consume every request field.
- Benchmark `seed_stored` → `memory_context_selected` → `memory_prompt_submitted`:
  IDs of the exact historical context included in an attempted API prompt. API
  failures remain failures; prompt submission is not proof of semantic use.

The existing incubation benchmark now matches lineage by seed ID whenever one is
present, and never substitutes text/archetype matching for an unknown ID. Legacy
reports without IDs retain approximate matching, labeled by their match type.

## Run locally and live

Activate the prepared cloud tools if using this environment. A dry run creates
recall diagnostics and unexecuted answer trials without provider requests:

```bash
python sr-dual-brain-llm/scripts/benchmark_controlled_memory.py run \
  --dry-run --repetitions 4 --output-dir target/benchmarks/memory-dry
```

Live mode uses the existing left-role or shared LLM configuration. In this cloud
environment use the `LLM_API_KEY` secret (the `OPENAI_API_KEY` target name is
reserved), `LLM_PROVIDER=openai`, and an available `LLM_MODEL_ID`. Permit
`api.openai.com`. Keys belong in environment secrets. The HTTP client honors
environment proxy settings for both JSON and streaming while retaining TLS
verification.

```bash
python sr-dual-brain-llm/scripts/benchmark_controlled_memory.py run \
  --live --repetitions 4 --max-tokens 1024 --timeout 40 \
  --output-dir target/benchmarks/memory-live
python sr-dual-brain-llm/scripts/benchmark_controlled_memory.py prepare \
  --report target/benchmarks/memory-live/report.json \
  --output-dir target/benchmarks/memory-review
```

Directories must be new. Full/partial reports, answers and reveal keys are mode
`0600` under ignored `target/`. Each completed call updates an atomic partial
checkpoint; interrupted runs stay incomplete and cannot be prepared for scoring.
Answer generation has no automatic resume/retry. Valid answers require a single successful
call, `finish_reason=stop`, and exact agreement between row text and stage output.
Preparation revalidates input hashes, frozen intervention text, selected IDs,
complete condition coverage and stage evidence; failed trials are not dropped.

## Independent evaluation

For human evaluation, provide `blind.json` and the rating template, withholding
`reveal_key.json`. Conditions/model names and seed IDs are absent from the packet.
Historical context remains visible so the reviewer can judge grounding, and may
reveal conditions; this is limited blinding. The current reference and criterion
definitions are visible only to the evaluator, not to answer generation.

For a separate model evaluator:

```bash
python sr-dual-brain-llm/scripts/benchmark_controlled_memory.py judge \
  --blind target/benchmarks/memory-review/blind.json \
  --key target/benchmarks/memory-review/reveal_key.json \
  --model gpt-6-sol --max-tokens 8192 --timeout 90 \
  --output-dir target/benchmarks/memory-judge
python sr-dual-brain-llm/scripts/benchmark_controlled_memory.py score \
  --blind target/benchmarks/memory-review/blind.json \
  --key target/benchmarks/memory-review/reveal_key.json \
  --ratings target/benchmarks/memory-judge/ratings.json \
  --output target/benchmarks/memory-judge/scores.json
```

The key is used locally to reject reuse of the generation model; only the blind
packet is sent to the evaluator. Model judging adds one call per packet (16 at
four repetitions). Incomplete/malformed judge results remain pending and cannot
be scored as complete. Saved automated ratings must match their validated judge
output. A separate model is not an independent human reviewer; same-family errors,
style preferences and grading mistakes still require human adjudication.
For an incomplete judge run, `judge --retry-from PATH` explicitly reuses validated
packet outputs and evaluates only missing/invalid packets into a new directory.
Packets and the complete public judge configuration (including cap and timeout)
must match. Earlier failures are retained; valid grades
are not rejudged, and the old result stays unchanged. A serialization reminder is
added to failed-packet requests without supplying old ratings or changing criteria.

Scores count pass/fail/uncertain/unavailable criteria, failed answers and grounding
failures per condition. Transitions from the no-memory baseline count repaired
coverage/defects and introduced failures with separate eligible denominators.
Unknown or unavailable comparisons are excluded from those denominators but
remain counted. These correlated case/repetition/criterion units do not support
independence assumptions or broad population-level significance claims.

## Checks with live credentials configured

Existing tests assume heuristic defaults. Run them in a child environment that
removes inherited model credentials/configuration and PostgreSQL DSN:

```bash
python sr-dual-brain-llm/scripts/run_offline_checks.py -- \
  python -m pytest -q --ignore=tests/test_gateway_contract.py
python sr-dual-brain-llm/scripts/run_offline_checks.py -- \
  python -m pytest -q tests/test_gateway_contract.py
node --test tests/test_trace_theater.mjs tests/test_trace_selection.mjs tests/test_live_history.mjs
```

The parent environment is unchanged; proxy/TLS settings and GitHub authentication
are preserved. Fake-provider tests validate budgets, stage binding and arithmetic,
and are not real model-quality evidence.
