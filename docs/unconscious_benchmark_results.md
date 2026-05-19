# Reading Unconscious Benchmark Results

The unconscious benchmark scripts are observability tools. They do not score
whether the model is "more unconscious"; they check whether latent signals are
captured in telemetry, remain out of plain user answers, and produce stable
trend metrics over time.

## Run The Suites

Creativity and leakage sweep:

```bash
python3 sr-dual-brain-llm/scripts/benchmark_unconscious_creativity.py \
  --questions sr-dual-brain-llm/examples/unconscious_creativity_benchmark_questions.json \
  --expect-no-leaks
```

Multi-turn incubation sweep:

```bash
python3 sr-dual-brain-llm/scripts/benchmark_unconscious_incubation.py \
  --sequences sr-dual-brain-llm/examples/unconscious_incubation_benchmark_sequences.json \
  --expect-no-leaks
```

Both scripts write a full report under `target/benchmarks/` and append compact
JSONL history by default. Pass `--history ""` to disable history writes for a
one-off local run.

Convert a JSON report into a short Markdown brief:

```bash
python3 sr-dual-brain-llm/scripts/summarize_benchmark_report.py \
  target/benchmarks/unconscious_incubation_last.json \
  --output target/benchmarks/unconscious_incubation_last.md \
  --top 8
```

The Markdown summary includes run/config metadata, aggregate metrics, count
breakdowns, tag coverage, role-level incubation metrics when present, and a
bounded detail table. Increase `--top` to show more case or sequence rows.

## Report Shape

Each report contains:

- `run_id` and `timestamp` for history correlation.
- `config` with the benchmark modes and filters used for the run.
- `summary` with aggregate trend metrics.
- `cases` or `sequences` with per-question or per-sequence telemetry extracts.

The per-item records intentionally keep only compact answer previews and
telemetry-derived signals. They are meant for engineering diagnosis, not for
showing hidden reasoning to users.

## Creativity Summary Metrics

- `leak_rate`: fraction of successful cases whose plain answer contained an
  internal debug marker. This should stay at `0.0` when `--expect-no-leaks` is
  used.
- `avg_unconscious_score`: blended signal strength from top archetypes,
  emergent ideas, psychoid resonance, coherence weave, and motifs.
- `avg_incubation_pressure`: pressure proxy from cache depth, emergence,
  released stress, psychoid resonance, signifier-chain length, and repeated
  motifs.
- `unreleased_cache_rate`: how often a latent cache remains active without
  emergence. A high value is not automatically bad; it may show seeds are being
  retained for later turns.
- `archetype_cue_top_k_alignment_rate`: how often explicit tags/cues align with
  the top archetype set.
- `archetype_cue_motif_only_rate`: how often the cue is visible in motif
  telemetry even when it is absent from `top_k`.
- `archetype_motif_top_k_divergence_rate`: how often motif evidence and top-k
  activation disagree. Treat spikes as a prompt to inspect examples.

## Incubation Summary Metrics

- `emergent_sequence_rate`: fraction of sequences where any turn produced an
  emergent idea.
- `target_emergent_sequence_rate`: fraction of labeled sequences where an
  expected target archetype emerged.
- `avg_first_emergent_turn_index`: average turn index of first emergence. Lower
  is not always better; compare against the fixture design.
- `avg_peak_incubation_pressure`: highest pressure observed per sequence,
  averaged across sequences.
- `seed_to_emergent_same_archetype_rate`: how often linked seed/emergent pairs
  share an archetype.
- `seed_to_emergent_origin_match_rate`: how often linked emergent ideas point
  back to the original seed text.
- `near_miss_attempts`: harvest attempts that did not emerge but had measurable
  threshold distance.
- `avg_closest_echo_near_miss_gap`: average closest threshold gap on echo turns.
  Smaller gaps mean the system was closer to emergence without crossing the
  threshold.

## How To Read A Run

Start with safety: `leak_rate` and `leak_sequences` should be zero for plain
answer runs. Then inspect signal health: emergence rates, pressure, and cache
depth should move in ways that fit the fixture. Finally, use lineage and
near-miss metrics to explain misses:

- Same-archetype lineage with no target emergence usually means the seed was
  preserved but thresholding stayed conservative.
- Low near-miss gaps on echo turns suggest the sequence is close to surfacing
  and is useful for regression tracking.
- Divergent cue/motif rates suggest the prompt is pulling symbolic evidence in
  more than one direction.

Prefer comparing a run against its own recent JSONL history over treating any
single scalar as a universal pass/fail score.
