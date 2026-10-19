# Same-topic memory precision — 2026-10-19

The previous pilot's 0.25 scoped threshold does not transfer safely to adjacent
unanswered facets. On this new evaluation it retrieves all required sources for
27/27 positive queries, but also retrieves something for **18/18 no-answer
queries**. The default 0.50 has 23/27 full-source hits and 12/18 false retrievals.
Runtime defaults remain unchanged. Similarity alone does not establish that a
record answers the requested facet.

This PR adds a harder benchmark, candidate diagnostics and calibration with an
explicit no-answer constraint. It does not fix runtime answerability. See the
[fixed protocol](memory-precision-protocol.md),
[fixture](../../sr-dual-brain-llm/examples/memory_precision_benchmark.json), and
[machine-readable measurements](memory-precision-2026-10-19.metrics.json).

## Evaluation

The fixture has 19 records and 90 queries, split equally between calibration
and evaluation. Each split contains 27 positives and 18 no-answer questions.
All queries identify a known scope, so unknown-name rejection cannot inflate
abstention. Three topic families per split, scope IDs and relevant sources are
disjoint; exact content-cue overlap across splits is zero. This is still authored,
correlated synthetic data, not independently labeled or blind evaluation.

| Selector | First source relevant | All sources within 3 | False retrieval on no-answer queries ↓ | Irrelevant / selected slots | Macro set-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous lexical | 16/27 | 18/27 | 9/18 | 62/82 | 0.424 |
| Scoped lexical | 15/27 | 18/27 | 9/18 | 57/77 | 0.436 |
| Previous hybrid, default 0.70 | 18/27 | 18/27 | 9/18 | 15/35 | 0.596 |
| Scoped hybrid, default 0.50 | 23/27 | 23/27 | 12/18 | 16/41 | 0.640 |
| Previous hybrid, F1 calibration 0.55 | 20/27 | 20/27 | 10/18 | 22/44 | 0.588 |
| Scoped hybrid, F1 calibration 0.60 | 20/27 | 20/27 | 9/18 | 12/34 | 0.644 |
| Scoped hybrid, prior pilot's 0.25 | 26/27 | 27/27 | 18/18 | 31/61 | 0.569 |
| Scoped hybrid, constrained calibration 0.85 | 9/27 | 9/27 | 2/18 | 3/13 | 0.556 |

The two conflicting active Nimbus batch-size records are both gold sources;
full-source recall requires both. Selected-slot precision also penalizes extra
sources on positive queries. A retrieved record is not a generated wrong answer:
these are retrieval measurements, with no new chat or answer-quality grading.

Ordinary F1 calibration follows the unchanged 0.20–0.80 grid. The additional
constraint searches 0.20–1.00, using calibration labels only, and requires zero
no-answer queries selecting a source. It selects 0.85: calibration has 12/27
full-source hits, 18/18 correct abstentions and no irrelevant selected slots.
On evaluation that point misses every English paraphrase and Japanese positive;
only the nine echoes succeed. It still falsely retrieves batch-size records for
a concurrency question and a minimum-length record for a maximum-length question.
**A satisfied calibration constraint is not a safety guarantee on evaluation.**
If the constraint is infeasible, the tool returns `feasible: false` and
`chosen_threshold: null`, retaining a separately labeled best-effort diagnostic.
It does not emit a supposedly safe selector in that case. No threshold from this
study is installed into runtime configuration.

## What the failures tell us

The old development misses have relevant-source content cosines of approximately
0.4924, 0.3279 and 0.2881. Lowering 0.50 to recover them cannot distinguish these
new adjacent-facet negatives. Those three values are post-hoc diagnostics of the
old fixture, not new held-out successes.

In the new default comparison, all nine English adjacent-facet negatives retrieve
records. Only three of nine Japanese negatives do, while five of nine Japanese
positives succeed. At 0.25, Japanese full-source hits rise to 9/9 (first-source
hits 8/9), and Japanese no-answer false retrievals rise to 9/9. Multilingual
recall and abstention must be evaluated together.

For example, asking for Saffron's **maximum** prefix-query length retrieves its
**minimum** length source with cosine **0.9154**, separated from the next distinct
source question by **0.4995**. A large candidate gap also fails to establish
answerability here. Nimbus consumer concurrency versus messages per batch scores
0.8738, with a 0.2658 gap. Both exceed even the constrained 0.85 threshold.
No selected candidate at the fixed 0.50 or 0.25 settings used the below-threshold
strong lexical override, across either split. These failures therefore cannot be
attributed to that bypass on this fixture.

The next runtime experiment should test whether a candidate's evidence supports
the requested relation or quantity: minimum versus maximum, path versus port,
interval versus timeout, batch size versus concurrency. An answerability check
after broad candidate recall is a plausible design, but is **not implemented or
validated by this study**. The present evaluation is development data for that
future design; a further untouched evaluation will be needed.

## Reproduction and limits

From the repository root, lexical-only mode requires no credentials:

```bash
python sr-dual-brain-llm/scripts/benchmark_memory_precision.py \
  --output-dir target/benchmarks/memory-precision-local
```

Live embedding is explicitly enabled by `--semantic-model text-embedding-3-small`
with the existing provider configuration. This run prepared 180 raw/content
question views in **3 provider requests, 3,779 reported input tokens**, using
`text-embedding-3-small` (served name identical), dimension 1,536, in approximately
4.05 seconds. This includes all baseline views, not production per-turn cost.
Answers, relevance labels and no-answer rationales were not embedded.

Complete private vectors permit zero-call replay:

```bash
python sr-dual-brain-llm/scripts/benchmark_memory_precision.py \
  --embedding-cache target/benchmarks/memory-precision-semantic-2026-10-19/embeddings.partial.json \
  --output-dir target/benchmarks/memory-precision-local-replay
```

Output directories must be new. The vector file is private, ignored and not
distributed with the repository; without it, reproducing the semantic study
requires a fresh paid run. Replay matched every source selection, summary,
calibration candidate and candidate diagnostic with zero provider calls.
The unchanged October 19 scope fixture and October 16 unannotated fixture also
replayed with identical source selections, calibration/default metrics and older
field recovery decisions. Random seed identities and state fingerprints are not
asserted identical across independent field constructions.

Local validation passed **435 Python unit tests and 5 gateway contracts**.
Tests exercise feasible and infeasible calibration, evaluation-label independence,
replay, question-only inputs, declared topic/cue separation, known-scope coverage,
lexical-override diagnosis and duplicate-question conflict handling. No new live
database or final-answer validation was performed.

The protocol was committed at `7c80ce1`; the final fixture and implementation
were frozen before paid embedding at clean revision
`f53aca0990b230187324ae91df528ce0e8baa203`. No ranking rule, threshold grid, alias
or relevance label changed after semantic evaluation inspection. Historical
fixtures and reports remain unchanged. Fixture and private-artifact hashes,
all calibration candidates and every evaluation selection are retained in the
metrics. Hashes identify local artifacts, not an independent execution audit.
