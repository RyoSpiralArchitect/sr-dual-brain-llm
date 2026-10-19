# Question/evidence support pilot — 2026-10-19

An optional support check preserved full-source recall on **26/26 positive
evaluation queries** while reducing false retrieval on **10 no-answer queries
from 10 to 0**. It checks bounded source Q&A against the requested facts and
requires exact answer quotes. This is a small authored retrieval experiment,
not evidence of perfect production answer correctness. The feature remains off
by default; see the [usage guide](memory-answerability.md),
[fixed protocol](memory-answerability-protocol.md) and
[measurements](memory-answerability-2026-10-19.metrics.json).

## What was repaired

The first development design checked the existing 0.25 content shortlist. It
removed every false retrieval on eight no-answer queries, but full-source hits
stayed at 13/16 positive queries. Three compound questions lacked one required
record **before** the checker: Japanese image width plus EXIF treatment, and
English/Japanese invoice deadline plus accepted currencies. Every checker
selection matched its available relevant candidates; candidate coverage was the
remaining failure.

The repair expands declared compatible scopes to at most six eligible records,
ranked by existing lexical/optional semantic scores, before checking support.
Accepted independent facts are not pruned again by a relative-score cutoff.
Development full-source hits rose to **16/16**, retaining **0/8** false retrievals.
This repair is included in the implementation, along with gating the exact
latent seed/source identity so a hash trigger cannot bypass support checking.
Unannotated memories retain their previous candidate policy before checking.

## Results

The new fixture has 11 records and 60 English/Japanese queries. Development uses
24 queries across image processing and invoice settlement; evaluation uses 36
across coordinates, parcel tracking and meeting recordings. Scopes, relevant
sources and declared topic families are disjoint, with zero exact content-cue
overlap. English echoes/paraphrases and Japanese variants are correlated.
Two explicit-absence questions count as positives: a record saying speaker names
are not stored can refute a retention question's premise. Silence cannot.

| Run / selector | First source relevant | All required sources within 3 | False retrieval on no-answer queries ↓ | Irrelevant / selected slots | Macro source-set F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development, existing 0.25 | 16/16 | 13/16 | 8/8 | 9/26 | 0.625 |
| Development, checked shortlist | 16/16 | 13/16 | 0/8 | 0/17 | 0.958 |
| Development, expanded + checked | 16/16 | 16/16 | 0/8 | 0/20 | 1.000 |
| Evaluation, existing 0.25 | 26/26 | 26/26 | 10/10 | 15/52 | 0.708 |
| Evaluation, expanded + checked | 26/26 | 26/26 | 0/10 | 0/37 | 1.000 |
| Prior precision development, existing 0.25 | 27/27 | 27/27 | 18/18 | 34/61 | 0.548 |
| Prior precision development, expanded + checked | 27/27 | 27/27 | 0/18 | 0/27 | 1.000 |

The 0.25 comparator is the earlier pilot's fixed experimental setting, not the
runtime default. Scope expansion in the enabled checker uses scores for ranking
and does not apply their acceptance thresholds. The raw/content runtime threshold
defaults remain 0.70/0.50 when the checker is disabled.

Evaluation retains six Japanese single-fact positives, three Japanese compound
queries, their English counterparts and both explicit-absence variants. All ten
unsupported adjacent-facet queries abstain. Both conflicting Maple recording
formats are retrieved; compound Maple questions also retain the separate speaker
record. No conflict was resolved by picking the newest answer. Candidate coverage
is 26/26 positives. No checker-rejection or provider-error failures remain in
this measured evaluation. Baseline evaluation candidate coverage was already
complete: precision improves here from checking; the observed coverage repair
comes from development.

The preceding precision fixture's **calibration split** supplies the additional
45-query measurement. Its 27 positives and 18 no-answer queries all select their
labeled source sets after checking. This is known development/regression data,
not an additional held-out result. No label, prompt, ranking rule or candidate
budget was tuned after inspecting new evaluation responses.

## Cost and operational limits

`gpt-6-luna` was requested and served under that name. The four live measurements
made **129 chat requests, 46,540 reported input tokens and 10,256 completion
tokens**, within the protocol's 150-call ceiling. Initial question-view preparation
made **two embedding requests, 2,429 input tokens**, using
`text-embedding-3-small`, dimension 1,536. All later runs reused complete vectors.
Embedding inputs exclude answers; the separate checker reads bounded answers.

The new 36-query evaluation used 36 chat requests, 12,949 input tokens and 3,094
completion tokens. Preparation with already cached embeddings took a median
**1.615 seconds**, nearest-rank p95 **2.209 seconds**, maximum **2.254 seconds**.
The evaluation and prior-fixture processes ran concurrently. These measurements
include local preparation/checkpoint overhead and are not an end-to-end service
latency guarantee. Synchronous lookup makes no network requests; unchanged
query/source approvals are cached. Each uncached preparation makes at most one
bounded request, with no retry or continuation.

The practical limits matter: scope annotations must be trusted, the scoped
candidate cap is six, normal delivery is three records, and oversized source
questions/answers are withheld rather than checked as incomplete excerpts
(240/640 characters). Large scopes, long records and more varied independently
authored questions need further measurement. Unannotated Japanese candidate misses
are not repaired automatically. Query-free historical QA access is suppressed
when checking is enabled. Failures cannot approve unchecked evidence; unchanged
previous approvals can still be served from cache.

Literal quote validation does not prove semantic entailment. A model could quote
the wrong relation, miss a qualification, follow hostile source text or generalize
incorrectly in a larger corpus. The fixture has only six underlying evaluation
fact questions plus a conflicting record, with correlated variants and ten
no-answer queries. These results do not establish a statistical safety bound,
resistance to adversarial sources, current factual truth or final-answer quality.
No new final-answer grading or live PostgreSQL test was performed.

## Validation and provenance

All four support measurements replayed from complete private response/vector
checkpoints with **zero provider calls**, preserving every source selection,
candidate list, check status and aggregate summary. The unchanged October 19
scope fixture and October 16 unannotated fixture also replayed with checking
disabled: selections, calibration/default metrics and field recovery decisions
match their prior results. Independent field seed identities/state hashes are
not asserted identical.

Local validation passed **459 Python unit tests and five gateway contracts**.
Tests exercise controller preparation without embeddings, approved/denied QA and
latent seeds, both actual worker/model prompt builders on detail/critic routes,
source/model invalidation, excerpt and scope limits, literal quotes, malformed
responses, duplicate JSON fields, provider failure and complete/corrupt replay.
Model calls in these pipeline tests are fake; the benchmark's support calls are
real. The implementation uses existing dependencies and requires explicit
`DUALBRAIN_MEMORY_ANSWERABILITY_MODEL` activation; no cloud-wide paid setting was
enabled.

Protocol and fixture were committed at `8457ab4`. Clean `b855883` measured the
initial shortlist; clean `2cf0eeb71b3bc48f0cd30749fdc6af16ebc92bd7` measured the
coverage repair and was frozen before new evaluation. Subsequent correctness
checks reject duplicate JSON fields/corrupt frozen requests, bind model
configuration changes and withhold sources changed by text sanitization. A removed
line could contain a condition rather than a debug instruction. Clean `b75cc43`
replays all four complete measurements without changing any measured selection
or summary; prompt, API payloads, ranking weights, candidate budget and labels
were unchanged. Original paid provenance remains attached to each
run. Private checkpoints are ignored and not distributed; the metrics retain
their hashes and every measured selection. Hashes identify local artifacts,
not an independent execution audit.
