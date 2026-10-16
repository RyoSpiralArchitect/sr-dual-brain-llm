# Question-centered memory pilot — 2026-10-16

The new path separates question retrieval from draft/archetype processing and
resolves recalled IDs to stored final QA contents with provenance. On a synthetic
competing-memory corpus, metadata/function-word filtering provided most of the
gain over the previous selector. A calibrated semantic index further reduced
irrelevant candidates, with a remaining recall/abstention tradeoff.

The [guide](question-memory-recall.md) explains configuration and selection rules.
The [metrics](question-memory-2026-10-16.metrics.json) retain all evaluation source
rankings, calibration candidates, usage, revisions and private-artifact hashes.

## Retrieval evaluation

The fixture has 21 records and 35 queries. Four project scenes supply 16 calibration
queries, including unsupported requests. The evaluation set has five other
positive scenes (13 correlated query variants) and six no-answer queries: 19 queries
total. Both records in the conflicting Mira pair are relevant; retrieving both is
required for a full top-three hit. Relevance labels were authored with the fixture.

The fixed calibration objective is macro set-F1 over returned/relevant source sets,
with an empty/empty set scoring one. A fixed grid selected **0.45** using calibration
labels only; the stricter threshold wins a tie. Model, weights, relative cutoff,
fixture and calibration objective were fixed before evaluating the held-out set.
No evaluation failure was used to tune the selector after this run.

| Selector | Relevant source first | All relevant sources within 3 | Correct no-answer abstention | Irrelevant selected slots | Macro set-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous lexical selector | 1/13 | 8/13 | 1/6 | 33/42 | 0.279 |
| New lexical/status filtering | 9/13 | 9/13 | 4/6 | 15/25 | 0.568 |
| Hybrid, calibrated threshold 0.45 | 10/13 | 10/13 | 5/6 | 1/12 | 0.789 |
| Hybrid, runtime default 0.70 | 5/13 | 5/13 | 6/6 | 0/6 | 0.579 |

The previous comparator intentionally cannot use the fixture's new superseded and
expiry annotations. Many of its first hits are old records stored later. Thus the
large 1→9 improvement is not a semantic-model result: it primarily demonstrates
the value of explicit evidence status plus removal of function-word-only matches.
Annotations are supplied by the fixture/trusted caller, not discovered by the model.

The calibrated semantic increment is 9→10 first hits and 15→1 irrelevant slots
relative to the new lexical path. Its remaining evaluation misses are Japanese
questions for Sable, Lumen and Atlas. Vega's unsupported lunch-menu query incorrectly
selects the database-backup record. These failures remain in the report.

The predeclared runtime default 0.70 rejects all irrelevant candidates here but
also rejects many useful sources. It is not the calibrated operating point and
must not be described as a universal improvement in recall. Semantic indexing
remains opt-in, and changing the runtime threshold is explicit. A lower threshold
should be calibrated against representative positive and negative cases, rather
than copied from this small pilot as a general default.

## Draft robustness and semantic cues

Three prior recall scenes were seeded once each, given a common exposure, and
forked before every question/draft variation. The draft rewrites are diagnostic
examples, not independently verified semantic equivalents. Counts below use the
rewritten draft; each column has three scenes.

| Recall path | Exact-question echo | Paraphrase | Motif-name-free |
| --- | ---: | ---: | ---: |
| Previous question+draft hash | 0/3 | 0/3 | 0/3 |
| Independent question hash | 3/3 | 0/3 | 0/3 |
| Semantic enabled, default 0.70 | 3/3 | 3/3 | 1/3 |
| Semantic enabled, calibrated 0.45 | 3/3 | 3/3 | 3/3 |

Question-only hashing preserves the exact-question recall across all three draft
conditions. It also removes the previous identical-draft assistance: with the
original draft, the old path recovered 1/3 paraphrases while the question-only
hash path recovers 0/3. Hashing does not learn semantic equivalence. The learned
index adds that separate capability. These three positive scenes do not measure
false recalls from unrelated latent seeds or establish answer-quality gains.

## API and model-path evidence

The embedding run used `text-embedding-3-small`, served under the same model name,
dimension 1,536: **one request, 781 reported input tokens**. Preparing the corpus
took about 1.78 seconds. Retrieval/calibration timing excludes that network stage.
The immutable vector checkpoint was replayed with zero provider calls, preserving
the calibrated retrieval summaries exactly.

Two live `gpt-6-luna` path smokes exercised actual stdio worker → RightBrainModel →
API calls: one detail request and one critic request. Both system prompts included
the final QA text, its original qid and unverified status. Both made one request,
finished with `stop`, and used 1,032 input / 168 output tokens in total. Detail used
the large 900-token budget; critic used an explicit 1,024-token cap. Continuation
was disabled. The detail response qualified the past 48-hour record; the critic
flagged a draft's unsupported claim of confirmed current seven-day retention.

These are representative path smokes, not independently graded full-controller
trials. Automated fake-client tests separately cover both detail/critic routes in
both worker implementations, identity ambiguity, gates, cache/dimension limits,
and metadata serialization. PostgreSQL adapter tests use a fake cursor; a running
database was not tested.

## Provenance and limits

The original embedding/ranking run used clean commit `a6cab6424430d30d5128ddcb054fbe87b7c73284`.
Frozen replay and the two live worker smokes used clean commit
`1436debb5f58af0c32b110c6cacd06e06d05e992`. That revision encodes a query once
per harvest and exposes the runtime-default comparison; the calibrated retrieval
summaries match the original run. Raw vectors, response receipts, request hashes and the
smoke runner remain under ignored `target/benchmarks/`, with hashes in the metrics.
Those public hashes identify artifacts but do not independently audit the raw
execution. No historic controlled-memory answer or rating was regraded.
The live smoke retains request hashes/capture flags, rather than full system prompts.

The next answer-quality study should separately assess current project facts,
action validity and evidence calibration. Retrieval hits, historical source
inclusion and two successful model paths do not substitute for those outcomes.
