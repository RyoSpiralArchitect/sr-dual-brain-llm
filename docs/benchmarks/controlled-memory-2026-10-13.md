# Controlled memory pilot — 2026-10-13

Correct synthetic history supplied 47 of 48 required factual items across four
questions and four repetitions. No-memory and unrelated-history answers supplied
none of those items, while appropriately acknowledging missing project facts.
This measures filling an information gap; it does not demonstrate 47 repaired
false claims or a benefit from the full controller.

Generation used `gpt-6-luna` (64 calls, all `finish_reason=stop`, 1,024-token cap,
40-second timeout, no continuation). Evaluation used `gpt-6-sol` on 16 packets
with randomized answer labels, reference facts and visible historical context.
One judge output had malformed JSON despite a normal finish. An explicit retry
retained the other 15 grades and retried only that packet, preserving the failed
output: 17 judge calls total. All packets passed structural validation before
aggregation. Model grades have not been adjudicated by a human.

The [protocol](controlled-memory-protocol.md) describes the fixed interventions,
balanced condition order, fresh clients, provenance checks and scoring. The
[metrics file](controlled-memory-2026-10-13.metrics.json) contains aggregate
results, model configuration, source checksums and private-artifact checksums.

## Answer results

Each condition has 16 answers, 48 factual-coverage grades and 16 grounding grades.
These are correlated case/repetition/criterion units from **four unique questions**.

| Historical context | Factual items passed | Grounding passed | Coverage improved over baseline | New failures on baseline passes |
| --- | ---: | ---: | ---: | ---: |
| None | 0/48 | 16/16 | 0/48 | 0/16 |
| Correct current facts | 47/48 | 16/16 | 47/48 | 0/16 |
| Dated erroneous facts | 4/48 | 16/16 | 4/48 | 0/16 |
| Unrelated content | 0/48 | 16/16 | 0/48 | 0/16 |

No criteria were uncertain/unavailable and no answers failed completion checks.
The one failed correct-history item was Orion's production guardrails on repetition
3: the answer stated that canaries should be bounded and reversible, but did not
explicitly reject globally disabling cache without bounds. The judge applied the
explicit fixture requirement. The dated-history answers qualified their evidence;
zero grounding failures in this small fixture is not evidence of general immunity
to stale or malicious memories.

| Historical context | Input tokens, total | Output tokens, total | Mean model-call latency |
| --- | ---: | ---: | ---: |
| None | 1,984 | 3,435 | 2,906.0 ms |
| Correct | 3,272 | 5,298 | 3,636.0 ms |
| Dated erroneous | 2,824 | 6,330 | 4,632.9 ms |
| Unrelated | 2,440 | 3,599 | 2,862.0 ms |

Usage comes from provider receipts and includes reasoning tokens where reported.
Latency is the isolated generation call, excluding retrieval and judging. These
descriptive means do not establish a full-system speed comparison.

## Recall diagnostics

Each of three scenes was seeded once, given one common exposure, and copied before
every probe. Question variants shared the same draft and initial cache/sampler
state. Recovery required the original seed ID.

| Probe | Recovered probes | Unique scenes recovered |
| --- | ---: | ---: |
| Exact question echo | 12/12 | 3/3 |
| Paraphrase | 4/12 | 1/3 |
| Motif-name-free question | 4/12 | 1/3 |

These use the deterministic hash embedder, dimension 128 and seed 17. Four
repetitions replay identical scenes and do not add independent evidence. The gap
supports investigating semantic retrieval next, with human-checked equivalent
questions and held-out scenes, before claiming a recall improvement.

## Evidence limits and provenance

The quality trial exercises actual `SharedMemory` selection/formatting followed
by an isolated model call; the recall trial exercises `UnconsciousField` only.
They do not test complete-controller routing, naturally accumulated memory,
adaptive feedback or PostgreSQL metadata persistence. Prompt inclusion and seed
IDs establish data flow, not internal model reliance.

Generation ran from a dirty working tree based on
`906c86dbc7e64f553f7a786a074ef4f0a8bb562f`. Relevant source files were captured
locally during the run; their checksums are in the metrics. Later judge retry and
validation changes are not part of that generation-source snapshot. It is not a
complete dependency or model-weight fingerprint.

Full answers, blind packets, reveal keys, original/retried judge outputs and source
copies remain private under ignored `target/benchmarks/`. Public aggregate counts
and hashes identify those files, but cannot independently audit the model's grades
without the raw evidence. The evaluator is a separate model from the same family,
and visible contexts limit blinding. Repeated grades do not justify population
significance claims.
