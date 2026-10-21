# Incremental evidence-support preparation — 2026-10-21 (UTC)

When one unchecked source joined a candidate set, support preparation previously
resent every candidate, including unchanged records with cached decisions. It now
sends only missing exact query/source decisions, reusing both positive and negative
results. Complete questions and answers retain their conditions and exceptions.
The feature uses the existing optional support checker; no new flag or dependency
is required. [Usage and limits](memory-answerability.md).

## Model-free request-volume experiment

Eight authored stages cover an initial four-record set, growth, reordering, an
explicitly revised record, a verification-status change, removal, a different
question and an exact repeat. A long Japanese policy ends with a regional exception.
The simulated checker returns authored independent decisions; it does not evaluate
natural-language understanding or final answers.

| Measurement | Previous whole-batch rule | Incremental |
| --- | ---: | ---: |
| Checker batches | 5 | 5 |
| Records sent | 22 | 10 |
| Serialized policy + request, UTF-8 bytes | 22,809 | 13,223 |
| Actual provider calls | 0 | 0 |

Records sent fall **54.5%**, and serialized policy/request bytes fall **42.0%**.
Approvals are identical at every stage. Growth, correction and status change each
send one record; reordering, removal and the exact repeat send none. A different
question checks all three remaining candidates. Both arms still make five checker
batches. Byte counts include the unchanged policy and compact JSON request, not
HTTP headers, model tokenization or responses. These are transmission-volume
measurements, not measured token savings, currency savings or service latency.

The reference adapter reproduces the previous all-candidates-on-any-miss rule for
this fault-free fixture. Its volumes and approvals were also checked against the
actual gate at integrated main commit `74d552b02fea8421ddb7e2284e03adf2af93fba9`.
The production change and experiment were committed at
`2d6eea8150ddcffb4a3a76676f96380970f5c993` before measurement. Public
[metrics](incremental-memory-support-2026-10-21.metrics.json) include fixture,
policy and artifact hashes.

## Failure handling and regression checks

Missing decisions are assigned request-local IDs. No new approval enters the
cache until every returned decision, boolean and supporting quote validates.
A failed or cancelled check retains earlier decisions and withholds unchecked
records. A model-identity change blocks old decisions too. A request-local
snapshot preserves valid cache hits if another asynchronous preparation evicts
them while the new check is pending; the shared LRU remains bounded.

Cache reuse does not override expiry, supersession, scope rules, pending correction
exclusions or complete-source limits. Text, question, identity, timestamp, expiry,
status or scope changes require their own exact key. Negative decisions are also
reused, so an unrelated record need not be repeatedly rejected.

Eight private frozen groups were verified with zero chat/embedding calls: the
conversation baseline, development trajectory, calibrated development regrading,
evaluation, and the previous shortlist/expanded/evaluation/precision retrieval
fixtures. The new conversation gate preserved answers, stored metadata and grades;
the four retrieval fixtures preserved selections and summaries. The historical
`checked` reference arm was held at the integrated main gate above. Applying the
new batching rule to that reference would create new support request shapes that
its strict historical HTTP journal cannot replay. No synthetic subset response
was substituted for a model response. Historical development grading uses its
archived diagnostic prompt; final quality uses the previously frozen calibrated
rubric. This validates replay compatibility with a fixed reference, not fresh
model judgments for newly shaped mixed-cache batches.

Local checks passed: **640 Python unit tests**, **5 gateway contract tests** and
**13 Node tests**. The ten isolated PostgreSQL tests were skipped locally and are
covered by the repository's dedicated database CI job. New regressions exercise
mixed positive/negative cache hits, complete late conditions, source changes,
batch validation failure, retries, model drift, concurrent eviction, cancellation
and lifecycle filtering.

## Reproduction and limits

This public mechanism experiment needs no credentials and makes no provider calls:

```bash
python sr-dual-brain-llm/scripts/benchmark_incremental_support.py \
  --output-dir target/benchmarks/incremental-support-local
```

The output directory must be new. The generated report includes the source
revision, policy/fixture hashes, stage decisions and volumes. Private historical
HTTP journals are not distributed with the repository.

This removes retransmission of already checked complete records when candidates
change. A first query still sends all eligible candidates, and different questions
still need independent checks. It does not shorten the long policies responsible
for the earlier pilot's increased support input tokens. That paid pilot is not
remeasured here; real token cost and final-answer quality under changed mixed
batches remain unmeasured.
