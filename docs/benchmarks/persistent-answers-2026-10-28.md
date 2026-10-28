# Persistent final-answer pilot — 2026-10-28 (Asia/Tokyo)

**Status: initial development measured; repair implemented and regression-tested;
repaired live measurement and untouched evaluation are blocked by provider quota.**
No quality improvement from the repair is claimed. The provider currently returns
HTTP 429 with error type `insufficient_quota`, including in a direct diagnostic
using the real Python wall clock. No additional paid requests are being made.

## What actually ran

The [prospective protocol](persistent-answer-protocol.md), authored fixture,
runner and failure/grade integrity tests were committed at `767bb2f` before paid
calls. This adds real PostgreSQL persistence and actual fresh Python workers to
the earlier in-memory, controlled final-answer tests. It uses `EngineSession` and
`_handle_process`, not the HTTP Gateway transport. There are two spawned workers
per arm: four initial probes, process exit, then a final conditional query in a
fresh interpreter. The running application database is never used.

One development episode has three real source writes, including an explicit
correction. Its stored QA and episodic snapshot is cloned identically to both
arms, followed by 1,030 authored answer-only acknowledgements. Those background
rows are not 1,030 model generations. Source acquisition receives caller-provided
scope/write/correction metadata; it does not test automatic fact extraction.

The left-centered arm clamps consultation off, including salience overrides. The
collaborative arm uses production right consultation and forced System2 on. Both
retain normal EngineSession brain components; optional executive/director models
are off. Responses are nonstreaming for complete JSON transport journals. This
is a bounded collaboration configuration, not every optional LLM or default auto
routing. Actual telemetry shows five right calls for five collaborative probes;
no additional left revision call was needed in this initial sample.

Generation/support used `gpt-6-luna`; blinded shuffled-label grading used the
separate `gpt-6-sol` model. Both served names matched. The same-provider-family
judge is not human or independent-provider assessment. Five correlated probes in
one authored episode cannot establish general quality or production reliability.

## Initial development results

Task, grounding and clarification were independently graded against prospectively
frozen criteria. Missing a known fact can fail task while passing grounding.
An unknown fee for an explicitly named project must not cause an unnecessary
project-identification question. An ambiguous follow-up after comparing two
projects should request clarification.

| Arm | Available / graded | Task + grounding | All three criteria | Grounding failures | Clarification failures |
| --- | :---: | :---: | :---: | :---: | :---: |
| Left-centered | 5/5 | 4/5 | 4/5 | 0/5 | 1/5 |
| Collaborative | 5/5 | 4/5 | 4/5 | 0/5 | 1/5 |

Corrected conditional values, the comparison, ambiguous follow-up and explicit
post-restart query passed in both arms. The unknown-fee probe failed task and
clarification in both: the answers asked which instance of the explicitly named
project was intended, rather than acknowledging that its fee was unrecorded.
There were no invalid grader packets or unavailable initial answers.

Only five probe tasks underlie these ten arm answers. No gain from forced
collaboration is observed in this development sample; this is not evidence that
right consultation is generally ineffective on harder reasoning tasks.

| Probe-path cost (five probes) | Left-centered | Collaborative |
| --- | ---: | ---: |
| Provider calls, excluding shared sources/judges | 9 | 14 |
| Input tokens | 3,512 | 10,323 |
| Completion tokens | 933 | 1,594 |
| Median handler duration | 4.313 s | 7.884 s |

Both arms made four support calls; the collaborative arm added five right calls.
Durations cover `_handle_process`, including live requests and persistence, but
exclude interpreter startup and initial session hydration. Hydration is separately
recorded per worker. Fixed arm order, controlled benchmark wall clocks/UUIDs,
small samples and uncontrolled service load prevent a causal latency claim.
No currency savings or speedup is inferred.

## Repair and its current evidence

At `a0cbd26`, SharedMemory exposes bounded reference-resolution states independently
of approved QA. The controller appends identity-only guidance to rebuilt contexts,
including when no factual QA is supplied. It includes no project identifiers,
source text or fact values, does not approve a QA, and does not inflate factual
memory salience/gating counts. It distinguishes resolved, multiple explicitly
named, ambiguous and unresolved references. Lifecycle/alias ambiguity is rechecked.

The left and right paths now receive the same identity status. A resolved project
can have an unconfirmed requested fact; lack of approved evidence must not by
itself turn that project into an unknown reference. The guidance asks for the
missing fact's source, rather than re-identifying an already resolved project.

Regressions verify rejected fact approvals remain rejected, no factual canary
leaks into either real model path, known + explicit unknown comparisons are not
marked resolved, ambiguous aliases remain ambiguous, expired/superseded sources
cannot establish identity and legacy mode is unchanged. These are mechanism/prompt
delivery checks. **They do not establish repaired model answer quality.**

## Preserved failures and actual spending

The first repaired-development attempt stopped before its first answer after a
successful support response and a provider POST error. The original journal
recorded its type but not HTTP status. Two further explicit development attempts
also stopped before the first answer; safe telemetry identified HTTP 429. A model
list check returned HTTP 200, but that did not establish usable completion quota.
Three isolated support diagnostics include one success and two subsequent HTTP
429 responses with `insufficient_quota`, under frozen and real Python clocks.
No completed repaired answer was sampled and discarded for a more favorable one.

All failures and diagnostic artifacts are retained. There have been **40 provider
POST attempts**: 31 initial-development calls plus nine failed-development/diagnostic
attempts. Seven attempts lack token usage. The known lower bound is **19,318 input
and 4,640 completion tokens**; complete usage and currency cost are unknown.
New runs stop making provider POSTs after their first transport failure. Earlier
attempts predate that guard and each made two calls within its first failed turn.

The original 240-call allocation remains: baseline development 64, repaired
development 64, evaluation 112. Nine repaired-development attempts are consumed,
leaving **55** for an explicit resumed measurement. The baseline's unused allocation
is not used to resample answers. Evaluation has made zero calls and its two
prospective episodes (ten probes per arm) remain untouched.

The initial protocol incorrectly described 2,048 as a universal output cap.
Recorded payloads show the production support backend requested 4,096 throughout;
source/left/judge requests used 2,048, and observed right requests used 520. No
request cap was changed between initial and repaired attempts. This deviation is
explicit; historical journals and metadata have not been rewritten to conceal it.

## Replay and artifacts

The complete initial run replayed at its recorded runtime revision `767bb2f` with
**zero provider calls**, identical source QAs, final answers and grades, and complete
exact-request coverage. Whole source-snapshot hashes differ because episodic
annotations contain measured runtime values; that is not described as identical
state replay. Current production context guidance changes request hashes, so the
old journal must use its recorded revision. No subset/synthetic HTTP response was
substituted to make a new prompt appear to match an old request.

[Public metrics](persistent-answers-2026-10-28.metrics.json) include per-probe
verdicts, hashes, actual output caps, cost denominators, failures, allocations and
artifact hashes. Answers, sources, literal grading quotes and HTTP journals stay
private under ignored `target/benchmarks/`. No credential or DSN value is published.

After quota becomes usable, resume development with the frozen source snapshot
and at most 55 calls, then freeze production code before the untouched evaluation:

```bash
python sr-dual-brain-llm/scripts/benchmark_persistent_answers.py \
  --phase development --model gpt-6-luna --max-calls 55 \
  --sources target/benchmarks/persistent-answer-dev-before-2026-10-28/sources.json \
  --output-dir target/benchmarks/persistent-answer-dev-resumed
python sr-dual-brain-llm/scripts/benchmark_persistent_answers.py \
  --phase evaluation --model gpt-6-luna --max-calls 112 \
  --output-dir target/benchmarks/persistent-answer-evaluation
```

Both need `requirements-dev.txt`, `requirements-pg.txt`, provider configuration and
an isolated `DUALBRAIN_TEST_PG_DSN`. Fresh output directories are required. The
benchmark drops only its unique table prefix. Replay replaces `--model` with
`--journal PATH` at the matching recorded code revision; private journals are not
included in the repository.

Local validation of the repair and benchmark passed **738 Python tests**, including
**14 real PostgreSQL cases**, with no skips. **17 Gateway tests** passed (two
unsupported binary-on-stdio cases skipped), and **13 Node UI tests** passed.
The PostgreSQL runner regression exercises paired source snapshots, real durable
receipts, fresh spawned worker PIDs, exact zero-network replay and snapshot reuse.
Separate tests cover atomic grader validation, provider fallback rejection,
quota/cancellation diagnostics, immutable prior records and the first-failure
transport stop. The unused evaluation remains a required live completion step.

Code review identified a configuration escape: inherited embedding settings could
create an unwrapped provider client, including during frozen replay. The runner
now rejects both external embedding and automatic-support selectors before any
session is created, in the parent and spawned workers. It creates only its own
journaled support backend. Source/probe and replay rejection regressions make this
configuration boundary explicit. Both selectors are disabled in the observed
execution environment. This hardening makes no additional provider calls and does
not change the frozen fixture, rubric or successful request payloads.

A second review repair reuses explicit-unknown project detection for every
reference cardinality, including two known scopes plus an unknown third scope.
It resolves against the current active registry, so expiry during preparation
cannot leave a partially matched multi-project request marked fully resolved.
The authored live fixture and rubric are unchanged; this boundary is covered by
a new model-free lifecycle regression, with no additional provider requests.
