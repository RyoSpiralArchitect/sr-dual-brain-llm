# Selective conversation consultation — 2026-11-01 (Asia/Tokyo)

**Status: measured, with one explicitly unavailable evaluation cell.**
Selective auto used 17 post-draft calls versus 20 in forced on, with both passing
7/8 trials. Existing auto passed all seven available trials and had one unavailable
cell; it also used 20 calls. This does not establish quality preservation versus
existing auto. The router remains opt-in and disabled by default.

## Motivation and behavior

The preceding [persistent-answer pilot](persistent-answers-2026-10-28.md) observed
identical 10/10 scores with 18 left-centered versus 28 forced-consultation probe
calls. That experiment did not measure ordinary auto routing. This experiment
explicitly compares existing auto, selective auto and forced System2 from the
same initial draft, with an unchanged-draft baseline.

`DUALBRAIN_CONVERSATION_SELECTIVE_CONSULT=1` enables a bounded post-draft router.
It is disabled by default, applies only to guarded left-led conversation memory
with a support gate in balanced auto mode, and makes no additional classifier
requests. A bounded lookup with one approved record, resolved identity, sufficient
numeric coverage, high heuristic confidence and no risk can skip consultation.
Explicit reasoning/review and new numeric claims request critique; missing,
ambiguous, multiple, qualitative or incomplete evidence retains normal routing.

Question length alone is insufficient evidence of review need: the existing
`medium` category includes questions with a question mark after 26 characters.
A supported lookup can supersede that surface signal and memory-driven
consultation. Hard questions and explicit controls remain protected. Source
status, support approvals and lifecycle checks never change. Explicit on/off,
right leads, precision preference, primary writes, director decisions and ACC
or amygdala overrides retain authority. Earlier automatic lead previews remain
outside this post-draft gate.

The numeric checks are estimates, not entailment or a proof of correctness.
Matching values cannot establish correct units, relationships or qualifications;
confidence is an existing heuristic, not a calibrated provider probability.
[Usage and control details](../selective-conversation-consultation.md) explain the
scope and final route telemetry. Wider evidence is required before default use.

## Prospective design

The [protocol](selective-consultation-protocol.md), fixture, implementation and
integrity tests were committed at `6defb14` before paid calls. Only two development
trials informed work. Review findings were repaired before evaluation:
comma-separated groups, leading-dot decimals and Unicode minus signs must preserve whole values and polarity;
unsupported/unparsed fragments must not silently disappear; missing token usage
must remain unknown rather than crashing final aggregation. The frozen original
development requests, routes, drafts, answers and grades exactly replay at `5bbfbd1`
with zero provider calls after these repairs. No new development answers were
sampled. Evaluation began at frozen production revision `5bbfbd1`. A runner-only
continuation at `3b999e7` completed unattempted cells after a returned empty detail
response; production core and engine code, fixture, prompts, caps and rubric were
unchanged. Its prefix hashes, call accounting and unavailable cell are retained.

The eight evaluation trials cover six distinct authored problems. Four trials
use one real model-generated initial draft each; four are paired synthetic
clean/faulty variants of two problems. Synthetic repair rates are not naturally
occurring quality gains. Version digits deliberately exercise identifier versus
arithmetic routing, so this stratified English fixture is not representative
traffic. Japanese paths have offline coverage, not live evidence in this pilot.

Model-origin drafts come from one clamped left-only production turn, with the
real support checker and complete memory context. All processing arms then
receive that exact initial draft and the same authored primary records. Actual
drafting is replaced with the frozen text; later support, critique, revision and
verification run through normal production components and live clients. Authored
synthetic drafts are public fixture inputs. Generated drafts/responses, grading
quotes and journals remain private under ignored `target/benchmarks/`.

Each arm has a fresh in-process EngineSession. This is not a new PostgreSQL or
restart experiment. Random/UUID/time sources are controlled; when Torch is present,
its newly initialized policy is seeded per trial, without a trained checkpoint.
This preserves the existing policy rather than claiming learned benefit prediction.
Trials are shuffled and processing-arm orders rotate/reverse. Execution positions
are approximately balanced, not repeated latency measurements.

All arms are explicitly left-led, executive/director off, conversation mode on,
System2 priority balanced and round-target minimum zero. Only the selective arm
enables the new flag. Forced on retains its production round policy. Shared
initial generation is counted once; unchanged draft has no processing calls.
Per-arm post-draft costs are not independent end-to-end production-turn costs.

Generation/support use `gpt-6-luna`; a separate blind `gpt-6-sol` judge scores
shuffled answer labels against prospectively authored task, grounding and
clarification criteria. Schema, all labels/criteria and literal answer quotes
are validated atomically. An invalid packet stays ungraded; there is no grading
retry or partial-score salvage. Same-family model grading is not human or
independent-provider assessment. Internal critic issue counts are not correctness.

## Development evidence

| Two development trials | Unchanged draft | Existing auto | Selective auto | Forced on |
| --- | :---: | :---: | :---: | :---: |
| All three criteria passed | 1/2 | 2/2 | 2/2 | 2/2 |
| Post-draft provider calls | 0 | 5 | 5 | 6 |
| Post-draft input tokens | 0 | 3,664 | 4,063 | 5,484 |
| Post-draft output tokens | 0 | 590 | 377 | 517 |

The one model-generated lookup was already correct. Selective used one support
call and no right call versus support plus right in existing auto. The synthetic
calculation was initially wrong. All processing arms repaired it; selective used
four calls versus existing auto's three, reflecting explicit critique/revision/
verification rather than a detail path. Overall development calls were equal,
and selective used more input tokens, so no aggregate development cost reduction
is claimed. There was no unavailable answer or invalid grader packet.

Including shared initial drafting/support and two judge calls, development used
20 provider calls, 15,625 input and 2,235 completion tokens. Its zero-call replay
reproduced the full request sequence. The unused development allocation is not
permission to resample answers for a better outcome.

## Evaluation

| Eight planned trials | Unchanged draft | Existing auto | Selective auto | Forced on |
| --- | :---: | :---: | :---: | :---: |
| Available / graded | 8/8 | 7/8 | 8/8 | 8/8 |
| All three criteria passed | 5 | 7 | 7 | 7 |
| Grounding failures among graded answers | 2 | 0 | 0 | 0 |
| Post-draft provider calls, failed call included | 0 | 20 | 17 | 20 |
| Post-draft input tokens | 0 | 15,988 | 13,539 | 17,809 |
| Post-draft output tokens | 0 | 1,300 | 1,430 | 1,629 |
| Median known handler duration | 0 | 3.509 s | 2.710 s | 3.705 s |
| Handler durations available | 8 | 7 | 8 | 8 |

Selective saved three calls (15%) relative to forced on, by skipping three
supported lookups. It deferred the unknown-fee case and consulted for the four
review/reasoning/drift cases. No graded processing-arm answer failed grounding
or clarification. The remaining failure was task completeness. There were no
invalid grader packets or uncertain grades. One existing-auto cell is unavailable,
not a wrong or passing answer. Handler timing for that interrupted cell was not
saved, and is not fabricated or silently treated as zero. These descriptive
medians have different availability and cannot establish a causal speedup.

| Stratum | Unchanged: passed / graded | Existing auto | Selective auto | Forced on |
| --- | :---: | :---: | :---: | :---: |
| Model initial drafts (four trials) | 4/4 | 3/3, one unavailable | 4/4 | 4/4 |
| Authored numerically correct drafts (two trials) | 1/2 | 2/2 | 1/2 | 1/2 |
| Authored faulty drafts (two trials) | 0/2 | 2/2 | 2/2 | 2/2 |

For the four model-origin trials, selective used six processing calls versus eight
in each other processing arm. Against forced on, this is matched 4/4 correctness
with two fewer calls in this tiny stratum. The model-generated initial drafts
were already correct, so no natural error repair is demonstrated. All processing
arms repaired the two authored faulty drafts, which is a controlled capability
observation, not a natural improvement rate.

The fixture's `synthetic_clean` label means authored numerically correct, not
independently established task-complete. The original expression in
`eval-clean-reasoning` had the correct result but did not explicitly label the
normal/urgent capacities. The frozen task criterion required those capacities.
Baseline, selective and forced-on received task=fail, grounding=pass and
clarification=pass. Existing auto expanded the capacities and passed. This is
consistent with a completeness gap rather than an arithmetic error; the judge
returns verdicts and literal quotes, not a detailed rationale. No rubric change,
regrading or new answer was used to overturn that result.

Among jointly graded trials, existing auto passed seven and selective passed six;
the remaining selective-only graded trial passed where existing auto was
unavailable. Therefore the aggregate call reduction does not justify a claim of
quality preservation versus existing auto. No initially passing graded draft was
harmed, but selective repaired only two of the three initially failing graded
drafts, whereas existing auto repaired all three. The feature stays disabled by
default. Further work should test coverage of explicit requested conditions,
not merely numeric correctness or a clean critic verdict, using new cases.

## Preserved empty completion and continuation

Evaluation stopped after 49 calls in `eval-explicit-review/legacy_auto`. The right
detail request had a 450-token cap and returned `finish_reason=stop`, empty content,
no refusal and three completion tokens. The frozen benchmark contract requires
nonempty completions, so it rejected the response and any fallback answer. The
product's optional detail path can accept an empty completion as no additive
content; this result is not proof of a production failure or intrinsic router
reliability difference.

The failed cell was never retried. Its original partial report, journal and
failure artifact remain immutable. Before further calls, the protocol amendment
and runner continuation were committed. Continuation verifies the same fixture,
model, runtime, router hash and unchanged production core/engine against the
prefix revision; requires an ungraded prefix; restores consumed calls/usage;
reuses shared drafts; and skips every attempted cell, including the failed one.
Only the remaining cells ran, without changing generation or grading conditions.

The continuation made 24 new calls, for **73 evaluation calls total**, including
the original 49. All 32 planned arm/trial cells are represented: 31 available and
graded, one unavailable. Eight grading packets ran once, with the unavailable
answer omitted from its packet and retained in the public denominator. The
interrupted cell cost two calls, 1,577 input and 78 completion tokens including
its support check. Its original full handler time is unknown.

The benchmark runner treats returned invalid completions as unavailable cells without a fallback or
retry; the next independent cell uses a fresh session. HTTP/quota/cancellation
failures still stop transport and cannot use this continuation path. Frozen
failure replay uses the actual occurrence of a repeated request key, so a later
successful occurrence cannot accidentally unblock the failed turn.

## Cost, replay and validation boundaries

This new experiment has separate allocations of 64 development + 160 evaluation
calls, at most 224. Actual consumption was **20 + 73 = 93 calls**, with
**72,506 input and 9,924 completion tokens**, all usage known. The unused 131 calls
were not used to resample answers. Evaluation included eight shared drafting/support
calls, 57 processing calls and eight judge calls. No currency cost is inferred.
Every POST is journaled by role, including shared drafting/support and grading.
Transport failures stop further POSTs. Returned unusable completions retain an
unavailable cell and permit later independent cells. Unknown usage remains unknown.
No automatic output continuations or transport retries are enabled. Production critic JSON-format repair, if invoked,
is a different counted request. Standard output caps are 2048 draft/left/judge,
520 right critique, 450 detail and 4096 support; format repair can request 900.
Actual observed right caps were 450 and 520 in both phases. Served generation/
support/draft models matched gpt-6-luna and judge matched gpt-6-sol.

Handler time includes processing requests and ordinary controller work, excluding
session construction. Initial draft creation is reported separately. The unchanged
arm's zero processing time is not end-to-end latency. Fixed small samples and
uncontrolled provider load mean timing is descriptive, not a causal speedup.

[Public metrics](selective-consultation-2026-11-01.metrics.json) include all strata,
per-trial statuses/verdicts, role costs, hashes, observed caps, continuation lineage
and original prefix artifact hashes. Development exactly replayed after parser
repairs with zero provider calls. The complete combined evaluation journal also
replayed with zero provider calls, reproducing every initial draft, final answer,
failed cell and grade, with full exact-request coverage. No synthetic success was
substituted for the empty response.

Actual evaluation commands, with the original allocation preserved:

```bash
python sr-dual-brain-llm/scripts/benchmark_selective_consultation.py \
  --phase evaluation --model gpt-6-luna --max-calls 160 \
  --output-dir target/benchmarks/selective-consultation-evaluation-2026-11-01
python sr-dual-brain-llm/scripts/benchmark_selective_consultation.py \
  --phase evaluation --model gpt-6-luna --max-calls 160 \
  --resume target/benchmarks/selective-consultation-evaluation-2026-11-01 \
  --output-dir target/benchmarks/selective-consultation-evaluation-continued-2026-11-01
```

Replay uses the completed combined journal with `--journal` in place of `--model`,
a fresh output directory, the same phase/budget, and no `--resume`. Complete model
replay is independent of the private original prefix directory. Neither the
completed trial plan nor its failed cell should be resampled for a better result.

Local validation passed **805 Python tests**, including **15 real PostgreSQL
cases**, with no skips. Focused routing/runner/failure checks passed 82 cases, including
same-draft identity, actual final routing telemetry, both complete and missing
provider usage, exact zero-network replay, protected explicit controls and numeric
format regressions, immutable failed-cell continuation and exact combined replay.
An additional 149 whole-value scalar checks passed. All eight CI checks passed at
`3b999e7`, and Codex re-review completed with no findings. The final report commit
changes documentation/metrics only and is checked separately before merge.
