# Persistent final-answer pilot — 2026-10-28 to 2026-11-01 (Asia/Tokyo)

**Status: measured.** After quota recovery, development exposed and repaired a
second weakness in collaborative review. Production code was then frozen at
`40cd2c3b2cb7706cc9f9ecb7524abeb4403bf3eb` before the two original evaluation
episodes ran once, in full. Both arms scored **10/10 on task, grounding and
clarification**, with no unavailable answers or invalid grader packets. All
successful runs also reproduced their source QAs, answers and grades through
exact-request replay with **zero provider calls**.

This is a small authored pilot. The untouched evaluation has no pre-repair arm,
so its score is not a measured before/after gain. Neither equal scores nor the
observed higher consultation cost establish general right-brain effectiveness.

## Results, including the development regression

Task, grounding and clarification use separate prospectively authored criteria.
Omitting a recorded value can fail task while passing grounding. An unknown fee
for an explicitly named project should elicit missing-information uncertainty;
an ambiguous follow-up after comparing projects should request identification.
In these runs, task + grounding and all-three pass counts coincide.

| Stage | Runtime revision | Left-centered: all three | Collaborative: all three |
| --- | --- | :---: | :---: |
| Initial full development, 2026-10-28 | `767bb2f` | 4/5 | 4/5 |
| Full development after identity repair, 2026-11-01 | `2b39d43` | 5/5 | 3/5 |
| Attribution diagnostic: two failed development probes only | `40cd2c3` | 2/2 | 2/2 |
| Original untouched evaluation: two episodes | `40cd2c3` | 10/10 | 10/10 |

All answers were available and graded; grounding failures were zero in every
stage. The initial development run had one clarification failure per arm. All
later stages had zero clarification failures. No invalid grader packet occurred.
The two-probe diagnostic is **not a replacement full five-probe development
score**. The final attribution repair was not remeasured over all five development
probes; the complete 5/5 versus 3/5 result remains part of the evidence.

Initially, both arms asked which instance of the explicitly named project was
intended when its fee was unrecorded. Identity-only reference metadata fixed that
behavior. However, in the resumed run the collaborative critic rejected two
otherwise correct conditional answers solely because their historical source was
unverified. One draft already attributed its values to the conversation. Revision
then removed the requested values, and the critic accepted the resulting omission.
These failures occurred before and after a real worker restart.

The second repair clarifies that supplied records can be reported with source
attribution and appropriate uncertainty, without claiming independent verification.
A critic should request that caveat rather than discard supported values solely
for lack of external verification. It must still challenge unsupported values,
lost conditions, contradictions and unjustified certainty. Stored evidence status
remains unverified; support approvals, source lifecycle exclusions and factual
memory gating are unchanged. Neither repair adds benchmark-specific answer
instructions or prospectively authored reference values to production prompts.

## What actually ran

The [prospective protocol](persistent-answer-protocol.md), fixture, initial runner
and grade/failure-integrity tests were committed at `767bb2f` before paid calls.
Only development outputs informed repairs. The bounded diagnostic amendment and
its CLI guard were committed before diagnostic calls and before evaluation.
There was no tuning or resampling on evaluation answers.

Each episode has three real source writes, including an explicit correction.
The resulting QA and episodic snapshot is cloned identically to both arms,
followed by 1,030 authored answer-only acknowledgements. Those background rows
are not 1,030 model generations. Source metadata, aliases and correction IDs are
caller supplied; automatic fact extraction is not tested. Development after both
repairs reused the exact original source snapshot.

The benchmark uses real isolated PostgreSQL, production `EngineSession` and
`_handle_process`, durable outbox/commit receipts, and actual fresh Python workers.
It does not use the HTTP Gateway transport. Each arm/episode runs four probes,
exits, then starts a second interpreter for its final conditional probe. Every
successful probe has a matching committed receipt. There are four distinct worker
processes per arm across the two evaluation episodes. In the two-probe diagnostic,
the first and post-restart probes likewise run in separate interpreters.

Five probes per episode cover corrected conditional values, a named unknown fee,
a comparison, an ambiguous follow-up and explicit conditions after restart. The
evaluation includes retention conditions and a transfer case about concurrency
and a region/urgency exception. Two episodes yield ten correlated probes per arm,
not twenty independent tasks or a production-reliability estimate.

Left-centered means left lead, System2 off, consult policy skip and a routing clamp
against salience overrides. Collaborative means left lead and forced System2 on
with production right consultation. Both instantiate normal brain components;
optional executive/director models are off. Responses are nonstreaming for complete
transport journals. This measures a bounded configuration, not every optional LLM
or default auto-routing. Evaluation made ten right calls for ten collaborative
probes; no extra left revision was needed.

Generation/support used `gpt-6-luna`; shuffled-label blind grading used the separate
`gpt-6-sol` model. Served names matched in all successful runs. The same-provider-
family judge is not a human or independent-provider assessment. Required JSON
schema, complete labels/criteria and nonempty literal quotes for pass/fail are
validated atomically, without partial-score salvage or automatic retries.

## Evaluation cost and time

| Probe paths, ten probes per arm | Left-centered | Collaborative |
| --- | ---: | ---: |
| Provider calls, excluding shared sources/judges | 18 | 28 |
| Input tokens | 8,183 | 23,663 |
| Completion tokens | 2,097 | 3,411 |
| Median handler duration | 3.924 s | 6.094 s |

Each arm made eight support calls. Collaboration added ten right calls, with no
observed score gain in this sample. The full evaluation used 62 calls: six shared
source writes, 46 probe-path calls and ten judge calls; 41,436 input and 8,893
completion tokens. Calls are a transport count, not a currency price estimate.

Handler durations include live requests and persistence but exclude interpreter
startup and initial hydration, which is separately recorded. Fixed arm order,
small correlated samples, controlled benchmark wall clocks/UUIDs and uncontrolled
service load prevent causal latency or speedup claims. Development measurements
also occurred on different dates, without a provider-side seed or immutable model
snapshot. The source snapshot was controlled; the service was not.

The next useful experiment is selective consultation on tasks where the critic
has a realistic opportunity to help, with both quality and total cost measured.
These easy factual tasks do not justify always consulting the right worker.

## Failures, quota recovery and spending

Historical failed runs and diagnostics are retained. Before quota recovery, three
repaired-development attempts each failed before their first answer and used two
calls. Three isolated support diagnostics used three more calls: one success and
two HTTP 429 responses with provider type `insufficient_quota`, under frozen and
real Python clocks. A model-list HTTP 200 had not established completion quota.
The first failed journal recorded error type but not HTTP status. No completed
repaired answer was discarded to obtain a more favorable one.

After the user confirmed restored quota on 2026-11-01, requests resumed normally.
There were no provider errors in the resumed full development, targeted diagnostic
or evaluation. The runner now stops provider POSTs after the first transport
failure, including failures otherwise swallowed by a model fallback. It rejects
external embedding/automatic-support selectors before parent/worker sessions,
preventing journal/budget bypass and configuration drift, including during replay.

| Allocation | Allowed calls | Consumed |
| --- | ---: | ---: |
| Initial development | 64 | 31 |
| Repaired development, failures/diagnostics included | 64 | 9 + 34 + 12 = 55 |
| Evaluation, run once | 112 | 62 |
| Total | 240 | 148 |

Unused allocations were not borrowed for answer resampling. Seven historical
attempts lack token usage. Across all stages, known usage is a lower bound of
**95,104 input and 20,644 completion tokens**. Complete usage and currency cost
remain unknown; rejected calls are not assumed to have used zero tokens.

The initial protocol incorrectly described 2,048 as a universal output cap.
Actual unchanged requests use 2,048 for source/left/judge, 520 for right critique
and 4,096 for support. The cap disclosure is retained; historical journals were
not rewritten to conceal it. Public metrics list observed cap sets for every stage.

## Replay, artifacts and validation

Exact-request replay succeeded at the recorded runtime revision for each stage:
`767bb2f` initial development, `2b39d43` resumed full development, and `40cd2c3`
for attribution diagnosis and evaluation. Each replay made zero provider calls,
matched complete request coverage and reproduced source QAs, final answers and
grades. Source snapshot hashes match for the two runs that reuse original sources.
For initial development and evaluation, whole snapshot hashes differ because
regenerated episodic annotations include measured runtime values. This is not
claimed as identical whole-engine state replay. A changed prompt must not be
replayed against an older request journal as if it matched.

[Public metrics](persistent-answers-2026-10-28.metrics.json) include revisions,
fixture/policy/request artifact hashes, per-probe verdicts, answer hashes, role
usage, allocations and all stages, including failures. Sources, answers, literal
grading quotes and journals stay private under ignored `target/benchmarks/`.
No key, credential header or DSN value is published.

The actual resumed commands were run with requirements-dev/pg, provider configuration
and isolated `DUALBRAIN_TEST_PG_DSN`; only unique benchmark table prefixes are
removed. The application database is never used:

```bash
python sr-dual-brain-llm/scripts/benchmark_persistent_answers.py \
  --phase development --model gpt-6-luna --max-calls 55 \
  --sources target/benchmarks/persistent-answer-dev-before-2026-10-28/sources.json \
  --output-dir target/benchmarks/persistent-answer-dev-resumed-2026-11-01
python sr-dual-brain-llm/scripts/benchmark_persistent_answers.py \
  --phase development --model gpt-6-luna --max-calls 21 \
  --development-probes conditions cold-conditions \
  --sources target/benchmarks/persistent-answer-dev-before-2026-10-28/sources.json \
  --output-dir target/benchmarks/persistent-answer-dev-attribution-diagnostic-2026-11-01
python sr-dual-brain-llm/scripts/benchmark_persistent_answers.py \
  --phase evaluation --model gpt-6-luna --max-calls 112 \
  --output-dir target/benchmarks/persistent-answer-evaluation-2026-11-01
```

Fresh output directories are required. Replay replaces `--model` with `--journal
PATH` at the recorded revision, preserves the development subset and source-cache
arguments where used, and writes to a distinct replay directory. These completed
live runs should not be repeated to select better answers.

Local validation passed **744 Python tests**, including **15 real PostgreSQL
cases**, without skips, and **13 Node UI tests**. At measured revision `40cd2c3`,
all eight CI checks passed, including Gateway contracts, and Codex re-review
completed with no findings. Earlier local Gateway validation passed 17 tests with
two unsupported binary-on-stdio cases skipped. Regression coverage includes both
approved and rejected fact delivery to actual left/right prompts, unchanged
unverified status, alias/lifecycle exclusions, partial multi-project matches,
paired snapshots/receipts, fresh spawned replay, development-only selection,
atomic grading, safe diagnostics, immutable prior responses and transport stop.
