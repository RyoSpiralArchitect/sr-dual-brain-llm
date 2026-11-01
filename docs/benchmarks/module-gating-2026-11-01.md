# First TRN measurement — 2026-11-01

The first matched-input pilot returned and graded all **8/8 answers in both
profiles**, with post-draft provider attempts **22 → 19** (13.6% fewer). This is
an observed mechanism result on eight trials/five problem groups, not established
quality equivalence, end-to-end savings, or evidence to enable AUTO by default.
The unchanged compatibility defaults remain appropriate.

The [prospective protocol](module-gating-protocol.md), fixture and runner were
committed before calls. Measured revision:
`514af7aaaad3df0a5ea02cff34b5c58c10f9ffe8`. Production roles used `gpt-6-luna`;
blind grading used `gpt-6-sol`. The configured output ceiling was 2,048 tokens;
smaller production substage limits and existing deadlines were preserved.
No policy, fixture, production threshold or model output was tuned after measurement.

## Quality and attempts

Six initial drafts were generated once; one complete-replay and one arithmetic
fault control were authored. Both profiles received identical draft bytes,
questions, source snapshots and requested caller modes in fresh EngineSessions.
Source support was prepared anew, so an identical source snapshot does not imply
identical support decisions across draft generation and processing. Arm order
alternated. Scores used blind labels and exact-quote validation; both literal
trials also passed a mechanical complete-string equality check in both profiles.

| Stratum | Trials | Frozen draft all-pass | Legacy all-pass | AUTO all-pass | Legacy attempts | AUTO attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Natural model drafts | 6 | 5/6 | 6/6 | 6/6 | 17 | 15 |
| Complete-replay authored control | 1 | 1/1 | 1/1 | 1/1 | 2 | 1 |
| Authored arithmetic fault | 1 | 0/1 | 1/1 | 1/1 | 3 | 3 |
| All trials | 8 | 6/8 | 8/8 | 8/8 | 22 | 19 |

All eight pairs were jointly graded, with no AUTO-only task/grounding/clarification
regression and no malformed grade packet. Both profiles passed all three criteria
on every returned answer. The Vireo trials share one source/problem, and the two
arithmetic trials share another: these are **five problem groups**, not eight
independent quality replications. There are no repeated stochastic trials.

The apparent natural-draft improvement needs qualification. For the retention
comparison, initial support returned `supported=false`; the initial speech prompt
did not contain the source answer. Both processing arms later received
`supported=true` and repaired the missing values/difference. This is not an
isolated demonstration of a critic repairing a reasoning error from the same
approved context. The authored multiplication error (`400` instead of `391`)
was repaired in both profiles, but does not estimate natural arithmetic-error rates.

| Trial | Legacy attempts | AUTO attempts | Observation |
| --- | ---: | ---: | --- |
| Queue metaphors | 5 | 5 | Same two-sentence answer; Director cancelled in both. |
| Natural literal replay | 3 | 1 | Exact answer retained; AUTO omitted consultation. Legacy had a failed right response and a subsequent integration. |
| Natural multiplication | 1 | 1 | Correct frozen result retained after critique. |
| Retention comparison | 4 | 4 | Both supplied labeled 13/3-day periods and the 10-day difference. |
| Injected multiplication fault | 3 | 3 | Both corrected the result to `391`. |
| Complete-replay control | 2 | 1 | Exact answer retained; AUTO omitted consultation. |
| English greeting | 2 | 2 | Advice/metrics calls disappeared, but detail/integration calls replaced them. |
| Japanese thanks | 2 | 2 | Same replacement pattern; answer unchanged. |

## Failure and cost accounting

The entire experiment used **57 provider attempts**: 8 shared draft-generation
and source-support attempts, 41 processing attempts, and 8 judge attempts. The
96-attempt cap was not exhausted. No failed or completed cell was resampled.

Two Director attempts were cancelled at the existing pre-draft wait boundary,
one in each creative-task profile. No response/usage receipt was recovered for
those attempts, so complete billed token totals and exact monetary savings are
**unknown**, not zero. Across the whole experiment, known receipts contain
36,034 input and 6,943 output tokens, plus those two unknown-usage attempts.
Processing known-token lower bounds were 15,177/2,710 for legacy and
13,651/1,668 for AUTO (input/output). These lower bounds cannot establish paid savings.

The legacy natural-literal right call returned `finish_reason=length` with empty
visible content. Its known usage, failure and subsequent production fallback
were retained. The row's `ok` availability means a final user answer was returned;
it does not mean every provider stage was healthy. Quality grading includes the
returned production fallback answer rather than silently excluding the cell.

As a **descriptive post-run sensitivity check**, the six pairs without cancellation
or failed provider stages still passed 6/6 in both profiles and used **14 → 13**
attempts. In the four natural-draft pairs inside that healthy subset, attempts
were **9 → 9**. These selected subsets are not replacements for the primary
eight-trial denominator; they show that a broad natural-workload saving is unproven.
Draft generation was shared and fresh-session support was repeated, so all
reported processing deltas concern the frozen-draft mechanism, not full end-to-end work.

## Gate exposure and next work

All twelve gates were reached at least once; only five had a positive AUTO
admission in this cold-session sample. Admission is not proof of an executed
provider task or an effective routing change.

| Gate | AUTO reached | AUTO eligible | AUTO admitted |
| --- | ---: | ---: | ---: |
| Selective consultation | 8 | 3 | 2 |
| Final-answer ACC monitor | 8 | 3 | 0 |
| Cerebellar correction | 8 | 0 | 0 |
| DMN reflection injection | 8 | 0 | 0 |
| Stalled System2 follow-up | 3 | 0 | 0 |
| Executive advice | 8 | 3 | 1 |
| Director advice | 8 | 3 | 1 |
| Executive metrics | 8 | 3 | 1 |
| Direct ACC consultation promotion | 8 | 1 | 1 |
| ACC System2 promotion | 8 | 0 | 0 |
| ACC temperature adjustment | 8 | 0 | 0 |
| Basal consultation promotion | 8 | 0 | 0 |

The admitted ACC consultation promotion occurred in the injected-fault trial
where System2 already required consultation; it did not demonstrate an extra
beneficial intervention. Temperature-drop amount was zero, DMN had no available
reflections, and verification did not stall. This pilot cannot establish efficacy
for the seven gates with no positive AUTO admission.

Priorities from the evidence:

1. **Preserve deterministic trivial-turn control when model work is inhibited.**
   With observer `both`, legacy enters the Director's free trivial-chat control
   (`consult=skip`). AUTO creates no Director task, and the pre-draft stage returns
   before that control. Under explicit left lead, ordinary policy consultation
   can then run. In both greetings, two Executive/metrics calls were replaced by
   one right-detail and one left-integration call. Declining a component did not
   reduce total work. Separate cheap control from optional provider execution,
   while preserving explicit System2/lead/director authority.
2. **Make optional-task admission aware of available time.** Both creative-task
   Director attempts missed the current wait boundary. Measure useful completion
   versus cancellation and unknown billing before changing timeouts or admitting
   more optional work.
3. **Expand controlled exposure.** Add paired warm-DMN, local cerebellar correction,
   bounded stalled-verification, positive temperature-drop and basal-loop cases.
   Freeze approved context as well as source/draft when isolating critic quality.

## Reproduction and artifacts

The complete journal was replayed with its live transport disabled: **0 new
provider calls**, all 57 recorded request events consumed, and zero differences
in status, final answer, frozen-draft identity, gate payloads or grades. Both
cancelled attempts and the invalid completion remained in replay; no fallback
network request was allowed. Replay latency is not a live latency measurement.

```bash
python sr-dual-brain-llm/scripts/benchmark_module_gating.py \
  --model gpt-6-luna --judge-model gpt-6-sol --max-calls 96 \
  --output-dir sr-dual-brain-llm/samples/module-gating-NEW-live
python sr-dual-brain-llm/scripts/benchmark_module_gating.py \
  --journal sr-dual-brain-llm/samples/module-gating-NEW-live/journal.json \
  --judge-model gpt-6-sol --max-calls 96 \
  --output-dir sr-dual-brain-llm/samples/module-gating-NEW-replay
```

Raw journals/reports remain private in ignored sample directories. The
[safe metrics projection](module-gating-2026-11-01.metrics.json) includes source,
fixture, runner, policy and endpoint fingerprints, per-row usage/health, blinded
verdicts without quotes, gate coverage and replay checks. It contains no auth
headers, raw prompts or complete responses. Existing local configuration is
needed to match a custom endpoint fingerprint during replay, but no live key is
sent while the journal is in replay mode.
