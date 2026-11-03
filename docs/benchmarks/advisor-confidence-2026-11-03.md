# Confirmed advisor recovery — 2026-11-03 (Asia/Tokyo)

Confirmation avoids one speculative timeout in the controlled jitter sequence
and restores the original cold opportunity after failed exploration. It also
misses one fast output in a deliberately unfavorable sequence. In the natural
live study, both arms pass all five tasks, but Director and metrics **never
complete within their windows**. No live confirmation occurs. Earlier retries
increase cancellations from **4 to 8** without adding usable advice. This is
evidence for bounded confidence behavior, not a live reliability or savings gain.

Code, complete independent fixture, parameters, schedules, judge contracts and
[prospective protocol](advisor-confidence-protocol.md) were frozen at
`9e9c3ac1f4a04765e84363d6636b6dcddf025e36` before paid calls. Nothing was tuned or
resampled after seeing outputs. The [safe metrics projection](advisor-confidence-2026-11-03.metrics.json)
retains every planned cell, original criterion verdict, role count and exposure,
including losses and unknown receipts. Earlier source-pinned studies stay intact.

## Admission behavior

Compare v3's single-success reset with v4 confirmation using one retained
production EngineSession per arm. Both use all twelve canonical module modes in
AUTO. The baseline's internal ledger disables confirmation; it preserves v3's
eager reset and normal TTL for failed probes. This is not a deployment switch.

Hypothalamus retains the timing facts, one provisional candidate and opaque
leases. TRN remains deterministic. A fast usable probe now requires one further
caller-requested eligible attempt before replacing old evidence. Two fresh
completions must satisfy:

`max(first, second) + abs(first - second) + 50 ms < claimed_window`.

Newer concurrent evidence must fit too. After promotion, the estimate includes
the range of recent completed latencies and all censored lower bounds. This is a
small conservative allowance, not a fitted distribution or confidence interval.
Unusable ordinary output after promotion suspends normal AUTO admission without
inventing a fast latency. Manual ON, OFF, caller modes and compatibility defaults
retain their authority; no AUTO default is promoted.

There is at most one confirmation per probe, claimed only at actual start.
Pre-start cancellation, model changes and insufficient windows consume nothing.
Original bounds remain: 15 seconds of quiet, one probe per retained role/model
profile per rolling 60 seconds, and unchanged native/RPC waits. Confirmation
requires at least the 1,600/2,500/2,000 ms Director/metrics/Executive window.
Executive's later wait is not its launch budget; only the outer RPC deadline
defines that window. Probes and confirmations create no background requests.

Failed exploratory observations keep their real timestamp but expire no later
than their pre-existing blocking witnesses. A provisional hold survives sample
ring eviction until that anchor. Failure therefore cannot renew the old cold
deadline. Successful promotion retains newer evidence and original completion
timestamps. Sample expiry does not erase an active probe rate limit; bounded LRU
eviction can discard a profile and restore normal unknown/cold behavior.

## Controlled uncertainty and opportunity cost

Five network-free interventions retain real production sessions and gates, with
logical service-time clocks, fixed answers and synthetic receipts. Only metrics
is caller-requested; Executive and Director are excluded, System2 is off and
consultation is skipped. Primary/support advance the clock by 30/2 ms. A metrics
service reaching its unchanged 2.5-second window is explicitly censored there.
These are logical cancellation interventions, not measurements of wall timers.

All arms start empty. The initial 2.5-second censor ends near 2.53 seconds;
opportunity starts at 19 seconds to satisfy the original quiet period. Subsequent
starts are 23, 27 and 63 seconds, except late recovery uses 23, 63 and 80 seconds.
No history is seeded or transplanted. The protocol specifies every service
fraction before measurement.

Values below are **v3 / v4**. Usable means structurally usable synthetic metrics
output; it does not establish content value or answer improvement.

| Regime | Whole-turn calls | Metrics usable | Metrics cancellations | First usable turn; logical finish |
| --- | --- | --- | --- | --- |
| Steady recovery | 14 / 14 | 4 / 4 | 1 / 1 | 2 / 2; 19.532 / 19.532 s |
| Near-window jitter | 13 / 13 | 2 / 3 | 2 / 1 | 2 / 2; 20.782 / 20.782 s |
| Lucky fast result | 12 / 13 | 1 / 2 | 2 / 2 | 2 / 2; 19.282 / 19.282 s |
| Missed fast opportunity | 14 / 13 | 4 / 3 | 1 / 1 | 2 / 2; 20.782 / 20.782 s |
| Recovery after failed probe | 12 / 13 | 1 / 2 | 2 / 2 | 5 / 4; 80.582 / 63.532 s |

Steady service at 500/550 ms confirms, yielding a 600 ms prediction and ordinary
admission. Under jitter, 1,750/2,250 ms completions yield a 2,750 ms prediction,
which fails the 2,500 ms window before adding reserve. v4 declines the next slow
attempt and resumes cold at 63 seconds; v3 makes that attempt, times out, and
remains inhibited. This exchanges one failed call for one completed call.

The lucky sample does not eliminate the next failed attempt: v4 spends its one
confirmation where v3 spends ordinary work. Preserving the original expiry then
allows an additional fast cold completion at 63 seconds. After an initially
failed probe, v4 first returns usable metrics **17.050 logical seconds and one
requested turn earlier**, at the cost of one extra attempt. v3's next successful
probe is at 80 seconds, outside the previous rolling 60-second interval.

The missed-opportunity sequence uses the same 1,750/2,250 ms pair but makes the
next service fast at 1,250 ms. v4 inhibits it and loses **one structurally usable
output** that v3 obtains. Both resume at 63 seconds. This is the measured cost of
the uncertainty allowance; reporting only avoided timeouts would hide it.

The following counts include every post-initial attempt, not just named probes.
Ordinary work includes cold resumption and subsequent work after promotion.

| Regime | v3 probe / confirmation / ordinary starts | v4 probe / confirmation / ordinary starts |
| --- | --- | --- |
| Steady | 1 / 0 / 3 | 1 / 1 / 2 |
| Jitter | 1 / 0 / 2 | 1 / 1 / 1 |
| Lucky | 1 / 0 / 1 | 1 / 1 / 1 |
| Missed opportunity | 1 / 0 / 3 | 1 / 1 / 1 |
| Late recovery | 2 / 0 / 0 | 1 / 0 / 2 |

Together, controls consume **131 synthetic attempts and zero paid calls**.
Primary/support work remains 5/4 per arm in every regime; there is no right or
integration/repair call. All 50 final answers are identical, and every cell drains
all advisor tasks. Synthetic fixed answers do not support quality or billing claims.

## Natural live sequence

Use `gpt-6-luna`, independently generated primary drafts and normal session state
updates. Matched caller settings are left lead, System2 auto/balanced, evaluative
dial, Executive observe and observer both, with real 120-second TurnDispatcher
caps. No records or PostgreSQL are seeded; record context is disabled, while
ordinary source/support work counts. Arms start empty, alternate order, and idle
**16.023 real seconds** after the initial pair. Phase names do not force latency.

| Phase / independent task | v3 calls | v4 calls | Director/metrics behavior |
| --- | --- | --- | --- |
| Initial: wall-clock adjustments | 5 | 5 | Both cold attempts time out in each arm |
| Opportunity: bulkheads and circuit breakers | 6 | 6 | Both recovery probes time out in each arm |
| Confirmation: cache miss and invalidation | 5 | 4 | Both roles inhibited; no candidate exists |
| Followup: pacing analogies | 4 | 6 | v4's original history expires; both cold retries time out |
| Release: retry amplification | 6 | 6 | v4 permits another probe per role after the rolling interval; both time out |
| Complete turns | **26** | **27** | **No successful probe or confirmation in either arm** |

v4 preserves a cold opportunity that v3's failed probe postpones. In this sample
the provider is still too slow: each restored attempt fails. The later v4 probes
also respect the rolling limit and fail. The sequence is retained rather than
repeated until a favorable recovery occurs. Its data does not validate the new
confirmation path with live successes; that path is exercised by the controlled
sessions and offline owned-task tests.

| Live advisor | v3 starts / usable / timeout | v4 starts / usable / timeout | Probe starts v3 / v4 |
| --- | --- | --- | --- |
| Director | 2 / 0 / 2 | 4 / 0 / 4 | 1 / 2 |
| Executive advice | 5 / 5 / 0 | 5 / 5 / 0 | 0 / 0 |
| Executive metrics | 2 / 0 / 2 | 4 / 0 / 4 | 1 / 2 |
| Total | **9 / 5 / 4** | **13 / 5 / 8** | **2 / 4** |

Executive first returns usable advice on turn 1 in both arms, taking
3,668.6/3,438.4 ms after task launch while overlapping other work. Director and
metrics have no usable output by turn 5; each session's final cell ends
102.669/105.923 seconds after its own initial start. The exact unobserved
completion times are unknown. Native Director and metrics waits remain 1,600
and 2,500 ms; timer scheduling produces slightly larger censored measurements.
Every admitted task starts, every cell has zero pending advisors, and source
counts advance from one to five in both sessions.

## Quality, content utility and full cost

Blind grading uses `gpt-6-sol`, shuffled labels, strict schemas and literal quotes.
All ten finals and ten primary drafts pass task, grounding and clarification:
**5/5 per arm**, with no grader failures or paired regression. Every format/content
obligation was explicitly requested in the new fixture; #72's unrequested-next-
check defect is not repeated. Passing drafts leave no measured final quality gain.

A separate blind judge evaluates actual completed external advice for relevance,
grounding and **potential_use**, excluding admission/latency/confidence labels.
All five Executive outputs in each arm pass all three criteria. There are no
Director or metrics outputs to grade: missing exposure is not a content pass or
failure. No malformed/uncertain advice grade occurs. A concrete requested-format
reminder can qualify; generic tone/privacy boilerplate alone cannot. These model
grades indicate potential use, not a causal contribution to the final answer.
Post-answer metrics could only supply monitoring value, never improve an answer
already returned; this study obtains none to assess.

| Whole-turn HTTP role | v3 | v4 |
| --- | --- | --- |
| Primary generation | 5 | 5 |
| Support from session sources | 4 | 4 |
| Right consultation/verification | 6 | 5 |
| Left integration/repair | 2 | 0 |
| Executive advice | 5 | 5 |
| Director | 2 | 4 |
| Executive metrics | 2 | 4 |
| Total | **26** | **27** |

Four more Director/metrics attempts are partly offset by three fewer other
consultation/integration calls. Independently sampled replies and differing
session histories prevent treating that aggregate difference as a causal saving
or a general cost estimate. Probe bounds do not cap all induced normal work.

| Study component | Attempts | Unknown cancellation receipts | Known input / output tokens |
| --- | --- | --- | --- |
| v3 complete turns | 26 | 4 | 16,411 / 3,492 |
| v4 complete turns | 27 | 8 | 13,916 / 3,324 |
| Blind final/draft judging | 5 | 0 | 4,608 / 2,739 |
| Blind advice judging | 5 | 0 | 4,651 / 1,177 |
| Whole study | **63** | **12** | **39,586 / 10,732** |

The frozen cap is 96 attempts, including all production/support/repair/judge work.
There is no warmup, resampling, continuation or paid replay. Only 51/63 attempts
have full token receipts. All twelve missing receipts are cancellations; no other
transport/auth/quota failure occurs. Both billed arm totals and the study total
remain unknown. Lower known tokens do not establish monetary savings.

## Replay, validation and next evidence

Replay at the frozen revision consumes all **63 recorded HTTP occurrences with
zero new requests**. All ten statuses, answers, drafts, canonical advisor outputs,
source/pending counts, role calls and usage occurrences match. All final/draft
and advice grades, full gate decisions and normalized resource read/claim/finish
occurrences match. Twelve real timeouts replay as immediate recorded
cancellations. Stop reasons, elapsed time and asynchronous completion order are
not reproduced; replay substitutes recorded facts and capabilities rather than
learning latency from instant cached responses.

Offline validation passes **1,214 Python tests**, including real isolated
PostgreSQL, plus **17 Gateway tests** with two expected skips and **13 Node UI
tests**. The 18 new confidence/benchmark cases include jitter, lucky samples,
late/missed recovery, pending expiry, unusable output, newer concurrent evidence,
bounded rings, duplicate claims, pre-start cancellation, profile changes,
manual modes, tight RPC limits and physical owned-task timeout/cancellation.
The overflow guard was additionally checked in the focused suite after the full
run. Production deadlines, cheap steering, Executive overlap and source/evidence
contracts retain their existing paths.

Raw prompts, answers, advice and journals remain in ignored
`sr-dual-brain-llm/samples/module-gating-advisor-confidence-2026-11-03-{steady,jitter,lucky,missed,late,live}/`;
the live directory contains `replay/`. Use the frozen revision to reproduce the
original request and rubric contracts:

```bash
python sr-dual-brain-llm/scripts/benchmark_advisor_confidence.py --controlled jitter \
  --max-calls 96 --output-dir /tmp/advisor-confidence-jitter
python sr-dual-brain-llm/scripts/benchmark_advisor_confidence.py --model gpt-6-luna \
  --judge-model gpt-6-sol --max-calls 96 --output-dir /tmp/advisor-confidence-live
python sr-dual-brain-llm/scripts/benchmark_advisor_confidence.py \
  --journal /path/to/original/journal.json --judge-model gpt-6-sol \
  --max-calls 96 --output-dir /tmp/advisor-confidence-replay
```

One correlated sequence per arm and synthetic interventions cannot establish
general latency or utility gains. The next useful work is to make Director and
metrics guidance deliverable within their native windows: prospectively compare
compact task-specific requests/output contracts and bounded deterministic
guidance, without extending waits or hiding failed calls. Measure content
exposure, omitted controls, time to usable guidance and full attempt cost on new
questions. Retain this sequence and its unfavorable retries as the baseline;
keep confirmation/default decisions separate from unproven downstream benefit.
