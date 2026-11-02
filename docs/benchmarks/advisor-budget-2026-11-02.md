# Advisor time-budget pilot — 2026-11-02 (Asia/Tokyo)

Both comparison profiles passed **3/3** nontrivial paired cases on all three
criteria. Post-draft HTTP attempts were **16 untimed versus 8 timed**. Six
Director/metrics deadline timeouts became six inhibited tasks; concurrent
Executive advice still completed usefully in all three timed cells. This is
evidence for the admission mechanism under matched observed warm history, not
an estimate of natural traffic savings or a recommendation to change defaults.

The [prospective protocol](advisor-budget-protocol.md), three new questions,
production code and runner were committed at
`641a7ec5afb1b720fde45d8e318b4e3b03224c60` before any live requests. No threshold,
fixture or output was changed after measurement. Production roles used
`gpt-6-luna`; the blind judge used `gpt-6-sol`, both with 2,048 output tokens,
60-second provider timeout and no continuation. Safe hashes, role counts,
gate decisions and timing facts are in the
[metrics projection](advisor-budget-2026-11-02.metrics.json).

## Matched intervention and quality

Generate three natural drafts once. For each case, run one real warmup from its
frozen draft with all three advisors ON, then seed both fresh comparison sessions
with the same observed history at age zero. Ordinary expiry still applies.
Compare three advisors ON (`untimed_advisors`) versus AUTO v2 (`timed_advisors`),
keeping the other nine gates AUTO, caller modes, policy seeds and draft bytes
matched. Caller modes are explicit left lead, System2 auto/balanced, Executive
observe and observer both. Alternate arm order; retain all cells and failures.
No outer RPC deadline is injected. No memory records or support calls are used.

All nine baseline/comparison answers were available and blind graded with literal
quote validation; no grader failed. No paired criterion regressed.

| Case | Frozen draft | Untimed | Timed | Processing attempts, untimed → timed |
| --- | --- | --- | --- | --- |
| UDP loss/detection, exactly two sentences | All pass | All pass | All pass | 6 → 2 |
| Event-loop analogies, exactly three sentences | All pass | All pass | All pass | 5 → 3 |
| Payment retry checks, exactly three bullets | Task fail; grounding/clarification pass | All pass | All pass | 5 → 3 |
| Total all-criterion passes | 2/3 | 3/3 | 3/3 | 16 → 8 |

The payment draft repeated the original-status check and omitted bounded retry
eligibility/policy. Both production paths repaired it. These three cases have no
AUTO-unique quality failure, but are too few to establish general quality parity.
The judge is model-based; structural advisor usability is separately measured
and does not certify that its guidance is correct or causally improves an answer.

## Admission, completion and accounting

Every warmup measured a Director cancellation lower bound of 1,601–1,602 ms and
a metrics lower bound of 2,503 ms. These are **not full completion latencies**.
With real local windows of 1,600/2,500 ms and the fixed 50 ms reserve, all six
timed gates returned `insufficient_time_budget` before task creation. The same
facts were present in ON cells but did not override manual admission.

Warm Executive completions took 2,660–4,036 ms. Executive advice overlaps drafting
and consultation, so its later wait alone is not its total available window.
With no RPC deadline in this study, its budget remained unknown and AUTO kept
the previous nontrivial-input admission. All six comparison Executive tasks
completed with structurally usable output; none was discarded due to an
incorrect assumption of a 1,200 ms launch budget.

| Role, across three comparison cells per profile | Untimed admitted / started | Untimed completed usable / timeout | Timed admitted / started | Timed completed usable / timeout |
| --- | --- | --- | --- | --- |
| Director | 3 / 3 | 0 / 3 | 0 / 0 | 0 / 0 |
| Executive advice | 3 / 3 | 3 / 0 | 3 / 3 | 3 / 0 |
| Executive metrics | 3 / 3 | 0 / 3 | 0 / 0 | 0 / 0 |

No other cancellation, task failure or incomplete provider response occurred in
comparison cells. The six timed omissions were recorded as `not_admitted`.
Task starts do not substitute for HTTP accounting: runtime marks started usage
unknown, while the journal provides the receipts below.

| Post-draft HTTP role | Untimed | Timed |
| --- | --- | --- |
| Director | 3 | 0 |
| Executive advice | 3 | 3 |
| Executive metrics | 3 | 0 |
| Right consultation/verification | 4 | 3 |
| Left integration/repair | 3 | 2 |
| Total | 16 | 8 |

Six fewer requests directly correspond to inhibited advisors. The other two
come from UDP's differing critic outcome: untimed used two System2 rounds,
timed one. Provider responses and task timing were not synchronized; do not
attribute that two-call difference to admission alone. Event-loop and retry
consultation/integration counts were unchanged. No replacement work increased
in these pairs; a deterministic nontrivial controller test also verifies equal
cooperative work and answers when only an unaffordable Director is inhibited.

| Study phase | HTTP attempts | Unknown-usage cancellations | Known input / output tokens |
| --- | --- | --- | --- |
| Shared natural draft generation | 3 | 0 | 641 / 277 |
| Three real warmups | 14 | 6 | 5,206 / 1,485 |
| Untimed processing | 16 | 6 | 7,195 / 2,225 |
| Timed processing | 8 | 0 | 5,214 / 1,903 |
| Shared blind judging | 3 | 0 | 2,554 / 1,253 |
| Whole experiment | **44** | **12** | **20,810 / 7,143** |

The prospective maximum was 96 attempts; no resampling or paid replay was used.
Only 32/44 attempts have known token receipts. The known-token columns exclude
unknown cancelled usage, so they are not complete billed totals. In particular,
untimed and whole-study total tokens remain unknown; no exact monetary saving
is claimed. The 14 warmup attempts are real overhead, not free training. Fresh
session setup, generation and judging also prevent interpreting 16 → 8 as
end-to-end savings. Each comparison has one observation, so elapsed times in the
JSON are descriptive, not a latency significance result.

## Reproduction and limits

Network-free replay consumed all **44** journal events with **0** new provider
requests. The nine baseline/comparison statuses, answers, all reached gate
decisions, frozen drafts, history inputs and grades matched exactly. Live timeout
events replay as immediate recorded cancellations: twelve lifecycle labels
(six warmup, six untimed) change from `timed_out` to `cancelled`. Stop reasons and
elapsed times are not reproduced. The replay therefore validates requests and
admission/answer paths, not real-time completion behavior. Original warm timing
facts are explicitly restored; recomputing them from cached responses would
invalidate this comparison.

Raw artifacts remain private under ignored
`sr-dual-brain-llm/samples/module-gating-advisor-budget-2026-11-02-live/`.
Use the frozen revision above to rerun or replay the original runner:

```bash
python sr-dual-brain-llm/scripts/benchmark_advisor_budget.py \
  --model gpt-6-luna --judge-model gpt-6-sol --max-calls 96 \
  --output-dir /tmp/advisor-budget-new-live
python sr-dual-brain-llm/scripts/benchmark_advisor_budget.py \
  --journal /path/to/original/journal.json --judge-model gpt-6-sol --max-calls 96 \
  --output-dir /tmp/advisor-budget-frozen-replay
```

This deliberately reseeded warm history does not measure cold-start frequency,
success after TTL expiry, variable provider recovery, or natural sequential
session economics. No positive fast Director/metrics AUTO completion was exposed;
deterministic tests cover known-fast admission and cold probe/timeout/expiry.
Executive admission under an actual constrained RPC deadline needs live coverage.
The next evaluation should follow natural same-session turns across cold, warm
and expired history, include fast and slow roles, and preserve complete failed
and cancelled usage accounting. Defaults remain unchanged.
