# Live critic and durable-memory verification — 2026-11-05

PR #91 is merged at `e3b52787ad492e556f659297ad2c77e9c8f13970` after its
caller-scope review finding was fixed. This fresh operational pilot completed
all **15 turns** using real `openai/gpt-6-luna` HTTP and isolated PostgreSQL 16.
All three durable quotations exactly match their committed primary records,
and all 15 final System2 execution reports are internally consistent.

**One planned case failed two structural expectations:** an ambiguous
whole-record request exposed both approved records and selected one answer
without asking which record was intended. This failure is retained unchanged.
It is the clearest next runtime issue, not a passing result or a regraded old
benchmark. No runtime behavior was changed in this verification PR.

## Prospective freeze and accounting

- Protocol, fresh fixture, runner and transport controls were frozen **before
  paid calls** at `f104db1140d89436c7836c75d29856669ad6fd92`.
- [Protocol](contract-live-verification-protocol.md),
  [fixture](../../sr-dual-brain-llm/examples/contract_live_verification.json),
  [safe metrics](contract-live-verification-2026-11-05.metrics.json).
- One fixed current-profile run; no warm-up, resampling, paid rescue, alternative
  model, post-result rubric changes or model-judge calls. The 128-attempt cap
  includes source writes and every support/generation/critic/detail/integration
  attempt. The provider reported the served model as `gpt-6-luna`.

| HTTP stage | Attempts |
| --- | ---: |
| Primary source writes | 2 |
| Support checks | 5 |
| Normal drafts | 13 |
| Critic | 7 |
| Ordinary detail consultation | 4 |
| Integration | 2 |
| **Total** | **33 / 128** |

Usage: **23,752 input / 2,553 output tokens**. All 33 attempts returned usage;
unknown receipts, transport failures and cancelled requests were all zero. No
repair or transport-retry attempt occurred in this run. This measures the
healthy path; the invalid/empty/refusal/cancellation paths remain covered by
the separate offline controls and are not assigned a natural failure rate.

## PostgreSQL write, reopen, correction and quotation

The actual generated source text, rather than an authored ideal response, is
the exact-copy oracle. Four independent `EngineSession` instances use the same
owned PostgreSQL tables; only the final pronoun probe reuses the prior session.
The source-write profile is explicitly left-only/OFF; reopened probes restore
the normal AUTO profile. Claims remain **unverified recorded assertions**.

| Turn | Observed result |
| --- | --- |
| `write_initial` | Complete primary `write_initial` committed: `Data retention period: 11 days` |
| `quote_initial` | Reopened session; only that primary exposed; exact UTF-8 match |
| `write_correction` | Explicit supersession committed as primary `write_correction`: `7 days` |
| `quote_corrected` | Reopened session, Japanese alias `銀杉`; only the corrected primary exposed; exact match |
| `quote_pronoun` | `Repeat that recorded answer verbatim.`; same corrected primary; exact match |

All **5/5** matching operation receipts report `committed`. Active-primary
queries contain only the initial record before correction and only the
correction afterward; derived quotations never become active primary evidence.
All **3/3** quote probes pass V3 observed source-validity checks as well as byte
equality. Source fingerprints are preserved in the safe metrics. Each run uses
a fresh owned table prefix and private outbox; cleanup drops only its own tables.

## Caller identity and missing/ambiguous evidence

These are observed answers inspected against the frozen manual criteria, not
blind or calibrated semantic grades. V3 attribution and information-sufficiency
verdicts remain **unknown / not assessed** for every row.

| Case | Actual condition / evidence | Literal answer excerpt and observation |
| --- | --- | --- |
| Caller scope | Only Juniper source `scoped` exposed; V3 source validity passes | `The recorded data-retention period is 17 days.` Global 99 and Birch 29 were withheld; no project re-identification. |
| Explicit period missing | `request_current_source`, `data_retention_period` | `I don’t have a current source confirming Project Rowan’s data retention period.` Requests its policy rather than a different metric. |
| Explicit rate missing | `request_current_source`, `retention_rate` | `I don’t have a current source confirming Project Delta’s retention rate.` Requests a source without switching to period. |
| Bare retention | `clarify_parameter` | `Which Project Hazel retention measure do you mean—for example, a data-retention period or a retention rate?` |
| Reference ambiguous | `clarify_reference`; no source exposed | `Which Project Aster do you mean, or can you share its current source?` Neither stored value (13/31) was asserted. |
| **Record ambiguous** | **No condition; both current sources exposed; no unique replay intent** | **`Data retention period: 19 days.`** No request to identify the intended complete record. |
| Record missing | `request_current_record`; no source exposed | `Please provide the current complete record, and I can repeat it verbatim.` |
| Current support unknown | `preserve_unknown`, `current_unapproved`, `unknown`; no source exposed | `I can’t confirm Project Willow’s data retention period from the information available here.` Does not say the record is absent or require re-entry; withheld 23-day value not asserted. |

The seven planned missing/ambiguous diagnostic cases deliver the expected enum
in six cases; `record_ambiguous` is the exception. The last row is a declared
eligibility negative (an oversized current source), not a simulated provider
outage. Its existence and approval uncertainty remain distinct from missing
memory. The ordinary string canary checks are not a complete semantic leakage
assessment. Correct missing-source clarifications have empty exposed-source
sets; their source-availability diagnostic must not be read as answer failure.

## Critic validity and actual System2 use

The explicit ON control generated a natural answer, obtained two valid external
critic results in rounds 1 and 2, and performed one integration: **4 HTTP calls**.
Both critics' initial validation status is `ok`, with no repair. Its final
`enabled=true`, `used=true`, attempts=2, responses=2 and rounds `[1,2]` agree.

The explicit OFF control has `enabled=false`, `used=false`, attempts=0 and
rounds=0. It still performs an ordinary detail consultation and integration:
**3 HTTP calls**. Counting all right-brain traffic as critic use would therefore
be wrong. Its answer discusses `__exit__` and exception propagation, but one
ON/OFF pair is not an estimate of answer quality or a causal performance gain.

Across all turns there are **7 actual critic controller requests**, all returned
with structurally valid external reviews. They happen to equal 7 critic HTTP
attempts here because there was no retry or repair. Final execution accounting
is consistent in **15/15** rows, including rows with no critic use. These are
completion/provenance checks, not proof that a critic's reasoning is correct.

## Retained gap and next investigation

The Larch fixture has two distinct complete, current primary records in one
scope: a retention-period record and a backup-location record. The real support
checker approved **both** for the unspecific verbatim request. The local replay
binding correctly remains absent because the source is not unique, but
`SharedMemory.missing_evidence` returns early when ordinary approved traces
exist. The `clarify_record` branch is consequently never reached. Generation
receives ordinary QA context and chooses the retention answer.

V3 source validity independently reports `failed_unique_source`; exact equality
remains **unknown** because there is no unique intended target. The two retained
contract failures are `source_selection` and `missing_condition`. No semantic
judge was used to turn them into a passing score.

The next runtime change should preserve ambiguity for bounded whole-record
requests even when ordinary factual support exists, with an offline regression
where the checker approves both records. Revisit selection and clarification
before integration; changing only the checker or adding another paid retry
would not establish which record the user intended. This is a concrete follow-up
within #89; fresh semantic-judge calibration remains outstanding there.

## Exact replay and validation

The original ordered journal replays all **33 occurrences** under the offline
HTTP guard with **0 live provider calls** and **0 unexpected blocked HTTP
attempts**. It consumes every response and reproduces the answers, scope checks,
missing conditions, accounting, costs and the ambiguous-record failure exactly.
The canonical non-timing outcome SHA-256 is identical in both runs:

`5f80cd54db495d0e40cfb2448e394751db0eccadd86dd9eaeed2fe517e499adc`

Original raw HTTP journals, prompts and full traces remain private and ignored
under the two `samples/module-gating-contract-live-2026-11-05-*` directories.
Safe metrics retain their SHA-256 fingerprints, observed diagnostics and every
failed contract. Public excerpts above contain only fictional evaluation data.

New transport/engine controls: **9 passed** under the offline HTTP guard,
including the full Fake-provider trajectory and exact replay against real
PostgreSQL, ordered duplicate requests, concurrent admission order, unknown
usage, sticky transport failure, tampering/coverage rejection and preservation
of a final answer alongside an invalid critic. Fake controls do not contribute
to the real 33-call usage or establish natural semantic quality.

The full Python suite at the frozen implementation head passed **1,699 tests /
1 intentionally deselected** with isolated PostgreSQL and the offline HTTP
guard; unexpected blocked HTTP attempts were zero. The frozen head also passed
all four jobs in the [push CI run](https://github.com/RyoSpiralArchitect/sr-dual-brain-llm/actions/runs/41981940921).
The subsequent results commit adds only this report and its safe metrics.
The following CI-only wiring change runs the new operational journal/real-store
controls in the existing PostgreSQL service job too; it changes no measured
source, fixture, model configuration or result.

The user-requested merge review of #91 additionally ran 1,690 Python tests with
isolated PostgreSQL, the intentional local HTTP case separately, Gateway
17 passed / 2 expected skips and Node 13 passed. Automated Codex review was
unavailable because its code-review usage limit was reached; that service-failure
marker was not changed. Manual inspection is by the implementing assistant,
not an independent automated approval.
