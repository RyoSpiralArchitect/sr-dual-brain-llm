# Bounded clarification routing and detail completion — 2026-11-06

In two fresh controlled comparisons, **AUTO removes one additional provider
request per clarification while preserving the exact final text**. Explicit
System2 ON still completes an external critic review; OFF still permits ordinary
detail without a critic. Empty/nonterminal detail handling is covered by injected
offline controls; no such failure occurred naturally in this live sample.

Code, inputs, criteria and runner were frozen before paid calls at
`ff0ad804fa088a1e147ba30a2725bf0fde7fafe2`, on merged PR #95
(`b1aac7c849c1033925cb04a68aa563305f3d43be`). See the
[prospective protocol](clarification-detail-protocol.md),
[fixture](../../sr-dual-brain-llm/examples/clarification_detail_confirmation.json),
[safe metrics](clarification-detail-2026-11-06.metrics.json) and
[runtime contract](../module-gating.md#ordinary-detail-completion).
Prior #95 reports, grades and journals remain unchanged.

## What the comparison establishes

The production left generator receives the fresh source-free memory condition
and PFC replay context, generating one natural draft for each case. Both outputs
match the prospectively defined English/Japanese choice clarification. That exact
draft is reused across fresh EngineSession arms; no model draft is resampled.

All evaluated arms use **fixed consultation pressure (policy action 1)** to ensure
positive exposure. Other registered modules retain the existing all-AUTO profile,
with Director/executive advice/observation OFF and balanced AUTO priority. This
is a controlled admission comparison, not a natural policy-frequency estimate or
an independent end-to-end quality comparison. Shared draft generation has a
separate cost; zero additional calls does not mean zero calls for the whole turn.

| Case / requested System2 | Selective module | Calls after shared draft | Actual critic | Final equals shared draft |
| --- | --- | ---: | ---: | --- |
| English, AUTO | OFF | 1 detail | 0 | yes |
| English, AUTO | AUTO | **0** | 0 | yes |
| English, ON | AUTO | 1 critic | 1 valid review | yes |
| English, OFF | AUTO | 1 detail | 0 | yes |
| Japanese, AUTO | OFF | 1 detail | 0 | yes |
| Japanese, AUTO | AUTO | **0** | 0 | yes |

English output:

> Please choose which complete record to quote from the current record choices.

Japanese output:

> 現在の候補から、再掲する記録をもう一度選んでください。

The English case retains `ambiguous_record / clarify_record`. The Japanese case
lists an eligible record, changes its body, then retains
`stale_selection / reselect_record`. Both AUTO arms propose and apply
`bounded_record_clarification`, changing prior action 1 to 0. No QA is exposed,
no source becomes approved and no replacement record is chosen. Predeclared
old/new/unselected value canaries are absent from every provider packet and final
answer. References remain outside provider payloads. These are literal checks,
not a universal semantic leakage claim.

All six source-free responses keep V3 source-validity **fail**
(`no_exposed_record`), with quotation equality not applicable. Attribution and
information-sufficiency grades remain **unknown / not assessed**. The equality
observation above concerns final versus shared draft, not a source quotation or
proof of semantic correctness. Inspection is unblinded by the implementing
assistant, with no paid judge or independent external adjudication.

## Cost, completion and replay

There are **6 / 24 actual provider HTTP attempts**: 2 shared drafts, 3 ordinary
details and 1 critic. Known usage is **6,893 input / 259 output tokens**, with no
unknown usage, transport failure, cancellation or denied admission. All six raw
envelopes end with `stop`; there is no repair, continuation, integration or direct
fallback in this live sample. Each ordinary detail repeats its case's shared
clarification rather than supplying additional information.

The two shared drafts cost 2 calls and 1,119 input / 39 output tokens. Within the
matched AUTO-mode comparisons, selective OFF adds 2 calls and 2,804 input / 62
output tokens in total; selective AUTO adds zero. The explicit ON/OFF controls
cost 2 further calls and 2,970 input / 158 output tokens. These sample-specific
counts do not estimate general savings or latency, and the total must not be
compared with the different six-case #95 study as a before/after improvement.

All **6/6** System2 execution reports agree with observed critic attempts, rounds
and latest validity. All three ordinary-detail deliveries separately report
`ok / external / usable / external_complete`, with no fallback. Delivery counts
are not billed HTTP counts. No natural empty/nonterminal live occurrence is
claimed as evidence for the new failure path.

Exact replay consumes all six ordered raw envelopes with **0 new provider calls**,
**0 unexpected blocked HTTP attempts**, and identical canonical outcomes,
including shared-draft identities/costs, final text, routing, source diagnostics,
detail status and execution accounting:

`db7727f5b2853a2a4bd0c7082f388edcb54ad3383ba785fcd717f8a3d0540a52`

Full journals, prompts and draft text stay private under ignored
`samples/module-gating-clarification-detail-2026-11-06-{live,replay}`. Public
metrics retain artifact SHA-256 hashes, complete cost accounting and separately
labeled post-run observations. No code, input or grading rule changed after
the live outputs.

## Failure controls and limits

The offline study injects empty `length` envelopes into all three ordinary-detail
deliveries and replays their exact JSON round-trip journals. Each is classified
`empty_response`, carries no usable notes, keeps the shared draft and makes one
provider attempt without the former repeated direct fallback. Separate real
client/worker tests cover empty terminal text, whitespace, nonempty nonterminal
text, provider failure, malformed/legacy output, cancellation and failed fallback.
A missing/error delivery can still use the existing single direct fallback;
that fallback cannot count as successful if its result is unusable. Configured
provider failures cannot become local heuristic success. Critic accounting stays
separate from these ordinary-detail observations.

The gate adds no general classifier: only the two fresh typed record-choice
conditions and exact fixed replies qualify. Additional claims, arbitrary
paraphrases, changed conditions, uncertain signals, protected modes and other
missing-evidence actions preserve normal routing. Default selective mode is
still OFF. Existing source approval, lifecycle, scope and persistence safeguards
remain mandatory.

Local verification at the frozen implementation: **1,887 Python tests passed /
1 intentional local-HTTP deselection**, including real isolated PostgreSQL and
the 34 new routing/completion/journal cases; zero unexpected blocked provider
requests. Gateway/local-HTTP checks pass **18 / 2 platform skips** and trace UI
checks pass **13**. PostgreSQL is integration coverage here, not a new live
persistence experiment. The broader semantic calibration in #89 remains open.
