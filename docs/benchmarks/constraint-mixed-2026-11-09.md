# Mixed constraint benchmark — 2026-11-09 (Asia/Tokyo)

**No observed score gain.** The frozen model judge passed both conditions on all
ten pairs and all three criteria. The six independent JSON-value/transcription
probes also passed. No local contradiction was witnessed, so this run does not
demonstrate a repair benefit. One SQL critic was cancelled at its native owner
deadline; that pair remains in the full report and is outside the runtime-
qualified subset. The full 53-request, 106-event tape replays exactly with zero
provider calls and zero blocked HTTP attempts.

See the [prospective protocol](constraint-mixed-protocol.md),
[frozen fixture](../../sr-dual-brain-llm/examples/constraint_mixed_comparison.json)
and [filtered metrics](constraint-mixed-2026-11-09.metrics.json). Raw answers,
quotes and original/partial journals remain private in ignored samples.

## Conditions and provenance

The answering model was `gpt-6-luna`; the blind paired judge was `gpt-6-sol`.
Both used provider-default reasoning effort. The runtime used the same native
RightBrainPolicy, fresh matched in-memory sessions, balanced AUTO profile with
goals AUTO and Director/executive OFF, and fixed per-case manual System2 modes.
Generation/critic caps were 2048, ordinary detail caps 450/900, the shared left
allocation 8192, and continuation was disabled. The total limit was 96 HTTP
attempts, reserving ten judge slots before the 86-attempt runtime phase.

`witnesses` retained the production checker. `ablated` changed only its micro-
critic adapter entry to return an empty report inside the benchmark. Legacy
checks, ordinary critics, source/literal protection and routing stayed active.
Initial request fingerprints matched in **10/10 pairs**. Each condition still
sampled its own drafts and later responses: differences in downstream calls or
latency cannot be attributed solely to the detector.

All inputs, criteria, allocation, code and tests were committed before the sole
paid run at `fb416b4e5cbf36678cb18a38f8013472e222499f`. No question, reference,
budget, profile or grade was changed after output. The previous inverse failure
from #105 was not an evaluation input. These ten new synthetic questions and
references were authored by the implementing assistant, without human
calibration. The unchanged judge instructions and strict quote/JSON validator
come from the earlier task-goal studies; this is an uncalibrated model observation.

## Quality observations

The judge assessed goal coverage, reasoning soundness and relevant contribution
independently. A pass is not a verified source, verified goal completion or a
human-validated guarantee. Proposed code/SQL tests were not executed by this study.

| Criterion | Ablated passes / all pairs | Witnesses passes / all pairs | Qualified passes, each condition | Uncertain / ungraded pairs |
| --- | ---: | ---: | ---: | ---: |
| Goal coverage | 10/10 | 10/10 | 9/9 | 0 / 0 |
| Reasoning soundness | 10/10 | 10/10 | 9/9 | 0 / 0 |
| Relevant contribution | 10/10 | 10/10 | 9/9 | 0 / 0 |

Every case is retained below. `P/P/P` means the three individual model criteria
passed, not a combined scalar score.

| Case | Domain | Ablated | Witnesses | Runtime qualified | Independent exact probe |
| --- | --- | --- | --- | --- | --- |
| `three_sensor_scope` | Numeric / withdrawn assumptions | P/P/P | P/P/P | Yes | Not assessed |
| `nonlinear_ambiguity` | Numeric / unsupported notation | P/P/P | P/P/P | Yes | Not assessed |
| `status_object` | JSON / supplied values | P/P/P | P/P/P | Yes | Both pass |
| `conditional_object` | JSON / inactive branch | P/P/P | P/P/P | Yes | Both pass |
| `dedup_call_boundary` | Code / mutable state | P/P/P | P/P/P | Yes | Not assessed |
| `cancellation_boundary` | Code / exception ownership | P/P/P | P/P/P | Yes | Not assessed |
| `latest_event_row` | SQL / row identity and NULL | P/P/P | P/P/P | **No** | Not assessed |
| `silent_monitor` | Evidence / missing observation | P/P/P | P/P/P | Yes | Not assessed |
| `customer_memo` | Prose / uncertainty and format | P/P/P | P/P/P | Yes | Not assessed |
| `literal_record` | Byte-exact source transcription | P/P/P | P/P/P | Yes | Both pass |

The JSON probes compared full objects, including their supplied string values,
and rejected duplicate keys. The literal probe required exact UTF-8 text. All
**6/6 probe outputs passed**; passing runtime key consistency alone would not have
passed a wrong-value probe. No exact semantic grade is invented for other tasks.

All 20 final ordinary generations were terminal-complete. Provisional goal review
statuses were 9 valid, 2 invalid (both evidence answers), 1 unavailable (the timed-
out SQL critic), 2 not reviewed (manual OFF prose), and 6 no-plan. Valid collaborator
reports remain provisional: `verified_complete` was never promoted to true.

## Detector exposure and limitations

The witness condition observed **zero contradiction snapshots**, hence no local
micro correction. Its unconditional JSON record was locally consistent in two
checker invocations. The two numeric answers' concrete recovered points were
correct, but the affine checker returned unknown: the model used the natural
form **“For the observations …”**, while the supported condition grammar expects
bare bindings immediately after `For`/`when`/`given`. This is an observed coverage
gap, not evidence that the unknown answer was wrong. The scalar/tuple formatting
and scope issue must be considered together; the checker was not widened after
seeing these outputs.

Of the three negative-control cases, nonlinear and conditional JSON each invoked
the checker twice and remained unknown with no false constraint flag. Literal
transcription bypassed the checker and passed its exact probe. Thus this is two
exposed negative cases with no flag, plus a protected bypass; it is not a measured
general false-positive rate. Zero local interventions also means the tied quality
scores cannot establish that the correction path improves or preserves quality
when a genuine contradiction actually occurs.

The affine reference's illustrative pair both misses r=4, while “distinguishes
them” can be read more narrowly as requiring one example to match it. Both actual
answers included a matching point and a nonmatching point and satisfied that
reading. The fixture/grades were retained unchanged; future questions should
state explicitly whether both examples may be ruled out. More generally, the
small authored set and all-pass model judge have a ceiling and calibration limit.

## Availability, usage and native work

| Scope | HTTP attempts | Known input subtotal | Known output subtotal | Unknown-usage calls |
| --- | ---: | ---: | ---: | ---: |
| Ablated runtime | 20 | 21,442 | 7,663 | 0 |
| Witnesses runtime | 23 | 22,755 | 8,797 | 1 |
| Paired judges | 10 | 10,658 | 2,752 | 0 |
| Whole run | **53** | **54,855** | **19,212** | **1** |

There were 52 responses and 1 owner cancellation, with no transport errors or
denied admissions. The cancelled request was the witness condition's SQL critic:
one critic attempt, zero critic responses, an owning `TimeoutError`,
`review_completed=false`, `external_review_completed=false`, and
`system2.resolved=false`. Its initial answer was complete and the judge passed
that answer; the missing review was never converted into successful verification.
Unknown usage for that cancelled HTTP remains unknown, so the true total input,
output and billing cost cannot be computed from the known subtotals.

Roles: 20 drafts, 17 critics, 2 integrations, 4 ordinary details, 10 judges. The extra
runtime work in the witness condition was an ordinary evidence revision/verify
and a literal-turn ordinary integration. There was no local witness triggering
those requests. Manual System2 OFF still permits existing ordinary detail work.
Summed runtime handler durations were 92,451.5 ms and 112,668.9 ms respectively;
these independent samples are descriptive, not a cost/latency effect estimate.

The SQL event is relevant to [critic availability #87](https://github.com/RyoSpiralArchitect/sr-dual-brain-llm/issues/87),
but differs from its original empty-stop envelope: this request had an actual
owner cancellation, not a completed empty response. Keep the distinction, bounded
deadlines and unknown usage when investigating availability.

## Replay and next measurement

The unchanged frozen source replayed **53/53 request occurrences** and **106/106
transport events** with 0 provider requests and 0 unexpected blocked HTTP attempts.
All non-timing rows, detector observations, exact probes, judgments, usage and
domain comparisons matched. The recorded cancellation was not injected from the
tape; the actual owner had to cancel it under the existing runtime contract.

Hashes:

| Artifact | SHA-256 |
| --- | --- |
| Code | `1cfabe4cb4ac57ac43efcfb99eff1d0f0bf18c45cc7d9fb4862d30702ddd40aa` |
| Fixture | `33e84487ee49de9064834012fb592efdfae6d7ae9de7472ae7850644170042be` |
| Protocol | `77992ef610d2ceccdef1305cd2a3fa24f16d7b1bd666100bb2ff870502f18b50` |
| Judge instructions | `b613be7b2762bb199688ef095c7eeb394d249f104c1734b425e3cda47918aa97` |
| Non-timing canonical report | `1b2c356d22a2b4e838a04f7885dd5307ad60eda893640f7d026dbaa087fe439b` |
| Transport events | `609f36c353c3725c701009921b367ef8a20ca12df2e165bec5f28ff9a851d962` |

Before widening extraction, build a bounded, labeled cross-domain development
corpus of genuine contradictions, correct claims, alternative surface forms,
withdrawn conditions, inactive branches and literal data. Separate extraction
coverage, witnessed precision, repair success and final answer quality; passing
local checks must not supply overall confidence. This run's outputs now belong to
development data. Another paid quality test needs independently frozen fresh
questions and criteria, with no current-fixture-only parser branch or score
recalculation. Investigate critic availability separately; defaults remain unchanged.
