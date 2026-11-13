# Completion work products and critic routing — 2026-11-13

**The five fresh authored tasks produced the same semantic result in all four
conditions: 3/5 exact, 60% mean item accuracy, and no initial-to-final repair or
incomplete-to-complete recovery.** Both Boolean entailment tasks remained
unavailable in all eight cells; SQL, event replay and source selection were
exact in all twelve. All 30 available arm pairs tied. This is a descriptive
result on independent provider samples, not evidence of statistical parity or
an isolated causal effect.

The [prospective protocol](completion-mechanisms-protocol.md) and
[CA-corrected protocol](completion-mechanisms-certified-protocol.md),
[corrected fixture](../../sr-dual-brain-llm/examples/completion_mechanisms_certified.json),
[frozen manifest](completion-mechanisms-certified-manifest.json) and
[filtered metrics with private-artifact hashes](completion-mechanisms-2026-11-13.metrics.json)
form the public evidence. The committed fixtures expose the authored questions
and evaluator reference answers; those references were never sent to the model.
Raw provider payloads, model answers, reviews and journals remain in ignored
private `samples/task-goals-completion-*` paths. The two frozen protocols say
"raw questions" are private; that wording is inaccurate for their committed
fixtures and is corrected here without changing the measured protocols.

## Stopped first study and separate correction

The first study was frozen at `a970d0844db77e16011197d5f85f10ca4596e510`.
Its first `logic_a:off` admission failed during TLS with
`ClientConnectorCertificateError`, before any provider response. It stopped
after one request, no scored model output and one unknown usage receipt. Its
single strict replay failed with `ValueError` because that frozen adapter did
not support the certificate exception. The parent artifacts and replay failure
were retained unchanged; neither is counted in the 20-cell comparison.

Local diagnosis found an empty Python default CA store; `certifi` supplied 137
certificates. A distinct study on five newly seeded, never-sent questions was
frozen at `00d0eba9cefa00086504da92d26d5f8be6fad2ed` with the original
failure hashes locked in its manifest. It required the `certifi` bundle through
`SSL_CERT_FILE` before admission. No paid health check, answer-based case
selection, provider model switch or retry of the first questions occurred.

## Conditions and measured results

The corrected study requested and received `gpt-6-luna` for all 46 completed
provider responses. It used manual native System2 ON, explicit left lead,
task goals AUTO, completion replanning AUTO and the same balanced AUTO module
profile in all arms. Director/executive were OFF. Only the work-product and
completion-aware critic switches changed. Case/arm order was fixed in the
manifest; each arm independently sampled its initial candidate.

| Condition | Exact / 5 | Initial complete / 5 | Work products available / attempted | Valid external review / 5 | Workflow valid / 5 | Requests | Unknown usage calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Both OFF | 3 | 3 | 0 / 0 | 2 | 0 | 12 | 3 |
| Work product ON | 3 | 3 | 1 / 2 | 3 | 1 | 14 | 2 |
| Critic aware ON | 3 | 3 | 0 / 0 | 2 | 2 | 12 | 0 |
| Both ON | 3 | 3 | 0 / 2 | 1 | 1 | 13 | 0 |

Every logic draft hit its 4096-token generation cap with an empty terminal
candidate. Each arm attempted the pre-existing completion replan, but every
integration also ended empty at its cap. The work-product path actually ran
four times at cap 1024: one product was terminally complete, shape-valid and
handed to integration; three ended at the length cap and were unavailable.
The one available product did not recover a complete answer. After a product
admission, integration was capped at 3072 to respect the fixed ordinary-left
allocation, versus 4096 without it. Product validity here means shape and
terminal completion only; its derivation was not verified.

The completion-aware critic deferred initial review on the four incomplete
logic candidates in its two arms. Those cells made no critic request, and no
later complete candidate became available for review. This confirms the
intended routing on these inputs, without a quality gain. On the three
complete-answer domains, valid external review ranged from one to three
cells per arm. Some complete answers remained exact despite invalid or
cancelled review, so semantic grade and workflow validity are reported
separately. The five cancelled requests were critic calls in the two arms
without completion-aware routing. The observed difference in critic outcomes
does not isolate the switch from independently sampled candidates and time.

No complete initial answer changed its semantic score. There were no semantic
repairs, semantic regressions, or incomplete-to-complete recoveries. The 30
case-level pairs all tied. The Boolean tasks were at one authored level and
the other domains at one level each; this is not a calibrated difficulty or
general performance estimate.

## Resource and replay receipts

The corrected run completed all 20 cells with 51/80 allowed admissions and
102 ordered transport events; none was denied. Serialized request caps were
20 drafts at 4096, nine integrations (five at 4096 and four at 3072), four
work products at 1024, 17 critics and one critic repair at 2048. Transport
had 25 `stop` completions, 21 `length` completions and five cancellations.
Known usage is **at least** 105,398 input and 99,167 output tokens; totals
and billing are unknown because the five cancelled HTTP attempts have no
usage receipt. Reservations were never refunded.

The one network-forbidden replay at the same frozen source reproduced all
51/51 request occurrences, 102/102 transport events, grades, reservations,
usage and the canonical report hash
`3e53bbe152388b3a509d3e85a5c98b53a5c567b88c0dcdc554c0da28d6390a2f`.
Its blocked-HTTP-attempt counter is zero; the live and replay journals have
identical SHA-256. Private artifact byte hashes and per-cell filtered
diagnostics are in the [metrics file](completion-mechanisms-2026-11-13.metrics.json).

## Next hypothesis, untested

The completion bottleneck on these two Boolean tasks is the large internal
reasoning demand before any terminal candidate appears. A 1024-token product
often failed to terminate and reduced the subsequent integration allowance.
A future, separately frozen experiment could compare a shorter structured
per-goal note and a direct larger integration budget on new questions, with
actual product admission and end-to-end completion as separate outcomes.
This study does not justify changing defaults or claiming score improvement.
