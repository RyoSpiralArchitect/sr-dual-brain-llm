# Coordinated Luna verification and integration — 2026-11-28

The [frozen protocol](coordinated-critic-protocol.md) compares unfiltered
regeneration, verified regeneration, and a coordinated selector that shares
the verifier and the verified regeneration result. The verifier receives the
question and the names of every answer field, with each emitted value tied
to exact excerpts of the question. When all verified values match the
draft, the selector returns the original bytes. When the values differ and
every difference matches a critic proposal and the generated final, the
selector applies only those fields to the original draft. Otherwise it
uses the verified final unchanged.

Both cohorts were fixed before the first provider admission. The fixed
screen uses twelve authored mixed critiques and four exact controls. The
natural holdout uses twelve separate questions with native draft
generation and native critique. Each question was measured twice. All
final generation calls used 24 seconds and 4,096 completion tokens.
Coordination is a research selector for known scalar JSON fields.
Neither the answer key nor the expected values enter the provider prompts.

## Fixed screen

| Condition | Wrong draft → exact final | Exact draft → nonexact | Exact finals | Final generation calls |
| --- | ---: | ---: | ---: | ---: |
| Unfiltered regeneration | 18/24 | 1/8 | 25/32 | 32 |
| Verified regeneration | 16/24 | 2/8 | 22/32 | 32 |
| Coordinated | 16/24 | 1/8 | 23/32 | 29 |

Compared with verified regeneration, the coordinated selector added one
exact final, preserved all existing exact results, repaired the same
16 wrong drafts and avoided three final calls. It meets the predeclared
quality criterion for this fixed screen. Unfiltered regeneration still
produced two more exact finals than the coordinated selector. That paired
comparison has three coordinated-only wins and five unfiltered-only wins.

The single exact gain against verified regeneration was the second sample
of logic_partition_control. The verifier's values matched the original
draft on all six fields. Neutral regeneration changed two correct
classifications, and direct return preserved them. The first sample of
the same task returned an incorrect checked value and went to the
fallback; its verified final changed those two fields to incorrect values. This shows why
verification coverage and correctness need separate measurements.

There were 3 retained drafts, 11 checked commits and 18 fallback results
(11 incomplete checks, 6 uncovered discrepancies, and 1 integration
disagreement). All 22 committed field edits matched the oracle. The
verifier approved 30/48 correct suggestions and 1/62 damaging suggestions.
The damaging suggestion was excluded from checked commits.

Eight fully returned verifier responses had trailing data after the first
JSON object. Three other trials had mismatched citations. An after-run
inspection of just the eight JSON prefixes could recover usable evidence
for 44/46 fields. Six of those 44 values still disagreed with the oracle.
This inspection diagnoses formatting separately from derivation and
does not alter recorded results.

## Natural holdout

| Condition | Wrong draft → exact final | Exact draft → nonexact | Exact finals | Final generation calls |
| --- | ---: | ---: | ---: | ---: |
| Unfiltered regeneration | 5/6 | 2/18 | 21/24 | 24 |
| Verified regeneration | 5/6 | 2/18 | 21/24 | 24 |
| Coordinated | 5/6 | 0/18 | 23/24 | 15 |

The coordinated selector met the predeclared quality criterion on these
12 fresh tasks. Against verified regeneration, there were 21 both-exact
pairs, two coordinated-only exact pairs, zero verified-only exact pairs
and one neither-exact pair. Against unfiltered regeneration the
corresponding counts were 20, 3, 1 and 0. The three arms repaired the
same number of wrong drafts. Verified regeneration lost three originally
correct fields, and the coordinated selector lost none.

Both exact gains against verified regeneration came from preserving
fully corroborated drafts. In natural_source_delegation_paths sample 2,
regeneration changed a correct gamma value. In natural_logic_mirror_caps
sample 2, it changed two correct classifications. In both cases the
coordinated selector returned the exact original text. Nine drafts were
retained, two wrong drafts were repaired by checked commits, and thirteen
used the verified fallback. No damaging proposal was applied.

The one pair where only unfiltered regeneration was exact was
natural_event_expiring_allocator sample 1. Its critic included one useful
fix and one harmful edit. The verifier produced trailing characters
after JSON, so neither passed. The neutral verified regeneration left
that field wrong; unfiltered regeneration corrected it. This case and
the worse fixed result relative to unfiltered critique limit the
comparison. Neither study shows universal superiority.

## Work and verification

| Natural holdout route accounting | Model calls including draft and critic | Input tokens | Output tokens |
| --- | ---: | ---: | ---: |
| Unfiltered | 72 | 52,875 | 55,677 |
| Verified regeneration | 96 | 72,599 | 111,342 |
| Coordinated | 87 | 66,673 | 103,393 |

These totals reuse observed stage receipts to account for route costs.
Unfiltered regeneration has no verifier on its logical route. The
coordinated selector reduces work relative to always regenerating after
the same verifier, while using more work than unfiltered regeneration.
The nine avoided final calls in the natural cohort account for 5,926 input
and 7,949 output tokens. The two live counterfactual finals ran in
concurrent waves, so these results do not establish deployed latency.

All 216 physical requests returned within the configured limits and
included usage. Independent checks verified actual request caps, matching
drafts and final systems, cited excerpts, recorded route behavior and
grades. Both scored files recomputed byte for byte from the journals with
network access blocked. Both frozen fixtures regenerated byte for byte
from their independent oracles. Focused regressions passed 177 tests.

Both studies ran from a5ae0bfb771ffc06f642c0a5167def32a6fc0288. Raw
journals remain local; the scores and route metadata are public.

| Artifact | SHA-256 |
| --- | --- |
| Fixed fixture | dabce463a431a555a4360e2377a78d2dfdab444a18274fbb5f48d3110d0fc895 |
| Natural fixture | 39724d26ccaac1f095f8417cc4ee2b36cf9586e273416f7e6ea084e6f2a5f036 |
| Private fixed journal | ae52d17a28b0867259dc56e2aff46a0a87ecb63599ceabcb6182d3b2d09ae917 |
| Private natural journal | fcee7c7aa861ad24a19b89fd7e50b7344678ffdb923503fdfe91771e4094bef3 |
| [Fixed metrics](coordinated-critic-fixed-2026-11-28.metrics.json) | 2da8e56aad2b7ae162392b3572a7151b74232e7e7671b249236ee3a1303913c3 |
| [Natural metrics](coordinated-critic-natural-2026-11-28.metrics.json) | 2a9472d21cfd2196e7bd79cf3ac0f8f8751919dd2f15cc6c70cab5427ded558b |
