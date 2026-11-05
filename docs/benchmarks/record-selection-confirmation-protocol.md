# Prospective record-selection confirmation after #92

Freeze this protocol, the fresh `record_selection_confirmation.json` fixture,
runtime fix, runner and controls in a clean commit before paid calls. The prior
#92 Larch ambiguity failure and its two failed expectations remain immutable.
This is an independent confirmation on fresh fictional inputs, not a rescore,
paired quality comparison or completion of #89's semantic-judge calibration.

## Fixed intervention and cases

Use the production `EngineSession` nonstream JSON path with `openai/gpt-6-luna`,
2,048 output tokens, no automatic continuation, unchanged runtime timeouts and
retries, and the existing all-AUTO profile with Director/executive advice and
metrics observation OFF. These are fresh in-memory source fixtures; #92's real
PostgreSQL results are separate. No new persistent-write claim is made here.

Run all **nine cases once in fixed fixture order**:

1. English whole-record ambiguity, explicit System2 ON.
2. Japanese whole-record ambiguity, AUTO.
3. Pronoun followup after declared focus, AUTO; focus setup makes no model call.
4. Read-only caller identity with two records and a global distractor, explicit
   System2 OFF (ordinary detail consultation remains allowed).
5. Two current primaries with one oversized/ineligible record and candidate cap 1.
6. Unique current whole-record quotation.
7. A unique current quotation plus a second expired record.
8. Ordinary factual lookup with two current same-project records.
9. A unique-source request with an extra instruction outside the bounded grammar.

For the first five cases, **before runtime**, ask the actual support backend once
about the prospectively listed `preapprove_qids` using the ordinary policy and
the exact runtime retrieval query. The ineligible case lists only the eligible
record. Retain every true/false/unknown receipt; no positive decision is forced,
selected from retries, or treated as an authority override. Runtime may reuse
only exact ordinary cache entries; unique literal replay still requires its own
typed full-source approval. All setup support calls count toward the total.

The intended runtime fix gates delivery independently of factual approval:
multiple current primary records in the resolved scope withhold QA and seed
evidence for bounded whole-record requests. They carry `clarify_record` through
both brains and integration. Count before eligibility, ranking and caps, and
recheck after awaits. Leave receipts, primary-write authority, ordinary factual
queries, extra-clause recognition and source completion/lifecycle guards intact.

## Criteria and reporting

Freeze each case's expected condition, intent, bounded-source count, source
delivery expectations and qualitative manual criterion. Inspect all answers;
retain gaps, unknowns and failures. For ambiguous cases, check that recorded-value
canaries occur in neither generation/review/integration packets nor final text.
The support model is deliberately allowed to see those source values. String
canaries are limited diagnostics, not general semantic leakage certification.

Record actual support-preparation decisions separately from delivery. Report
completion and answer availability independently, all controller critic attempts
separately from HTTP stage counts, and final System2 accounting consistency.
For unique quotations, compare exact UTF-8 bytes against the prospectively named
actual source and retain V3 observed source-validity guards. Ambiguous requests
have no unique equality oracle. All V3 semantic criteria stay **unknown / not
assessed**; no judge is called. Manual observations are unblinded assistant
inspection, not calibrated semantic grades or independent adjudication.

## Budget, failure and replay

Hard cap: **64 actual provider HTTP attempts total**, including support setup,
runtime checking, generation, detail, critique, repairs, retries and integration.
No warm-up, paid rescue, resampling or extra diagnostic allocation. Missing usage
stays unknown with known subtotals. A real transport/auth/quota failure closes
paid admission using #92's sticky raw journal; preserve denied local retries and
partial checkpoints. Infrastructure failure leaves an incomplete run.

Use the existing raw journal with exact ordered occurrences, endpoint hashes,
private raw responses, safe failure categories and no stored headers/credentials.
Returned empty/length/refusal envelopes reach production validation unchanged.
Replay with no transport object under the offline HTTP guard; consume every
occurrence and compare canonical non-timing outcomes, including failures and
support decisions. A mismatch is retained and does not authorize paid rescue or
post-output changes to the measured implementation or criteria.

```bash
python sr-dual-brain-llm/scripts/benchmark_record_selection.py \
  --model gpt-6-luna --max-calls 64 \
  --output-dir sr-dual-brain-llm/samples/module-gating-record-selection-2026-11-05-live
python sr-dual-brain-llm/scripts/run_offline_checks.py -- \
  python sr-dual-brain-llm/scripts/benchmark_record_selection.py \
  --journal sr-dual-brain-llm/samples/module-gating-record-selection-2026-11-05-live/journal.json \
  --max-calls 64 \
  --output-dir sr-dual-brain-llm/samples/module-gating-record-selection-2026-11-05-replay
```
