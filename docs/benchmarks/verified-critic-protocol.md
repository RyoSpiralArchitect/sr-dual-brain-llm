# Luna claim verification before integration

Test whether independent checking of each proposed correction preserves
repair and reduces regressions. Run three final generations from the same
natural draft using the native draft, external critic and integration
components of the system.

## Fixed design

- Freeze 12 authored tasks, three in each of Boolean logic, SQLite, event
  state and evidence authority. Each task requests one JSON object with
  four to six short answer fields. The oracle uses Boolean enumeration,
  SQLite execution or deterministic simulation. Its answers and rationale
  stay in the evaluator. Sample two natural drafts per task, retain all
  attempted drafts and run two blocks. Use seed 2713, two trials at a time.
  Each block includes each of the six final launch orders twice.
- Use OpenAI gpt-6-luna with provider-default effort for every call,
  4,096 completion tokens and a 24-second owner deadline. Use one request
  attempt per stage, no auto continuation, retry, repair, local fallback
  critic, preview, work product, verification round or controller
  intervention. Ordinary native integration can retain a complete draft
  on generation failure. Record delivered answers separately from calls.
- Request one atomic issue per answer field from the native critic on
  the shared draft. Validate exact field and current draft value, reject
  no-ops, missing values, malformed claims and conflicts, and remove
  duplicates. Both critic arms use that same post-parse list. Each final
  call receives the same question, complete draft text and system prompt.
  The control gets a neutral no-critique note. The standard arm gets all
  eligible proposed changes in a field-patch format. The verified arm
  gets the subset of eligible changes that pass verification, in the
  same format, or the neutral note. Proposal and verifier reasoning
  prose are retained privately.
- Make one extra verifier call in a fresh context if there are eligible
  proposals. Send only the original question with numbered lines and
  the requested field names. The verifier independently derives values
  and supplies exact quotations anchored to question lines. Admit a
  claim only if its typed replacement agrees with the independent
  answer and all evidence quotations match the claimed lines. Missing,
  ambiguous, incomplete and invalid verifier replies give no support to
  the affected claims. Agreement and exact source anchoring both count
  as signals. The verifier can still misderive a value.
- Each trial has five planned calls plus up to one verifier, for a
  maximum of 144 admissions. The three final requests launch in parallel
  after the verifier or its skip decision. All arms stay in the
  denominator when drafts, reviews or finals fail. A request contract,
  authentication or model error stops the run after its current wave.
  Freeze source and grading before network access. Keep raw receipts
  privately and recompute grades from the recorded decisions without
  provider access.

## Readout

Record strict exact JSON scores against the frozen oracle, type and value
matched per field, for all drafts and all three finals. Track complete but
wrong, malformed and unavailable answers separately. For each arm report
wrong-to-exact repair and loss of an exact draft (including format or
completion loss) using matching denominators. Compare matched outcomes by
task and report the subset where verification changed the final note,
alongside all attempted drafts. Record each eligible claim's verification
decision and compare its proposed and verified values with the oracle
offline. Report repair suggestions retained, harmful suggestions blocked
or admitted, and missing usage.

The predeclared success criterion for this cohort is at least as many
wrong-to-exact repairs as standard critique and strictly fewer exact
drafts lost, with both draft strata present and the study complete.
Two samples per task make this a descriptive check. The additional
verifier call changes total work, and eligible issues identify which
fields to verify. The audit checks exact quotations and independently
derived values. It does not mathematically prove that the evidence
entails the proposed correction.

Run from the frozen commit and verify the recorded scores:

    python sr-dual-brain-llm/scripts/benchmark_verified_critic.py --live /private/path/new-run
    python sr-dual-brain-llm/scripts/benchmark_verified_critic.py --score /private/path/new-run/journal.json
