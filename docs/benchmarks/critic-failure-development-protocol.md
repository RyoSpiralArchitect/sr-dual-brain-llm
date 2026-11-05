# Critic failure development controls

This model-free suite exercises the production `EngineSession`, controller,
Callosum, right worker, critic parser, and integration path. It addresses the
offline portion of Issue #87. It does not read or change the #85/#88 studies,
their all-stages-valid availability rule, or any historical answer grades.

## Fixed contract

`scripts/benchmark_critic_failures.py` defines 18 authored controls before execution:

| Boundary | Controls |
| --- | --- |
| Initial critic | clean; empty; length; filtered; missing termination; missing fields; scalar issue; transport error; unknown usage |
| Existing single repair | clean; empty; filtered; invalid schema |
| Verification critic | empty; invalid schema |
| Answer integration | transport error |
| Caller cancellation | initial critic; integration |

The same current, complete Cedar record, question, logical clock, and per-case
seed are used. Source approval is obtained through the actual answerability
backend using a scripted HTTP response. Generation may reproduce the authored
record only when its text is present in its actual input. Record claims remain
unverified. The scripted critic verdicts exercise control flow, not review quality.

The runtime uses explicit System2 ON, a round target of two, and fixed disabled
optional module controls. Micro fallback and sanity override are disabled for
these external-critic controls. The existing one-shot unstructured-text repair is
enabled. Auto-continuation is disabled. The total cap is 128 scripted transport
attempts, including support, drafts, repair, integration, verification, and recovery
turns. The runner has no live mode, model argument, or credential lookup.

## Independent observations

Each case retains these facts independently:

- Raw transport outcomes, completion status, and usage receipts for every attempt.
- Initial and repair validation status, effective critic provenance, and external
  versus local review validity. A failed verification is the latest review even
  when the first review was valid.
- Whether a response was returned; whether its bytes match a successfully completed
  provider response; whether it came from the draft, integration, or local fallback.
- Current source eligibility and actual source approval, with no factual-truth claim.
- Consultation draining, abandoned-turn publication, and reuse after cancellation.

The integration-error control can return local fallback text without a confirmed
provider completion for that final text. Both facts remain visible. An empty or
invalid critic can leave a completed, approved-record draft available while its
external review is unavailable. No aggregate score converts one into the other.

Fake responses normally carry fixed receipts of 10 input and 5 output tokens.
These are synthetic counts. Missing receipts, transport failures, and cancellation
retain unknown usage; the complete total stays null while known subtotals remain
available. Cancelled turns are followed by a fresh turn in the same session, and
that recovery's attempts are reported separately and included in the total budget.

## Exact replay

The raw journal preserves responses before production validation. This matters:
rejecting a malformed completion in the harness would hide which production
boundary accepted or rejected it. It binds the fixture, code, runtime controls,
and every ordered case/role/endpoint/payload occurrence. A mismatch remains fatal
even if a production fallback catches the immediate exception.

Replay serves only the recorded responses and error categories. Cancellation is
re-enacted at the same blocked boundary, then drained by the caller. Every receipt
must be consumed exactly once. Canonical comparison excludes only the number of
new scripted transport invocations, which is zero during replay.

```sh
python sr-dual-brain-llm/scripts/benchmark_critic_failures.py \
  --output-dir sr-dual-brain-llm/samples/module-gating-critic-development
python sr-dual-brain-llm/scripts/benchmark_critic_failures.py \
  --journal sr-dual-brain-llm/samples/module-gating-critic-development/journal.json \
  --output-dir sr-dual-brain-llm/samples/module-gating-critic-replay
pytest -q tests/test_critic_contract.py tests/test_benchmark_critic_failures.py
```

Raw artifacts are ignored private files. The report contains case IDs, status,
hashes, and receipt counts; the journal additionally contains scripted requests
and responses. These controls establish deterministic contracts and recovery
behavior. Any later live-provider confirmation requires a separately frozen
protocol and budget.
