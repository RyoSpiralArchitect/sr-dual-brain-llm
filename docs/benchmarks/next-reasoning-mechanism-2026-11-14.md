# Next reasoning mechanism after the critic-cap holdout

The [eight-case holdout](critic-cap-holdout-2026-11-14.md) returned
`winner: none`. Keep the ordinary and critic defaults unchanged. The
development advantage (3/5 versus 2/5 exact) did not persist: holdout was
6/8 versus 6/8. Raising only the critic cap from 2048 to 4096 did not
increase valid external reviews or workflow-valid cases (both 2/8 versus
the low-effort cap control). This rules out promoting that cap change from
these data; it does not rule out other reviewer configurations.

Three separable failure modes deserve the next tests:

1. Provider-default ordinary generation can use a 4096-token stage without
   a visible Boolean answer. Low ordinary effort makes an answer available,
   but both held-out Boolean answers were still inexact. Completion and
   correctness require separate gates.
2. Critics often return empty, malformed or deadline-limited reviews. The
   critic has a default 24-second local window, so a larger token ceiling
   alone does not guarantee a completed review. Measure **false `ok` on
   evaluator-known wrong candidates**, not JSON validity alone.
3. Duration-based replay changes an owner timeout into a returned
   `deadline_exhausted` result near a millisecond boundary. Before another
   live qualification, make the owner terminal event an explicit journal
   receipt and test replay under schedule jitter. Keep the current failed
   replay receipt as failed.

Recommended next experiment, in order:

- Add an opt-in critic reasoning-effort setting with provider default as
  the unchanged default. Test serialized role separation, cap enforcement,
  and reset of the effort context on exceptions.
- Freeze a **reviewer-only** fixture with balanced, evaluator-generated
  correct and single-error Boolean/SQL/event/evidence candidates. The
  reviewer sees the task and candidate, never the oracle label. Compare
  critic default versus low/medium effort at the same cap and 24-second
  owner window. Record valid review, wrong-candidate detection, false `ok`,
  false issue, abstention, latency, cancellation and unknown usage. Qualify
  a configuration only if wrong-candidate detection improves without more
  false `ok` or non-Boolean regression.
- Only after that gate, freeze a new end-to-end task set. Compare the
  qualified reviewer with the current baseline and low ordinary control at
  matched per-trial token ceilings. Require more exact answers, no scoped
  evidence/SQL/event harm, valid review and a fresh exact replay before
  considering a default change. A longer reviewer window should be a
  separate arm, because it changes latency rather than token allowance.

The candidate remains a research proposal. No new provider run, retry of
the holdout or default change is implied by this document.
