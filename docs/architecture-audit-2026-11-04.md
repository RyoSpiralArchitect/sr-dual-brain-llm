# Director and orchestration audit — 2026-11-04 (Asia/Tokyo)

This inspection follows #80's local Director prototype and #81's advice-grader
calibration. It covers the integration boundaries in `dual_brain`, Callosum,
turn lifecycle, Hypothalamus/TRN, advisor timing/delivery, shared-memory controls,
source completion and EngineSession metrics. Existing memory, Gateway and UI
suites provide broader regression coverage; this is not a claim that every
behavior or external provider has been exhaustively verified.

## Confirmed defect and correction

Callosum's local consultation timeout and the controller's outer consultation
timer still used `asyncio.wait_for`. On Python 3.11+, its local timeout can cancel
and uncancel the current RPC `_TurnTask`, spending that task's cancel-once guard.
A reproduced handler first timed out locally, then accepted user cancellation
without ending; the later RPC deadline also could not stop it. This is the same
family previously fixed for advisor timers, exposed here by checking the wider
consultation path.

Callosum now observes its existing response future with `asyncio.wait` and bounds
both slot and response wait by the original end-to-end deadline. It still removes
pending/queued state and cancels owned physical work through its existing cleanup
path. The controller creates an owned consultation child, observes it without
cancelling the parent, and cancels/drains the child on local timeout or RPC
abandonment. Slot restoration remains in `finally`; cleanup is awaited before
the next stateful request runs.

Six regression cases cover repeated local timeouts, slot before/after deadline,
subsequent explicit cancellation and outer deadline, asynchronous child cleanup
and serialized subsequent work. Direct channel cases also run in the minimal
Python 3.10 CI job. The full Python 3.12 suite exercises controller integration.
There are no remaining executable `asyncio.wait_for` calls in the core modules;
that search supports this specific review, not a universal cancellation proof.

## Director boundary checks

Hypothalamus projects only normalized authoritative facts. Explicit active,
capable System2 ON supports consultation preservation. Only an existing explicit
all-stream long-term restriction supports LT context drop; mixed relay flags,
fallback-to-long-term notes and absent memory remain neutral. Source-write intent
such as `remember=False` is not reinterpreted as a deletion or forgetting request.

The registered Director AUTO policy decides whether known local constraints can
replace optional remote work. The local subset is applied before actual drafting
with `local_facts` provenance, without external latency training. Manual ON and
default requests keep the remote contract; OFF generates no new local plan.
Existing trivial-turn controls and mandatory explicit System2/source/evidence
rules retain their separate authority. Unknown answer length, temperature and
clarification needs remain absent, rather than invented controls.

Local plans add no output clamp or appended question. This matters for primary
memory writes: their exact-answer completion guard can deliberately reject a
postprocessed answer that no longer matches a confirmed completion. Memory
suppression filters context; it does not perform deletion, persistence or a
remote operation, and local advice must not claim those outcomes.

## Evaluation boundary checks

The new advice evaluator retains structured memo/control/directive/mix-in fields,
canonical quote paths, full selected-field context and explicit interpretation.
It does not turn every forbidden phrase into a positive assertion or automatically
excuse all prohibitions. Six independent paired calibration packets also cover
genuine unsupported assertions, irrelevant context, requested-count drift,
omitted controls and unknown outcomes. Original #78 grades stay unchanged.

The new natural sequence measures whole pre-draft-stage duration, ordering before
drafting, routed control use, local versus external exposure and all provider
attempts. Calibration shares the same prospective call budget. Historical
benchmarks pin their prior local-control/request behavior; source-pinned old
studies remain the reference for their original replay.

Final full-suite results, live limitations and calibration disagreements belong
in the measured local-Director report. Remaining semantic controls are explicitly
neutral when trusted facts do not establish them; future work should measure that
missing coverage rather than infer intent or extend deadlines.
