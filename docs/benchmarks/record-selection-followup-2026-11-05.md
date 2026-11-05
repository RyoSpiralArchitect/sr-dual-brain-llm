# Whole-record selection: controlled offline followup (2026-11-05)

## Result and scope

The same independently authored **23-case regression file** produced **16 failures
and 7 passes** against the unchanged PR #92 baseline, then **23 passes** against
the fixed implementation. Both arms used fixed Fake support responses and
recorded **zero unexpected blocked HTTP attempts**. No real provider or model
judge was called.

The fix preserves a recognized whole-record request when its exact source cannot
be bound. Ordinary support approval cannot choose the intended record. The
controller and delayed right-worker consumer retain that condition at existing
generation boundaries, including when a previously approved source becomes
ambiguous or expires.

This is evidence for deterministic selection, evidence handling and prompt
delivery. It is not a new live answer-quality study. The original PR #92 run,
its ambiguity failure, its 33 HTTP occurrences, fixtures, metrics and historical
grades remain unchanged. See the [original operational report](contract-live-verification-2026-11-05.md),
the [followup protocol](record-selection-followup-protocol.md), and the
[per-case safe metrics](record-selection-followup-2026-11-05.metrics.json).

## Provenance and fixed controls

- Baseline: `53498ef57e6ed41ec5f885154391b6e66e0ae16b`, the unchanged #92 head.
- Published implementation snapshot: `27e37f1f10485c3179699e2a219174f7dfdc0a1a`.
  Its tree, `fef72b3bf4f97cfc88b886a5634adf7b05886d61`, matches the locally tested
  tree exactly. This report and metrics are a subsequent documentation-only change.
- Fixed comparison: `tests/test_ambiguous_replay_contract.py`, SHA-256
  `8e4f6e0d0a1082348b0537d7bc6d4e7eb50778aee41486aa208a7d6833eae2cf`.
  The test bytes and JUnit case identity sets are identical in both arms.
- The baseline imports production modules from its own detached checkout.
  `--noconftest`, explicit module paths, disabled bytecode/cache writes and the
  shared `forbid_provider_network` guard keep the baseline unchanged and prevent
  accidentally importing the patched implementation.
- Synthetic sources, current/expired states, reference forms and positive,
  negative or unknown support states are fixed. Ordinary approvals are primed
  explicitly for both arms, so withholding an unbound runtime request does not
  make the approval-positive regression disappear.

The complete changed-code/test hash manifest and baseline/patched
`shared_memory.py` hashes are in the metrics. The original baseline failure
observations were retained; only the patched arm was rerun after the final
implementation freeze.

## Why the old path failed

`replay_intent()` represented only a successful binding to one eligible current
record. With two records, it returned `None`, which allowed downstream code to
fall back to the ordinary support policy. Positive ordinary decisions then made
retrieval nonempty. `missing_evidence()` returned early on that approved context,
and the controller's additional empty-memory-context condition could also hide
the clarification. Generation received the sources without an explicit record
selection requirement.

The fix separates request recognition from local binding through immutable
`RecordReplayResolution`. Every lookup recomputes reference resolution and the
full current primary set in the resolved resident scope before eligibility,
ranking, candidate limits or support decisions. A `bound` result still needs its
own exact complete-source replay-policy approval.

An `ambiguous_record` result withholds direct source text and latent seed
evidence and carries `current_unselected / unknown / clarify_record`. Here
`unknown` concerns exact replay approval of an unbound target; it does not claim
that existing ordinary approvals were negative. The same existing bounded
grammar covers named, caller-declared and pronominal forms. Ordinary semantic
and extra-clause questions retain their existing support path.

## Same-case comparison

| Fixed case group | Cases | Baseline passes | Fixed passes |
| --- | ---: | ---: | ---: |
| Both ordinary decisions positive, across four reference forms | 4 | 0 | 4 |
| One ordinary decision positive and one negative, same forms | 4 | 0 | 4 |
| One ordinary decision positive and one unknown, same forms | 4 | 0 | 4 |
| A current record omitted by candidate limits or source eligibility | 2 | 0 | 2 |
| Two records become one: expiry, supersession or derived-record status; fresh exact replay approval required | 3 | 3 | 3 |
| A second current record appears after prior replay and ordinary approvals | 1 | 0 | 1 |
| Ordinary single-fact, compound and extra-clause controls | 3 | 3 | 3 |
| Disabled checker must not select a record through lexical relevance | 1 | 0 | 1 |
| Non-conversation legacy retrieval | 1 | 1 | 1 |
| **Total** | **23** | **7** | **23** |

Neither arm has errors or skips. All 16 baseline failures and all 7 passing
controls remain represented in the metrics. Source records remain unverified;
neither replay binding nor approval certifies their claims as true.

## Review findings and generation checks

Separate assistant-agent review found two lifecycle boundaries beyond the fixed
23-case matrix, and each received its own regression tests.

First, if the last source and live alias disappeared after context admission,
the explicit named operation could stop being recognized, leaving old context
untouched. The current turn now retains only whether its exact original/rewritten
input matched the existing bounded grammar. It retains no source, scope, body or
approval. Reference and binding are recomputed, and the request flag is cleared
on the next turn, abort or committed-memory reload. Eleven added controls cover
expiry, supersession, derived-record status and turn/question ownership.

Second, a copied consultation packet could become stale after controller
dispatch while waiting in the queue. The right worker now uses the same shared
refresh helper immediately before `ASK_DETAIL`, `ASK_CRITIC` or `ASK_LEAD`
generation. The **same 12-case worker file** produced **6 failures / 6 passes**
with the saved previous worker and **12 passes** with the fixed worker, using
the patched memory implementation in both arms. This is a separate consumer
comparison, not another run of the original #92 implementation. Its guards also
recorded zero unexpected HTTP attempts.

Those tests enqueue a real copied Callosum packet, pause its consumer with an
event, change the sources, and then capture the actual `RightBrainModel` prompt
through a Fake completion. Changed-source cases withhold both old text and seed
provenance. Unique approved replay retains the full body and provenance; ordinary
semantic controls retain their multiple sources and supplementary context.

Another **17 EngineSession cases** capture actual Fake Provider inputs and
outputs across System2 ON/OFF, initial generation after the director stage,
changes during draft/review awaits, copied critic requests, revision,
verification and existing direct fallbacks. The shared refresh retains only an
already admitted exact memory block that is still approved and rechecks copied
seed evidence. Current clarification also reaches all five existing integration
entrypoints. Drafts, reviewer notes and earlier quotations do not establish a
new target or approval. These checks preserve existing call counts.

## Verification

| Check | Result |
| --- | --- |
| Full local Python suite under the shared HTTP guard | **1,770 passed, 18 skipped, 1 deliberate deselection** |
| Unexpected blocked HTTP attempts in that full run | **0** |
| New regression cases included in the full run | **89**: 23 comparison, 37 resolution, 17 EngineSession, 12 queued-worker |
| Deliberately excluded local-HTTP human-review test, run separately | **1 passed** |
| Local Gateway subset | **2 passed, 17 skipped** because the .NET SDK is absent |
| Node trace/history tests | **13 passed** |
| Diff whitespace check | Passed |

The 18 full-suite skips are **17 isolated-PostgreSQL tests** with no local test
DSN and **1 optional Torch test** with Torch unavailable. The configured GitHub
CI runs isolated PostgreSQL operational-journal/commit controls, the complete
Gateway contract suite, the Python 3.10 consultation lifecycle and Python/Node
unit tests. Its final-head status is recorded on the pull request.

## Interpretation and limits

The comparison isolates a reproducible software contract failure with fixed
inputs and responses. It does not estimate how often an unmodified real model
will obey the clarification prompt, calibrate a judge, or revise any historical
model grade. Exact UTF-8 equality, attribution, source validity and information
sufficiency remain separate existing assessment axes.

Revalidation governs inputs to the **next existing generation boundary**. If a
source changes after the last generation and a completed draft is returned with
no later call, this change does not schedule a new call to rewrite that draft or
revoke already streamed text. Existing source residency limits also remain:
the uniqueness check covers the current resident working set, not an unlimited
durable archive search. The fix adds no semantic classifier, new selector
language, live resampling or model-call budget.

## Reproducing the fixed comparison

Use the regression file from the implementation snapshot for both arms and
create a detached checkout of the baseline SHA above. Run the following through
`sr-dual-brain-llm/scripts/run_offline_checks.py -- python` in the fixed checkout,
setting `REPLAY_SOURCE_ROOT` to each production checkout in turn. Expected exits
are **1** for the baseline's 16 assertion failures and **0** for the fixed arm.

```python
import os
from pathlib import Path
import sys
import pytest

sys.dont_write_bytecode = True
source = Path(os.environ["REPLAY_SOURCE_ROOT"]).resolve()
test = Path("tests/test_ambiguous_replay_contract.py").resolve()
sys.path.insert(0, str(source / "sr-dual-brain-llm"))
sys.path.insert(0, str(source / "sr-dual-brain-llm/scripts"))
from core import shared_memory
from offline_provider_guard import forbid_provider_network

assert Path(shared_memory.__file__).resolve() == source / "sr-dual-brain-llm/core/shared_memory.py"
with forbid_provider_network() as audit:
    result = pytest.main(["-q", "--noconftest", "-p", "no:cacheprovider", str(test)])
assert audit.blocked_requests == 0
raise SystemExit(result)
```
