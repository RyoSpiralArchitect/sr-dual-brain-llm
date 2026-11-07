# Ordinary generation contracts — development checks, 2026-11-07

#102 now preserves ordinary generation outcomes, retains a confirmed draft after
failed integration, and allocates bounded draft/integration output ceilings.
The former no-provider placeholder cannot become a successful solution. This is
a completion/ownership change; **no new paid run or real-model quality gain is
claimed**. Reasoning-effort selection remains opt-in, with provider defaults
unchanged. See the [contract](../ordinary-answer-generation.md) and
[PR #104](https://github.com/RyoSpiralArchitect/sr-dual-brain-llm/pull/104).

The first implementation freeze is
`e95046dd3feb5f9c9c608cf2183df4e441bba7e4`, based on merged #101
`869be0267802cbe669a938509f99ac3b46df11dd`. A subsequent narrow presentation fix
keeps plain-answer Director limits and explicit debug sections when adding an
incomplete/failure notice. The original #101 source, raw results, rubric, scores
and failed replay remain unchanged. #103 owns that separate cancellation/fallback
ordering race; it is not fixed or declared successful by these checks.

## Applying the contract to preserved empty envelopes

A zero-network development check reads the eight actual empty `length`
responses from #101's private raw journal. It substitutes each unchanged response
above the real HTTP boundary and invokes the strict current generation contract
in a fresh scoped check. It does not ask a model to answer again, assess new
answers, reproduce the whole original request trajectory or change old scores.
The original journal bytes are checked before and after execution.

The check runs at `e95046d`; its core/script digest is
`627a0acd720e710f5c601ced337c9607e97b031da3ce247576b00dca682f456f`.
The later notice-formatting change does not change these isolated terminal
observations. The [filtered data](ordinary-generation-2026-11-07.metrics.json)
retains source/script/journal hashes and every check.

| Original response role | Envelopes | Isolated helper result | What it establishes |
| --- | ---: | --- | --- |
| Left draft | 4 | empty; no terminal completion | Empty length output cannot become a completed ordinary draft. |
| Integration | 2 | empty; no terminal completion | Empty length output cannot establish a new integrated answer. |
| Detail-sized control | 2 | empty; no terminal completion | Generic helper control only; this does not invoke the production right-detail method. |

The two detail controls do not demonstrate a newly fixed right-detail behavior.
That path already withheld empty/incomplete output before #102, and actual
`RightBrainModel.deepen` controls separately verify its strict usability and
450-token cap. The opt-in reasoning-effort setting now reaches compatible detail
requests, without enlarging that cap. Real provider acceptance and answer-quality
effects of that setting have not been measured here.

All eight isolated checks preserve `empty`, make **zero provider HTTP calls** and
record zero unexpected blocked HTTP attempts. This artifact publishes no raw
runtime messages, question/answer text or judge quotes; the original synthetic
fixture is already tracked. This is development evidence from previously
observed failures, not fresh benchmark data. The old cancellations and unknown
usage are not rerun, refunded or converted to success by an output allocation.
The private check report/script are retained under ignored
`samples/task-goals-2026-11-07-ordinary-contract-development`.

## Regression evidence and limits

The first broad development run found eleven regressions. An unobserved fixed
control draft was incorrectly assigned low confidence; confidence now changes
only for an actually observed empty/unconfirmed result. Organ-wiring tests that
needed successful generation now use explicit local provider receipts, while
separate tests exercise unavailable models. The old critic failure matrix now
expects the actually confirmed draft after failed integration, with the failed
request/unknown usage still retained. Historical reports keep their original
local-fallback observation.

At the first freeze, **2,067 local Python tests pass**, with isolated PostgreSQL,
one intentional local-HTTP deselection and zero unexpected blocked provider
requests. Affected regression controls pass 77; envelope/source/detail/goal focus
passes 147; Node passes 13. The subsequent presentation change passes 53 focused
checks, including an effective 80-character Director bound and retained debug
sections. Its full local run at `12b59d11e0ffe3da74f25fb2d43a848eb617081b`
passes **2,069 tests**, with the same isolated PostgreSQL/deselection/HTTP guard;
all eight CI checks at that revision pass. Gateway/local HTTP passes 18 with two
platform skips. Final review additionally rejects malformed falsey SSE terminal
objects instead of converting them to an empty valid delta. Its focused
envelope/source/detail/goal/cancellation set passes **161 tests**. Final-head CI
evidence is attached to PR #104 at its reviewed head.

Actual RPC/controller tests distinguish handler `ok` from complete/partial/empty
answers and preserve manual System2 OFF. JSON/SSE tests cover terminal reasons,
partial disconnects, malformed/refused/custom output, same-text completions,
retained drafts, lowered/exhausted budgets, no continuation, deadline bounds,
concurrent context isolation, cancellation and goal-report authority. Scripted
responses establish mechanics; they cannot establish model reasoning quality.

The shared allocation covers ordinary **left draft/integration ceilings**, not
all provider work or billed usage. Critics, detail, previews and source/support
work keep their existing budgets. Integration output is buffered before
selection, which can delay visible deltas. Stripped-text-changing cleanup can
conservatively leave a final answer unverified. A complete provider terminal
still does not establish correctness, and a retained draft may have unresolved
semantic issues. Broader calibration remains #89; critic-specific availability
remains #87. Any next paid quality comparison needs fresh frozen tasks,
references, criteria and allocation; these old envelopes are development data.
