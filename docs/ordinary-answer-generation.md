# Ordinary answer completion and bounded reasoning

Ordinary left drafts and integrations retain request-local completion outcomes.
An RPC `ok` means the handler returned; it does not assert a completed solution.
`result.answer_status` and `metrics.answer_generation` distinguish `complete`,
`partial`, `empty`, `invalid`, `failed`, `unavailable`, `budget_exhausted` and
`unverified` output. Cancellation propagates to the owner and records a cancelled
stage. `complete` establishes a nonempty successful provider terminal, never
semantic correctness. `semantic_correctness` remains `not_assessed`.

This implements #102's ordinary-generation boundary. It adds no brain, module
gate, planner, repair loop or permission to retrieve, approve sources, persist
facts or take external action. Source writes keep their existing independent
completion and commit requirements. Literal memory replay keeps its own source
selection and byte-preserving path. Critic-specific availability and broad
semantic calibration remain separate work.

## Selecting an answer

The LLM client still returns strings. A scoped collector observes the actual
validated terminal result from that request; it is not shared client state.
Nonempty output stopped by a length limit, or visible stream output without a
terminal completion, stays partial. Invalid/refused envelopes are withheld.
Connection failure after stream text retains partial provenance. Custom clients
without a validated provider receipt can supply text, but remain `unverified`.
Adapters must accept the bounded completion options; rejecting them fails the
stage rather than retrying without its limits. An unconfigured ordinary model
returns unavailable output instead of the former `Draft answer for:` placeholder.
Offline role-wiring tests use explicit scripted provider receipts when they need
successful generation; they do not treat that placeholder as model capability.

A failed or partial integration cannot replace a known terminally complete draft.
Rejected integration deltas are buffered, and the selected text is emitted only
after the result is known. If no complete draft exists, a visible partial result
can be retained with a plain incomplete-answer notice. With no usable text, the
response says generation failed and invites a retry. An auditor's empty-answer
clarification cannot turn a generation failure into a claim that the user's
question was incomplete. Internal placeholder drafts and raw collaborator notes
do not become a substitute successful answer.

Confidence is zero for an empty or unconfirmed ordinary draft. Existing routing
still decides whether to consult; manual System2 OFF does not authorize a critic,
and no new request is made merely to obtain a completion label. A final answer
without confirmed generation cannot earn the ordinary success signal. Goal
reports remain provisional; an addressed report against unconfirmed generation
cannot discharge a goal, and revised snapshots retain their unreviewed state.

Output cleanup and later edits can change the confirmed text. Such an answer
stays `unverified` unless its exact stripped text matches a recorded terminal
completion. This is conservative provenance, not a claim that every edit is
wrong. A retained good draft can remain complete while its failed integration is
still visible in the stage list.

## Allocating bounded work

Every ordinary draft/integration request disables automatic continuation,
including streaming. One stage makes at most one generation request. Existing
critic/detail/source contracts keep their own limits. Ordinary generation never
silently enlarges the configured per-request token ceiling or retries a failed
request.

A turn reserves output ceilings for ordinary left draft/integration work from a
shared allocation. The default upper bound is four times the configured stage
cap: a draft and at most three existing integration opportunities. The optional
`DUALBRAIN_ANSWER_TOKEN_BUDGET` can lower that bound; larger values are clamped to
the bound. Invalid/nonpositive values inhibit ordinary requests. A final request
can use the remaining smaller allowance; exhausted allocation makes no provider
request. Failed/cancelled requests keep their reservation, so unavailable billed
usage cannot be optimistically refunded. These are allocations of output
ceilings, **not measured token usage or provider cost**. Existing transport
accounting remains authoritative for actual attempts and known/unknown usage.
Right detail, critics, previews, source writes and support queries have their own
existing budgets; this allocation is not an all-provider billing cap. Native
round limits and the RPC deadline remain additional turn-wide bounds. Per-request
timeouts are capped by the remaining RPC deadline when one exists.

Reasoning models can spend an entire completion cap before emitting visible text.
The opt-in `DUALBRAIN_ANSWER_REASONING_EFFORT=low|medium|high` selects a request
reasoning effort for ordinary draft, integration and detail on OpenAI GPT-5/6
Chat Completions. It leaves the token cap unchanged. The default is
`provider_default`, which omits the parameter. Other provider/model families omit
it. This is an explicit compatibility choice for APIs supporting that parameter;
unsupported provider responses remain failures with no retry using another
profile. It does not change critic or primary-source generation requests.

```bash
export DUALBRAIN_ANSWER_REASONING_EFFORT=low
# Optional lower shared allocation, never an automatic budget increase:
export DUALBRAIN_ANSWER_TOKEN_BUDGET=4096
```

The new effort profile is not promoted to a default. Terminal availability alone
cannot establish a reasoning-quality gain. Before any paid comparison, freeze
fresh questions, references, allocation, ordering and answer-quality criteria.
The eight prior empty completions in #101 are development evidence, not fresh
quality test data. Their original scores and failed replay remain unchanged;
#103 owns the separate cancellation/fallback replay race.

## Observability and checks

`answer_generation` stage records expose stage, status, bounded finish/error
categories, whether text existed, the reserved request ceiling and the selected
reasoning-effort request parameter. They contain no prompt, response text, goal
quote, provider exception text or credential. `reserved_output_tokens` and
`total_output_token_cap` describe the shared left-generation allocation.
`terminal_complete` describes the selected answer's provenance, while stage
failures remain visible even when a complete draft was retained.

Tests exercise actual validated JSON and SSE envelopes, empty reasoning-only
length outputs, partial streams/disconnects, refused/invalid output, failed
integration preserving a good draft, finite shared allocations, explicit effort
payloads, unavailable/custom clients, goal-report authority and owned
cancellation. RPC/controller checks distinguish normal handler success from
complete, partial and empty generation. Scripted completions demonstrate those
contracts and do not measure real model quality.
