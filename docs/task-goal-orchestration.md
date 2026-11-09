# Explicit task goals and role-specific reasoning (prototype)

`Orchestrator` can carry a bounded single-turn plan through existing reasoning
phases. The default-OFF TRN module `task_goal_orchestration` admits the feature;
AUTO requires an eligible explicit list of two to four user goals. There is no
extra planner-model call, per-goal agent loop or change to manual consultation,
System2, source approval, persistence or external-operation authority.

```bash
export DUALBRAIN_MODULE_TASK_GOAL_ORCHESTRATION_MODE=auto
```

The first prototype recognizes a standalone `Goals:`, `Requirements:`, `Tasks:`,
`Please do:`, `要件：`, `依頼：` or `やること：` heading followed by a complete
numbered/bulleted list. Each goal retains its exact user-text span. Numbered
`step N`, `item N`, `goal N`, `項目N` and `手順N` backward references can establish
dependencies. Missing references do not establish semantic independence.
Single goals, unsupported structure, more than four goals, overlong goals,
ambiguous headings and forward/cyclic dependencies keep the direct pipeline;
no requested item is silently truncated. Fenced/quoted headings are not goals.
Unstructured intent inference is outside this prototype.

For example:

```text
Explain how to make a bounded queue safe.
Requirements:
1. Give a concrete backpressure strategy.
2. Give a counterexample that can deadlock.
3. Using step 1, describe an executable test without claiming it was run.
```

The role assignments enter actual model inputs:

| Existing phase | Goal-specific responsibility |
| --- | --- |
| Left draft | Derive an answer to each requested goal and respect constraints/dependencies. |
| Right lead/detail | Supply concrete useful alternatives, counterexamples or missing assumptions tied to the task. |
| System2 critic | Check each goal and report anchored findings alongside ordinary issues/fixes. |
| Integration | Address the goals and reported omissions; retain uncertainty and necessary clarification. |

The right worker receives a copied plan and re-derives it against the exact
question before use. Direct fallbacks use the same turn-local context. Primary
writes and literal reproduction retain their own paths, including protection in
the source generator and delayed worker. Goal instructions grant no permission
to retrieve, approve, persist or act outside the request.

An active critic is asked for `goal_findings` in its existing JSON completion:
one `goal_id`, `status` (`addressed`, `unresolved`, `uncertain`), literal draft
`quote` and bounded `reason` per goal. Positive claims require a quote present in
the actual reviewed snapshot. Duplicate/unknown IDs, incomplete coverage,
invented quotes and inconsistent `ok`/unresolved reports stay unknown/invalid.
Missing/malformed goal observations make no new repair call or review round.
The existing critic completion/schema and external-validity contracts remain
separate, including unavailable/local fallback provenance.

These are **provisional model reports**, not semantic proof. `reported_addressed`
never means verified completion. A revised answer or a partially reviewed
snapshot becomes `unreviewed_answer`, while earlier reports retain their
provenance. A later unavailable review cannot inherit an earlier valid one.
Valid unresolved reports for an exact current snapshot can feed the existing
integration phase even when generic issue filtering omitted the goal-specific
signal. They do not change round budgets or force a critic in System2 OFF.

Plans and observations remain turn-local and clear on success, failure and
cancellation. Plan cleanup cannot reset existing round-trip limits. No model
request is added solely to construct a plan. Existing stage output/time limits,
fallbacks and task ownership remain mandatory; at most four goals share each
existing phase's budget.

`task_goal_execution` / `metrics.task_goals` expose goal counts, generic goal IDs,
dependencies, review status and whether the reviewed snapshot matches the final
answer. They contain no goal text, quote or reason. `verified_complete` remains
false: useful output and requested-goal coverage require separate evaluation.
The module gate itself contains only the existing bounded admission metadata.

[Grounded constraint checks](constraint-consistency.md) can independently witness
an explicit local contradiction and feed the existing bounded correction path.
They apply to supported numeric substitutions and JSON key sets, without changing
goal extraction or treating provisional coverage as semantic verification.

The [first frozen OFF/AUTO comparison](benchmarks/task-goals-2026-11-07.md)
records actual prompt handoffs and conservative unresolved states, but does not
establish a reliable quality improvement. Empty ordinary completions and a
cancel/fallback replay divergence remain visible limitations. The feature stays
default OFF.

An independent default-OFF [completion-replanning module](completion-replanning.md)
can carry all goals and dependencies into reconstruction after a recorded length-
limited generation. It uses the existing integration and remaining allocation,
without a planner-model call, per-goal execution loop or verified-goal promotion.
