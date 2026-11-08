# Grounded constraint consistency

The micro-critic can supply **local witnesses**, independently of a model's
confidence or well-formed review. `core/constraint_checks.py` shares one small
contract between exact numeric substitution and nonnumeric JSON key comparison.
The checker itself performs no provider request. Its findings can activate an
existing budgeted revision; it adds no brain, planner, router or per-goal loop.

Each internal finding contains exact user constraint spans, an explicit user
scope, an answer claim span, a check kind, an obligation identity and a witness.
Offsets always refer to the original text, including supported TeX wrappers.
The outcomes are `contradiction`, `consistent` and `unknown`. Consistency covers
only the claims actually checked; it never proves overall correctness, goal
completion, uniqueness, evidence quality or factual source approval.

## Prototype boundary

| Adapter | Accepted | Unknown / outside its scope |
| --- | --- | --- |
| `exact_substitution` | Explicit affine measurement definitions in the user prefix; matching explicit `when`/`for`/`given` numeric conditions in question and answer; named 2–4 variable tuple equalities in the same answer clause. Exact integer, finite decimal, rational and simple integer TeX fractions. | Inferred goal alignment, implicit/unnamed tuple points, symbolic families, nonlinear expressions, units, approximate/noisy measurements, unsupported notation, ambiguous/repeated cases, conflicting definitions, ungrounded conditions. |
| `json_key_set` | One standalone `Return a JSON object with exactly these keys: name, state.` directive (also `Output`, optional `only`, bare/backtick/double-quoted identifier keys); a whole valid JSON object as the answer. Compares top-level key sets. | Conditional/example instructions, malformed JSON, duplicate keys at any depth, nonstandard constants, non-object output, surrounding prose/fences, multiple key directives. Values, nested schemas and all other semantic requirements are unassessed. |

For example, given `p=3a+b` and `q=2b`, a user request `when p=8,q=4`
and an answer `For p=8,q=4, (a,b)=(1,2).` witness `p=5`, residual `-3`.
The checker does not solve the system or silently replace the point. With only
`p=a+b`, two different points can both satisfy the same measurement: no unique
solution is invented.

Numeric definitions must start with `Given`, `Measurements`, `Equations`,
`Constraints`, or `Exact measurements/relations/constraints`, and precede the
request heading/first numeric case. Supported single-letter ASCII variables
are arbitrary; no benchmark IDs, coefficients, expected answers or model names
are inspected. Only explicitly selected measurements are active. The checker
does not import a withdrawn measurement into a later counterexample. Clauses
must start with their explicit answer condition; embedded quotations and fenced
answers are withheld. Recognized negative, hypothetical or approximate wording
also withholds a clause; unsupported natural-language scope remains a substantial
recall limitation. Unsupported numeric suffixes, scientific notation, numeric
commas, disjunctions and question-condition connectors do not silently truncate
into a supported constraint.

Both texts are limited to 10,000 characters. Numeric checks allow at most four
measurement definitions, eight assignments/cases, sixteen findings, 160-character
expressions, 64 AST nodes/tokens, depth 12 and 256-bit rational components.
Only affine arithmetic is evaluated; calls, powers and arbitrary execution are
unsupported. An adapter's parsing/work-limit failure discards its partial result.

## Existing correction and ownership

An enabled local check that witnesses a contradiction takes precedence over
ordinary critic `ok`, including when the older heuristic sanity-override option
is disabled. `DUALBRAIN_CRITIC_MICRO_FALLBACK=0` and per-call
`allow_micro_fallback=False` still disable the local critic. External completion,
schema, review validity and failure provenance remain separate. An empty external
critic is still an unavailable review even if a local witness is present.

Witnesses feed the existing System2 integration or ACC/Cerebellum path. Passing
a structural/local check never supplies overall micro-critic approval or routing
confidence, and never suppresses existing content detectors. Positive consistency
is used directly to recheck witnessed repairs. Manual
System2 modes and TRN gates retain authority; there is no additional review round,
retry, implicit continuation or output-budget escalation. Primary writes,
including a delayed right worker, and byte-exact/literal reproduction do not
receive local micro-correction. A local check cannot approve or rewrite a recorded
source, grant tool permissions or promote a model report to verified completion.

A revision discharges a witnessed obligation only if that same user scope and
output variables remain checkable and consistent. Removing the point, changing
to unsupported notation, a failed revision or a later critic `ok` cannot by itself
erase the earlier witness. Turn-local obligations clear on return, exception and
cancellation. The final displayed answer is assessed after audit and generation
presentation; unresolved obligations veto resolution/success without another
model call. Existing cancellation, consultation draining and token caps apply.

`constraint_consistency` telemetry and `metrics.constraint_check` contain only
bounded status/counts/check kinds and `semantic_correctness=not_assessed`.
They contain no constraint text, answer quote or residual. A final unresolved
observation can veto `metrics.system2.resolved`, never grant semantic approval.

## Extension and evaluation rule

New deterministic adapters should use this anchored contract and existing
correction ownership. Require an explicit user constraint, a claimed output and
a reproducible local witness; abstain on unsupported scope. Add cross-domain
negative controls and evidence of bounded work before widening the grammar. Do
not add another module router, numeric-task route or model preference as a side
effect of an adapter.

The inverse-measurement failure from #105/#106 is **development data**. Tests also
generate 72 affine systems across 2–4 renamed variables and varied coefficients,
test exact rational/decimal arithmetic and multiple compatible solutions, and
exercise JSON repair through the same critic/Cerebellum/System2 machinery.
These are deterministic implementation checks, not an answer-quality benchmark.

No paid evaluation is performed for this prototype. Before any future quality
test, freeze fresh questions, success/abstention criteria, balanced allocation,
model/profile settings and call/token/time limits. Include numeric and nonnumeric
tasks (code, SQL, constraints, evidence/uncertainty and prose), supported and
unsupported scope, withdrawn assumptions and valid counterexamples. Retain all
failures and unknown usage. Report local exact witnesses separately from answer
quality and uncalibrated model-judge observations; a gain on numeric development
cases cannot establish a general orchestration gain.
