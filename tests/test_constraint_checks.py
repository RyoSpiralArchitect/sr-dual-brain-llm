"""Exact witnesses, scope abstention and nonnumeric contract checks (no model)."""
import asyncio
from fractions import Fraction
import json
import random

import pytest

from core.constraint_checks import (
    ConstraintState, check_constraints, current_constraints, guard_constraint_turn,
    observe_constraints, repaired,
)

QUESTION = ('Exact measurements follow u=2x+y and v=3y. No noise is assumed.\n'
    'Requirements:\n'
    '1. Recover x,y when u=5,v=6.\n'
    '2. Suppose v is unavailable and only u=5 is observed: describe all compatible pairs.\n'
    '3. Give counterexamples for step 2 that do not match the additional observation v=6.')
BAD = r'For \(u=5,\ v=6\), this yields \[(x,y)=(1,2).\]'
GOOD = BAD.replace('(1,2)', r'(\frac{3}{2},2)')
JSON_QUESTION = 'Return a JSON object with exactly these keys: name, state.'


def test_development_failure_has_exact_visible_anchors_and_residual():
    answer = BAD + '\n\nIf only u=5, examples are (0,5) and (1,3); they do not satisfy v=6.'
    report = check_constraints(QUESTION, answer)
    assert report.status == 'contradiction' and len(report.findings) == 1
    finding = report.findings[0]
    assert 'substituted 4, required 5, residual -1' in finding.witness
    assert finding.claim.text == '(x,y)=(1,2)'
    assert answer[finding.claim.start:finding.claim.end] == finding.claim.text
    assert QUESTION[finding.scope.start:finding.scope.end] == finding.scope.text
    assert all(QUESTION[s.start:s.end] == s.text for s in finding.constraints)
    assert repaired(report, check_constraints(QUESTION, GOOD))
    assert not repaired(report, check_constraints(QUESTION, 'The result is now correct.'))


@pytest.mark.parametrize('wrapper', [('\\[', '\\]'), ('$$', '$$')])
def test_display_math_wrappers_do_not_create_paragraph_boundaries(wrapper):
    question = 'Given p=3a+b and q=2b. Recover a,b when p=8,q=4.'
    left, right = wrapper
    answer = ('The symbolic inverse follows from substitution.\n'
              r'For \(p=8,\ q=4\), this yields' + '\n' + left + '\n'
              r'(a,b)=\left(1,2\right).' + '\n' + right + '\n\n'
              'If only p=8 is available, (a,b)=(0,8) is a compatible example.')
    report = check_constraints(question, answer)
    assert report.status == 'contradiction' and len(report.findings) == 1
    assert 'substituted 5, required 8, residual -3' in report.findings[0].witness
    assert check_constraints(question, answer.replace('(1,2', '(2,2')).status == 'consistent'
    # An actual paragraph break still withholds unsupported scope inheritance.
    assert check_constraints(question, answer.replace('yields\n', 'yields\n\n')).status == 'unknown'


@pytest.mark.parametrize('dimension', [2, 3, 4])
def test_generated_affine_systems_with_renamed_variables_and_exact_perturbations(dimension):
    # Generate points/constraints independently of parser mechanics; no solving,
    # fixture numbers, float tolerance or expected-answer bag is used by checks.
    rng = random.Random(710 + dimension)
    variables = 'abcd'[:dimension]
    for _ in range(24):
        point = [Fraction(rng.randint(-8, 8), rng.randint(1, 7)) for _ in variables]
        rows = [[rng.choice([-5, -3, -1, 1, 2, 4]) for _ in variables] for _ in range(2)]
        offsets = [rng.randint(-5, 5) for _ in rows]
        names = 'pq'
        definitions = [name + '=' + '+'.join(f'({coefficient}*{v})' for coefficient, v in zip(row, variables))
                       + f'+({offset})' for name, row, offset in zip(names, rows, offsets)]
        values = [sum((c * v for c, v in zip(row, point)), Fraction(offset)) for row, offset in zip(rows, offsets)]
        condition = ','.join(f'{name}={value}' for name, value in zip(names, values))
        question = 'Exact measurements: ' + ', '.join(definitions) + '. Recover variables when ' + condition + '.'
        prefix = f'For {condition}, ({",".join(variables)})='
        correct = prefix + '(' + ','.join(map(str, point)) + ').'
        report = check_constraints(question, correct)
        assert report.status == 'consistent', (question, correct)
        point[0] += Fraction(1, 11)
        incorrect = prefix + '(' + ','.join(map(str, point)) + ').'
        assert check_constraints(question, incorrect).status == 'contradiction', (question, incorrect)


@pytest.mark.parametrize('question,answer,status', [
    ('Given p=0.1a+0.2b. Recover a,b when p=0.3.', 'For p=0.3, (a,b)=(1,1).', 'consistent'),
    ('Given p=a/3+b/7. Recover a,b when p=2/3.', 'For p=2/3, (a,b)=(1,7/3).', 'consistent'),
    ('Given p=-2a+b+1. Recover a,b when p=-3.', 'For p=-3, (a,b)=(2,0).', 'consistent'),
    ('Given p=a+b. Recover a,b when p=3.', 'For p=3, (a,b)=(0,3) and (a,b)=(1,2).', 'consistent'),
    ('Given p=a+b. Recover a,b when p=3.', 'For p=3, (a,b)=(0,3) and (a,b)=(2,2).', 'contradiction'),
    ('Given p=a+b. Recover a,b when p=3.', 'For p=3, (b,a)=(1,2).', 'consistent'),
    ('Given p=2a+b. Recover a,b when p=3.', 'For p=3, (b,a)=(1,2).', 'contradiction'),
    ('Given p=a+b and q=a-b. Recover a,b when p=3.', 'For p=3, (a,b)=(0,3).', 'consistent'),
])
def test_supported_subset_exact_decimals_fractions_and_multiple_solutions(question, answer, status):
    assert check_constraints(question, answer).status == status


@pytest.mark.parametrize('answer', [
    'If only u=5, (x,y)=(0,5).',
    'For u=5,v=6, the incorrect example is (x,y)=(1,2).',
    'It is wrong that for u=5,v=6, (x,y)=(1,2).',
    'For u=5,v=6, approximately (x,y)=(1,2).',
    'For u=5,v=6, (x,y)=(1,2) fails.',
    'For u=5,v=6, is (x,y)=(1,2)?',
    'For u=5,v=6, if v is unavailable, (x,y)=(0,5).',
    'For u=5,v=6, (x,y)=(t,5-2t).',
    'For u=5,v=6, (x,x)=(1,2).',
    'For u=5,v=6, (x,z)=(1,2).',
    'The answer is (x,y)=(1,2).',  # No explicit answer scope.
    'For u=5,v=6, (x,y)=(1e0,2).',
    'For u=5,v=6, (x,y)=(1/0,2).',
    'For u=5,v=6, (x,y)=(3/2,2) and (x,y)=(1,(2)).',
    'For u=5,v=6, (x,y)=(3/2,2) and (x,z)=(1,2).',
    'For u=5,v=6, counterexamples are (x,y)=(1,2).',
    'Someone wrote: "For u=5,v=6, (x,y)=(1,2)."',
    '```text\nFor u=5,v=6, (x,y)=(1,2).\n```',
])
def test_unsupported_hypothetical_and_ambiguous_claims_abstain(answer):
    assert check_constraints(QUESTION, answer).status == 'unknown'


@pytest.mark.parametrize('change', [
    lambda q: q.replace('u=2x+y', 'u=x^2+y'),
    lambda q: q.replace('u=2x+y', 'u=sin(x)+y'),
    lambda q: q.replace('u=2x+y', 'u=xy+y'),
    lambda q: q.replace('u=2x+y', 'u=2x+y, u=3x'),
    lambda q: q.replace('u=2x+y', 'u=x*y+y'),
    lambda q: q.replace('u=2x+y', 'u=x/y'),
    lambda q: q.replace('No noise is assumed', 'Measurements have noise'),
    lambda q: q + '\n4. Again recover x,y when u=5,v=6.',
    lambda q: q.replace('Recover x,y when', 'Suppose another assumption holds when'),
    lambda q: 'Reproduce verbatim:\n' + q,
    lambda q: '```\n' + q + '\n```',
])
def test_unsupported_conflicting_and_quoted_constraints_abstain(change):
    assert check_constraints(change(QUESTION), BAD).status == 'unknown'


@pytest.mark.parametrize('value', [None, {}, [], 17, '', 'x' * 10001])
def test_bad_shapes_and_global_work_limit(value):
    assert check_constraints(value, BAD).status == 'unknown'
    assert check_constraints(QUESTION, value).status == 'unknown'


@pytest.mark.parametrize('expression', ['9' * 31 + '*x+y', '('*20 + 'x' + ')'*20 + '+y', 'x**999999+y'])
def test_expression_work_limits_abstain(expression):
    assert check_constraints(QUESTION.replace('2x+y', expression), BAD).status == 'unknown'


def test_json_key_sets_share_witness_contract_without_numeric_reasoning():
    report = check_constraints(JSON_QUESTION, '{"name":"cat","debug":true}')
    assert report.status == 'contradiction'
    finding = report.findings[0]
    assert finding.kind == 'json_key_set' and 'state' in finding.witness and 'debug' in finding.witness
    good = check_constraints(JSON_QUESTION, '{"state":"awake","name":"cat"}')
    assert repaired(report, good)
    assert good.summary()['semantic_correctness'] == 'not_assessed'
    # Structural consistency is not evidence that values or prose are true.
    assert check_constraints(JSON_QUESTION, '{"name":null,"state":"invented"}').status == 'consistent'


@pytest.mark.parametrize('directive', [
    'Output only a JSON object with exactly these keys: `route`, `reason`, `alternatives`.',
    'Return JSON object with exactly the keys: "route", "reason", "alternatives".',
])
def test_other_json_contracts_and_balanced_key_quotes(directive):
    answer = '{"alternatives":[],"reason":"insufficient evidence","route":"clarify"}'
    assert check_constraints(directive, answer).status == 'consistent'


@pytest.mark.parametrize('answer', ['{"name":1,"name":2}', '{', '[1,2]',
    '{"name":NaN}', '```json\n{"name":1}\n```', 'Text before {"name":1}', '[' * 1000])
def test_malformed_and_unsupported_json_stays_unknown(answer):
    assert check_constraints(JSON_QUESTION, answer).status == 'unknown'


@pytest.mark.parametrize('question', [
    JSON_QUESTION + '\nReturn a JSON object with exactly these keys: other.',
    'Quote this exactly:\n' + JSON_QUESTION, '> ' + JSON_QUESTION,
    'Return a JSON object with exactly these keys: name, name.',
    'Return a JSON object with exactly these keys: "name, state.',
    'If an error occurs:\nReturn a JSON object with exactly these keys: error.\nOtherwise return success.',
    'An example instruction:\nReturn a JSON object with exactly these keys: name, state.',
    'Design a bounded queue with cancellation and shutdown.',
    'Explain a SQL left join without dropping unmatched orders.',
    'Compare randomized and uncontrolled evidence; keep uncertainty explicit.',
])
def test_nonnumeric_unsupported_requests_receive_no_invented_contract(question):
    assert check_constraints(question, '{"name":1}').status == 'unknown'


def test_removed_claim_cannot_discharge_prior_obligation_and_summary_has_no_text():
    state = ConstraintState()
    state.observe(check_constraints(QUESTION, BAD))
    unknown = state.finish(QUESTION, 'I am confident it is fixed.')
    assert unknown['status'] == 'unknown' and unknown['unresolved_obligations'] == 1
    assert state.finish(QUESTION, GOOD)['unresolved_obligations'] == 0
    encoded = json.dumps(unknown)
    assert BAD not in encoded and QUESTION not in encoded and 'claim' not in unknown


def test_turn_cleanup_on_cancel_and_exception():
    class Turn:
        @guard_constraint_turn
        async def run(self, question, error):
            assert current_constraints() is not None
            observe_constraints(check_constraints(question, BAD))
            raise error

    async def run():
        for error in [asyncio.CancelledError(), RuntimeError('failure')]:
            with pytest.raises(type(error)):
                await Turn().run(QUESTION, error)
            assert current_constraints() is None
    asyncio.run(run())
