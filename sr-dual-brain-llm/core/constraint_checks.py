"""Bounded witnesses against explicit user constraints, never source approval.

Adapters share anchored observations and three outcomes. ``consistent`` means
only that the supported claims passed substitution/set comparison. Unparsed
claims, assumptions and semantics remain unknown. There is no model or solver.
"""
from contextvars import ContextVar
from dataclasses import dataclass
from fractions import Fraction
from functools import wraps
import ast
import json
import re

VERSION = 'grounded_constraints_v1'
MAX_TEXT = 10000
MAX_FINDINGS = 16
_LITERAL = re.compile(r'\b(?:verbatim|quote|reproduce|literal)\b|そのまま|再掲|逐語', re.I)
_UNSCOPED = re.compile(
    r'\b(?:if|only|without|unavailable|suppose|hypothetical|counterexamples?|'
    r'instead|incorrect|wrong|invalid|false|impossible|fails?|violates?|reject|cannot|could|would|'
    r'candidate|test|try|perhaps|maybe|not|approximately|approximate|\w+n[\'’]t)\b|反例|仮定|近似', re.I)
_TEX_SPACE = re.compile(r'\\(?:left|right|qquad|quad)\b|\\[()[\],;! ]|[$`]')
_FRAC = re.compile(r'\\(?:dfrac|tfrac|frac)\s*\{\s*([+-]?\d{1,30})\s*\}\s*\{\s*([+-]?\d{1,30})\s*\}')
_NUMBER = r'(?:\d+(?:\.\d+)?)'
_TOKEN = re.compile(r'\\(?:dfrac|tfrac|frac)\s*\{[+-]?\d{1,30}\}\s*\{[+-]?\d{1,30}\}|'
                    + _NUMBER + r'|[A-Za-z](?![\w])|[()+*/-]')
_ASSIGN = re.compile(r'(?<![\w])([A-Za-z])\s*=\s*(?!=)')
_CASE = re.compile(r'\b(?:when|for|given)\s+(?=[A-Za-z]\s*=)', re.I)
_TUPLE = re.compile(r'\(\s*([A-Za-z](?:\s*,\s*[A-Za-z]){1,3})\s*\)\s*=\s*\(([^()\n]{1,200})\)')
_TUPLE_HEAD = re.compile(r'\(\s*[A-Za-z](?:\s*,\s*[A-Za-z]){1,8}\s*\)\s*=')
_KEYS = re.compile(r'^\s*(?:Return|Output) (?:only )?(?:a )?JSON object with exactly '
                   r'(?:(?:these|the) )?keys:\s*([^\n]+?)\.?\s*$', re.I | re.M)
_active = ContextVar('grounded_constraint_state', default=None)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str


def _span(text, start, end):
    return Span(start, end, text[start:end])


@dataclass(frozen=True)
class Finding:
    kind: str
    status: str
    constraints: tuple[Span, ...]
    scope: Span
    claim: Span
    witness: str
    # A repair must still address the same user obligation and output variables.
    obligation: tuple


@dataclass(frozen=True)
class CheckReport:
    findings: tuple[Finding, ...] = ()

    @property
    def status(self):
        if any(f.status == 'contradiction' for f in self.findings):
            return 'contradiction'
        return 'consistent' if self.findings else 'unknown'

    @property
    def contradictions(self):
        return tuple(f for f in self.findings if f.status == 'contradiction')

    def summary(self):
        return {'version': VERSION, 'status': self.status,
                'checked_claims': len(self.findings),
                'contradictions': len(self.contradictions),
                'kinds': sorted({f.kind for f in self.findings}),
                'semantic_correctness': 'not_assessed'}


def _mask(text):
    # Preserve every offset; evidence is always sliced from the original text.
    return _TEX_SPACE.sub(lambda m: ' ' * len(m[0]), text)


def _bounded(value):
    if value.numerator.bit_length() > 256 or value.denominator.bit_length() > 256:
        raise ValueError('rational size limit')
    return value


def _linear(expression):
    """Exact affine arithmetic; reject powers/calls/products of variables."""
    if not 1 <= len(expression) <= 160:
        raise ValueError('expression size limit')
    if re.search(r'\d(?:\.\d*)?[eE][+-]?\d', expression):
        raise ValueError('unsupported scientific notation')
    expression = _FRAC.sub(r'(\1/\2)', expression).strip()
    expression = re.sub(r'(?<=[\d)])\s*(?=[A-Za-z])', '*', expression)
    nesting = 0
    for character in expression:
        nesting += (character == '(') - (character == ')')
        if nesting > 12:
            raise ValueError('parenthesis depth limit')
    tree = ast.parse(expression, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError('node limit')

    def visit(node, depth=0):
        if depth > 12:
            raise ValueError('depth limit')
        if isinstance(node, ast.Name) and re.fullmatch('[A-Za-z]', node.id):
            return {node.id: Fraction(1)}
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            token = ast.get_source_segment(expression, node)
            if token is None or not re.fullmatch(_NUMBER, token) or len(token) > 30:
                raise ValueError('unsupported number')
            return {'': _bounded(Fraction(token))}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            factor = -1 if isinstance(node.op, ast.USub) else 1
            return {k: _bounded(factor * v) for k, v in visit(node.operand, depth + 1).items()}
        if not isinstance(node, ast.BinOp):
            raise ValueError('unsupported expression')
        left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            sign = -1 if isinstance(node.op, ast.Sub) else 1
            result = dict(left)
            for key, value in right.items():
                result[key] = _bounded(result.get(key, Fraction(0)) + sign * value)
        elif isinstance(node.op, ast.Mult) and (set(left) <= {''} or set(right) <= {''}):
            scalar, other = (left, right) if set(left) <= {''} else (right, left)
            result = {k: _bounded(scalar.get('', Fraction(0)) * v) for k, v in other.items()}
        elif isinstance(node.op, ast.Div) and set(right) <= {''}:
            result = {k: _bounded(v / right.get('', Fraction(0))) for k, v in left.items()}
        else:
            raise ValueError('not affine arithmetic')
        return {k: v for k, v in result.items() if v}

    return visit(tree.body)


def _read_expression(text, start):
    position, end = start, start
    count = 0
    while position < len(text) and count < 64:
        while position < len(text) and text[position] in ' \t':
            position += 1
        match = _TOKEN.match(text, position)
        if match is None:
            break
        end = position = match.end()
        count += 1
    tail = text[end:]
    if re.match(r'\s*,\s*[+-]?\d', tail):
        raise ValueError('ambiguous numeric comma')
    if tail and not re.match(r'\s*(?:[,;\n]|[.:?](?=\s|$)|$|\band\b)', tail):
        raise ValueError('unsupported expression tail')
    return _linear(text[start:end]), end


def _assignments(text):
    result = []
    for match in _ASSIGN.finditer(text):
        expression, end = _read_expression(text, match.end())
        result.append((match[1], expression, match.start(), end))
        if len(result) > 8:
            raise ValueError('assignment limit')
    return result


def _cases(text, *, claims=False, original=None):
    """Require explicit numeric conditions in one clause; no scope inference."""
    original = text if original is None else original
    result = []
    for index, case in enumerate(_CASE.finditer(text)):
        if index >= 8:
            raise ValueError('case limit')
        # A period in a decimal is not a sentence boundary.
        # Removing a standalone display-math wrapper produces a whitespace-only
        # line. Paragraph boundaries must come from the original, not that mask.
        boundaries = [m for m in (
            re.search(r'\n[ \t\r]*\n', original[case.end():]),
            re.search(r'[.!?](?=\s|$)', text[case.end():]),
        ) if m is not None]
        boundary = min(boundaries, key=lambda m: m.start()) if boundaries else None
        if claims and boundary is not None and boundary[0].startswith('?'):
            continue
        end = case.end() + boundary.start() if boundary else len(text)
        previous = (list(re.finditer(r'[.!?](?=\s|$)', text[:case.start()]))
                    + list(re.finditer(r'\n[ \t\r]*\n', original[:case.start()])))
        start = max((m.end() for m in previous), default=0)
        if claims and text[start:case.start()].strip():
            continue  # Do not reinterpret a quoted/embedded conditional as a claim.
        segment = text[start:end]
        # A changed assumption/negative example invalidates this clause's scope.
        if _UNSCOPED.search(segment) or re.search(r'\bor\b', segment, re.I):
            continue
        values, consumed = {}, case.end()
        for match in _ASSIGN.finditer(text, case.end(), end):
            if text[consumed:match.start()].strip(' ,\t\r\n') not in ('', 'and'):
                if not claims:
                    raise ValueError('unsupported question condition connector')
                break
            expression, rhs_end = _read_expression(text, match.end())
            if set(expression) - {''}:
                if values:
                    raise ValueError('partially symbolic condition')
                break  # A symbolic definition is not a numeric case.
            if match[1] in values:
                raise ValueError('ambiguous case')
            values[match[1]] = expression.get('', Fraction(0))
            consumed = rhs_end
        if values:
            result.append((case.start(), consumed, end, values))
    return result


def _numeric(question, draft):
    q, a = _mask(question), _mask(draft)
    question_cases = _cases(q, original=question)
    if not question_cases:
        return []
    # Definitions must precede the request/first numeric case. Never import an
    # answer-derived equation or a constraint from a different requested goal.
    heading = re.search(r'^\s*(?:Requirements|Goals|Tasks|要件|依頼)\s*[:：]', q, re.I | re.M)
    prefix_end = heading.start() if heading else question_cases[0][0]
    prefix = q[:prefix_end]
    if not re.match(r'\s*(?:Exact\s+(?:measurements|relations|constraints)|Given\b|'
                    r'Equations\b|Measurements\b|Constraints\b)', prefix, re.I):
        return []
    noise_context = re.sub(r'\bno noise\b', '', prefix, flags=re.I)
    if (_UNSCOPED.search(prefix) or re.search(r'\b(?:noise|rounded|uncertainty)\b', noise_context, re.I)
            or any(marker in prefix for marker in ('>', '?', '"', '“', '”'))
            or any(marker in value for marker in ('```', '~~~') for value in (question, draft))):
        return []
    definitions = {}
    parsed = _assignments(prefix)
    if len(parsed) != prefix.count('='):
        return []
    for index, (name, expression, start, end) in enumerate(parsed):
        gap = prefix[parsed[index - 1][3]:start] if index else ''
        statement_tail = re.split(r'[.;\n]', prefix[end:], maxsplit=1)[0]
        if re.search(r'\bor\b', gap + ' ' + statement_tail, re.I):
            return []
        if not set(expression) - {''} or name in definitions:
            return []
        definitions[name] = (expression, _span(question, start, end))
    if not 1 <= len(definitions) <= 4:
        return []
    findings = []
    for begin, condition_end, clause_end, values in _cases(a, claims=True, original=draft):
        scopes = [(s, e, v) for s, e, _, v in question_cases if v == values]
        if len(scopes) != 1 or not set(values) <= definitions.keys():
            continue
        scope_start, scope_end, _ = scopes[0]
        scope = _span(question, scope_start, scope_end)
        heads = list(_TUPLE_HEAD.finditer(a, condition_end, clause_end))
        claims = list(_TUPLE.finditer(a, condition_end, clause_end))
        if {m.start() for m in heads} != {m.start() for m in claims}:
            raise ValueError('unsupported scoped tuple claim')
        for index, claim in enumerate(claims):
            if index >= MAX_FINDINGS:
                raise ValueError('claim limit')
            variables = tuple(v.strip() for v in claim[1].split(','))
            components = claim[2].split(',')
            if len(set(variables)) != len(variables) or len(components) != len(variables):
                raise ValueError('ambiguous tuple')
            constants = [_linear(c.strip()) for c in components]
            if any(set(c) - {''} for c in constants):
                raise ValueError('symbolic tuple')
            point = {v: c.get('', Fraction(0)) for v, c in zip(variables, constants)}
            applicable = [definitions[name] for name in values]
            needed = set().union(*(set(expr) - {''} for expr, _ in applicable))
            if needed != set(point):
                raise ValueError('incomplete tuple variables')
            residuals = []
            for name, expected in values.items():
                expr, _ = definitions[name]
                actual = _bounded(expr.get('', Fraction(0)) + sum(
                    (coefficient * point[v] for v, coefficient in expr.items() if v), Fraction(0)))
                residuals.append((name, actual, expected, _bounded(actual - expected)))
            witness = '; '.join(f'{name}: substituted {actual}, required {expected}, residual {residual}'
                                for name, actual, expected, residual in residuals)
            findings.append(Finding('exact_substitution',
                'contradiction' if any(r for _, _, _, r in residuals) else 'consistent',
                tuple(anchor for _, anchor in applicable), scope,
                _span(draft, claim.start(), claim.end()), witness,
                ('exact_substitution', scope.start, scope.end, tuple(sorted(variables)))))
            if len(findings) > MAX_FINDINGS:
                raise ValueError('finding limit')
    return findings


def _json_keys(question, draft):
    matches = list(_KEYS.finditer(question))
    if len(matches) != 1 or '```' in question or '~~~' in question or re.search(r'^\s*>', question, re.M):
        return []
    match = matches[0]
    outside = question[:match.start()] + question[match.end():]
    if (_UNSCOPED.search(outside) or re.search(
            r'\b(?:when|unless|otherwise|except|examples?|samples?|conditional)\b', outside, re.I)):
        return []  # No inference about whether a conditional output rule is active.
    tokens = [key.strip() for key in match[1].split(',')]
    identifier = r'[A-Za-z_][A-Za-z_0-9]{0,39}'
    if not all(re.fullmatch(r'(?:' + identifier + '|`' + identifier + '`|"' + identifier + '")', t)
               for t in tokens):
        return []
    keys = [t[1:-1] if t.startswith(('`', '"')) else t for t in tokens]
    if (not 1 <= len(keys) <= 12 or len(set(keys)) != len(keys)
            or not all(re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,39}', key) for key in keys)):
        return []

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate key')
            result[key] = value
        return result

    # A malformed/ambiguous output remains unknown, not an invented key set.
    value = json.loads(draft, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('non-JSON constant')))
    if not isinstance(value, dict) or len(value) > 32:
        return []
    required, actual = set(keys), set(value)
    anchor = _span(question, match.start(), match.end())
    return [Finding('json_key_set', 'consistent' if actual == required else 'contradiction',
        (anchor,), anchor, _span(draft, 0, len(draft)),
        f'missing keys: {sorted(required - actual)}; extra keys: {sorted(actual - required)}',
        ('json_key_set', anchor.start, anchor.end))]


def check_constraints(question, draft):
    if (not isinstance(question, str) or not isinstance(draft, str)
            or not 1 <= len(question) <= MAX_TEXT or not 1 <= len(draft) <= MAX_TEXT
            or _LITERAL.search(question)):
        return CheckReport()
    findings = []
    for adapter in (_numeric, _json_keys):
        try:
            findings.extend(adapter(question, draft))
        except (ValueError, SyntaxError, ZeroDivisionError, RecursionError, OverflowError):
            # No partially parsed result escapes an adapter's work/shape limit.
            pass
    if len(findings) > MAX_FINDINGS:
        return CheckReport()
    return CheckReport(tuple(findings))


def literal_requested(question):
    return isinstance(question, str) and bool(_LITERAL.search(question))


def repaired(before, after):
    required = {f.obligation for f in before.contradictions}
    checked = {f.obligation for f in after.findings if f.status == 'consistent'}
    return bool(required and after.status == 'consistent' and required <= checked)


class ConstraintState:
    def __init__(self, question=''):
        self.question = question
        self.required = set()
        self.observed = False

    def observe(self, report):
        self.observed |= bool(report.findings)
        self.required.update(f.obligation for f in report.contradictions)

    def finish(self, question, answer):
        report = check_constraints(question, answer)
        checked = {f.obligation for f in report.findings if f.status == 'consistent'}
        summary = report.summary()
        summary['unresolved_obligations'] = len(self.required - checked)
        if summary['status'] != 'contradiction' and self.required - checked:
            summary['status'] = 'unknown'
        return summary


def observe_constraints(report):
    state = _active.get()
    if state is not None:
        state.observe(report)


def current_constraints():
    return _active.get()


def guard_constraint_turn(method):
    @wraps(method)
    async def guarded(*args, **kwargs):
        from .source_completion import source_completion_required
        question = kwargs.get('question', args[1] if len(args) > 1 else '')
        token = _active.set(None if source_completion_required() or literal_requested(question)
                            else ConstraintState(question))
        try:
            return await method(*args, **kwargs)
        finally:
            _active.reset(token)
    return guarded
