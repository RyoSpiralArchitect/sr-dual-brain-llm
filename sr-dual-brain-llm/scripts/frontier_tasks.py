"""Prospective closed-world tasks and exact graders; never execute model code.

The generators depend only on fixed seeds, partition, domain and level. Expected
answers are evaluator data and must never enter answering/review requests.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import random
import sqlite3

DOMAINS = ('logic', 'sql', 'events', 'evidence')
ORIGIN = 'prospective_reasoning_frontier_v1'
MAX_CALIBRATION_LEVEL = 6


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field')
        result[key] = value
    return result


def decode(answer):
    if not isinstance(answer, str) or not answer.strip() or len(answer) > 30000:
        raise ValueError('No bounded answer')
    return json.loads(answer, object_pairs_hook=unique_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-JSON constant')))


def logic_models(data):
    """Enumerate assignments by bit masks; clauses contain signed 1-based IDs."""
    n = len(data['variables'])
    masks = [(sum(1 << (x-1) for x in clause if x > 0),
              sum(1 << (-x-1) for x in clause if x < 0)) for clause in data['clauses']]
    models = []
    limit = (1 << n) - 1
    for assignment in range(1 << n):
        inverse = limit ^ assignment
        if all(assignment & positive or inverse & negative for positive, negative in masks):
            models.append(assignment)
    return models


def logic_answer(data):
    models = logic_models(data)
    if not models:
        raise ValueError('Generated task must be satisfiable')
    return {'status': {name: ('entailed_true' if all(m & (1 << i) for m in models)
        else 'entailed_false' if all(not m & (1 << i) for m in models) else 'undetermined')
        for i, name in enumerate(data['variables'])}}


def make_logic(rng, level):
    n = 4 + 2*level
    names = [f'v{i:02d}' for i in range(n)]
    planted = [rng.choice((False, True)) for _ in names]
    clauses = set()
    while len(clauses) < (3*n if level == 1 else 5*n):
        selected = rng.sample(range(n), 3)
        clause = tuple(sorted((i+1 if rng.choice((False, True)) else -(i+1)) for i in selected))
        if any(planted[abs(x)-1] == (x > 0) for x in clause):
            clauses.add(clause)
    data = {'variables': names, 'clauses': [list(c) for c in sorted(clauses)]}
    rendered = [[('~' if x < 0 else '') + names[abs(x)-1] for x in c] for c in data['clauses']]
    text = ('This is a satisfiable Boolean constraint system. A clause is an OR of its terms; '
        'every clause must hold. A ~ prefix means NOT. Classify a variable as entailed_true '
        'when it is true in every satisfying assignment, entailed_false when false in every '
        'satisfying assignment, and undetermined when both values occur. Do not select a '
        'single satisfying assignment as if it proved entailment.\nVariables: '
        + json.dumps(names) + '\nClauses: ' + json.dumps(rendered, separators=(',', ':')))
    return data, text


SQL_SCHEMA = (
    'CREATE TABLE accounts(id TEXT);',
    'CREATE TABLE events(id INTEGER,account_id TEXT,ts INTEGER,kind TEXT,amount INTEGER);',
    'CREATE TABLE tags(account_id TEXT,label TEXT);',
    'CREATE TABLE bans(account_id TEXT,since INTEGER,until INTEGER,reason TEXT);',
)
SQL_QUERY = '''WITH ranked AS (
 SELECT id,account_id,kind,
 ROW_NUMBER() OVER(PARTITION BY account_id ORDER BY ts DESC,id DESC) AS rn
 FROM events WHERE ts<=:cutoff AND kind<>'void'
)
SELECT a.id AS account,
 COALESCE(SUM(CASE WHEN t.label='active' AND e.kind='credit' THEN e.amount ELSE 0 END),0) AS credit_total,
 COUNT(e.id) AS joined_events, COUNT(DISTINCT t.label) AS tag_kinds,
 r.id AS latest_event, r.kind AS latest_kind
FROM accounts a
LEFT JOIN events e ON e.account_id=a.id AND e.ts<=:cutoff
LEFT JOIN tags t ON t.account_id=a.id
LEFT JOIN ranked r ON r.account_id=a.id AND r.rn=1
WHERE NOT EXISTS (
 SELECT 1 FROM bans b WHERE b.account_id=a.id AND b.reason='hold'
 AND b.since<=:cutoff AND (b.until IS NULL OR b.until>:cutoff)
)
GROUP BY a.id,r.id,r.kind ORDER BY a.id'''


def sql_answer(data):
    with sqlite3.connect(':memory:') as conn:
        conn.row_factory = sqlite3.Row
        for statement in SQL_SCHEMA:
            conn.execute(statement)
        for table, width in (('accounts', 1), ('events', 5), ('tags', 2), ('bans', 4)):
            conn.executemany(f'INSERT INTO {table} VALUES({",".join("?" for _ in range(width))})', data[table])
        return {'rows': [dict(row) for row in conn.execute(SQL_QUERY, {'cutoff': data['cutoff']})]}


def sql_reference(data):
    """Independent Python bag/join/window calculation for fixture verification."""
    cutoff, rows = data['cutoff'], []
    for (account,) in sorted(data['accounts']):
        if any(a == account and reason == 'hold' and since <= cutoff and (until is None or until > cutoff)
               for a, since, until, reason in data['bans']):
            continue
        events = [e for e in data['events'] if e[1] == account and e[2] <= cutoff]
        labels = [label for a, label in data['tags'] if a == account]
        eligible = [e for e in events if e[3] != 'void']
        last = max(eligible, key=lambda e: (e[2], e[0])) if eligible else None
        rows.append({'account': account,
            'credit_total': sum(e[4] or 0 for e in events if e[3] == 'credit') * labels.count('active'),
            'joined_events': len(events) * max(1, len(labels)),
            'tag_kinds': len({x for x in labels if x is not None}),
            'latest_event': last[0] if last else None, 'latest_kind': last[3] if last else None})
    return {'rows': rows}


def make_sql(rng, level):
    accounts = [[f'acct_{i:02d}'] for i in range(2+level)]
    data = {'cutoff': 5, 'accounts': accounts, 'events': [], 'tags': [], 'bans': []}
    for i, (account,) in enumerate(accounts):
        for _ in range(rng.randint(0, 2+level)):
            data['events'].append([len(data['events'])+31, account, rng.randint(2, 7),
                rng.choice(['credit', 'debit', 'void']), rng.choice([None, -2, 0, 3, 7, 9])])
        for _ in range(rng.randint(0, 3)):
            data['tags'].append([account, rng.choice(['active', 'active', 'blue', None])])
        if i and rng.random() < .7:
            data['bans'].append([account, rng.randint(3, 7), rng.choice([None, 4, 5, 8]), rng.choice(['hold', 'note'])])
    # Stable edge conditions independent of model outputs: duplicates, tied
    # timestamps, a latest void row, and a present NULL amount.
    a = accounts[0][0]
    data['events'] += [[901, a, 5, 'credit', 3], [902, a, 5, 'void', None]]
    data['tags'] += [[a, 'active'], [a, 'active']]
    text = ('Evaluate the following SQLite 3 query exactly using bag semantics and SQL NULL rules. '
        'Tables contain only the supplied rows; cutoff is a bound parameter. Do not execute tools '
        'or claim execution. Return the query result, not a corrected query.\nSchemas:\n'
        + '\n'.join(SQL_SCHEMA) + '\nData: ' + json.dumps(data, separators=(',', ':'))
        + '\nQuery:\n' + SQL_QUERY)
    return data, text


EVENT_RULES = (
    'Each job starts at epoch 0, state absent, owner null, result null. Events apply in supplied order. '
    'submit(job,v): only v greater than the current epoch resets the job to queued at epoch v, '
    'owner/result null; otherwise ignore. claim(job,v,w): only the current epoch in queued state '
    'becomes running with owner w. transfer(job,v,from,to): only current epoch, running state, '
    'and matching from owner replaces the owner with to; otherwise ignore. '
    'cancel(job,v): only current epoch; queued becomes cancelled, '
    'running becomes cancel_requested; all other states are unchanged. '
    'finish(job,v,w,value): only current epoch, matching owner, and state running or cancel_requested '
    'commits the value and changes state to committed, owner null. A request without its acknowledgement '
    'does not prevent finish. ack_cancel(job,v,w): only current epoch, matching owner, and state '
    'cancel_requested becomes cancelled, owner/result null. Other events are ignored. '
    'Terminal states ignore claim/cancel/finish/ack_cancel; a later higher-epoch submit can reset them. '
    'A stale epoch, wrong owner, duplicate submit or unsolicited acknowledgement grants no authority.'
)


def events_answer(data):
    states = {j: {'epoch': 0, 'state': 'absent', 'owner': None, 'result': None} for j in data['jobs']}
    for op, job, epoch, *args in data['events']:
        s = states[job]
        if op == 'submit':
            if epoch > s['epoch']:
                s.update(epoch=epoch, state='queued', owner=None, result=None)
            continue
        if epoch != s['epoch']:
            continue
        if op == 'claim' and s['state'] == 'queued':
            s.update(state='running', owner=args[0])
        elif op == 'transfer' and s['state'] == 'running' and s['owner'] == args[0]:
            s['owner'] = args[1]
        elif op == 'cancel':
            if s['state'] == 'queued': s.update(state='cancelled', owner=None, result=None)
            elif s['state'] == 'running': s['state'] = 'cancel_requested'
        elif op == 'finish' and s['state'] in {'running', 'cancel_requested'} and s['owner'] == args[0]:
            s.update(state='committed', owner=None, result=args[1])
        elif op == 'ack_cancel' and s['state'] == 'cancel_requested' and s['owner'] == args[0]:
            s.update(state='cancelled', owner=None, result=None)
    return {'jobs': states}


def make_events(rng, level):
    jobs = [f'job_{i:02d}' for i in range(2+level)]
    stories = []
    for j in jobs:
        worker, other = rng.sample(['amber', 'birch', 'coral'], 2)
        story = [['submit', j, 1], ['claim', j, 1, worker], ['cancel', j, 1]]
        terminals = [['finish', j, 1, worker, 'old'], ['ack_cancel', j, 1, worker]]
        rng.shuffle(terminals)
        story += terminals
        if level >= 2:
            story += [['submit', j, 2], ['claim', j, 2, other], ['finish', j, 1, worker, 'late'],
                      ['ack_cancel', j, 2, other], ['claim', j, 2, worker]]
            if level >= 3 and rng.choice([True, False]):
                story += [['transfer', j, 2, other, worker]]
            if rng.choice([True, False]): story += [['cancel', j, 2]]
            if rng.choice([True, False]): story += [['ack_cancel', j, 2, other]]
            if rng.choice([True, False]): story += [['finish', j, 2, other, 'new']]
        if level >= 4 and rng.random() < .65:
            story += [['submit', j, 1], ['submit', j, 3], ['ack_cancel', j, 2, other],
                      ['claim', j, 3, worker], ['cancel', j, 3], ['ack_cancel', j, 3, other]]
            if rng.choice([True, False]): story += [['finish', j, 3, worker, 'third']]
            if rng.choice([True, False]): story += [['ack_cancel', j, 3, worker]]
        stories.append(story)
    events = []
    while any(stories):
        story = rng.choice([s for s in stories if s])
        events.append(story.pop(0))
    data = {'jobs': jobs, 'events': events}
    return data, ('Apply this closed-world async job-state specification. No external observations '
        'are needed.\nRules: ' + EVENT_RULES + '\nInput: ' + json.dumps(data, separators=(',', ':')))


EVIDENCE_RULES = (
    'Select sources separately for each query scope/topic/time. A record is a candidate only if '
    'kind is primary, scope and topic exactly match, issued_at<=time, valid_from<=time, '
    'and time<valid_to when valid_to is non-null. It must also satisfy time<expires_at '
    'and time<revoked_at for each non-null bound. Endpoints are exclusive. '
    'A primary record permanently supersedes the IDs in its supersedes list from its issued_at '
    'onward, but only targets with the same scope AND topic. Its own later expiry/revocation '
    'does not resurrect superseded sources. Derived records cannot supersede or provide primary '
    'support; cross-scope/topic links have no effect. A supersession need not have an otherwise '
    'active replacement. Among surviving candidates: no source means unknown/value null/IDs []; '
    'one distinct value means supported with that value and all surviving source IDs; multiple '
    'distinct values mean conflict/value null with all surviving source IDs. Recency alone '
    'never resolves a conflict. Return source_ids sorted lexicographically.'
)


def evidence_answer(data):
    by_id = {s['id']: s for s in data['sources']}
    answers = {}
    for q in data['queries']:
        time = q['time']
        superseded = {old for new in data['sources'] if new['kind'] == 'primary' and new['issued_at'] <= time
            for old in new['supersedes'] if old in by_id and all(new[k] == by_id[old][k] for k in ('scope', 'topic'))}
        active = [s for s in data['sources'] if s['id'] not in superseded and s['kind'] == 'primary'
            and all(s[k] == q[k] for k in ('scope', 'topic')) and s['issued_at'] <= time
            and s['valid_from'] <= time and all(s[k] is None or time < s[k]
                for k in ('valid_to', 'expires_at', 'revoked_at'))]
        values = {s['value'] for s in active}
        answers[q['id']] = {'status': 'unknown' if not active else 'supported' if len(values) == 1 else 'conflict',
            'value': next(iter(values)) if len(values) == 1 else None,
            'source_ids': sorted(s['id'] for s in active)}
    return {'queries': answers}


def make_evidence(rng, level):
    sources, queries = [], []
    for i in range(2+level):
        scope, topic = f'project_{i%3}', f'field_{i//3}'
        qid = f'q{i:02d}'
        sources.append({'id': qid+'a', 'scope': scope, 'topic': topic, 'kind': 'primary',
            'value': rng.choice(['amber', 'birch', 'coral']), 'issued_at': 1, 'valid_from': 1,
            'valid_to': None, 'expires_at': None, 'revoked_at': None, 'supersedes': []})
        sources.append({'id': qid+'b', 'scope': scope, 'topic': topic, 'kind': rng.choice(['primary', 'derived']),
            'value': rng.choice(['amber', 'birch', 'coral']), 'issued_at': 3, 'valid_from': 2,
            'valid_to': rng.choice([None, 6]), 'expires_at': rng.choice([None, 5]),
            'revoked_at': rng.choice([None, 4]), 'supersedes': [qid+'a'] if rng.choice([True, False]) else []})
        if level >= 3:
            sources.append({'id': qid+'c', 'scope': scope, 'topic': topic, 'kind': 'primary',
                'value': rng.choice(['amber', 'birch', 'coral']), 'issued_at': 4, 'valid_from': 4,
                'valid_to': None, 'expires_at': rng.choice([None, 5]), 'revoked_at': rng.choice([None, 6]),
                'supersedes': [qid+'b']})
        if level >= 5:
            sources.append({'id': qid+'d', 'scope': 'unrelated', 'topic': topic, 'kind': 'primary',
                'value': 'foreign', 'issued_at': 2, 'valid_from': 1, 'valid_to': None,
                'expires_at': None, 'revoked_at': None, 'supersedes': [qid+'a', qid+'b']})
        queries.append({'id': qid, 'scope': scope, 'topic': topic, 'time': rng.choice([3, 4, 5, 6])})
    rng.shuffle(sources)
    data = {'sources': sources, 'queries': queries}
    return data, ('This is a closed-world source-selection exercise under explicit symbolic rules, '
        'not a request to retrieve or certify real-world facts.\nRules: ' + EVIDENCE_RULES
        + '\nInput: ' + json.dumps(data, separators=(',', ':')))


ORACLES = {'logic': logic_answer, 'sql': sql_answer, 'events': events_answer, 'evidence': evidence_answer}
GENERATORS = {'logic': make_logic, 'sql': make_sql, 'events': make_events, 'evidence': make_evidence}
SHAPES = {
    'logic': '{"status":{"v00":"entailed_true|entailed_false|undetermined",...}}',
    'sql': '{"rows":[{"account":"...","credit_total":0,"joined_events":0,"tag_kinds":0,"latest_event":null,"latest_kind":null},...]}',
    'events': '{"jobs":{"job_00":{"epoch":0,"state":"...","owner":null,"result":null},...}}',
    'evidence': '{"queries":{"q00":{"status":"supported|conflict|unknown","value":null,"source_ids":[]},...}}',
}


def make_case(partition, level, domain):
    identity = f'{partition}:L{level}:{domain}'
    seed = int(hashlib.sha256((ORIGIN+':'+identity).encode()).hexdigest()[:16], 16)
    data, text = GENERATORS[domain](random.Random(seed), level)
    expected = ORACLES[domain](data)
    if domain == 'sql' and expected != sql_reference(data):
        raise ValueError('Independent SQL oracles disagree')
    question = text + ('\nRequirements:\n1. Compute every requested output from the supplied data and rules; '
        'retain ambiguity where the rules require it.\n2. Return one complete JSON object with no Markdown, '
        'explanation, omitted entries or additional fields. Use JSON integers for integer fields '
        'and JSON null for absent values. Use this shape: ' + SHAPES[domain])
    if len(question) > 10000:
        raise ValueError('Question exceeded the frozen input bound')
    return {'id': identity, 'partition': partition, 'level': level, 'domain': domain,
            'data': data, 'question': question, 'expected': expected}


def build_fixture():
    return {'schema_version': 1, 'origin': ORIGIN, 'clock': 1794182400,
        'calibration_levels': MAX_CALIBRATION_LEVEL,
        'cases': [make_case(partition, level, domain)
            for partition in ('calibration', 'holdout')
            for level in range(1, MAX_CALIBRATION_LEVEL + (2 if partition == 'holdout' else 1))
            for domain in DOMAINS]}


def grade(case, answer):
    """Case-normalized exact semantic items; whole-answer format kept separate."""
    root = next(iter(case['expected']))
    expected = case['expected'][root]
    result = {'json_valid': False, 'schema_valid': False, 'exact': False, 'item_accuracy': 0.0}
    try:
        value = decode(answer)
    except (ValueError, RecursionError):
        return result
    result['json_valid'] = True
    if not isinstance(value, dict) or set(value) != {root}:
        return result
    actual = value[root]
    if root == 'rows':
        keys = {'account', 'credit_total', 'joined_events', 'tag_kinds', 'latest_event', 'latest_kind'}
        if not isinstance(actual, list) or any(not isinstance(x, dict) or set(x) != keys for x in actual):
            return result
        if any(not isinstance(x['account'], str) or any(type(x[k]) is not int for k in
               ('credit_total', 'joined_events', 'tag_kinds'))
               or x['latest_event'] is not None and type(x['latest_event']) is not int
               or x['latest_kind'] is not None and not isinstance(x['latest_kind'], str) for x in actual):
            return result
        expected_items = Counter(json.dumps(x, sort_keys=True) for x in expected)
        actual_items = Counter(json.dumps(x, sort_keys=True) for x in actual)
        correct = sum((expected_items & actual_items).values())
        denominator = max(sum(expected_items.values()), sum(actual_items.values()), 1)
    else:
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return result
        if root == 'status' and any(not isinstance(x, str) or x not in
                {'entailed_true', 'entailed_false', 'undetermined'} for x in actual.values()):
            return result
        if root in {'jobs', 'queries'} and any(not isinstance(actual[k], dict) or set(actual[k]) != set(expected[k]) for k in expected):
            return result
        if root == 'jobs' and any(type(x['epoch']) is not int or not isinstance(x['state'], str)
                or x['state'] not in {'absent', 'queued', 'running', 'cancel_requested', 'cancelled', 'committed'}
                or any(x[k] is not None and not isinstance(x[k], str) for k in ('owner', 'result')) for x in actual.values()):
            return result
        if root == 'queries':
            if any(not isinstance(x['status'], str) or x['status'] not in {'unknown', 'supported', 'conflict'}
                   or x['value'] is not None and not isinstance(x['value'], str)
                   or not isinstance(x['source_ids'], list) or any(not isinstance(s, str) for s in x['source_ids'])
                   or len(set(x['source_ids'])) != len(x['source_ids']) for x in actual.values()):
                return result
            actual = deepcopy(actual)
            for x in actual.values(): x['source_ids'].sort()
        correct = sum(json.dumps(actual[k], sort_keys=True) == json.dumps(expected[k], sort_keys=True) for k in expected)
        denominator = len(expected)
    result.update(schema_valid=True, exact=actual == expected if root != 'rows' else actual_items == expected_items,
                  item_accuracy=correct/denominator)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refuse to replace a materialized fixture')
    args.output.write_text(json.dumps(build_fixture(), ensure_ascii=False, indent=2)+'\n')
