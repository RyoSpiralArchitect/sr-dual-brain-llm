"""Untouched tasks for a separately frozen critic-cap holdout."""
import json
from pathlib import Path

import frontier_tasks as base
from replanning_stress_tasks import GOALS

ORIGIN = 'prospective_critic_cap_holdout_v1'
SPECS = (('logic_a', 'logic', 4), ('sql_a', 'sql', 7),
         ('events_a', 'events', 6), ('evidence_a', 'evidence', 6),
         ('logic_b', 'logic', 4), ('sql_b', 'sql', 7),
         ('events_b', 'events', 6), ('evidence_b', 'evidence', 6))


def make_case(case_id, domain, level):
    case = base.make_case(ORIGIN + ':' + case_id, level, domain)
    text = case['question'].split('\nRequirements:\n', 1)[0]
    goals = GOALS[domain] + (
        'Using steps 1 and 2, return only the complete JSON answer, without '
        'Markdown, explanations, omitted items or extra fields. Use JSON '
        'integers and nulls where required. Shape: ' + base.SHAPES[domain],)
    case.update(id=case_id, partition='holdout', tier='fresh',
                question=text + '\nRequirements:\n' +
                '\n'.join(f'{i}. {goal}' for i, goal in enumerate(goals, 1)))
    if len(case['question']) > 10000 or any(len(goal) > 400 for goal in goals):
        raise ValueError('Task exceeds bounded question/goal limits')
    return case


def build_fixture():
    return {'schema_version': 1, 'origin': ORIGIN, 'clock': 1794960000,
            'cases': [make_case(*spec) for spec in SPECS]}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refuse to replace a frozen fixture')
    args.output.write_text(json.dumps(build_fixture(), indent=2, ensure_ascii=False) + '\n')
