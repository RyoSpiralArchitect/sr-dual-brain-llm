"""Prospective fresh, closed-world tasks for the completion mechanisms study."""
import json
from pathlib import Path

import frontier_tasks as base
from replanning_stress_tasks import GOALS

ORIGIN = 'prospective_completion_mechanisms_v1'
SPECS = (('logic_a', 'logic', 4), ('logic_b', 'logic', 4),
         ('sql', 'sql', 7), ('events', 'events', 6),
         ('evidence', 'evidence', 6))


def make_case(case_id, domain, level):
    # The partition is part of frontier_tasks' seed. Each name is new, so no
    # model has seen this generated input in the earlier frontier/stress runs.
    case = base.make_case(ORIGIN + ':' + case_id, level, domain)
    text = case['question'].split('\nRequirements:\n', 1)[0]
    goals = GOALS[domain] + (
        'Using steps 1 and 2, return only the complete JSON answer, without '
        'Markdown, explanations, omitted items or extra fields. Use JSON '
        'integers and nulls where required. Shape: ' + base.SHAPES[domain],)
    case.update(id=case_id, partition='evaluation', tier='fresh',
                question=text + '\nRequirements:\n' +
                '\n'.join(f'{i}. {goal}' for i, goal in enumerate(goals, 1)))
    if len(case['question']) > 10000 or any(len(goal) > 400 for goal in goals):
        raise ValueError('Task exceeds bounded question/goal limits')
    return case


def build_fixture():
    return {'schema_version': 1, 'origin': ORIGIN, 'clock': 1794528000,
            'cases': [make_case(*spec) for spec in SPECS]}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refuse to replace a frozen fixture')
    args.output.write_text(json.dumps(build_fixture(), indent=2, ensure_ascii=False) + '\n')
