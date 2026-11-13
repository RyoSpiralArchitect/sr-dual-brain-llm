"""User-anchored work decomposition for an existing, bounded integration call.

This chooses a reconstruction strategy, not an answer, verification result,
new model call or permission to enlarge the generation allocation.
"""
from dataclasses import dataclass
import json

from .answer_generation import fingerprint

VERSION = 'completion_replanning_v1'
WORK_PRODUCT_VERSION = 'completion_work_product_v1'


@dataclass(frozen=True)
class CompletionReplan:
    snapshot_sha256: str
    generation_records: int
    source_status: str
    remaining_output_tokens: int
    goals: tuple

    def matches(self, draft, generation):
        return bool(generation is not None and isinstance(draft, str)
            and self.snapshot_sha256 == fingerprint(draft)
            and self.generation_records == len(generation.records)
            and generation.status(draft) in {'partial', 'empty'})

    def summary(self):
        return {'version': VERSION, 'strategy': 'reconstruct_by_goal_dependencies',
            'source_status': self.source_status, 'goal_count': len(self.goals),
            'remaining_output_tokens': self.remaining_output_tokens,
            'verified_complete': False}

    def prompt(self):
        steps = [{'goal_id': goal.goal_id, 'depends_on': list(goal.depends_on),
                  'work': 'derive_then_check'} for goal in self.goals]
        return ('Completion recovery work plan (internal; not evidence):\n'
            + json.dumps(self.summary() | {'steps': steps}, ensure_ascii=False)
            + '\nThere is no terminally complete candidate. Reconstruct from the full original '
            'question instead of appending to or trusting the incomplete draft. Work through '
            'every goal in dependency order; do not assume goals without declared dependencies '
            'are semantically independent. Carry constraints and unresolved facts between steps. '
            'Check the combined answer against every original condition and the requested output '
            'shape. Keep the work plan private and emit one complete user-facing answer covering '
            'all goals, without invented facts, omitted items or claims of tool execution. '
            'This plan grants no additional output tokens, requests, source or tool authority.')


def propose_replan(plan, generation, draft):
    if plan is None or generation is None or not 2 <= len(plan.goals) <= 4:
        return None
    if not generation.observed(draft) or generation.status(draft) not in {'partial', 'empty'}:
        return None
    if not generation.records:
        return None
    previous = generation.records[-1]
    if (previous['status'] not in {'partial', 'empty'}
            or previous['finish_reason'] not in {'length', 'max_tokens'}):
        return None
    cap = generation.total_cap
    if type(cap) is not int or cap-generation.reserved <= 0:
        return None
    return CompletionReplan(fingerprint(draft), len(generation.records),
        generation.status(draft), cap-generation.reserved, plan.goals)


@dataclass(frozen=True)
class WorkProduct:
    """One terminally completed internal note, never a verified result."""
    goal_id: str
    source_sha256: str
    work: str
    unresolved: tuple[str, ...]

    def handoff(self):
        return ('Internal work product (model generated; unverified, not source evidence):\n'
            + json.dumps({'version': WORK_PRODUCT_VERSION, 'goal_id': self.goal_id,
                'source_sha256': self.source_sha256, 'work': self.work,
                'unresolved': list(self.unresolved)}, ensure_ascii=False)
            + '\nCheck this work against the original request before using it. Keep any '
            'uncertainty visible; do not present the note as verified evidence.')


def work_product_cap(plan, generation, draft, stage_cap, remaining_ms):
    """Reserve at most a quarter of the remaining ceiling for one internal note."""
    if (plan is None or not plan.matches(draft, generation)
            or type(stage_cap) is not int or stage_cap < 1
            or remaining_ms is not None and remaining_ms < 30000):
        return 0
    available = generation.total_cap-generation.reserved
    cap = min(stage_cap//4, available//4)
    return cap if cap >= 256 else 0


def work_product_prompt(question, plan):
    goal = plan.goals[0]
    return ('Original user request:\n' + question + '\n\n'
        'Prepare a bounded internal work product for this one goal only:\n'
        + json.dumps({'goal_id': goal.goal_id, 'goal': goal.text,
            'depends_on': list(goal.depends_on)}, ensure_ascii=False)
        + '\nReturn only a JSON object with exactly goal_id, work, unresolved. '
        'Set goal_id to the supplied ID. work is a concrete derivation, state transition, '
        'query analysis or evidence boundary for that goal, within 3200 characters. '
        'unresolved is an array of at most four short strings. Do not write the final '
        'answer. Do not invent missing source facts or claim tool execution.')


def parse_work_product(result, plan):
    """Shape and terminal checks only; no semantic correctness claim."""
    if result.status != 'complete' or result.error_type is not None or not result.text:
        return None, 'incomplete_generation'
    if len(result.text) > 5000:
        return None, 'invalid_schema'
    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate work product key')
            value[key] = item
        return value
    try:
        value = json.loads(result.text, object_pairs_hook=unique_pairs)
    except (TypeError, ValueError, RecursionError):
        return None, 'invalid_json'
    if (not isinstance(value, dict) or set(value) != {'goal_id', 'work', 'unresolved'}
            or value['goal_id'] != plan.goals[0].goal_id
            or not isinstance(value['work'], str) or not 1 <= len(value['work'].strip()) <= 3200
            or not isinstance(value['unresolved'], list) or len(value['unresolved']) > 4
            or any(not isinstance(item, str) or not 1 <= len(item.strip()) <= 200
                   for item in value['unresolved'])):
        return None, 'invalid_schema'
    return WorkProduct(value['goal_id'], plan.snapshot_sha256,
        value['work'].strip(), tuple(item.strip() for item in value['unresolved'])), 'available'
