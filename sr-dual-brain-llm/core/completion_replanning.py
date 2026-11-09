"""User-anchored work decomposition for an existing, bounded integration call.

This chooses a reconstruction strategy, not an answer, verification result,
new model call or permission to enlarge the generation allocation.
"""
from dataclasses import dataclass
import json

from .answer_generation import fingerprint

VERSION = 'completion_replanning_v1'


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
