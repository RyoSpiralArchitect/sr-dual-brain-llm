"""Rubric integrity and failure-boundary checks without provider calls."""
from copy import deepcopy
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_persistent_answers as bench


def good_grades():
    return {'scores': {label: {criterion: {'verdict': 'pass', 'quote': answer}
                             for criterion in bench.CRITERIA} for label, answer in {'A': '記録なし', 'B': '7日'}.items()}}


@pytest.mark.parametrize('fault', ['missing-label', 'extra-field', 'nested-label', 'wrong-quote', 'duplicate'])
def test_entire_grader_packet_rejected_on_invalid_structure_or_evidence(fault):
    packet = good_grades()
    if fault == 'missing-label': del packet['scores']['B']
    if fault == 'extra-field': packet['scores']['A']['task']['other'] = True
    if fault == 'nested-label': packet['scores']['A']['B'] = packet['scores'].pop('B')
    if fault == 'wrong-quote': packet['scores']['B']['task']['quote'] = 'invented quote'
    raw = json.dumps(packet, ensure_ascii=False)
    if fault == 'duplicate': raw = raw.replace('"verdict": "pass"', '"verdict": "pass", "verdict": "fail"', 1)
    with pytest.raises(ValueError):
        bench.validate_grades(raw, {'A': '記録なし', 'B': '7日'})


def test_schema_and_parser_require_all_labels_and_distinguish_criteria():
    schema = bench.grade_schema(['A', 'B'])['json_schema']
    assert schema['strict']
    assert schema['schema']['properties']['scores']['required'] == ['A', 'B']
    packet = good_grades()
    packet['scores']['A']['clarification']['verdict'] = 'fail'
    result = bench.validate_grades(json.dumps(packet), {'A': '記録なし', 'B': '7日'})
    assert result['A']['task']['verdict'] == 'pass' and result['A']['clarification']['verdict'] == 'fail'


@pytest.mark.asyncio
async def test_hidden_provider_fallback_aborts_before_scoring_or_a_later_turn(monkeypatch):
    client = SimpleNamespace(failed=0)
    async def fallback(*args, **kwargs):
        client.failed += 1
        return {'answer': 'heuristic fallback'}
    monkeypatch.setattr(bench, '_handle_process', fallback)
    with pytest.raises(RuntimeError, match='episode aborted'):
        await bench.process(None, [client], {}, SimpleNamespace(events=[]), {})


def test_prospective_fixture_contains_disjoint_scopes_and_both_reference_cases():
    fixture = bench.read_json(bench.FIXTURE)
    bench.validate_fixture(fixture)
    scopes = {}
    for ep in fixture['episodes']:
        scopes.setdefault(ep['split'], set()).update(w['memory_context']['id'] for w in ep['writes'])
        assert {p['id'] for p in ep['probes']} >= {'named-unknown', 'ambiguous', 'cold-conditions'}
        assert ep['probes'][-1]['restart_before']
    assert scopes['development'].isdisjoint(scopes['evaluation'])
