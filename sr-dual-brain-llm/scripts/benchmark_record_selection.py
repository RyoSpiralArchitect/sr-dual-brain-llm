#!/usr/bin/env python3
"""Prospective approval-positive record-selection confirmation; no old rescoring."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import uuid
from unittest.mock import patch

import benchmark_contract_live as pilot

CALL_LIMIT = 64
FIXTURE = pilot.base.ROOT / 'examples/record_selection_confirmation.json'
CASE_IDS = ('english_selection', 'japanese_selection', 'pronoun_selection', 'caller_selection',
            'ineligible_selection', 'unique_quote', 'expired_second', 'factual_lookup', 'extra_clause')


def validate(fixture):
    if (fixture.get('schema_version') != 1 or fixture.get('origin') != 'prospective_record_selection_confirmation'
            or tuple(c['id'] for c in fixture.get('cases', [])) != CASE_IDS):
        raise ValueError('Complete fresh selection-confirmation fixture required')
    for case in fixture['cases']:
        if not case['manual_criterion'] or not case['question'].strip():
            raise ValueError('Prospective question and criterion required')
        records = [pilot.MemoryTrace(**r) for r in case['records']]
        identities = {r.qid for r in records}
        if (len(identities) != len(records) or not set(case['preapprove_qids']) <= identities
                or case.get('target_qid') is not None and case['target_qid'] not in identities):
            raise ValueError('Unambiguous fixture identities required')


def metadata(fixture, config):
    return pilot.metadata(fixture, config) | {
        'call_limit': CALL_LIMIT, 'case_ids': list(CASE_IDS), 'persistence': 'in_memory_engine_sessions',
        'support_preparation': 'once_on_predeclared_qids_under_ordinary_policy_before_runtime',
        'repeat_rule': 'one_fixed_order_no_paid_rescue',
    }


def canonical(report):
    return pilot.canonical(report)


async def cell(case, config, journal, clock, options):
    before = len(journal.events)
    journal.case = case['id']
    session, clients = await pilot.make_session('record-selection:' + case['id'], None, config, journal, clock)
    try:
        memory = session.memory
        for record in case['records']:
            memory.store(pilot.MemoryTrace(**record))
        if 'candidate_limit' in case:
            memory.answerability_gate.candidate_limit = case['candidate_limit']
        if case.get('focus'):
            memory.begin_turn(case['focus'], 'setup-focus')
        # This setup is fixed before outputs. It asks the real support model
        # about the named records without choosing a unique literal target.
        # True/false/unknown receipts are retained; none is a delivery override.
        memory.begin_turn(case['question'], case['id'], case.get('memory_context'))
        query = memory.retrieval_question(case['question'])
        sources = [memory._answerability_source(t) for t in memory.past_qas if t.qid in case['preapprove_qids']]
        journal.case = case['id'] + ':ordinary-support'
        prepared = await memory.answerability_gate.prepare(query, sources) if sources else {'status': 'not_planned'}
        preparation = {
            'result': deepcopy(prepared),
            'decisions': {s['qid']: memory.answerability_gate.decision(query, s) for s in sources},
            'usage': pilot.bench.usage(journal.events[before:]),
        }
        runtime_start = len(journal.events)
        target = next((t for t in memory.past_qas if t.qid == case.get('target_qid')), None)
        row = await pilot.turn(session, clients, case, journal, options)
        pilot.assess_row(case, row, target)
        row['support_preparation'] = preparation
        row['runtime_usage'] = row['usage']
        row['runtime_calls_by_role'] = row['calls_by_role']
        row['usage'] = pilot.bench.usage(journal.events[before:])
        row['calls_by_role'] = dict(pilot.Counter(e['role'] for e in journal.events[before:]))
        row['bounded_source_count'] = len(memory._record_replay_sources(case['question']))
        packets = [r['request'] for r in journal.data['occurrences'][runtime_start:]
                   if r['request']['role'] != 'support']
        leaked = [p['role'] for p in packets if any(
            value in json.dumps(p['payload'], ensure_ascii=False) for value in case.get('forbidden_values', []))]
        row['packet_leak_roles'] = leaked
        row['contracts'].update(
            provider_stages_available=row['stages_available'] and prepared.get('status') != 'withheld',
            expected_condition=(row.get('condition') or {}).get('action') == case.get('expected_action'),
            expected_intent=(row.get('intent') is not None) == case['expected_intent'],
            bounded_source_count=row['bounded_source_count'] == case['expected_bounded_source_count'],
            no_record_text_in_generation=not leaked if case.get('forbidden_values') else None,
            required_source_available=(case['required_source'] in row['source_qids'])
                                       if case.get('required_source') else None)
        return row
    finally:
        await session.close()


async def run(output_dir, fixture, config, journal):
    validate(fixture)
    pilot.base.enforce_runtime()
    report = {'schema_version': 1, 'metadata': metadata(fixture, config), 'complete': False, 'rows': [],
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pilot.base.ROOT.parent, text=True).strip()}
    save = lambda: pilot.bench.checkpoint(output_dir / 'report.partial.json', report)
    save()
    clock = [fixture['clock']]
    options = dict(blob_root=output_dir, blob_index={}, vision_cache={}, vision_cache_limit=0)
    with patch.dict(os.environ, pilot.profile.environment(), clear=True), patch('time.time', lambda: clock[0]), \
            patch('core.unconscious_field.uuid', SimpleNamespace(
                uuid4=lambda: uuid.UUID(int=pilot.base.UUID_RNG.getrandbits(128)))):
        for index, case in enumerate(fixture['cases']):
            clock[0] = fixture['clock'] + index
            row = await cell(case, config, journal, clock, options)
            report['rows'].append(row)
            save()
            print(json.dumps({'case': case['id'], 'status': row['status'], 'contracts': row['contracts'],
                              'support_preparation': row['support_preparation']['decisions'],
                              'usage': row['usage']}), flush=True)
    journal.finish()
    report.update(complete=True, live_provider_calls=journal.calls, usage=pilot.bench.usage(journal.events),
        calls_by_role=dict(pilot.Counter(e['role'] for e in journal.events)),
        served_models=sorted({e['served_model'] for e in journal.events if e.get('served_model')}))
    report['contract_failures'] = [{'case': r['id'], 'contract': key}
        for r in report['rows'] for key, value in r['contracts'].items() if value is False]
    report['canonical_sha256'] = pilot.bench.digest(canonical(report))
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model')
    mode.add_argument('--journal', type=Path)
    parser.add_argument('--replay-api-base')
    parser.add_argument('--max-calls', choices=(CALL_LIMIT,), type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        fixture = pilot.bench.read_json(FIXTURE)
        validate(fixture)
        frozen = pilot.bench.read_json(args.journal) if args.journal else None
        if args.model and subprocess.check_output(['git', 'status', '--porcelain'], cwd=pilot.base.ROOT.parent, text=True).strip():
            raise ValueError('Freeze code and fixtures in a clean commit before paid calls')
        model = args.model or frozen['metadata']['config']['model']
        config = pilot.load_llm_config(default_model=model) if not frozen else pilot.LLMConfig(
            'openai', model, 'REPLAY_ONLY', api_base=args.replay_api_base)
        if config is None or config.provider != 'openai':
            raise ValueError('An OpenAI-compatible configuration is required')
        config = replace(config, model=model, max_output_tokens=2048, timeout_seconds=60,
                         auto_continue=False, max_continuations=0)
        journal = pilot.RawJournal(config, metadata(fixture, config), frozen=frozen, max_calls=CALL_LIMIT,
            save=lambda data: pilot.bench.checkpoint(args.output_dir / 'journal.partial.json', data))
        if frozen:
            with pilot.forbid_provider_network() as audit:
                report = asyncio.run(run(args.output_dir, fixture, config, journal))
            report['unexpected_blocked_http_attempts'] = audit.blocked_requests
        else:
            report = asyncio.run(run(args.output_dir, fixture, config, journal))
        pilot.bench.write_private(args.output_dir / 'report.json', report)
        pilot.bench.write_private(args.output_dir / 'journal.json', frozen or journal.data)
        pilot.bench.write_private(args.output_dir / 'summary.json', pilot.public_summary(report))
        print(json.dumps({'complete': True, 'live_provider_calls': journal.calls, 'usage': report['usage'],
                          'contract_failures': report['contract_failures']}))
    except BaseException as exc:
        pilot.bench.checkpoint(args.output_dir / 'failure.json', {'complete': False, 'error_type': type(exc).__name__})
        print('Record-selection confirmation incomplete: ' + type(exc).__name__, file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
