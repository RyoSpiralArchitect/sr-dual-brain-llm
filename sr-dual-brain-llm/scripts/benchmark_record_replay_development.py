#!/usr/bin/env python3
"""Network-free production-path controls for the independent replay contract.

The provider and semantic verdicts are scripted, not a model-quality estimate.
Historical fixtures, journals and grades are neither read nor changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time
from unittest.mock import patch

import answer_assessment_v3 as assessment
import benchmark_record_replay as previous
from offline_provider_guard import forbid_provider_network
from core.memory_answerability import AnswerabilityGate, OpenAIAnswerability
from core.shared_memory import SharedMemory

bench = previous.bench
ARMS = previous.ARMS
CALL_LIMIT = 192
FIXTURE = bench.base.ROOT / 'examples/record_replay_development.json'
CASE_IDS = ('multiword_quote', 'corrected_japanese', 'corrected_normalized_pronoun',
            'cross_project', 'expired_source', 'superseded_source', 'incomplete_source',
            'repeat_zero_denial', 'partial_approval', 'extra_clause', 'semantic_question')


def validate(fixture):
    if (fixture.get('schema_version') != 1
            or fixture.get('origin') != 'scripted_record_replay_development'
            or fixture.get('support_repeats') != 3
            or tuple(c['id'] for c in fixture.get('cases', [])) != CASE_IDS):
        raise ValueError('Complete fresh development fixture required')
    for case in fixture['cases']:
        if (case.get('requested_operation') not in {'recorded_answer_replay', 'fact_lookup'}
                or set(case.get('support_script', {})) != set(ARMS)
                or set(case.get('expectations', {})) != set(ARMS)
                or set(case.get('semantic_verdicts', {})) != set(assessment.SEMANTIC_CRITERIA)):
            raise ValueError('Explicit operation, scripts and expectations required')
        for arm in ARMS:
            if (len(case['support_script'][arm]) != 3
                    or not set(case['support_script'][arm]) <= {'approve', 'deny', 'partial'}
                    or len(case['expectations'][arm]['support_sources']) != 3):
                raise ValueError('Exactly three predeclared support outcomes required')


class FakeProvider:
    """Deterministic HTTP-shaped replies, using only actual generation context."""
    def __init__(self):
        self.calls = 0
        self.case = self.arm = None
        self.repeat = 0

    def select(self, case, arm, repeat=0):
        self.case, self.arm, self.repeat = case, arm, repeat

    async def _post_json(self, _url, payload, _headers, **_kwargs):
        self.calls += 1
        messages = payload['messages']
        try:
            packet = json.loads(messages[-1]['content'])
        except (ValueError, TypeError):
            packet = {}
        if 'records' in packet:
            mode = self.case['support_script'][self.arm][self.repeat]
            content = {'decisions': [
                {'id': record['id'], 'supported': mode != 'deny',
                 'quote': '' if mode == 'deny' else
                 record['answer'][:1] if mode == 'partial' else record['answer']}
                for record in packet['records']]}
        elif 'answers' in packet:
            # These authored verdicts exercise the assessment contract only.
            # The expected source/answer and legacy task grade are absent from
            # the V3 packet, and are not used to choose an answer here.
            content = {'scores': {label: {
                key: {'verdict': verdict, 'quote': answer[:1] if verdict != 'uncertain' else ''}
                for key, verdict in self.case['semantic_verdicts'].items()}
                for label, answer in packet['answers'].items()}}
        elif 'critic' in messages[0]['content'].lower():
            content = {'verdict': 'ok', 'issues': [], 'fixes': [],
                       'critic_sum': 'No changes in the scripted control.', 'confidence_r': .9}
        else:
            # Exercise normal generation/completion. Do not return a fixture's
            # expected answer when support withheld it from the actual prompt.
            system = messages[0]['content']
            content = 'Please provide a current complete recorded answer.'
            for line in system.splitlines():
                if line.startswith('{"source_question":'):
                    content = json.loads(line)['recorded_answer']
                    break
            else:
                ordinary = re.search(r'^Q:[^\n]* A:([^\n]*)', system, re.MULTILINE)
                if ordinary:
                    content = 'Recorded source: ' + ordinary.group(1)
        return {'choices': [{'finish_reason': 'stop', 'message': {
            'content': content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)}}],
            'model': 'fake-record-replay-v1',
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}


def configuration():
    # No environment credential lookup and no live mode exist in this runner.
    return bench.LLMConfig('openai', 'fake-record-replay-v1', 'FAKE_NO_CREDENTIAL',
        api_base='https://fake.invalid/v1', auto_continue=False, max_continuations=0,
        max_output_tokens=2048, timeout_seconds=60)


def metadata(fixture, config):
    files = sorted(set((bench.base.ROOT / 'core').glob('*.py'))
                   | set((bench.base.ROOT / 'scripts').glob('*.py')))
    return {'experiment': 'scripted_record_replay_development', 'schema_version': 1,
        'call_limit': CALL_LIMIT, 'arms': list(ARMS), 'support_repeats': 3,
        'case_ids': list(CASE_IDS), 'fixture_sha256': bench.digest(fixture),
        'code_sha256': bench.digest([(str(p.relative_to(bench.base.ROOT)),
            hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]),
        'assessment_version': assessment.VERSION,
        'assessment_contract_sha256': bench.digest(assessment.INSTRUCTIONS),
        'config': bench.public_config(config), 'runtime': previous.previous.environment_contract(),
        'provider_kind': 'deterministic_fake', 'usage_kind': 'synthetic_fixed_receipts',
        'legacy_grade_origin': 'scripted_development_control'}


def observed_case(case, row, config, clock):
    """Reconstruct the fixed source snapshot; approval comes from actual exposure."""
    memory = SharedMemory(conversation_context=True,
        answerability_gate=AnswerabilityGate(OpenAIAnswerability(config, config.model)))
    previous.configure(memory, case, row['arm'], clock)
    memory.begin_turn(case['question'], 'probe')
    targets = [t for t in memory.past_qas if t.qid == case['target_qid']]
    context = deepcopy(row['available_context'])
    context['evaluation_time'] = clock[0]
    checks = {key: None for key in assessment.SOURCE_CHECKS}
    result = {**case, 'available_context': context}
    if len(targets) == 1:
        target = targets[0]
        source = memory._answerability_source(target)
        cue = memory.scope_resolver().query(memory.retrieval_question(case['question']))
        checks.update(
            current=memory._available(target, clock[0]) and not target.recalled_answer
                    and target.evidence_status in {'verified', 'unverified'},
            complete=(not source['truncated'] and bool(source['question'].strip())
                      and bool(source['answer'].strip())
                      and len(source['question']) <= memory.answerability_gate.max_question_chars
                      and len(source['answer']) <= memory.answerability_gate.max_answer_chars),
            scope=(not cue.ambiguous and target.retrieval_scope in cue.scopes) if cue.scopes else None,
            source_approved=any(record == source for record in context['approved_records']))
        if case['requested_operation'] == 'recorded_answer_replay':
            result['replay_target'] = {key: getattr(target, key) for key in ('qid', 'retrieval_scope', 'answer')}
    context['source_checks'] = checks
    for record in context['approved_records']:
        matches = [trace for trace in memory.past_qas if memory._answerability_source(trace) == record]
        record['recalled_answer'] = matches[0].recalled_answer if len(matches) == 1 else None
    return result


def expectation_failures(report, fixture):
    failures = []
    cases = {case['id']: case for case in fixture['cases']}
    for row in report['support']:
        expected = cases[row['case']]['expectations'][row['arm']]
        if row['source_qids'] != expected['support_sources'][row['repeat']]:
            failures.append({'phase': 'support', 'case': row['case'], 'arm': row['arm'], 'repeat': row['repeat']})
        if row['intent'] != expected['intent']:
            failures.append({'phase': 'intent', 'case': row['case'], 'arm': row['arm'], 'repeat': row['repeat']})
    for row in report['rows']:
        expected = cases[row['case']]['expectations'][row['arm']]
        if row['status'] != 'ok' or row['source_qids'] != expected['support_sources'][0]:
            failures.append({'phase': 'runtime', 'case': row['case'], 'arm': row['arm']})
        if row['exact_stored_answer'] != expected['exact_stored_answer']:
            failures.append({'phase': 'exact_answer', 'case': row['case'], 'arm': row['arm']})
    for row in report['assessments']:
        expected = cases[row['case']]['expectations'][row['arm']]
        if any(row['diagnostics'][key]['verdict'] != value for key, value in expected['axes'].items()):
            failures.append({'phase': 'assessment', 'case': row['case'], 'arm': row['arm']})
    return failures


def retain_support_error_receipt(row, events):
    """Use the recorded transport category without hiding replay's wrapper.

    The inherited HTTP journal raises RuntimeError for a replayed transport
    exception. The immutable receipt still has the original exception category.
    Invalid returned support JSON has no transport exception and is unchanged.
    """
    error = next((event['error_type'] for event in events if event.get('error_type')), None)
    if error is not None and row['result']['status'] == 'withheld':
        row['local_gate_error_type'] = row['result'].get('error_type')
        row['result'] = {**row['result'], 'error_type': error, 'error_type_basis': 'provider_receipt'}


async def run(output_dir, fixture, config, journal, provider=None):
    validate(fixture)
    report = {'schema_version': 1, 'complete': False, 'metadata': metadata(fixture, config),
              'support': [], 'rows': [], 'assessments': []}
    save = lambda: bench.checkpoint(output_dir / 'report.partial.json', report)
    clock = [fixture['clock']]
    options = dict(blob_root=output_dir, blob_index={}, vision_cache={}, vision_cache_limit=0)
    # JSON and streaming HTTP entries fail locally, including swallowed errors.
    # Journal transport is the FakeProvider object, or None in exact replay.
    with patch.dict(os.environ, previous.previous.environment(), clear=True), \
            patch('time.time', lambda: clock[0]), \
            forbid_provider_network() as transport_audit:
        for index, case in enumerate(fixture['cases']):
            for repeat in range(3):
                for arm in ARMS if (index + repeat) % 2 == 0 else ARMS[::-1]:
                    if provider:
                        provider.select(case, arm, repeat)
                    before = len(journal.events)
                    row = await previous.support_cell(case, arm, repeat, config, journal, clock)
                    retain_support_error_receipt(row, journal.events[before:])
                    row['expected_source_qids'] = case['expectations'][arm]['support_sources'][repeat]
                    report['support'].append(row)
                    save()
            for arm in ARMS if index % 2 == 0 else ARMS[::-1]:
                if provider:
                    provider.select(case, arm)
                first = next(row for row in report['support'] if row['case'] == case['id']
                             and row['arm'] == arm and row['repeat'] == 0)
                row = await previous.runtime_cell(case, arm, first['receipt'], config, journal, clock, options)
                report['rows'].append(row)
                packet_case = observed_case(case, row, config, clock)
                row['available_context'] = packet_case['available_context']
                # An authored failing legacy task verdict is deliberately kept
                # beside equality. It is not a historical model measurement.
                legacy = {'task': {'verdict': case['legacy_task_verdict'],
                                   'quote': (row['answer'] or '')[:1]}}
                before, started = len(journal.events), time.perf_counter()
                try:
                    if row['status'] != 'ok':
                        raise ValueError('Unavailable runtime answer')
                    result = await assessment.assess(packet_case,
                        {'A': {'answer': row['answer'], 'legacy_model_grade': legacy}}, config, journal)
                    diagnostics = result['A']
                except Exception as exc:
                    diagnostics = assessment.diagnostics(packet_case, row['answer'],
                        legacy_model_grade=legacy, model_error=previous.previous.recorded_error(journal.events[before:], exc))
                report['assessments'].append({'case': case['id'], 'arm': arm,
                    'legacy_grade_origin': 'scripted_development_control', 'diagnostics': diagnostics,
                    'usage': bench.usage(journal.events[before:]),
                    'elapsed_ms': (time.perf_counter() - started) * 1000})
                save()
    journal.finish()
    report.update(complete=True, transport_attempts=journal.calls, provider_network_calls=0,
        blocked_provider_requests=transport_audit.blocked_requests,
        usage=bench.usage(journal.events), calls_by_role=dict(Counter(e['role'] for e in journal.events)))
    report['expectation_failures'] = expectation_failures(report, fixture)
    save()
    return report


def comparable(report):
    """Compare receipts/outcomes, excluding durations and local exception wrappers."""
    return {key: [{name: value for name, value in row.items()
                  if name not in {'elapsed_ms', 'local_gate_error_type'}}
                  for row in report[key]] for key in ('support', 'rows', 'assessments')} | {
        key: report[key] for key in ('usage', 'calls_by_role', 'expectation_failures')}


def public_summary(report):
    return {key: report[key] for key in ('schema_version', 'complete', 'metadata', 'transport_attempts',
        'provider_network_calls', 'blocked_provider_requests', 'usage', 'calls_by_role', 'expectation_failures')} | {
        'support': [{key: value for key, value in row.items() if key != 'receipt'} for row in report['support']],
        'rows': [{key: value for key, value in row.items()
                  if key not in {'answer', 'available_context', 'receipt_keys'}} | {
            'receipt_keys_sha256': bench.digest(row['receipt_keys'])} for row in report['rows']],
        'assessments': [{key: value for key, value in row.items() if key != 'diagnostics'} | {
            'diagnostics': assessment.public_diagnostics(row['diagnostics'])} for row in report['assessments']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--journal', type=Path, help='Exact replay journal; transport is removed')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        fixture, config = bench.read_json(FIXTURE), configuration()
        validate(fixture)
        frozen = bench.read_json(args.journal) if args.journal else None
        provider = None if frozen else FakeProvider()
        journal = bench.Journal(config, metadata(fixture, config), frozen=frozen, max_calls=CALL_LIMIT,
            save=lambda data: bench.checkpoint(args.output_dir / 'journal.partial.json', data))
        journal.transport = provider
        report = __import__('asyncio').run(run(args.output_dir, fixture, config, journal, provider))
        bench.write_private(args.output_dir / 'report.json', report)
        bench.write_private(args.output_dir / 'journal.json', frozen or journal.data)
        bench.write_private(args.output_dir / 'summary.json', public_summary(report))
    except BaseException as exc:
        bench.checkpoint(args.output_dir / 'failure.json', {'complete': False, 'error_type': type(exc).__name__})
        print('Development replay incomplete: ' + type(exc).__name__, file=__import__('sys').stderr)
        raise SystemExit(1) from None
    print(json.dumps({'complete': report['complete'], 'provider_network_calls': 0,
        'transport_attempts': report['transport_attempts'], 'usage': report['usage'],
        'expectation_failures': report['expectation_failures']}))
    if report['expectation_failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
