#!/usr/bin/env python3
"""Fixed, model-free critic failure controls through the production engine.

The raw journal deliberately leaves completion validation to production code.
Scripted token receipts and review verdicts do not estimate model quality/cost.
Historical studies and grades are neither read nor modified.
"""
from __future__ import annotations

import argparse
import asyncio
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from core.llm_client import LLMClient, LLMConfig
from core.memory_answerability import AnswerabilityGate, OpenAIAnswerability
from core.policy import TORCH_AVAILABLE
from core.shared_memory import MemoryTrace
from core.thalamic_reticular_nucleus import DEFAULT_REGISTRY
from engine_stdio import EngineSession, _handle_process
from benchmark_frozen_drafts import digest, read_json, write_private
from benchmark_module_gating import usage
from benchmark_controlled_memory import checkpoint
from offline_provider_guard import forbid_provider_network

CLOCK = 1793836800.
CALL_LIMIT = 128
REPLAYABLE_ERRORS = frozenset({'OSError', 'CancelledError', 'ValueError'})
QUESTION = 'What does the Project Cedar record say about retention?'
SOURCE = 'Recorded policy: normal retention is 11 days; urgent retention is 2 days.'
CLEAN = {'verdict': 'ok', 'issues': [], 'fixes': []}
ISSUES = {'verdict': 'issues', 'issues': ['State that the policy comes from the supplied record.'],
          'fixes': ['Attribute both retention periods to the supplied record.']}
# These expectations are authored before running the controls. A failure is
# retained as a row, including when the initial answer remains available.
CASES = (
    ('clean', 'critic_initial', 'clean', 'ok'),
    ('empty', 'critic_initial', 'empty', 'empty_response'),
    ('length', 'critic_initial', 'length', 'incomplete_response'),
    ('filtered', 'critic_initial', 'filtered', 'incomplete_response'),
    ('missing_finish', 'critic_initial', 'missing_finish', 'incomplete_response'),
    ('missing_fields', 'critic_initial', 'missing_fields', 'invalid_schema'),
    ('scalar_issue', 'critic_initial', 'scalar_issue', 'invalid_schema'),
    ('transport_error', 'critic_initial', 'error', 'provider_error'),
    ('unknown_usage', 'critic_initial', 'unknown_usage', 'ok'),
    ('repair_clean', 'critic_repair', 'clean', 'ok'),
    ('repair_empty', 'critic_repair', 'empty', 'empty_response'),
    ('repair_filtered', 'critic_repair', 'filtered', 'incomplete_response'),
    ('repair_schema', 'critic_repair', 'scalar_issue', 'invalid_schema'),
    ('verify_empty', 'critic_verify', 'empty', 'empty_response'),
    ('verify_schema', 'critic_verify', 'scalar_issue', 'invalid_schema'),
    ('integration_error', 'integration', 'error', None),
    ('cancel_critic', 'critic_initial', 'cancel', None),
    ('cancel_integration', 'integration', 'cancel', None),
)
STAGE = ContextVar('critic_development_stage', default='untracked')


def configuration():
    return LLMConfig('openai', 'fake-critic-contract-v1', 'FAKE_NO_CREDENTIAL',
        api_base='https://fake.invalid/v1', auto_continue=False, max_continuations=0,
        max_output_tokens=2048, timeout_seconds=10)


def environment():
    env = {spec.mode_env: 'off' for spec in DEFAULT_REGISTRY.values()}
    env.update(DUALBRAIN_MEMORY_CONVERSATION_CONTEXT='1',
        DUALBRAIN_SYSTEM2_ROUND_TARGET_MIN='2', DUALBRAIN_SYSTEM2_LOW_SIGNAL_FILTER='0',
        DUALBRAIN_CRITIC_MICRO_FALLBACK='0', DUALBRAIN_CRITIC_MICRO_SANITY_OVERRIDE='0',
        DUALBRAIN_CRITIC_JSON_REPAIR_CALL='1', REASONING_DIAL='evaluative')
    return env


def metadata():
    files = sorted([*(ROOT / 'core').glob('*.py'), *(ROOT / 'scripts').glob('*.py')])
    return {'version': 1, 'origin': 'scripted_critic_failure_development',
        'cases': [list(case) for case in CASES], 'clock': CLOCK, 'call_limit': CALL_LIMIT,
        'fixture_sha256': digest([QUESTION, SOURCE, CLEAN, ISSUES, CASES]),
        'code_sha256': digest([(str(p.relative_to(ROOT)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]),
        'runtime': environment(), 'model': configuration().model,
        'dependencies': {'python': list(sys.version_info[:3]), 'numpy': np.__version__,
                         'torch': str(__import__('torch').__version__) if TORCH_AVAILABLE else None},
        'usage_kind': 'synthetic_receipts', 'provider_kind': 'deterministic_fake'}


class RawJournal:
    """Ordered raw responses/errors; never sanitize a failed response into success."""
    def __init__(self, *, frozen=None, save=None):
        self.metadata = metadata()
        if frozen is not None:
            if (not isinstance(frozen, dict) or frozen.get('metadata') != self.metadata
                    or frozen.get('complete') is not True):
                raise ValueError('Frozen critic journal differs or is incomplete')
            responses = frozen.get('responses')
            if (not isinstance(responses, list) or len(responses) > CALL_LIMIT
                    or frozen.get('responses_sha256') != digest(responses)):
                raise ValueError('Frozen response integrity differs')
            for row in responses:
                if (not isinstance(row, dict) or not isinstance(row.get('request'), dict)
                        or set(row) not in ({'request', 'response'}, {'request', 'error_type'})
                        or 'error_type' in row and (not isinstance(row['error_type'], str)
                            or row['error_type'] not in REPLAYABLE_ERRORS)):
                    raise ValueError('Unsupported frozen receipt envelope or failure category')
        self.frozen, self.rows, self.events = frozen, [], []
        self.save = save
        self.position = self.transport_attempts = 0
        self.mismatch = False
        self.recovering = False
        self.cancel_started = asyncio.Event()

    async def post(self, case, role, url, payload, _headers, **_kwargs):
        request = {'case': case[0], 'phase': 'recovery' if self.recovering else 'primary',
                   'role': role, 'endpoint_sha256': digest(url), 'payload': payload}
        if self.position >= CALL_LIMIT:
            self.mismatch = True
            raise ValueError('Fixed attempt budget exceeded')
        if role not in {'support', 'draft', 'integration', 'critic_initial', 'critic_repair', 'critic_verify', 'detail'}:
            self.mismatch = True
            raise ValueError('Untracked production call')
        position = self.position
        self.position += 1
        if self.frozen is not None:
            if position >= len(self.frozen['responses']) or self.frozen['responses'][position]['request'] != request:
                self.mismatch = True
                raise ValueError('Exact request occurrence differs')
            row = deepcopy(self.frozen['responses'][position])
        else:
            self.transport_attempts += 1
            row = {'request': request}
        try:
            if self.frozen is None:
                row['response'] = await self.fake_response(case, role, payload)
            elif row.get('error_type') == 'CancelledError':
                self.cancel_started.set()
                await asyncio.Event().wait()  # The same caller cancellation ends this occurrence.
            elif row.get('error_type') == 'OSError':
                raise OSError('Recorded scripted transport failure')
            elif row.get('error_type') == 'ValueError':
                raise ValueError('Recorded scripted input failure')
            elif row.get('error_type'):
                self.mismatch = True
                raise ValueError('Unsupported recorded failure category')
            return deepcopy(row['response'])
        except BaseException as exc:
            if self.frozen is None:
                row['error_type'] = type(exc).__name__
                if row['error_type'] not in REPLAYABLE_ERRORS:
                    # A production fallback may catch this harness failure, but
                    # it must not turn an unreplayable journal into a complete run.
                    self.mismatch = True
            raise
        finally:
            self.rows.append(row)
            response = row.get('response')
            receipt = response.get('usage') if isinstance(response, dict) else None
            receipt = receipt if isinstance(receipt, dict) else {}
            tokens = lambda key: receipt[key] if type(receipt.get(key)) is int and receipt[key] >= 0 else None
            self.events.append({'role': role, 'error_type': row.get('error_type'),
                'input_tokens': tokens('prompt_tokens'), 'output_tokens': tokens('completion_tokens')})
            if self.save:
                self.save({'metadata': self.metadata, 'complete': False, 'responses': self.rows,
                           'responses_sha256': digest(self.rows)})

    async def fake_response(self, case, role, payload):
        _, target, fault, _ = case
        if self.recovering:
            target, fault = None, None
        mode = fault if role == target else 'clean'
        if target == 'critic_repair' and role == 'critic_initial':
            mode = 'unstructured'
        if target in {'critic_verify', 'integration'} and role == 'critic_initial':
            mode = 'issues'
        if mode == 'cancel':
            self.cancel_started.set()
            await asyncio.Event().wait()
        if mode == 'error':
            raise OSError('Scripted provider unavailable')
        if role == 'support':
            packet = json.loads(payload['messages'][-1]['content'])
            content = json.dumps({'decisions': [{'id': r['id'], 'supported': True, 'quote': r['answer']}
                                               for r in packet['records']]})
        elif role in {'draft', 'integration'}:
            if not any(SOURCE in str(message.get('content', '')) for message in payload['messages']):
                raise ValueError('Scripted source is absent from the actual generation context')
            content = SOURCE
        elif role == 'detail':
            content = 'The supplied record contains both retention periods.'
        else:
            content = {'empty': '', 'missing_fields': '{"verdict":"ok"}',
                'scalar_issue': '{"verdict":"ok","issues":"The record contradicts the draft.","fixes":[]}',
                'unstructured': 'The attribution needs review.'}.get(mode, json.dumps(ISSUES if mode == 'issues' else CLEAN))
        reason = {'length': 'length', 'filtered': 'content_filter', 'missing_finish': None}.get(mode, 'stop')
        response = {'choices': [{'message': {'content': content}, 'finish_reason': reason}],
                    'model': configuration().model}
        if mode != 'unknown_usage':
            response['usage'] = {'prompt_tokens': 10, 'completion_tokens': 5}
        return response

    def finish(self):
        if self.mismatch or self.frozen is not None and self.position != len(self.frozen['responses']):
            raise ValueError('Unconsumed response occurrences')
        return {'metadata': self.metadata, 'complete': True, 'responses': self.rows,
                'responses_sha256': digest(self.rows)}


class RecordedClient:
    def __init__(self, config, journal, case):
        self.client = LLMClient(config)
        async def post(url, payload, headers, **kwargs):
            role = STAGE.get()
            if role.startswith('critic_') and payload['messages'][-1]['content'].startswith('Your previous response'):
                role = 'critic_repair'
            return await journal.post(case, role, url, payload, headers, **kwargs)
        self.client._post_json = post

    async def complete(self, *args, **kwargs):
        return await self.client.complete(*args, **kwargs)


def track_stage(model, method, role, reviews):
    original = getattr(model, method)
    count = 0
    async def tracked(*args, **kwargs):
        nonlocal count
        count += 1
        stage = role if role != 'critic_initial' or count == 1 else 'critic_verify'
        token = STAGE.set(stage)
        try:
            result = await original(*args, **kwargs)
            if role == 'critic_initial':
                reviews.append({'stage': stage, **{key: deepcopy(result.get(key)) for key in
                    ('critic_status', 'critic_kind', 'critic_review_valid', 'critic_external_review_valid',
                     'critic_validation', 'verdict')}})
            return result
        finally:
            STAGE.reset(token)
    setattr(model, method, tracked)


async def run_cell(case, journal, output_dir):
    journal.recovering = False
    seed = int(digest(case[0])[:8], 16)
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        import torch
        torch.manual_seed(seed)
    config = configuration()
    session = await EngineSession.create('critic-failure:' + case[0], left_llm_config=config,
        right_llm_config=config, executive_llm_config=config)
    before = len(journal.events)
    reviews = []
    for model in (session.left, session.right):
        model._llm_client = RecordedClient(config, journal, case)
    backend = OpenAIAnswerability(config, config.model)
    async def support_post(url, payload, headers, **kwargs):
        return await journal.post(case, 'support', url, payload, headers, **kwargs)
    backend.client._post_json = support_post
    session.memory.answerability_gate = AnswerabilityGate(backend, max_question_chars=2048, max_answer_chars=2048)
    session.memory.store(MemoryTrace('Project Cedar retention policy', SOURCE, qid='cedar-current',
        retrieval_scope='cedar', scope_aliases=('Cedar',), timestamp=CLOCK - 1))
    track_stage(session.left, 'generate_answer', 'draft', reviews)
    track_stage(session.left, 'integrate_info_async', 'integration', reviews)
    track_stage(session.right, 'criticise_reasoning', 'critic_initial', reviews)
    track_stage(session.right, 'deepen', 'detail', reviews)
    params = {'question': QUESTION, 'qid': 'probe-' + case[0], 'leading_brain': 'left',
              'system2_mode': 'on', 'executive_mode': 'off', 'executive_observer_mode': 'off'}
    journal.cancel_started = asyncio.Event()
    task = asyncio.create_task(_handle_process(session, params, blob_root=output_dir,
        blob_index={}, vision_cache={}, vision_cache_limit=0))
    row = {'case': case[0], 'answer_available': False, 'answer_origin': None, 'cancelled': False,
           'answer_completion_confirmed': False, 'error_type': None, 'reviews': reviews,
           'next_turn_reusable': None}
    end = before
    try:
        if case[2] == 'cancel':
            await asyncio.wait_for(journal.cancel_started.wait(), 3.)
            task.cancel()
        result = await task
        answer = result['answer']
        row.update(answer_available=bool(isinstance(answer, str) and answer.strip()),
            answer_sha256=digest(answer), exact_stored_text=answer == SOURCE)
        # Identify the returned provider text separately from review validity.
        matching = [event['role'] for receipt, event in zip(journal.rows[before:], journal.events[before:])
                    if event['role'] in {'draft', 'integration'} and 'response' in receipt
                    and receipt['response']['choices'][0]['message']['content'] == answer]
        row['answer_origin'] = matching[-1] if matching else 'local_fallback'
        row['answer_completion_confirmed'] = any(
            receipt['request']['role'] in {'draft', 'integration'}
            and len(receipt.get('response', {}).get('choices', [])) == 1
            and receipt['response']['choices'][0].get('finish_reason') == 'stop'
            and receipt['response']['choices'][0]['message'].get('content') == answer
            for receipt in journal.rows[before:])
    except asyncio.CancelledError:
        row.update(cancelled=True, error_type='CancelledError')
    except Exception as exc:
        row['error_type'] = type(exc).__name__
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await session.callosum.wait_idle()
        row['source_qids'] = [t.qid for t in session.memory.retrieve_related_traces(QUESTION)]
        source = next(t for t in session.memory.past_qas if t.qid == 'cedar-current')
        checked = session.memory._answerability_source(source)
        row['source_context'] = {'current': session.memory._available(source, CLOCK),
            'complete': bool(checked['question'] and checked['answer'] and not checked['truncated']
                             and len(checked['question']) <= 2048 and len(checked['answer']) <= 2048),
            'scope': source.retrieval_scope,
            'source_approved': row['source_qids'] == ['cedar-current'], 'verifies_recorded_claims': False}
        row['occupied_after_turn'] = session.callosum.snapshot()['occupied']
        row['trace_published'] = params['qid'] in session.trace_cache
        # Cancellation clears turn-local state; it must not create a QA/source.
        row['stored_probe'] = any(t.qid == params['qid'] for t in session.memory.past_qas)
        end = len(journal.events)
        if row['cancelled']:
            journal.recovering = True
            class SkipConsult:
                def decide(self, _state):
                    return 0
            session.controller.policy = SkipConsult()
            followup = await _handle_process(session, {**params, 'qid': 'recovery-' + case[0],
                'system2_mode': 'off'}, blob_root=output_dir, blob_index={}, vision_cache={}, vision_cache_limit=0)
            row['next_turn_reusable'] = bool(followup['answer'] == SOURCE
                and session.callosum.snapshot()['occupied'] == 0
                and params['qid'] not in session.trace_cache)
            row['recovery_usage'] = usage(journal.events[end:])
        await session.close()
    row['usage'] = usage(journal.events[before:end])
    row['draft_completion_confirmed'] = any(
        receipt['request']['role'] == 'draft'
        and receipt.get('response', {}).get('choices', [{}])[0].get('finish_reason') == 'stop'
        for receipt in journal.rows[before:end])
    row['receipts'] = [dict(event) for event in journal.events[before:end]]
    row['last_review_valid'] = reviews[-1]['critic_review_valid'] if reviews else None
    row['last_external_review_valid'] = reviews[-1]['critic_external_review_valid'] if reviews else None
    expected = case[3]
    row['expectation_failures'] = []
    if expected is not None and (not reviews or reviews[-1]['critic_status'] != expected):
        row['expectation_failures'].append('critic_status')
    if expected is not None and row['last_external_review_valid'] != (expected == 'ok'):
        row['expectation_failures'].append('review_validity')
    if case[2] == 'cancel':
        if (not row['cancelled'] or row['answer_available'] or row['stored_probe']
                or row['trace_published'] or not row['next_turn_reusable']):
            row['expectation_failures'].append('cancelled_turn_published')
    elif not row['answer_available']:
        row['expectation_failures'].append('retained_answer_unavailable')
    if row['occupied_after_turn']:
        row['expectation_failures'].append('undrained_consultation')
    return row


async def run(output_dir, *, frozen=None):
    journal = RawJournal(frozen=frozen,
        save=lambda data: checkpoint(output_dir / 'journal.partial.json', data))
    rows = []
    with patch.dict(os.environ, environment(), clear=True), patch('time.time', lambda: CLOCK), \
            forbid_provider_network() as transport_audit:
        for case in CASES:
            rows.append(await run_cell(case, journal, output_dir))
            checkpoint(output_dir / 'report.partial.json', {'metadata': journal.metadata,
                'complete': False, 'rows': rows, 'usage': usage(journal.events)})
    recorded = journal.finish()
    report = {'metadata': journal.metadata, 'complete': True, 'rows': rows,
        'transport_attempts': journal.transport_attempts, 'provider_network_calls': 0,
        'blocked_provider_requests': transport_audit.blocked_requests,
        'usage': usage(journal.events),
        'expectation_failures': [{'case': row['case'], 'failures': row['expectation_failures']}
                                 for row in rows if row['expectation_failures']]}
    return report, recorded


def comparable(report):
    return {key: value for key, value in report.items() if key != 'transport_attempts'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--journal', type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        report, journal = asyncio.run(run(args.output_dir, frozen=read_json(args.journal) if args.journal else None))
    except BaseException as exc:
        checkpoint(args.output_dir / 'failure.json', {'complete': False, 'error_type': type(exc).__name__})
        raise SystemExit(1) from None
    write_private(args.output_dir / 'report.json', report)
    write_private(args.output_dir / 'journal.json', journal)
    print(json.dumps({key: report[key] for key in ('complete', 'transport_attempts', 'provider_network_calls',
                                                  'blocked_provider_requests', 'usage', 'expectation_failures')}))
    if report['expectation_failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
