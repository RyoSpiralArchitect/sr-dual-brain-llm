"""A caller's record choice narrows identity, never lifecycle or approval."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
from core.memory_answerability import AnswerabilityGate, REPLAY_POLICY, policy_for
from core.prefrontal_cortex import PrefrontalCortex
from core.record_selection import RecordSelectionCodec
from core.shared_memory import MemoryTrace, SharedMemory
from engine_stdio import EngineSession, _dispatch_request, _handle_process, _handle_process_stream
from test_missing_evidence import Decisions

Q = 'Repeat the recorded answer verbatim.'
META = {'id': 'alpha', 'aliases': ['Alpha', 'アルファ'], 'remember': False}


def sources():
    return [MemoryTrace('Alpha retention policy?', 'PRIVATE_PERIOD_14_DAYS', qid='period',
                seed_id='period-seed', retrieval_scope='alpha', scope_aliases=('Alpha', 'アルファ')),
            MemoryTrace('Alpha backup location?', 'PRIVATE_BACKUP_KYOTO', qid='backup',
                seed_id='backup-seed', retrieval_scope='alpha', scope_aliases=('Alpha', 'アルファ'))]


def memory():
    m = SharedMemory(conversation_context=True, answerability_gate=AnswerabilityGate(Decisions(True)))
    for t in sources():
        m.store(t)
    return m


def ref_for(m, label='backup'):
    return next(r['record_ref'] for r in m.record_choices('alpha')['choices'] if label in r['label'])


def begin(m, ref, question=Q, **metadata):
    m.begin_turn(question, 'read-probe', {**META, 'record_ref': ref, **metadata})


def test_inventory_is_read_only_and_does_not_disclose_answers_or_approve_sources():
    m = memory()
    before = list(m.past_qas)
    page = m.record_choices('alpha', limit=1)
    assert page['total'] == 2 and page['next_offset'] == 1
    assert len(m.record_choices('alpha', offset=1)['choices']) == 1
    assert m.record_choices('unknown')['total'] == 0
    assert 'PRIVATE_' not in json.dumps(page) and 'answer' not in page['choices'][0]
    assert not m.answerability_gate.cache and not m.answerability_gate.backend.requests
    assert m.past_qas == before and m._turn_question is None
    assert not m.pending_memory_writes and not m.pending_memory_updates


@pytest.mark.parametrize('limit,offset', [(0, 0), (21, 0), (True, 0), (1, -1), (1, False)])
def test_invalid_pagination_is_rejected(limit, offset):
    with pytest.raises(ValueError):
        memory().record_choices('alpha', limit=limit, offset=offset)


def test_all_current_records_count_and_ineligible_identity_is_disabled():
    m = memory()
    m.past_qas[1] = replace(m.past_qas[1], answer='x' * 2049)
    m.store(replace(sources()[0], qid='expired', valid_until=0))
    m.store(replace(sources()[0], qid='derived', recalled_answer=True))
    page = m.record_choices('alpha')
    assert page['total'] == 2
    bad = next(c for c in page['choices'] if 'backup' in c['label'])
    assert bad['record_ref'] is None and bad['reason'] == 'incomplete_source'
    m.store(replace(sources()[0], answer='Another', seed_id=None))
    assert all(c['record_ref'] is None for c in m.record_choices('alpha')['choices'])


def test_choice_targets_exact_source_before_cap_and_still_needs_typed_approval():
    m = memory()
    ref = ref_for(m)
    m.answerability_gate.candidate_limit = 1
    # A generic positive receipt is insufficient for the selected literal call.
    asyncio.run(m.answerability_gate.prepare(Q, [m._answerability_source(m.past_qas[1])]))
    begin(m, ref)
    assert m.replay_resolution(Q).intent.source_qid == 'backup'
    assert not m.retrieve_related(Q)
    assert [t.qid for t in m.answerability_candidates(Q)] == ['backup']
    asyncio.run(m.prepare_retrieval(Q))
    assert policy_for(m.answerability_gate.backend.requests[-1]) == REPLAY_POLICY
    assert [t.qid for t in m.retrieve_related_traces(Q)] == ['backup']
    assert 'PRIVATE_BACKUP_KYOTO' in m.retrieve_related(Q) and 'PRIVATE_PERIOD' not in m.retrieve_related(Q)
    assert m.evidence_for_seeds(['period-seed'], question=Q) == []
    assert m.selection_status(Q) == {'status': 'bound', 'action': None}


@pytest.mark.parametrize('change', ['expired', 'superseded', 'derived', 'body', 'scope', 'duplicate', 'removed'])
def test_changed_selection_never_substitutes_another_record(change):
    m = memory(); ref = ref_for(m); begin(m, ref)
    asyncio.run(m.prepare_retrieval(Q))
    old_context = m.retrieve_related(Q)
    if change == 'removed':
        m.past_qas.pop(1)
    elif change == 'duplicate':
        m.store(replace(m.past_qas[1], seed_id=None))
    else:
        fields = {'expired': {'valid_until': 0}, 'superseded': {'evidence_status': 'superseded'},
                  'derived': {'recalled_answer': True}, 'body': {'answer': 'PRIVATE_CHANGED_VALUE'},
                  'scope': {'retrieval_scope': 'beta', 'scope_aliases': ('Beta',)}}[change]
        m.past_qas[1] = replace(m.past_qas[1], **fields)
    assert m.replay_resolution(Q).binding_status == 'stale_selection'
    assert not m.retrieve_related(Q) and not m.evidence_for_seeds(['backup-seed'], question=Q)
    assert m.missing_evidence(Q).action == 'reselect_record'
    context = PrefrontalCortex.refresh_record_replay_context(m, Q, old_context)
    assert 'PRIVATE_' not in context and 'reselect_record' in context and ref not in context


@pytest.mark.parametrize('kind', ['tampered', 'other_instance', 'other_scope'])
def test_reference_cannot_cross_context_or_be_modified(kind):
    m = memory(); ref = ref_for(m)
    if kind == 'tampered':
        ref = ref[:-1] + ('0' if ref[-1] != '0' else '1')
    elif kind == 'other_instance':
        m = memory()
    elif kind == 'other_scope':
        m.store(MemoryTrace('Beta policy?', 'PRIVATE_BETA', qid='beta', retrieval_scope='beta', scope_aliases=('Beta',)))
    begin(m, ref, **({'id': 'beta', 'aliases': ['Beta']} if kind == 'other_scope' else {}))
    assert m.replay_resolution(Q).binding_status == 'invalid_selection'
    assert not m.answerability_candidates(Q) and not m.retrieve_related(Q)
    assert m.missing_evidence(Q).action == 'reselect_record'


@pytest.mark.parametrize('metadata,question', [
    ({'remember': True}, Q), ({'supersedes': ['period']}, Q),
    ({'record_ref': ''}, Q), ({'record_ref': 'x' * 4097}, Q),
    ({}, 'What is Alpha retention policy?'), ({}, Q + ' Then summarize it.'),
])
def test_selection_is_only_an_explicit_read_only_literal_request(metadata, question):
    m = memory()
    with pytest.raises(ValueError):
        begin(m, ref_for(m), question, **metadata)
    assert not m.pending_memory_writes and all(t.evidence_status == 'unverified' for t in m.past_qas)


def test_selection_is_not_sticky_and_refs_survive_only_unchanged_same_instance_reload():
    m = memory(); ref = ref_for(m); begin(m, ref)
    m.abort_turn(None)
    assert m._turn_record_ref is None
    begin(m, ref)
    m.restore_committed_traces(list(m.past_qas))
    assert m._turn_record_ref is None
    begin(m, ref)
    assert m.replay_resolution(Q).binding_status == 'bound'
    m.begin_turn(Q, 'next', META)
    assert m.replay_resolution(Q).binding_status == 'ambiguous_record'


@pytest.mark.asyncio
async def test_source_change_during_awaited_approval_invalidates_the_reference():
    m = memory(); ref = ref_for(m); begin(m, ref)
    ready, release = asyncio.Event(), asyncio.Event()
    original = m.answerability_gate.backend.evaluate
    async def delayed(request):
        ready.set(); await release.wait(); return await original(request)
    m.answerability_gate.backend.evaluate = delayed
    pending = asyncio.create_task(m.prepare_retrieval(Q))
    await ready.wait()
    m.past_qas[1] = replace(m.past_qas[1], answer='PRIVATE_CHANGED_VALUE')
    release.set(); await pending
    assert m.replay_resolution(Q).binding_status == 'stale_selection' and not m.retrieve_related(Q)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['ASK_DETAIL', 'ASK_CRITIC', 'ASK_LEAD'])
@pytest.mark.parametrize('change', ['expire', 'unchanged'])
async def test_selected_record_is_rechecked_at_actual_queued_worker_boundary(kind, change):
    from test_replay_worker_context import consume_after_change
    m = memory(); ref = ref_for(m, 'retention'); begin(m, ref)
    await m.prepare_retrieval(Q)
    request = {'type': kind, 'qid': 'queued-choice', 'question': Q,
        'draft': 'Prior draft without source text.', 'draft_sum': 'Prior draft without source text.',
        'context': m.retrieve_related(Q), 'memory_evidence_format': 'complete_v1',
        'memory_evidence': m.evidence_for_seeds(['period-seed'], question=Q)}
    captured = await consume_after_change(m, request, change)
    combined = captured['system'] + '\n' + captured['prompt']
    assert ref not in combined and 'PRIVATE_BACKUP_KYOTO' not in combined
    if change == 'expire':
        assert 'PRIVATE_PERIOD_14_DAYS' not in combined and 'reselect_record' in combined
    else:
        assert 'PRIVATE_PERIOD_14_DAYS' in combined and 'reselect_record' not in combined


class Client:
    def __init__(self, calls): self.calls = calls
    async def complete(self, prompt, *, system='', **kwargs):
        self.calls.append(system)
        if 'rigorous reviewer (critic)' in system:
            return '{"verdict":"ok","issues":[],"fixes":[],"confidence_r":0.9}'
        for line in system.splitlines():
            if line.startswith('{"source_question":'):
                return json.loads(line)['recorded_answer']
        return 'Please choose a current record again.'


async def configure(session, calls):
    session.memory.answerability_gate = AnswerabilityGate(Decisions(True))
    session.left._llm_client = Client(calls); session.right._llm_client = Client(calls)
    session.left.uses_external_llm = session.right.uses_external_llm = True


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_stdio_inventory_to_selected_quote_round_trip(tmp_path, stream):
    sessions, replies, calls = {}, [], []
    options = dict(blob_root=tmp_path, blob_index={}, vision_cache={}, vision_cache_limit=0)
    with patch.dict(os.environ, {'DUALBRAIN_MEMORY_CONVERSATION_CONTEXT': '1'}, clear=True):
        session = await EngineSession.create('s'); sessions['s'] = session
        try:
            await configure(session, calls)
            for t in sources(): session.memory.store(t)
            reply = await _dispatch_request({'id': 'choices', 'method': 'record_choices', 'params': {
                'session_id': 's', 'scope': 'alpha'}}, sessions=sessions, state_store=None,
                blob_pipe=None, send=replies.append, **options)
            page = reply['result']; assert reply['ok'] and not calls
            ref = next(c['record_ref'] for c in page['choices'] if 'backup' in c['label'])
            params = {'session_id': 's', 'question': Q, 'qid': 'read', 'system2_mode': 'on',
                      'memory_context': {**META, 'record_ref': ref}}
            reply = await _dispatch_request({'id': 'quote', 'method': 'process_stream' if stream else 'process',
                'params': params}, sessions=sessions, state_store=None, blob_pipe=None, send=replies.append, **options)
            assert reply['ok'], reply
            result = reply['result']
            assert result['answer'] == 'PRIVATE_BACKUP_KYOTO'
            assert result['record_selection'] == {'status': 'bound', 'action': None}
            assert ref not in '\n'.join(calls) and 'PRIVATE_PERIOD_14_DAYS' not in '\n'.join(calls)
            assert all(t.recalled_answer for t in session.memory.past_qas if t.qid == 'read')
        finally:
            await session.close()
