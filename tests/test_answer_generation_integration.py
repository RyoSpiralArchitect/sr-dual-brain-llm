"""Real RPC/controller behavior with non-network provider completion envelopes."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_contract_live as pilot
from engine_stdio import _dispatch_request
from scheduled_http_journal import ScheduledJournal
from offline_provider_guard import forbid_provider_network
from test_benchmark_contract_live import config, response
from test_dual_brain import AlwaysSkipPolicy


@pytest.mark.asyncio
@pytest.mark.parametrize('text,finish,expected', [
    ('102', 'stop', 'complete'), ('A partial solution', 'length', 'partial'), ('', 'length', 'empty'),
])
async def test_engine_ok_exposes_generation_status_without_claiming_answer_completion(tmp_path, text, finish, expected):
    journal = ScheduledJournal(config(), {'test': 'ordinary-completion'}, max_calls=10)
    class Provider:
        async def _post_json(self, _url, payload, _headers, **kwargs):
            return response(text, finish)
    journal.transport = Provider()
    env = pilot.profile.environment() | {
        'DUALBRAIN_MODULE_ACC_CONSULT_PROMOTION_MODE': 'off',
        'DUALBRAIN_MODULE_BASAL_CONSULT_PROMOTION_MODE': 'off',
        'DUALBRAIN_MODULE_CONVERSATION_SELECTIVE_CONSULTATION_MODE': 'off',
    }
    with patch.dict('os.environ', env, clear=True), forbid_provider_network() as audit:
        session, _ = await pilot.make_session('completion-rpc', None, config(), journal, [1794009600.])
        session.controller.policy = AlwaysSkipPolicy()
        try:
            journal.case = 'ordinary'
            result = await _dispatch_request({'id': 'ordinary', 'method': 'process', 'params': {
                'session_id': session.session_id, 'question': 'What is 17 × 6? Answer with only the number.',
                'qid': 'ordinary', 'leading_brain': 'left', 'system2_mode': 'off',
                'executive_mode': 'off', 'executive_observer_mode': 'off', 'return_telemetry': True}},
                sessions={session.session_id: session}, state_store=None, blob_pipe=None,
                blob_root=tmp_path, blob_index={}, vision_cache={}, vision_cache_limit=0, send=lambda *_: None)
        finally:
            await session.close()
    assert audit.blocked_requests == 0 and result['ok']
    value = result['result']
    observation = value['metrics']['answer_generation']
    assert value['answer_status'] == observation['answer_status'] == expected, (value['answer'], observation, value['metrics']['policy'])
    assert observation['terminal_complete'] is (expected == 'complete')
    assert observation['semantic_correctness'] == 'not_assessed'
    assert not value['metrics']['system2']['used']
    roles = [o['request']['role'] for o in journal.data['occurrences']]
    assert journal.calls <= 2 and roles.count('draft') == 1 and 'integration' not in roles
    assert len(observation['stages']) == 1
    if expected == 'empty':
        assert value['answer'] == 'I could not generate an answer. Please try again.'
    elif expected == 'partial':
        assert value['answer'].startswith('This answer is incomplete.')
    else:
        assert value['answer'] == '102'
    assert 'Draft answer for:' not in value['answer']
    safe = json.dumps(observation)
    assert all(key not in safe for key in ('A partial solution', 'question', 'prompt', 'FAKE_NO_CREDENTIAL'))
