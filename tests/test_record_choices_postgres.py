"""Choice refs bind one memory incarnation; durable receipts retain retry semantics."""
from dataclasses import replace
import os
from unittest.mock import patch

import pytest

from core.memory_commits import MemoryCommitConflict, memory_payload
from engine_stdio import EngineSession, _dispatch_request, _handle_process, _reload_committed_memory
from test_memory_commit_postgres import pg_store
from test_record_choices import Q, META, sources, configure, ref_for


@pytest.mark.asyncio
async def test_durable_choice_quote_restart_relist_and_receipt_replay(pg_store, tmp_path):
    await pg_store.append_memory_traces('s', sources())
    env = {'DUALBRAIN_MEMORY_CONVERSATION_CONTEXT': '1', 'DUALBRAIN_MEMORY_OUTBOX_PATH': str(tmp_path/'outbox.sqlite3'),
           'LLM_PROVIDER': 'openai', 'LLM_MODEL_ID': 'fake-choice', 'LLM_API_KEY': 'FAKE_NO_CREDENTIAL',
           'DUALBRAIN_MEMORY_ANSWERABILITY_MODEL': 'fake-choice', 'DUALBRAIN_PG_PERSIST_TELEMETRY': '0'}
    sessions, calls = {}, []
    options = dict(blob_root=tmp_path, blob_index={}, vision_cache={}, vision_cache_limit=0)
    with patch.dict(os.environ, env, clear=True):
        reply = await _dispatch_request({'id':'list','method':'record_choices','params':{
            'session_id':'s','scope':'alpha'}}, sessions=sessions, state_store=pg_store,
            blob_pipe=None, send=lambda *_:None, **options)
        assert reply['ok'] and reply['result']['total'] == 2
        session = sessions['s']
        assert session.memory.answerability_gate.backend.calls == 0
        await configure(session, calls)
        ref = ref_for(session.memory)
        params = {'question':Q,'qid':'chosen-read','memory_operation_id':'chosen-op',
                  'memory_context':{**META,'record_ref':ref},'system2_mode':'on','leading_brain':'left'}
        first = await _handle_process(session, params, **options)
        assert first['answer']=='PRIVATE_BACKUP_KYOTO' and first['memory_commit']['state']=='committed'
        assert first['record_selection']=={'status':'bound','action':None}
        assert len(await pg_store.load_conversation_sources('s'))==2
        await session.close()
        restarted = await EngineSession.create('s',state_store=pg_store)
        try:
            await configure(restarted, calls)
            before=len(calls)
            # Exactly the same committed request is a historical receipt replay,
            # not a fresh quote or a restoration of its old reference capability.
            retry=await _handle_process(restarted,params,**options)
            assert retry['replayed'] and retry['answer']==first['answer'] and len(calls)==before
            changed={**params,'memory_context':{**META,'record_ref':ref_for(restarted.memory)}}
            with pytest.raises(MemoryCommitConflict):
                await _handle_process(restarted,changed,**options)
            fresh={**params,'qid':'after-restart','memory_operation_id':'new-op'}
            stale=await _handle_process(restarted,fresh,**options)
            assert stale['record_selection']=={'status':'invalid_selection','action':'reselect_record'}
            assert 'PRIVATE_' not in stale['answer']
            new_ref=ref_for(restarted.memory)
            fresh.update(qid='relisted',memory_operation_id='relisted-op',memory_context={**META,'record_ref':new_ref})
            current=await _handle_process(restarted,fresh,**options)
            assert current['answer']=='PRIVATE_BACKUP_KYOTO'
        finally:
            await restarted.close()


@pytest.mark.asyncio
async def test_committed_correction_invalidates_choice_in_same_instance(pg_store, tmp_path):
    await pg_store.append_memory_traces('s',sources())
    with patch.dict(os.environ, {'DUALBRAIN_MEMORY_CONVERSATION_CONTEXT':'1',
            'DUALBRAIN_MEMORY_OUTBOX_PATH':str(tmp_path/'outbox.sqlite3')}, clear=True):
        session=await EngineSession.create('s',state_store=pg_store)
        try:
            await configure(session,[])
            ref=ref_for(session.memory)
            old=next(t for t in session.memory.past_qas if t.qid=='backup')
            new=replace(old,qid='backup-v2',seed_id=None,answer='PRIVATE_CURRENT_OSAKA')
            await pg_store.commit_memory_operation('s','correction',memory_payload([new],[replace(old,evidence_status='superseded')]))
            # A committed external update is observed at the selected-request
            # boundary, without a caller manually reloading the session.
            result=await _handle_process(session,{'question':Q,'qid':'probe','system2_mode':'on',
                'memory_context':{**META,'record_ref':ref}},blob_root=tmp_path,blob_index={},
                vision_cache={},vision_cache_limit=0)
            assert result['record_selection']=={'status':'stale_selection','action':'reselect_record'}
            assert 'PRIVATE_' not in result['answer']
            assert session.memory.record_choices('alpha')['total']==2
        finally:
            await session.close()
