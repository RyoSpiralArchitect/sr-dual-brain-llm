#!/usr/bin/env python3
"""Choose a stored record by number and request its original wording via stdio."""
import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys


def recorded_at_label(value):
    try:
        return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec='seconds')
    except (TypeError, ValueError, OverflowError, OSError):
        return '日時不明'


async def choose(args):
    env = {key:value for key,value in os.environ.items() if key not in {
        'DUALBRAIN_PIPE_NAME','DUALBRAIN_PIPE_PATH','DUALBRAIN_BLOB_PIPE_NAME','DUALBRAIN_BLOB_PIPE_PATH'}}
    process = await asyncio.create_subprocess_exec(sys.executable, '-u',
        str(Path(__file__).with_name('engine_stdio.py')), stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, env=env)
    sequence = 0
    async def request(method, params):
        nonlocal sequence
        sequence += 1
        rid = str(sequence)
        process.stdin.write((json.dumps({'id':rid,'method':method,'params':params},ensure_ascii=False)+'\n').encode())
        await process.stdin.drain()
        while True:
            line = await process.stdout.readline()
            if not line:
                raise RuntimeError('The memory engine closed before replying')
            reply = json.loads(line)
            if str(reply.get('id')) != rid or 'event' in reply:
                continue
            if not reply.get('ok'):
                raise RuntimeError('Memory request failed: '+str((reply.get('error') or {}).get('type','Error')))
            return reply['result']
    try:
        page = await request('record_choices', {'session_id':args.session_id,'scope':args.scope,
                                               'offset':args.offset})
        choices = page['choices']
        if not choices:
            print('現在の候補はありません。')
            return
        for index, item in enumerate(choices, 1):
            status = '' if item['selectable'] else '（現在は再掲できません）'
            recorded = recorded_at_label(item['recorded_at'])
            print(f"{index}. {item['label']} [{recorded}] {status}")
        if page['next_offset'] is not None:
            print(f"続きは --offset {page['next_offset']} で表示できます。")
        index = args.index if args.index is not None else int(input('再掲する記録の番号: '))
        if not 1 <= index <= len(choices) or not choices[index-1]['selectable']:
            raise ValueError('再掲できる候補の番号を選んでください。')
        result = await request('process', {'session_id':args.session_id,
            'question':'記録された回答をそのまま再掲して','system2_mode':args.system2_mode,
            'memory_context':{'id':args.scope,'aliases':[args.scope],'remember':False,
                              'record_ref':choices[index-1]['record_ref']}})
        print('\n'+result['answer'])
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), 5)
        except asyncio.TimeoutError:
            process.terminate()
            await process.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-id',required=True)
    parser.add_argument('--scope',required=True)
    parser.add_argument('--index',type=int)
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--system2-mode',choices=('auto','on','off'),default='auto')
    args=parser.parse_args()
    try:
        asyncio.run(choose(args))
    except (ValueError, RuntimeError) as exc:
        print(str(exc),file=sys.stderr)
        raise SystemExit(1) from None


if __name__=='__main__':
    main()
