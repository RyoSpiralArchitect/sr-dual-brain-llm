"""Post-output offline diagnostic; the original frozen replay failure stays intact."""
import asyncio
import hashlib
from pathlib import Path

import sys
sys.path.insert(0, str(Path.cwd() / "sr-dual-brain-llm/scripts"))
import benchmark_natural_memory as study

original = study.pilot.RawJournal.post
previews = {}
started = {}


async def scheduled(self, role, url, payload, headers, **kwargs):
    if self.frozen is None or self.transport is not None:
        raise RuntimeError('This diagnostic permits frozen offline replay only')
    case = self.case
    event = started.setdefault(case, asyncio.Event())
    if role == 'right':
        expected = self.frozen['occurrences'][self.position]
        if expected['request']['case'] != case or expected['request']['role'] != role:
            raise ValueError('Preview admission differs')
        previews[case] = (asyncio.current_task(), 'response' in expected)
        event.set()
    try:
        result = await original(self, role, url, payload, headers, **kwargs)
    except asyncio.CancelledError:
        if role != 'right':
            raise
        # Consume the recorded cancelled occurrence, but let the actual owner
        # cancel its preview at the ordinary post-draft boundary.
        await asyncio.Future()
        raise AssertionError('A cancelled preview must not return')
    if role in {'draft', 'source_write'}:
        next_row = (self.frozen['occurrences'][self.position]
                    if self.position < len(self.frozen['occurrences']) else None)
        if next_row and next_row['request']['case'] == case and next_row['request']['role'] == 'right':
            waiter = asyncio.create_task(event.wait())
            done, _ = await asyncio.wait({waiter}, timeout=1.)
            if not done:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
                raise ValueError('The actual controller did not start the recorded preview')
            task, completed = previews[case]
            if completed:
                done, _ = await asyncio.wait({task}, timeout=1.)
                if not done:
                    raise ValueError('The recorded completed preview did not finish')
    return result


study.pilot.RawJournal.post = scheduled
print('Supplemental offline diagnostic SHA256:', hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
study.main()
