"""Owned speculative work; starting a model coroutine is not an HTTP receipt."""
import asyncio

from .turn_lifecycle import cancel_task, create_turn_task


class OwnedPreview:
    def __init__(self, *, requested, suppressed=False):
        self.task = None
        self.notes = None
        self.observation = {
            'version': 'owned_preview_v1', 'requested': bool(requested),
            'admitted': False, 'started': False, 'available': False,
            'outcome': 'suppressed' if requested and suppressed else 'not_requested',
        }
        self.allowed = requested and not suppressed

    def start(self, generate):
        if not self.allowed or self.task is not None:
            return
        async def run():
            self.observation['started'] = True
            return await generate()
        coroutine = run()
        try:
            self.task = create_turn_task(coroutine)
            self.observation['admitted'] = True
        except Exception:
            coroutine.close()
            self.observation['outcome'] = 'failed'

    async def close(self):
        if self.task is None:
            return
        try:
            if not self.task.done():
                cancel_task(self.task)
            await asyncio.gather(self.task, return_exceptions=True)
        finally:
            # Even cancellation arriving during cleanup retains an observation.
            if self.task.done():
                if self.task.cancelled():
                    self.observation['outcome'] = ('cancelled_after_start' if self.observation['started']
                                                   else 'cancelled_before_start')
                else:
                    try:
                        self.notes = self.task.result()
                        self.observation['available'] = bool(self.notes)
                        self.observation['outcome'] = 'completed' if self.notes else 'empty'
                    except Exception:
                        self.observation['outcome'] = 'failed'
