"""Opt-in ordered transport events for concurrent, zero-network raw replay.

Transport starts and terminal events are observations. Replay waits for their
order, never wall time or policy actions. A recorded cancellation is delivered
only by the actual owning task; the journal never cancels that task itself.
Legacy RawJournal behavior and historical artifacts remain separate.
"""
import asyncio
from copy import deepcopy

from benchmark_contract_live import AdmissionError, RawJournal, bench


class ScheduledJournal(RawJournal):
    def __init__(self, *args, replay_wait_seconds=5., **kwargs):
        super().__init__(*args, **kwargs)
        self.schedule_position = 0
        self.schedule_errors = []
        self.changed = asyncio.Event()
        self.replay_wait_seconds = replay_wait_seconds
        self.data.update(transport_events=[], transport_events_sha256=bench.digest([]))
        if self.frozen:
            self._validate_schedule()

    @staticmethod
    def terminal(row):
        return ('cancelled' if row.get('error_type') == 'CancelledError' else
                'error' if row.get('error_type') else 'response')

    def _validate_schedule(self):
        events = self.frozen.get('transport_events')
        if (not isinstance(events, list)
                or self.frozen.get('transport_events_sha256') != bench.digest(events)):
            raise ValueError('Missing or corrupt transport schedule')
        active = set()
        admitted = 0
        for event in events:
            if (not isinstance(event, dict) or set(event) != {'event', 'occurrence'}
                    or type(event['occurrence']) is not int):
                raise ValueError('Malformed transport event')
            index = event['occurrence']
            if event['event'] == 'start':
                if index != admitted or index >= len(self.frozen['occurrences']):
                    raise ValueError('Transport admission order differs')
                admitted += 1
                active.add(index)
            else:
                if index not in active or event['event'] != self.terminal(self.frozen['occurrences'][index]):
                    raise ValueError('Transport completion order is invalid')
                active.remove(index)
        if active or admitted != len(self.frozen['occurrences']):
            raise ValueError('Transport event coverage differs')

    def _next(self):
        events = self.frozen['transport_events']
        return events[self.schedule_position] if self.schedule_position < len(events) else None

    def _observe(self, event):
        if self.frozen:
            if self._next() != event:
                self.schedule_errors.append('event_differs')
                raise AdmissionError('Transport event differs')
            self.schedule_position += 1
        self.data['transport_events'].append(event)
        self.data['transport_events_sha256'] = bench.digest(self.data['transport_events'])
        previous, self.changed = self.changed, asyncio.Event()
        previous.set()

    def _transport_admitted(self, index):
        self._observe({'event': 'start', 'occurrence': index})
        self.checkpoint()

    async def _wait_recorded_transport(self, index):
        expected = {'event': self.terminal(self.frozen['occurrences'][index]), 'occurrence': index}
        deadline = asyncio.get_running_loop().time() + self.replay_wait_seconds
        while True:
            if expected['event'] != 'cancelled' and self._next() == expected:
                return
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                self.schedule_errors.append('event_timeout')
                raise AdmissionError('Recorded transport event was not reached')
            # Observe a separate waiter without spending the caller's cancel-once
            # guard on a local timeout (asyncio.wait_for can cancel that caller).
            # Cancellation receipts still require the actual owning task.
            waiter = asyncio.create_task(self.changed.wait())
            try:
                done, _ = await asyncio.wait({waiter}, timeout=remaining)
                if not done:
                    self.schedule_errors.append('event_timeout')
                    raise AdmissionError('Recorded transport event was not reached')
            finally:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)

    def _transport_finished(self, index, row):
        try:
            self._observe({'event': self.terminal(row), 'occurrence': index})
        except AdmissionError:
            # Never replace an owning turn's cancellation from a finally hook.
            # finish() rejects divergence even if a production guard caught it.
            pass

    def finish(self):
        if self.schedule_errors or self.frozen and (
                self.schedule_position != len(self.frozen['transport_events'])
                or self.data['transport_events'] != self.frozen['transport_events']):
            raise ValueError('Exact transport schedule coverage differs')
        self._validate_live_coverage()
        super().finish()

    def _validate_live_coverage(self):
        # Reuse exactly the frozen validator for a completed live observation.
        original = self.frozen
        try:
            self.frozen = deepcopy(self.data)
            self._validate_schedule()
        finally:
            self.frozen = original
