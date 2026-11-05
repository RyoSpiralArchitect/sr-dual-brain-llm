"""Fail locally if an offline experiment enters a real HTTP transport.

Apply around the entire experiment, including task draining. Scripted transports
and raw-journal replay stay above these boundaries. This guard is opt-in: tests
that intentionally run a local Gateway/PostgreSQL service keep their own setup.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from unittest.mock import patch


class ProviderNetworkBlocked(RuntimeError):
    """A real HTTP entry was attempted; no request was sent by that entry."""


@dataclass
class TransportAudit:
    blocked_requests: int = 0

    def reject(self, *_args, **_kwargs):
        self.blocked_requests += 1
        # Never include the URL, headers, body, or a credential in an error.
        raise ProviderNetworkBlocked('Offline experiment attempted real HTTP transport')

    async def reject_async(self, *_args, **_kwargs):
        self.reject()


@contextmanager
def forbid_provider_network():
    """Block JSON, SSE, and common fallback HTTP paths, even on existing clients.

    A swallowed rejection still fails on exit. The caller must join all owned
    tasks before exiting so a delayed request cannot escape the guarded lifetime.
    Patches are process-wide; use in a serial offline experiment/test process.
    """
    audit = TransportAudit()
    with ExitStack() as stack:
        stack.enter_context(patch('aiohttp.ClientSession._request', audit.reject_async))
        stack.enter_context(patch('urllib.request.urlopen', audit.reject))
        stack.enter_context(patch('http.client.HTTPConnection.request', audit.reject))
        try:
            yield audit
        finally:
            if audit.blocked_requests:
                raise ProviderNetworkBlocked(
                    f'Offline experiment blocked {audit.blocked_requests} real HTTP request(s)') from None
