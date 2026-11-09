"""One critic's remaining time, bounded by its actual consultation/turn owners."""
import asyncio
from contextvars import ContextVar
from functools import wraps
import math
import os

from .callosum import remaining_consultation_ms
from .turn_lifecycle import remaining_turn_ms

_deadline = ContextVar('critic_review_deadline', default=None)


def _now():
    return asyncio.get_running_loop().time()


def configured_review_seconds(config):
    try:
        requested = float(os.environ.get('DUALBRAIN_CRITIC_TIMEOUT_SECONDS', '24'))
        if not math.isfinite(requested):
            requested = 24.
    except ValueError:
        requested = 24.
    try:
        client = float(config.timeout_seconds if config else 40)
    except (TypeError, ValueError):
        return 0.
    return min(client, max(1., min(requested, 120.))) if math.isfinite(client) and client > 0 else 0.


def review_timeout_seconds(limit):
    """Leave other requests alone; a critic cannot restart a consumed deadline."""
    deadline = _deadline.get()
    if deadline is None:
        return limit
    bounds = [float(limit), deadline-_now()]
    for remaining in (remaining_consultation_ms(), remaining_turn_ms()):
        if remaining is not None:
            bounds.append(remaining/1000.)
    return max(0., min(bounds))


def review_http_timeout_seconds(limit):
    seconds = review_timeout_seconds(limit)
    if _deadline.get() is not None and seconds <= 0:
        raise TimeoutError('Critic review deadline elapsed before HTTP admission')
    return seconds


def bounded_critic_review(method):
    @wraps(method)
    async def guarded(self, *args, **kwargs):
        end = _now()+configured_review_seconds(self.llm_config)
        parent = _deadline.get()
        token = _deadline.set(min(end, parent) if parent is not None else end)
        try:
            return await method(self, *args, **kwargs)
        finally:
            _deadline.reset(token)
    return guarded
