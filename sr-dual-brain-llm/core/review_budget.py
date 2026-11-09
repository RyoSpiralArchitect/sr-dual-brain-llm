"""One critic's remaining time, bounded by its actual consultation/turn owners."""
import asyncio
from contextvars import ContextVar
from functools import wraps
import math
import os

from .callosum import remaining_consultation_ms
from .turn_lifecycle import remaining_turn_ms

_deadline = ContextVar('critic_review_deadline', default=None)


class ReviewBudgetExhausted(TimeoutError):
    """A local review window expired, not a provider health/auth verdict."""
    def __init__(self,message='Critic review window elapsed',*,http_attempted=False):
        super().__init__(message)
        self.http_attempted=http_attempted


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


def review_budget_active():
    return _deadline.get() is not None


def review_io_timeout_seconds(limit):
    """The RPC/bus owner cancels its work; do not duplicate its timer in HTTP.

    Local/client limits still bound direct reviews without a shorter owner.
    Admission and late-result checks continue to consult every actual owner.
    """
    deadline=_deadline.get()
    return limit if deadline is None else max(0.,min(float(limit),deadline-_now()))


def review_http_timeout_seconds(limit):
    if review_budget_active() and review_timeout_seconds(limit)<=0:
        raise ReviewBudgetExhausted('Critic review window elapsed before HTTP admission')
    return review_io_timeout_seconds(limit)


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
