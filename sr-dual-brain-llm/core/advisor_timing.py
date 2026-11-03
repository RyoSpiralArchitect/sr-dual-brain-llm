"""Bounded session-local latency facts; never prompts, replies or billed usage."""
from collections import OrderedDict, deque
from dataclasses import dataclass
import hashlib
import json
import math
import re
import time

from .thalamic_reticular_nucleus import (
    EXECUTIVE_ADVICE, DIRECTOR_ADVICE, EXECUTIVE_METRICS_OBSERVATION,
)

ADVISORS = frozenset({EXECUTIVE_ADVICE, DIRECTOR_ADVICE, EXECUTIVE_METRICS_OBSERVATION})
_PROBE_MIN_BUDGET_MS = {DIRECTOR_ADVICE: 1600., EXECUTIVE_METRICS_OBSERVATION: 2500.,
                        EXECUTIVE_ADVICE: 2000.}


def finite_nonnegative(value):
    if type(value) not in (float, int):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except (OverflowError, ValueError):
        return False


def model_profile(model):
    """Separate role histories when effective provider/model/options change.

    Auth keys/headers and conversation content never enter this fingerprint.
    Custom/non-provider models without a known config retain unknown estimates.
    """
    if not getattr(model, 'uses_external_llm', False):
        return None
    client = getattr(model, '_llm_client', None)
    config = getattr(client, 'config', None) or getattr(model, 'llm_config', None)
    if config is None:
        return None
    fields = ('provider', 'model', 'api_base', 'max_output_tokens', 'timeout_seconds',
              'auto_continue', 'max_continuations')
    payload = {key: getattr(config, key, None) for key in fields}
    if not payload['provider'] or not payload['model']:
        return None
    try:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class LatencyEstimate:
    ms: float | None = None
    kind: str = 'unknown'
    completed_samples: int = 0
    censored_samples: int = 0


@dataclass(frozen=True)
class RecoveryStatus:
    allowed: bool = False
    reason: str = 'no_history'
    retry_after_ms: float | None = None


@dataclass(frozen=True, eq=False)
class _ProbeLease:
    key: tuple
    cutoff: int
    budget_ms: float


@dataclass
class _Profile:
    samples: deque
    sequence: int = 0
    last_probe_at: float | None = None
    active_probe: _ProbeLease | None = None


class AdvisorTiming:
    """Recent usable completions and deadline-censored lower bounds.

    A timeout is not a completed latency. The maximum recent observation is a
    conservative estimate, not a guarantee about a future request. Expiry allows
    a new cold probe; history is neither persisted nor shared between sessions.
    """
    def __init__(self, *, clock=time.monotonic, ttl_seconds=60., max_profiles=32, max_samples=16,
                 recovery_enabled=True):
        if (not callable(clock) or not finite_nonnegative(ttl_seconds) or ttl_seconds <= 0
                or type(max_profiles) is not int or max_profiles < 1
                or type(max_samples) is not int or max_samples < 1
                or type(recovery_enabled) is not bool):
            raise ValueError('Positive finite history bounds required')
        self.clock, self.ttl_seconds = clock, ttl_seconds
        self.max_profiles, self.max_samples = max_profiles, max_samples
        # Internal dependency injection supports a v2-equivalent benchmark.
        # Production admission remains controlled by the registered module mode.
        self.recovery_enabled = recovery_enabled
        self._history = OrderedDict()

    @staticmethod
    def _key(module, profile):
        if not isinstance(module, str) or module not in ADVISORS or not isinstance(profile, str) or re.fullmatch('[0-9a-f]{64}', profile) is None:
            return None
        return module, profile

    def record(self, module, profile, elapsed_ms, *, censored=False):
        key, now = self._key(module, profile), self.clock()
        if key is None or not finite_nonnegative(elapsed_ms) or not math.isfinite(now):
            return
        state = self._history.setdefault(key, _Profile(deque(maxlen=self.max_samples)))
        state.sequence += 1
        state.samples.append((now, float(elapsed_ms), bool(censored), state.sequence))
        self._history.move_to_end(key)
        while len(self._history) > self.max_profiles:
            self._history.popitem(last=False)

    def _recent(self, key, now):
        state = self._history.get(key)
        if state is None or not math.isfinite(now):
            return None, []
        recent = [s for s in state.samples if 0 <= now - s[0] < self.ttl_seconds]
        state.samples = deque(recent, maxlen=self.max_samples)
        if not recent:
            # Sample expiry must not erase a still-active rolling probe limit.
            if state.active_probe is None and (state.last_probe_at is None or now - state.last_probe_at >= 60.):
                self._history.pop(key, None)
                return None, []
        self._history.move_to_end(key)
        return state, recent

    @staticmethod
    def _estimate(recent):
        if not recent:
            return LatencyEstimate()
        slowest = max(recent, key=lambda s: (s[1], s[2]))
        return LatencyEstimate(slowest[1], 'censored_bound' if slowest[2] else 'completed_max',
                               sum(not s[2] for s in recent), sum(s[2] for s in recent))

    def estimate(self, module, profile):
        _, recent = self._recent(self._key(module, profile), self.clock())
        return self._estimate(recent)

    def recovery_status(self, module, profile, budget_ms):
        """Read-only permission facts: 15 s quiet time, one start per 60 s.

        A probe needs a known, ample window; strict outer deadlines and unknown
        windows never establish probe eligibility. ON/OFF do not consume it.
        """
        if not self.recovery_enabled:
            return RecoveryStatus(reason='disabled')
        key, now = self._key(module, profile), self.clock()
        state, recent = self._recent(key, now)
        if not recent:
            return RecoveryStatus()
        if not finite_nonnegative(budget_ms):
            return RecoveryStatus(reason='unknown_budget')
        if budget_ms < _PROBE_MIN_BUDGET_MS[module]:
            return RecoveryStatus(reason='budget_too_small')
        if self._estimate(recent).ms + 50. < budget_ms:
            return RecoveryStatus(reason='estimate_fits')
        if state.active_probe is not None:
            return RecoveryStatus(reason='in_flight')
        cooldown = 15. - (now - recent[-1][0])
        rate_limit = 0. if state.last_probe_at is None else 60. - (now - state.last_probe_at)
        if max(cooldown, rate_limit) > 0:
            return RecoveryStatus(reason='rate_limited' if rate_limit >= cooldown else 'cooldown',
                                  retry_after_ms=max(cooldown, rate_limit) * 1000.)
        return RecoveryStatus(True, 'ready', 0.)

    def start_probe(self, module, profile, budget_ms):
        """Atomically claim at actual work start, without awaiting a provider."""
        if not self.recovery_status(module, profile, budget_ms).allowed:
            return None
        now = self.clock()
        if not math.isfinite(now):
            return None
        key = self._key(module, profile)
        state = self._history[key]
        lease = _ProbeLease(key, state.sequence, float(budget_ms))
        state.last_probe_at, state.active_probe = now, lease
        return lease

    def finish_probe(self, lease, elapsed_ms, *, usable=False, censored=False):
        """A fast usable probe supersedes older evidence, never newer work.

        Failed/empty work still spends the probe allowance but cannot remove
        slow history. Token identity prevents an evicted profile or duplicate
        completion from changing a new profile's history.
        """
        if not isinstance(lease, _ProbeLease):
            return 'profile_unavailable'
        state = self._history.get(lease.key)
        if state is None or state.active_probe is not lease:
            return 'profile_unavailable'
        state.active_probe = None
        if not finite_nonnegative(elapsed_ms) or not math.isfinite(self.clock()):
            return 'unusable'
        if usable and not censored:
            recovered = elapsed_ms + 50. < lease.budget_ms
            if recovered:
                state.samples = deque((s for s in state.samples if s[3] > lease.cutoff), maxlen=self.max_samples)
            self.record(*lease.key, elapsed_ms)
            return 'recovered' if recovered else 'still_slow'
        if censored:
            self.record(*lease.key, elapsed_ms, censored=True)
            return 'cancelled'
        return 'unusable'


@dataclass
class AdvisorWork:
    module: str
    profile: str | None
    qid: str | None
    started_at: float | None = None
    stop_elapsed_ms: float | None = None
    stop_reason: str = 'none'
    finished: bool = False
    recovery_probe: bool = False
    probe_lease: object | None = None
