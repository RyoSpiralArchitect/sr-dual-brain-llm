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
    contract = getattr(model, 'advisor_contract', None)
    if contract is not None:
        if not isinstance(contract, str) or re.fullmatch('[a-z][a-z0-9_]{0,63}', contract) is None:
            return None
        payload['advisor_contract'] = contract
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
    kind: str = 'probe'
    output_blocked: bool = False
    confidence: str = 'none'
    confirmation_required: bool = False


@dataclass(frozen=True)
class _Sample:
    observed_at: float
    elapsed_ms: float
    censored: bool
    sequence: int
    expires_at: float
    provisional: bool = False


@dataclass(frozen=True, eq=False)
class _ProbeLease:
    key: tuple
    cutoff: int
    budget_ms: float
    kind: str = 'probe'
    history_expiry: float = 0.
    first: _Sample | None = None


@dataclass(frozen=True)
class _Candidate:
    first: _Sample
    cutoff: int
    budget_ms: float
    history_expiry: float


@dataclass
class _Profile:
    samples: deque
    sequence: int = 0
    last_probe_at: float | None = None
    active_probe: _ProbeLease | None = None
    candidate: _Candidate | None = None
    confirmed: bool = False
    unusable_until: float = 0.
    uncertain_until: float = 0.


class AdvisorTiming:
    """Recent usable completions and deadline-censored lower bounds.

    A timeout is not a completed latency. The maximum recent observation is a
    conservative estimate, not a guarantee about a future request. Expiry allows
    a new cold probe; history is neither persisted nor shared between sessions.
    """
    def __init__(self, *, clock=time.monotonic, ttl_seconds=60., max_profiles=32, max_samples=16,
                 recovery_enabled=True, confirmation_enabled=True):
        if (not callable(clock) or not finite_nonnegative(ttl_seconds) or ttl_seconds <= 0
                or type(max_profiles) is not int or max_profiles < 1
                or type(max_samples) is not int or max_samples < 1
                or type(recovery_enabled) is not bool or type(confirmation_enabled) is not bool
                or confirmation_enabled and max_samples < 2):
            raise ValueError('Positive finite history bounds required')
        self.clock, self.ttl_seconds = clock, ttl_seconds
        self.max_profiles, self.max_samples = max_profiles, max_samples
        # Internal dependency injection supports a v2-equivalent benchmark.
        # Production admission remains controlled by the registered module mode.
        self.recovery_enabled = recovery_enabled
        self.confirmation_enabled = confirmation_enabled
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
        state = self._history.get(key)
        if state is not None and state.candidate is not None and elapsed_ms + 50. >= state.candidate.budget_ms:
            state.candidate = None
        self._append(key, now, elapsed_ms, censored, now + self.ttl_seconds)

    def _append(self, key, now, elapsed_ms, censored, expires_at, *, provisional=False):
        if expires_at <= now:
            return None
        state = self._history.setdefault(key, _Profile(deque(maxlen=self.max_samples)))
        state.sequence += 1
        sample = _Sample(now, float(elapsed_ms), bool(censored), state.sequence, expires_at, provisional)
        state.samples.append(sample)
        self._history.move_to_end(key)
        while len(self._history) > self.max_profiles:
            self._history.popitem(last=False)
        return sample

    def _recent(self, key, now):
        state = self._history.get(key)
        if state is None or not math.isfinite(now):
            return None, []
        recent = [s for s in state.samples if s.observed_at <= now < s.expires_at]
        state.samples = deque(recent, maxlen=self.max_samples)
        if state.candidate is not None and state.candidate.history_expiry <= now:
            state.candidate = None
        if not recent:
            # Sample expiry must not erase a still-active rolling probe limit.
            if (state.active_probe is None and now >= max(state.uncertain_until, state.unusable_until)
                    and (state.last_probe_at is None or now - state.last_probe_at >= 60.)):
                self._history.pop(key, None)
                return None, []
        self._history.move_to_end(key)
        return state, recent

    @staticmethod
    def _estimate(recent, confirmed=False):
        if not recent:
            return LatencyEstimate()
        slowest = max(recent, key=lambda s: (s.elapsed_ms, s.censored))
        value, kind = slowest.elapsed_ms, 'censored_bound' if slowest.censored else 'completed_max'
        completed = [s.elapsed_ms for s in recent if not s.censored and not s.provisional]
        if confirmed and len(completed) >= 2:
            margin = max(completed) + (max(completed) - min(completed))
            if not math.isfinite(margin):
                margin = math.nextafter(math.inf, 0.)
            if margin > value:
                value, kind = margin, 'confirmation_margin'
        return LatencyEstimate(value, kind, sum(not s.censored for s in recent), sum(s.censored for s in recent))

    def estimate(self, module, profile):
        state, recent = self._recent(self._key(module, profile), self.clock())
        return self._estimate(recent, bool(state and state.confirmed))

    @staticmethod
    def _status(state, now, reason, *, allowed=False, retry=None, kind='probe'):
        blocked = bool(state and now < state.unusable_until)
        uncertain = bool(state and now < state.uncertain_until)
        confidence = ('suspended' if blocked else 'confirming' if state and state.active_probe and state.active_probe.kind == 'confirmation'
                      else 'probing' if state and state.active_probe else 'pending' if state and state.candidate
                      else 'unconfirmed' if uncertain else 'confirmed' if state and state.confirmed else 'none')
        return RecoveryStatus(allowed, reason, retry, kind, blocked, confidence, uncertain)

    @staticmethod
    def _blocking_expiry(state, recent, budget):
        """Expiry of the pre-existing witnesses that currently inhibit work."""
        expiry = max((s.expires_at for s in recent if s.elapsed_ms + 50. >= budget), default=0.)
        completed = [s for s in recent if not s.censored and not s.provisional]
        if state.confirmed and len(completed) >= 2:
            low, high = min(s.elapsed_ms for s in completed), max(s.elapsed_ms for s in completed)
            if high + (high - low) + 50. >= budget:
                support_expiry = min(max(s.expires_at for s in completed if s.elapsed_ms == low),
                                     max(s.expires_at for s in completed if s.elapsed_ms == high))
                expiry = max(expiry, support_expiry)
        return max(expiry, state.unusable_until, state.uncertain_until)

    def recovery_status(self, module, profile, budget_ms):
        """Read-only permission facts: 15 s quiet time, one start per 60 s.

        A probe needs a known, ample window; strict outer deadlines and unknown
        windows never establish probe eligibility. ON/OFF do not consume it.
        """
        if not self.recovery_enabled:
            return RecoveryStatus(reason='disabled')
        key, now = self._key(module, profile), self.clock()
        state, recent = self._recent(key, now)
        if state is None or not recent and now >= max(state.unusable_until, state.uncertain_until):
            return RecoveryStatus()
        status = lambda reason, **kw:self._status(state, now, reason, **kw)
        if not finite_nonnegative(budget_ms):
            return status('unknown_budget')
        if budget_ms < _PROBE_MIN_BUDGET_MS[module]:
            return status('budget_too_small')
        estimate = self._estimate(recent, state.confirmed).ms
        if (estimate is not None and estimate + 50. < budget_ms
                and now >= max(state.unusable_until, state.uncertain_until)):
            return status('estimate_fits')
        if state.active_probe is not None:
            return status('in_flight')
        if self.confirmation_enabled and state.candidate is not None:
            candidate = state.candidate
            if candidate.first.elapsed_ms + 50. >= min(budget_ms, candidate.budget_ms):
                return status('budget_too_small', kind='confirmation')
            return status('ready', allowed=True, retry=0., kind='confirmation')
        cooldown = 15. - (now - max((s.observed_at for s in recent), default=state.last_probe_at or now))
        rate_limit = 0. if state.last_probe_at is None else 60. - (now - state.last_probe_at)
        if max(cooldown, rate_limit) > 0:
            return status('rate_limited' if rate_limit >= cooldown else 'cooldown', retry=max(cooldown, rate_limit) * 1000.)
        return status('ready', allowed=True, retry=0.)

    def start_probe(self, module, profile, budget_ms, *, kind='probe'):
        """Atomically claim at actual work start, without awaiting a provider."""
        status = self.recovery_status(module, profile, budget_ms)
        if not status.allowed or kind != status.kind:
            return None
        now = self.clock()
        if not math.isfinite(now):
            return None
        key = self._key(module, profile)
        state, recent = self._recent(key, now)
        if state is None:
            return None
        if kind == 'confirmation':
            candidate = state.candidate
            if candidate is None:
                return None
            lease = _ProbeLease(key, candidate.cutoff, min(float(budget_ms), candidate.budget_ms), kind,
                                candidate.history_expiry, candidate.first)
            state.candidate = None
        else:
            lease = _ProbeLease(key, state.sequence, float(budget_ms), kind,
                                self._blocking_expiry(state, recent, budget_ms))
            state.last_probe_at = now
        state.active_probe = lease
        return lease

    def finish_probe(self, lease, elapsed_ms, *, usable=False, censored=False):
        """Confirmed usable recovery supersedes older evidence, never newer work.

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
        if self.confirmation_enabled:
            return self._finish_confirmed_probe(state, lease, elapsed_ms, usable=usable, censored=censored)
        if usable and not censored:
            recovered = elapsed_ms + 50. < lease.budget_ms
            if recovered:
                state.samples = deque((s for s in state.samples if s.sequence > lease.cutoff), maxlen=self.max_samples)
            self.record(*lease.key, elapsed_ms)
            return 'recovered' if recovered else 'still_slow'
        if censored:
            self.record(*lease.key, elapsed_ms, censored=True)
            return 'cancelled'
        return 'unusable'

    def _finish_confirmed_probe(self, state, lease, elapsed_ms, *, usable, censored):
        now = self.clock()
        state.samples = deque((s for s in state.samples if s.observed_at <= now < s.expires_at), maxlen=self.max_samples)
        # Ring-buffer eviction cannot turn one provisional sample into normal
        # admission. This hold ends no later than the original blocking evidence.
        state.uncertain_until = max(state.uncertain_until, lease.history_expiry)
        if usable and not censored and elapsed_ms + 50. < lease.budget_ms:
            if lease.kind == 'probe':
                if now >= lease.history_expiry:
                    self.record(*lease.key, elapsed_ms)
                    return 'history_expired'
                if any(s.sequence > lease.cutoff and s.elapsed_ms + 50. >= lease.budget_ms for s in state.samples):
                    self._append(lease.key, now, elapsed_ms, False, lease.history_expiry, provisional=True)
                    return 'unstable'
                sample = self._append(lease.key, now, elapsed_ms, False, lease.history_expiry, provisional=True)
                state.candidate = _Candidate(sample, lease.cutoff, lease.budget_ms, lease.history_expiry)
                return 'confirmation_pending'
            first = lease.first
            prediction = max(first.elapsed_ms, elapsed_ms) + abs(first.elapsed_ms - elapsed_ms)
            retained = [s for s in state.samples if s.sequence > lease.cutoff and s.sequence != first.sequence]
            proposed = retained + [_Sample(first.observed_at, first.elapsed_ms, False, 0, first.observed_at + self.ttl_seconds),
                                   _Sample(now, elapsed_ms, False, 0, now + self.ttl_seconds)]
            prediction = max(prediction, self._estimate(proposed, True).ms)
            if prediction + 50. < lease.budget_ms and first.observed_at + self.ttl_seconds > now:
                state.samples = deque(retained, maxlen=self.max_samples)
                self._append(lease.key, first.observed_at, first.elapsed_ms, False, first.observed_at + self.ttl_seconds)
                self.record(*lease.key, elapsed_ms)
                state.confirmed, state.unusable_until, state.uncertain_until = True, 0., 0.
                return 'confirmed'
            self._append(lease.key, now, elapsed_ms, False, lease.history_expiry, provisional=True)
            return 'unstable'
        if usable or censored:
            # Exploratory failure cannot postpone the original cold opportunity.
            self._append(lease.key, now, elapsed_ms, censored, lease.history_expiry, provisional=True)
        return 'cancelled' if censored else 'still_slow' if usable else 'unusable'

    def record_work(self, module, profile, elapsed_ms, *, usable=False, censored=False):
        """Observe ordinary work, including unusable output after promotion."""
        if usable or censored:
            self.record(module, profile, elapsed_ms, censored=censored)
            return
        if self.confirmation_enabled:
            state, recent = self._recent(self._key(module, profile), self.clock())
            if state is not None and state.confirmed and recent:
                state.candidate = None
                state.unusable_until = max(s.expires_at for s in recent)


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
    recovery_kind: str = 'probe'
