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


class AdvisorTiming:
    """Recent usable completions and deadline-censored lower bounds.

    A timeout is not a completed latency. The maximum recent observation is a
    conservative estimate, not a guarantee about a future request. Expiry allows
    a new cold probe; history is neither persisted nor shared between sessions.
    """
    def __init__(self, *, clock=time.monotonic, ttl_seconds=60., max_profiles=32, max_samples=16):
        if (not callable(clock) or not finite_nonnegative(ttl_seconds) or ttl_seconds <= 0
                or type(max_profiles) is not int or max_profiles < 1
                or type(max_samples) is not int or max_samples < 1):
            raise ValueError('Positive finite history bounds required')
        self.clock, self.ttl_seconds = clock, ttl_seconds
        self.max_profiles, self.max_samples = max_profiles, max_samples
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
        samples = self._history.setdefault(key, deque(maxlen=self.max_samples))
        samples.append((now, float(elapsed_ms), bool(censored)))
        self._history.move_to_end(key)
        while len(self._history) > self.max_profiles:
            self._history.popitem(last=False)

    def estimate(self, module, profile):
        key, now = self._key(module, profile), self.clock()
        samples = self._history.get(key)
        if not samples or not math.isfinite(now):
            return LatencyEstimate()
        recent = [s for s in samples if 0 <= now - s[0] < self.ttl_seconds]
        if not recent:
            self._history.pop(key, None)
            return LatencyEstimate()
        self._history[key] = deque(recent, maxlen=self.max_samples)
        self._history.move_to_end(key)
        slowest = max(recent, key=lambda s: (s[1], s[2]))
        return LatencyEstimate(slowest[1], 'censored_bound' if slowest[2] else 'completed_max',
                               sum(not s[2] for s in recent), sum(s[2] for s in recent))


@dataclass
class AdvisorWork:
    module: str
    profile: str | None
    qid: str | None
    started_at: float | None = None
    stop_elapsed_ms: float | None = None
    stop_reason: str = 'none'
    finished: bool = False
