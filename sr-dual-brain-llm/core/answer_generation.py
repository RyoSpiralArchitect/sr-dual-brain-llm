"""Turn-local ordinary generation outcomes and bounded completion allocations.

Terminal completion is transport provenance, never proof of a correct solution.
Source generation retains its separate completion/commit authority.
"""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import hashlib
import math
import os
import re

from .turn_lifecycle import remaining_turn_ms

_capture = ContextVar('ordinary_completion_capture', default=None)
_turn = ContextVar('ordinary_answer_generation', default=None)
_effort = ContextVar('ordinary_reasoning_effort', default=None)
_SUCCESS = {'openai': {'stop'}, 'mistral': {'stop'}, 'xai': {'stop'},
            'anthropic': {'end_turn', 'stop_sequence'}, 'google': {'stop'},
            'huggingface': {'eos_token', 'stop_sequence'}}
_REASONS = {'stop', 'end_turn', 'stop_sequence', 'eos_token', 'length', 'max_tokens',
            'content_filter', 'tool_calls', 'tool_use', 'safety'}


def fingerprint(text):
    return hashlib.sha256(str(text or '').strip().encode()).hexdigest()


@dataclass(frozen=True)
class Generation:
    text: str = ''
    status: str = 'unavailable'
    finish_reason: str = 'unknown'
    error_type: str | None = None
    request_reasoning_effort: str | None = None


def record_completion(provider, text, finish_reason):
    """Called only after a provider envelope has passed its strict validation."""
    capture = _capture.get()
    if capture is not None:
        finish = str(finish_reason or '').lower()
        status = ('empty' if not text.strip() else 'complete' if finish in _SUCCESS.get(provider, ())
                  else 'partial' if finish in {'', 'length', 'max_tokens'} else 'invalid')
        capture['result'] = Generation(text.strip(), status,
            finish if finish in _REASONS else 'unknown', request_reasoning_effort=capture.get('effort'))


def record_stream_text(text):
    # A connection can fail after visible deltas. Retain partial provenance,
    # without treating a missing terminal event as completion.
    capture = _capture.get()
    if capture is not None:
        capture['result'] = Generation(text, 'partial' if text else 'empty',
            request_reasoning_effort=capture.get('effort'))


def record_request_effort(effort):
    capture = _capture.get()
    if capture is not None:
        capture['effort'] = effort


def requested_reasoning_effort():
    return _effort.get()


def capturing_completion():
    return _capture.get() is not None


@contextmanager
def reasoning_profile():
    value = os.environ.get('DUALBRAIN_ANSWER_REASONING_EFFORT', 'provider_default').strip().lower()
    if value not in {'provider_default', 'low', 'medium', 'high'}:
        raise ValueError('Invalid ordinary reasoning profile')
    token = _effort.set(None if value == 'provider_default' else value)
    try:
        yield
    finally:
        _effort.reset(token)


class AnswerGeneration:
    def __init__(self):
        self.records = []
        self.answers = {}
        self.total_cap = None
        self.reserved = 0

    def reserve(self, stage_cap):
        if type(stage_cap) is not int or stage_cap < 1:
            return 0
        if self.total_cap is None:
            # At most a draft and three existing integration opportunities.
            # This is an allocation of ceilings, not billed-token accounting.
            bound = 4 * stage_cap
            try:
                requested = int(os.environ.get('DUALBRAIN_ANSWER_TOKEN_BUDGET', str(bound)))
            except (TypeError, ValueError):
                requested = 0
            self.total_cap = min(bound, max(0, requested))
        available = min(stage_cap, self.total_cap - self.reserved)
        self.reserved += available
        return available

    def record(self, stage, result, returned, cap, *, preserve_existing=False):
        self.records.append({'stage': stage, 'status': result.status,
            'finish_reason': result.finish_reason, 'nonempty_text': bool(result.text.strip()),
            'error_type': result.error_type, 'max_output_tokens': cap,
            'request_reasoning_effort': result.request_reasoning_effort})
        # An unsuccessful integration returning a known good draft cannot
        # erase its provenance. Actual new output always gets its own status.
        key = fingerprint(returned)
        if not preserve_existing or key not in self.answers:
            self.answers[key] = result.status if returned == result.text or not returned else 'unverified'

    def observed(self, text):
        return fingerprint(text) in self.answers

    def status(self, text):
        return self.answers.get(fingerprint(text), 'unverified')

    def summary(self, answer):
        status = self.status(answer) if self.records else 'not_observed'
        return {'version': 'ordinary_generation_v1', 'answer_status': status,
            'terminal_complete': status == 'complete', 'semantic_correctness': 'not_assessed',
            'stages': list(self.records), 'reserved_output_tokens': self.reserved,
            'total_output_token_cap': self.total_cap}

    def present(self, answer, question, *, constrain=None):
        status = self.status(answer)
        japanese = bool(re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', question))
        text = answer
        if status == 'partial' and answer.strip():
            text = ('回答は途中までです。\n\n' if japanese else 'This answer is incomplete.\n\n') + answer
        elif self.records and not answer.strip():
            text = ('回答を生成できませんでした。もう一度お試しください。' if japanese else
                    'I could not generate an answer. Please try again.')
        if text != answer:
            if constrain is not None:
                text = constrain(text)
            self.answers[fingerprint(text)] = status
        return text


def current_generation():
    return _turn.get()


def guard_answer_generation(method):
    @wraps(method)
    async def guarded(self, *args, **kwargs):
        from .source_completion import source_completion_required
        state = None if source_completion_required() else AnswerGeneration()
        token = _turn.set(state)
        try:
            return await method(self, *args, **kwargs)
        except BaseException as exc:
            if state is not None:
                try:
                    self.telemetry.log('answer_generation', **state.summary(''),
                        turn_status='cancelled' if isinstance(exc, asyncio.CancelledError) else 'failed')
                except Exception:
                    pass
            raise
        finally:
            _turn.reset(token)
    return guarded


async def generate(client, config, stage, prompt, *, system, temperature, on_delta=None):
    """One ordinary request, no continuation or retry; propagate cancellation."""
    state = current_generation()
    if client is None:
        return Generation(), 0
    cap = getattr(config, 'max_output_tokens', 1024)
    if type(cap) is not int or cap < 1:
        return Generation(status='failed', error_type='ValueError'), 0
    cap = state.reserve(cap) if state is not None else cap
    remaining = remaining_turn_ms()
    if not cap or remaining is not None and remaining <= 0:
        return Generation(status='budget_exhausted'), 0
    capture = {}
    token = _capture.set(capture)
    try:
        with reasoning_profile():
            timeout = float(getattr(config, 'timeout_seconds', 40))
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError('Invalid ordinary generation timeout')
            if remaining is not None:
                timeout = min(timeout, remaining / 1000.)
            method = client.complete_stream if hasattr(client, 'complete_stream') else client.complete
            kwargs = dict(system=system, temperature=temperature, max_output_tokens=cap,
                timeout_seconds=timeout, require_complete=True, allow_continuation=False)
            if hasattr(client, 'complete_stream'):
                kwargs['on_delta'] = on_delta
            text = await method(prompt, **kwargs)
            if not isinstance(text, str):
                return Generation(status='invalid', error_type='TypeError'), cap
            result = capture.get('result')
            if result is None or result.text.strip() != text.strip():
                result = Generation(text.strip(), 'unverified' if text.strip() else 'empty')
            if on_delta and not hasattr(client, 'complete_stream'):
                value = on_delta(result.text)
                if asyncio.iscoroutine(value):
                    await value
            return result, cap
    except asyncio.CancelledError:
        if state is not None:
            partial = capture.get('result')
            state.record(stage, Generation(text=partial.text if partial else '', status='cancelled',
                request_reasoning_effort=capture.get('effort')), '', cap)
        raise
    except Exception as exc:
        result = capture.get('result')
        name = type(exc).__name__
        error = name if name in {'CompletionValidationError', 'TimeoutError', 'RuntimeError', 'ValueError', 'TypeError', 'OSError'} else 'Error'
        reason = getattr(exc, 'reason', None)
        if result is not None and reason not in {'invalid_response', 'refused_response'}:
            return Generation(result.text, result.status, result.finish_reason, error, result.request_reasoning_effort), cap
        status = 'invalid' if reason in {'invalid_response', 'refused_response'} else 'empty' if reason == 'empty_response' else 'failed'
        return Generation(status=status, error_type=error, request_reasoning_effort=capture.get('effort')), cap
    finally:
        _capture.reset(token)


def record_generation(stage, result, returned, cap, *, preserve_existing=False):
    state = current_generation()
    if state is not None:
        state.record(stage, result, returned, cap, preserve_existing=preserve_existing)
