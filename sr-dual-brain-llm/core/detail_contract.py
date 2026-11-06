"""Bounded ordinary-detail results, separate from critic and HTTP accounting."""
import math

STATUSES = frozenset({'ok', 'empty_response', 'incomplete_response', 'invalid_response',
                      'refused_response', 'failed', 'invalid_result', 'delivery_error'})
KINDS = frozenset({'external', 'heuristic', 'legacy', 'unavailable'})
ERROR_TYPES = frozenset({'TimeoutError', 'RuntimeError', 'ValueError', 'TypeError',
                         'OSError', 'ConnectionError', 'CompletionValidationError'})
FIELDS = ('notes_sum', 'confidence_r', 'detail_status', 'detail_kind',
          'detail_usable', 'detail_complete', 'detail_error_type')


def error_category(error):
    name = error if isinstance(error, str) else type(error).__name__
    return name if name in ERROR_TYPES else 'Error'


def detail_result(qid, text=None, *, status='ok', kind='external', error=None):
    if status == 'ok' and (not isinstance(text, str) or not text.strip()):
        status = 'empty_response'
    usable = status == 'ok' and isinstance(text, str) and bool(text.strip())
    return {'qid': qid, 'notes_sum': text.strip() if usable else None,
            'confidence_r': (.9 if kind == 'external' else .85) if usable else 0.,
            'detail_status': status, 'detail_kind': kind, 'detail_usable': usable,
            'detail_complete': usable and kind == 'external',
            'detail_error_type': error_category(error) if error is not None else None}


def detail_observation(response):
    """Revalidate at delivery; legacy nonempty notes carry no external proof."""
    if not isinstance(response, dict):
        response = {'detail_status': 'invalid_result'}
    error = response.get('error')
    missing_delivery = error or not any(key in response for key in ('notes_sum', 'detail_status', 'detail_kind'))
    kind = response.get('detail_kind', 'unavailable' if missing_delivery else 'legacy')
    status = response.get('detail_status', 'delivery_error' if missing_delivery else 'ok')
    if type(kind) is not str or type(status) is not str or kind not in KINDS or status not in STATUSES:
        kind, status = 'unavailable', 'invalid_result'
    text = response.get('notes_sum')
    nonempty = isinstance(text, str) and bool(text.strip())
    if error:
        status = 'delivery_error'
    elif status == 'ok' and not nonempty:
        status = 'empty_response'
    elif status == 'ok' and (kind == 'unavailable' or
            'detail_usable' in response and response['detail_usable'] is not True):
        status = 'invalid_result'
    elif status == 'ok' and kind == 'external' and (
            response.get('detail_usable') is not True or response.get('detail_complete') is not True):
        status = 'invalid_result'
    usable = status == 'ok' and nonempty
    return {'status': status, 'kind': kind, 'usable': usable,
            'external_complete': usable and kind == 'external',
            'error_type': error_category(error or response.get('detail_error_type'))
                          if error or response.get('detail_error_type') else None}


def detail_confidence(response):
    value = response.get('confidence_r')
    return float(value) if type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value) else 0.
