"""Prospective obligation-audited advice assessment; never runtime admission."""
from __future__ import annotations

import json

import advisor_assessment as original
import advisor_assessment_v2 as previous

VERSION = 'explicit_obligations_v3'
CRITERIA = previous.CRITERIA
BASIS = previous.BASIS
MAX_OUTPUT_TOKENS = 4096
INSTRUCTIONS = previous.INSTRUCTIONS + (
    ' Additional explicit boundary contract: obligations is a finite mapping of required role obligations. '
    'For critical_coverage return obligation_checks with exactly those IDs and a pass/fail/uncertain for each. '
    'Check only what each obligation actually requires. Do not invent subrequirements, extra tests, execution '
    'steps, sensor identity, timing or operational detail. An explicitly unknown outcome plus a concrete '
    'proposed way to obtain the missing observation satisfies an obligation asking for those two things; '
    'it does not need an already completed observation or a more detailed procedure. Read across all fields. '
    'Coverage is fail if any required obligation fails, pass if all pass, otherwise uncertain. '
    'This audit does not automatically settle relevance, grounding or potential_use. '
    'For relevance distinguish a claim ABOUT the assigned monitoring task from taking a DIFFERENT actor role '
    'or changing the requested scope. A false claim that the assigned observer already saw hidden task-related '
    'text is an observer_access grounding defect and may contradict a coverage obligation, but the false '
    'attestation alone does not change its topic/actor role: relevance passes if otherwise aligned. '
    'Advice directing that observer to execute an external action instead of monitoring, or discussing an '
    'unrelated topic, fails relevance. Explicitly requested scope/count restrictions still apply. '
    'Do not derive scores from labels, expected values or whether another criterion failed.'
)


def obligations_for(case, role):
    obligations = case['advice_obligations'][role] if 'advice_obligations' in case else case['obligations']
    if (not isinstance(obligations, dict) or not 1 <= len(obligations) <= 8
            or any(not isinstance(k, str) or not k.isidentifier() or not isinstance(v, str) or not v.strip()
                   for k, v in obligations.items())):
        raise ValueError('Explicit bounded obligation IDs required')
    return obligations


def schema(case, labels):
    result = previous.schema(labels)
    outputs = result['json_schema']['schema']['properties']['scores']['properties']
    for label, output in labels.items():
        coverage = outputs[label]['properties']['critical_coverage']
        obligations = obligations_for(case, output['role'])
        coverage['properties']['obligation_checks'] = {'type': 'object', 'properties': {
            name: {'type': 'string', 'enum': ['pass', 'fail', 'uncertain']} for name in obligations},
            'required': list(obligations), 'additionalProperties': False}
        coverage['required'].append('obligation_checks')
    return result


def validate_scores(raw, case, labels):
    parsed = json.loads(raw, object_pairs_hook=original._unique_object)
    if not isinstance(parsed, dict) or set(parsed) != {'scores'} or set(parsed['scores']) != set(labels):
        raise ValueError('Assessment labels differ')
    audits = {}
    for label, output in labels.items():
        coverage = parsed['scores'][label]['critical_coverage']
        audit = coverage.pop('obligation_checks')
        if (not isinstance(audit, dict) or set(audit) != set(obligations_for(case, output['role']))
                or any(v not in {'pass', 'fail', 'uncertain'} for v in audit.values())):
            raise ValueError('Obligation coverage differs')
        overall = 'fail' if 'fail' in audit.values() else 'pass' if set(audit.values()) == {'pass'} else 'uncertain'
        if coverage['verdict'] != overall:
            raise ValueError('Coverage verdict differs from obligation audit')
        audits[label] = audit
    scores = previous.validate_scores(json.dumps(parsed, ensure_ascii=False), labels)
    for label in labels:
        scores[label]['critical_coverage']['obligation_checks'] = audits[label]
    return scores


async def assess(case, labels, config, journal, *, role='advice_judge_v3'):
    packet = original.build_packet(case, labels)
    client = original.bench.LLMClient(config)
    payload = client._openai_style_payload([
        {'role': 'system', 'content': INSTRUCTIONS},
        {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False)}],
        temperature=0., max_output_tokens=MAX_OUTPUT_TOKENS)
    payload['response_format'] = schema(case, labels)
    headers = {'Authorization': 'Bearer ' + config.api_key, 'Content-Type': 'application/json', **config.extra_headers}
    if config.organization:
        headers['OpenAI-Organization'] = config.organization
    url = (config.api_base or client._default_base()).rstrip('/') + '/chat/completions'
    response = await journal.post(role, url, payload, headers, timeout_seconds=60)
    choice = response['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Assessment completion is not complete')
    return validate_scores(choice['message']['content'], case, labels)
