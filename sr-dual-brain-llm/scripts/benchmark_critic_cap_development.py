#!/usr/bin/env python3
"""Fresh native-only comparison of low ordinary effort and critic capacity."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
from unittest.mock import patch

import benchmark_frontload_development as study
import critic_cap_tasks as tasks

ROOT = study.ROOT
PARENT_SOURCE = '98f77d35bd4278c77952af0a57d3b470d8b4a53b'
PARENT_REPORT_SHA256 = '192d11b35aa9903bfe10f79f64992b8ccfca9a3d9fa21b32c0b1b6efd1d75d97'
PARENT_JOURNAL_SHA256 = '1c0150eec419385320f02babd02ca3711f0f13e4d8192cdfc7c2384507e85858'
PARENT_REPLAY_REPORT_SHA256 = 'ba873d557c84af0d35cbc548792cd505ad23fcdc234454e2f8c89025cbb213aa'
PARENT_REPLAY_FAILURE_SHA256 = '3c1222d95add4c543c9507d17c3fbe7fbc30fdae4a6f8b0ff36887b1626cd0cf'
ARMS = ('default_off', 'default_aware', 'low_aware', 'low_critic4096')
ORDERS = tuple(ARMS[i % 4:]+ARMS[:i % 4] for i in range(5))
PROFILES = {
    'default_off': {'effort': 'provider_default', 'critic_aware': False, 'critic_cap': 2048},
    'default_aware': {'effort': 'provider_default', 'critic_aware': True, 'critic_cap': 2048},
    'low_aware': {'effort': 'low', 'critic_aware': True, 'critic_cap': 2048},
    'low_critic4096': {'effort': 'low', 'critic_aware': True, 'critic_cap': 4096},
}
CAPS = {arm: {'draft': 4096, 'integration': 4096,
              'critic': profile['critic_cap'], 'critic_repair': profile['critic_cap']}
        for arm, profile in PROFILES.items()}
original_manifest = study.expected_manifest
original_dual = study.base.dual
original_journal = study.Journal


class ProfileJournal(original_journal):
    async def post(self, role, url, payload, headers, **kwargs):
        arm = (self.case or '').rsplit(':', 1)[-1]
        effort = PROFILES.get(arm, {}).get('effort') if role in {'draft', 'integration'} else None
        expected = None if effort == 'provider_default' else effort
        actual = payload.get('reasoning_effort')
        mismatch = (('reasoning_effort' in payload) if expected is None else actual != expected)
        if mismatch:
            self.configuration_fault = {'arm': arm, 'role': role,
                                        'actual_effort': actual, 'expected_effort': expected}
            self.data['configuration_fault'] = self.configuration_fault
            self.checkpoint()
            raise study.base.pilot.AdmissionError('Serialized reasoning effort differs')
        return await super().post(role, url, payload, headers, **kwargs)


async def profile_dual(case, arm, config, journal, output_dir, clock):
    profile = PROFILES[arm]
    with patch.dict(os.environ, {
        'DUALBRAIN_ANSWER_REASONING_EFFORT': profile['effort'],
        'DUALBRAIN_CRITIC_COMPLETION_AWARE': '1' if profile['critic_aware'] else '0',
        'DUALBRAIN_CRITIC_MAX_OUTPUT_TOKENS': str(profile['critic_cap'])}):
        return await original_dual(case, arm, config, journal, output_dir, clock)


def configure():
    study.tasks = tasks
    study.FIXTURE = ROOT/'sr-dual-brain-llm/examples/critic_cap_development.json'
    study.PROTOCOL = ROOT/'docs/benchmarks/critic-cap-development-protocol.md'
    study.MANIFEST = ROOT/'docs/benchmarks/critic-cap-development-manifest.json'
    study.ARMS, study.ORDERS, study.CAPS = ARMS, ORDERS, CAPS
    study.CALL_LIMIT = 80
    study.Journal = ProfileJournal
    study.expected_manifest = expected_manifest
    study.base.dual = profile_dual


def expected_manifest(fixture):
    return original_manifest(fixture) | {
        'reasoning_effort': 'per_arm_default_or_low',
        'profiles': PROFILES,
        'adaptive_from_stopped_profile_screen': True,
        'parent_source': PARENT_SOURCE,
        'parent_report_sha256': PARENT_REPORT_SHA256,
        'parent_journal_sha256': PARENT_JOURNAL_SHA256,
        'parent_replay_report_sha256': PARENT_REPLAY_REPORT_SHA256,
        'parent_replay_failure_sha256': PARENT_REPLAY_FAILURE_SHA256}


def validate_parent(report_path, journal_path, replay_report_path, replay_failure_path):
    read = study.base.pilot.bench.read_json
    paths = ((report_path, PARENT_REPORT_SHA256), (journal_path, PARENT_JOURNAL_SHA256),
             (replay_report_path, PARENT_REPLAY_REPORT_SHA256),
             (replay_failure_path, PARENT_REPLAY_FAILURE_SHA256))
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != expected for path, expected in paths):
        raise ValueError('Stopped parent bytes differ from frozen lineage')
    report, journal, replay, failure = map(read,
        (report_path, journal_path, replay_report_path, replay_failure_path))
    if (report['source_revision'] != PARENT_SOURCE or report['outcome'] != 'transport_blocked'
            or report['usage']['calls'] != 10 or len(report['rows']) != 5
            or len(journal['occurrences']) != 10
            or journal['occurrences'][-1]['request']['case'] != 'logic_a:self_review'
            or journal['occurrences'][-1].get('error_type') != 'ClientOSError'
            or 'response' in journal['occurrences'][-1]
            or replay['usage'] != report['usage']
            or replay['phase_receipt'] != report['phase_receipt']
            or replay['canonical_sha256'] == report['canonical_sha256']
            or failure != {'complete': False, 'error_type': 'ValueError'}):
        raise ValueError('Only the recorded stopped profile screen permits this study')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--parent-report', type=Path, required=True)
    parser.add_argument('--parent-journal', type=Path, required=True)
    parser.add_argument('--parent-replay-report', type=Path, required=True)
    parser.add_argument('--parent-replay-failure', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    validate_parent(args.parent_report, args.parent_journal,
                    args.parent_replay_report, args.parent_replay_failure)
    configure()
    sys.argv = [sys.argv[0], *remaining]
    study.main()


if __name__ == '__main__':
    main()
