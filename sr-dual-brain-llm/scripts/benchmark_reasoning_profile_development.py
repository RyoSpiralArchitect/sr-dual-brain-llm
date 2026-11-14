#!/usr/bin/env python3
"""Separate new-task screen of ordinary reasoning effort and critic routing."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
from unittest.mock import patch

import benchmark_frontload_development as study
import reasoning_profile_tasks as tasks

ROOT = study.ROOT
PARENT_SOURCE = 'd1b79f887d71d0e5418b69fb8d9c9959ba53cb11'
PARENT_REPORT_SHA256 = 'e486484288afc1f8fcf315fa8c795a8682123cfd612ebf61e4bd6c0962f3ffd0'
PARENT_JOURNAL_SHA256 = '8f36c332b3a0a3611d033dff00e4109c7471e66127746432f73556ba3cfcc30c'
PARENT_REPLAY_SHA256 = '3c1222d95add4c543c9507d17c3fbe7fbc30fdae4a6f8b0ff36887b1626cd0cf'
ARMS = ('default_off', 'default_aware', 'medium_aware', 'low_aware', 'self_review')
ORDERS = tuple(ARMS[i:]+ARMS[:i] for i in range(len(ARMS)))
PROFILES = {
    'default_off': {'effort': 'provider_default', 'critic_aware': False},
    'default_aware': {'effort': 'provider_default', 'critic_aware': True},
    'medium_aware': {'effort': 'medium', 'critic_aware': True},
    'low_aware': {'effort': 'low', 'critic_aware': True},
    'self_review': {'effort': 'provider_default', 'critic_aware': False},
}
CAPS = {arm: {'draft': 4096, 'integration': 4096,
              'critic': 2048, 'critic_repair': 2048} for arm in ARMS if arm != 'self_review'}
CAPS['self_review'] = {'self_initial': 4096, 'self_critique': 2048,
                       'self_revision': 4096, 'self_audit': 2048}
original_manifest = study.expected_manifest
original_dual = study.base.dual
original_journal = study.Journal


class ProfileJournal(original_journal):
    async def post(self, role, url, payload, headers, **kwargs):
        arm = (self.case or '').rsplit(':', 1)[-1]
        expected = (PROFILES.get(arm, {}).get('effort')
                    if role in {'draft', 'integration'} else 'provider_default')
        expected = None if expected == 'provider_default' else expected
        actual = payload.get('reasoning_effort')
        if expected is None and 'reasoning_effort' in payload or expected is not None and actual != expected:
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
        'DUALBRAIN_CRITIC_COMPLETION_AWARE': '1' if profile['critic_aware'] else '0'}):
        return await original_dual(case, arm, config, journal, output_dir, clock)


def configure():
    study.tasks = tasks
    study.FIXTURE = ROOT/'sr-dual-brain-llm/examples/reasoning_profile_development.json'
    study.PROTOCOL = ROOT/'docs/benchmarks/reasoning-profile-development-protocol.md'
    study.MANIFEST = ROOT/'docs/benchmarks/reasoning-profile-development-manifest.json'
    study.ARMS, study.ORDERS, study.CAPS = ARMS, ORDERS, CAPS
    study.CALL_LIMIT = 96
    study.Journal = ProfileJournal
    study.expected_manifest = expected_manifest
    study.base.dual = profile_dual


def expected_manifest(fixture):
    return original_manifest(fixture) | {
        'reasoning_effort': 'per_arm_default_medium_low',
        'profiles': PROFILES,
        'parent_source': PARENT_SOURCE,
        'parent_report_sha256': PARENT_REPORT_SHA256,
        'parent_journal_sha256': PARENT_JOURNAL_SHA256,
        'parent_replay_failure_sha256': PARENT_REPLAY_SHA256}


def validate_parent(report_path, journal_path, replay_failure_path):
    read = study.base.pilot.bench.read_json
    paths = ((report_path, PARENT_REPORT_SHA256), (journal_path, PARENT_JOURNAL_SHA256),
             (replay_failure_path, PARENT_REPLAY_SHA256))
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != expected for path, expected in paths):
        raise ValueError('Stopped parent bytes differ from frozen lineage')
    report, journal, failure = map(read, (report_path, journal_path, replay_failure_path))
    if (report['source_revision'] != PARENT_SOURCE or report['outcome'] != 'transport_blocked'
            or report['usage']['calls'] != 7 or len(report['rows']) != 4
            or len(journal['occurrences']) != 7
            or journal['occurrences'][-1]['request']['case'] != 'logic_a:self_review'
            or journal['occurrences'][-1].get('error_type') != 'ClientOSError'
            or 'response' in journal['occurrences'][-1]
            or failure != {'complete': False, 'error_type': 'ValueError'}):
        raise ValueError('Only the recorded stopped first screen permits this separate study')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--parent-report', type=Path, required=True)
    parser.add_argument('--parent-journal', type=Path, required=True)
    parser.add_argument('--parent-replay-failure', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    validate_parent(args.parent_report, args.parent_journal, args.parent_replay_failure)
    configure()
    sys.argv = [sys.argv[0], *remaining]
    study.main()


if __name__ == '__main__':
    main()
