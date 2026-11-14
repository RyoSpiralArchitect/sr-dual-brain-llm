#!/usr/bin/env python3
"""Untouched eight-case holdout after a fully replayed development gate."""
import argparse
import hashlib
from pathlib import Path
import sys

import benchmark_frontload_development as study
import benchmark_critic_cap_development as dev
import critic_cap_holdout_tasks as tasks

ROOT = study.ROOT
PARENT_SOURCE = 'f9507dde60672ba9c99b5d11336012cbd30ad22c'
PARENT_REPORT_SHA256 = 'b3b5ee9135d1a56d1a799e0c56ec53d79c501302f72576a221d20f9e664fe3b0'
PARENT_JOURNAL_SHA256 = '8ca16cc3394246aaaeb47d5c24d59e13c0d6074ea15eb4157a33969ac74e209d'
PARENT_REPLAY_REPORT_SHA256 = '855736ede9a24ae514c94bb3d2837db95b14437659771bcc950532b7143f3cba'
PARENT_REPLAY_JOURNAL_SHA256 = PARENT_JOURNAL_SHA256
ARMS = ('default_off', 'low_aware', 'low_critic4096')
ORDERS = tuple(ARMS[i % 3:]+ARMS[:i % 3] for i in range(8))
PROFILES = {arm: dev.PROFILES[arm] for arm in ARMS}
CAPS = {arm: dev.CAPS[arm] for arm in ARMS}
original_manifest = dev.original_manifest


def configure():
    study.tasks = tasks
    study.FIXTURE = ROOT/'sr-dual-brain-llm/examples/critic_cap_holdout.json'
    study.PROTOCOL = ROOT/'docs/benchmarks/critic-cap-holdout-protocol.md'
    study.MANIFEST = ROOT/'docs/benchmarks/critic-cap-holdout-manifest.json'
    study.ARMS, study.ORDERS, study.CAPS = ARMS, ORDERS, CAPS
    study.CALL_LIMIT, study.PHASE = 80, 'holdout'
    study.Journal = dev.ProfileJournal
    study.expected_manifest = expected_manifest
    study.base.dual = dev.profile_dual


def expected_manifest(fixture):
    return original_manifest(fixture) | {
        'reasoning_effort': 'per_arm_default_or_low',
        'profiles': PROFILES,
        'phase': 'untouched_holdout_after_adaptive_development',
        'selected_candidate': 'low_critic4096',
        'parent_source': PARENT_SOURCE,
        'parent_report_sha256': PARENT_REPORT_SHA256,
        'parent_journal_sha256': PARENT_JOURNAL_SHA256,
        'parent_replay_report_sha256': PARENT_REPLAY_REPORT_SHA256,
        'parent_replay_journal_sha256': PARENT_REPLAY_JOURNAL_SHA256}


def _score(row):
    return row['grade']['item_accuracy'] if row['answer_complete'] else 0.


def development_gate(report):
    if (report['source_revision'] != PARENT_SOURCE or report['outcome'] != 'comparisons_complete'
            or len(report['rows']) != 20 or report['usage']['calls'] != 45):
        return False
    rows = {(row['case'], row['arm']): row for row in report['rows']}
    if len(rows) != 20:
        return False
    baseline, low, candidate = (report['summary']['arms'][arm]
                                for arm in ('default_off', 'low_aware', 'low_critic4096'))
    if (candidate['exact'] < baseline['exact'] + 1
            or candidate['mean_item_accuracy'] < baseline['mean_item_accuracy']
            or candidate['workflow_valid'] < low['workflow_valid']):
        return False
    if any(_score(rows[(case, 'low_critic4096')]) < _score(rows[(case, 'default_off')])
           for case in ('sql', 'events', 'evidence')):
        return False
    reviews = lambda arm: sum(bool(row['system2'].get('external_review_completed'))
                              for row in report['rows'] if row['arm'] == arm)
    return reviews('low_critic4096') >= reviews('low_aware') + 1


def validate_parent(report_path, journal_path, replay_report_path, replay_journal_path):
    read = study.base.pilot.bench.read_json
    paths = ((report_path, PARENT_REPORT_SHA256), (journal_path, PARENT_JOURNAL_SHA256),
             (replay_report_path, PARENT_REPLAY_REPORT_SHA256),
             (replay_journal_path, PARENT_REPLAY_JOURNAL_SHA256))
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != expected for path, expected in paths):
        raise ValueError('Development parent bytes differ from frozen lineage')
    report, journal, replay, replay_journal = map(read,
        (report_path, journal_path, replay_report_path, replay_journal_path))
    if (not development_gate(report)
            or report['canonical_sha256'] != replay['canonical_sha256']
            or report['usage'] != replay['usage']
            or report['phase_receipt'] != replay['phase_receipt']
            or replay.get('unexpected_blocked_http_attempts') != 0
            or len(journal['occurrences']) != 45
            or journal != replay_journal):
        raise ValueError('Only the replayed development winner permits this holdout')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--parent-report', type=Path, required=True)
    parser.add_argument('--parent-journal', type=Path, required=True)
    parser.add_argument('--parent-replay-report', type=Path, required=True)
    parser.add_argument('--parent-replay-journal', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    validate_parent(args.parent_report, args.parent_journal,
                    args.parent_replay_report, args.parent_replay_journal)
    configure()
    sys.argv = [sys.argv[0], *remaining]
    study.main()


if __name__ == '__main__':
    main()
