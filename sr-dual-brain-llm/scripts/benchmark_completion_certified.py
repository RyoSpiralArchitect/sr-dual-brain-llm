#!/usr/bin/env python3
"""Separate fresh completion study after a diagnosed local CA-store failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import ssl
import sys

import benchmark_completion_mechanisms as study
import completion_benchmark_certified_tasks as tasks

ROOT = study.ROOT
PARENT_SOURCE = 'a970d0844db77e16011197d5f85f10ca4596e510'
PARENT_REPORT_SHA256 = 'e889935662b546d65f8ddf6efbbc0c0e5534f41799b50845f36c2fa43e4c067b'
PARENT_JOURNAL_SHA256 = 'df46b895484373890f82e4ba07e45ee67663aff1e20945e941e8f4826e0900be'
PARENT_REPLAY_SHA256 = '3c1222d95add4c543c9507d17c3fbe7fbc30fdae4a6f8b0ff36887b1626cd0cf'
PARENT_CANONICAL_SHA256 = 'f2cd03de45063e9c8903c1217f153cf199ac8ac17664766a06f50a34b39d25f6'
original_manifest = study.expected_manifest


def configure():
    study.tasks = tasks
    study.FIXTURE = ROOT/'sr-dual-brain-llm/examples/completion_mechanisms_certified.json'
    study.PROTOCOL = ROOT/'docs/benchmarks/completion-mechanisms-certified-protocol.md'
    study.MANIFEST = ROOT/'docs/benchmarks/completion-mechanisms-certified-manifest.json'
    study.expected_manifest = expected_manifest


def expected_manifest(fixture):
    return original_manifest(fixture) | {
        'parent_source': PARENT_SOURCE,
        'parent_report_sha256': PARENT_REPORT_SHA256,
        'parent_journal_sha256': PARENT_JOURNAL_SHA256,
        'parent_replay_failure_sha256': PARENT_REPLAY_SHA256,
        'parent_canonical_sha256': PARENT_CANONICAL_SHA256,
        'ca_preflight': 'SSL_CERT_FILE=certifi.where, nonempty default context'}


def validate_parent(report_path, journal_path, replay_failure_path):
    read = study.base.pilot.bench.read_json
    paths = ((report_path, PARENT_REPORT_SHA256), (journal_path, PARENT_JOURNAL_SHA256),
             (replay_failure_path, PARENT_REPLAY_SHA256))
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != expected for path, expected in paths):
        raise ValueError('Stopped parent bytes differ from frozen lineage')
    report, journal, replay = map(read, (report_path, journal_path, replay_failure_path))
    if (report['source_revision'] != PARENT_SOURCE or report['outcome'] != 'transport_blocked'
            or report['canonical_sha256'] != PARENT_CANONICAL_SHA256
            or report['canonical_sha256'] != study.base.pilot.bench.digest(study.canonical(report))
            or report['usage']['calls'] != 1 or len(report['rows']) != 1
            or report['rows'][0]['case'] != 'logic_a' or report['rows'][0]['arm'] != 'off'
            or len(journal['occurrences']) != 1
            or journal['occurrences'][0]['request']['case'] != 'logic_a:off'
            or journal['occurrences'][0].get('error_type') != 'ClientConnectorCertificateError'
            or 'response' in journal['occurrences'][0]
            or journal['occurrences'][0].get('http_status') is not None
            or replay != {'complete': False, 'error_type': 'ValueError'}):
        raise ValueError('Only the recorded certificate failure permits this separate study')


def validate_ca_environment():
    import certifi
    chosen = os.environ.get('SSL_CERT_FILE')
    if (not chosen or Path(chosen).resolve() != Path(certifi.where()).resolve()
            or len(ssl.create_default_context().get_ca_certs()) < 100):
        raise ValueError('Corrected local CA bundle is not active')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--parent-report', type=Path, required=True)
    parser.add_argument('--parent-journal', type=Path, required=True)
    parser.add_argument('--parent-replay-failure', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    validate_parent(args.parent_report, args.parent_journal, args.parent_replay_failure)
    if '--model' in remaining:
        validate_ca_environment()
    configure()
    sys.argv = [sys.argv[0], *remaining]
    study.main()


if __name__ == '__main__':
    main()
