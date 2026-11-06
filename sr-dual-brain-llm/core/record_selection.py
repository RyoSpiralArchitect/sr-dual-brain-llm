"""Instance-bound read-only references; selecting a record never approves it."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
from dataclasses import dataclass

from .memory_intent import _digest


@dataclass(frozen=True)
class RecordSelectionRef:
    scope: str
    qid: str
    source_sha256: str


class RecordSelectionCodec:
    """A fresh memory instance gets a fresh issuer key; refs survive no restart.

    The signed payload contains identity/fingerprint metadata only, never QA
    text. Verification is stateless; current scope, source and approval are
    checked separately on every use. The key is never persisted or serialized.
    """
    def __init__(self):
        self._key = secrets.token_bytes(32)

    def issue(self, scope, source):
        data = {'scope': scope, 'qid': source['qid'], 'source_sha256': _digest(source)}
        raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        payload = base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii')
        signature = hmac.new(self._key, ('r1.' + payload).encode(), hashlib.sha256).hexdigest()
        return 'r1.' + payload + '.' + signature

    def decode(self, token):
        if not isinstance(token, str) or not 1 <= len(token) <= 4096:
            return None
        match = re.fullmatch(r'r1\.([A-Za-z0-9_-]+)\.([0-9a-f]{64})', token)
        if match is None:
            return None
        payload, signature = match.groups()
        expected = hmac.new(self._key, ('r1.' + payload).encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        try:
            raw = base64.b64decode(payload + '=' * (-len(payload) % 4), altchars=b'-_', validate=True)
            data = json.loads(raw)
            if (not isinstance(data, dict) or set(data) != {'scope', 'qid', 'source_sha256'}
                    or not isinstance(data['scope'], str) or not 1 <= len(data['scope']) <= 128
                    or not isinstance(data['qid'], str) or not 1 <= len(data['qid']) <= 256
                    or not isinstance(data['source_sha256'], str)
                    or re.fullmatch(r'[0-9a-f]{64}', data['source_sha256']) is None):
                return None
            return RecordSelectionRef(**data)
        except (ValueError, TypeError, UnicodeError):
            return None

    @staticmethod
    def unchanged(selection, source):
        try:
            return hmac.compare_digest(selection.source_sha256, _digest(source))
        except (ValueError, TypeError, OverflowError):
            return False
