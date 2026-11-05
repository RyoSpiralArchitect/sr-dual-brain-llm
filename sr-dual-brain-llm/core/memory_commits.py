"""Immutable conversation-memory operations and a durable, local retry outbox."""
from __future__ import annotations

from contextlib import closing
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3


class MemoryCommitConflict(ValueError):
    pass


class MemoryCommitPending(RuntimeError):
    def __init__(self, operation_id):
        self.operation_id = operation_id
        super().__init__(f"Memory commit unconfirmed; reconcile operation {operation_id}")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def payload_fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_operation_id(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError("memory_operation_id must be a nonempty string of at most 128 characters")


def memory_payload(traces, updates):
    return {"version": 1, "traces": [asdict(t) for t in traces], "updates": [asdict(t) for t in updates]}


def decode_payload(payload):
    from .shared_memory import MemoryTrace

    if (not isinstance(payload, dict) or set(payload) != {"version", "traces", "updates"}
            or payload["version"] != 1 or not isinstance(payload["traces"], list)
            or not isinstance(payload["updates"], list) or not payload["traces"]):
        raise ValueError("Expected a versioned, nonempty memory operation")
    canonical_json(payload)  # Reject nonfinite numbers before any writes.

    def trace(row):
        values = dict(row)
        for key in ("tags", "question_tokens", "scope_aliases"):
            values[key] = tuple(values.get(key, ()))
        return MemoryTrace(**values)

    return [trace(row) for row in payload["traces"]], [trace(row) for row in payload["updates"]]


class MemoryOutbox:
    """Fsync the exact generated payload before attempting a remote commit.

    Keep the file on persistent local storage and reuse it after restart. A
    database-target namespace prevents accidentally replaying into another store.
    SQLite is only an outbox; PostgreSQL receipts decide whether memory committed.
    """

    def __init__(self, path, namespace, *, legacy_namespaces=()):
        self.path, self.namespace = Path(path), namespace
        self.legacy_namespaces = ()
        self.add_legacy_guards(legacy_namespaces)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        with closing(self._connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS memory_outbox (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL, session_id TEXT NOT NULL, operation_id TEXT NOT NULL,
                request TEXT NOT NULL, payload TEXT NOT NULL, result TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL, receipt TEXT,
                UNIQUE(namespace, session_id, operation_id)
            )""")

    def _connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        return db

    def add_legacy_guards(self, namespaces):
        """Keep every known legacy constraint when binding this outbox to a store."""
        self.legacy_namespaces = tuple(dict.fromkeys((
            *self.legacy_namespaces, *(item for item in namespaces if item != self.namespace))))

    def _check_legacy_target(self, db, session_id):
        if not self.legacy_namespaces:
            return
        # Reads share one snapshot; writes begin their transaction before this
        # check. Both pending operations and saved acknowledged responses count.
        if not db.in_transaction:
            db.execute("BEGIN")
        for namespace in self.legacy_namespaces:
            if db.execute("SELECT 1 FROM memory_outbox WHERE namespace=? AND session_id=? LIMIT 1",
                          (namespace, session_id)).fetchone() is not None:
                raise MemoryCommitConflict(
                    "Legacy outbox target is ambiguous for this session; retain the original outbox "
                    "and verify its database/schema before recovery. Automatic migration is disabled."
                )

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        result = dict(row)
        for key in ("request", "payload", "result", "receipt"):
            result[key] = json.loads(result[key]) if result[key] is not None else None
        if payload_fingerprint(result["payload"]) != result["payload_sha256"]:
            raise MemoryCommitConflict("Outbox payload fingerprint mismatch")
        return result

    def get(self, session_id, operation_id):
        validate_operation_id(operation_id)
        with closing(self._connect()) as db:
            self._check_legacy_target(db, session_id)
            row = db.execute("SELECT * FROM memory_outbox WHERE namespace=? AND session_id=? AND operation_id=?",
                             (self.namespace, session_id, operation_id)).fetchone()
        return self._decode(row)

    def entries(self, session_id, *, pending_only=False):
        with closing(self._connect()) as db:
            self._check_legacy_target(db, session_id)
            rows = db.execute("SELECT * FROM memory_outbox WHERE namespace=? AND session_id=? "
                              + ("AND receipt IS NULL " if pending_only else "") + "ORDER BY sequence",
                              (self.namespace, session_id)).fetchall()
        return [self._decode(row) for row in rows]

    def prepare(self, session_id, operation_id, payload, *, request=None, result=None):
        validate_operation_id(operation_id)
        decode_payload(payload)
        encoded, request_json = canonical_json(payload), canonical_json(request)
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            self._check_legacy_target(db, session_id)
            db.execute("INSERT OR IGNORE INTO memory_outbox "
                       "(namespace, session_id, operation_id, request, payload, result, payload_sha256) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (self.namespace, session_id, operation_id, request_json, encoded,
                        canonical_json(result), payload_fingerprint(payload)))
            row = db.execute("SELECT * FROM memory_outbox "
                             "WHERE namespace=? AND session_id=? AND operation_id=?",
                             (self.namespace, session_id, operation_id)).fetchone()
            if row["request"] != request_json or row["payload"] != encoded:
                raise MemoryCommitConflict("Operation ID already has a different request or memory payload")
            return self._decode(row)

    def acknowledge(self, session_id, operation_id, receipt):
        validate_operation_id(operation_id)
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            self._check_legacy_target(db, session_id)
            entry = self._decode(db.execute(
                "SELECT * FROM memory_outbox WHERE namespace=? AND session_id=? AND operation_id=?",
                (self.namespace, session_id, operation_id)).fetchone())
            if (entry is None or receipt.get("operation_id") != operation_id
                    or receipt.get("payload_sha256") != entry["payload_sha256"] or receipt.get("state") != "committed"
                    or (entry["receipt"] is not None and entry["receipt"] != receipt)):
                raise MemoryCommitConflict("Memory receipt does not match the prepared operation")
            db.execute("UPDATE memory_outbox SET receipt=? WHERE namespace=? AND session_id=? AND operation_id=?",
                       (canonical_json(receipt), self.namespace, session_id, operation_id))


async def reconcile_memory(outbox, store, session_id, *, operation_id=None):
    if outbox.namespace != store.memory_namespace:
        raise MemoryCommitConflict("Outbox target does not match the configured memory store")
    # Direct callers may have used the original two-argument constructor. The
    # store's upgrade guard still applies before any lookup or backend call,
    # and remains attached for subsequent reads/writes through this object.
    outbox.add_legacy_guards(getattr(store, "memory_legacy_namespaces", ()))
    entries = ([outbox.get(session_id, operation_id)] if operation_id is not None
               else outbox.entries(session_id, pending_only=True))
    receipts = []
    for entry in entries:
        if entry is None:
            raise ValueError("Unknown memory operation; the original outbox is required to retry")
        op_id = entry["operation_id"]
        try:
            if entry["receipt"] is not None:
                # An old receipt is evidence of a past commit, not a command to
                # reapply its old revision (including after an explicit reset).
                receipt = await store.memory_commit_receipt(session_id, op_id)
                if receipt != entry["receipt"]:
                    raise MemoryCommitConflict("Durable receipt no longer matches the outbox; do not replay")
            else:
                receipt = await store.commit_memory_operation(session_id, op_id, entry["payload"])
            outbox.acknowledge(session_id, op_id, receipt)
        except MemoryCommitConflict:
            raise
        except Exception:
            raise MemoryCommitPending(op_id) from None
        receipts.append(receipt)
    return receipts
