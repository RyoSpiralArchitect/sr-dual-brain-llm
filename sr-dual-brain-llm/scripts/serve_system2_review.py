#!/usr/bin/env python3
"""Serve a private, resumable human rating form for blind System2 packets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import tempfile

METRICS = {"correctness", "completeness", "unsupported_claims"}
EXPOSURES = {"aggregate_results_seen", "case_results_seen", "no_prior_results", "unsure"}
STATIC = Path(__file__).resolve().parents[2] / "csharp/SrDualBrain.Gateway/wwwroot"
ASSETS = {
    "/": ("benchmark-review.html", "text/html; charset=utf-8"),
    "/benchmark-review.js": ("benchmark-review.js", "text/javascript; charset=utf-8"),
    "/benchmark-review.css": ("benchmark-review.css", "text/css; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def complete(row):
    if row["preferred"] == "uncertain":
        return bool(row["rationale"].strip())
    return (row["preferred"] in {"a", "b", "tie"}
            and all(row[side][m] is not None for side in ("answer_a", "answer_b") for m in METRICS)
            and (row["preferred"] == "tie" or bool(row["rationale"].strip())))


class ReviewStore:
    def __init__(self, packets_path: Path, output: Path):
        raw = packets_path.read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        self.packets = json.loads(raw)
        if (set(self.packets) != {"schema", "rubric", "packets"}
                or self.packets["schema"] != "system2-blind-pairs-v1"):
            raise ValueError("Expected blind packets only; do not supply a report or reveal key")
        self.ids = []
        for packet in self.packets["packets"]:
            if set(packet) != {"packet_id", "question", "answer_a", "answer_b"}:
                raise ValueError("Blind packets must not contain mode or run metadata")
            if not all(isinstance(v, str) and v.strip() for v in packet.values()):
                raise ValueError("Packet fields must be nonempty strings")
            self.ids.append(packet["packet_id"])
        if not self.ids or len(self.ids) != len(set(self.ids)):
            raise ValueError("Packet IDs must be nonempty and unique")
        self.output = output
        output.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.draft = output / "draft.json"
        self.locked = output / "ratings.json"
        for path in (self.draft, self.locked):
            if path.exists() and json.loads(path.read_text())["input_sha256"] != self.digest:
                raise ValueError("Output directory belongs to a different packet file")

    def read(self):
        path = self.locked if self.locked.exists() else self.draft
        if path.exists():
            return json.loads(path.read_text())
        return {"reviewer": "", "exposure": "unsure", "ratings": [], "state": "draft"}

    def validate(self, data, *, final=False):
        if not isinstance(data, dict) or set(data) != {"reviewer", "exposure", "ratings"}:
            raise ValueError("Invalid rating document")
        if not isinstance(data["reviewer"], str) or not 1 <= len(data["reviewer"].strip()) <= 80:
            raise ValueError("Enter a reviewer name")
        if data["exposure"] not in EXPOSURES:
            raise ValueError("Select prior exposure")
        if not isinstance(data["ratings"], list):
            raise ValueError("Ratings must be a list")
        seen = set()
        for row in data["ratings"]:
            if not isinstance(row, dict) or set(row) != {"packet_id", "answer_a", "answer_b", "preferred", "rationale"}:
                raise ValueError("Invalid rating row")
            if row["packet_id"] not in self.ids or row["packet_id"] in seen:
                raise ValueError("Unknown or duplicate packet ID")
            seen.add(row["packet_id"])
            for side in ("answer_a", "answer_b"):
                scores = row[side]
                if not isinstance(scores, dict) or set(scores) != METRICS:
                    raise ValueError("Missing scoring dimensions")
                if any(v is not None and (type(v) is not int or v not in (0, 1, 2)) for v in scores.values()):
                    raise ValueError("Scores must be 0, 1, 2, or unfilled")
            if row["preferred"] not in {"", "a", "b", "tie", "uncertain"}:
                raise ValueError("Invalid preference")
            if not isinstance(row["rationale"], str) or len(row["rationale"]) > 4000:
                raise ValueError("Rationale must be text up to 4000 characters")
            if final and not complete(row):
                raise ValueError("Complete every rating or explain uncertainty before locking")
        if final and seen != set(self.ids):
            raise ValueError("Every packet must be rated before locking")

    def save(self, data, *, final=False):
        if self.locked.exists():
            raise FileExistsError("Ratings are already locked")
        self.validate(data, final=final)
        now = datetime.now(timezone.utc).isoformat()
        record = {"schema": "system2-human-blind-ratings-v1", "input_sha256": self.digest,
                  **data, "state": "locked" if final else "draft", "saved_at": now}
        encoded = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode()
        if final:
            fd = os.open(self.locked, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self.locked.chmod(0o400)
            receipt = {"locked_at": now, "ratings_sha256": hashlib.sha256(encoded).hexdigest(),
                       "input_sha256": self.digest, "count": len(data["ratings"])}
            receipt_path = self.output / "receipt.json"
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            receipt_path.chmod(0o600)
        else:
            fd, name = tempfile.mkstemp(dir=self.output, prefix=".draft-")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(encoded)
                os.replace(name, self.draft)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
        return record


def make_server(store, port):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, code, body, mime="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self):
            if not self.valid_host():
                return self.send(403, {"error": "Invalid host"})
            if self.path in ASSETS:
                filename, mime = ASSETS[self.path]
                return self.send(200, (STATIC / filename).read_bytes(), mime)
            if self.path == "/api/session":
                return self.send(200, {"packets": store.packets["packets"], "input_sha256": store.digest,
                                       "review": store.read(), "token": token})
            if self.path == "/api/export" and store.locked.exists():
                return self.send(200, store.locked.read_bytes())
            return self.send(404, {"error": "Not found"})

        def do_POST(self):
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if (not self.valid_host() or self.headers.get("Origin") != origin
                    or self.headers.get("X-Review-Token") != token):
                return self.send(403, {"error": "Invalid local session"})
            if self.path not in {"/api/draft", "/api/lock"}:
                return self.send(404, {"error": "Not found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2_000_000:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                review = store.save(data, final=self.path == "/api/lock")
            except FileExistsError as error:
                return self.send(409, {"error": str(error)})
            except (ValueError, KeyError, TypeError) as error:
                return self.send(400, {"error": str(error)})
            return self.send(200, {"state": review["state"], "saved_at": review["saved_at"]})

    return HTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8871)
    args = parser.parse_args()
    os.umask(0o077)
    store = ReviewStore(args.packets, args.output_dir)
    with make_server(store, args.port) as server:
        print(f"Human review: http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
