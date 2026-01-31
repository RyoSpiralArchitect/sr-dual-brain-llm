# ============================================================================
#  SpiralReality Proprietary
#  Copyright (c) 2025 SpiralReality. All Rights Reserved.
#
#  NOTICE: This file contains confidential and proprietary information of
#  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
#  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
#  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
#
#  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
#  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
#  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
#  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ============================================================================

from __future__ import annotations

import asyncio
import json
import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .shared_memory import MemoryTrace
from .temporal_hippocampal_indexing import (
    EMBEDDING_VERSION,
    EpisodicTrace,
    TemporalHippocampalIndexing,
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class PostgresStateStore:
    """Best-effort Postgres persistence for SharedMemory + Hippocampus.

    This store is intentionally simple and dependency-light:
    - No pgvector requirement (vectors are stored as JSON arrays for now)
    - Schema is auto-created
    - Writes are idempotent for episodes (session_id + qid unique)
    """

    def __init__(self, dsn: str, *, table_prefix: str = "srdb") -> None:
        self.dsn = dsn
        self.table_prefix = table_prefix.strip().lower() or "srdb"
        self._has_pgvector: Optional[bool] = None
        self._pgvector_dim: int = 128

    # ---------------------------------------------------------------------
    # Public async API (wraps sync psycopg in asyncio.to_thread)
    async def ensure_schema(self) -> None:
        await asyncio.to_thread(self._ensure_schema_sync)

    async def load_session(
        self,
        session_id: str,
        *,
        memory_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
    ) -> Tuple[List[MemoryTrace], List[EpisodicTrace]]:
        return await asyncio.to_thread(
            self._load_session_sync,
            session_id,
            memory_limit,
            episode_limit,
        )

    async def append_memory_traces(
        self,
        session_id: str,
        traces: Sequence[MemoryTrace],
        *,
        qid: Optional[str] = None,
    ) -> None:
        if not traces:
            return
        await asyncio.to_thread(self._append_memory_traces_sync, session_id, traces, qid)

    async def upsert_episodes(self, session_id: str, episodes: Sequence[EpisodicTrace]) -> None:
        if not episodes:
            return
        await asyncio.to_thread(self._upsert_episodes_sync, session_id, episodes)

    async def reset_session(self, session_id: str) -> None:
        await asyncio.to_thread(self._reset_session_sync, session_id)

    # ---------------------------------------------------------------------
    # Sync implementation
    def _require_psycopg(self):
        try:
            import psycopg  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PostgresStateStore requires psycopg. Install: pip install -r requirements-pg.txt"
            ) from exc
        return psycopg

    def _connect(self):
        psycopg = self._require_psycopg()
        try:
            return psycopg.connect(self.dsn)
        except psycopg.OperationalError as exc:  # type: ignore[attr-defined]
            try:
                info = psycopg.conninfo.conninfo_to_dict(self.dsn)  # type: ignore[attr-defined]
            except Exception:
                info = {}

            host = info.get("host") or info.get("hostaddr")
            port = info.get("port")
            dbname = info.get("dbname")
            user = info.get("user")

            hint = ""
            if host in {"host", "postgres", "db"}:
                hint = (
                    "Hint: your DSN host looks like a Docker/compose service name. "
                    "If you're running this from your machine (not inside that Docker network), use "
                    "'localhost' (or 127.0.0.1) with the published port instead."
                )
            elif not host:
                hint = "Hint: DSN is missing 'host'."

            raise RuntimeError(
                f"Failed to connect to Postgres (host={host!r}, port={port!r}, dbname={dbname!r}, user={user!r}). {hint}"
            ) from exc

    def _table(self, name: str) -> str:
        # Table names are not user-generated SQL: table_prefix is controlled via env only.
        return f"{self.table_prefix}_{name}"

    def _ensure_schema_sync(self) -> None:
        sessions = self._table("sessions")
        memory = self._table("shared_memory")
        episodes = self._table("hippocampal_episodes")
        telemetry = self._table("telemetry_events")
        schema_memory = self._table("schema_memory")
        consolidation = self._table("consolidation_state")

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {sessions} (
                      session_id TEXT PRIMARY KEY,
                      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                      updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                      metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {memory} (
                      id BIGSERIAL PRIMARY KEY,
                      session_id TEXT NOT NULL REFERENCES {sessions}(session_id) ON DELETE CASCADE,
                      qid TEXT NULL,
                      question TEXT NOT NULL,
                      answer TEXT NOT NULL,
                      ts DOUBLE PRECISION NOT NULL,
                      tags JSONB NOT NULL,
                      question_tokens JSONB NOT NULL
                    );
                    """
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {memory}_session_ts ON {memory}(session_id, ts DESC);"
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {episodes} (
                      id BIGSERIAL PRIMARY KEY,
                      session_id TEXT NOT NULL REFERENCES {sessions}(session_id) ON DELETE CASCADE,
                      qid TEXT NOT NULL,
                      question TEXT NOT NULL,
                      answer TEXT NOT NULL,
                      ts DOUBLE PRECISION NOT NULL,
                      embedding_version TEXT NULL,
                      leading TEXT NULL,
                      collaboration_strength DOUBLE PRECISION NULL,
                      selection_reason TEXT NULL,
                      tags JSONB NOT NULL,
                      annotations JSONB NOT NULL,
                      vector JSONB NOT NULL
                    );
                    """
                )
                cur.execute(
                    f"""
                    ALTER TABLE {episodes}
                    ADD COLUMN IF NOT EXISTS embedding_version TEXT NULL;
                    """
                )
                cur.execute(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS {episodes}_session_qid ON {episodes}(session_id, qid);"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {episodes}_session_ts ON {episodes}(session_id, ts DESC);"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {episodes}_session_id_id ON {episodes}(session_id, id);"
                )
                self._ensure_pgvector(cur, episodes)
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {telemetry} (
                      id BIGSERIAL PRIMARY KEY,
                      session_id TEXT NOT NULL REFERENCES {sessions}(session_id) ON DELETE CASCADE,
                      qid TEXT NULL,
                      event TEXT NOT NULL,
                      ts DOUBLE PRECISION NOT NULL,
                      payload JSONB NOT NULL
                    );
                    """
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {telemetry}_session_ts ON {telemetry}(session_id, ts DESC);"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {telemetry}_session_qid ON {telemetry}(session_id, qid);"
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {schema_memory} (
                      id BIGSERIAL PRIMARY KEY,
                      session_id TEXT NOT NULL REFERENCES {sessions}(session_id) ON DELETE CASCADE,
                      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                      ts_from DOUBLE PRECISION NOT NULL,
                      ts_to DOUBLE PRECISION NOT NULL,
                      episode_ids JSONB NOT NULL,
                      qids JSONB NOT NULL,
                      tags JSONB NOT NULL,
                      summary TEXT NOT NULL,
                      metrics JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    );
                    """
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {schema_memory}_session_created ON {schema_memory}(session_id, created_at DESC);"
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {consolidation} (
                      session_id TEXT PRIMARY KEY REFERENCES {sessions}(session_id) ON DELETE CASCADE,
                      last_episode_id BIGINT NOT NULL DEFAULT 0,
                      updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )
            conn.commit()

    def _ensure_pgvector(self, cur, episodes_table: str) -> None:
        self._has_pgvector = False
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception:
            # Extension may not be installed or permissions may be missing.
            pass

        try:
            cur.execute("SELECT 1 FROM pg_type WHERE typname='vector' LIMIT 1;")
        except Exception:
            return
        if cur.fetchone() is None:
            return

        # If we got here, the vector type exists in this database.
        self._has_pgvector = True
        try:
            cur.execute(
                f"ALTER TABLE {episodes_table} ADD COLUMN IF NOT EXISTS vector_pg vector({int(self._pgvector_dim)}) NULL;"
            )
        except Exception:
            # Column creation can still fail if the type is not available in the current schema.
            self._has_pgvector = False
            return

        # Best-effort index. If it fails (old pgvector / permissions), we still proceed.
        try:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {episodes_table}_vector_pg_cosine ON {episodes_table} USING ivfflat (vector_pg vector_cosine_ops) WITH (lists = 100);"
            )
        except Exception:
            pass

    def _vector_literal(self, vector: Sequence[float]) -> str:
        return "[" + ",".join(f"{float(x):.10f}" for x in vector) + "]"

    def _normalise_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0.0:
            return vector
        return vector / norm

    def _touch_session(self, cur, session_id: str) -> None:
        sessions = self._table("sessions")
        cur.execute(
            f"""
            INSERT INTO {sessions}(session_id, updated_at)
            VALUES (%s, now())
            ON CONFLICT (session_id) DO UPDATE SET updated_at = EXCLUDED.updated_at;
            """,
            (session_id,),
        )

    def _load_session_sync(
        self, session_id: str, memory_limit: Optional[int], episode_limit: Optional[int]
    ) -> Tuple[List[MemoryTrace], List[EpisodicTrace]]:
        memory_table = self._table("shared_memory")
        episode_table = self._table("hippocampal_episodes")

        memory_traces: List[MemoryTrace] = []
        episodes: List[EpisodicTrace] = []
        embedder = TemporalHippocampalIndexing(dim=self._pgvector_dim)

        with self._connect() as conn:
            with conn.cursor() as cur:
                if memory_limit is None:
                    cur.execute(
                        f"""
                        SELECT
                          qid,
                          question,
                          answer,
                          ts,
                          tags::text,
                          question_tokens::text
                        FROM {memory_table}
                        WHERE session_id = %s
                        ORDER BY ts ASC;
                        """,
                        (session_id,),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT
                          qid,
                          question,
                          answer,
                          ts,
                          tags::text,
                          question_tokens::text
                        FROM (
                          SELECT
                            qid,
                            question,
                            answer,
                            ts,
                            tags,
                            question_tokens
                          FROM {memory_table}
                          WHERE session_id = %s
                          ORDER BY ts DESC
                          LIMIT %s
                        ) recent
                        ORDER BY ts ASC;
                        """,
                        (session_id, int(memory_limit)),
                    )
                for qid, question, answer, ts, tags_json, tokens_json in cur.fetchall():
                    tags = tuple(json.loads(tags_json or "[]"))
                    question_tokens = tuple(json.loads(tokens_json or "[]"))
                    memory_traces.append(
                        MemoryTrace(
                            qid=qid,
                            question=str(question),
                            answer=str(answer),
                            timestamp=float(ts),
                            tags=tags,
                            question_tokens=question_tokens,
                        )
                    )

                if episode_limit is None:
                    cur.execute(
                        f"""
                        SELECT
                          id,
                          qid,
                          question,
                          answer,
                          ts,
                          embedding_version,
                          leading,
                          collaboration_strength,
                          selection_reason,
                          tags::text,
                          annotations::text,
                          vector::text
                        FROM {episode_table}
                        WHERE session_id = %s
                        ORDER BY ts ASC;
                        """,
                        (session_id,),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT
                          id,
                          qid,
                          question,
                          answer,
                          ts,
                          embedding_version,
                          leading,
                          collaboration_strength,
                          selection_reason,
                          tags::text,
                          annotations::text,
                          vector::text
                        FROM (
                          SELECT
                            id,
                            qid,
                            question,
                            answer,
                            ts,
                            embedding_version,
                            leading,
                            collaboration_strength,
                            selection_reason,
                            tags,
                            annotations,
                            vector
                          FROM {episode_table}
                          WHERE session_id = %s
                          ORDER BY ts DESC
                          LIMIT %s
                        ) recent
                        ORDER BY ts ASC;
                        """,
                        (session_id, int(episode_limit)),
                    )
                pending_updates: List[Tuple[int, str, str]] = []
                for (
                    row_id,
                    qid,
                    question,
                    answer,
                    ts,
                    embedding_version,
                    leading,
                    collaboration_strength,
                    selection_reason,
                    tags_json,
                    annotations_json,
                    vector_json,
                ) in cur.fetchall():
                    tags = tuple(json.loads(tags_json or "[]"))
                    annotations = json.loads(annotations_json or "{}")
                    vector_list = json.loads(vector_json or "[]")
                    vector = np.asarray(vector_list, dtype=np.float32)
                    version = str(embedding_version) if embedding_version is not None else ""
                    if version != EMBEDDING_VERSION:
                        payload = f"Q: {question}\nA: {answer}"
                        vector = embedder.embed_text(payload)
                        vector = self._normalise_vector(vector.astype(np.float32))
                        pending_updates.append(
                            (
                                int(row_id),
                                EMBEDDING_VERSION,
                                _json_dumps(vector.astype(np.float32).tolist()),
                            )
                        )
                    episodes.append(
                        EpisodicTrace(
                            qid=str(qid),
                            question=str(question),
                            answer=str(answer),
                            vector=vector,
                            embedding_version=EMBEDDING_VERSION if version != EMBEDDING_VERSION else version,
                            timestamp=float(ts),
                            leading=str(leading) if leading is not None else None,
                            collaboration_strength=float(collaboration_strength)
                            if collaboration_strength is not None
                            else None,
                            selection_reason=str(selection_reason)
                            if selection_reason is not None
                            else None,
                            tags=tags,
                            annotations=dict(annotations) if isinstance(annotations, dict) else {},
                        )
                    )
                if pending_updates:
                    self._apply_embedding_updates(cur, episode_table, pending_updates)
                    conn.commit()

        return memory_traces, episodes

    def _apply_embedding_updates(
        self, cur, episode_table: str, updates: Sequence[Tuple[int, str, str]]
    ) -> None:
        if not updates:
            return
        for row_id, version, vector_json in updates:
            if self._has_pgvector:
                try:
                    vector_list = json.loads(vector_json or "[]")
                except Exception:
                    vector_list = []
                vec_literal = self._vector_literal(vector_list) if vector_list else None
                cur.execute(
                    f"""
                    UPDATE {episode_table}
                    SET embedding_version = %s,
                        vector = %s::jsonb,
                        vector_pg = CASE WHEN %s IS NULL THEN vector_pg ELSE %s::vector END
                    WHERE id = %s;
                    """,
                    (version, vector_json, vec_literal, vec_literal, int(row_id)),
                )
            else:
                cur.execute(
                    f"""
                    UPDATE {episode_table}
                    SET embedding_version = %s,
                        vector = %s::jsonb
                    WHERE id = %s;
                    """,
                    (version, vector_json, int(row_id)),
                )

    def _append_memory_traces_sync(
        self, session_id: str, traces: Sequence[MemoryTrace], qid: Optional[str]
    ) -> None:
        table = self._table("shared_memory")
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._touch_session(cur, session_id)
                for trace in traces:
                    trace_qid = trace.qid or qid
                    cur.execute(
                        f"""
                        INSERT INTO {table}
                          (session_id, qid, question, answer, ts, tags, question_tokens)
                        VALUES
                          (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb);
                        """,
                        (
                            session_id,
                            trace_qid,
                            trace.question,
                            trace.answer,
                            float(trace.timestamp),
                            _json_dumps(list(trace.tags)),
                            _json_dumps(list(trace.question_tokens)),
                        ),
                    )
            conn.commit()

    def _upsert_episodes_sync(self, session_id: str, episodes: Sequence[EpisodicTrace]) -> None:
        table = self._table("hippocampal_episodes")
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._touch_session(cur, session_id)
                for episode in episodes:
                    vector = (
                        episode.vector.astype(np.float32).tolist()
                        if isinstance(episode.vector, np.ndarray)
                        else list(episode.vector)
                    )
                    vector_json = _json_dumps(vector)
                    embedding_version = (
                        str(getattr(episode, "embedding_version", "") or "") or EMBEDDING_VERSION
                    )
                    if self._has_pgvector:
                        vec_literal = self._vector_literal(vector)
                        cur.execute(
                            f"""
                            INSERT INTO {table}
                              (
                                session_id,
                                qid,
                                question,
                                answer,
                                ts,
                                embedding_version,
                                leading,
                                collaboration_strength,
                                selection_reason,
                                tags,
                                annotations,
                                vector,
                                vector_pg
                              )
                            VALUES
                              (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::vector)
                            ON CONFLICT (session_id, qid) DO UPDATE SET
                              question = EXCLUDED.question,
                              answer = EXCLUDED.answer,
                              ts = EXCLUDED.ts,
                              embedding_version = EXCLUDED.embedding_version,
                              leading = EXCLUDED.leading,
                              collaboration_strength = EXCLUDED.collaboration_strength,
                              selection_reason = EXCLUDED.selection_reason,
                              tags = EXCLUDED.tags,
                              annotations = EXCLUDED.annotations,
                              vector = EXCLUDED.vector,
                              vector_pg = EXCLUDED.vector_pg;
                            """,
                            (
                                session_id,
                                episode.qid,
                                episode.question,
                                episode.answer,
                                float(episode.timestamp),
                                embedding_version,
                                episode.leading,
                                episode.collaboration_strength,
                                episode.selection_reason,
                                _json_dumps(list(episode.tags)),
                                _json_dumps(episode.annotations or {}),
                                vector_json,
                                vec_literal,
                            ),
                        )
                    else:
                        cur.execute(
                            f"""
                            INSERT INTO {table}
                              (
                                session_id,
                                qid,
                                question,
                                answer,
                                ts,
                                embedding_version,
                                leading,
                                collaboration_strength,
                                selection_reason,
                                tags,
                                annotations,
                                vector
                              )
                            VALUES
                              (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)
                            ON CONFLICT (session_id, qid) DO UPDATE SET
                              question = EXCLUDED.question,
                              answer = EXCLUDED.answer,
                              ts = EXCLUDED.ts,
                              embedding_version = EXCLUDED.embedding_version,
                              leading = EXCLUDED.leading,
                              collaboration_strength = EXCLUDED.collaboration_strength,
                              selection_reason = EXCLUDED.selection_reason,
                              tags = EXCLUDED.tags,
                              annotations = EXCLUDED.annotations,
                              vector = EXCLUDED.vector;
                            """,
                            (
                                session_id,
                                episode.qid,
                                episode.question,
                                episode.answer,
                                float(episode.timestamp),
                                embedding_version,
                                episode.leading,
                                episode.collaboration_strength,
                                episode.selection_reason,
                                _json_dumps(list(episode.tags)),
                                _json_dumps(episode.annotations or {}),
                                vector_json,
                            ),
                        )
            conn.commit()

    def _search_episodes_sync(
        self,
        session_id: str,
        query_vector: Sequence[float],
        *,
        limit: int,
        candidate_limit: int,
    ) -> List[Tuple[float, EpisodicTrace]]:
        episode_table = self._table("hippocampal_episodes")
        qv = np.asarray(list(query_vector), dtype=np.float32)
        qv = self._normalise_vector(qv)

        hits: List[Tuple[float, EpisodicTrace]] = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                if self._has_pgvector:
                    vec_literal = self._vector_literal(qv.tolist())
                    cur.execute(
                        f"""
                        SELECT
                          qid,
                          question,
                          answer,
                          ts,
                          embedding_version,
                          leading,
                          collaboration_strength,
                          selection_reason,
                          tags::text,
                          annotations::text,
                          vector::text,
                          (1 - (vector_pg <=> %s::vector)) AS similarity
                        FROM {episode_table}
                        WHERE session_id = %s AND vector_pg IS NOT NULL
                        ORDER BY vector_pg <=> %s::vector
                        LIMIT %s;
                        """,
                        (vec_literal, session_id, vec_literal, int(limit)),
                    )
                    rows = cur.fetchall()
                    for (
                        qid,
                        question,
                        answer,
                        ts,
                        embedding_version,
                        leading,
                        collaboration_strength,
                        selection_reason,
                        tags_json,
                        annotations_json,
                        vector_json,
                        similarity,
                    ) in rows:
                        tags = tuple(json.loads(tags_json or "[]"))
                        annotations = json.loads(annotations_json or "{}")
                        vector_list = json.loads(vector_json or "[]")
                        vector = np.asarray(vector_list, dtype=np.float32)
                        trace = EpisodicTrace(
                            qid=str(qid),
                            question=str(question),
                            answer=str(answer),
                            vector=vector,
                            embedding_version=str(embedding_version or EMBEDDING_VERSION),
                            timestamp=float(ts),
                            leading=str(leading) if leading is not None else None,
                            collaboration_strength=float(collaboration_strength)
                            if collaboration_strength is not None
                            else None,
                            selection_reason=str(selection_reason)
                            if selection_reason is not None
                            else None,
                            tags=tags,
                            annotations=dict(annotations) if isinstance(annotations, dict) else {},
                        )
                        hits.append((float(similarity), trace))

                if hits and len(hits) >= limit:
                    return hits[:limit]

                # Fallback: compute cosine similarity in Python using stored JSON vectors.
                cur.execute(
                    f"""
                    SELECT
                      qid,
                      question,
                      answer,
                      ts,
                      embedding_version,
                      leading,
                      collaboration_strength,
                      selection_reason,
                      tags::text,
                      annotations::text,
                      vector::text
                    FROM {episode_table}
                    WHERE session_id = %s
                    ORDER BY ts DESC
                    LIMIT %s;
                    """,
                    (session_id, int(candidate_limit)),
                )
                scored: List[Tuple[float, EpisodicTrace]] = []
                for (
                    qid,
                    question,
                    answer,
                    ts,
                    embedding_version,
                    leading,
                    collaboration_strength,
                    selection_reason,
                    tags_json,
                    annotations_json,
                    vector_json,
                ) in cur.fetchall():
                    tags = tuple(json.loads(tags_json or "[]"))
                    annotations = json.loads(annotations_json or "{}")
                    vector_list = json.loads(vector_json or "[]")
                    vector = np.asarray(vector_list, dtype=np.float32)
                    vector = self._normalise_vector(vector)
                    sim = float(np.dot(qv, vector)) if vector.size else 0.0
                    trace = EpisodicTrace(
                        qid=str(qid),
                        question=str(question),
                        answer=str(answer),
                        vector=vector,
                        embedding_version=str(embedding_version or EMBEDDING_VERSION),
                        timestamp=float(ts),
                        leading=str(leading) if leading is not None else None,
                        collaboration_strength=float(collaboration_strength)
                        if collaboration_strength is not None
                        else None,
                        selection_reason=str(selection_reason)
                        if selection_reason is not None
                        else None,
                        tags=tags,
                        annotations=dict(annotations) if isinstance(annotations, dict) else {},
                    )
                    scored.append((sim, trace))
                scored.sort(key=lambda item: item[0], reverse=True)
                return scored[:limit]

    async def search_episodes(
        self,
        session_id: str,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        candidate_limit: int = 500,
    ) -> List[Tuple[float, EpisodicTrace]]:
        return await asyncio.to_thread(
            self._search_episodes_sync,
            session_id,
            list(query_vector),
            limit=int(limit),
            candidate_limit=int(candidate_limit),
        )

    async def append_telemetry_events(
        self,
        session_id: str,
        events: Sequence[dict[str, Any]],
        *,
        qid: Optional[str] = None,
    ) -> None:
        if not events:
            return
        await asyncio.to_thread(self._append_telemetry_events_sync, session_id, list(events), qid)

    def _append_telemetry_events_sync(
        self, session_id: str, events: Sequence[dict[str, Any]], qid: Optional[str]
    ) -> None:
        table = self._table("telemetry_events")
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._touch_session(cur, session_id)
                for ev in events:
                    try:
                        event_name = str(ev.get("event") or "")
                    except Exception:
                        event_name = ""
                    if not event_name:
                        event_name = "event"
                    ts = float(ev.get("ts") or 0.0)
                    ev_qid = ev.get("qid") or qid
                    cur.execute(
                        f"""
                        INSERT INTO {table}
                          (session_id, qid, event, ts, payload)
                        VALUES
                          (%s, %s, %s, %s, %s::jsonb);
                        """,
                        (
                            session_id,
                            str(ev_qid) if ev_qid is not None else None,
                            event_name,
                            ts,
                            _json_dumps(ev),
                        ),
                    )
            conn.commit()

    async def query_telemetry_events(
        self,
        session_id: str,
        *,
        limit: int = 250,
        qid: Optional[str] = None,
        event: Optional[str] = None,
        since_ts: Optional[float] = None,
        until_ts: Optional[float] = None,
    ) -> List[dict[str, Any]]:
        return await asyncio.to_thread(
            self._query_telemetry_events_sync,
            session_id,
            int(limit),
            qid,
            event,
            since_ts,
            until_ts,
        )

    def _query_telemetry_events_sync(
        self,
        session_id: str,
        limit: int,
        qid: Optional[str],
        event: Optional[str],
        since_ts: Optional[float],
        until_ts: Optional[float],
    ) -> List[dict[str, Any]]:
        table = self._table("telemetry_events")
        clauses = ["session_id = %s"]
        params: List[Any] = [session_id]
        if qid:
            clauses.append("qid = %s")
            params.append(str(qid))
        if event:
            clauses.append("event = %s")
            params.append(str(event))
        if since_ts is not None:
            clauses.append("ts >= %s")
            params.append(float(since_ts))
        if until_ts is not None:
            clauses.append("ts <= %s")
            params.append(float(until_ts))

        where_sql = " AND ".join(clauses)
        params.append(int(limit))

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT payload::text
                    FROM {table}
                    WHERE {where_sql}
                    ORDER BY ts DESC
                    LIMIT %s;
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
        events: List[dict[str, Any]] = []
        for (payload_json,) in rows:
            try:
                payload = json.loads(payload_json or "{}")
            except Exception:
                payload = {"raw": str(payload_json)}
            if isinstance(payload, dict):
                events.append(payload)
            else:
                events.append({"value": payload})
        return events

    async def get_consolidation_cursor(self, session_id: str) -> int:
        return await asyncio.to_thread(self._get_consolidation_cursor_sync, session_id)

    def _get_consolidation_cursor_sync(self, session_id: str) -> int:
        table = self._table("consolidation_state")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT last_episode_id FROM {table} WHERE session_id = %s;",
                    (session_id,),
                )
                row = cur.fetchone()
        if not row:
            return 0
        try:
            return int(row[0] or 0)
        except Exception:
            return 0

    async def set_consolidation_cursor(self, session_id: str, last_episode_id: int) -> None:
        await asyncio.to_thread(
            self._set_consolidation_cursor_sync, session_id, int(last_episode_id)
        )

    def _set_consolidation_cursor_sync(self, session_id: str, last_episode_id: int) -> None:
        table = self._table("consolidation_state")
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._touch_session(cur, session_id)
                cur.execute(
                    f"""
                    INSERT INTO {table}(session_id, last_episode_id, updated_at)
                    VALUES (%s, %s, now())
                    ON CONFLICT (session_id) DO UPDATE SET
                      last_episode_id = EXCLUDED.last_episode_id,
                      updated_at = now();
                    """,
                    (session_id, int(last_episode_id)),
                )
            conn.commit()

    async def fetch_episodes_after_id(
        self, session_id: str, *, after_id: int, limit: int
    ) -> List[dict[str, Any]]:
        return await asyncio.to_thread(
            self._fetch_episodes_after_id_sync,
            session_id,
            int(after_id),
            int(limit),
        )

    def _fetch_episodes_after_id_sync(
        self, session_id: str, after_id: int, limit: int
    ) -> List[dict[str, Any]]:
        table = self._table("hippocampal_episodes")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                      id,
                      qid,
                      question,
                      answer,
                      ts,
                      leading,
                      collaboration_strength,
                      selection_reason,
                      tags::text,
                      annotations::text
                    FROM {table}
                    WHERE session_id = %s AND id > %s
                    ORDER BY id ASC
                    LIMIT %s;
                    """,
                    (session_id, int(after_id), int(limit)),
                )
                rows = cur.fetchall()
        results: List[dict[str, Any]] = []
        for (
            row_id,
            qid,
            question,
            answer,
            ts,
            leading,
            collaboration_strength,
            selection_reason,
            tags_json,
            annotations_json,
        ) in rows:
            try:
                tags = list(json.loads(tags_json or "[]"))
            except Exception:
                tags = []
            try:
                annotations = json.loads(annotations_json or "{}")
            except Exception:
                annotations = {}
            results.append(
                {
                    "id": int(row_id),
                    "qid": str(qid),
                    "question": str(question),
                    "answer": str(answer),
                    "ts": float(ts),
                    "leading": str(leading) if leading is not None else None,
                    "collaboration_strength": float(collaboration_strength)
                    if collaboration_strength is not None
                    else None,
                    "selection_reason": str(selection_reason)
                    if selection_reason is not None
                    else None,
                    "tags": tags,
                    "annotations": annotations if isinstance(annotations, dict) else {},
                }
            )
        return results

    async def insert_schema_memory(
        self,
        session_id: str,
        *,
        ts_from: float,
        ts_to: float,
        episode_ids: Sequence[int],
        qids: Sequence[str],
        tags: Sequence[str],
        summary: str,
        metrics: dict[str, Any],
    ) -> int:
        return await asyncio.to_thread(
            self._insert_schema_memory_sync,
            session_id,
            float(ts_from),
            float(ts_to),
            list(episode_ids),
            list(qids),
            list(tags),
            str(summary),
            dict(metrics),
        )

    def _insert_schema_memory_sync(
        self,
        session_id: str,
        ts_from: float,
        ts_to: float,
        episode_ids: Sequence[int],
        qids: Sequence[str],
        tags: Sequence[str],
        summary: str,
        metrics: dict[str, Any],
    ) -> int:
        table = self._table("schema_memory")
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._touch_session(cur, session_id)
                cur.execute(
                    f"""
                    INSERT INTO {table}
                      (session_id, ts_from, ts_to, episode_ids, qids, tags, summary, metrics)
                    VALUES
                      (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb)
                    RETURNING id;
                    """,
                    (
                        session_id,
                        float(ts_from),
                        float(ts_to),
                        _json_dumps([int(v) for v in episode_ids]),
                        _json_dumps([str(v) for v in qids]),
                        _json_dumps([str(v) for v in tags]),
                        summary,
                        _json_dumps(metrics),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return int(row[0]) if row else 0

    async def load_schema_memories(
        self,
        session_id: str,
        *,
        limit: int = 16,
    ) -> List[dict[str, Any]]:
        return await asyncio.to_thread(
            self._load_schema_memories_sync, session_id, int(limit)
        )

    def _load_schema_memories_sync(
        self, session_id: str, limit: int
    ) -> List[dict[str, Any]]:
        table = self._table("schema_memory")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                      id,
                      extract(epoch from created_at),
                      ts_from,
                      ts_to,
                      episode_ids::text,
                      qids::text,
                      tags::text,
                      summary,
                      metrics::text
                    FROM {table}
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (session_id, int(limit)),
                )
                rows = cur.fetchall()
        results: List[dict[str, Any]] = []
        for (
            row_id,
            created_at_epoch,
            ts_from,
            ts_to,
            episode_ids_json,
            qids_json,
            tags_json,
            summary,
            metrics_json,
        ) in rows:
            try:
                episode_ids = list(json.loads(episode_ids_json or "[]"))
            except Exception:
                episode_ids = []
            try:
                qids = list(json.loads(qids_json or "[]"))
            except Exception:
                qids = []
            try:
                tags = list(json.loads(tags_json or "[]"))
            except Exception:
                tags = []
            try:
                metrics = json.loads(metrics_json or "{}")
            except Exception:
                metrics = {}
            results.append(
                {
                    "id": int(row_id),
                    "created_at": float(created_at_epoch or 0.0),
                    "ts_from": float(ts_from),
                    "ts_to": float(ts_to),
                    "episode_ids": episode_ids,
                    "qids": qids,
                    "tags": tags,
                    "summary": str(summary),
                    "metrics": metrics if isinstance(metrics, dict) else {},
                }
            )
        return results

    def _reset_session_sync(self, session_id: str) -> None:
        sessions = self._table("sessions")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {sessions} WHERE session_id = %s;", (session_id,))
            conn.commit()
