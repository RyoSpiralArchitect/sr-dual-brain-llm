#!/usr/bin/env python3
"""Sleep/consolidation job that turns episodic traces into schema memories.

This script is intentionally dependency-light and deterministic by default.
It reads hippocampal episodes from Postgres, summarizes recent activity into a
"schema memory" row, and advances a per-session consolidation cursor.

Usage:
  export DUALBRAIN_PG_DSN="postgresql://user:pass@localhost:5432/dualbrain"
  python3 sr-dual-brain-llm/scripts/sleep_consolidate.py --session-id default
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.postgres_state_store import PostgresStateStore


_WORD_RE = re.compile(r"[\w']+")


def _keywords(text: str, *, min_len: int = 4, limit: int = 24) -> list[str]:
    tokens = [tok.lower() for tok in _WORD_RE.findall(text)]
    tokens = [tok for tok in tokens if len(tok) >= min_len]
    counts = Counter(tokens)
    return [tok for tok, _ in counts.most_common(limit)]


def _summarise_batch(episodes: list[dict[str, Any]]) -> tuple[str, list[str], dict[str, Any]]:
    if not episodes:
        return "", [], {}

    tags_counter: Counter[str] = Counter()
    lead_counter: Counter[str] = Counter()
    collab_strengths: list[float] = []
    question_keywords: Counter[str] = Counter()

    for ep in episodes:
        for tag in ep.get("tags") or []:
            if tag:
                tags_counter[str(tag)] += 1
        leading = ep.get("leading")
        if leading:
            lead_counter[str(leading)] += 1
        strength = ep.get("collaboration_strength")
        if strength is not None:
            try:
                collab_strengths.append(float(strength))
            except Exception:
                pass
        question_keywords.update(_keywords(str(ep.get("question") or "")))

    top_tags = [tag for tag, _ in tags_counter.most_common(12)]
    top_keywords = [kw for kw, _ in question_keywords.most_common(12) if kw not in top_tags]
    combined_tags = (top_tags + top_keywords)[:16]

    avg_collab = sum(collab_strengths) / len(collab_strengths) if collab_strengths else 0.0
    metrics = {
        "episodes": len(episodes),
        "lead_counts": dict(lead_counter),
        "avg_collaboration_strength": float(avg_collab),
        "tag_top": top_tags[:8],
        "keyword_top": top_keywords[:8],
    }

    recent_questions = [str(ep.get("question") or "")[:140] for ep in episodes[-5:]]
    summary_lines = [
        f"Consolidated {len(episodes)} episodes into schema memory.",
    ]
    if combined_tags:
        summary_lines.append("Themes: " + ", ".join(combined_tags[:12]))
    if lead_counter:
        summary_lines.append("Leading: " + ", ".join(f"{k}={v}" for k, v in lead_counter.items()))
    if collab_strengths:
        summary_lines.append(f"Avg collaboration strength: {avg_collab:.2f}")
    if recent_questions:
        summary_lines.append("Recent questions:")
        summary_lines.extend(f"- {q}" for q in recent_questions)
    return "\n".join(summary_lines), combined_tags, metrics


async def _run(args: argparse.Namespace) -> int:
    dsn = args.dsn or os.environ.get("DUALBRAIN_PG_DSN")
    if not dsn:
        print("Missing DSN. Set DUALBRAIN_PG_DSN or pass --dsn.", file=sys.stderr)
        return 2

    table_prefix = args.table_prefix or os.environ.get("DUALBRAIN_PG_TABLE_PREFIX", "srdb")
    store = PostgresStateStore(dsn, table_prefix=table_prefix)
    await store.ensure_schema()

    session_id = args.session_id
    max_batches = max(1, int(args.max_batches))
    batch_size = max(1, int(args.batch_size))

    total_inserted = 0
    for _ in range(max_batches):
        cursor = await store.get_consolidation_cursor(session_id)
        episodes = await store.fetch_episodes_after_id(session_id, after_id=cursor, limit=batch_size)
        if not episodes:
            break

        ts_from = float(episodes[0]["ts"])
        ts_to = float(episodes[-1]["ts"])
        episode_ids = [int(ep["id"]) for ep in episodes]
        qids = [str(ep["qid"]) for ep in episodes]

        summary, tags, metrics = _summarise_batch(episodes)
        if not summary:
            # Should never happen, but be defensive.
            await store.set_consolidation_cursor(session_id, episode_ids[-1])
            continue

        schema_id = await store.insert_schema_memory(
            session_id,
            ts_from=ts_from,
            ts_to=ts_to,
            episode_ids=episode_ids,
            qids=qids,
            tags=tags,
            summary=summary,
            metrics=metrics,
        )
        await store.set_consolidation_cursor(session_id, episode_ids[-1])

        total_inserted += 1
        print(
            f"[sleep_consolidate] session={session_id} schema_id={schema_id} episodes={len(episodes)} cursor={episode_ids[-1]} tags={len(tags)}"
        )

    if total_inserted == 0:
        print(f"[sleep_consolidate] session={session_id} no new episodes")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate episodic memory into schema memory.")
    parser.add_argument("--session-id", default="default")
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--table-prefix", default=None)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-batches", type=int, default=4)
    args = parser.parse_args()

    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()

