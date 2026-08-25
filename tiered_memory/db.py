"""SQLite layer for the medium-term tier (session_summaries) and the
promotion-candidate dedup table that decides when a recurring summary
graduates to a long-term markdown fact."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_PATH.read_text())
    return conn


def dedup_key(text: str) -> str:
    """Normalize text (lowercase, collapse whitespace/punctuation) and hash it,
    so near-identical summaries across sessions land on the same promotion
    candidate without needing embeddings."""
    normalized = re.sub(r"[^\w\s]", "", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def insert_summary(
    conn: sqlite3.Connection,
    session_id: str,
    profile: str,
    summary: str,
    source_turns: int,
    importance: float = 0.0,
    ttl_days: int = 30,
) -> int:
    created_at = now_iso()
    expires_at = (
        datetime.now(timezone.utc) + timedelta(days=ttl_days)
    ).isoformat(timespec="seconds")
    cur = conn.execute(
        """INSERT INTO session_summaries
           (session_id, profile, summary, source_turns, importance,
            created_at, ttl_days, expires_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, profile, summary, source_turns, importance,
         created_at, ttl_days, expires_at),
    )
    conn.commit()
    return cur.lastrowid


def prune_expired(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        "DELETE FROM session_summaries WHERE expires_at < ? AND promoted = 0",
        (now_iso(),),
    )
    conn.commit()
    return cur.rowcount


def recent_summaries(
    conn: sqlite3.Connection, profile: str, limit: int = 10
) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM session_summaries
           WHERE profile = ? AND expires_at >= ?
           ORDER BY created_at DESC LIMIT ?""",
        (profile, now_iso(), limit),
    ).fetchall()


def upsert_promotion_candidate(
    conn: sqlite3.Connection, key: str, profile: str, sample_text: str
) -> int:
    row = conn.execute(
        "SELECT hit_count FROM promotion_candidates WHERE dedup_key = ?", (key,)
    ).fetchone()
    ts = now_iso()
    if row is None:
        conn.execute(
            """INSERT INTO promotion_candidates
               (dedup_key, profile, sample_text, hit_count, first_seen, last_seen)
               VALUES (?, ?, ?, 1, ?, ?)""",
            (key, profile, sample_text, ts, ts),
        )
        hit_count = 1
    else:
        hit_count = row["hit_count"] + 1
        conn.execute(
            """UPDATE promotion_candidates
               SET hit_count = ?, sample_text = ?, last_seen = ?
               WHERE dedup_key = ?""",
            (hit_count, sample_text, ts, key),
        )
    conn.commit()
    return hit_count


def mark_promoted(conn: sqlite3.Connection, key: str) -> None:
    conn.execute(
        "UPDATE promotion_candidates SET promoted = 1 WHERE dedup_key = ?", (key,)
    )
    conn.execute(
        """UPDATE session_summaries SET promoted = 1
           WHERE id IN (
               SELECT id FROM session_summaries
               WHERE summary IN (
                   SELECT sample_text FROM promotion_candidates WHERE dedup_key = ?
               )
           )""",
        (key,),
    )
    conn.commit()
