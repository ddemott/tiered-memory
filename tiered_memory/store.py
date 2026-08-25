"""TieredMemory: the public API. Short-term (in-process buffer), medium-term
(SQLite session summaries, TTL-pruned, per-fact), long-term (markdown facts,
promoted by recurrence, importance, or explicit save; contradiction-checked
against existing facts before writing). Zero external deps -- stdlib only,
so it drops into Hermes or any other Python engine identically.

Promotion gate (medium -> long), promote on ANY:
  1. recurrence   -- same fact (hashed) seen >= promote_threshold times
  2. importance   -- summarizer scored it >= importance_promote_threshold
  3. explicit ask -- caller calls promote() directly, bypasses both signals

Long-term writes go through a contradiction check (classifier) against
existing facts: ADD (new) / UPDATE (refine in place) / SUPERSEDE (contradicts
an existing fact -- old one marked superseded, not deleted) / NOOP (exact
duplicate, skip). No classifier wired in -> always ADD (dumb-write, same as
before this existed).
"""

from __future__ import annotations

import uuid
from collections import deque
from pathlib import Path
from typing import Callable, Optional

from . import db, long_store

# summarizer(turns) -> [{"fact": str, "importance": 0-10}, ...]
Summarizer = Callable[[list[tuple[str, str]]], list[dict]]

# classifier(new_fact, existing_facts=[{"slug","body"}, ...]) ->
#   {"action": "ADD"|"UPDATE"|"SUPERSEDE"|"NOOP", "target_slug": str|None}
Classifier = Callable[[str, list[dict]], dict]

_VALID_ACTIONS = {"ADD", "UPDATE", "SUPERSEDE", "NOOP"}


def _naive_summarizer(turns: list[tuple[str, str]]) -> list[dict]:
    """Fallback when no LLM summarizer is wired in: one fact, raw join,
    importance 0 (never trips the importance gate -- only recurrence or
    an explicit promote() can promote it). Good enough to unblock testing;
    replace with a real LLM call in prod."""
    lines = [f"{role}: {text}" for role, text in turns]
    return [{"fact": "\n".join(lines)[:2000], "importance": 0}]


class TieredMemory:
    def __init__(
        self,
        root: str | Path,
        profile: str = "default",
        session_id: Optional[str] = None,
        short_max_turns: int = 40,
        summarizer: Optional[Summarizer] = None,
        classifier: Optional[Classifier] = None,
        promote_threshold: int = 3,
        importance_promote_threshold: int = 8,
    ):
        self.root = Path(root).expanduser()
        self.profile = profile
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.summarizer = summarizer or _naive_summarizer
        self.classifier = classifier
        self.promote_threshold = promote_threshold
        self.importance_promote_threshold = importance_promote_threshold

        self._long_dir = self.root / "long"
        self._db_path = self.root / "medium" / "sessions.db"
        self._conn = db.connect(self._db_path)

        self._short: deque[tuple[str, str]] = deque(maxlen=short_max_turns)

    # ---- short-term ----

    def remember(self, role: str, text: str) -> None:
        self._short.append((role, text))

    def short_buffer(self) -> list[tuple[str, str]]:
        return list(self._short)

    # ---- medium-term ----

    def end_session(self, ttl_days: int = 30) -> list[int]:
        """Summarize the short-term buffer into one or more atomic
        medium-term facts, check each against the promotion gate, then
        clear the buffer. Returns the inserted summary row ids."""
        if not self._short:
            return []

        turns = list(self._short)
        facts = self.summarizer(turns)
        if isinstance(facts, str):  # tolerate an old-style string summarizer
            facts = [{"fact": facts, "importance": 0}]

        summary_ids: list[int] = []
        for item in facts:
            fact_text = (item.get("fact") or "").strip()
            if not fact_text:
                continue
            importance = int(item.get("importance") or 0)

            summary_id = db.insert_summary(
                self._conn, self.session_id, self.profile, fact_text,
                source_turns=len(turns), importance=importance, ttl_days=ttl_days,
            )
            summary_ids.append(summary_id)

            key = db.dedup_key(fact_text)
            hit_count = db.upsert_promotion_candidate(self._conn, key, self.profile, fact_text)

            if hit_count >= self.promote_threshold or importance >= self.importance_promote_threshold:
                self._promote_with_contradiction_check(fact_text, fact_type="derived", hit_count=hit_count)
                db.mark_promoted(self._conn, key)

        self._short.clear()
        return summary_ids

    def recent_summaries(self, limit: int = 10) -> list[str]:
        rows = db.recent_summaries(self._conn, self.profile, limit=limit)
        return [row["summary"] for row in rows]

    def prune(self) -> int:
        return db.prune_expired(self._conn)

    # ---- long-term ----

    def promote(self, text: str, fact_type: str = "project", name: Optional[str] = None) -> Optional[Path]:
        """Explicit save: bypasses the recurrence/importance gate. Still
        runs the contradiction check UNLESS a target `name` is given, in
        which case the caller is naming an exact file to overwrite and
        the classifier is skipped."""
        if name:
            path = long_store.write_fact(self._long_dir, text, fact_type=fact_type, name=name)
        else:
            path = self._promote_with_contradiction_check(text, fact_type=fact_type)
        db.mark_promoted(self._conn, db.dedup_key(text))
        return path

    def _promote_with_contradiction_check(
        self, text: str, fact_type: str = "project", hit_count: int = 1
    ) -> Optional[Path]:
        existing = long_store.list_active_facts(self._long_dir)
        decision = {"action": "ADD"}
        if self.classifier and existing:
            try:
                result = self.classifier(text, existing)
                if isinstance(result, dict) and result.get("action", "").upper() in _VALID_ACTIONS:
                    decision = {"action": result["action"].upper(), "target_slug": result.get("target_slug")}
            except Exception:
                pass  # classifier failure must not block the write -- fall back to ADD

        action = decision["action"]
        target_slug = decision.get("target_slug")

        if action == "NOOP":
            return None

        if action == "UPDATE" and target_slug:
            return long_store.write_fact(
                self._long_dir, text, fact_type=fact_type, hit_count=hit_count, name=target_slug
            )

        if action == "SUPERSEDE" and target_slug:
            new_path = long_store.write_fact(self._long_dir, text, fact_type=fact_type, hit_count=hit_count)
            long_store.mark_superseded(self._long_dir, target_slug, superseded_by=new_path.stem)
            return new_path

        return long_store.write_fact(self._long_dir, text, fact_type=fact_type, hit_count=hit_count)

    def long_facts(self) -> list[str]:
        return [f["body"] for f in long_store.list_active_facts(self._long_dir)]

    # ---- assembled context ----

    def context(self, token_budget: int = 2000) -> str:
        """Assemble long facts (always, highest priority) + recent medium
        summaries (recency order, fill remaining budget) + current short
        buffer (always appended in full -- it's the live turn).
        ~4 chars/token, no tokenizer dependency."""
        char_budget = token_budget * 4
        parts: list[str] = []
        used = 0

        facts = self.long_facts()
        if facts:
            block = "## Long-term memory\n" + "\n\n".join(facts)
            if len(block) > char_budget:
                block = block[:char_budget]  # long-term itself exceeds budget: hard cap
            parts.append(block)
            used += len(block)

        remaining = char_budget - used
        summaries = self.recent_summaries()
        if summaries and remaining > 0:
            kept: list[str] = []
            for s in summaries:  # already recency DESC; fill until budget runs out
                if remaining - len(s) <= 0:
                    break
                kept.append(s)
                remaining -= len(s)
            if kept:
                parts.append("## Recent sessions\n" + "\n\n".join(kept))

        if self._short:
            turns = "\n".join(f"{role}: {text}" for role, text in self._short)
            parts.append("## Current session\n" + turns)

        return "\n\n".join(parts)
