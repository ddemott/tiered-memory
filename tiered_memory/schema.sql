-- medium-term tier: rolling session summaries, auto-expiring
-- long-term tier lives as markdown files (long/*.md + INDEX.md), not in this DB
-- short-term tier is in-process only, never persisted here

PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS session_summaries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    profile       TEXT NOT NULL DEFAULT 'default',  -- hermes profile / engine instance name
    summary       TEXT NOT NULL,
    source_turns  INTEGER NOT NULL DEFAULT 0,        -- how many short-term turns this compresses
    importance    REAL NOT NULL DEFAULT 0.0,          -- 0-1, scoring input for promotion
    created_at    TEXT NOT NULL,                      -- ISO8601 UTC
    ttl_days      INTEGER NOT NULL DEFAULT 30,
    expires_at    TEXT NOT NULL,                      -- created_at + ttl_days, indexed for pruning
    promoted      INTEGER NOT NULL DEFAULT 0           -- 1 once folded into a long/*.md fact
);

CREATE INDEX IF NOT EXISTS idx_session_summaries_expires
    ON session_summaries (expires_at);

CREATE INDEX IF NOT EXISTS idx_session_summaries_profile_created
    ON session_summaries (profile, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_session_summaries_promoted
    ON session_summaries (promoted);

-- promotion candidates: dedup key groups recurring summaries across sessions
-- (hash/slug of the normalized fact text) so N-recurrence can trigger promotion
-- without embeddings or an LLM call just to check "have I seen this before"
CREATE TABLE IF NOT EXISTS promotion_candidates (
    dedup_key     TEXT PRIMARY KEY,
    profile       TEXT NOT NULL DEFAULT 'default',
    sample_text   TEXT NOT NULL,       -- most recent occurrence, for promotion write-up
    hit_count     INTEGER NOT NULL DEFAULT 1,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    promoted      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_promotion_candidates_hits
    ON promotion_candidates (profile, hit_count DESC);
