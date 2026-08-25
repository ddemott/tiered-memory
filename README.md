# tiered-memory

Short/medium/long-term memory for LLM agents. Stdlib-only Python package,
plus a drop-in plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).
No external service, no vector DB, no API keys of its own.

- **Short-term** — in-process buffer, never persisted, cleared each session.
- **Medium-term** — SQLite, one row per extracted fact, TTL-pruned.
- **Long-term** — markdown + frontmatter files, one per fact, human-readable and git-diffable.

Promotion (medium → long) fires on any of:
1. **Recurrence** — the same fact (hashed) shows up `promote_threshold` times across sessions.
2. **Importance** — a summarizer-scored fact (0–10) clears `importance_promote_threshold` and promotes on first mention (e.g. an allergy, an explicit constraint).
3. **Explicit save** — caller (or the agent, via a tool) asks directly.

Every long-term write runs a contradiction check against existing facts —
**ADD** / **UPDATE** / **SUPERSEDE** / **NOOP** — before it lands. A
superseded fact is never deleted: its frontmatter gets `status: superseded`
+ `superseded_by`, it drops out of active context, but the file stays on
disk as history.

`summarizer` and `classifier` are both plain injectable callables — the
core package makes no model calls itself. Bring your own LLM.

## Install

```bash
pip install git+https://github.com/ddemott/tiered-memory
# or, from a checkout:
pip install -e .
```

## Quick start

```python
from tiered_memory import TieredMemory

mem = TieredMemory(root="~/.myapp/memory", profile="dale")

mem.remember("user", "I'm allergic to peanuts")
mem.remember("assistant", "Noted.")

# call at session end -- summarizer turns the buffer into atomic,
# importance-scored facts; the promotion gate runs automatically
mem.end_session()

mem.context(token_budget=2000)   # long facts + recent summaries + live buffer, ready to inject
```

Without a `summarizer`/`classifier`, it degrades to a dumb fallback (raw
join, importance 0, always ADD) — safe to use with zero wiring, but you
want a real LLM call behind both for the promotion/contradiction logic to
mean anything. Shape:

```python
def summarizer(turns: list[tuple[str, str]]) -> list[dict]:
    # turns = [(role, text), ...] -- return atomic facts, scored 0-10
    return [{"fact": "...", "importance": 7}, ...]

def classifier(new_fact: str, existing_facts: list[dict]) -> dict:
    # existing_facts = [{"slug": ..., "body": ...}, ...]
    return {"action": "ADD" | "UPDATE" | "SUPERSEDE" | "NOOP", "target_slug": "..." }

mem = TieredMemory(root="...", summarizer=summarizer, classifier=classifier)
```

## Hermes Agent plugin

`hermes-plugin/tiered/` is a ready-to-drop-in [Hermes memory provider](https://github.com/NousResearch/hermes-agent) — it wires `summarizer`/`classifier` to Hermes's own host-owned LLM facade (`ctx.llm`), so it always calls through whichever model Hermes currently has active. No separate credentials, no hardcoded provider.

Install (no core Hermes files touched — this is the officially sanctioned third-party plugin path):

```bash
# 1. package needs to be importable from the Hermes venv
~/.hermes/bin/uv pip install --python ~/.hermes/hermes-agent/venv/bin/python3 -e /path/to/tiered-memory

# 2. drop the plugin into your profile's user-plugins dir
mkdir -p $HERMES_HOME/plugins
ln -s /path/to/tiered-memory/hermes-plugin/tiered $HERMES_HOME/plugins/tiered
# ($HERMES_HOME is ~/.hermes for the default profile, ~/.hermes/profiles/<name> for a named one)

# 3. activate it in that profile's config.yaml
```

```yaml
memory:
  provider: tiered
  tiered:
    promote_threshold: 3
    importance_promote_threshold: 8
    ttl_days: 30
    short_max_turns: 40
```

Only one external memory provider runs per profile — this replaces
whatever was active (e.g. Honcho) for that profile only.

## Config reference

| Key | Default | Meaning |
|---|---|---|
| `promote_threshold` | 3 | Recurring hits before auto-promotion |
| `importance_promote_threshold` | 8 | 0–10 score that promotes on first mention |
| `ttl_days` | 30 | Medium-term summary lifespan before pruning |
| `short_max_turns` | 40 | In-process short-term buffer size |

## Layout

```
tiered_memory/          # the standalone package
  store.py              # TieredMemory -- the public API
  db.py                 # SQLite layer, medium-term
  long_store.py          # markdown layer, long-term, supersession
  schema.sql
hermes-plugin/tiered/    # drop-in Hermes memory provider
  plugin.yaml
  __init__.py
```

## License

MIT — see `LICENSE`.
