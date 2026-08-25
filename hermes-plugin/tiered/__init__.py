"""Tiered memory plugin — MemoryProvider interface.

Local short/medium/long-term memory via the `tiered_memory` package
(~/projects/tiered-memory). No external service, no network calls --
SQLite for medium-term (TTL-pruned session summaries) and markdown
files for long-term (promoted by recurrence or explicit save).

Promotion (medium -> long) fires on ANY of: recurrence >= promote_threshold,
importance >= importance_promote_threshold (scored by the summarizer itself,
so a critical fact promotes on first mention), or an explicit save. Every
long-term write runs a contradiction check (ADD/UPDATE/SUPERSEDE/NOOP)
against existing facts via the same host-model call -- a contradicted fact
is marked superseded, never deleted.

Config via config.yaml:
  memory:
    tiered:
      promote_threshold: 3              # recurring facts needed to auto-promote
      importance_promote_threshold: 8   # 0-10 score that promotes on first mention
      ttl_days: 30                       # medium-term summary lifespan
      short_max_turns: 40                # in-process short-term buffer size

Working directory: $HERMES_HOME/tiered-memory/ (profile-scoped, since
hermes_home is already per-profile -- see agent/memory_provider.py).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_MIN_QUERY_LEN = 6

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Pull a fenced code block out if present -- models sometimes wrap
    JSON in ```json fences despite being told not to."""
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _load_plugin_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config()
        memory_config = config.get("memory", {})
        if not isinstance(memory_config, dict):
            return {}
        provider_config = memory_config.get("tiered", {})
        return dict(provider_config) if isinstance(provider_config, dict) else {}
    except Exception:
        return {}


TIERED_PROMOTE_SCHEMA = {
    "name": "tiered_promote",
    "description": (
        "Save an important fact directly to long-term tiered memory, "
        "bypassing the recurrence threshold. Use for anything the user "
        "explicitly asks to remember, or a fact clearly worth keeping "
        "across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to remember."},
            "fact_type": {
                "type": "string",
                "enum": ["project", "user", "feedback", "reference"],
                "description": "Category of the fact (default: project).",
            },
        },
        "required": ["content"],
    },
}


class TieredMemoryProvider(MemoryProvider):
    """Local tiered (short/medium/long) memory, no external service."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config) if config is not None else _load_plugin_config()
        self._promote_threshold = int(self._config.get("promote_threshold", 3))
        self._importance_promote_threshold = int(self._config.get("importance_promote_threshold", 8))
        self._ttl_days = int(self._config.get("ttl_days", 30))
        self._short_max_turns = int(self._config.get("short_max_turns", 40))
        self._agent_context = "primary"
        self._mem = None
        self._last_recall_count = 0
        self._llm = None  # lazy PluginLlm facade -- see _get_llm()

    @property
    def name(self) -> str:
        return "tiered"

    def is_available(self) -> bool:
        return importlib.util.find_spec("tiered_memory") is not None

    def unavailable_reason(self) -> str:
        return (
            "tiered_memory package not installed. From ~/projects/tiered-memory: "
            "~/.hermes/bin/uv pip install --python venv/bin/python3 -e ."
        )

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "promote_threshold",
                "description": "Recurring summaries needed before auto-promotion to long-term",
                "type": "integer",
                "default": 3,
                "minimum": 1,
            },
            {
                "key": "importance_promote_threshold",
                "description": "0-10 importance score that promotes a fact to long-term on first mention",
                "type": "integer",
                "default": 8,
                "minimum": 0,
                "maximum": 10,
            },
            {
                "key": "ttl_days",
                "description": "Days a medium-term session summary lives before pruning",
                "type": "integer",
                "default": 30,
                "minimum": 1,
            },
            {
                "key": "short_max_turns",
                "description": "Turns kept in the in-process short-term buffer",
                "type": "integer",
                "default": 40,
                "minimum": 1,
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        from tiered_memory import TieredMemory

        hermes_home = kwargs.get("hermes_home") or str(Path.home() / ".hermes")
        self._agent_context = kwargs.get("agent_context", "primary")
        profile = kwargs.get("agent_identity") or "default"
        root = Path(hermes_home) / "tiered-memory"

        self._mem = TieredMemory(
            root=root,
            profile=profile,
            session_id=session_id,
            short_max_turns=self._short_max_turns,
            promote_threshold=self._promote_threshold,
            importance_promote_threshold=self._importance_promote_threshold,
            summarizer=self._summarize_via_host,
            classifier=self._classify_via_host,
        )
        self._mem.prune()

    def _get_llm(self):
        """Lazily build the host-owned LLM facade, same construction the
        memory-plugin loader itself uses for forwarding register_* calls
        (plugins/memory/__init__.py:_ProviderCollector._plugin_context).
        No provider/model override is requested, so every call routes
        through whatever model Hermes currently has active -- a later
        model switch just changes what this resolves to, nothing to
        rewire on our end."""
        if self._llm is None:
            from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager

            manifest = PluginManifest(name="tiered", key="tiered")
            self._llm = PluginContext(manifest, get_plugin_manager()).llm
        return self._llm

    def _summarize_via_host(self, turns: List[tuple]) -> List[Dict[str, Any]]:
        """Summarizer passed into TieredMemory: extract atomic facts (with
        an importance score each) from the short-term buffer using the
        host's active model. Falls back to one raw-join fact, importance 0,
        on any host-call or parse failure -- summarization must never
        crash session-end and lose the buffer."""
        transcript = "\n".join(f"{role}: {text}" for role, text in turns)
        fallback = [{"fact": transcript[:2000], "importance": 0}]
        try:
            llm = self._get_llm()
            result = llm.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "Extract distinct, atomic, factual statements worth "
                            "remembering long-term from this conversation excerpt. "
                            "Score each 0-10 for importance (10 = critical, e.g. "
                            "allergies, hard constraints, explicit 'remember this' "
                            "requests; 0 = trivial). Respond with ONLY a JSON array: "
                            '[{"fact": "...", "importance": 0}, ...]. Empty array if '
                            "nothing worth keeping. No prose, no markdown fences."
                        ),
                    },
                    {"role": "user", "content": transcript[:6000]},
                ],
                # generous headroom: reasoning-capable models spend part of
                # max_tokens on hidden reasoning before any visible text,
                # so a tight budget here truncates the response to nothing
                max_tokens=1024,
                purpose="tiered_memory_summarize",
            )
            parsed = json.loads(_strip_fences(result.text or ""))
            if not isinstance(parsed, list):
                return fallback
            facts = [
                {"fact": str(item.get("fact", "")).strip(), "importance": int(item.get("importance", 0))}
                for item in parsed
                if isinstance(item, dict) and str(item.get("fact", "")).strip()
            ]
            return facts or fallback
        except Exception:
            logger.debug("tiered memory: host summarize failed, falling back to raw join", exc_info=True)
            return fallback

    def _classify_via_host(self, new_fact: str, existing_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classifier passed into TieredMemory: decide how a fact about to
        be promoted relates to what's already in long-term memory. Falls
        back to ADD (dumb-write, pre-contradiction-check behavior) on any
        host-call or parse failure."""
        catalog = "\n".join(f"[{f['slug']}] {f['body'][:300]}" for f in existing_facts[:30])
        try:
            llm = self._get_llm()
            result = llm.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "Decide how a new fact relates to existing long-term "
                            "memory facts. Respond with ONLY a JSON object: "
                            '{"action": "ADD"|"UPDATE"|"SUPERSEDE"|"NOOP", '
                            '"target_slug": "<slug or null>"}. '
                            "ADD: genuinely new, unrelated to any existing fact. "
                            "UPDATE: refines/adds detail to an existing fact, same "
                            "underlying truth -- target_slug required. "
                            "SUPERSEDE: contradicts and replaces an existing fact "
                            "(e.g. changed address, changed preference) -- "
                            "target_slug required. "
                            "NOOP: exact duplicate of an existing fact, adds nothing."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Existing facts:\n{catalog}\n\nNew fact:\n{new_fact}",
                    },
                ],
                max_tokens=1024,
                purpose="tiered_memory_classify",
            )
            decision = json.loads(_strip_fences(result.text or ""))
            if not isinstance(decision, dict) or decision.get("action") not in {
                "ADD", "UPDATE", "SUPERSEDE", "NOOP",
            }:
                return {"action": "ADD"}
            return decision
        except Exception:
            logger.debug("tiered memory: host classify failed, falling back to ADD", exc_info=True)
            return {"action": "ADD"}

    def system_prompt_block(self) -> str:
        if not self._mem:
            return ""
        facts = self._mem.long_facts()
        if not facts:
            return ""
        return "# Tiered Memory (long-term)\n" + "\n\n".join(facts)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_recall_count = 0
        if not self._mem or is_trivial_prompt(query) or len(query.strip()) < _MIN_QUERY_LEN:
            return ""
        summaries = self._mem.recent_summaries(limit=5)
        if not summaries:
            return ""
        self._last_recall_count = len(summaries)
        return "## Tiered Memory (recent sessions)\n" + "\n\n".join(summaries)

    def recall_status(self) -> Optional[RecallStatus]:
        if self._last_recall_count <= 0:
            return None
        return RecallStatus(provider_label="tiered", count=self._last_recall_count, glyph="🗂️")

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not self._mem or self._agent_context != "primary":
            return
        if user_content:
            self._mem.remember("user", user_content)
        if assistant_content:
            self._mem.remember("assistant", assistant_content)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._mem or self._agent_context != "primary":
            return
        self._mem.end_session(ttl_days=self._ttl_days)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        if not self._mem:
            return
        # flush whatever the old session accumulated before rotating ids
        self._mem.end_session(ttl_days=self._ttl_days)
        self._mem.session_id = new_session_id

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._mem or action not in {"add", "replace"} or not content:
            return
        fact_type = "user" if target == "user" else "project"
        self._mem.promote(content, fact_type=fact_type)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [TIERED_PROMOTE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != "tiered_promote":
            return tool_error(f"Unknown tool: {tool_name}")
        if not self._mem:
            return tool_error("tiered memory not initialized")
        content = args.get("content", "")
        if not content:
            return tool_error("content is required")
        fact_type = args.get("fact_type", "project")
        path = self._mem.promote(content, fact_type=fact_type)
        if path is None:
            return json.dumps({"result": "Already known -- exact duplicate of an existing memory, nothing saved."})
        return json.dumps({"result": f"Saved to {path.name}"})

    def shutdown(self) -> None:
        if not self._mem or self._agent_context != "primary":
            return
        self._mem.end_session(ttl_days=self._ttl_days)
        self._mem.prune()


def register(ctx) -> None:
    """Register tiered memory as a memory provider plugin."""
    ctx.register_memory_provider(TieredMemoryProvider())
