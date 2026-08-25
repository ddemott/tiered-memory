"""Stdlib-only smoke test -- no network, no LLM. Exercises the promotion
gate (recurrence + importance) and the contradiction-check paths (ADD/
UPDATE/SUPERSEDE/NOOP) with injected fake summarizer/classifier callables,
same shape a real host-LLM integration would provide."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from tiered_memory import TieredMemory


class TieredMemorySmokeTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="tiered-memory-test-"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_importance_gate_promotes_on_first_mention(self):
        summarizer = lambda turns: [{"fact": "Dale is allergic to peanuts", "importance": 9}]
        mem = TieredMemory(self.root, profile="t", summarizer=summarizer,
                            promote_threshold=3, importance_promote_threshold=8)
        mem.remember("user", "heads up, allergic to peanuts")
        mem.end_session()
        self.assertEqual(mem.long_facts(), ["Dale is allergic to peanuts"])

    def test_recurrence_gate_needs_repeats(self):
        summarizer = lambda turns: [{"fact": "Dale prefers dark roast coffee", "importance": 2}]
        for _ in range(2):
            mem = TieredMemory(self.root, profile="t", summarizer=summarizer,
                                promote_threshold=3, importance_promote_threshold=8)
            mem.remember("user", "turn")
            mem.end_session()
        self.assertEqual(mem.long_facts(), [])  # only 2 of 3 needed hits so far

        mem = TieredMemory(self.root, profile="t", summarizer=summarizer,
                            promote_threshold=3, importance_promote_threshold=8)
        mem.remember("user", "turn")
        mem.end_session()
        self.assertEqual(mem.long_facts(), ["Dale prefers dark roast coffee"])

    def test_supersede_replaces_without_deleting(self):
        add_classifier = lambda fact, existing: {"action": "ADD"}
        mem = TieredMemory(self.root, profile="t",
                            summarizer=lambda t: [{"fact": "Dale lives in Austin", "importance": 9}],
                            classifier=add_classifier, importance_promote_threshold=8)
        mem.remember("user", "i live in austin")
        mem.end_session()

        def supersede_classifier(fact, existing):
            for f in existing:
                if "lives in" in f["body"]:
                    return {"action": "SUPERSEDE", "target_slug": f["slug"]}
            return {"action": "ADD"}

        mem2 = TieredMemory(self.root, profile="t",
                             summarizer=lambda t: [{"fact": "Dale lives in Seattle", "importance": 9}],
                             classifier=supersede_classifier, importance_promote_threshold=8)
        mem2.remember("user", "i moved to seattle")
        mem2.end_session()

        active = mem2.long_facts()
        self.assertIn("Dale lives in Seattle", active)
        self.assertNotIn("Dale lives in Austin", active)

        superseded_files = [p for p in (self.root / "long").glob("*.md")
                             if "status: superseded" in p.read_text()]
        self.assertEqual(len(superseded_files), 1)  # old fact kept on disk, just inactive

    def test_noop_skips_duplicate_write(self):
        noop_classifier = lambda fact, existing: (
            {"action": "NOOP", "target_slug": existing[0]["slug"]} if existing else {"action": "ADD"}
        )
        mem = TieredMemory(self.root, profile="t",
                            summarizer=lambda t: [{"fact": "Dale likes tea", "importance": 9}],
                            classifier=noop_classifier, importance_promote_threshold=8)
        mem.remember("user", "i like tea")
        mem.end_session()
        self.assertEqual(len(mem.long_facts()), 1)

        mem2 = TieredMemory(self.root, profile="t",
                             summarizer=lambda t: [{"fact": "Dale likes tea", "importance": 9}],
                             classifier=noop_classifier, importance_promote_threshold=8)
        mem2.remember("user", "i like tea, again")
        mem2.end_session()
        self.assertEqual(len(mem2.long_facts()), 1)  # NOOP -- no duplicate written

    def test_context_prioritizes_long_over_medium_over_short(self):
        mem = TieredMemory(self.root, profile="t",
                            summarizer=lambda t: [{"fact": "durable fact", "importance": 9}],
                            importance_promote_threshold=8)
        mem.remember("user", "trigger")
        mem.end_session()
        mem.remember("user", "live turn text")

        ctx = mem.context(token_budget=2000)
        self.assertIn("## Long-term memory", ctx)
        self.assertIn("durable fact", ctx)
        self.assertIn("## Current session", ctx)
        self.assertIn("live turn text", ctx)

    def test_naive_fallback_with_no_summarizer_or_classifier(self):
        mem = TieredMemory(self.root, profile="t")  # no summarizer/classifier wired
        mem.remember("user", "hello")
        mem.remember("assistant", "hi")
        ids = mem.end_session()
        self.assertEqual(len(ids), 1)
        self.assertEqual(mem.recent_summaries(limit=1)[0].count("hello"), 1)


if __name__ == "__main__":
    unittest.main()
