"""Long-term tier: durable facts as markdown+frontmatter files under long/,
indexed by INDEX.md. Same shape as Claude Code's own memory system, on
purpose -- human-readable, git-diffable, no DB needed to inspect it.

Supersession (Zep-style): a contradicted fact is never deleted -- its
frontmatter gets `status: superseded` + `superseded_by: <slug>` and it
drops out of the active set (list_active_facts / INDEX.md), but the file
stays on disk as history."""

from __future__ import annotations

import re
from pathlib import Path

from .db import now_iso

_FRONTMATTER = """---
name: {name}
description: "{description}"
metadata:
  type: {type}
  hit_count: {hit_count}
  created: {created}
  status: {status}
  superseded_by: {superseded_by}
  superseded_at: {superseded_at}
---

{body}
"""

_INDEX_HEADER = "# Long-term memory index\n\n"

_META_LINE_RE = re.compile(r"^\s*(\w[\w-]*):\s*(.*)$")


def slugify(text: str, max_words: int = 6) -> str:
    words = re.sub(r"[^\w\s-]", "", text.lower()).split()[:max_words]
    slug = "-".join(words) or "fact"
    return slug[:60]


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    """Parse our own generated frontmatter shape. Line-scan, not a YAML
    parser -- safe only because we control the format we write."""
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    meta_block, body = parts[1], parts[2].strip()
    meta: dict = {}
    for line in meta_block.splitlines():
        m = _META_LINE_RE.match(line)
        if m:
            meta[m.group(1)] = m.group(2).strip().strip('"')
    return meta, body


def write_fact(
    long_dir: Path,
    text: str,
    fact_type: str = "project",
    hit_count: int = 1,
    name: str | None = None,
) -> Path:
    long_dir.mkdir(parents=True, exist_ok=True)
    slug = name or slugify(text)
    path = long_dir / f"{slug}.md"

    if name is None:
        # avoid clobbering an existing distinct fact that happens to slugify
        # the same -- but a caller-provided `name` IS the target, overwrite it
        suffix = 1
        base_slug = slug
        while path.exists():
            existing = path.read_text()
            if text.strip() in existing:
                break  # same fact, just re-promoted -- overwrite in place
            suffix += 1
            slug = f"{base_slug}-{suffix}"
            path = long_dir / f"{slug}.md"

    description = text.strip().splitlines()[0][:120]
    path.write_text(
        _FRONTMATTER.format(
            name=slug,
            description=description.replace('"', "'"),
            type=fact_type,
            hit_count=hit_count,
            created=now_iso(),
            status="active",
            superseded_by="",
            superseded_at="",
            body=text.strip(),
        )
    )
    _update_index(long_dir, slug, description)
    return path


def mark_superseded(long_dir: Path, slug: str, superseded_by: str) -> None:
    path = long_dir / f"{slug}.md"
    if not path.exists():
        return
    meta, body = _split_frontmatter(path.read_text())
    meta["status"] = "superseded"
    meta["superseded_by"] = superseded_by
    meta["superseded_at"] = now_iso()
    path.write_text(
        _FRONTMATTER.format(
            name=meta.get("name", slug),
            description=meta.get("description", ""),
            type=meta.get("type", "project"),
            hit_count=meta.get("hit_count", 1),
            created=meta.get("created", now_iso()),
            status="superseded",
            superseded_by=superseded_by,
            superseded_at=meta["superseded_at"],
            body=body,
        )
    )
    _remove_from_index(long_dir, slug)


def _update_index(long_dir: Path, slug: str, description: str) -> None:
    index_path = long_dir.parent / "INDEX.md"
    line = f"- [{slug}](long/{slug}.md) — {description}\n"
    if not index_path.exists():
        index_path.write_text(_INDEX_HEADER + line)
        return
    existing = index_path.read_text()
    marker = f"({slug}.md)"
    lines = existing.splitlines(keepends=True)
    replaced = False
    for i, existing_line in enumerate(lines):
        if marker in existing_line:
            lines[i] = line
            replaced = True
            break
    if not replaced:
        lines.append(line)
    index_path.write_text("".join(lines))


def _remove_from_index(long_dir: Path, slug: str) -> None:
    index_path = long_dir.parent / "INDEX.md"
    if not index_path.exists():
        return
    marker = f"({slug}.md)"
    lines = index_path.read_text().splitlines(keepends=True)
    lines = [line for line in lines if marker not in line]
    index_path.write_text("".join(lines))


def list_active_facts(long_dir: Path) -> list[dict]:
    """Return [{"slug", "body", "description"}, ...] for facts not marked
    superseded. This is what feeds context assembly and the classifier's
    view of "what already exists"."""
    if not long_dir.exists():
        return []
    facts = []
    for path in sorted(long_dir.glob("*.md")):
        meta, body = _split_frontmatter(path.read_text())
        if meta.get("status") == "superseded":
            continue
        facts.append({"slug": path.stem, "body": body, "description": meta.get("description", "")})
    return facts


def read_all_facts(long_dir: Path) -> list[str]:
    """Back-compat helper: active fact bodies only."""
    return [f["body"] for f in list_active_facts(long_dir)]
