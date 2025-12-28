from __future__ import annotations

from typing import Iterable, List, Tuple

from lmstudio_client import LMStudioClient

from .memory_store import MemoryItem, MemoryStore, now_ts


def _format_memory_lines(items: Iterable[MemoryItem]) -> str:
    lines: List[str] = []
    for idx, m in enumerate(items, 1):
        tags = ", ".join(m.tags or [])
        meta = f" source={m.source}" if m.source else ""
        lines.append(
            f"[{idx}] kind={m.kind} imp={m.importance:.2f} tags=[{tags}] {m.content}{meta}"
        )
    return "\n".join(lines)


def _extract_bullets(text: str, limit: int) -> List[str]:
    bullets: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = line.lstrip("-*•\t ").strip()
        if not line:
            continue
        bullets.append(line)
        if len(bullets) >= limit:
            break
    return bullets


def condense_memories(
    client: LMStudioClient,
    model: str,
    store: MemoryStore,
    *,
    keep_recent: int = 40,
    target_lessons: int = 10,
    max_items: int = 800,
    min_items: int = 120,
) -> Tuple[bool, int]:
    """Cluster older memories into concise lessons and rewrite the store.

    Returns (did_condense, lesson_count).
    """

    items = list(store.iter_items(limit=max_items))
    if len(items) < min_items:
        return False, 0

    items = sorted(items, key=lambda m: m.ts)
    recent: List[MemoryItem] = items[-keep_recent:] if keep_recent > 0 else []
    backlog: List[MemoryItem] = items[:-keep_recent] if keep_recent > 0 else items

    if not backlog:
        return False, 0

    sys = (
        "You condense long-term memories for an offline, real-time companion.\n"
        "Group similar items into themes and return concise lessons to carry forward.\n"
        "Make the lessons timeless and identity-aware (names, preferences, relationships).\n"
        "Keep the lessons small enough for live prompting. Prefer 6-12 bullets.\n"
        "Avoid raw transcripts; each bullet should be standalone and actionable.\n"
    )

    mem_lines = _format_memory_lines(backlog)
    messages = [
        {"role": "system", "content": sys},
        {
            "role": "user",
            "content": (
                "Condense the following memories.\n"
                f"Return at most {target_lessons} bullet lessons.\n"
                "Focus on shared context, preferences, identities, and project steps.\n"
                "Memories to condense:\n"
                f"{mem_lines}\n\n"
                "Only return the bullet lessons."
            ),
        },
    ]

    raw = client.chat(model=model, messages=messages, temperature=0.15, max_tokens=420)
    bullets = _extract_bullets(raw, target_lessons)
    if not bullets:
        return False, 0

    now = now_ts()
    lessons: List[MemoryItem] = [
        MemoryItem(
            ts=now,
            kind="lesson",
            content=b,
            tags=["lesson", "summary"],
            importance=0.82,
            source="system",
            meta={"condensed": True},
        )
        for b in bullets
    ]

    merged: List[MemoryItem] = recent + lessons
    store.rewrite(merged)
    return True, len(lessons)
