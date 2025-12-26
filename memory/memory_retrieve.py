from __future__ import annotations

import re
from typing import List, Tuple
from .memory_store import MemoryItem


def _tok(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return [t for t in s.split() if len(t) >= 3]


def score_memory(query: str, mem: MemoryItem) -> float:
    q = set(_tok(query))
    if not q:
        return 0.0
    m = set(_tok(mem.content + " " + " ".join(mem.tags)))
    overlap = len(q.intersection(m))
    if overlap == 0:
        return 0.0
    kind_bonus = 0.15 if mem.kind in ("preference", "fact") else 0.05
    return (overlap / max(3, len(q))) * (0.6 + 0.4 * mem.importance) + kind_bonus


def retrieve(query: str, memories: List[MemoryItem], k: int = 8) -> List[Tuple[float, MemoryItem]]:
    scored: List[Tuple[float, MemoryItem]] = []
    for mem in memories:
        s = score_memory(query, mem)
        if s > 0:
            scored.append((s, mem))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]
