from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .memory_store import MemoryItem, now_ts


@dataclass
class MemoryDecision:
    should_save: bool
    ask_user: bool
    proposed: Optional[MemoryItem]
    question: Optional[str]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def hybrid_decide(user_text: str) -> MemoryDecision:
    t = (user_text or "").strip()
    tl = t.lower()

    pref_markers = [
        "i prefer", "i like", "i don't like", "i do not like",
        "i want", "i’d like", "i would like", "please call me", "my name is",
    ]
    if any(p in tl for p in pref_markers):
        kind = "preference" if ("prefer" in tl or " like" in tl) else "fact"
        item = MemoryItem(
            ts=now_ts(),
            kind=kind,
            content=_norm(t),
            tags=["user_preference" if kind == "preference" else "user_fact"],
            importance=0.75,
            source="user",
            meta=None,
        )
        return MemoryDecision(True, False, item, None)

    project_markers = ["astra", "lm studio", "lmstudio", "orchestrator", "piper", "whisper", "camera", "microphone"]
    if any(p in tl for p in project_markers) and len(t) >= 25:
        item = MemoryItem(
            ts=now_ts(),
            kind="project",
            content=_norm(t),
            tags=["project_astra", "implementation"],
            importance=0.65,
            source="user",
            meta=None,
        )
        return MemoryDecision(True, False, item, None)

    if len(t) >= 30:
        q = "Should I remember that for next time? (yes/no)"
        item = MemoryItem(
            ts=now_ts(),
            kind="event",
            content=_norm(t),
            tags=["maybe_important"],
            importance=0.5,
            source="user",
            meta=None,
        )
        return MemoryDecision(False, True, item, q)

    return MemoryDecision(False, False, None, None)
