from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class MemoryItem:
    ts: float
    kind: str  # "preference" | "fact" | "event" | "project" | "relationship"
    content: str
    tags: List[str]
    importance: float = 0.5  # 0..1
    source: str = "user"  # "user" | "assistant" | "system"
    meta: Optional[Dict[str, Any]] = None


def now_ts() -> float:
    return time.time()


class MemoryStore:
    def __init__(self, path_jsonl: str):
        self.path = path_jsonl
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)

    def append(self, item: MemoryItem) -> None:
        line = json.dumps(asdict(item), ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def iter_items(self, limit: int = 5000) -> Iterable[MemoryItem]:
        if not os.path.exists(self.path):
            return []
        items: List[MemoryItem] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    items.append(MemoryItem(**d))
                except Exception:
                    continue
        return items[-limit:]
