"""A simple fixed-size ring buffer for storing recent events."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple


class RingBuffer:
    """Fixed-size buffer that stores (category, text) tuples."""

    def __init__(self, max_events: int = 10) -> None:
        self.max_events = max_events
        self.events: Deque[Tuple[str, str]] = deque(maxlen=max_events)

    def add(self, category: str, text: str) -> None:
        self.events.append((category, text))

    def snapshot_text(self) -> str:
        """Return a joined string of all events currently in the buffer."""
        return "\n".join(f"{cat}: {text}" for (cat, text) in self.events)

    def clear(self) -> None:
        self.events.clear()