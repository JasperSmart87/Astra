from __future__ import annotations

from typing import Any, Dict, List

from .memory_retrieve import retrieve
from .memory_store import MemoryItem


def build_messages(
    system_prompt: str,
    session_summary: str,
    memories: List[MemoryItem],
    user_text: str,
    recent_turns: List[Dict[str, Any]],
    max_memories: int = 8,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    if session_summary:
        messages.append({"role": "system", "content": f"SESSION SUMMARY:\n{session_summary}"})

    scored = retrieve(user_text, memories, k=max_memories)
    if scored:
        mem_lines = []
        for _, m in scored:
            mem_lines.append(f"- ({m.kind}, imp={m.importance:.2f}) {m.content}")
        messages.append({"role": "system", "content": "RELEVANT LONG-TERM MEMORY:\n" + "\n".join(mem_lines)})

    messages.extend(recent_turns[-6:])
    messages.append({"role": "user", "content": user_text})
    return messages
