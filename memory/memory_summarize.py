from __future__ import annotations

import os
from typing import Any, Dict, List

from lmstudio_client import LMStudioClient


def load_summary(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_summary(path: str, text: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write((text or "").strip() + "\n")


def update_summary(
    client: LMStudioClient,
    model: str,
    summary_path: str,
    recent_dialogue: List[Dict[str, Any]],
) -> str:
    old = load_summary(summary_path)

    sys = (
        "Update a running session summary for an offline companion agent.\n"
        "Write compact bullet points of what matters for continuity.\n"
        "Keep it under ~12 bullets. Include names, preferences, plans, and ongoing tasks.\n"
        "Do not include raw transcripts.\n"
    )

    messages: List[Dict[str, Any]] = [{"role": "system", "content": sys}]
    if old:
        messages.append({"role": "user", "content": f"Current summary:\n{old}"})

    messages.append({"role": "user", "content": "New dialogue to incorporate:"})
    messages.extend(recent_dialogue)

    new_sum = client.chat(model=model, messages=messages, temperature=0.2, max_tokens=350)
    save_summary(summary_path, new_sum)
    return new_sum
