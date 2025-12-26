"""Piper TTS wrapper.

Adds a small text-cleaning pass to avoid weird characters being spoken
(eg, mojibake like "â" sequences), and to reduce stage-direction clutter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

try:
    from piper import PiperVoice  # type: ignore
except Exception:
    PiperVoice = None  # noqa: N816


@dataclass
class PiperConfig:
    model_name: str = "en_GB-alba-medium"
    speaker: Optional[int] = None
    noise_scale: float = 0.667
    length_scale: float = 1.0


def clean_tts_text(text: str) -> str:
    if not text:
        return ""

    t = text

    # Remove common mojibake / encoding artifacts
    t = t.replace("â€™", "'")
    t = t.replace("â€œ", '"').replace("â€", '"')
    t = t.replace("â€¦", "...")
    t = t.replace("â€“", "-").replace("â€”", "-")
    t = t.replace("â¦", "...")

    # Strip non-printable characters
    t = "".join(ch for ch in t if ch.isprintable() or ch in "\n\t")

    # Remove bracketed stage directions: ( ... ) and [ ... ]
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = re.sub(r"\([^\)]*\)", "", t)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Piper can stumble on lots of punctuation; keep it simple
    t = t.replace("*", "")

    return t


class PiperTTS:
    def __init__(self, cfg: PiperConfig):
        self.cfg = cfg
        self.enabled = PiperVoice is not None
        self.voice = None

        if self.enabled:
            try:
                self.voice = PiperVoice.load(cfg.model_name)
            except Exception:
                self.enabled = False

    def speak(self, text: str) -> None:
        if not self.enabled or self.voice is None:
            return

        cleaned = clean_tts_text(text)
        if not cleaned:
            return

        # Optional: debug print what we actually send to TTS
        # print("[TTS][CLEAN]", cleaned)

        try:
            self.voice.synthesize(
                cleaned,
                speaker_id=self.cfg.speaker,
                noise_scale=self.cfg.noise_scale,
                length_scale=self.cfg.length_scale,
            )
        except Exception as e:
            print("[TTS] error:", e)
