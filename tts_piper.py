"""Piper TTS wrapper.

Adds a small text-cleaning pass to avoid weird characters being spoken
(eg, mojibake like "â" sequences), and to reduce stage-direction clutter.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

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

    # --- Streaming helpers -------------------------------------------------
    def _split_flush_segment(self, buf: str) -> Tuple[str, str]:
        """Split buffered text into a speakable chunk and a remainder.

        Prefer to flush at sentence boundaries so the speech sounds natural.
        If no obvious boundary exists, flush the whole buffer.
        """

        last_sentence_end = -1
        for delim in (". ", "? ", "! "):
            idx = buf.rfind(delim)
            last_sentence_end = max(last_sentence_end, idx)

        if last_sentence_end != -1:
            cut = last_sentence_end + 1
            return buf[:cut].strip(), buf[cut:]

        return buf.strip(), ""

    def stream_fragments(
        self,
        fragments: Iterable[str],
        flush_chars: int = 120,
        max_wait_s: float = 1.2,
        on_speak: Optional[Callable[[str], None]] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Speak fragments as they arrive while preserving flow.

        Accumulates fragments, flushing to TTS when enough content is present
        or when a small amount of time passes. Returns the concatenated text.
        """

        full_text = ""

        # If TTS is disabled, just join fragments without blocking.
        if not self.enabled or self.voice is None:
            for frag in fragments:
                if frag:
                    if on_chunk:
                        on_chunk(frag)
                    full_text += frag
            if on_complete:
                on_complete(full_text)
            return full_text

        buf = ""
        last_flush = time.time()
        if on_start:
            on_start()

        for frag in fragments:
            if not frag:
                continue

            full_text += frag
            if on_chunk:
                on_chunk(frag)
            buf += frag
            now = time.time()

            enough_chars = len(buf) >= flush_chars
            waited = (now - last_flush) >= max_wait_s
            has_sentence = any(d in buf for d in (". ", "? ", "! ", "\n"))
            impatient = waited and len(buf) >= max(12, flush_chars // 3)

            if enough_chars or (waited and has_sentence) or impatient:
                speakable, buf = self._split_flush_segment(buf)
                if speakable:
                    if on_speak:
                        on_speak(speakable)
                    self.speak(speakable)
                    last_flush = time.time()

        # Flush any trailing text
        if buf.strip():
            if on_speak:
                on_speak(buf)
            self.speak(buf)

        if on_complete:
            on_complete(full_text)

        return full_text
