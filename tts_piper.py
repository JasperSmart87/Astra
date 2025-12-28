"""Piper TTS wrapper.

Adds a small text-cleaning pass to avoid weird characters being spoken
(eg, mojibake like "â" sequences), and to reduce stage-direction clutter.
"""

from __future__ import annotations

import inspect
import io
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import sounddevice as sd
import soundfile as sf

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
    @staticmethod
    def _resolve_model_path(name: str) -> Optional[Path]:
        """Return a usable ONNX model path for Piper.

        Accepts either a bare basename ("en_GB-alba-medium"), a path without an
        extension, or a direct path to an ``.onnx`` file. If a matching ``.onnx``
        exists next to a provided ``.json`` file, it is preferred.
        """

        candidates = []

        raw = Path(name)
        if raw.suffix == ".json":
            candidates.append(raw.with_suffix(""))
        candidates.append(raw)
        candidates.append(raw.with_suffix(".onnx"))

        cwd_raw = Path.cwd() / name
        if cwd_raw != raw:
            if cwd_raw.suffix == ".json":
                candidates.append(cwd_raw.with_suffix(""))
            candidates.append(cwd_raw)
            candidates.append(cwd_raw.with_suffix(".onnx"))

        for cand in candidates:
            if cand.is_file() and cand.suffix == ".onnx":
                return cand

        return None

    def __init__(self, cfg: PiperConfig, play_fn: Optional[Callable[[object, int], None]] = None):
        self.cfg = cfg
        self.enabled = PiperVoice is not None
        self.status_reason = "piper package not installed" if not self.enabled else ""
        self.voice = None
        self._play_fn = play_fn or self._play_audio

        if self.enabled:
            model_path = self._resolve_model_path(cfg.model_name)
            if not model_path:
                self.enabled = False
                self.status_reason = f"model not found: {cfg.model_name} (.onnx file required)"
                return

            try:
                self.voice = PiperVoice.load(str(model_path))
                self.status_reason = "ready"
            except Exception as exc:  # pragma: no cover - runtime environment dependent
                self.enabled = False
                self.status_reason = f"failed to load {model_path}: {exc}"

    def _write_wave(self, audio_iter, buffer: io.BytesIO) -> None:
        writer = getattr(self.voice, "write_wav", None) or getattr(self.voice, "write_wave", None)
        if not writer:
            raise RuntimeError("PiperVoice missing write_wav/write_wave helpers")
        writer(buffer, audio_iter)

    def _render_audio(self, cleaned: str, kwargs: dict) -> Tuple[Optional[object], Optional[int]]:
        params = set()
        try:
            params = set(inspect.signature(self.voice.synthesize).parameters)
        except (TypeError, ValueError):  # pragma: no cover - signature may be missing
            params = set()

        synth_kwargs = {}
        if "speaker_id" in params and self.cfg.speaker is not None:
            synth_kwargs["speaker_id"] = self.cfg.speaker
        if "noise_scale" in params:
            synth_kwargs["noise_scale"] = self.cfg.noise_scale
        if "length_scale" in params:
            synth_kwargs["length_scale"] = self.cfg.length_scale

        synth_kwargs.update(kwargs)

        try:
            audio_iter = self.voice.synthesize(cleaned, **synth_kwargs)
        except TypeError:
            audio_iter = self.voice.synthesize(cleaned)

        buf = io.BytesIO()
        self._write_wave(audio_iter, buf)
        buf.seek(0)
        data, rate = sf.read(buf, dtype="float32")
        return data, rate

    @staticmethod
    def _play_audio(data, rate: int) -> None:
        sd.stop()
        sd.play(data, rate)
        sd.wait()

    def speak(self, text: str) -> None:
        if not self.enabled or self.voice is None:
            return

        cleaned = clean_tts_text(text)
        if not cleaned:
            return

        # Optional: debug print what we actually send to TTS
        # print("[TTS][CLEAN]", cleaned)

        try:
            audio_data, sample_rate = self._render_audio(cleaned, {})
            if audio_data is None or sample_rate is None:
                return
            self._play_fn(audio_data, sample_rate)
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
