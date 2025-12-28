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

import numpy as np
import requests
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
    """Clean text before sending to Piper."""

    if not text:
        return ""

    t = text

    # Remove common mojibake / encoding artifacts using safe escape sequences
    replacements = {
        "\u00e2\u20ac\u2122": "'",  # curly apostrophe mis-decoded as UTF-8 -> cp1252
        "\u00e2\u20ac\u0153": '"',  # open quote garble
        "\u00e2\u20ac\u009d": '"',  # close quote garble
        "\u00e2\u20ac\u00a6": "...",  # ellipsis garble
        "\u00e2\u20ac\u201c": "-",  # en dash garble
        "\u00e2\u20ac\u201d": "-",  # em dash garble
        "\u2026": "...",  # real ellipsis
    }
    for bad, good in replacements.items():
        t = t.replace(bad, good)

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
    def _voice_metadata_path(name: str) -> Optional[Path]:
        """Return the companion ``.onnx.json`` metadata path, if present."""

        candidates = []

        raw = Path(name)
        if raw.suffix == ".onnx":
            candidates.append(raw.with_suffix(".onnx.json"))
        candidates.append(raw)
        candidates.append(raw.with_suffix(".onnx.json"))

        cwd_raw = Path.cwd() / name
        if cwd_raw != raw:
            if cwd_raw.suffix == ".onnx":
                candidates.append(cwd_raw.with_suffix(".onnx.json"))
            candidates.append(cwd_raw)
            candidates.append(cwd_raw.with_suffix(".onnx.json"))

        for cand in candidates:
            if cand.is_file() and cand.suffix == ".json":
                return cand

        return None

    @staticmethod
    def _voice_download_url(name: str) -> Optional[str]:
        """Best-effort derivation of the HuggingFace Piper voice URL.

        Voice bundles follow the pattern ``<lang>_<variant>-<voice>-<size>``. For
        example, ``en_GB-alba-medium`` maps to the path
        ``en/en_GB/alba/medium/en_GB-alba-medium.onnx``.
        """

        match = re.match(r"^(?P<lang>[a-z]{2})_(?P<variant>[A-Za-z]+)-(?P<voice>[A-Za-z]+)-(?P<size>[a-z]+)$", name)
        if not match:
            return None

        parts = match.groupdict()
        return (
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            f"{parts['lang']}/{parts['lang']}_{parts['variant']}/{parts['voice']}/{parts['size']}/{name}.onnx"
        )

    @staticmethod
    def _metadata_to_voice_name(meta_path: Path) -> str:
        """Strip known suffixes from a metadata filename to recover the voice id."""

        if meta_path.name.endswith(".onnx.json"):
            return meta_path.name[: -len(".onnx.json")]
        return meta_path.stem

    @staticmethod
    def _download_voice_model(name: str, meta_path: Path) -> Optional[Path]:
        """Download a Piper voice next to its metadata if missing.

        This keeps TTS working out-of-the-box when only the ``.onnx.json``
        metadata is present in the repository.
        """

        url = PiperTTS._voice_download_url(name)
        if not url:
            return None

        dest = meta_path.with_suffix(".onnx")
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()

            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
            return dest
        except Exception:
            # Fail quietly; caller will surface a friendly status_reason.
            if dest.exists():
                dest.unlink(missing_ok=True)
            return None

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

        meta = PiperTTS._voice_metadata_path(name)
        if meta:
            voice_name = PiperTTS._metadata_to_voice_name(meta)
            downloaded = PiperTTS._download_voice_model(voice_name, meta)
            if downloaded and downloaded.is_file():
                return downloaded

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
                meta = self._voice_metadata_path(cfg.model_name)
                download_hint = " (download failed)" if meta else ""
                self.status_reason = (
                    f"model not found: {cfg.model_name} (.onnx file required){download_hint}"
                )
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

    def _iter_to_audio(self, audio_iter) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Convert a Piper audio iterator into float32 PCM data and sample rate."""

        chunks: list[np.ndarray] = []
        sample_rate: Optional[int] = None

        for chunk in audio_iter:
            if chunk is None:
                continue

            # Piper may emit custom chunk containers; prefer the raw PCM payload when present.
            pcm = getattr(chunk, "pcm", None)
            if pcm is not None:
                arr = np.asarray(pcm)
                chunk_rate = getattr(chunk, "sample_rate", None)
                if chunk_rate:
                    sample_rate = int(chunk_rate)
            elif isinstance(chunk, (bytes, bytearray)):
                arr = np.frombuffer(chunk, dtype=np.int16)
            else:
                arr = np.asarray(chunk)

                # Some Piper builds return opaque AudioChunk objects without a direct array
                # representation. If ``np.asarray`` yields an object array, try common payload
                # fields, and skip unconvertible chunks to avoid ``float(AudioChunk)`` errors.
                if arr.dtype == object:
                    payload = getattr(chunk, "audio", None) or getattr(chunk, "data", None)
                    if payload is not None:
                        arr = np.asarray(payload)
                    if arr.dtype == object:
                        try:
                            arr = arr.astype(np.float32)
                        except (TypeError, ValueError):
                            continue

            if arr.size == 0:
                continue
            if arr.ndim > 1:
                arr = arr[:, 0]
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.float32) / 32768.0
            else:
                arr = arr.astype(np.float32)
            arr = arr.reshape(-1)
            chunks.append(arr)

        if not chunks:
            return None, None

        data = np.concatenate(chunks)
        if sample_rate is None:
            sample_rate = getattr(self.voice, "sample_rate", None)
        return data, int(sample_rate) if sample_rate else None

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

        has_writer = hasattr(self.voice, "write_wav") or hasattr(self.voice, "write_wave")
        if has_writer:
            buf = io.BytesIO()
            self._write_wave(audio_iter, buf)
            buf.seek(0)
            data, rate = sf.read(buf, dtype="float32")
        else:
            data, rate = self._iter_to_audio(audio_iter)
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

        try:
            audio_data, sample_rate = self._render_audio(cleaned, {})
            if audio_data is None:
                return

            # ``sounddevice`` expects a numeric array; if a custom audio container sneaks
            # through, normalize it to float32 PCM before playback.
            pcm = getattr(audio_data, "pcm", None)
            if pcm is not None:
                sample_rate = sample_rate or getattr(audio_data, "sample_rate", None)
                audio_data = pcm

            audio_arr = np.asarray(audio_data)
            if audio_arr.size == 0:
                return
            if audio_arr.ndim > 1:
                audio_arr = audio_arr[:, 0]
            if np.issubdtype(audio_arr.dtype, np.integer):
                audio_arr = audio_arr.astype(np.float32) / 32768.0
            else:
                audio_arr = audio_arr.astype(np.float32)

            if sample_rate is None:
                return

            self._play_fn(audio_arr, int(sample_rate))
            params = set()
            try:
                params = set(inspect.signature(self.voice.synthesize).parameters)
            except (TypeError, ValueError):  # pragma: no cover - signature may be missing
                params = set()

            kwargs = {}
            if "speaker_id" in params and self.cfg.speaker is not None:
                kwargs["speaker_id"] = self.cfg.speaker
            if "noise_scale" in params:
                kwargs["noise_scale"] = self.cfg.noise_scale
            if "length_scale" in params:
                kwargs["length_scale"] = self.cfg.length_scale

            try:
                self.voice.synthesize(cleaned, **kwargs)
                return
            except TypeError:
                # Fallback to positional-only synthesize signatures that don't accept kwargs.
                pass

            self.voice.synthesize(cleaned)
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
