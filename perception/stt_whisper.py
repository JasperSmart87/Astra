"""Speech-to-text helpers using faster-whisper.

This module provides:
- WhisperSTT: one-shot (utterance) transcription.
- StreamingWhisperSTT: chunked "streaming" transcription for live readouts.

Notes:
- faster-whisper expects 16 kHz mono float32 audio.
- On CPU, smaller models + small beam size help responsiveness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel
from scipy.signal import resample_poly


# ---------------------------
# Data types
# ---------------------------

@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    language: Optional[str] = None
    segments: Optional[List[Segment]] = None


@dataclass
class StreamUpdate:
    committed: str  # stable text added since last update
    partial: str    # current unstable hypothesis
    segments: Optional[List[Segment]] = None  # debugging aid


# ---------------------------
# Utilities
# ---------------------------

def _to_mono_float32(pcm: np.ndarray) -> np.ndarray:
    x = np.asarray(pcm)
    if x.ndim == 2 and x.shape[1] >= 2:
        x = x.mean(axis=1)

    # int16 -> float32 [-1, 1]
    if x.dtype != np.float32 and x.dtype != np.float64:
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32, copy=False)

    return x


def _resample_to_16k(pcm_f32: np.ndarray, sample_rate: int) -> np.ndarray:
    target_sr = 16000
    if sample_rate == target_sr:
        return pcm_f32
    return resample_poly(pcm_f32, target_sr, sample_rate).astype(np.float32)


def _segments_to_list(segments) -> List[Segment]:
    out: List[Segment] = []
    for s in segments:
        out.append(Segment(start=float(getattr(s, "start", 0.0)), end=float(getattr(s, "end", 0.0)), text=str(getattr(s, "text", ""))))
    return out


# ---------------------------
# One-shot STT
# ---------------------------

class WhisperSTT:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(
        self,
        pcm: np.ndarray,
        sample_rate: int,
        *,
        language: Optional[str] = "en",
        vad_filter: bool = True,
        beam_size: int = 1,
        debug_segments: bool = False,
    ) -> Transcript:
        """Transcribe a full utterance.

        pcm: mono or stereo, int16 or float32
        sample_rate: source sample rate
        """
        x = _to_mono_float32(pcm)
        x = _resample_to_16k(x, sample_rate)

        segments, info = self.model.transcribe(
            x,
            language=language,
            vad_filter=vad_filter,
            beam_size=beam_size,
            condition_on_previous_text=False,
        )

        seg_list = _segments_to_list(segments)
        text = "".join(s.text for s in seg_list).strip()
        lang = getattr(info, "language", None)

        return Transcript(text=text, language=lang, segments=seg_list if debug_segments else None)

    def make_streamer(
        self,
        *,
        input_sr: int,
        update_every_s: float = 0.5,
        window_s: float = 6.0,
        lookback_s: float = 1.5,
        language: Optional[str] = "en",
    ) -> "StreamingWhisperSTT":
        return StreamingWhisperSTT(
            model=self.model,
            input_sr=input_sr,
            update_every_s=update_every_s,
            window_s=window_s,
            lookback_s=lookback_s,
            language=language,
        )


# ---------------------------
# Streaming STT
# ---------------------------

class StreamingWhisperSTT:
    """Chunked "streaming" ASR using faster-whisper.

    Strategy:
    - Keep a rolling 16k buffer of last window_s seconds.
    - Every update_every_s seconds (approx), re-run transcription on the window.
    - Commit segments that end before (window_s - lookback_s).
      The newest words are unstable; we keep them as partial.
    """

    def __init__(
        self,
        *,
        model: WhisperModel,
        input_sr: int,
        update_every_s: float,
        window_s: float,
        lookback_s: float,
        language: Optional[str],
    ):
        self.model = model
        self.input_sr = input_sr
        self.target_sr = 16000
        self.update_every_s = float(update_every_s)
        self.window_s = float(window_s)
        self.lookback_s = float(lookback_s)
        self.language = language

        self._buf = np.zeros((0,), dtype=np.float32)  # 16k mono
        self._last_update_at = 0  # samples
        self._committed_text = ""

    def reset(self) -> None:
        self._buf = np.zeros((0,), dtype=np.float32)
        self._last_update_at = 0
        self._committed_text = ""

    def push(
        self,
        pcm: np.ndarray,
        sample_rate: int,
        *,
        debug_segments: bool = False,
    ) -> Optional[StreamUpdate]:
        x = _to_mono_float32(pcm)
        x = _resample_to_16k(x, sample_rate)
        if x.size == 0:
            return None

        self._buf = np.concatenate([self._buf, x])

        # Trim to window
        max_samples = int(self.window_s * self.target_sr)
        if self._buf.shape[0] > max_samples:
            self._buf = self._buf[-max_samples:]

        # Update cadence
        needed = int(self.update_every_s * self.target_sr)
        if (self._buf.shape[0] - self._last_update_at) < needed:
            return None
        self._last_update_at = self._buf.shape[0]

        segments, _info = self.model.transcribe(
            self._buf,
            language=self.language,
            vad_filter=False,
            beam_size=1,
            condition_on_previous_text=False,
            initial_prompt=self._committed_text[-600:],
        )
        seg_list = _segments_to_list(segments)

        stable_cutoff = max(0.0, self.window_s - self.lookback_s)
        stable_text = "".join(s.text for s in seg_list if s.end <= stable_cutoff).strip()
        full_text = "".join(s.text for s in seg_list).strip()

        committed_delta = ""
        if stable_text:
            # Append only the new tail beyond what's already committed.
            base = self._committed_text

            # Find largest overlap where base endswith prefix(stable_text)
            overlap = 0
            max_overlap = min(len(base), len(stable_text))
            for k in range(max_overlap, 0, -1):
                if base.endswith(stable_text[:k]):
                    overlap = k
                    break

            to_add = stable_text[overlap:].strip()
            if to_add:
                sep = "" if (not base or base.endswith((" ", "\n"))) else " "
                self._committed_text = (base + sep + to_add).strip()
                committed_delta = to_add

        partial = full_text
        if self._committed_text and partial.startswith(self._committed_text):
            partial = partial[len(self._committed_text):].strip()

        return StreamUpdate(
            committed=committed_delta,
            partial=partial,
            segments=seg_list if debug_segments else None,
        )
