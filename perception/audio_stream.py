"""Audio capture helper using sounddevice.

This module defines a simple microphone sensor that records short chunks
of audio from the default or specified input device.  It returns a data
structure containing the PCM waveform and associated metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd


@dataclass
class AudioChunk:
    """Container for a chunk of recorded audio."""

    pcm: np.ndarray
    sample_rate: int
    duration: float


class MicrophoneSensor:
    """Records audio from the system microphone in fixed-length chunks."""

    def __init__(self, sample_rate: int = 44100, chunk_seconds: float = 4.0, device: Optional[str] = None) -> None:
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.device = device

    def start(self) -> None:
        """Prepare the microphone sensor (no-op for sounddevice)."""
        pass

    def read_chunk(self) -> AudioChunk:
        """Record a chunk of audio and return it as an ``AudioChunk``."""
        frames = int(self.sample_rate * self.chunk_seconds)
        recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32", device=self.device)
        sd.wait()
        recording = np.squeeze(recording)
        return AudioChunk(pcm=recording, sample_rate=self.sample_rate, duration=self.chunk_seconds)

    def stop(self) -> None:
        """Stop the microphone sensor (no-op for sounddevice)."""
        pass