"""Audio capture helper using sounddevice.

This module defines a simple microphone sensor that records short chunks
of audio from the default or specified input device.  It returns a data
structure containing the PCM waveform and associated metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import queue
import sounddevice as sd
import time


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
        self._stream: Optional[sd.InputStream] = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._frames_per_chunk = int(self.sample_rate * self.chunk_seconds)
        self._fallback_blocking = False

    def _enqueue(self, indata: np.ndarray, frames: int) -> None:
        """Add captured frames to the queue (best-effort)."""
        try:
            self._queue.put_nowait(np.copy(indata))
        except queue.Full:
            # Drop the oldest data if we're falling behind; realtime beats perfect recall.
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(np.copy(indata))
            except queue.Full:
                pass

    def start(self) -> None:
        """Start a low-latency input stream."""
        if self._stream:
            return

        blocksize = max(128, int(self._frames_per_chunk // 4) or 128)
        try:
            self._stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=blocksize,
                dtype="float32",
                device=self.device,
                callback=lambda indata, frames, _time, status: self._enqueue(indata, frames),
            )
            self._stream.start()
            self._fallback_blocking = False
        except Exception as e:  # pragma: no cover - hardware dependent
            print("[AUDIO] falling back to blocking microphone capture:", e)
            self._stream = None
            self._fallback_blocking = True

    def read_chunk(self) -> AudioChunk:
        """Record a chunk of audio and return it as an ``AudioChunk``."""
        if self._fallback_blocking:
            frames = int(self.sample_rate * self.chunk_seconds)
            recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32", device=self.device)
            sd.wait()
            recording = np.squeeze(recording)
            return AudioChunk(pcm=recording, sample_rate=self.sample_rate, duration=self.chunk_seconds)

        target_frames = self._frames_per_chunk
        collected: list[np.ndarray] = []
        total = 0
        start_ts = time.time()
        timeout = max(0.01, self.chunk_seconds / 2.0)

        while total < target_frames:
            try:
                block = self._queue.get(timeout=timeout)
            except queue.Empty:
                # If we have some audio already, break early to reduce latency
                if collected:
                    break
                continue

            block = np.asarray(block, dtype=np.float32)
            if block.ndim > 1:
                block = block[:, 0]
            collected.append(block)
            total += block.shape[0]

            # Don't over-accumulate: preserve responsiveness.
            if (time.time() - start_ts) > self.chunk_seconds:
                break

        if not collected:
            silence = np.zeros((target_frames,), dtype=np.float32)
            return AudioChunk(pcm=silence, sample_rate=self.sample_rate, duration=self.chunk_seconds)

        pcm = np.concatenate(collected)
        duration = pcm.shape[0] / float(self.sample_rate)
        return AudioChunk(pcm=pcm, sample_rate=self.sample_rate, duration=duration)

    def stop(self) -> None:
        """Stop the microphone sensor and release the stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        with self._queue.mutex:
            self._queue.queue.clear()
