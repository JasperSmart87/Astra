from dataclasses import dataclass
import numpy as np
import time


@dataclass
class UtteranceResult:
    pcm: np.ndarray
    sample_rate: int
    seconds: float


class UtteranceCollector:
    def __init__(
        self,
        sample_rate: int,
        speech_rms: float = 0.012,
        end_silence_s: float = 0.70,
        min_speech_s: float = 0.35,
        max_utterance_s: float = 10.0,
        pre_roll_s: float = 0.25,   # keep a tiny lead-in before speech triggers
    ):
        self.sample_rate = sample_rate
        self.speech_rms = speech_rms
        self.end_silence_s = end_silence_s
        self.min_speech_s = min_speech_s
        self.max_utterance_s = max_utterance_s
        self.pre_roll_s = pre_roll_s
        self.reset()

    @staticmethod
    def rms(x: np.ndarray) -> float:
        if x is None:
            return 0.0
        x = np.asarray(x)
        if x.size == 0:
            return 0.0
        x = x.astype(np.float32, copy=False)
        return float(np.sqrt(np.mean(x * x)))

    def reset(self):
        self._buf: list[np.ndarray] = []
        self._pre: list[np.ndarray] = []
        self._speech_started = False
        self._last_voice_ts: float | None = None
        self._start_ts = time.time()

    def push(self, pcm: np.ndarray) -> bool:
        """Return True when utterance should finalize."""
        now = time.time()

        pcm = np.asarray(pcm)
        if pcm.size == 0:
            return False

        # Keep a small pre-roll buffer until speech starts
        if not self._speech_started:
            self._pre.append(pcm)
            # Trim pre-roll to last pre_roll_s seconds
            max_pre_samples = int(self.pre_roll_s * self.sample_rate)
            total = 0
            trimmed = []
            for chunk in reversed(self._pre):
                total += int(chunk.shape[0])
                trimmed.append(chunk)
                if total >= max_pre_samples:
                    break
            self._pre = list(reversed(trimmed))

        # Hard cap: if speech never started, don't "finalize" a fake utterance
        if (now - self._start_ts) > self.max_utterance_s:
            return self._speech_started

        r = self.rms(pcm)
        if r >= self.speech_rms:
            if not self._speech_started:
                # Start speech: move pre-roll into main buffer once
                self._buf.extend(self._pre)
                self._pre.clear()
                self._speech_started = True
            self._buf.append(pcm)
            self._last_voice_ts = now
            return False

        # If speech has started, keep buffering even during silence
        if self._speech_started:
            self._buf.append(pcm)
            if self._last_voice_ts is not None and (now - self._last_voice_ts) >= self.end_silence_s:
                return True

        return False

    def finalize(self) -> UtteranceResult | None:
        if not self._buf:
            self.reset()
            return None

        pcm = np.concatenate(self._buf)
        seconds = pcm.shape[0] / float(self.sample_rate)

        if (not self._speech_started) or (seconds < self.min_speech_s):
            self.reset()
            return None

        out = UtteranceResult(pcm=pcm, sample_rate=self.sample_rate, seconds=seconds)
        self.reset()
        return out
