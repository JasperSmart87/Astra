"""Implements a simple presence and response decision engine for Astra.

This engine maintains an internal state (energy, cooldown timer) and uses
heuristics to decide whether Astra should speak in response to a given
transcript. It also updates its internal energy level based on how much
conversation is happening.

Wake detection:
- Exact match on wake name (word boundary)
- Fuzzy match per-word (RapidFuzz) to tolerate ASR glitches (asra/astera/etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time

try:
    from rapidfuzz.fuzz import ratio as _rf_ratio  # type: ignore
except Exception:  # pragma: no cover
    _rf_ratio = None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", (s or "").lower()).strip()


def fuzzy_wake(t_norm: str, wake_norm: str, threshold: int = 80) -> bool:
    """Return True if any word in t_norm is a fuzzy match for wake_norm."""
    if not _rf_ratio:
        return False
    for w in t_norm.split():
        if _rf_ratio(w, wake_norm) >= threshold:
            return True
    return False


@dataclass
class PresenceState:
    energy: float = 0.65  # Normalised energy level (0..1)
    last_spoke_ts: float = 0.0
    cooldown_s: float = 6.0  # Minimum seconds between spoken replies
    wake_name: str = "Astra"


@dataclass
class Decision:
    should_respond: bool
    reason: str
    confidence: float


class PresenceEngine:
    """Decides when Astra should respond based on recent activity and mood."""

    def __init__(self, wake_name: str = "Astra", cooldown_s: float = 6.0) -> None:
        self.state = PresenceState(wake_name=wake_name, cooldown_s=cooldown_s)
        self._last_tick = time.time()

        if _rf_ratio is None:
            print(
                "[PRESENCE] rapidfuzz not installed; wake-word fuzzy matching disabled."
                " Install requirements to improve wake detection."
            )

    def tick(self, transcript_text: str) -> None:
        """Update slow-changing internal state based on conversation activity."""
        now = time.time()
        dt = max(0.0, now - self._last_tick)
        self._last_tick = now

        quiet = 1.0 if not (transcript_text or "").strip() else 0.0
        talky = 1.0 if (transcript_text or "").strip() else 0.0

        self.state.energy += 0.03 * dt * quiet
        self.state.energy -= 0.0015 * dt * talky
        self.state.energy = max(0.0, min(1.0, self.state.energy))

    def decide(self, transcript_text: str) -> Decision:
        t = transcript_text.strip()
        now = time.time()

        # --- Filler-only guard (prevents interrupting "um", "erm", etc.) ---
        FILLER_ONLY = re.compile(r"^(um+|uh+|erm+|mm+)$", re.I)
        if FILLER_ONLY.match(t):
            return Decision(False, "filler", 0.0)

        # Enforce cooldown between spoken replies
        if now - self.state.last_spoke_ts < self.state.cooldown_s:
            return Decision(False, "cooldown", 0.0)

        # Determine if the user addressed Astra
        wake = self._contains_wake_name(t)
        question = self._looks_like_question(t)
        direct = self._looks_like_direct_address(t)

        confidence = 0.0
        if wake:
            confidence = 0.95
        elif direct:
            confidence = 0.7
        elif question:
            confidence = 0.55

        if confidence <= 0.0:
            return Decision(False, "not_addressed", 0.0)

        # Low-energy behavior: still answer if explicitly addressed by name.
        if self.state.energy < 0.12:
            if wake:
                return Decision(True, "low_energy_but_addressed", confidence)
            return Decision(False, "low_energy", confidence)

        return Decision(True, "addressed", confidence)

    def mark_spoke(self) -> None:
        self.state.last_spoke_ts = time.time()
        self.state.energy = max(0.0, self.state.energy - 0.03)

    def _contains_wake_name(self, text: str) -> bool:
        if not text:
            return False

        t_norm = _norm(text)
        wake_norm = _norm(self.state.wake_name)

        if re.search(rf"\b{re.escape(wake_norm)}\b", t_norm):
            return True

        return fuzzy_wake(t_norm, wake_norm)

    @staticmethod
    
    def _looks_like_question(text: str) -> bool:
        if not text:
            return False
        if text.endswith("?"):
            return True
        return bool(re.match(r"^(can you|could you|would you|what|why|how|when|where|who)\b", text.strip(), re.I))

    @staticmethod
    def _looks_like_direct_address(text: str) -> bool:
        if not text:
            return False
        return bool(
            re.match(
                r"^(hey|hello|hi|oi|excuse me|listen|you there|mate)\b",
                text.strip(),
                re.I,
            )
        )
    