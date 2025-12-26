"""Main entry point for the Astra LM Studio sensory bridge.

Windows + Python 3.10 + LM Studio + Piper + faster-whisper.

What this orchestrator tries to do:
- Keep audio responsive (small chunks)
- Provide streaming ASR commits for presence gating
- Keep vision optional and low-cost
- Inject to LM Studio on a fixed cadence (inject_every_seconds)
"""

from __future__ import annotations


class _SkipAudio(Exception):
    """Internal control-flow: skip processing a mic chunk (e.g., while TTS is speaking)."""
    pass

import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

from agent.prompts import SYSTEM_PROMPT
from config import SETTINGS
from lmstudio_client import LMStudioClient
from presence_engine import PresenceEngine

from perception.audio_stream import MicrophoneSensor
from perception.camera_stream import CameraSensor
from perception.utterance_collector import UtteranceCollector
from perception.stt_whisper import WhisperSTT

from memory.ring_buffer import RingBuffer
from memory.memory_store import MemoryStore, MemoryItem
from memory.memory_policy import hybrid_decide
from memory.memory_summarize import update_summary, load_summary
from memory.memory_pack import build_messages

from tts_piper import PiperTTS, PiperConfig
from vision_vlm import VLMCaptioner
from vision_identity import IdentityRecognizer

load_dotenv()


def env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None and v != "" else default


def safe_add_vision(buf: RingBuffer, obs: Any) -> Optional[Any]:
    """Add a vision observation to the buffer.

    Returns a usable frame (BGR) if present, else None.
    """
    frame_bgr = None

    if isinstance(obs, str):
        buf.add("vision", obs)
        return None

    if hasattr(obs, "to_text") and callable(getattr(obs, "to_text")):
        try:
            buf.add("vision", obs.to_text())
        except Exception:
            buf.add("vision", str(obs))
    elif isinstance(obs, dict):
        txt = obs.get("text") or obs.get("summary") or str(obs)
        buf.add("vision", txt)
    else:
        buf.add("vision", str(obs))

    if hasattr(obs, "frame_bgr"):
        frame_bgr = getattr(obs, "frame_bgr")
    elif hasattr(obs, "frame"):
        frame_bgr = getattr(obs, "frame")

    return frame_bgr


def make_wake_normalizer(wake_name: str):
    aliases = {
        wake_name.lower(),
        "astra",
        "aster",
        "asra",
        "astor",
        "astrea",
        "astro",
    }

    def normalize(text: str) -> str:
        if not text:
            return text
        t = text
        for a in sorted(aliases - {wake_name.lower()}, key=len, reverse=True):
            t = re.sub(rf"\b{re.escape(a)}\b", wake_name, t, flags=re.IGNORECASE)
        return t

    return normalize


def audio_levels(pcm: np.ndarray) -> tuple[float, float, bool]:
    """Return (rms, peak, clipped) for float32 [-1, 1] pcm."""
    if pcm.size == 0:
        return 0.0, 0.0, False
    rms = float(np.sqrt(np.mean(np.square(pcm))) + 1e-12)
    peak = float(np.max(np.abs(pcm)))
    clipped = bool(peak >= 0.999)
    return rms, peak, clipped


def loudest_channel(pcm: np.ndarray) -> np.ndarray:
    """Pick the loudest channel instead of averaging away signal energy."""
    if pcm.ndim != 2 or pcm.shape[1] <= 1:
        return pcm

    # Compute RMS per channel and keep the strongest one so VAD sees it.
    rms_per_ch = np.sqrt(np.mean(np.square(pcm), axis=0) + 1e-12)
    loudest_idx = int(np.argmax(rms_per_ch))
    return pcm[:, loudest_idx]


class ResponseSession:
    """Run LM -> TTS streaming in a background thread.

    This keeps the main loop free to continue sensing while speech plays
    out, improving the feeling of a live, full-duplex assistant.
    """

    def __init__(
        self,
        *,
        client: LMStudioClient,
        tts: PiperTTS,
        voice_model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        user_content: str,
        on_tts_chunk: Optional[Callable[[], None]] = None,
        on_stream_start: Optional[Callable[[], None]] = None,
        on_stream_done: Optional[Callable[[], None]] = None,
        on_fragment: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.client = client
        self.tts = tts
        self.voice_model = voice_model
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.on_tts_chunk = on_tts_chunk
        self.on_stream_start = on_stream_start
        self.on_stream_done = on_stream_done
        self.on_fragment = on_fragment
        self.user_content = user_content

        self.full_reply: str = ""
        self.error: Optional[Exception] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._done = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def is_active(self) -> bool:
        return not self._done.is_set()

    def wait(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        try:
            def _on_speak(_: str) -> None:
                if self.on_tts_chunk:
                    self.on_tts_chunk()

            def _on_start() -> None:
                if self.on_stream_start:
                    self.on_stream_start()

            def _on_chunk(chunk: str) -> None:
                if self.on_fragment:
                    self.on_fragment(chunk)

            def _on_done(_: str) -> None:
                if self.on_stream_done:
                    self.on_stream_done()

            def _chat_fragments():
                for frag in self.client.chat_stream(
                    self.voice_model,
                    self.messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ):
                    print(frag, end="", flush=True)
                    if self.on_fragment:
                        self.on_fragment(frag)
                    yield frag

            self.full_reply = self.tts.stream_fragments(
                _chat_fragments(),
                flush_chars=90,
                max_wait_s=0.4,
                on_speak=_on_speak,
                on_start=_on_start,
                on_chunk=_on_chunk,
                on_complete=_on_done,
            )
        except Exception as e:  # pragma: no cover - safety net for background worker
            self.error = e
        finally:
            self._done.set()


class SpeechBridge:
    """Coordinate mic gating with live TTS streaming."""

    def __init__(self, block_s: float) -> None:
        self.block_s = block_s
        self._block_until = 0.0
        self._lock = threading.Lock()

    def _extend(self) -> None:
        with self._lock:
            self._block_until = max(self._block_until, time.time() + self.block_s)

    def on_stream_start(self) -> None:
        self._extend()

    def pulse(self) -> None:
        self._extend()

    def on_stream_end(self) -> None:
        # Leave a small tail to let the room settle before listening again
        self._extend()

    def should_skip_audio(self) -> bool:
        with self._lock:
            return time.time() < self._block_until


def main() -> None:
    base_url = SETTINGS.lmstudio_base_url
    api_key = SETTINGS.lmstudio_api_key
    voice_model = env("LMSTUDIO_MODEL", SETTINGS.lmstudio_model)
    vlm_model = env("VLM_MODEL", "").strip()
    wake_name = env("WAKE_NAME", SETTINGS.wake_name)

    normalize_wake = make_wake_normalizer(wake_name)

    inject_every = SETTINGS.inject_every_seconds
    max_events = SETTINGS.max_context_events

    # Debug knobs (safe to leave off)
    DEBUG_WHISPER_SEGS = env("DEBUG_WHISPER_SEGS", "0") in ("1", "true", "TRUE")
    DEBUG_PARTIALS = env("DEBUG_PARTIALS", "0") in ("1", "true", "TRUE")

    client = LMStudioClient(base_url, api_key)
    vlm = VLMCaptioner(client, vlm_model) if vlm_model else VLMCaptioner(client, "")
    presence = PresenceEngine(wake_name=wake_name, cooldown_s=6.0)

    tts = PiperTTS(PiperConfig(model_name="en_GB-alba-medium"))
    if tts.enabled:
        print("Piper TTS enabled (en_GB-alba-medium)")
    else:
        print("Piper TTS disabled (piper not installed)")

    buf = RingBuffer(max_events=max_events)

    # --- Memory system ---
    mem_store = MemoryStore("memory/memory_store.jsonl")
    summary_path = "memory/session_summary.txt"
    recent_dialogue: List[Dict[str, Any]] = []
    pending_memory: Optional[MemoryItem] = None
    inject_count = 0

    # Sensors
    cam = CameraSensor(SETTINGS.camera_index, SETTINGS.camera_fps, SETTINGS.camera_downscale_width)
    mic = MicrophoneSensor(SETTINGS.audio_sample_rate, SETTINGS.audio_chunk_seconds, SETTINGS.audio_device)

    # ASR
    stt = WhisperSTT(model_size="small", device="cpu", compute_type="int8")
    streamer = stt.make_streamer(
        input_sr=SETTINGS.audio_sample_rate,
        update_every_s=0.5,
        window_s=6.0,
        lookback_s=1.5,
        language="en",
    )
    last_partial_print = 0.0

    # Presence helper: hold most recent heard speech briefly
    last_heard_text = ""
    last_heard_ts = 0.0
    HEARD_HOLD_S = 2.5

    # Identity
    identity = IdentityRecognizer(
        known_faces_dir=r"C:\astra_main\faces",
        cooldown_s=10.0,
        debug=False,
    )
    current_identities: set[str] = set()
    last_identity_ts = 0.0
    last_known_identities: set[str] = set()

    # Utterance collector
    collector = UtteranceCollector(
        sample_rate=SETTINGS.audio_sample_rate,
        speech_rms=0.006,
        min_speech_s=0.15,
        end_silence_s=1.2,
        max_utterance_s=15.0,
        pre_roll_s=0.25,
    )

    live_thought_accum = ""

    def on_generation_fragment(frag: str) -> None:
        nonlocal live_thought_accum
        if not frag:
            return
        live_thought_accum += frag
        # Keep a small rolling window of what was just said for sensory context
        if len(live_thought_accum) > 320:
            live_thought_accum = live_thought_accum[-320:]
        buf.add("audio", f"[voice-stream] {frag.strip()}")
        speech_bridge.pulse()

    cam.start()
    mic.start()

    # Block microphone processing for a short time after TTS plays, to avoid the mic
    # re-transcribing Astra's own voice (feedback loop).
    speech_bridge = SpeechBridge(block_s=float(env("TTS_BLOCK_S", "1.2")))

    # Live response session (LM -> TTS) runs in background to keep sensing snappy.
    response_session: Optional[ResponseSession] = None

    last_inject = 0.0
    last_frame_bgr = None

    print("Astra Orchestrator running (Ctrl+C to stop)")
    print(f"LM Studio base URL: {base_url}")
    print(f"Voice model:         {voice_model}")
    print(f"Vision model:        {vlm_model if vlm_model else '(disabled)'}")
    print(f"Wake name:           {wake_name}")

    try:
        while True:
            # ---------------------------
            # Vision
            # ---------------------------
            if response_session and response_session.is_active():
                speech_bridge.pulse()

            try:
                obs = cam.read()
                frame_bgr = safe_add_vision(buf, obs)
                if frame_bgr is not None:
                    _ = identity.process(frame_bgr)
                    current_identities = set(identity.present)
                    if current_identities:
                        last_known_identities = set(current_identities)
                        last_identity_ts = time.time()
                    last_frame_bgr = frame_bgr
            except Exception as e:
                buf.add("vision", f"[vision] error: {e}")

            # ---------------------------
            # Audio
            # ---------------------------
            transcript = ""  # final utterance transcript

            try:
                chunk = mic.read_chunk()
                if speech_bridge.should_skip_audio():
                    # Avoid transcribing our own TTS (headphones still recommended)
                    raise _SkipAudio()
                pcm = np.asarray(chunk.pcm)

                # force float32 [-1,1]
                if pcm.dtype != np.float32 and pcm.dtype != np.float64:
                    pcm = pcm.astype(np.float32) / 32768.0
                else:
                    pcm = pcm.astype(np.float32, copy=False)

                # keep the loudest input channel to avoid losing level when averaging
                pcm = loudest_channel(pcm)

                rms, peak, clipped = audio_levels(pcm)
                print(f"[AUDIO][LEVEL] rms={rms:.6f} peak={peak:.3f} clip={clipped}")

                thr = getattr(collector, "speech_rms", 0.0)
                print(f"[AUDIO][VAD] rms={rms:.6f} thr={thr:.6f} above={rms >= thr}")

                # Streaming ASR update
                upd = streamer.push(pcm, chunk.sample_rate)
                if upd is not None:
                    if DEBUG_WHISPER_SEGS and upd.segments:
                        print("\n[WHISPER][STREAM][SEGS]")
                        for (a, b, t) in upd.segments:
                            print(f"  {a:5.2f}s → {b:5.2f}s | {t!r}")
                        print("[WHISPER][STREAM][END]\n")

                    if DEBUG_PARTIALS and upd.partial:
                        nowp = time.time()
                        if (nowp - last_partial_print) >= 0.5:
                            print(f"[WHISPER][STREAM][PARTIAL] {upd.partial!r}")
                            last_partial_print = nowp

                    if upd.committed:
                        committed = normalize_wake(upd.committed.strip())
                        if committed:
                            print(f"[WHISPER][STREAM][COMMIT] {committed!r}")
                            buf.add("audio", f"[audio-live] {committed}")
                            last_heard_text = committed
                            last_heard_ts = time.time()

                # Utterance collector (final transcript)
                ready = collector.push(pcm)
                if ready:
                    print("[AUDIO][VAD] utterance ready -> finalizing")
                    utt = collector.finalize()
                    streamer.reset()  # treat boundary as fresh context

                    if utt is not None and utt.pcm is not None:
                        print(f"[AUDIO][DBG] utt seconds={utt.seconds:.2f} samples={int(np.asarray(utt.pcm).shape[0])}")

                        tr = stt.transcribe(utt.pcm, utt.sample_rate)
                        transcript = normalize_wake((tr.text or "").strip())

                        print(f"[WHISPER][FINAL] {transcript!r}")
                        if DEBUG_WHISPER_SEGS and tr.segments:
                            print("[WHISPER][FINAL][SEGS]")
                            for (a, b, t) in tr.segments:
                                print(f"  {a:5.2f}s → {b:5.2f}s | {t!r}")
                            print("[WHISPER][FINAL][END]")

                        if transcript:
                            buf.add("audio", f"[audio] {transcript}")
                            last_heard_text = transcript
                            last_heard_ts = time.time()
            except _SkipAudio:
                # Mic is muted for a moment (usually while TTS is speaking).
                pass

            except Exception as e:
                print("[DEBUG] audio exception:", repr(e))
                buf.add("audio", f"[audio] error: {e}")
                transcript = ""

            # ---------------------------
            # Memory handling (only on final transcript)
            # ---------------------------
            tl = transcript.lower().strip()

            if pending_memory is not None and tl:
                if tl in ("yes", "yep", "yeah", "sure", "remember that", "please do", "do it"):
                    mem_store.append(pending_memory)
                    print("[MEMORY] saved pending memory")
                    pending_memory = None
                elif tl in ("no", "nope", "don't", "do not", "forget it", "dont"):
                    print("[MEMORY] discarded pending memory")
                    pending_memory = None

            if transcript:
                dec = hybrid_decide(transcript)
                if dec.should_save and dec.proposed:
                    mem_store.append(dec.proposed)
                    print("[MEMORY] autosaved:", dec.proposed.kind)
                elif dec.ask_user and dec.proposed and dec.question:
                    pending_memory = dec.proposed

            # ---------------------------
            # Presence + inject tick
            # ---------------------------
            now = time.time()

            effective_transcript = transcript
            if (not effective_transcript) and (now - last_heard_ts) <= HEARD_HOLD_S:
                effective_transcript = last_heard_text

            presence.tick(transcript_text=effective_transcript)

            if now - last_inject >= inject_every:
                # Identity context (hold last known briefly)
                HOLD_IDENTITY_S = 3.0

                who_set = set(current_identities)
                if not who_set and (now - last_identity_ts) <= HOLD_IDENTITY_S:
                    who_set = set(last_known_identities)

                if who_set:
                    who = ", ".join(sorted(who_set))
                    buf.add("vision", f"[identity] present={who}")
                else:
                    who = "unknown"
                    buf.add("vision", "[identity] present=unknown")

                # VLM caption occasionally
                if vlm.enabled() and last_frame_bgr is not None:
                    try:
                        prompt = (
                            "You are describing a live camera frame.\n"
                            f"Identity label (authoritative): {who}\n"
                            "Rules:\n"
                            "- If identity label is not 'unknown', you MUST refer to the person as that name.\n"
                            "- If identity label is 'unknown', do NOT guess names.\n"
                            "- Write 1-3 sentences. Be natural and personal.\n"
                        )
                        cap = vlm.caption(last_frame_bgr, prompt, max_tokens=120)
                        if cap:
                            buf.add("vision", f"[vision-vlm] {cap}")
                    except Exception as e:
                        buf.add("vision", f"[vision-vlm] error: {e}")

                sensory_block = buf.snapshot_text() or "- (no observations captured)"
                decision = presence.decide(transcript_text=effective_transcript)

                internal_note = (
                    f"energy={presence.state.energy:.2f} "
                    f"decision={decision.should_respond} "
                    f"reason={decision.reason} "
                    f"confidence={decision.confidence:.2f}"
                )

                active_response = response_session and response_session.is_active()
                if decision.should_respond and not active_response:
                    user_content = (
                        f"SENSORY_UPDATE:\n{sensory_block}\n\n"
                        f"INTERNAL_STATE:\n{internal_note}\n\n"
                        f"IMPORTANT:\n"
                        f"- Your name is {wake_name}.\n"
                        f"- Respond naturally, like a person in the room.\n"
                        f"- Keep it short and conversational unless asked otherwise.\n"
                    )
                    if pending_memory is not None:
                        user_content += "\nMEMORY_QUESTION:\nShould I remember the last thing you said for next time? (yes/no)\n"

                    session_summary = load_summary(summary_path)
                    memories = list(mem_store.iter_items(limit=3000))

                    messages = build_messages(
                        system_prompt=SYSTEM_PROMPT,
                        session_summary=session_summary,
                        memories=memories,
                        user_text=user_content,
                        recent_turns=recent_dialogue,
                        max_memories=8,
                    )

                    print("\n[Inject -> VOICE]\n" + sensory_block)
                    print("[Astra]>", end=" ", flush=True)

                    live_thought_accum = ""

                    response_session = ResponseSession(
                        client=client,
                        tts=tts,
                        voice_model=voice_model,
                        messages=messages,
                        temperature=0.6,
                        max_tokens=int(env("LM_MAX_TOKENS", "550")),
                        user_content=user_content,
                        on_tts_chunk=speech_bridge.pulse,
                        on_stream_start=speech_bridge.on_stream_start,
                        on_stream_done=speech_bridge.on_stream_end,
                        on_fragment=on_generation_fragment,
                    )
                    response_session.start()
                    speech_bridge.on_stream_start()
                else:
                    print("\n[Silent tick]", internal_note)

                buf.clear()
                last_inject = now

            if response_session and (not response_session.is_active()):
                print("\n" + "-" * 60)
                if response_session.error:
                    print("[VOICE] response error:", response_session.error)
                    buf.add("audio", f"[voice] error: {response_session.error}")
                else:
                    full_reply = response_session.full_reply
                    if full_reply.strip():
                        speech_bridge.on_stream_end()
                    presence.mark_spoke()
                    recent_dialogue.append({"role": "user", "content": response_session.user_content})
                    recent_dialogue.append({"role": "assistant", "content": full_reply})
                    recent_dialogue = recent_dialogue[-20:]

                    inject_count += 1
                    if inject_count % 3 == 0:
                        try:
                            update_summary(
                                client=client,
                                model=voice_model,
                                summary_path=summary_path,
                                recent_dialogue=recent_dialogue[-10:],
                            )
                            print("[SUMMARY] updated")
                        except Exception as e:
                            print("[SUMMARY] update failed:", e)

                response_session = None

    except KeyboardInterrupt:
        print("\nStopping orchestrator...")
    finally:
        cam.stop()
        mic.stop()


if __name__ == "__main__":
    main()