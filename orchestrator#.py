"""
Main entry point for the Astra LM Studio sensory bridge.

- Captures camera + mic.
- Mic uses an utterance collector: keeps listening until end-of-silence, then transcribes.
- Presence decides whether to respond.
- Optional VLM captioning on inject ticks.
- Hybrid memory + session summary.
- Optional Piper TTS for spoken output.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from agent.prompts import SYSTEM_PROMPT
from config import SETTINGS
from lmstudio_client import LMStudioClient
from presence_engine import PresenceEngine

from perception.audio_stream import MicrophoneSensor
from perception.stt_whisper import WhisperSTT
from perception.camera_stream import CameraSensor
from perception.utterance_collector import UtteranceCollector

from memory.ring_buffer import RingBuffer
from memory.memory_store import MemoryStore, MemoryItem
from memory.memory_policy import hybrid_decide
from memory.memory_summarize import update_summary, load_summary
from memory.memory_pack import build_messages

from tts_piper import PiperTTS, PiperConfig
from vision_vlm import VLMCaptioner
from vision_identity import IdentityRecognizer
import perception.utterance_collector as uc
print("[AUDIO] utterance_collector file:", uc.__file__)
load_dotenv()


def env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None and v != "" else default


def safe_add_vision(buf: RingBuffer, obs: Any) -> Optional[Any]:
    """Add a vision observation to the buffer, robust to different obs types.
    Returns a usable frame (BGR) if present, else None.
    """
    frame_bgr = None

    # If obs is already a string
    if isinstance(obs, str):
        buf.add("vision", obs)
        return None

    # If obs has to_text()
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

    # Try to pull a frame out
    if hasattr(obs, "frame_bgr"):
        frame_bgr = getattr(obs, "frame_bgr")
    elif hasattr(obs, "frame"):
        frame_bgr = getattr(obs, "frame")

    return frame_bgr


def main() -> None:
    base_url = SETTINGS.lmstudio_base_url
    api_key = SETTINGS.lmstudio_api_key
    voice_model = env("LMSTUDIO_MODEL", SETTINGS.lmstudio_model)
    vlm_model = env("VLM_MODEL", "").strip()
    wake_name = env("WAKE_NAME", SETTINGS.wake_name)

    inject_every = SETTINGS.inject_every_seconds
    max_events = SETTINGS.max_context_events

    client = LMStudioClient(base_url, api_key)

    # VLM captioner (Qwen3-VL etc.)
    vlm = VLMCaptioner(client, vlm_model) if vlm_model else VLMCaptioner(client, "")

    presence = PresenceEngine(wake_name=wake_name, cooldown_s=6.0)

    tts = PiperTTS(PiperConfig(model_name="en_GB-alba-medium"))
    print("Piper TTS enabled (en_GB-alba-medium)")

    buf = RingBuffer(max_events=max_events)

    # --- Memory system (Hybrid) ---
    mem_store = MemoryStore("memory/memory_store.jsonl")
    summary_path = "memory/session_summary.txt"
    recent_dialogue: List[Dict[str, Any]] = []
    pending_memory: Optional[MemoryItem] = None
    inject_count = 0

    # Sensors
    cam = CameraSensor(SETTINGS.camera_index, SETTINGS.camera_fps, SETTINGS.camera_downscale_width)
    mic = MicrophoneSensor(SETTINGS.audio_sample_rate, SETTINGS.audio_chunk_seconds, SETTINGS.audio_device)
    stt = WhisperSTT(model_size="small")

    streamer = stt.make_streamer(
        input_sr=SETTINGS.audio_sample_rate,
        update_every_s=0.7,
        window_s=8.0,
        lookback_s=2.0,
    )
    last_partial_print = 0.0
    

    # Identity (load known faces once)
    identity = IdentityRecognizer(
        known_faces_dir=r"C:\astra_main\faces",
        cooldown_s=10.0,
        debug=True,
    )

    # Identity state snapshot (no 'kind' needed)
    current_identities: set[str] = set()
    last_identity_ts = 0.0
    last_known_identities: set[str] = set()

    # IMPORTANT: Collector must live OUTSIDE the loop (stateful)
    collector = UtteranceCollector(
        sample_rate=SETTINGS.audio_sample_rate,        
        speech_rms=0.006,
        min_speech_s=0.15,
        end_silence_s=1.2,
        max_utterance_s=15.0,
    )

    cam.start()
    mic.start()

    last_inject = 0.0
    last_frame_bgr = None  # keep the most recent frame we saw

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
            try:
                obs = cam.read()
                frame_bgr = safe_add_vision(buf, obs)
                if frame_bgr is not None:
                    _ = identity.process(frame_bgr)            # updates identity.present internally
                    current_identities = set(identity.present)  # snapshot
                    if current_identities:
                        last_known_identities = set(current_identities)
                        last_identity_ts = time.time()
                    last_frame_bgr = frame_bgr
            except Exception as e:
                buf.add("vision", f"[vision] error: {e}")

            # ---------------------------
            # Audio (collect until silence) + STT
            # ---------------------------
            transcript = ""
            try:
                chunk = mic.read_chunk()

                import numpy as np

                pcm = np.asarray(chunk.pcm)

                # --- Force mono ---
                if pcm.ndim == 2 and pcm.shape[1] >= 2:
                    pcm = pcm.mean(axis=1)

                # --- Force float32 [-1,1] ---
                if pcm.dtype != np.float32 and pcm.dtype != np.float64:
                    pcm = pcm.astype(np.float32) / 32768.0
                else:
                    pcm = pcm.astype(np.float32, copy=False)

                # --- RMS + VAD debug ---
                rms = float(np.sqrt(np.mean(np.square(pcm))) + 1e-12)
                thr = getattr(collector, "speech_rms", 0.0)
                print(f"[AUDIO][VAD] rms={rms:.6f} thr={thr:.6f} above={rms >= thr}")

                upd = streamer.push(pcm, chunk.sample_rate)
                if upd is not None:
                    nowp = time.time()

                    # Print partials occasionally (so console isn't a waterfall)
                    if upd.partial and (nowp - last_partial_print) >= 0.7:
                        print(f"[PARTIAL] {upd.partial}")
                        last_partial_print = nowp

                    # Committed text is "stable" streaming output
                    if upd.committed:
                        print(f"[COMMIT] {upd.committed}")
                        buf.add("audio", f"[audio-live] {upd.committed}")

                
                ready = collector.push(pcm)

                utt = None
                if ready:
                    print("[AUDIO][VAD] utterance ready -> finalizing")
                    utt = collector.finalize()

                    # Inspect what finalize produced
                    if utt is None:
                        print("[AUDIO][DBG] finalize returned None")
                    elif utt.pcm is None:
                        print("[AUDIO][DBG] finalize returned utt but pcm is None")
                    else:
                        samples = int(np.asarray(utt.pcm).shape[0])
                        print(f"[AUDIO][DBG] utt seconds={utt.seconds:.2f} samples={samples}")

                # Only transcribe when we have a real utterance
                if utt is not None and utt.pcm is not None:
                    tr = stt.transcribe(utt.pcm, utt.sample_rate)
                    transcript = (tr.text or "").strip()

                    print(f"[DEBUG] utterance_seconds={utt.seconds:.2f}")
                    print(f"[DEBUG] whisper_text={repr(tr.text)}")
                    print(f"[DEBUG] transcript_final={repr(transcript)}")

                streamer.reset()


                    if transcript:
                        buf.add("audio", f"[audio] {transcript}")

            except Exception as e:
                print("[DEBUG] audio exception:", repr(e))
                buf.add("audio", f"[audio] error: {e}")
                transcript = ""  # keep downstream logic sane


            # ---------------------------
            # Memory handling (AFTER transcript exists)
            # ---------------------------
            tl = transcript.lower().strip()

            # Pending yes/no confirmation
            if pending_memory is not None and tl:
                if tl in ("yes", "yep", "yeah", "sure", "remember that", "please do", "do it"):
                    mem_store.append(pending_memory)
                    print("[MEMORY] saved pending memory")
                    pending_memory = None
                elif tl in ("no", "nope", "don't", "do not", "forget it", "dont"):
                    print("[MEMORY] discarded pending memory")
                    pending_memory = None

            # Hybrid decide
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
            presence.tick(transcript_text=transcript)
            now = time.time()

            # ---------------------------
            # Identity context (inject-tick only, avoids spamming)
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

            # Add a real VLM caption occasionally (only on inject ticks)
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
                    print(f"[VLM][DEBUG] prompt identity={who!r}")

                    cap = vlm.caption(
                        last_frame_bgr,
                        prompt,
                        max_tokens=120,
                    )
                    if cap:
                        buf.add("vision", f"[vision-vlm] {cap}")
                except Exception as e:
                    buf.add("vision", f"[vision-vlm] error: {e}")


                sensory_block = buf.snapshot_text() or "- (no observations captured)"
                decision = presence.decide(transcript_text=transcript)

                internal_note = (
                    f"energy={presence.state.energy:.2f} "
                    f"decision={decision.should_respond} "
                    f"reason={decision.reason} "
                    f"confidence={decision.confidence:.2f}"
                )

                if decision.should_respond:
                    user_content = (
                        f"SENSORY_UPDATE:\n{sensory_block}\n\n"
                        f"INTERNAL_STATE:\n{internal_note}\n\n"
                        f"IMPORTANT:\n"
                        f"- Your name is {wake_name}.\n"
                        f"- You may choose to respond briefly, or say you're low-energy and staying quiet.\n"
                        f"- If you respond, keep it natural, like a person in the room.\n"
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

                    full_reply = ""
                    for frag in client.chat_stream(voice_model, messages, temperature=0.6, max_tokens=300):
                        print(frag, end="", flush=True)
                        full_reply += frag

                    print("\n" + "-" * 60)
                    presence.mark_spoke()

                    if full_reply.strip():
                        tts.speak(full_reply)

                    # Short-term rolling context
                    recent_dialogue.append({"role": "user", "content": user_content})
                    recent_dialogue.append({"role": "assistant", "content": full_reply})
                    recent_dialogue = recent_dialogue[-20:]

                    # Periodic summary update
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

                else:
                    print("\n[Silent tick]", internal_note)

                buf.clear()
                last_inject = now

    except KeyboardInterrupt:
        print("\nStopping orchestrator...")
    finally:
        cam.stop()
        mic.stop()


if __name__ == "__main__":
    main()
