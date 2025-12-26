from __future__ import annotations
import time
from typing import List, Dict

from config import SETTINGS
from lmstudio_client import LMStudioClient
from memory.ring_buffer import RingBuffer
from perception.camera_stream import CameraSensor
from perception.audio_stream import MicrophoneSensor
from perception.stt_whisper import WhisperSTT
from agent.prompts import SYSTEM_PROMPT, build_injection_message

def run() -> None:
    client = LMStudioClient(SETTINGS.lmstudio_base_url, SETTINGS.lmstudio_api_key)
    buf = RingBuffer(max_events=SETTINGS.max_context_events)

    cam = CameraSensor(SETTINGS.camera_index, SETTINGS.camera_fps, SETTINGS.camera_downscale_width)
    mic = MicrophoneSensor(SETTINGS.audio_sample_rate, SETTINGS.audio_chunk_seconds, SETTINGS.audio_device)
    stt = WhisperSTT(model_size="small")

    cam.start()
    mic.start()

    last_inject = 0.0

    print("Sensory Bridge started. Press Ctrl+C to stop.")
    print(f"LM Studio: {SETTINGS.lmstudio_base_url} | model={SETTINGS.lmstudio_model}")

    try:
        while True:
            # 1) Collect camera observation
            try:
                obs = cam.read()
                buf.add("vision", obs.to_text())
            except Exception as e:
                buf.add("vision", f"[vision] error: {e}")

            # 2) Collect microphone chunk + transcription
            try:
                chunk = mic.read_chunk()
                tr = stt.transcribe(chunk.pcm, chunk.sample_rate)
                if tr.to_text():
                    buf.add("audio", tr.to_text())
            except Exception as e:
                buf.add("audio", f"[audio] error: {e}")

            # 3) Periodically inject into LM Studio
            now = time.time()
            if now - last_inject >= SETTINGS.inject_every_seconds:
                block = buf.snapshot_text()
                if not block:
                    block = "- (no observations captured)"
                messages: List[Dict[str, str]] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_injection_message(block)},
                ]

                print("\n\n[Injecting sensory update -> LM Studio]\n" + block + "\n")
                print("[Model reply]\n", end="", flush=True)
                for frag in client.chat_stream(SETTINGS.lmstudio_model, messages):
                    print(frag, end="", flush=True)
                print("\n" + "-"*60)
                buf.clear()
                last_inject = now

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cam.stop()
        mic.stop()

