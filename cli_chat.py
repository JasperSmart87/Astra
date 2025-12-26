from __future__ import annotations
import time
from typing import List, Dict

from config import SETTINGS
from lmstudio_client import LMStudioClient
from memory.ring_buffer import RingBuffer
from perception.camera_stream import CameraSensor
from perception.audio_stream import MicrophoneSensor
from perception.stt_whisper import WhisperSTT
from agent.prompts import SYSTEM_PROMPT

def main() -> None:
    client = LMStudioClient(SETTINGS.lmstudio_base_url, SETTINGS.lmstudio_api_key)
    buf = RingBuffer(max_events=SETTINGS.max_context_events)

    cam = CameraSensor(SETTINGS.camera_index, SETTINGS.camera_fps, SETTINGS.camera_downscale_width)
    mic = MicrophoneSensor(SETTINGS.audio_sample_rate, SETTINGS.audio_chunk_seconds, SETTINGS.audio_device)
    stt = WhisperSTT(model_size="small")

    cam.start()
    mic.start()

    conversation: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("CLI chat + sensors. Type your message and press enter. Ctrl+C to quit.")
    try:
        while True:
            # background sensory capture (one step)
            try:
                obs = cam.read()
                buf.add("vision", obs.to_text())
            except Exception:
                pass
            try:
                chunk = mic.read_chunk()
                tr = stt.transcribe(chunk.pcm, chunk.sample_rate)
                if tr.to_text():
                    buf.add("audio", tr.to_text())
            except Exception:
                pass

            user = input("\nYou> ").strip()
            if not user:
                continue

            sensory_block = buf.snapshot_text()
            if sensory_block:
                user = user + "\n\n(Recent sensory observations)\n" + sensory_block
                buf.clear()

            conversation.append({"role": "user", "content": user})

            print("Model> ", end="", flush=True)
            for frag in client.chat_stream(SETTINGS.lmstudio_model, conversation):
                print(frag, end="", flush=True)
            print()
    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        cam.stop()
        mic.stop()

if __name__ == "__main__":
    main()
