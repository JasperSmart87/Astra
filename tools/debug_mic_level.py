import sounddevice as sd
import numpy as np
import time

DEVICE = 1 # set to an int like 3 if you want
SR = 44100

print("Speak into the mic. Showing RMS level...")
def callback(indata, frames, time_info, status):
    rms = np.sqrt(np.mean(indata**2))
    print(f"RMS: {rms:.4f}")

with sd.InputStream(samplerate=SR, channels=1, dtype="float32", device=DEVICE, callback=callback):
    time.sleep(8)
