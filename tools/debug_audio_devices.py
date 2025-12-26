import sounddevice as sd

print("Default input device:", sd.default.device)
print("\nAll devices:")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        print(f"[{i}] {d['name']} | inputs={d['max_input_channels']} | default_sr={d['default_samplerate']}")