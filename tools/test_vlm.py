import base64, cv2, requests, json

BASE = "http://127.0.0.1:1234/v1"
MODEL = "qwen/qwen3-vl-8b"

cap = cv2.VideoCapture(0)
ok, frame = cap.read()
cap.release()
assert ok

_, jpg = cv2.imencode(".jpg", frame)
b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a vision assistant. Be concise."},
        {"role": "user", "content": [
            {"type": "text", "text": "What do you see?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}
    ],
    "temperature": 0.2,
    "max_tokens": 200
}

r = requests.post(BASE + "/chat/completions", json=payload, timeout=120)
print("status:", r.status_code)
print(r.text[:2000])
