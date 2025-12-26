import base64
import cv2
from typing import Any, Dict, List

class VLMCaptioner:
    def __init__(self, client, model_id: str):
        self.client = client
        self.model_id = (model_id or "").strip()

    def enabled(self) -> bool:
        return bool(self.model_id)

    @staticmethod
    def _to_data_url(frame_bgr, max_side: int = 1288, quality: int = 85) -> str:
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        ok, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")

        b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def caption(self, frame_bgr, prompt: str, max_tokens: int = 120) -> str:
        data_url = self._to_data_url(frame_bgr)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a vision assistant. Be concise and concrete."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]
        return self.client.chat(self.model_id, messages, temperature=0.2, max_tokens=max_tokens).strip()
