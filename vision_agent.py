"""Vision agent for calling a vision-language model served by LM Studio.

This module provides a small helper class that converts OpenCV frames into
base64-encoded JPEG data URLs and sends them to a vision‑language model via
LM Studio's chat API.  If no vision model is configured, the agent can be
disabled and will return ``None``.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import cv2

from lmstudio_client import LMStudioClient


def frame_to_data_url_jpg(frame_bgr: Any, quality: int = 80) -> str:
    """Encode a BGR frame to a JPEG data URL."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", frame_bgr, encode_params)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


class VisionAgent:
    """Helper for calling a vision-language model via LM Studio."""

    def __init__(self, client: LMStudioClient, vlm_model_id: str | None = None) -> None:
        self.client = client
        self.vlm_model_id = (vlm_model_id or "").strip()

    def enabled(self) -> bool:
        return bool(self.vlm_model_id)

    def caption_frame(self, frame_bgr: Any) -> Optional[str]:
        """Return a short caption for the given frame, or ``None`` if disabled."""
        if not self.enabled():
            return None

        img_url = frame_to_data_url_jpg(frame_bgr)

        # Build an OpenAI-compatible multi-modal message
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are a vision assistant. Describe the image briefly and concretely.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe what you see in 1-2 sentences. Mention people, actions, and notable objects.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": img_url},
                    },
                ],
            },
        ]

        try:
            text = self.client.chat(self.vlm_model_id, messages, temperature=0.2)
            text = (text or "").strip()
            return text if text else None
        except Exception:
            # If anything goes wrong (e.g. model not loaded), fail gracefully
            return None