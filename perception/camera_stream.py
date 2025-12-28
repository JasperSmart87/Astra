"""Video capture helper using OpenCV.

This module defines a camera sensor that captures frames from the default or
specified webcam and computes simple summary statistics such as brightness
and motion.  It returns an object that can be converted to text for logging
purposes.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import cv2
except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
    cv2 = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None
import numpy as np
import time

from dataclasses import dataclass
from typing import Any

@dataclass
class CameraObservation:
    motion: float
    brightness: float
    frame: Any  # <-- add this
    ts: float

    def to_text(self) -> str:
        return f"[vision-summary] motion={self.motion:.2f}, brightness={self.brightness:.2f}"



class CameraSensor:
    """Captures frames from a webcam and computes simple metrics."""

    def __init__(self, camera_index: int = 0, fps: int = 1, downscale_width: int = 320) -> None:
        self.camera_index = camera_index
        self.fps = fps
        self.downscale_width = downscale_width
        self.cap: cv2.VideoCapture | None = None
        self.prev_gray: np.ndarray | None = None

    def start(self) -> None:
        if cv2 is None:  # pragma: no cover - environment-specific
            raise ImportError(
                "OpenCV (cv2) is required for vision capture. Install opencv-python from requirements.txt"
            ) from _IMPORT_ERROR
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")
        self.prev_gray = None

    def read(self) -> CameraObservation:
        if self.cap is None:
            raise RuntimeError("Camera has not been started")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        # Downscale the frame for performance
        if self.downscale_width:
            ratio = self.downscale_width / frame.shape[1]
            height = int(frame.shape[0] * ratio)
            frame = cv2.resize(frame, (self.downscale_width, height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        motion = 0.0
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            motion = float(np.mean(diff))
        self.prev_gray = gray
        return CameraObservation(
            motion=motion,
            brightness=brightness,
            frame=frame,  # <-- PASS THE FRAME THROUGH
            ts=time.time(),
        )

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
