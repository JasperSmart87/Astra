# vision_identity.py
import os
import time
from typing import List, Dict
import numpy as np

try:
    import face_recognition
except ImportError:
    face_recognition = None
print("[IDENTITY] vision_identity.py imported")



class IdentityEvent:
    def __init__(self, name: str, confidence: float, ts: float):
        self.name = name
        self.confidence = confidence
        self.ts = ts


class IdentityRecognizer:
    """
    Passive face identity recognizer.
    It never interrupts. It only reports changes.
    """

    def __init__(self, known_faces_dir="faces", cooldown_s=10.0, debug: bool = False):
        self.known_faces_dir = known_faces_dir
        self.cooldown_s = cooldown_s
        self.debug = debug
        print(f"[IDENTITY] init known_faces_dir={os.path.abspath(self.known_faces_dir)} debug={self.debug}")
        print(f"[IDENTITY] init dir={os.path.abspath(self.known_faces_dir)} debug={self.debug}")
        self.known_encodings = []
        self.known_names = []

        self.last_seen: Dict[str, float] = {}
        self.present: set[str] = set()

        self._load_faces()

    def _load_faces(self):
        if not face_recognition:
            print("[IDENTITY] face_recognition not installed, identity disabled")
            return

        if not os.path.isdir(self.known_faces_dir):
            print(f"[IDENTITY] no faces directory at {self.known_faces_dir}")
            return

        for fn in os.listdir(self.known_faces_dir):
            path = os.path.join(self.known_faces_dir, fn)
            name, _ = os.path.splitext(fn)

            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    self.known_encodings.append(encs[0])
                    self.known_names.append(name)
                    print(f"[IDENTITY] loaded {name}")
            except Exception as e:
                print(f"[IDENTITY] failed to load {fn}: {e}")

    def process(self, frame_bgr) -> List[IdentityEvent]:
        if not face_recognition or not self.known_encodings:
            return []

        now = time.time()
        events = []

        # Ensure correct layout for dlib: RGB, uint8, contiguous
        try:
            rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        except Exception as e:
            if self.debug:
                print(f"[IDENTITY][DEBUG] rgb convert failed: {e}")
            return []

        try:
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)
        except Exception as e:
            # Prevents your whole vision block from erroring out
            if self.debug:
                print(f"[IDENTITY][DEBUG] face_encodings failed: {e}")
            return []

        if self.debug:
            print(f"[IDENTITY][DEBUG] faces_detected={len(encodings)}")

        seen_now = set()

        for enc in encodings:
            matches = face_recognition.compare_faces(self.known_encodings, enc, tolerance=0.48)
            if True in matches:
                i = matches.index(True)
                name = self.known_names[i]
                seen_now.add(name)

                if self.debug:
                    dist = float(face_recognition.face_distance([self.known_encodings[i]], enc)[0])
                    print(f"[IDENTITY][DEBUG] matched={name} dist={dist:.4f} tol=0.48")

                last = self.last_seen.get(name, 0)
                if now - last > self.cooldown_s:
                    self.last_seen[name] = now
                    if name not in self.present:
                        events.append(IdentityEvent(name, 0.95, now))

        for name in list(self.present):
            if name not in seen_now:
                events.append(IdentityEvent(name, 0.95, now))
                self.present.remove(name)

        self.present = seen_now
        return events
