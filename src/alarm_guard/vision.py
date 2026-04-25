from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


@dataclass
class DrowsySignal:
    eyes_closed_for: float
    triggered: bool


class DrowsinessDetector:
    def __init__(self, eye_model_path: str, threshold_seconds: float, frame_skip: int = 2) -> None:
        self.threshold_seconds = threshold_seconds
        self.frame_skip = max(1, frame_skip)
        self.eye_closed_since: float | None = None
        self.counter = 0
        self.model = load_model(eye_model_path) if eye_model_path else None

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame: np.ndarray) -> DrowsySignal:
        self.counter += 1
        if self.counter % self.frame_skip != 0:
            return DrowsySignal(eyes_closed_for=0.0, triggered=False)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            self.eye_closed_since = None
            return DrowsySignal(eyes_closed_for=0.0, triggered=False)

        h, w = frame.shape[:2]
        lm = result.multi_face_landmarks[0].landmark
        left_eye_img = self._extract_eye_patch(frame, lm, LEFT_EYE, w, h)
        right_eye_img = self._extract_eye_patch(frame, lm, RIGHT_EYE, w, h)

        left_closed = self._predict_closed(left_eye_img)
        right_closed = self._predict_closed(right_eye_img)
        closed = left_closed and right_closed

        if closed:
            if self.eye_closed_since is None:
                self.eye_closed_since = time.time()
            elapsed = time.time() - self.eye_closed_since
            return DrowsySignal(eyes_closed_for=elapsed, triggered=elapsed >= self.threshold_seconds)

        self.eye_closed_since = None
        return DrowsySignal(eyes_closed_for=0.0, triggered=False)

    def _extract_eye_patch(self, frame: np.ndarray, landmarks, eye_idx: list[int], width: int, height: int) -> np.ndarray:
        points = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in eye_idx]
        xs, ys = zip(*points)
        x1, x2 = max(min(xs) - 4, 0), min(max(xs) + 4, width)
        y1, y2 = max(min(ys) - 4, 0), min(max(ys) + 4, height)
        eye = frame[y1:y2, x1:x2]
        if eye.size == 0:
            return np.zeros((24, 24, 1), dtype=np.float32)
        gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (24, 24)).astype(np.float32) / 255.0
        return resized[..., None]

    def _predict_closed(self, eye_patch: np.ndarray) -> bool:
        if self.model is None:
            return False
        pred = self.model.predict(np.expand_dims(eye_patch, axis=0), verbose=0)[0][0]
        return pred >= 0.5
