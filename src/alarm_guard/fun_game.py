from __future__ import annotations

import cv2
import numpy as np
from tensorflow.keras.models import load_model


MONKEY_MAP = {
    "happy": "🙈",
    "sad": "🙊",
    "angry": "🙉",
    "surprised": "🐵",
}


class MonkeyEmojiGame:
    def __init__(self, model_path: str) -> None:
        self.model = load_model(model_path)
        self.labels = ["angry", "happy", "sad", "surprised"]

    def classify(self, frame: np.ndarray) -> tuple[str, str, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (48, 48)).astype(np.float32) / 255.0
        face = face[..., None]
        probs = self.model.predict(np.expand_dims(face, axis=0), verbose=0)[0]
        idx = int(np.argmax(probs))
        label = self.labels[idx]
        return label, MONKEY_MAP.get(label, "🐵"), float(probs[idx])
