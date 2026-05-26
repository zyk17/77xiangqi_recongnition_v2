from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .labels import load_label_names


class OnnxPieceClassifier:
    def __init__(self, model_path: Path, label_map_path: Path, image_size: int = 48) -> None:
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.labels = load_label_names(label_map_path)
        self.image_size = image_size

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize((self.image_size, self.image_size))
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        arr = (arr - mean) / std
        return arr[None, ...]

    def predict(self, image: Image.Image) -> tuple[str, float]:
        logits = self.session.run(None, {self.input_name: self._preprocess(image)})[0][0]
        logits = np.asarray(logits, dtype=np.float32)
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)
        index = int(np.argmax(probs))
        return self.labels[index], float(probs[index])
