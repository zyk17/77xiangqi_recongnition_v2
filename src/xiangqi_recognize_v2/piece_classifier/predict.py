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
        resized = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.uint8)
        return np.ascontiguousarray(arr)[None, ...]

    def predict(self, image: Image.Image) -> tuple[str, float]:
        logits = self.session.run(None, {self.input_name: self._preprocess(image)})[0][0]
        logits = np.asarray(logits, dtype=np.float32)
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)
        index = int(np.argmax(probs))
        return self.labels[index], float(probs[index])
