from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from ..types import Roi
from .decode import cell_bboxes_from_evidence_maps, roi_from_cell_boxes
from .export import load_board_detector_meta


class OnnxBoardDetector:
    def __init__(
        self,
        model_path: Path,
        *,
        image_size: int | None = None,
        map_downsample: int | None = None,
        recovery_threshold: float | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(str(self.model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.maps_output_name = self.session.get_outputs()[0].name
        runtime = load_board_detector_meta(
            model_path=self.model_path,
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            map_downsample=map_downsample,
            recovery_threshold=recovery_threshold,
        )
        self.image_size = int(runtime["image_size"])
        self.map_downsample = int(runtime["map_downsample"])
        self.evidence_channels = int(runtime["evidence_channels"])
        self.recovery_threshold = float(runtime["recovery_threshold"])
        self.last_recovery_ok = True

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize((self.image_size, self.image_size))
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        arr = (arr - mean) / std
        return arr[None, ...]

    def predict_evidence_maps(self, image: Image.Image) -> np.ndarray:
        maps = self.session.run(None, {self.input_name: self._preprocess(image)})[0]
        output = maps[0].astype(np.float32)
        if output.shape[0] != self.evidence_channels:
            raise ValueError(f"ONNX evidence maps 通道数应为 {self.evidence_channels}")
        expected_h = max(1, self.image_size // self.map_downsample)
        expected_w = max(1, self.image_size // self.map_downsample)
        if output.shape[-2:] != (expected_h, expected_w):
            raise ValueError(
                "ONNX evidence maps 空间尺寸与 meta 不一致: "
                f"got={output.shape[-2:]}, expected={(expected_h, expected_w)}, "
                f"map_downsample={self.map_downsample}"
            )
        return output

    def predict_cell_boxes(self, image: Image.Image) -> np.ndarray:
        width, height = image.size
        maps = self.predict_evidence_maps(image)
        boxes, recovery_ok = cell_bboxes_from_evidence_maps(
            torch.tensor(maps, dtype=torch.float32),
            input_width=self.image_size,
            input_height=self.image_size,
            image_width=width,
            image_height=height,
            downsample=self.map_downsample,
            threshold=self.recovery_threshold,
            allow_fallback=True,
        )
        self.last_recovery_ok = recovery_ok
        return boxes.numpy()

    def predict(self, image: Image.Image, *, cell_boxes: np.ndarray | None = None) -> Roi:
        """外接矩形，仅供兼容旧接口。可传入已算好的 cell_boxes 避免重复推理。"""
        width, height = image.size
        boxes = cell_boxes if cell_boxes is not None else self.predict_cell_boxes(image)
        x1, y1, x2, y2 = roi_from_cell_boxes(
            torch.tensor(boxes, dtype=torch.float32),
            image_width=width,
            image_height=height,
        )
        return Roi(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
