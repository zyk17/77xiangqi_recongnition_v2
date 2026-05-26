from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..board_detector.predict import OnnxBoardDetector
from ..piece_classifier.predict import OnnxPieceClassifier
from ..types import CellPrediction, ReferencePipelineResult
from .grid import SampledCell


class ReferenceVisionPipeline:
    """
    仅用于本地验证与给 Rust 提供参考逻辑。
    交付目标仍然是两个 ONNX，而不是 Python 线上服务。
    """

    def __init__(
        self,
        *,
        board_detector_path: Path,
        piece_classifier_path: Path,
        label_map_path: Path,
        board_input_size: int = 224,
        piece_input_size: int = 48,
    ) -> None:
        self.board_detector = OnnxBoardDetector(board_detector_path, image_size=board_input_size)
        self.piece_classifier = OnnxPieceClassifier(
            piece_classifier_path,
            label_map_path,
            image_size=piece_input_size,
        )

    def _sample_cells_from_boxes(self, image: Image.Image, boxes) -> list[SampledCell]:
        cells: list[SampledCell] = []
        index = 0
        for row in range(10):
            for col in range(9):
                x1, y1, x2, y2 = (int(round(float(v))) for v in boxes[index])
                width, height = image.size
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(x1 + 1, min(width, x2))
                y2 = max(y1 + 1, min(height, y2))
                cells.append(
                    SampledCell(
                        row=row,
                        col=col,
                        bbox=(x1, y1, x2, y2),
                        image=image.crop((x1, y1, x2, y2)),
                    )
                )
                index += 1
        return cells

    def run(self, image: Image.Image) -> ReferencePipelineResult:
        cell_boxes = self.board_detector.predict_cell_boxes(image)
        board_bbox = self.board_detector.predict(image, cell_boxes=cell_boxes)
        sampled = self._sample_cells_from_boxes(image, cell_boxes)
        cells: list[CellPrediction] = []
        for cell in sampled:
            label, confidence = self.piece_classifier.predict(cell.image)
            cells.append(
                CellPrediction(
                    row=cell.row,
                    col=cell.col,
                    label=label,
                    confidence=confidence,
                    bbox=cell.bbox,
                )
            )
        return ReferencePipelineResult(
            board_bbox=board_bbox,
            cells=cells,
            board_recovery_ok=self.board_detector.last_recovery_ok,
        )
