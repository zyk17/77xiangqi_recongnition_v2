from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from ..piece_classifier.eval_cases import fen_to_grid, orient_grid
from .service import ReferenceVisionPipeline


class RawPipeline(Protocol):
    def run(self, image: Image.Image):  # pragma: no cover - protocol only
        ...


@dataclass(slots=True)
class RawWrongPrediction:
    row: int
    col: int
    expected: str
    predicted: str
    confidence: float


@dataclass(slots=True)
class RawCaseResult:
    case_name: str
    raw_name: str
    image_size: tuple[int, int]
    board_bbox: dict[str, int]
    board_recovery_ok: bool
    cell_acc: float
    piece_acc: float
    wrong_count: int
    wrong_examples: list[RawWrongPrediction]


def find_raw_image(case_dir: Path) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        path = case_dir / f"raw{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing raw image in {case_dir}")


def iter_raw_eval_cases(eval_root: Path) -> list[Path]:
    out: list[Path] = []
    for path in sorted(eval_root.iterdir()):
        if not path.is_dir():
            continue
        if not (path / "expected.json").is_file():
            continue
        try:
            find_raw_image(path)
        except FileNotFoundError:
            continue
        out.append(path)
    return out


def evaluate_raw_case(case_dir: Path, pipeline: RawPipeline) -> RawCaseResult:
    raw_path = find_raw_image(case_dir)
    expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
    grid = orient_grid(fen_to_grid(str(expected["fen"])), str(expected.get("view_side", "red")))
    image = Image.open(raw_path).convert("RGB")
    result = pipeline.run(image)

    total = 0
    correct = 0
    piece_total = 0
    piece_correct = 0
    wrongs: list[RawWrongPrediction] = []
    for cell in result.cells:
        expected_label = grid[cell.row][cell.col]
        total += 1
        if expected_label != "empty":
            piece_total += 1
        if cell.label == expected_label:
            correct += 1
            if expected_label != "empty":
                piece_correct += 1
        else:
            wrongs.append(
                RawWrongPrediction(
                    row=cell.row,
                    col=cell.col,
                    expected=expected_label,
                    predicted=cell.label,
                    confidence=cell.confidence,
                )
            )

    return RawCaseResult(
        case_name=case_dir.name,
        raw_name=raw_path.name,
        image_size=image.size,
        board_bbox={
            "x": result.board_bbox.x,
            "y": result.board_bbox.y,
            "w": result.board_bbox.w,
            "h": result.board_bbox.h,
        },
        board_recovery_ok=result.board_recovery_ok,
        cell_acc=correct / max(total, 1),
        piece_acc=piece_correct / max(piece_total, 1),
        wrong_count=len(wrongs),
        wrong_examples=wrongs[:12],
    )


def build_reference_pipeline(
    *,
    repo_root: Path,
    board_detector_path: Path,
    piece_classifier_path: Path,
    label_map_path: Path,
) -> ReferenceVisionPipeline:
    resolved_board_detector = (
        board_detector_path if board_detector_path.is_absolute() else (repo_root / board_detector_path).resolve()
    )
    resolved_piece_classifier = (
        piece_classifier_path if piece_classifier_path.is_absolute() else (repo_root / piece_classifier_path).resolve()
    )
    resolved_label_map = label_map_path if label_map_path.is_absolute() else (repo_root / label_map_path).resolve()
    return ReferenceVisionPipeline(
        board_detector_path=resolved_board_detector,
        piece_classifier_path=resolved_piece_classifier,
        label_map_path=resolved_label_map,
    )
