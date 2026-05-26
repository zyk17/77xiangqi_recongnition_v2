from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from ..reference.grid import sample_board_cells
from ..types import Roi

PIECE_MAP = {
    "k": "k",
    "a": "a",
    "b": "b",
    "n": "n",
    "r": "r",
    "c": "c",
    "p": "p",
}


class PiecePredictor(Protocol):
    def predict(self, image: Image.Image) -> tuple[str, float]:
        ...


@dataclass(slots=True)
class WrongPrediction:
    row: int
    col: int
    expected: str
    predicted: str
    confidence: float


@dataclass(slots=True)
class PieceCaseResult:
    case_name: str
    image_size: tuple[int, int]
    cell_acc: float
    piece_acc: float
    wrong_count: int
    wrong_examples: list[WrongPrediction]


def fen_to_grid(fen: str) -> list[list[str]]:
    board = fen.split()[0]
    rows = board.split("/")
    if len(rows) != 10:
        raise ValueError(f"fen board rows should be 10: {fen!r}")
    out: list[list[str]] = []
    for row in rows:
        cells: list[str] = []
        for ch in row:
            if ch.isdigit():
                cells.extend(["empty"] * int(ch))
            else:
                side = "r" if ch.isupper() else "b"
                cells.append(f"{side}{PIECE_MAP[ch.lower()]}")
        if len(cells) != 9:
            raise ValueError(f"fen row should have 9 cells: {row!r}")
        out.append(cells)
    return out


def orient_grid(grid: list[list[str]], view_side: str) -> list[list[str]]:
    if view_side.lower() == "black":
        return [list(reversed(row)) for row in reversed(grid)]
    return grid


def find_clipped_image(case_dir: Path) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        path = case_dir / f"clipped{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing clipped image in {case_dir}")


def iter_piece_eval_cases(eval_root: Path) -> list[Path]:
    out: list[Path] = []
    for path in sorted(eval_root.iterdir()):
        if not path.is_dir():
            continue
        if not (path / "expected.json").is_file():
            continue
        try:
            find_clipped_image(path)
        except FileNotFoundError:
            continue
        out.append(path)
    return out


def evaluate_piece_case(case_dir: Path, predictor: PiecePredictor) -> PieceCaseResult:
    expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
    grid = orient_grid(fen_to_grid(str(expected["fen"])), str(expected.get("view_side", "red")))
    image = Image.open(find_clipped_image(case_dir)).convert("RGB")
    roi = Roi(x=0, y=0, w=image.size[0], h=image.size[1])
    sampled = sample_board_cells(image, roi)

    total = 0
    correct = 0
    piece_total = 0
    piece_correct = 0
    wrongs: list[WrongPrediction] = []
    for cell in sampled:
        predicted, confidence = predictor.predict(cell.image)
        expected_label = grid[cell.row][cell.col]
        total += 1
        if expected_label != "empty":
            piece_total += 1
        if predicted == expected_label:
            correct += 1
            if expected_label != "empty":
                piece_correct += 1
        else:
            wrongs.append(
                WrongPrediction(
                    row=cell.row,
                    col=cell.col,
                    expected=expected_label,
                    predicted=predicted,
                    confidence=confidence,
                )
            )

    return PieceCaseResult(
        case_name=case_dir.name,
        image_size=image.size,
        cell_acc=correct / max(total, 1),
        piece_acc=piece_correct / max(piece_total, 1),
        wrong_count=len(wrongs),
        wrong_examples=wrongs[:12],
    )

