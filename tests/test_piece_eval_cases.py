from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from xiangqi_recognize_v2.piece_classifier.eval_cases import evaluate_piece_case, fen_to_grid, orient_grid


class FakePredictor:
    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.index = 0

    def predict(self, _image: Image.Image) -> tuple[str, float]:
        label = self.labels[self.index]
        self.index += 1
        return label, 0.99


def test_evaluate_piece_case_red_view(tmp_path: Path) -> None:
    case_dir = tmp_path / "case01"
    case_dir.mkdir()
    Image.new("RGB", (450, 500), color=(220, 210, 190)).save(case_dir / "clipped.png")
    expected = {"fen": "9/9/9/9/9/9/9/9/9/4K4 w - - 0 1", "view_side": "red"}
    (case_dir / "expected.json").write_text(json.dumps(expected), encoding="utf-8")

    grid = orient_grid(fen_to_grid(expected["fen"]), expected["view_side"])
    labels = [label for row in grid for label in row]
    result = evaluate_piece_case(case_dir, FakePredictor(labels))
    assert result.case_name == "case01"
    assert result.cell_acc == 1.0
    assert result.piece_acc == 1.0
    assert result.wrong_count == 0


def test_evaluate_piece_case_black_view_rotation(tmp_path: Path) -> None:
    case_dir = tmp_path / "case02"
    case_dir.mkdir()
    Image.new("RGB", (450, 500), color=(220, 210, 190)).save(case_dir / "clipped.jpg")
    expected = {"fen": "4k4/9/9/9/9/9/9/9/9/9 w - - 0 1", "view_side": "black"}
    (case_dir / "expected.json").write_text(json.dumps(expected), encoding="utf-8")

    grid = orient_grid(fen_to_grid(expected["fen"]), expected["view_side"])
    labels = [label for row in grid for label in row]
    result = evaluate_piece_case(case_dir, FakePredictor(labels))
    assert result.cell_acc == 1.0
    assert result.piece_acc == 1.0
