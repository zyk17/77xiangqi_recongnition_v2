from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from xiangqi_recognize_v2.reference.eval_cases import evaluate_raw_case
from xiangqi_recognize_v2.types import CellPrediction, ReferencePipelineResult, Roi


class FakePipeline:
    def __init__(self, labels: list[str]) -> None:
        self.labels = labels

    def run(self, _image: Image.Image) -> ReferencePipelineResult:
        cells = []
        index = 0
        for row in range(10):
            for col in range(9):
                cells.append(
                    CellPrediction(
                        row=row,
                        col=col,
                        label=self.labels[index],
                        confidence=0.99,
                        bbox=(0, 0, 10, 10),
                    )
                )
                index += 1
        return ReferencePipelineResult(
            board_bbox=Roi(x=1, y=2, w=300, h=330),
            cells=cells,
            board_recovery_ok=True,
        )


def test_evaluate_raw_case(tmp_path: Path) -> None:
    case_dir = tmp_path / "case01"
    case_dir.mkdir()
    Image.new("RGB", (500, 600), color=(200, 180, 150)).save(case_dir / "raw.png")
    expected = {"fen": "9/9/9/9/9/9/9/9/9/4K4 w - - 0 1", "view_side": "red"}
    (case_dir / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    labels = ["empty"] * 90
    labels[9 * 9 + 4] = "rk"

    result = evaluate_raw_case(case_dir, FakePipeline(labels))
    assert result.case_name == "case01"
    assert result.board_bbox["x"] == 1
    assert result.board_recovery_ok is True
    assert result.cell_acc == 1.0
    assert result.piece_acc == 1.0
