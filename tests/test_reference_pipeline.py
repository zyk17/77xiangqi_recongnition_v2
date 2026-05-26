from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from xiangqi_recognize_v2.board_detector.export import export_board_detector_onnx
from xiangqi_recognize_v2.board_detector.model import build_board_detector
from xiangqi_recognize_v2.piece_classifier.export import export_piece_classifier_onnx
from xiangqi_recognize_v2.piece_classifier.model import build_piece_classifier
from xiangqi_recognize_v2.reference.service import ReferenceVisionPipeline


def test_reference_pipeline_runs(tmp_path: Path) -> None:
    board_model = build_board_detector(pretrained=False)
    board_checkpoint = tmp_path / "board.pt"
    torch.save(
        {
            "image_size": 224,
            "map_downsample": 4,
            "evidence_channels": 2,
            "model_state_dict": board_model.state_dict(),
        },
        board_checkpoint,
    )
    board_onnx = tmp_path / "board.onnx"
    export_board_detector_onnx(checkpoint_path=board_checkpoint, output_path=board_onnx)

    piece_model = build_piece_classifier(15, pretrained=False)
    piece_checkpoint = tmp_path / "piece.pt"
    labels = [
        "empty",
        "rk",
        "ra",
        "rb",
        "rn",
        "rr",
        "rc",
        "rp",
        "bk",
        "ba",
        "bb",
        "bn",
        "br",
        "bc",
        "bp",
    ]
    torch.save(
        {
            "image_size": 48,
            "label_names": labels,
            "model_state_dict": piece_model.state_dict(),
        },
        piece_checkpoint,
    )
    piece_onnx = tmp_path / "piece.onnx"
    export_piece_classifier_onnx(checkpoint_path=piece_checkpoint, output_path=piece_onnx)

    label_map = tmp_path / "label_map.json"
    label_map.write_text(json.dumps({"labels": labels}), encoding="utf-8")

    image = Image.new("RGB", (256, 256), color=(30, 30, 30))
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 32, 224, 224), outline=(200, 160, 120), width=3)

    pipeline = ReferenceVisionPipeline(
        board_detector_path=board_onnx,
        piece_classifier_path=piece_onnx,
        label_map_path=label_map,
    )
    result = pipeline.run(image)
    assert result.board_bbox.w > 0
    assert len(result.cells) == 90
    assert hasattr(result, "board_recovery_ok")
