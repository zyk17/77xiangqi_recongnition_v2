from __future__ import annotations

import json
from pathlib import Path

import onnxruntime as ort
import torch

from xiangqi_recognize_v2.board_detector.export import (
    board_detector_meta_path,
    export_board_detector_onnx,
    load_board_detector_meta,
)
from xiangqi_recognize_v2.board_detector.labels import EVIDENCE_CHANNELS, MAP_DOWNSAMPLE
from xiangqi_recognize_v2.board_detector.model import build_board_detector
from xiangqi_recognize_v2.board_detector.predict import OnnxBoardDetector


def test_board_detector_export(tmp_path: Path) -> None:
    model = build_board_detector(pretrained=False)
    checkpoint = tmp_path / "board.pt"
    image_size = 224
    torch.save(
        {
            "image_size": image_size,
            "map_downsample": MAP_DOWNSAMPLE,
            "evidence_channels": EVIDENCE_CHANNELS,
            "recovery_threshold": 0.35,
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    output = tmp_path / "board.onnx"
    export_board_detector_onnx(checkpoint_path=checkpoint, output_path=output)
    assert output.is_file()
    assert not output.with_name(f"{output.name}.data").exists()
    meta_path = board_detector_meta_path(output)
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["map_downsample"] == MAP_DOWNSAMPLE
    assert meta["recovery_threshold"] == 0.35

    runtime = load_board_detector_meta(model_path=output)
    assert runtime["map_downsample"] == MAP_DOWNSAMPLE
    assert runtime["image_size"] == image_size

    session = ort.InferenceSession(str(output))
    output_names = {item.name for item in session.get_outputs()}
    assert output_names == {"evidence_maps"}
    result = session.run(None, {"input": torch.randn(1, 3, image_size, image_size).numpy()})[0]
    assert result.shape == (1, EVIDENCE_CHANNELS, image_size // MAP_DOWNSAMPLE, image_size // MAP_DOWNSAMPLE)

    detector = OnnxBoardDetector(output)
    assert detector.map_downsample == MAP_DOWNSAMPLE
    assert detector.recovery_threshold == 0.35
