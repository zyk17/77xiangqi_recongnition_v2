from __future__ import annotations

from pathlib import Path

import onnxruntime as ort
import torch

from xiangqi_recognize_v2.piece_classifier.export import export_piece_classifier_onnx
from xiangqi_recognize_v2.piece_classifier.model import build_piece_classifier


def test_piece_classifier_export(tmp_path: Path) -> None:
    model = build_piece_classifier(15, pretrained=False)
    checkpoint = tmp_path / "piece.pt"
    torch.save(
        {
            "image_size": 48,
            "label_names": [str(i) for i in range(15)],
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    output = tmp_path / "piece.onnx"
    export_piece_classifier_onnx(checkpoint_path=checkpoint, output_path=output)
    assert output.is_file()
    assert not output.with_name(f"{output.name}.data").exists()
    session = ort.InferenceSession(str(output))
    assert session.get_outputs()[0].name == "logits"
