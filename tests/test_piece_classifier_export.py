from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms

from xiangqi_recognize_v2.piece_classifier.export import export_piece_classifier_onnx
from xiangqi_recognize_v2.piece_classifier.model import build_piece_classifier
from xiangqi_recognize_v2.piece_classifier.predict import OnnxPieceClassifier


def test_piece_preprocess_resize_matches_train_resize() -> None:
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, size=(96, 80, 3), dtype=np.uint8), mode="RGB")
    classifier = OnnxPieceClassifier.__new__(OnnxPieceClassifier)
    classifier.image_size = 48
    predicted = classifier._preprocess(image)[0]
    trained = np.asarray(transforms.Resize((48, 48))(image))
    assert np.array_equal(predicted, trained)


def test_piece_classifier_forward_accepts_nhwc_uint8() -> None:
    model = build_piece_classifier(15)
    model.eval()
    images = torch.randint(0, 256, (4, 48, 48, 3), dtype=torch.uint8)
    logits = model(images)
    assert logits.shape == (4, 15)


def test_piece_classifier_export(tmp_path: Path) -> None:
    model = build_piece_classifier(15)
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
    model_input = session.get_inputs()[0]
    assert model_input.name == "input"
    assert model_input.type == "tensor(uint8)"
    assert session.get_outputs()[0].name == "logits"
    sample = np.zeros((2, 48, 48, 3), dtype=np.uint8)
    logits = session.run(None, {model_input.name: sample})[0]
    assert logits.shape == (2, 15)
