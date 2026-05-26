from __future__ import annotations

from pathlib import Path

import torch

from .model import build_piece_classifier

ONNX_INPUT_NAME = "input"
ONNX_OUTPUT_NAME = "logits"
ONNX_OPSET_VERSION = 18
BATCH_DIM_NAME = "batch"


def export_piece_classifier_onnx(
    *,
    checkpoint_path: Path,
    output_path: Path,
    dynamic_batch: bool = True,
) -> Path:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_piece_classifier(len(payload["label_names"]), pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    image_size = int(payload["image_size"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    external_data_path = output_path.with_name(f"{output_path.name}.data")
    dummy = torch.randn(1, 3, image_size, image_size)
    dynamic_shapes = None
    if dynamic_batch:
        # torch.export-based ONNX export expects forward arg names here, not ONNX tensor names.
        dynamic_shapes = {"x": {0: torch.export.Dim(BATCH_DIM_NAME)}}
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        verbose=False,
        input_names=[ONNX_INPUT_NAME],
        output_names=[ONNX_OUTPUT_NAME],
        opset_version=ONNX_OPSET_VERSION,
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        external_data=False,
    )
    if external_data_path.exists():
        external_data_path.unlink()
    return output_path.resolve()
