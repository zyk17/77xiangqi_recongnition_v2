from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .labels import EVIDENCE_CHANNELS, MAP_DOWNSAMPLE
from .model import build_board_detector

ONNX_INPUT_NAME = "input"
ONNX_OUTPUT_MAPS = "evidence_maps"
ONNX_OPSET_VERSION = 18
BATCH_DIM_NAME = "batch"
META_SUFFIX = ".meta.json"


def board_detector_meta_path(onnx_path: Path) -> Path:
    return onnx_path.with_name(f"{onnx_path.name}{META_SUFFIX}")


def write_board_detector_meta(
    *,
    output_path: Path,
    image_size: int,
    map_downsample: int,
    evidence_channels: int = EVIDENCE_CHANNELS,
    recovery_threshold: float = 0.25,
) -> Path:
    meta_path = board_detector_meta_path(output_path)
    payload = {
        "task": "board_detector",
        "image_size": int(image_size),
        "map_downsample": int(map_downsample),
        "evidence_channels": int(evidence_channels),
        "recovery_threshold": float(recovery_threshold),
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta_path


def load_board_detector_meta(
    *,
    model_path: Path,
    checkpoint_path: Path | None = None,
    image_size: int | None = None,
    map_downsample: int | None = None,
    recovery_threshold: float | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    meta_path = board_detector_meta_path(model_path)
    if meta_path.is_file():
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            meta.update(raw)

    if checkpoint_path is not None and checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            meta.setdefault("image_size", payload.get("image_size"))
            meta.setdefault("map_downsample", payload.get("map_downsample"))
            meta.setdefault("evidence_channels", payload.get("evidence_channels"))
            meta.setdefault("recovery_threshold", payload.get("recovery_threshold"))

    if image_size is not None:
        meta["image_size"] = image_size
    if map_downsample is not None:
        meta["map_downsample"] = map_downsample
    if recovery_threshold is not None:
        meta["recovery_threshold"] = recovery_threshold

    resolved_image_size = int(meta.get("image_size", 224))
    resolved_map_downsample = int(meta.get("map_downsample", MAP_DOWNSAMPLE))
    resolved_evidence_channels = int(meta.get("evidence_channels", EVIDENCE_CHANNELS))
    resolved_recovery_threshold = float(meta.get("recovery_threshold", 0.25))
    if resolved_map_downsample <= 0:
        raise ValueError(f"map_downsample 非法: {resolved_map_downsample}")

    return {
        "image_size": resolved_image_size,
        "map_downsample": resolved_map_downsample,
        "evidence_channels": resolved_evidence_channels,
        "recovery_threshold": resolved_recovery_threshold,
    }


def export_board_detector_onnx(
    *,
    checkpoint_path: Path,
    output_path: Path,
    dynamic_batch: bool = True,
    recovery_threshold: float = 0.25,
) -> Path:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    map_downsample = int(payload.get("map_downsample", MAP_DOWNSAMPLE))
    model = build_board_detector(pretrained=False, map_downsample=map_downsample)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    image_size = int(payload["image_size"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    external_data_path = output_path.with_name(f"{output_path.name}.data")
    dummy = torch.randn(1, 3, image_size, image_size)
    dynamic_shapes = None
    if dynamic_batch:
        dynamic_shapes = {"x": {0: torch.export.Dim(BATCH_DIM_NAME)}}
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        verbose=False,
        input_names=[ONNX_INPUT_NAME],
        output_names=[ONNX_OUTPUT_MAPS],
        opset_version=ONNX_OPSET_VERSION,
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        external_data=False,
    )
    if external_data_path.exists():
        external_data_path.unlink()
    write_board_detector_meta(
        output_path=output_path,
        image_size=image_size,
        map_downsample=map_downsample,
        evidence_channels=int(payload.get("evidence_channels", EVIDENCE_CHANNELS)),
        recovery_threshold=float(payload.get("recovery_threshold", recovery_threshold)),
    )
    return output_path.resolve()
