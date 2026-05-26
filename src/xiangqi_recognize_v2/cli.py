from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .reference.service import ReferenceVisionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("参考串联：board detector ONNX + piece classifier ONNX，输出棋盘 bbox 与 90 格分类结果。")
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--board-detector", type=Path, default="artifacts/models/board_detector.onnx")
    parser.add_argument("--piece-classifier", type=Path, default="artifacts/models/piece_classifier.onnx")
    parser.add_argument(
        "--label-map",
        type=Path,
        default="configs/label_map.json",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"找不到输入图像: {args.image}")
    image = Image.open(args.image).convert("RGB")
    pipeline = ReferenceVisionPipeline(
        board_detector_path=args.board_detector,
        piece_classifier_path=args.piece_classifier,
        label_map_path=args.label_map,
    )
    result = pipeline.run(image)
    print(
        json.dumps(
            {
                "board_bbox": {
                    "x": result.board_bbox.x,
                    "y": result.board_bbox.y,
                    "w": result.board_bbox.w,
                    "h": result.board_bbox.h,
                },
                "cells": [
                    {
                        "row": cell.row,
                        "col": cell.col,
                        "label": cell.label,
                        "confidence": cell.confidence,
                        "bbox": list(cell.bbox),
                    }
                    for cell in result.cells
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
