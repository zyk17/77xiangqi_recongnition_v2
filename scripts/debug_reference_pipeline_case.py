from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xiangqi_recognize_v2.reference.eval_cases import build_reference_pipeline, find_raw_image


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for raw in (
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    ):
        path = Path(raw)
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_overlay(image: Image.Image, payload: dict[str, object]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _load_font(14)
    board_bbox = payload["board_bbox"]
    x = int(board_bbox["x"])
    y = int(board_bbox["y"])
    w = int(board_bbox["w"])
    h = int(board_bbox["h"])
    draw.rectangle((x, y, x + w, y + h), outline=(255, 64, 64), width=4)

    for cell in payload["cells"]:
        x1, y1, x2, y2 = (int(v) for v in cell["bbox"])
        draw.rectangle((x1, y1, x2, y2), outline=(64, 200, 255), width=1)
        label = f"{cell['row']},{cell['col']}:{cell['label']}"
        draw.text((x1 + 2, y1 + 2), label, fill=(255, 255, 0), font=font)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="对单个 raw eval case 生成中间可视化：棋盘 ROI、90 格网线和预测标签。")
    parser.add_argument("case_dir", type=Path)
    parser.add_argument(
        "--board-detector",
        type=Path,
        default=ROOT / "artifacts" / "models" / "board_detector.onnx",
    )
    parser.add_argument(
        "--piece-classifier",
        type=Path,
        default=ROOT / "artifacts" / "models" / "piece_classifier.onnx",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=ROOT / "configs" / "label_map.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "debug",
    )
    args = parser.parse_args()

    pipeline = build_reference_pipeline(
        repo_root=ROOT,
        board_detector_path=args.board_detector,
        piece_classifier_path=args.piece_classifier,
        label_map_path=args.label_map,
    )
    raw_path = find_raw_image(args.case_dir)
    image = Image.open(raw_path).convert("RGB")
    result = pipeline.run(image)

    payload = {
        "case": args.case_dir.name,
        "raw": raw_path.name,
        "board_bbox": {
            "x": result.board_bbox.x,
            "y": result.board_bbox.y,
            "w": result.board_bbox.w,
            "h": result.board_bbox.h,
        },
        "board_recovery_ok": result.board_recovery_ok,
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
    }

    target_dir = args.output_dir / args.case_dir.name
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "pipeline_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    overlay = _draw_overlay(image, payload)
    overlay.save(target_dir / "overlay.png")
    board_bbox = payload["board_bbox"]
    x = int(board_bbox["x"])
    y = int(board_bbox["y"])
    w = int(board_bbox["w"])
    h = int(board_bbox["h"])
    image.crop((x, y, x + w, y + h)).save(target_dir / "board_crop.png")
    print(target_dir)


if __name__ == "__main__":
    main()
