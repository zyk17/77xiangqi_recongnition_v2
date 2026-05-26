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

from xiangqi_recognize_v2.board_detector.predict import OnnxBoardDetector
from xiangqi_recognize_v2.reference.eval_cases import find_raw_image, iter_raw_eval_cases


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


def _draw_full_overlay(
    image: Image.Image,
    *,
    board_bbox: tuple[int, int, int, int],
    cell_boxes: list[tuple[int, int, int, int]],
) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _load_font(14)

    x1, y1, x2, y2 = board_bbox
    draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=4)
    draw.text((x1 + 4, max(0, y1 - 18)), "ROI", fill=(255, 64, 64), font=font)

    for index, (cx1, cy1, cx2, cy2) in enumerate(cell_boxes):
        row = index // 9
        col = index % 9
        draw.rectangle((cx1, cy1, cx2, cy2), outline=(64, 200, 255), width=1)
        center_x = int(round((cx1 + cx2) * 0.5))
        center_y = int(round((cy1 + cy2) * 0.5))
        draw.ellipse((center_x - 2, center_y - 2, center_x + 2, center_y + 2), fill=(255, 220, 0))
        if row == 0 or col == 0:
            draw.text((cx1 + 1, cy1 + 1), f"{row},{col}", fill=(255, 255, 0), font=font)
    return out


def _draw_crop_overlay(
    image: Image.Image,
    *,
    board_bbox: tuple[int, int, int, int],
    cell_boxes: list[tuple[int, int, int, int]],
) -> Image.Image:
    x1, y1, x2, y2 = board_bbox
    crop = image.crop((x1, y1, x2, y2)).convert("RGB")
    draw = ImageDraw.Draw(crop)
    font = _load_font(13)

    for index, (cx1, cy1, cx2, cy2) in enumerate(cell_boxes):
        row = index // 9
        col = index % 9
        lx1 = cx1 - x1
        ly1 = cy1 - y1
        lx2 = cx2 - x1
        ly2 = cy2 - y1
        draw.rectangle((lx1, ly1, lx2, ly2), outline=(64, 200, 255), width=1)
        if row == 0 or col == 0:
            draw.text((lx1 + 1, ly1 + 1), f"{row},{col}", fill=(255, 255, 0), font=font)
    return crop


def main() -> None:
    parser = argparse.ArgumentParser(description="对真实 eval cases 输出 board ROI 与 90 格 bbox 可视化。")
    parser.add_argument(
        "--board-detector",
        type=Path,
        default=ROOT / "artifacts" / "models" / "board_detector.onnx",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=ROOT / "data" / "eval_cases",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "debug_board_detector_eval_cases",
    )
    args = parser.parse_args()

    detector = OnnxBoardDetector(args.board_detector)
    case_dirs = iter_raw_eval_cases(args.eval_root)
    if not case_dirs:
        raise SystemExit(f"未找到可评估的 raw cases: {args.eval_root}")

    for case_dir in case_dirs:
        raw_path = find_raw_image(case_dir)
        image = Image.open(raw_path).convert("RGB")
        evidence_maps = detector.predict_evidence_maps(image)
        boxes_array = detector.predict_cell_boxes(image)
        board_roi = detector.predict(image, cell_boxes=boxes_array)
        board_bbox = board_roi.as_xyxy()
        cell_boxes = [tuple(int(round(float(v))) for v in row) for row in boxes_array.tolist()]

        target_dir = args.output_dir / case_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "case": case_dir.name,
            "raw": raw_path.name,
            "image_size": list(image.size),
            "board_bbox": {
                "x": board_roi.x,
                "y": board_roi.y,
                "w": board_roi.w,
                "h": board_roi.h,
            },
            "evidence_maps_shape": list(evidence_maps.shape),
            "recovery_ok": detector.last_recovery_ok,
            "map_downsample": detector.map_downsample,
            "cell_boxes": [list(item) for item in cell_boxes],
        }
        (target_dir / "debug.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        _draw_full_overlay(image, board_bbox=board_bbox, cell_boxes=cell_boxes).save(target_dir / "overlay_full.png")
        _draw_crop_overlay(image, board_bbox=board_bbox, cell_boxes=cell_boxes).save(target_dir / "overlay_crop.png")
        image.crop(board_bbox).save(target_dir / "board_crop.png")
        print(target_dir)


if __name__ == "__main__":
    main()
