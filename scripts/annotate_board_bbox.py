from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from tkinter import Canvas, Label, Tk

from PIL import Image, ImageTk

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(slots=True)
class BboxState:
    start_x: int | None = None
    start_y: int | None = None
    end_x: int | None = None
    end_y: int | None = None
    rect_id: int | None = None

    def as_xyxy(self) -> tuple[int, int, int, int] | None:
        if None in (self.start_x, self.start_y, self.end_x, self.end_y):
            return None
        x1 = min(int(self.start_x), int(self.end_x))
        y1 = min(int(self.start_y), int(self.end_y))
        x2 = max(int(self.start_x), int(self.end_x))
        y2 = max(int(self.start_y), int(self.end_y))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)


def _fit_size(width: int, height: int, max_width: int = 1400, max_height: int = 900) -> tuple[int, int, float]:
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale))), scale


def _resolve_output(raw_image: Path, output: Path | None, output_dir: Path | None) -> Path:
    if output is not None:
        return output
    if output_dir is not None:
        return output_dir / f"{raw_image.stem}.json"
    return raw_image.with_suffix(".json")


def _collect_images(target: Path, recursive: bool) -> list[Path]:
    if target.is_file():
        if target.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"不支持的图像后缀: {target}")
        return [target]
    if not target.is_dir():
        raise SystemExit(f"找不到图像或目录: {target}")
    pattern = "**/*" if recursive else "*"
    images = sorted(path for path in target.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"目录中没有图像: {target}")
    return images


def _load_existing_bbox(output_path: Path) -> tuple[int, int, int, int] | None:
    if not output_path.is_file():
        return None
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    bbox = payload.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


class BatchAnnotatorApp:
    def __init__(
        self,
        *,
        images: list[Path],
        explicit_output: Path | None,
        output_dir: Path | None,
    ) -> None:
        self.images = images
        self.explicit_output = explicit_output
        self.output_dir = output_dir
        self.index = 0
        self.original_image: Image.Image | None = None
        self.display_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.scale = 1.0
        self.output_path: Path | None = None
        self.state = BboxState()

        self.root = Tk()
        self.info_label = Label(self.root, text="")
        self.info_label.pack()
        self.canvas = Canvas(self.root, cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.root.bind("<Key>", self.on_key)

    def current_image_path(self) -> Path:
        return self.images[self.index]

    def current_output_path(self) -> Path:
        return _resolve_output(self.current_image_path(), self.explicit_output, self.output_dir)

    def run(self) -> None:
        self.load_current_image()
        self.root.mainloop()

    def load_current_image(self) -> None:
        image_path = self.current_image_path()
        self.output_path = self.current_output_path()
        self.original_image = Image.open(image_path).convert("RGB")
        display_w, display_h, self.scale = _fit_size(*self.original_image.size)
        if self.scale != 1.0:
            self.display_image = self.original_image.resize((display_w, display_h), Image.Resampling.LANCZOS)
        else:
            self.display_image = self.original_image

        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.config(width=display_w, height=display_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.root.title(f"Annotate Board BBox - {image_path.name} ({self.index + 1}/{len(self.images)})")
        self.reset(redraw=False)

        existing = _load_existing_bbox(self.output_path)
        if existing is not None:
            self.set_bbox_from_raw(existing)
        self.refresh_label()

    def refresh_label(self, extra: str | None = None) -> None:
        image_path = self.current_image_path()
        bbox = self.state.as_xyxy()
        prefix = f"[{self.index + 1}/{len(self.images)}] {image_path.name}"
        if bbox is None:
            message = f"{prefix} | 拖拽画出棋盘框。Enter 保存并下一张，S 保存，A/Left 上一张，D/Right 下一张，R 重置，Q 退出。"
        else:
            raw_bbox = [int(round(v / self.scale)) for v in bbox]
            message = f"{prefix} | display={bbox} raw={tuple(raw_bbox)}"
        if extra:
            message = f"{message} | {extra}"
        self.info_label.config(text=message)

    def set_bbox_from_raw(self, raw_bbox: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = (int(round(v * self.scale)) for v in raw_bbox)
        self.state.start_x = x1
        self.state.start_y = y1
        self.state.end_x = x2
        self.state.end_y = y2
        if self.state.rect_id is not None:
            self.canvas.delete(self.state.rect_id)
        self.state.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=3)

    def on_press(self, event) -> None:  # type: ignore[no-untyped-def]
        self.state.start_x = event.x
        self.state.start_y = event.y
        self.state.end_x = event.x
        self.state.end_y = event.y
        if self.state.rect_id is not None:
            self.canvas.delete(self.state.rect_id)
        self.state.rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=3)
        self.refresh_label()

    def on_drag(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.display_image is None or self.state.rect_id is None:
            return
        display_w, display_h = self.display_image.size
        self.state.end_x = max(0, min(display_w, event.x))
        self.state.end_y = max(0, min(display_h, event.y))
        self.canvas.coords(self.state.rect_id, self.state.start_x, self.state.start_y, self.state.end_x, self.state.end_y)
        self.refresh_label()

    def reset(self, *, redraw: bool = True) -> None:
        if self.state.rect_id is not None:
            self.canvas.delete(self.state.rect_id)
        self.state = BboxState()
        if redraw and self.tk_image is not None:
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.refresh_label()

    def save(self) -> bool:
        bbox = self.state.as_xyxy()
        if bbox is None or self.output_path is None:
            self.refresh_label("没有有效框，无法保存。")
            return False
        raw_bbox = [int(round(v / self.scale)) for v in bbox]
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps({"bbox": raw_bbox}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.refresh_label(f"已保存到 {self.output_path.name}")
        return True

    def step(self, delta: int) -> None:
        next_index = self.index + delta
        if not (0 <= next_index < len(self.images)):
            self.refresh_label("已经到边界了。")
            return
        self.index = next_index
        self.load_current_image()

    def save_and_next(self) -> None:
        if self.save():
            self.step(1)

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        key = str(event.keysym).lower()
        if key in {"q", "escape"}:
            self.root.destroy()
        elif key == "r":
            self.reset()
        elif key == "s":
            self.save()
        elif key in {"d", "right", "space"}:
            self.step(1)
        elif key in {"a", "left"}:
            self.step(-1)
        elif key in {"return", "kp_enter"}:
            self.save_and_next()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "手动框选棋盘 bbox，并保存为 JSON。支持单图或目录批量标注。"
            "拖拽画框，Enter 保存并下一张，S 保存，A/Left 上一张，D/Right 下一张，R 重置，Q 或 ESC 退出。"
        )
    )
    parser.add_argument("target", type=Path, help="单张图像路径，或包含多张图像的目录。")
    parser.add_argument("--output", type=Path, default=None, help="单图模式下显式指定输出 JSON。")
    parser.add_argument("--output-dir", type=Path, default=None, help="目录模式下统一指定 JSON 输出目录。")
    parser.add_argument("--recursive", action="store_true", help="目录模式下递归收集子目录图像。")
    args = parser.parse_args()

    images = _collect_images(args.target, args.recursive)
    if args.output is not None and len(images) > 1:
        raise SystemExit("--output 只适用于单图模式。批量标注请使用 --output-dir。")

    app = BatchAnnotatorApp(images=images, explicit_output=args.output, output_dir=args.output_dir)
    app.run()


if __name__ == "__main__":
    main()
