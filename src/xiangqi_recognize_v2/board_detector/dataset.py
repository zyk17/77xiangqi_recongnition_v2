from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .labels import MAP_DOWNSAMPLE, load_evidence_maps_file

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class BoardDetectionDataset:
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        *,
        image_size: int,
        map_downsample: int = MAP_DOWNSAMPLE,
        transform: object | None = None,
    ) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.map_downsample = map_downsample
        self.transform = transform
        self.samples: list[tuple[Path, Path]] = []
        self._scan()

    def _scan(self) -> None:
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"图像目录不存在: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"标注目录不存在: {self.labels_dir}")
        for image_path in sorted(self.images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = self.labels_dir / f"{image_path.stem}.json"
            if not label_path.is_file():
                raise FileNotFoundError(f"缺少标注文件: {label_path}")
            self.samples.append((image_path, label_path))
        if not self.samples:
            raise FileNotFoundError(f"未找到可训练样本: {self.images_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, torch.Tensor]:
        image_path, label_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        evidence_maps = load_evidence_maps_file(
            label_path,
            image_width=width,
            image_height=height,
            target_width=self.image_size,
            target_height=self.image_size,
            downsample=self.map_downsample,
        )
        if self.transform is not None:
            image = self.transform(image)
        return image, evidence_maps
