from __future__ import annotations

from pathlib import Path

from PIL import Image
from torchvision import transforms

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class OrderedPieceImageFolder:
    def __init__(
        self,
        root: Path,
        class_names: list[str],
        *,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.root = root
        self.class_names = class_names
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self._scan()

    def _scan(self) -> None:
        for class_index, class_name in enumerate(self.class_names):
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"缺少类别目录: {class_dir}")
            files = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if not files:
                raise FileNotFoundError(f"类别目录无图像: {class_dir}")
            for image_path in files:
                self.samples.append((image_path, class_index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int]:
        image_path, class_index = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, class_index
