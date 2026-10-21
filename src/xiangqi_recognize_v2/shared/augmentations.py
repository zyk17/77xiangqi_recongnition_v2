from __future__ import annotations

import io
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class OptionalImageAugment:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.color_jitter = transforms.ColorJitter(
            brightness=float(cfg.get("brightness", 0.18)),
            contrast=float(cfg.get("contrast", 0.18)),
            saturation=float(cfg.get("saturation", 0.14)),
            hue=float(cfg.get("hue", 0.03)),
        )
        blur_enabled = bool(cfg.get("gaussian_blur_enabled", False))
        blur_kernel = int(cfg.get("gaussian_blur_kernel_size", 3))
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        self.blur = transforms.GaussianBlur(
            kernel_size=blur_kernel,
            sigma=(
                float(cfg.get("gaussian_blur_sigma_min", 0.1)),
                float(cfg.get("gaussian_blur_sigma_max", 0.75)),
            ),
        )
        self.blur_enabled = blur_enabled
        self.blur_prob = float(cfg.get("gaussian_blur_prob", 0.12))
        self.jpeg_enabled = bool(cfg.get("jpeg_enabled", False))
        self.jpeg_prob = float(cfg.get("jpeg_prob", 0.18))
        self.jpeg_quality_min = int(cfg.get("jpeg_quality_min", 65))
        self.jpeg_quality_max = int(cfg.get("jpeg_quality_max", 92))
        self.noise_enabled = bool(cfg.get("noise_enabled", False))
        self.noise_prob = float(cfg.get("noise_prob", 0.32))
        self.noise_std = float(cfg.get("noise_std", 4.0))

    def __call__(self, image: Image.Image) -> Image.Image:
        if not self.enabled:
            return image

        out = self.color_jitter(image)

        if self.blur_enabled and random.random() < self.blur_prob:
            out = self.blur(out)

        if self.noise_enabled and random.random() < self.noise_prob:
            arr = np.asarray(out, dtype=np.float32)
            arr += np.random.normal(loc=0.0, scale=self.noise_std, size=arr.shape)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            out = Image.fromarray(arr, mode="RGB")

        if self.jpeg_enabled and random.random() < self.jpeg_prob:
            quality = random.randint(self.jpeg_quality_min, self.jpeg_quality_max)
            buffer = io.BytesIO()
            out.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            out = Image.open(buffer).convert("RGB")

        return out


class ToNhwcUint8:
    """棋子分类输入：HWC uint8。/255 和 NCHW 由模型完成。"""

    def __call__(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.uint8)
        return torch.tensor(array, dtype=torch.uint8)


def build_image_only_train_transform(
    image_size: int,
    augment: Mapping[str, Any] | None = None,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            OptionalImageAugment(augment),
            transforms.ToTensor(),
            IMAGENET_NORMALIZE,
        ]
    )


def build_piece_classifier_train_transform(
    image_size: int,
    augment: Mapping[str, Any] | None = None,
) -> transforms.Compose:
    cfg = augment or {}
    ops: list[Any] = [transforms.Resize((image_size, image_size))]
    if bool(cfg.get("random_affine_enabled", False)):
        translate = float(cfg.get("random_affine_translate", 0.04))
        scale_min = float(cfg.get("random_affine_scale_min", 0.96))
        scale_max = float(cfg.get("random_affine_scale_max", 1.04))
        ops.append(
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(translate, translate),
                        scale=(scale_min, scale_max),
                        fill=(124, 116, 104),
                    )
                ],
                p=float(cfg.get("random_affine_prob", 0.28)),
            )
        )
    ops.extend(
        [
            OptionalImageAugment(cfg),
            ToNhwcUint8(),
        ]
    )
    return transforms.Compose(ops)


def build_piece_classifier_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            ToNhwcUint8(),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            IMAGENET_NORMALIZE,
        ]
    )
