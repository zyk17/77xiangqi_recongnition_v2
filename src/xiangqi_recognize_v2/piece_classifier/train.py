from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..shared.augmentations import build_eval_transform, build_piece_classifier_train_transform
from .dataset import OrderedPieceImageFolder
from .labels import load_label_names
from .model import build_piece_classifier


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"配置应为 mapping: {path}")
    return raw


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _ensure_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _num_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    pred = logits.argmax(dim=1)
    return int((pred == targets).sum().item())


def _make_progress(
    loader: DataLoader,
    *,
    epoch: int,
    epochs: int,
    stage: str,
) -> tqdm:
    return tqdm(loader, desc=f"epoch {epoch}/{epochs} {stage}", leave=False, dynamic_ncols=True)


def run_piece_classifier_training(
    *,
    config_path: Path,
    repo_root: Path,
    overrides: dict[str, Any] | None = None,
    resume_checkpoint: Path | None = None,
) -> Path:
    cfg = _load_yaml(config_path)
    if overrides:
        for key, value in overrides.items():
            cfg.setdefault(key, {}).update(value)

    train_cfg = cfg["train"]
    dataset_cfg = cfg["dataset"]
    checkpoint_cfg = cfg["checkpoint"]
    runtime_cfg = cfg["runtime"]

    labels = load_label_names(_ensure_path(repo_root, str(cfg["labels"]["path"])))
    image_size = int(train_cfg.get("image_size", 48))
    batch_size = int(train_cfg.get("batch_size", 128))
    epochs = int(train_cfg.get("epochs", 10))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    seed = int(runtime_cfg.get("seed", 42))
    num_workers = int(runtime_cfg.get("num_workers", 0))
    pretrained = bool(runtime_cfg.get("pretrained", False))
    device = _resolve_device(str(runtime_cfg.get("device", "auto")))

    random.seed(seed)
    torch.manual_seed(seed)

    train_set = OrderedPieceImageFolder(
        _ensure_path(repo_root, str(dataset_cfg["train_dir"])),
        labels,
        transform=build_piece_classifier_train_transform(image_size, train_cfg.get("augment")),
    )
    val_set = OrderedPieceImageFolder(
        _ensure_path(repo_root, str(dataset_cfg["val_dir"])),
        labels,
        transform=build_eval_transform(image_size),
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_piece_classifier(len(labels), pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    output_dir = _ensure_path(repo_root, str(checkpoint_cfg["out_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / str(checkpoint_cfg.get("best_name", "best.pt"))
    metrics_path = output_dir / "metrics.jsonl"

    start_epoch = 0
    best_val_acc = -1.0
    metrics_mode = "w"
    configured_resume = runtime_cfg.get("resume_checkpoint")
    if resume_checkpoint is None and configured_resume:
        resume_checkpoint = _ensure_path(repo_root, str(configured_resume))
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload.get("epoch", 0))
        best_val_acc = float(payload.get("val_acc", -1.0))
        metrics_mode = "a"

    with metrics_path.open(metrics_mode, encoding="utf-8") as handle:
        for epoch in range(start_epoch, epochs):
            start = time.perf_counter()
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_count = 0
            train_bar = _make_progress(train_loader, epoch=epoch + 1, epochs=epochs, stage="train")
            for images, targets in train_bar:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                batch_size_now = int(targets.size(0))
                train_loss_sum += float(loss.item()) * batch_size_now
                train_correct += _num_correct(logits.detach(), targets)
                train_count += batch_size_now
                train_bar.set_postfix(
                    loss=f"{train_loss_sum / max(train_count, 1):.4f}",
                    acc=f"{train_correct / max(train_count, 1):.4f}",
                )

            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_count = 0
            with torch.no_grad():
                val_bar = _make_progress(val_loader, epoch=epoch + 1, epochs=epochs, stage="val")
                for images, targets in val_bar:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    logits = model(images)
                    loss = criterion(logits, targets)
                    batch_size_now = int(targets.size(0))
                    val_loss_sum += float(loss.item()) * batch_size_now
                    val_correct += _num_correct(logits, targets)
                    val_count += batch_size_now
                    val_bar.set_postfix(
                        loss=f"{val_loss_sum / max(val_count, 1):.4f}",
                        acc=f"{val_correct / max(val_count, 1):.4f}",
                    )

            train_acc = train_correct / max(train_count, 1)
            val_acc = val_correct / max(val_count, 1)
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss_sum / max(train_count, 1),
                        "train_acc": train_acc,
                        "val_loss": val_loss_sum / max(val_count, 1),
                        "val_acc": val_acc,
                        "seconds": round(time.perf_counter() - start, 3),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "task": "piece_classifier",
                        "epoch": epoch + 1,
                        "val_acc": val_acc,
                        "image_size": image_size,
                        "label_names": labels,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    best_path,
                )

    return best_path
