from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..shared.augmentations import build_eval_transform, build_image_only_train_transform
from .dataset import BoardDetectionDataset
from .decode import batch_decoded_grid_geom_mae
from .labels import EVIDENCE_CHANNELS, MAP_DOWNSAMPLE
from .losses import board_detector_loss
from .model import build_board_detector


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


def _map_peak_recall(pred_maps: torch.Tensor, target_maps: torch.Tensor, *, threshold: float) -> float:
    """正样本区域中，预测峰值是否达到阈值（粗略验收证据图是否学到点）。"""
    target_mask = target_maps >= 0.5
    if not bool(target_mask.any()):
        return 1.0
    hit = (pred_maps >= threshold) & target_mask
    return float(hit.float().sum().item() / target_mask.float().sum().item())


def _make_progress(loader: DataLoader, *, epoch: int, epochs: int, stage: str) -> tqdm:
    return tqdm(loader, desc=f"epoch {epoch}/{epochs} {stage}", leave=False, dynamic_ncols=True)


def run_board_detector_training(
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
    loss_cfg = cfg.get("loss") or {}

    image_size = int(train_cfg.get("image_size", 224))
    map_downsample = int(train_cfg.get("map_downsample", MAP_DOWNSAMPLE))
    batch_size = int(train_cfg.get("batch_size", 64))
    epochs = int(train_cfg.get("epochs", 10))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    map_threshold = float(train_cfg.get("map_threshold", 0.35))
    seed = int(runtime_cfg.get("seed", 42))
    num_workers = int(runtime_cfg.get("num_workers", 0))
    pretrained = bool(runtime_cfg.get("pretrained", False))
    device = _resolve_device(str(runtime_cfg.get("device", "auto")))

    random.seed(seed)
    torch.manual_seed(seed)

    train_set = BoardDetectionDataset(
        _ensure_path(repo_root, str(dataset_cfg["train_images"])),
        _ensure_path(repo_root, str(dataset_cfg["train_labels"])),
        image_size=image_size,
        map_downsample=map_downsample,
        transform=build_image_only_train_transform(image_size, train_cfg.get("augment")),
    )
    val_set = BoardDetectionDataset(
        _ensure_path(repo_root, str(dataset_cfg["val_images"])),
        _ensure_path(repo_root, str(dataset_cfg["val_labels"])),
        image_size=image_size,
        map_downsample=map_downsample,
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

    model = build_board_detector(pretrained=pretrained, map_downsample=map_downsample).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    output_dir = _ensure_path(repo_root, str(checkpoint_cfg["out_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / str(checkpoint_cfg.get("best_name", "best.pt"))
    metrics_path = output_dir / "metrics.jsonl"

    start_epoch = 0
    best_val_grid_geom_mae = float("inf")
    best_val_loss_at_best_grid = float("inf")
    metrics_mode = "w"
    configured_resume = runtime_cfg.get("resume_checkpoint")
    if resume_checkpoint is None and configured_resume:
        resume_checkpoint = _ensure_path(repo_root, str(configured_resume))
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload.get("epoch", 0))
        best_val_grid_geom_mae = float(payload.get("val_grid_geom_mae", float("inf")))
        best_val_loss_at_best_grid = float(payload.get("val_loss", float("inf")))
        metrics_mode = "a"

    piece_weight = float(loss_cfg.get("piece_weight", 1.0))
    cross_weight = float(loss_cfg.get("cross_weight", 1.0))
    pos_weight = float(loss_cfg.get("pos_weight", 8.0))

    with metrics_path.open(metrics_mode, encoding="utf-8") as handle:
        for epoch in range(start_epoch, epochs):
            start = time.perf_counter()
            model.train()
            train_loss_sum = 0.0
            train_peak_recall_sum = 0.0
            train_count = 0
            train_bar = _make_progress(train_loader, epoch=epoch + 1, epochs=epochs, stage="train")
            for images, target_maps in train_bar:
                images = images.to(device, non_blocking=True)
                target_maps = target_maps.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                pred_maps = model(images)
                loss, _parts = board_detector_loss(
                    pred_maps,
                    target_maps,
                    piece_weight=piece_weight,
                    cross_weight=cross_weight,
                    pos_weight=pos_weight,
                )
                loss.backward()
                optimizer.step()
                batch_size_now = int(images.shape[0])
                train_loss_sum += float(loss.item()) * batch_size_now
                train_peak_recall_sum += (
                    _map_peak_recall(pred_maps.detach(), target_maps, threshold=map_threshold) * batch_size_now
                )
                train_count += batch_size_now
                train_bar.set_postfix(
                    loss=f"{train_loss_sum / max(train_count, 1):.4f}",
                    peak_recall=f"{train_peak_recall_sum / max(train_count, 1):.4f}",
                )

            model.eval()
            val_loss_sum = 0.0
            val_peak_recall_sum = 0.0
            val_grid_geom_mae_sum = 0.0
            val_grid_geom_mae_ok_sum = 0.0
            val_recovery_ok_sum = 0.0
            val_count = 0
            with torch.no_grad():
                val_bar = _make_progress(val_loader, epoch=epoch + 1, epochs=epochs, stage="val")
                for images, target_maps in val_bar:
                    images = images.to(device, non_blocking=True)
                    target_maps = target_maps.to(device, non_blocking=True)
                    pred_maps = model(images)
                    loss, _parts = board_detector_loss(
                        pred_maps,
                        target_maps,
                        piece_weight=piece_weight,
                        cross_weight=cross_weight,
                        pos_weight=pos_weight,
                    )
                    batch_size_now = int(images.shape[0])
                    grid_geom_mae, recovery_ok_rate, grid_geom_mae_ok = batch_decoded_grid_geom_mae(
                        pred_maps,
                        target_maps,
                        image_size=image_size,
                        map_downsample=map_downsample,
                        threshold=map_threshold,
                    )
                    val_loss_sum += float(loss.item()) * batch_size_now
                    val_peak_recall_sum += (
                        _map_peak_recall(pred_maps, target_maps, threshold=map_threshold) * batch_size_now
                    )
                    val_grid_geom_mae_sum += grid_geom_mae * batch_size_now
                    val_grid_geom_mae_ok_sum += grid_geom_mae_ok * batch_size_now
                    val_recovery_ok_sum += recovery_ok_rate * batch_size_now
                    val_count += batch_size_now
                    val_bar.set_postfix(
                        loss=f"{val_loss_sum / max(val_count, 1):.4f}",
                        grid_mae=f"{val_grid_geom_mae_sum / max(val_count, 1):.4f}",
                        peak_recall=f"{val_peak_recall_sum / max(val_count, 1):.4f}",
                    )

            val_loss = val_loss_sum / max(val_count, 1)
            val_peak_recall = val_peak_recall_sum / max(val_count, 1)
            val_grid_geom_mae = val_grid_geom_mae_sum / max(val_count, 1)
            val_grid_geom_mae_ok = val_grid_geom_mae_ok_sum / max(val_count, 1)
            val_recovery_ok_rate = val_recovery_ok_sum / max(val_count, 1)

            handle.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss_sum / max(train_count, 1),
                        "train_peak_recall": train_peak_recall_sum / max(train_count, 1),
                        "val_loss": val_loss,
                        "val_peak_recall": val_peak_recall,
                        "val_grid_geom_mae": val_grid_geom_mae,
                        "val_grid_geom_mae_ok_only": val_grid_geom_mae_ok,
                        "val_recovery_ok_rate": val_recovery_ok_rate,
                        "best_selection": "val_grid_geom_mae",
                        "map_threshold": map_threshold,
                        "seconds": round(time.perf_counter() - start, 3),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()

            grid_improved = val_grid_geom_mae < best_val_grid_geom_mae - 1e-6
            grid_tied_better_loss = (
                abs(val_grid_geom_mae - best_val_grid_geom_mae) <= 1e-6 and val_loss < best_val_loss_at_best_grid
            )
            if grid_improved or grid_tied_better_loss:
                best_val_grid_geom_mae = val_grid_geom_mae
                best_val_loss_at_best_grid = val_loss
                torch.save(
                    {
                        "task": "board_detector",
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "val_peak_recall": val_peak_recall,
                        "val_grid_geom_mae": val_grid_geom_mae,
                        "val_grid_geom_mae_ok_only": val_grid_geom_mae_ok,
                        "val_recovery_ok_rate": val_recovery_ok_rate,
                        "image_size": image_size,
                        "map_downsample": map_downsample,
                        "recovery_threshold": map_threshold,
                        "evidence_channels": EVIDENCE_CHANNELS,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    best_path,
                )

    return best_path
