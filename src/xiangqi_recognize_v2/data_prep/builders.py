from __future__ import annotations

import json
import random
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

from .assets import (
    discover_board_images,
    find_cross_split_piece_asset_overlaps,
    list_complete_piece_skins,
    load_board_grid_roi,
    load_piece_rgba_from_skin,
    resolve_piece_skin_subset,
)
from .layouts import merge_layout_cfg, sample_board_labels
from .render import grid_point_center, grid_roi_spacing, render_synthetic_board, sample_board_cells


def _ensure_piece_imagefolder_layout(root: Path, clear: bool) -> None:
    if clear and root.is_dir():
        shutil.rmtree(root)
    for split in ("train", "val"):
        for label in (
            "empty",
            "rk",
            "ra",
            "rb",
            "rn",
            "rr",
            "rc",
            "rp",
            "bk",
            "ba",
            "bb",
            "bn",
            "br",
            "bc",
            "bp",
        ):
            (root / split / label).mkdir(parents=True, exist_ok=True)


def _ensure_board_dataset_layout(root: Path, clear: bool) -> None:
    if clear and root.is_dir():
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)


def scan_next_global_image_index(output_root: Path) -> int:
    max_index = -1
    for path in output_root.rglob("*.png"):
        match = re.match(r"^(\d+)_", path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1 if max_index >= 0 else 0


def _build_split_skin_sets(
    all_skins: list[Path],
    *,
    piece_skin_ids: str | list[str],
    train_piece_skin_ids: str | list[str] | None,
    val_piece_skin_ids: list[str] | None,
) -> tuple[list[Path], list[Path]]:
    if val_piece_skin_ids:
        val_skins = resolve_piece_skin_subset(
            all_skins,
            list(val_piece_skin_ids),
            field_name="val_piece_skin_ids",
        )
        if train_piece_skin_ids is None:
            val_names = {path.name for path in val_skins}
            train_skins = [path for path in all_skins if path.name not in val_names]
        else:
            train_skins = resolve_piece_skin_subset(
                all_skins,
                train_piece_skin_ids,
                field_name="train_piece_skin_ids",
            )
    else:
        train_skins = resolve_piece_skin_subset(all_skins, piece_skin_ids, field_name="piece_skin_ids")
        val_skins = list(train_skins)
    if not train_skins or not val_skins:
        raise ValueError("train/val 棋子套不能为空")
    return train_skins, val_skins


def _build_skin_schedule(
    skins: list[Path],
    sample_count: int,
    *,
    random_seed: int,
    mode: str,
) -> list[Path]:
    if mode == "random":
        rng = random.Random(random_seed)
        return [rng.choice(skins) for _ in range(sample_count)]
    if mode != "balanced":
        raise ValueError("skin_sampling 应为 'balanced' 或 'random'")

    rng = random.Random(random_seed)
    schedule: list[Path] = []
    while len(schedule) < sample_count:
        cycle = list(skins)
        rng.shuffle(cycle)
        schedule.extend(cycle)
    return schedule[:sample_count]


def _render_one_sample(
    *,
    board_path: Path,
    skin_dir: Path,
    labels: list[list[str]],
    pad_fraction: float,
    inset_fraction: Sequence[float],
    piece_scale_in_cell: float,
    rng: random.Random,
    synthetic_augmentation: Mapping[str, Any] | None,
    realistic_nuisance: Mapping[str, Any] | None,
    piece_halo: Mapping[str, Any] | None,
    empty_marker_nuisance: Mapping[str, Any] | None,
    scene_placement: Mapping[str, Any] | None,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    pieces = load_piece_rgba_from_skin(skin_dir)
    board_rgb = Image.open(board_path).convert("RGB")
    grid_roi_xyxy = load_board_grid_roi(board_path)
    image, roi = render_synthetic_board(
        board_rgb=board_rgb,
        pieces_rgba=pieces,
        labels=labels,
        grid_roi_xyxy=grid_roi_xyxy,
        pad_fraction=pad_fraction,
        inset_fraction=inset_fraction,
        piece_scale_in_cell=piece_scale_in_cell,
        rng=rng,
        synthetic_augmentation=synthetic_augmentation,
        realistic_nuisance_cfg=realistic_nuisance,
        piece_halo_cfg=piece_halo,
        empty_marker_cfg=empty_marker_nuisance,
        scene_placement_cfg=scene_placement,
    )
    return image, roi.as_xyxy()


def build_piece_classifier_dataset(
    *,
    boards_root: Path,
    pieces_root: Path,
    output_root: Path,
    samples_total: int | None,
    train_fraction: float,
    random_seed: int,
    fen_variants: list[str],
    piece_skin_ids: str | list[str],
    train_piece_skin_ids: str | list[str] | None,
    val_piece_skin_ids: list[str] | None,
    pad_fraction: float,
    inset_fraction: Sequence[float],
    piece_scale_in_cell: float,
    clear_output: bool,
    layout: Mapping[str, Any] | None,
    synthetic_augmentation: Mapping[str, Any] | None,
    realistic_nuisance: Mapping[str, Any] | None,
    piece_halo: Mapping[str, Any] | None,
    empty_marker_nuisance: Mapping[str, Any] | None,
    scene_placement: Mapping[str, Any] | None,
    skin_sampling: str = "balanced",
    samples_per_skin: int | None = None,
) -> dict[str, Any]:
    boards = discover_board_images(boards_root)
    skins = list_complete_piece_skins(pieces_root)
    train_skins, val_skins = _build_split_skin_sets(
        skins,
        piece_skin_ids=piece_skin_ids,
        train_piece_skin_ids=train_piece_skin_ids,
        val_piece_skin_ids=val_piece_skin_ids,
    )
    overlaps = find_cross_split_piece_asset_overlaps(train_skins, val_skins)
    if overlaps:
        raise ValueError(f"train/val 棋子原图重复，会造成验证集泄漏: {overlaps}")
    _ensure_piece_imagefolder_layout(output_root, clear_output)
    layout_cfg = merge_layout_cfg(layout)
    if samples_per_skin is not None:
        if samples_per_skin < 1:
            raise ValueError("samples_per_skin 必须至少为 1")
        split_specs = [
            ("train", samples_per_skin * len(train_skins), train_skins),
            ("val", samples_per_skin * len(val_skins), val_skins),
        ]
    else:
        if samples_total is None:
            raise ValueError("未指定 samples_total 或 samples_per_skin")
        train_count = min(max(int(round(samples_total * train_fraction)), 1), samples_total - 1)
        split_specs = [("train", train_count, train_skins), ("val", samples_total - train_count, val_skins)]
    global_index = 0 if clear_output else scan_next_global_image_index(output_root)
    stats: dict[str, Any] = {"samples": 0, "written_cells": 0, "skin_samples": {}}

    for split_index, (split_name, sample_count, split_skins) in enumerate(split_specs):
        skin_schedule = _build_skin_schedule(
            split_skins,
            sample_count,
            random_seed=random_seed + split_index,
            mode=skin_sampling,
        )
        skin_samples = {skin.name: 0 for skin in split_skins}
        split_bar = tqdm(range(sample_count), desc=f"build {split_name} piece_dataset", leave=False, dynamic_ncols=True)
        for sample_offset in split_bar:
            rng = random.Random(random_seed + global_index * 1009 + sample_offset)
            labels = sample_board_labels(rng, fen_variants, layout_cfg)
            skin_dir = skin_schedule[sample_offset]
            image, roi_xyxy = _render_one_sample(
                board_path=rng.choice(boards),
                skin_dir=skin_dir,
                labels=labels,
                pad_fraction=pad_fraction,
                inset_fraction=inset_fraction,
                piece_scale_in_cell=piece_scale_in_cell,
                rng=rng,
                synthetic_augmentation=synthetic_augmentation,
                realistic_nuisance=realistic_nuisance,
                piece_halo=piece_halo if split_name == "train" else None,
                empty_marker_nuisance=empty_marker_nuisance if split_name == "train" else None,
                scene_placement=scene_placement,
            )
            from ..types import Roi

            roi = Roi(
                x=roi_xyxy[0],
                y=roi_xyxy[1],
                w=roi_xyxy[2] - roi_xyxy[0],
                h=roi_xyxy[3] - roi_xyxy[1],
            )
            for row, col, _, crop in sample_board_cells(image, roi):
                label = labels[row][col]
                crop.save(output_root / split_name / label / f"{global_index:08d}_{row:02d}_{col:02d}.png")
                stats["written_cells"] += 1
            stats["samples"] += 1
            skin_samples[skin_dir.name] += 1
            global_index += 1
            split_bar.set_postfix(samples=stats["samples"])
        stats["skin_samples"][split_name] = skin_samples

    manifest = {
        "task": "piece_classifier_dataset",
        "piece_skin_names": {"train": [path.name for path in train_skins], "val": [path.name for path in val_skins]},
        "layout": layout_cfg,
        "stats": stats,
    }
    (output_root / "build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_board_detector_dataset(
    *,
    boards_root: Path,
    pieces_root: Path,
    output_root: Path,
    samples_total: int,
    train_fraction: float,
    random_seed: int,
    fen_variants: list[str],
    piece_skin_ids: str | list[str],
    train_piece_skin_ids: str | list[str] | None,
    val_piece_skin_ids: list[str] | None,
    pad_fraction: float,
    inset_fraction: Sequence[float],
    piece_scale_in_cell: float,
    clear_output: bool,
    layout: Mapping[str, Any] | None,
    synthetic_augmentation: Mapping[str, Any] | None,
    realistic_nuisance: Mapping[str, Any] | None,
    scene_placement: Mapping[str, Any] | None,
) -> dict[str, Any]:
    boards = discover_board_images(boards_root)
    skins = list_complete_piece_skins(pieces_root)
    train_skins, val_skins = _build_split_skin_sets(
        skins,
        piece_skin_ids=piece_skin_ids,
        train_piece_skin_ids=train_piece_skin_ids,
        val_piece_skin_ids=val_piece_skin_ids,
    )
    overlaps = find_cross_split_piece_asset_overlaps(train_skins, val_skins)
    if overlaps:
        raise ValueError(f"train/val 棋子原图重复，会造成验证集泄漏: {overlaps}")
    _ensure_board_dataset_layout(output_root, clear_output)
    layout_cfg = merge_layout_cfg(layout)
    train_count = min(max(int(round(samples_total * train_fraction)), 1), samples_total - 1)
    split_specs = [("train", train_count, train_skins), ("val", samples_total - train_count, val_skins)]
    global_index = 0 if clear_output else scan_next_global_image_index(output_root)
    stats = {"samples": 0}

    for split_name, sample_count, split_skins in split_specs:
        split_bar = tqdm(range(sample_count), desc=f"build {split_name} board_dataset", leave=False, dynamic_ncols=True)
        for sample_offset in split_bar:
            rng = random.Random(random_seed + global_index * 1009 + sample_offset)
            labels = sample_board_labels(rng, fen_variants, layout_cfg)
            image, roi_xyxy = _render_one_sample(
                board_path=rng.choice(boards),
                skin_dir=rng.choice(split_skins),
                labels=labels,
                pad_fraction=pad_fraction,
                inset_fraction=inset_fraction,
                piece_scale_in_cell=piece_scale_in_cell,
                rng=rng,
                synthetic_augmentation=synthetic_augmentation,
                realistic_nuisance=realistic_nuisance,
                piece_halo=None,
                empty_marker_nuisance=None,
                scene_placement=scene_placement,
            )
            image_name = f"{global_index:08d}_board.png"
            label_name = f"{global_index:08d}_board.json"
            image.save(output_root / split_name / "images" / image_name)
            from ..types import Roi

            point_roi = Roi(
                x=roi_xyxy[0],
                y=roi_xyxy[1],
                w=roi_xyxy[2] - roi_xyxy[0],
                h=roi_xyxy[3] - roi_xyxy[1],
            )
            dx, dy = grid_roi_spacing(point_roi)
            point_rows: list[list[dict[str, Any]]] = []
            type_rows: list[list[str]] = []
            for row in range(10):
                point_row: list[dict[str, Any]] = []
                type_row: list[str] = []
                for col in range(9):
                    center_x, center_y = grid_point_center(point_roi, row, col)
                    piece_label = labels[row][col]
                    point_type = "cross" if piece_label == "empty" else "piece"
                    type_row.append(point_type)
                    point_row.append(
                        {
                            "row": row,
                            "col": col,
                            "x": round(center_x, 3),
                            "y": round(center_y, 3),
                            "type": point_type,
                        }
                    )
                point_rows.append(point_row)
                type_rows.append(type_row)
            (output_root / split_name / "labels" / label_name).write_text(
                json.dumps(
                    {
                        "grid_roi": list(roi_xyxy),
                        "dx": round(dx, 6),
                        "dy": round(dy, 6),
                        "point_types": type_rows,
                        "grid_points": point_rows,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            stats["samples"] += 1
            global_index += 1
            split_bar.set_postfix(samples=stats["samples"])

    manifest = {
        "task": "board_detector_dataset",
        "piece_skin_names": {"train": [path.name for path in train_skins], "val": [path.name for path in val_skins]},
        "layout": layout_cfg,
        "stats": stats,
    }
    (output_root / "build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
