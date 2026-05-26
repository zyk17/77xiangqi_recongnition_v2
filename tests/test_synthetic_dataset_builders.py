from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from xiangqi_recognize_v2.data_prep import (
    build_board_detector_dataset,
    build_piece_classifier_dataset,
)
from xiangqi_recognize_v2.data_prep.render import _draw_semi_transparent_overlay


def _write_board(board_dir: Path) -> None:
    board_dir.mkdir(parents=True)
    image = Image.new("RGB", (256, 256), color=(186, 145, 92))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 236, 236), outline=(80, 50, 30), width=3)
    image.save(board_dir / "board.png")


def _write_skin(skin_dir: Path) -> None:
    skin_dir.mkdir(parents=True)
    labels = ["rk", "ra", "rb", "rn", "rr", "rc", "rp", "bk", "ba", "bb", "bn", "br", "bc", "bp"]
    for index, label in enumerate(labels):
        image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((4, 4, 28, 28), fill=(80 + index * 5, 30, 30, 255))
        image.save(skin_dir / f"{label}.png")


def test_build_piece_classifier_dataset(tmp_path: Path) -> None:
    boards_root = tmp_path / "boards"
    _write_board(boards_root / "theme_a")
    pieces_root = tmp_path / "pieces"
    _write_skin(pieces_root / "train_skin")
    _write_skin(pieces_root / "val_skin")

    manifest = build_piece_classifier_dataset(
        boards_root=boards_root,
        pieces_root=pieces_root,
        output_root=tmp_path / "piece_out",
        meta_dir=None,
        samples_total=2,
        train_fraction=0.5,
        random_seed=42,
        fen_variants=["9/9/9/9/9/9/9/9/9/9", "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"],
        piece_skin_ids="all_complete",
        train_piece_skin_ids=["train_skin"],
        val_piece_skin_ids=["val_skin"],
        pad_fraction=0.0,
        inset_fraction=(0.0, 0.0, 0.0, 0.0),
        piece_scale_in_cell=0.82,
        clear_output=True,
        layout={"mode": "variants_only"},
        synthetic_augmentation={"enabled": False, "paste_jitter_frac": 0.0},
        realistic_nuisance={"enabled": False},
        piece_halo={"enabled": True, "prob": 0.0},
        empty_marker_nuisance={"enabled": True, "prob_empty_cell": 0.0},
        scene_placement={
            "enabled": True,
            "top_margin_frac_min": 0.2,
            "top_margin_frac_max": 0.2,
            "scene_scale_min": 1.0,
            "scene_scale_max": 1.0,
        },
    )
    assert manifest["piece_skin_names"]["val"] == ["val_skin"]
    assert any((tmp_path / "piece_out" / "train" / "empty").iterdir())


def test_build_board_detector_dataset(tmp_path: Path) -> None:
    boards_root = tmp_path / "boards"
    _write_board(boards_root / "theme_a")
    pieces_root = tmp_path / "pieces"
    _write_skin(pieces_root / "train_skin")
    _write_skin(pieces_root / "val_skin")

    manifest = build_board_detector_dataset(
        boards_root=boards_root,
        pieces_root=pieces_root,
        output_root=tmp_path / "board_out",
        meta_dir=None,
        samples_total=2,
        train_fraction=0.5,
        random_seed=42,
        fen_variants=["9/9/9/9/9/9/9/9/9/9", "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"],
        piece_skin_ids="all_complete",
        train_piece_skin_ids=["train_skin"],
        val_piece_skin_ids=["val_skin"],
        pad_fraction=0.0,
        inset_fraction=(0.0, 0.0, 0.0, 0.0),
        piece_scale_in_cell=0.82,
        clear_output=True,
        layout={"mode": "variants_only"},
        synthetic_augmentation={"enabled": False, "paste_jitter_frac": 0.0},
        realistic_nuisance={"enabled": False},
        scene_placement={
            "enabled": True,
            "left_margin_frac_min": 0.15,
            "left_margin_frac_max": 0.15,
            "top_margin_frac_min": 0.25,
            "top_margin_frac_max": 0.25,
            "right_margin_frac_min": 0.05,
            "right_margin_frac_max": 0.05,
            "bottom_margin_frac_min": 0.1,
            "bottom_margin_frac_max": 0.1,
            "wide_screen_prob": 1.0,
            "wide_aspect_min": 1.8,
            "wide_aspect_max": 1.8,
            "scene_scale_min": 1.0,
            "scene_scale_max": 1.0,
        },
    )
    image_files = list((tmp_path / "board_out" / "train" / "images").glob("*.png"))
    label_files = list((tmp_path / "board_out" / "train" / "labels").glob("*.json"))
    assert manifest["piece_skin_names"]["val"] == ["val_skin"]
    assert image_files
    assert label_files
    data = json.loads(label_files[0].read_text(encoding="utf-8"))
    image = Image.open(image_files[0])
    assert "grid_roi" in data and len(data["grid_roi"]) == 4
    assert "grid_points" in data and len(data["grid_points"]) == 10
    assert all(len(row) == 9 for row in data["grid_points"])
    assert "point_types" in data and len(data["point_types"]) == 10
    assert {point["type"] for row in data["grid_points"] for point in row} <= {"piece", "cross"}
    assert image.size[0] > 256
    assert image.size[0] > image.size[1]
    assert data["grid_roi"][0] > 0
    assert data["grid_roi"][1] > 0


def test_build_board_detector_dataset_rich_scene(tmp_path: Path) -> None:
    boards_root = tmp_path / "boards"
    _write_board(boards_root / "theme_a")
    pieces_root = tmp_path / "pieces"
    _write_skin(pieces_root / "train_skin")
    _write_skin(pieces_root / "val_skin")

    build_board_detector_dataset(
        boards_root=boards_root,
        pieces_root=pieces_root,
        output_root=tmp_path / "board_out",
        meta_dir=None,
        samples_total=4,
        train_fraction=0.75,
        random_seed=7,
        fen_variants=["9/9/9/9/9/9/9/9/9/9"],
        piece_skin_ids="all_complete",
        train_piece_skin_ids=["train_skin"],
        val_piece_skin_ids=["val_skin"],
        pad_fraction=0.0,
        inset_fraction=(0.0, 0.0, 0.0, 0.0),
        piece_scale_in_cell=0.82,
        clear_output=True,
        layout={"mode": "variants_only"},
        synthetic_augmentation={"enabled": False, "paste_jitter_frac": 0.0},
        realistic_nuisance={"enabled": False},
        scene_placement={
            "enabled": True,
            "clean_scene_prob": 0.0,
            "top_margin_frac_min": 0.2,
            "top_margin_frac_max": 0.35,
            "bottom_margin_frac_min": 0.18,
            "bottom_margin_frac_max": 0.35,
            "wide_screen_prob": 1.0,
            "wide_aspect_min": 1.9,
            "wide_aspect_max": 2.1,
            "background_texture_prob": 1.0,
            "background_photo_blobs_prob": 1.0,
            "portrait_prob": 1.0,
            "icon_cluster_prob": 1.0,
            "semi_transparent_overlay_prob": 0.5,
            "top_bar_prob": 1.0,
            "bottom_panel_prob": 1.0,
            "side_panel_prob": 1.0,
            "text_nuisance_prob": 1.0,
            "secondary_board_prob": 0.0,
            "board_occluder_prob": 0.0,
            "scene_scale_min": 1.0,
            "scene_scale_max": 1.0,
        },
    )
    image_path = next((tmp_path / "board_out" / "train" / "images").glob("*.png"))
    label_path = tmp_path / "board_out" / "train" / "labels" / f"{image_path.stem}.json"
    data = json.loads(label_path.read_text(encoding="utf-8"))
    image = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = (int(v) for v in data["grid_roi"])
    arr = np.asarray(image, dtype=np.int16)
    outside = np.concatenate(
        [
            arr[: max(y1, 1), :, :].reshape(-1, 3),
            arr[y2:, :, :].reshape(-1, 3),
            arr[:, : max(x1, 1), :].reshape(-1, 3),
            arr[:, x2:, :].reshape(-1, 3),
        ]
    )
    assert outside.shape[0] > 1000
    assert float(outside.std()) > 8.0


def test_semi_transparent_overlay_preserves_board_region() -> None:
    scene = Image.new("RGB", (240, 240), color=(120, 130, 140))
    board_bbox = (70, 70, 170, 170)
    rng = random.Random(7)
    out = _draw_semi_transparent_overlay(
        scene,
        board_bbox,
        rng,
        {
            "semi_transparent_overlay_prob": 1.0,
            "overlay_alpha_min": 80,
            "overlay_alpha_max": 120,
            "board_safe_padding": 4,
        },
    )
    before = np.asarray(scene)[80:160, 80:160]
    after = np.asarray(out)[80:160, 80:160]
    assert np.array_equal(before, after)
