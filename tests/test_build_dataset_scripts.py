from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


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


def _write_piece_config(config_path: Path, output_root: Path, boards_root: Path, pieces_root: Path) -> None:
    config = {
        "paths": {
            "boards_root": str(boards_root),
            "pieces_root": str(pieces_root),
            "output_root": str(output_root),
        },
        "synthesis": {
            "samples_total": 2,
            "train_fraction": 0.5,
            "random_seed": 42,
            "piece_scale_in_cell": 0.82,
            "piece_skin_ids": "all_complete",
            "train_piece_skin_ids": ["train_skin"],
            "val_piece_skin_ids": ["val_skin"],
            "fen_variants": [
                "9/9/9/9/9/9/9/9/9/9",
                "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR",
            ],
        },
        "layout": {"mode": "variants_only"},
        "synthetic_augmentation": {"enabled": False, "paste_jitter_frac": 0.0},
        "realistic_nuisance": {"enabled": False},
        "piece_halo": {"enabled": True, "prob": 0.0},
        "empty_marker_nuisance": {"enabled": True, "prob_empty_cell": 0.0},
        "scene_placement": {"enabled": True, "scene_scale_min": 1.0, "scene_scale_max": 1.0},
        "sampling": {"pad_fraction": 0.0, "inset_fraction": [0.0, 0.0, 0.0, 0.0]},
        "output": {"clear_before_build": True},
    }
    config_path.write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")


def _write_board_config(config_path: Path, output_root: Path, boards_root: Path, pieces_root: Path) -> None:
    config = {
        "paths": {
            "boards_root": str(boards_root),
            "pieces_root": str(pieces_root),
            "output_root": str(output_root),
        },
        "synthesis": {
            "samples_total": 2,
            "train_fraction": 0.5,
            "random_seed": 42,
            "piece_scale_in_cell": 0.82,
            "piece_skin_ids": "all_complete",
            "train_piece_skin_ids": ["train_skin"],
            "val_piece_skin_ids": ["val_skin"],
            "fen_variants": [
                "9/9/9/9/9/9/9/9/9/9",
                "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR",
            ],
        },
        "layout": {"mode": "variants_only"},
        "synthetic_augmentation": {"enabled": False, "paste_jitter_frac": 0.0},
        "realistic_nuisance": {"enabled": False},
        "scene_placement": {"enabled": True, "scene_scale_min": 1.0, "scene_scale_max": 1.0},
        "sampling": {"pad_fraction": 0.0, "inset_fraction": [0.0, 0.0, 0.0, 0.0]},
        "output": {"clear_before_build": True},
    }
    config_path.write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")


def test_build_piece_classifier_dataset_script(tmp_path: Path) -> None:
    boards_root = tmp_path / "boards"
    _write_board(boards_root / "theme_a")
    pieces_root = tmp_path / "pieces"
    _write_skin(pieces_root / "train_skin")
    _write_skin(pieces_root / "val_skin")
    config_path = tmp_path / "piece_dataset.yaml"
    output_root = tmp_path / "piece_out"
    _write_piece_config(config_path, output_root, boards_root, pieces_root)

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_piece_classifier_dataset.py"
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    manifest = json.loads((output_root / "build_manifest.json").read_text(encoding="utf-8"))
    assert manifest["piece_skin_names"]["val"] == ["val_skin"]
    assert result.stdout.strip()


def test_build_board_detector_dataset_script(tmp_path: Path) -> None:
    boards_root = tmp_path / "boards"
    _write_board(boards_root / "theme_a")
    pieces_root = tmp_path / "pieces"
    _write_skin(pieces_root / "train_skin")
    _write_skin(pieces_root / "val_skin")
    config_path = tmp_path / "board_dataset.yaml"
    output_root = tmp_path / "board_out"
    _write_board_config(config_path, output_root, boards_root, pieces_root)

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_board_detector_dataset.py"
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    label_files = list((output_root / "train" / "labels").glob("*.json"))
    assert label_files
    payload = json.loads(label_files[0].read_text(encoding="utf-8"))
    assert len(payload["grid_roi"]) == 4
    assert len(payload["grid_points"]) == 10
    assert all(len(row) == 9 for row in payload["grid_points"])
    assert result.stdout.strip()
