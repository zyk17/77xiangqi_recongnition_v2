from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from xiangqi_recognize_v2.board_detector.train import run_board_detector_training


def _write_grid_sample(image_path: Path, label_path: Path, *, width: int = 320, height: int = 256) -> None:
    image = Image.new("RGB", (width, height), color=(40, 60, 80))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 20, width - 24, height - 20), outline=(220, 180, 120), width=3)
    image.save(image_path)

    grid_points: list[list[dict[str, object]]] = []
    x0, y0 = 40.0, 32.0
    dx, dy = (width - 80) / 8.0, (height - 64) / 9.0
    for row in range(10):
        row_points: list[dict[str, object]] = []
        for col in range(9):
            row_points.append(
                {
                    "row": row,
                    "col": col,
                    "x": round(x0 + col * dx, 3),
                    "y": round(y0 + row * dy, 3),
                    "type": "piece" if (row + col) % 3 == 0 else "cross",
                }
            )
        grid_points.append(row_points)

    label_path.write_text(
        json.dumps(
            {
                "grid_roi": [int(x0), int(y0), int(x0 + 8 * dx), int(y0 + 9 * dy)],
                "grid_points": grid_points,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_board_detector_train_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    for split in ("train", "val"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        for idx in range(2):
            _write_grid_sample(
                tmp_path / split / "images" / f"sample_{idx}.png",
                tmp_path / split / "labels" / f"sample_{idx}.json",
            )

    cfg = yaml.safe_load((repo / "configs" / "train" / "board_detector.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["train_images"] = str((tmp_path / "train" / "images").resolve())
    cfg["dataset"]["train_labels"] = str((tmp_path / "train" / "labels").resolve())
    cfg["dataset"]["val_images"] = str((tmp_path / "val" / "images").resolve())
    cfg["dataset"]["val_labels"] = str((tmp_path / "val" / "labels").resolve())
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = 2
    cfg["checkpoint"]["out_dir"] = str((tmp_path / "ckpt").resolve())
    cfg["runtime"]["device"] = "cpu"
    cfg["runtime"]["num_workers"] = 0

    config_path = tmp_path / "board_detector.yaml"
    config_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")

    best_path = run_board_detector_training(config_path=config_path, repo_root=repo)
    assert best_path.is_file()
    metrics = (tmp_path / "ckpt" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    row = json.loads(metrics[0])
    assert "val_peak_recall" in row
    assert "val_grid_geom_mae" in row
    assert "val_grid_geom_mae_ok_only" in row
    assert row["best_selection"] == "val_grid_geom_mae"

    cfg["train"]["epochs"] = 2
    cfg["runtime"]["resume_checkpoint"] = str(best_path)
    config_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")

    resumed_path = run_board_detector_training(config_path=config_path, repo_root=repo)
    assert resumed_path.is_file()
    metrics = (tmp_path / "ckpt" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics) == 2
