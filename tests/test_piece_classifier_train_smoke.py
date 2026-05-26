from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

from xiangqi_recognize_v2.piece_classifier.labels import load_label_names
from xiangqi_recognize_v2.piece_classifier.train import run_piece_classifier_training


def _write_piece_dataset(root: Path, labels: list[str]) -> None:
    for split in ("train", "val"):
        for label in labels:
            target_dir = root / split / label
            target_dir.mkdir(parents=True)
            Image.new("RGB", (24, 24), color=(50, 60, 70)).save(target_dir / "sample.png")


def test_piece_classifier_train_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    labels = load_label_names(repo / "configs" / "label_map.json")
    _write_piece_dataset(tmp_path, labels)

    cfg = yaml.safe_load((repo / "configs" / "train" / "piece_classifier.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["train_dir"] = str((tmp_path / "train").resolve())
    cfg["dataset"]["val_dir"] = str((tmp_path / "val").resolve())
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = 15
    cfg["checkpoint"]["out_dir"] = str((tmp_path / "ckpt").resolve())
    cfg["runtime"]["device"] = "cpu"
    cfg["runtime"]["num_workers"] = 0

    config_path = tmp_path / "piece_classifier.yaml"
    config_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")

    best_path = run_piece_classifier_training(config_path=config_path, repo_root=repo)
    assert best_path.is_file()
    metrics = (tmp_path / "ckpt" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    row = json.loads(metrics[0])
    assert "val_acc" in row

    cfg["train"]["epochs"] = 2
    cfg["runtime"]["resume_checkpoint"] = str(best_path)
    config_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")

    resumed_path = run_piece_classifier_training(config_path=config_path, repo_root=repo)
    assert resumed_path.is_file()
    metrics = (tmp_path / "ckpt" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics) == 2
