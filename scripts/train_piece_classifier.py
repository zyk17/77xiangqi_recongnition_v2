from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xiangqi_recognize_v2.piece_classifier.train import run_piece_classifier_training


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 15 类棋子分类模型。")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train" / "piece_classifier.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    overrides: dict[str, dict[str, object]] = {}
    if args.epochs is not None:
        overrides.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("train", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides.setdefault("train", {})["lr"] = args.lr
    if args.device is not None:
        overrides.setdefault("runtime", {})["device"] = args.device
    if args.train_dir is not None:
        overrides.setdefault("dataset", {})["train_dir"] = str(args.train_dir)
    if args.val_dir is not None:
        overrides.setdefault("dataset", {})["val_dir"] = str(args.val_dir)
    if args.checkpoint_dir is not None:
        overrides.setdefault("checkpoint", {})["out_dir"] = str(args.checkpoint_dir)

    best_path = run_piece_classifier_training(
        config_path=args.config,
        repo_root=ROOT,
        overrides=overrides or None,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(best_path)


if __name__ == "__main__":
    main()
