from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xiangqi_recognize_v2.board_detector.train import run_board_detector_training


def main() -> None:
    parser = argparse.ArgumentParser(description="训练棋盘定位模型（MobileNetV3-Small）")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train" / "board_detector.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument("--pretrained", action="store_true", dest="pretrained")
    pretrained_group.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.set_defaults(pretrained=None)
    parser.add_argument("--train-images", type=Path, default=None)
    parser.add_argument("--train-labels", type=Path, default=None)
    parser.add_argument("--val-images", type=Path, default=None)
    parser.add_argument("--val-labels", type=Path, default=None)
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
    if args.pretrained is not None:
        overrides.setdefault("runtime", {})["pretrained"] = args.pretrained
    if args.train_images is not None:
        overrides.setdefault("dataset", {})["train_images"] = str(args.train_images)
    if args.train_labels is not None:
        overrides.setdefault("dataset", {})["train_labels"] = str(args.train_labels)
    if args.val_images is not None:
        overrides.setdefault("dataset", {})["val_images"] = str(args.val_images)
    if args.val_labels is not None:
        overrides.setdefault("dataset", {})["val_labels"] = str(args.val_labels)
    if args.checkpoint_dir is not None:
        overrides.setdefault("checkpoint", {})["out_dir"] = str(args.checkpoint_dir)

    best_path = run_board_detector_training(
        config_path=args.config,
        repo_root=ROOT,
        overrides=overrides or None,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(best_path)


if __name__ == "__main__":
    main()
