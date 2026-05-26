from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xiangqi_recognize_v2.piece_classifier.export import export_piece_classifier_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description="导出棋子分类 ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "models" / "piece_classifier.onnx",
    )
    args = parser.parse_args()
    print(export_piece_classifier_onnx(checkpoint_path=args.checkpoint, output_path=args.output))


if __name__ == "__main__":
    main()
