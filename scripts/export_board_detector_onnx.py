from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xiangqi_recognize_v2.board_detector.export import export_board_detector_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description="导出棋盘定位 ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "models" / "board_detector.onnx",
    )
    args = parser.parse_args()
    print(export_board_detector_onnx(checkpoint_path=args.checkpoint, output_path=args.output))


if __name__ == "__main__":
    main()
