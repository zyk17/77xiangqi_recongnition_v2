from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xiangqi_recognize_v2.piece_classifier.eval_cases import evaluate_piece_case, iter_piece_eval_cases
from xiangqi_recognize_v2.piece_classifier.predict import OnnxPieceClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 piece_classifier.onnx 在 data/eval_cases/*/clipped.xx 上的逐格分类表现。")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "models" / "piece_classifier.onnx",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=ROOT / "configs" / "label_map.json",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=ROOT / "data" / "eval_cases",
    )
    args = parser.parse_args()

    predictor = OnnxPieceClassifier(args.model, args.label_map, image_size=48)
    case_dirs = iter_piece_eval_cases(args.eval_root)
    if not case_dirs:
        raise SystemExit(f"未找到可评估的 clipped cases: {args.eval_root}")

    results = [evaluate_piece_case(case_dir, predictor) for case_dir in case_dirs]
    summary = {
        "model": str(args.model.resolve()),
        "case_count": len(results),
        "avg_cell_acc": sum(item.cell_acc for item in results) / max(len(results), 1),
        "avg_piece_acc": sum(item.piece_acc for item in results) / max(len(results), 1),
        "cases": [
            {
                "case": item.case_name,
                "size": list(item.image_size),
                "cell_acc": item.cell_acc,
                "piece_acc": item.piece_acc,
                "wrong_count": item.wrong_count,
                "wrong_examples": [
                    {
                        "row": wrong.row,
                        "col": wrong.col,
                        "expected": wrong.expected,
                        "predicted": wrong.predicted,
                        "confidence": wrong.confidence,
                    }
                    for wrong in item.wrong_examples
                ],
            }
            for item in results
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
