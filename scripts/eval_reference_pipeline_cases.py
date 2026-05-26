from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xiangqi_recognize_v2.reference.eval_cases import build_reference_pipeline, evaluate_raw_case, iter_raw_eval_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 raw.xx -> board_detector.onnx -> 90格 -> piece_classifier.onnx 的整链表现。")
    parser.add_argument(
        "--board-detector",
        type=Path,
        default=ROOT / "artifacts" / "models" / "board_detector.onnx",
    )
    parser.add_argument(
        "--piece-classifier",
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

    pipeline = build_reference_pipeline(
        repo_root=ROOT,
        board_detector_path=args.board_detector,
        piece_classifier_path=args.piece_classifier,
        label_map_path=args.label_map,
    )
    case_dirs = iter_raw_eval_cases(args.eval_root)
    if not case_dirs:
        raise SystemExit(f"未找到可评估的 raw cases: {args.eval_root}")

    results = [evaluate_raw_case(case_dir, pipeline) for case_dir in case_dirs]
    summary = {
        "case_count": len(results),
        "avg_cell_acc": sum(item.cell_acc for item in results) / max(len(results), 1),
        "avg_piece_acc": sum(item.piece_acc for item in results) / max(len(results), 1),
        "board_recovery_ok_count": sum(1 for item in results if item.board_recovery_ok),
        "board_recovery_failed_cases": [item.case_name for item in results if not item.board_recovery_ok],
        "perfect_cases": [item.case_name for item in results if item.wrong_count == 0],
        "cases": [
            {
                "case": item.case_name,
                "raw": item.raw_name,
                "image_size": list(item.image_size),
                "board_bbox": item.board_bbox,
                "board_recovery_ok": item.board_recovery_ok,
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
