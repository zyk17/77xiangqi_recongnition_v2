from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xiangqi_recognize_v2.data_prep import build_piece_classifier_dataset


def _abs(repo_root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (repo_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="从 raw 棋盘 + 棋子套合成棋子分类训练集。")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "datasets" / "piece_classifier_synthetic.yaml",
    )
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("配置应为 mapping")

    paths = raw.get("paths") or {}
    synthesis = raw.get("synthesis") or {}
    sampling = raw.get("sampling") or {}
    manifest = build_piece_classifier_dataset(
        boards_root=_abs(ROOT, paths.get("boards_root", "data/raw/boards")),
        pieces_root=_abs(ROOT, paths.get("pieces_root", "data/raw/pieces")),
        output_root=_abs(ROOT, paths.get("output_root", "data/processed/train_cells")),
        samples_total=(None if synthesis.get("samples_per_skin") is not None else int(synthesis.get("samples_total", 200))),
        train_fraction=float(synthesis.get("train_fraction", 0.9)),
        random_seed=int(synthesis.get("random_seed", 42)),
        fen_variants=list(synthesis.get("fen_variants") or []),
        piece_skin_ids=synthesis.get("piece_skin_ids", "all_complete"),
        train_piece_skin_ids=synthesis.get("train_piece_skin_ids"),
        val_piece_skin_ids=synthesis.get("val_piece_skin_ids"),
        pad_fraction=float(sampling.get("pad_fraction", 0.0)),
        inset_fraction=tuple(sampling.get("inset_fraction", [0.0, 0.0, 0.0, 0.0])),
        piece_scale_in_cell=float(synthesis.get("piece_scale_in_cell", 0.82)),
        clear_output=bool((raw.get("output") or {}).get("clear_before_build", True)),
        layout=raw.get("layout"),
        synthetic_augmentation=raw.get("synthetic_augmentation"),
        realistic_nuisance=raw.get("realistic_nuisance"),
        piece_halo=raw.get("piece_halo"),
        empty_marker_nuisance=raw.get("empty_marker_nuisance"),
        scene_placement=raw.get("scene_placement"),
        skin_sampling=str(synthesis.get("skin_sampling", "balanced")),
        samples_per_skin=(int(synthesis["samples_per_skin"]) if synthesis.get("samples_per_skin") is not None else None),
    )
    print(manifest)


if __name__ == "__main__":
    main()
