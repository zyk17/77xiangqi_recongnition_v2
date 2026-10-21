from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from PIL import Image

CANONICAL_PIECE_CLASSES = frozenset(("rk", "ra", "rb", "rn", "rr", "rc", "rp", "bk", "ba", "bb", "bn", "br", "bc", "bp"))
IMAGE_SUFFIXES = frozenset((".png",))
BOARD_FILENAMES = ("board.png", "board.webp", "board.jpg", "board.jpeg")


def load_board_grid_roi(board_image_path: Path) -> tuple[int, int, int, int] | None:
    meta_path = board_image_path.with_name("grid_roi.json")
    if not meta_path.is_file():
        return None
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    grid_roi = raw.get("grid_roi")
    if not isinstance(grid_roi, list) or len(grid_roi) != 4:
        raise ValueError(f"grid_roi 标注非法: {meta_path}")
    x1, y1, x2, y2 = (int(value) for value in grid_roi)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"grid_roi 非法: {meta_path}")
    return x1, y1, x2, y2


def _normalize_piece_stem(stem: str) -> str | None:
    value = stem.strip().lower()
    if value in CANONICAL_PIECE_CLASSES:
        return value
    return None


def load_piece_rgba_from_skin(skin_dir: Path) -> dict[str, Image.Image]:
    out: dict[str, Image.Image] = {}
    for file_path in sorted(skin_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = _normalize_piece_stem(file_path.stem)
        if label is None:
            continue
        out[label] = Image.open(file_path).convert("RGBA")
    missing = CANONICAL_PIECE_CLASSES - set(out)
    if missing:
        raise ValueError(f"棋子套 {skin_dir} 缺少类别: {sorted(missing)}")
    return out


def list_complete_piece_skins(pieces_root: Path) -> list[Path]:
    if not pieces_root.is_dir():
        raise FileNotFoundError(f"棋子目录不存在: {pieces_root}")
    out: list[Path] = []
    for path in sorted(pieces_root.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_dir():
            continue
        try:
            load_piece_rgba_from_skin(path)
        except ValueError:
            continue
        out.append(path)
    return out


def find_cross_split_piece_asset_overlaps(
    train_skins: list[Path],
    val_skins: list[Path],
) -> dict[str, dict[str, list[str]]]:
    """Return exact source-image overlaps between train and validation skins."""
    def by_hash(skins: list[Path]) -> dict[str, dict[str, set[str]]]:
        out: dict[str, dict[str, set[str]]] = {}
        for skin_dir in skins:
            for file_path in skin_dir.iterdir():
                if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                label = _normalize_piece_stem(file_path.stem)
                if label is None:
                    continue
                digest = sha256(file_path.read_bytes()).hexdigest()
                out.setdefault(label, {}).setdefault(digest, set()).add(skin_dir.name)
        return out

    train_by_hash = by_hash(train_skins)
    val_by_hash = by_hash(val_skins)
    overlaps: dict[str, dict[str, list[str]]] = {}
    for label in sorted(CANONICAL_PIECE_CLASSES):
        shared_hashes = set(train_by_hash.get(label, {})) & set(val_by_hash.get(label, {}))
        if not shared_hashes:
            continue
        overlaps[label] = {
            "train_skins": sorted({name for digest in shared_hashes for name in train_by_hash[label][digest]}),
            "val_skins": sorted({name for digest in shared_hashes for name in val_by_hash[label][digest]}),
        }
    return overlaps


def discover_board_images(boards_root: Path) -> list[Path]:
    if not boards_root.is_dir():
        raise FileNotFoundError(f"棋盘目录不存在: {boards_root}")
    out: list[Path] = []
    for sub_dir in sorted(boards_root.iterdir(), key=lambda item: item.name.lower()):
        if not sub_dir.is_dir():
            continue
        for filename in BOARD_FILENAMES:
            path = sub_dir / filename
            if path.is_file():
                out.append(path)
                break
    return out


def resolve_piece_skin_subset(
    all_skins: list[Path],
    spec: str | list[str] | None,
    *,
    field_name: str,
) -> list[Path]:
    if spec in ("all", "all_complete", None):
        return list(all_skins)
    if isinstance(spec, list):
        by_name = {path.name: path for path in all_skins}
        out: list[Path] = []
        for skin_id in spec:
            if skin_id not in by_name:
                raise FileNotFoundError(f"{field_name} 中的棋子套 {skin_id!r} 不存在或不完整")
            out.append(by_name[skin_id])
        return out
    raise ValueError(f"{field_name} 应为 'all_complete' 或目录名列表")
