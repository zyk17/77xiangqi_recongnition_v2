from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from .fen_utils import board_part_to_labels


def default_layout_cfg() -> dict[str, Any]:
    return {
        "mode": "random_mixed",
        "random_mixed_fixed_prob": 0.18,
        "density_weights": {"sparse": 0.38, "normal": 0.42, "dense": 0.20},
        "sparse": {"min_pieces": 2, "max_pieces": 12},
        "normal": {"min_pieces": 10, "max_pieces": 24},
        "dense": {"min_pieces": 20, "max_pieces": 32},
        "keep_side_prob": {"red": 0.9, "black": 0.9},
    }


def merge_layout_cfg(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    base = default_layout_cfg()
    if not raw:
        return base
    out = dict(base)
    for key, value in raw.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            merged = dict(base[key])
            merged.update(dict(value))
            out[key] = merged
        else:
            out[key] = value
    return out


def _pick_density_tier(rng: random.Random, weights: Mapping[str, Any]) -> str:
    items = [(str(key), float(value)) for key, value in weights.items() if float(value) > 0.0]
    total = sum(weight for _, weight in items)
    marker = rng.random() * total
    cursor = 0.0
    for name, weight in items:
        cursor += weight
        if marker <= cursor:
            return name
    return items[-1][0]


def _full_side_piece_bag(*, red: bool) -> list[str]:
    if red:
        return ["rk", "ra", "ra", "rb", "rb", "rn", "rn", "rr", "rr", "rc", "rc"] + ["rp"] * 5
    return ["bk", "ba", "ba", "bb", "bb", "bn", "bn", "br", "br", "bc", "bc"] + ["bp"] * 5


def random_board_labels(rng: random.Random, layout_cfg: dict[str, Any]) -> list[list[str]]:
    tier = _pick_density_tier(rng, layout_cfg["density_weights"])
    piece_range = layout_cfg[tier]
    keep_cfg = layout_cfg["keep_side_prob"]
    bag: list[str] = []
    if rng.random() < float(keep_cfg.get("red", 0.9)):
        bag.extend(_full_side_piece_bag(red=True))
    if rng.random() < float(keep_cfg.get("black", 0.9)):
        bag.extend(_full_side_piece_bag(red=False))
    rng.shuffle(bag)

    target_count = rng.randint(int(piece_range["min_pieces"]), int(piece_range["max_pieces"]))
    target_count = min(target_count, len(bag), 90)
    cells = [(row, col) for row in range(10) for col in range(9)]
    rng.shuffle(cells)
    out = [["empty" for _ in range(9)] for _ in range(10)]
    for index in range(target_count):
        row, col = cells[index]
        out[row][col] = bag[index]
    return out


def sample_board_labels(
    rng: random.Random,
    fen_variants: list[str],
    layout_cfg: dict[str, Any],
) -> list[list[str]]:
    mode = str(layout_cfg.get("mode", "random_mixed"))
    if mode == "variants_only":
        return board_part_to_labels(rng.choice(fen_variants))
    if mode == "random_only":
        return random_board_labels(rng, layout_cfg)
    if rng.random() < float(layout_cfg.get("random_mixed_fixed_prob", 0.18)):
        return board_part_to_labels(rng.choice(fen_variants))
    return random_board_labels(rng, layout_cfg)
