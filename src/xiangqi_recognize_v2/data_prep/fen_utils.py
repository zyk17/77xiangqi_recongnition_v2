from __future__ import annotations

LABEL_ORDER = [
    "empty",
    "rk",
    "ra",
    "rb",
    "rn",
    "rr",
    "rc",
    "rp",
    "bk",
    "ba",
    "bb",
    "bn",
    "br",
    "bc",
    "bp",
]

FEN_PIECE_TO_LABEL = {
    "K": "rk",
    "A": "ra",
    "B": "rb",
    "N": "rn",
    "R": "rr",
    "C": "rc",
    "P": "rp",
    "k": "bk",
    "a": "ba",
    "b": "bb",
    "n": "bn",
    "r": "br",
    "c": "bc",
    "p": "bp",
}


def board_part_to_labels(board_part: str) -> list[list[str]]:
    rows = board_part.strip().split("/")
    if len(rows) != 10:
        raise ValueError(f"board_part 应有 10 行，实际 {len(rows)}: {board_part!r}")
    out: list[list[str]] = []
    for segment in rows:
        cells: list[str] = []
        for ch in segment:
            if ch.isdigit():
                cells.extend(["empty"] * int(ch))
            else:
                cells.append(FEN_PIECE_TO_LABEL[ch])
        if len(cells) != 9:
            raise ValueError(f"行格数应为 9，实际 {len(cells)}: {segment!r}")
        out.append(cells)
    return out


def board_labels_to_board_part(labels: list[list[str]]) -> str:
    label_to_fen = {value: key for key, value in FEN_PIECE_TO_LABEL.items()}
    rows: list[str] = []
    for row in labels:
        empty_count = 0
        parts: list[str] = []
        for label in row:
            piece = label_to_fen.get(label, "")
            if not piece:
                empty_count += 1
                continue
            if empty_count:
                parts.append(str(empty_count))
                empty_count = 0
            parts.append(piece)
        if empty_count:
            parts.append(str(empty_count))
        rows.append("".join(parts) or "9")
    return "/".join(rows)
