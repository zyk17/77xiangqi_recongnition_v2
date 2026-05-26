from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from ..types import Roi


@dataclass(slots=True)
class SampledCell:
    row: int
    col: int
    bbox: tuple[int, int, int, int]
    image: Image.Image


def sample_board_cells(
    image: Image.Image,
    roi: Roi,
    rows: int = 10,
    cols: int = 9,
) -> list[SampledCell]:
    cell_w = roi.w / cols
    cell_h = roi.h / rows
    cells: list[SampledCell] = []
    for row in range(rows):
        for col in range(cols):
            x1 = int(round(roi.x + col * cell_w))
            y1 = int(round(roi.y + row * cell_h))
            x2 = int(round(roi.x + (col + 1) * cell_w))
            y2 = int(round(roi.y + (row + 1) * cell_h))
            cells.append(
                SampledCell(
                    row=row,
                    col=col,
                    bbox=(x1, y1, x2, y2),
                    image=image.crop((x1, y1, x2, y2)),
                )
            )
    return cells
