from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Roi:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x2, self.y2)


@dataclass(slots=True)
class CellPrediction:
    row: int
    col: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class ReferencePipelineResult:
    board_bbox: Roi
    cells: list[CellPrediction]
    board_recovery_ok: bool = True
