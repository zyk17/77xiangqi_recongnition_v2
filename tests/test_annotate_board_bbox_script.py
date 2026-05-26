from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.annotate_board_bbox import _collect_images


def test_annotate_board_bbox_script_help() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "annotate_board_bbox.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--output" in result.stdout
    assert "--output-dir" in result.stdout


def test_collect_images_from_directory(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a.png").write_bytes(b"fake")
    (images_dir / "b.jpg").write_bytes(b"fake")
    (images_dir / "note.txt").write_text("ignore", encoding="utf-8")

    images = _collect_images(images_dir, recursive=False)

    assert [path.name for path in images] == ["a.png", "b.jpg"]
