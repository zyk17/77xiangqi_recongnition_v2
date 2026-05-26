from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_eval_piece_classifier_cases_script_help() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_piece_classifier_cases.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--eval-root" in result.stdout


def test_eval_cases_readme_mentions_raw_and_clipped() -> None:
    readme = (Path(__file__).resolve().parents[1] / "data" / "eval_cases" / "README.md").read_text(encoding="utf-8")
    assert "raw.xx" in readme
    assert "clipped.xx" in readme
    assert "expected.json" in readme
