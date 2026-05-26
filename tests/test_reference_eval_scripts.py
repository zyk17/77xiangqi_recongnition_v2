from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_eval_reference_pipeline_cases_script_help() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_reference_pipeline_cases.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--board-detector" in result.stdout


def test_debug_reference_pipeline_case_script_help() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "debug_reference_pipeline_case.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--output-dir" in result.stdout
