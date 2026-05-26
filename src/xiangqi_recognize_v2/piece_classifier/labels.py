from __future__ import annotations

import json
from pathlib import Path

EXPECTED_LABEL_COUNT = 15


def load_label_names(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data.get("labels")
    if not isinstance(labels, list) or len(labels) != EXPECTED_LABEL_COUNT:
        raise ValueError(f"label_map.json 应包含 {EXPECTED_LABEL_COUNT} 个 labels: {path}")
    return [str(label) for label in labels]
