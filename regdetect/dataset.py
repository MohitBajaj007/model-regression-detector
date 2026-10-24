"""Load and validate the golden dataset."""
from __future__ import annotations

import json
from pathlib import Path

from .schemas import GoldenDataset


def load_dataset(path: str | Path) -> GoldenDataset:
    """Read the JSON file. Pydantic rejects missing fields and duplicate IDs."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GoldenDataset(**data)
