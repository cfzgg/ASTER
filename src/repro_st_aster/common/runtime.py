from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def file_report(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return f"{path} [missing]"
    size_mb = path.stat().st_size / 1024 / 1024
    return f"{path} ({size_mb:.2f} MB)"


def find_repo_root(start: Optional[Path] = None, markers: Optional[Iterable[str]] = None) -> Path:
    start = (start or Path.cwd()).resolve()
    expected = tuple(markers or ("README.md", "requirements.txt", "src", "raw_data", "preprocess_data"))
    for candidate in (start, *start.parents):
        if all((candidate / marker).exists() for marker in expected):
            return candidate
    raise FileNotFoundError(f"Could not locate repository root from {start}.")
