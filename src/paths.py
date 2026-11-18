"""Portable repository paths with optional machine-local storage configuration."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def storage_root(override: str | Path | None = None, *, root: Path = ROOT) -> Path:
    """Resolve CLI > environment > local config > repository, relative to root."""
    value = override or os.environ.get("BOOK_STORAGE")
    config = root / "settings.local.json"
    if not value and config.is_file():
        value = json.loads(config.read_text(encoding="utf-8-sig")).get("storage")
    if value is not None and not isinstance(value, (str, Path)):
        raise ValueError("storage must be a filesystem path")
    path = Path(value or ".").expanduser()
    return (path if path.is_absolute() else root / path).resolve()
