"""Read/write .acatome bundle files (gzipped JSON)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


def write_bundle(data: dict[str, Any], path: str | Path) -> Path:
    """Write a bundle dict as gzipped JSON.

    Args:
        data: Bundle dict (header + blocks + enrichment_meta).
        path: Output path (should end in .acatome).

    Returns:
        Path to written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    return path


def read_bundle(path: str | Path) -> dict[str, Any]:
    """Read a .acatome bundle file.

    Args:
        path: Path to .acatome file.

    Returns:
        Parsed bundle dict.
    """
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def update_bundle(
    path: str | Path,
    update_fn: callable,
) -> dict[str, Any]:
    """Read-modify-write a bundle file atomically.

    Args:
        path: Path to .acatome file.
        update_fn: Function that takes bundle dict, mutates it, returns it.

    Returns:
        Updated bundle dict.
    """
    path = Path(path)
    data = read_bundle(path)
    data = update_fn(data)
    write_bundle(data, path)
    return data
