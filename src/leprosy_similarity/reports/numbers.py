"""The shared store of every number the thesis quotes: ``reports/numbers.json``.

Several report steps write to the same file, each owning its own keys (by
convention prefixed with the step's area, for example ``search.``), so a step
must merge its keys into the file rather than overwrite it. ``record_numbers``
does that under a lock, so steps run in parallel cannot lose each
other's keys.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from leprosy_similarity import paths


def _plain(value: Any) -> Any:
    """JSON-ready copy: numpy scalars become Python numbers, non-finite floats become None."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_numbers(path: Path = paths.NUMBERS) -> dict[str, Any]:
    """Every number recorded so far (empty when the file does not exist yet)."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def record_numbers(values: Mapping[str, Any], path: Path = paths.NUMBERS) -> dict[str, Any]:
    """Merge ``values`` into the store, replacing only the keys given.

    Keys are written in sorted order so the committed file diffs cleanly.
    Returns the merged store.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # The lock is held on the folder, so no lock file is left next to the store.
    folder = os.open(path.parent, os.O_RDONLY)
    try:
        fcntl.flock(folder, fcntl.LOCK_EX)
        merged = read_numbers(path)
        merged.update(_plain(dict(values)))
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        os.close(folder)
    return merged
