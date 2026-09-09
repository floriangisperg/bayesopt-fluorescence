"""Run metadata helpers."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import botorch
import gpytorch
import numpy as np
import pandas as pd
import torch


def file_sha256(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_run_metadata(
    path: str | Path,
    *,
    command: str,
    config: Mapping[str, Any],
    inputs: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save JSON metadata for a training or optimization run."""
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            "torch": torch.__version__,
            "gpytorch": gpytorch.__version__,
            "botorch": botorch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "config": dict(config),
        "inputs": dict(inputs or {}),
        "extra": dict(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=str)
    return metadata
