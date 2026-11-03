"""Artifact IO and portable paths for frozen experimental records."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checkpoint_path(value):
    """Resolve historical absolute paths against this checkout, without changing records."""
    value = str(value)
    marker = "/results/"
    if marker in value:
        return ROOT / "results" / value.split(marker, 1)[1]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def acquisition_entry(name):
    """Map an original manifest key to the portable source-archive entry."""
    path = Path(name)
    if path.is_absolute():
        marker = "/self-editing-agents/"
        if marker not in str(path):
            raise ValueError("Unknown acquisition source path")
        path = Path(str(path).split(marker, 1)[1])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe acquisition source path")
    return str(path)
