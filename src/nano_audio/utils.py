"""Common utility functions for hashing, atomic JSON serialization, and git repository inspection.

Provides:
  - SHA-256 file hashing (chunked for memory efficiency).
  - Deterministic object digestion for experiment configuration fingerprinting.
  - Atomic JSON and JSON Lines (.jsonl) read/write operations to prevent corruptions during concurrent access.
  - Git revision inspection for experiment reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

# Absolute path to the repository root directory
ROOT = Path(__file__).resolve().parents[2]

# Root directory for storing checkpoints, datasets, features, and experiment artifacts.
# Can be overridden via the AUDIO2TOOL_ARTIFACTS environment variable.
ARTIFACTS = Path(os.environ.get("AUDIO2TOOL_ARTIFACTS", ROOT / "artifacts")).resolve()


def sha(path: Path | str) -> str:
    """Compute the SHA-256 hexadecimal hash digest of a file.
    
    Reads in 8MB chunks to handle large model weights and audio archives efficiently without memory spikes.
    """
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest(obj: object) -> str:
    """Compute a deterministic SHA-256 digest of any JSON-serializable Python object.
    
    Sorts dictionary keys to guarantee that identical configurations always yield identical digests.
    """
    encoded = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path | str, obj: object) -> None:
    """Atomically write a JSON object to disk using a temporary file.
    
    Writing to a temporary file first and renaming it guarantees that readers never
    observe incomplete or corrupted writes if the process is interrupted.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)


def read_json(path: Path | str) -> dict | list:
    """Read and parse a JSON document from disk."""
    return json.loads(Path(path).read_text())


def append_json(path: Path | str, obj: object) -> None:
    """Append a single JSON object as a newline-delimited entry to a JSON Lines (.jsonl) ledger."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(obj, default=str) + "\n")


def gitrev(path: Path | str) -> str:
    """Query and return the current HEAD commit hash of a git repository directory."""
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
