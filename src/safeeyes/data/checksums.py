"""Checksum verification for downloaded dataset archives.

The hardening checklist commits to verifying dataset archives against a recorded
checksum before they are unpacked, so the provenance noted in docs/source
actually matches the bytes on disk. This module is that check.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

_CHUNK = 1 << 20


def sha256_of_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(expected: Mapping[str, str], base_dir: str | Path) -> None:
    base = Path(base_dir)
    for name, want in expected.items():
        path = base / name
        if not path.exists():
            raise FileNotFoundError(f"expected dataset file not found: {path}")
        got = sha256_of_file(path)
        if got != want:
            raise ValueError(f"checksum mismatch for {name}: expected {want}, got {got}")
