import hashlib
from pathlib import Path

import pytest

from safeeyes.data.checksums import sha256_of_file, verify_checksums


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"safeeyes")
    assert sha256_of_file(f) == hashlib.sha256(b"safeeyes").hexdigest()


def test_verify_checksums_passes_when_all_match(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"data")
    digest = hashlib.sha256(b"data").hexdigest()
    verify_checksums({"a.bin": digest}, tmp_path)  # must not raise


def test_verify_checksums_raises_on_mismatch(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"data")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksums({"a.bin": "0" * 64}, tmp_path)


def test_verify_checksums_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_checksums({"missing.bin": "0" * 64}, tmp_path)
