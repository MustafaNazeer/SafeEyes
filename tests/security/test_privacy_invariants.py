"""Privacy hard-rail guards.

The public threat model promises the runtime is fully on device: no network
calls, no cloud inference, and no raw video persisted. These tests enforce that
statically over the source tree, so a future change cannot silently break the
promise. They parse each module with the ast module rather than scanning text,
so imports inside comments or docstrings do not register.

The legitimate disk writes in the project (trained model weights, extracted
feature arrays, exported ONNX graphs) are deliberately not flagged: only raw
frame writes via OpenCV count as persisted video.

The promise protects the on-device runtime. Offline dataset preparation tools
that write frames from licensed training datasets on a development machine are
exempted by name in FRAME_WRITE_ALLOWLIST; any module not listed there still
fails the scan, so a new frame writer must be deliberately allowlisted here to
land.
"""

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "safeeyes"

# Module roots that would let the runtime reach the network. None of these
# belong in an on-device computer vision pipeline.
BANNED_NETWORK_ROOTS = frozenset(
    {
        "socket",
        "ssl",
        "requests",
        "urllib",
        "http",
        "httpx",
        "aiohttp",
        "ftplib",
        "smtplib",
        "telnetlib",
        "websocket",
        "websockets",
        "grpc",
        "paramiko",
        "boto3",
    }
)

# OpenCV calls that would write a frame or a video to disk.
FRAME_WRITE_ATTRS = frozenset({"imwrite", "VideoWriter"})

# Offline dataset preparation modules allowed to write training frames. These
# never run on the device; they turn licensed dataset video into training
# images under the gitignored data tree.
FRAME_WRITE_ALLOWLIST = frozenset({"data/interval_frames.py"})


def network_imports(source: str) -> set[str]:
    """Banned network module roots imported anywhere in a source string."""
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BANNED_NETWORK_ROOTS:
                    found.add(root)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            root = node.module.split(".")[0]
            if root in BANNED_NETWORK_ROOTS:
                found.add(root)
    return found


def frame_persistence_calls(source: str) -> list[str]:
    """Raw-frame write calls (for example cv2.imwrite) in a source string."""
    tree = ast.parse(source)
    hits: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in FRAME_WRITE_ATTRS
        ):
            hits.append(node.func.attr)
    return hits


def _source_files() -> list[Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_source_tree_is_present() -> None:
    # Guard against the scan silently passing because it found no files.
    assert _source_files(), f"no source files found under {SRC_ROOT}"


def test_no_network_imports_in_source() -> None:
    offenders = {
        str(path.relative_to(SRC_ROOT)): roots
        for path in _source_files()
        if (roots := network_imports(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}, f"network imports found in on-device source: {offenders}"


def test_no_raw_frame_persistence_in_source() -> None:
    offenders = {
        rel: calls
        for path in _source_files()
        if (rel := str(path.relative_to(SRC_ROOT))) not in FRAME_WRITE_ALLOWLIST
        and (calls := frame_persistence_calls(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}, f"raw frame writes found in source: {offenders}"


def test_frame_write_allowlist_is_exact() -> None:
    # The allowlist must not accumulate stale entries or cover runtime modules.
    existing = {str(p.relative_to(SRC_ROOT)) for p in _source_files()}
    assert FRAME_WRITE_ALLOWLIST <= existing, "allowlist names a missing module"
    runtime_prefixes = ("alert/", "edge/", "observability/")
    assert not {
        entry for entry in FRAME_WRITE_ALLOWLIST if entry.startswith(runtime_prefixes)
    }, "allowlist must never cover runtime modules"


def test_network_scanner_flags_a_violation() -> None:
    # Positive control: planted network imports are caught in both forms.
    assert network_imports("import socket") == {"socket"}
    assert network_imports("from urllib.request import urlopen") == {"urllib"}
    assert network_imports("import numpy as np") == set()


def test_frame_persistence_scanner_flags_a_violation() -> None:
    # Positive control: a frame write is caught, an allowed artifact save is not.
    assert frame_persistence_calls("cv2.imwrite('f.png', frame)") == ["imwrite"]
    assert frame_persistence_calls("torch.save(model.state_dict(), out)") == []
