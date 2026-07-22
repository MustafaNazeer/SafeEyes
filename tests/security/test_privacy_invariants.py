"""Privacy hard-rail guards.

The public threat model promises the runtime is fully on device: no network
calls, no cloud inference, and no raw video persisted. These tests enforce that
statically over the source tree, so a future change cannot silently break the
promise. They parse each module with the ast module rather than scanning text,
so imports inside comments or docstrings do not register.

The legitimate disk writes in the project (trained model weights, extracted
feature arrays, exported ONNX graphs) are deliberately not flagged: only pixel
writes count as persisted video.

Persisting pixels means any route, not only an OpenCV one. The call scanner
catches cv2.imwrite and cv2.VideoWriter, which is the shape a runtime leak
would most likely take, but a module that saves a stack of frames or crops
through a numpy array write persists exactly the same pixels and is covered by
the same promise. Because an array write is indistinguishable from a legitimate
feature save at the call site, that half of the rail is enforced by the
allowlist rather than by the scanner: a module that writes dataset pixels in
any form must name itself in FRAME_WRITE_ALLOWLIST.

The promise protects the on-device runtime. Offline dataset preparation tools
that write frames from licensed training datasets on a development machine are
exempted by name in FRAME_WRITE_ALLOWLIST; any module not listed there still
fails the scan, so a new frame writer must be deliberately allowlisted here to
land. The allowlist is pinned to its exact membership below so a third entry
cannot appear without a deliberate edit.
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

# Offline dataset preparation modules allowed to write training pixels, whether
# through OpenCV or through an array write. These never run on the device; they
# turn licensed dataset video into training images and crops under the
# gitignored data and features trees.
FRAME_WRITE_ALLOWLIST = frozenset({"data/interval_frames.py", "data/yawdd_crops.py"})

# Modules that run inside the live loop and therefore must stay within the
# scanned source tree, so the network and frame-write invariants above cover
# them. Naming them explicitly means relocating one outside src/safeeyes (out
# of the rglob scan) fails here rather than silently dropping its coverage. The
# gaze and distraction paths were added after the invariants and are the reason
# this pin exists.
RUNTIME_MODULES = frozenset(
    {
        "alert/gaze_track.py",
        "alert/gaze_smoothing.py",
        "alert/hud.py",
        "alert/state_machine.py",
        "perception/gaze_features.py",
        "perception/iris.py",
        "perception/frame.py",
        "data/dmd_gaze.py",
        "distraction/scheduler.py",
        "distraction/labels.py",
        "edge/preprocess.py",
        "edge/run.py",
        "observability/observer.py",
    }
)


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


def _imported_module_names(source: str) -> set[str]:
    """Every module name a source string imports, in dotted form."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_frame_write_allowlist_is_exact() -> None:
    # The allowlist must not accumulate stale entries, must stay confined to
    # offline dataset tooling, and must only name import leaves: a module the
    # rest of the package imports is reachable from the runtime import graph
    # and may never write frames.
    existing = {str(p.relative_to(SRC_ROOT)) for p in _source_files()}
    assert FRAME_WRITE_ALLOWLIST <= existing, "allowlist names a missing module"
    assert all(entry.startswith("data/") for entry in FRAME_WRITE_ALLOWLIST), (
        "allowlist may only cover offline dataset tooling under data/"
    )
    allowlisted_modules = {
        "safeeyes." + entry.removesuffix(".py").replace("/", "."): entry
        for entry in FRAME_WRITE_ALLOWLIST
    }
    importers: dict[str, list[str]] = {}
    for path in _source_files():
        rel = str(path.relative_to(SRC_ROOT))
        if rel in FRAME_WRITE_ALLOWLIST:
            continue
        imported = _imported_module_names(path.read_text(encoding="utf-8"))
        for module in allowlisted_modules:
            if module in imported:
                importers.setdefault(module, []).append(rel)
    assert importers == {}, f"allowlisted frame writers must be import leaves: {importers}"


def test_frame_write_allowlist_is_pinned() -> None:
    # A new module writing dataset pixels, by any route including an array
    # write the call scanner cannot see, must be added here deliberately.
    assert FRAME_WRITE_ALLOWLIST == frozenset(
        {"data/interval_frames.py", "data/yawdd_crops.py"}
    )


def test_runtime_modules_are_within_the_scanned_tree() -> None:
    # Every live-loop module must sit under the scanned source root, so the
    # network and frame-write invariants cover it. A module moved out of src/
    # would leave the runtime scan without a silent gap; this catches that.
    existing = {str(p.relative_to(SRC_ROOT)) for p in _source_files()}
    missing = sorted(RUNTIME_MODULES - existing)
    assert missing == [], f"runtime modules no longer under the privacy scan: {missing}"


def test_runtime_modules_are_free_of_network_and_frame_writes() -> None:
    # A direct assertion that the specific gaze and distraction runtime modules
    # added after these invariants carry no network import and no raw frame
    # write, rather than relying only on the whole-tree scan to have reached them.
    offenders: dict[str, list[str]] = {}
    for rel in sorted(RUNTIME_MODULES):
        source = (SRC_ROOT / rel).read_text(encoding="utf-8")
        problems = sorted(network_imports(source)) + frame_persistence_calls(source)
        if problems:
            offenders[rel] = problems
    assert offenders == {}, f"runtime modules with a network or frame-write leak: {offenders}"


def test_network_scanner_flags_a_violation() -> None:
    # Positive control: planted network imports are caught in both forms.
    assert network_imports("import socket") == {"socket"}
    assert network_imports("from urllib.request import urlopen") == {"urllib"}
    assert network_imports("import numpy as np") == set()


def test_frame_persistence_scanner_flags_a_violation() -> None:
    # Positive control: a frame write is caught, an allowed artifact save is not.
    assert frame_persistence_calls("cv2.imwrite('f.png', frame)") == ["imwrite"]
    assert frame_persistence_calls("torch.save(model.state_dict(), out)") == []
