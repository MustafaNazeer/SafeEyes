"""Selective extraction of the DMD gaze archives.

Each subject archive carries face, body, hands and mosaic video, but only the
face video and the gaze annotation are ever read here. Unpacking the archives
whole would cost tens of gigabytes for material the project never opens, so
extraction filters to the two wanted members per subject.

The archives are gzip streams, so finding those members still costs a full
decompression pass per archive; the saving is disk, not time.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

WANTED_MEMBERS: tuple[str, ...] = ("*_rgb_face.mp4", "*_rgb_ann_gaze.json")
ARCHIVE_GLOB = "dmd-dataset-gaze-*.tar.gz"


def extract_archive(archive: Path, out_dir: Path) -> None:
    subprocess.run(
        ["tar", "xzf", str(archive), "-C", str(out_dir), "--wildcards", *WANTED_MEMBERS],
        check=True,
    )


def find_archives(archive_dir: Path) -> list[Path]:
    archives = sorted(archive_dir.glob(ARCHIVE_GLOB))
    if not archives:
        raise FileNotFoundError(f"no gaze archives matching {ARCHIVE_GLOB} under {archive_dir}")
    return archives


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract the DMD gaze face video and annotations.")
    parser.add_argument("--archive-dir", required=True, help="directory holding the tar.gz files")
    parser.add_argument("--out-dir", required=True, help="destination, must be gitignored")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip an archive whose annotation is already extracted",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archives = find_archives(Path(args.archive_dir))

    extracted = 0
    for index, archive in enumerate(archives, start=1):
        subject = archive.stem.replace("dmd-dataset-gaze-", "").replace(".tar", "")
        if args.skip_existing and _already_extracted(out_dir, subject):
            print(f"[{index}/{len(archives)}] {subject} already extracted, skipping", flush=True)
            continue
        print(f"[{index}/{len(archives)}] extracting {subject}", flush=True)
        extract_archive(archive, out_dir)
        extracted += 1

    print(f"extracted {extracted} of {len(archives)} archives into {out_dir}")
    return 0


def _already_extracted(out_dir: Path, subject: str) -> bool:
    group, _, number = subject.partition("-")
    candidate = out_dir / "dmd" / group / number / "s6"
    return candidate.is_dir() and any(candidate.glob("*_rgb_ann_gaze.json"))


if __name__ == "__main__":
    sys.exit(main())
