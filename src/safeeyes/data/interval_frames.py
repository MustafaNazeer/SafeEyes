"""Frame extraction for interval labeled video manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from safeeyes.data.intervals import IntervalSample, read_interval_manifest


def interval_frame_indices(
    start_frame: int, end_frame: int, stride: int, max_frames: int | None
) -> list[int]:
    if end_frame < start_frame:
        raise ValueError(f"invalid interval: [{start_frame}, {end_frame}]")
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}")
    if max_frames is not None and max_frames < 1:
        raise ValueError(f"max_frames must be positive, got {max_frames}")
    indices = list(range(start_frame, end_frame + 1, stride))
    if indices[-1] != end_frame:
        indices.append(end_frame)
    if max_frames is not None and len(indices) > max_frames:
        if max_frames == 1:
            return [indices[len(indices) // 2]]
        span = len(indices) - 1
        picks = [round(i * span / (max_frames - 1)) for i in range(max_frames)]
        indices = [indices[p] for p in picks]
    return indices


def sanitize_sample_id(sample_id: str) -> str:
    return sample_id.replace("#", "_")


def extract_manifest_frames(
    manifest_paths: Sequence[str | Path],
    video_root: str | Path,
    out_root: str | Path,
    stride: int = 5,
    max_frames: int | None = 32,
    skip_existing: bool = True,
    progress: Callable[[str, int], None] | None = None,
) -> list[Path]:
    import cv2

    video_root = Path(video_root)
    out_root = Path(out_root)
    samples: list[IntervalSample] = []
    for manifest_path in manifest_paths:
        samples.extend(read_interval_manifest(manifest_path))
    seen: set[str] = set()
    for s in samples:
        if s.sample_id in seen:
            raise ValueError(f"duplicate sample id across manifests: {s.sample_id}")
        seen.add(s.sample_id)
    by_video: dict[str, list[IntervalSample]] = defaultdict(list)
    for s in samples:
        by_video[s.sample_id.split("#", 1)[0]].append(s)
    written: list[Path] = []
    for video_rel, video_samples in sorted(by_video.items()):
        video_path = video_root / video_rel
        if not video_path.is_file():
            raise FileNotFoundError(f"video not found: {video_path}")
        wanted: dict[int, list[Path]] = defaultdict(list)
        for s in video_samples:
            out_dir = out_root / sanitize_sample_id(s.sample_id)
            indices = interval_frame_indices(s.start_frame, s.end_frame, stride, max_frames)
            if skip_existing and out_dir.is_dir():
                if len(list(out_dir.glob("frame_*.jpg"))) == len(indices):
                    continue
            for index in indices:
                wanted[index].append(out_dir / f"frame_{index:06d}.jpg")
        if not wanted:
            continue
        capture = cv2.VideoCapture(str(video_path))
        try:
            decoded = 0
            index = 0
            remaining = dict(wanted)
            while remaining:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded += 1
                for dest in remaining.pop(index, []):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dest), frame)
                    written.append(dest)
                index += 1
            if decoded == 0:
                raise ValueError(f"no frames decoded from video: {video_path}")
            if remaining:
                unwritten = sum(len(dests) for dests in remaining.values())
                raise ValueError(
                    f"video ended after {decoded} frames with {unwritten} requested "
                    f"frames unwritten (first missing index {min(remaining)}): {video_path}"
                )
        finally:
            capture.release()
        if progress is not None:
            progress(video_rel, len(video_samples))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="extract frames for interval manifests")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args(argv)
    written = extract_manifest_frames(
        args.manifest,
        args.video_root,
        args.out_root,
        stride=args.stride,
        max_frames=args.max_frames,
        skip_existing=not args.no_skip_existing,
        progress=lambda video, n: print(f"{video}: {n} intervals"),
    )
    print(f"wrote {len(written)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
