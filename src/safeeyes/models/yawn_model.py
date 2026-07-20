"""Frozen backbone yawn event classifier.

An event proposed by the geometric MAR rule (any run of feature rows at or
above the preregistered threshold) is verified here by a small trainable head
sitting on top of a frozen ImageNet backbone. Five mouth crops sampled across
the event are each pushed through the backbone in eval mode, the per crop
features are mean and max pooled into one vector per event, and only the head
is trained on those cached vectors. This is the same freeze and cache pattern
``frozen_distraction.py`` documents for the distraction classifier: it turns a
training run into seconds of work on a four core CPU, at the cost of training
the head on non augmented, dropout free features.

The validation fold is never a file on disk. ``--train-manifest`` names the
outer 70 subject train split, and this module carves the inner train and
validation subjects from it itself with ``carve_validation``, seeded the same
as ``--seed``, so the 20 held out test subjects are never read here and the
split is reproducible without a manifest nobody writes. The trained head is
verified on the carved validation videos, a decision threshold tau is
selected there, and both the head weights and tau are saved to the checkpoint.

    python -m safeeyes.models.yawn_model \\
        --train-manifest splits/yawdd/mirror-train.csv \\
        --crop-root features/yawdd-crops \\
        --out models/yawn.pt
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from safeeyes.data.manifest import read_manifest
from safeeyes.data.splits import Sample
from safeeyes.data.yawdd import is_yawning
from safeeyes.data.yawdd_splits import carve_validation
from safeeyes.models.train_distraction import BACKBONES, build_transform, default_size
from safeeyes.models.yawn_decision import select_tau, video_scores
from safeeyes.models.yawn_events import YawnEvent, proposal_events, training_events
from safeeyes.perception.frame import FEATURE_COLUMNS
from safeeyes.temporal.yawn_validation import CROP_MARGIN_STEPS, MAR_YAWN_THRESHOLD

BACKBONE_NAME = "mobilenet_v3_small"
CROP_SIZE = 96
N_FRAMES = 5
# Bound to FEATURE_COLUMNS rather than duplicated as a bare literal: that tuple
# is the published source of truth for the feature column order, so this stays
# in lockstep with safeeyes.data.yawdd_crops without importing it directly.
# yawdd_crops.py is on the frame write allowlist and the privacy invariant
# requires it to stay an import leaf, so nothing else in the package may
# import from it, but nothing stops it from importing FEATURE_COLUMNS itself.
MAR_COLUMN = FEATURE_COLUMNS.index("mar")
# The extraction keeps crops for every row within CROP_MARGIN_STEPS of a row
# that clears the crop gate (yawdd_crops.py's margin_steps default), and every
# row inside a detection event already clears that lower gate directly. A
# nearest row substitution should therefore never need to reach further than
# that guaranteed radius; anything beyond it signals a real mismatch, not a
# rare crop_mouth failure on an adjacent frame. Bound to CROP_MARGIN_STEPS
# rather than a duplicated literal, for the same reason MAR_COLUMN is bound to
# FEATURE_COLUMNS above: yawdd_crops.py is a privacy allowlisted import leaf,
# so this module cannot import margin_steps' default from it directly, and
# instead shares CROP_MARGIN_STEPS through yawn_validation.py, which both
# modules already import.
MAX_SUBSTITUTION_ROWS = CROP_MARGIN_STEPS

EventsBuilder = Callable[[Sequence[tuple[str, str, str, np.ndarray]], float], list[YawnEvent]]


def sample_event_rows(start: int, end: int, n: int = 5) -> list[int]:
    """Pick n feature row indices evenly spread across [start, end].

    A short event, one shorter than n rows, repeats rows rather than raising,
    because a repeated row still lets the pooled feature vector see the whole
    event; it simply weights the rows already available more heavily than an
    even spread would. round() here is Python's banker's rounding, so ties at
    exactly .5 go to the nearest even integer, not always up.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if end < start:
        raise ValueError(f"end must not be before start, got start={start} end={end}")
    if n == 1:
        return [start]
    span = end - start
    return [start + round(i * span / (n - 1)) for i in range(n)]


def event_feature_vector(
    crops: np.ndarray, backbone: nn.Module, transform: Callable[[np.ndarray], torch.Tensor]
) -> np.ndarray:
    """Mean and max pool one event's crops through a frozen backbone.

    crops is stacked, transformed one at a time (the transform expects BGR,
    matching how crops come straight out of OpenCV), and forwarded through the
    backbone under no_grad in eval mode, mirroring the caching invariant in
    frozen_distraction.py: the backbone's final linear must already be an
    identity, so what comes back is the feature the real head would consume.
    The mean captures what is typical of the event; the max captures its most
    extreme frame, which for a yawn is usually the widest opening.
    """
    backbone.eval()
    batch = torch.stack([transform(crop) for crop in crops])
    with torch.no_grad():
        features = backbone(batch)
    array = features.cpu().numpy().astype(np.float32)
    mean = array.mean(axis=0)
    peak = array.max(axis=0)
    return np.concatenate([mean, peak]).astype(np.float32)


def train_yawn_head(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 0,
    class_weighted: bool = True,
) -> nn.Module:
    """Train the small trainable head on cached event feature vectors."""
    torch.manual_seed(seed)

    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    dim = int(x.shape[1])

    head = build_yawn_head(dim)

    weight: torch.Tensor | None = None
    if class_weighted:
        counts = torch.bincount(y, minlength=2).float().clamp(min=1.0)
        weight = counts.sum() / (counts.shape[0] * counts)

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    head.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(head(x), y)
        loss.backward()
        optimizer.step()

    head.eval()
    return head


def score_events(head: nn.Module, features: np.ndarray) -> np.ndarray:
    """Per event probability of being a yawn, softmax(...)[:, 1]."""
    head.eval()
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    with torch.no_grad():
        probabilities = torch.softmax(head(x), dim=1)[:, 1]
    return probabilities.cpu().numpy().astype(np.float32)


def save_yawn_checkpoint(
    path: str | Path,
    head: nn.Module,
    backbone_name: str,
    crop_size: int,
    n_frames: int,
    tau: float,
) -> None:
    checkpoint = {
        "state_dict": head.state_dict(),
        "backbone": backbone_name,
        "crop_size": crop_size,
        "n_frames": n_frames,
        "tau": tau,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)


def build_yawn_head(dim: int) -> nn.Module:
    """The head architecture, defined once so training and loading cannot drift."""
    return nn.Sequential(
        nn.Linear(dim, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 2),
    )


def load_yawn_checkpoint(path: str | Path) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the trained head and return it with the frozen metadata.

    The input dimension is read back from the saved weights rather than
    recomputed from a backbone, so a checkpoint always loads into exactly the
    shape it was saved from. The head comes back in eval mode, with the
    checkpoint's own tau in the returned metadata: nothing here may choose or
    override a decision threshold.
    """
    checkpoint = cast(dict[str, Any], torch.load(Path(path), map_location="cpu", weights_only=True))
    state_dict = checkpoint["state_dict"]
    dim = int(state_dict["0.weight"].shape[1])
    head = build_yawn_head(dim)
    head.load_state_dict(state_dict)
    head.eval()
    metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    return head, metadata


def _frozen_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Build the ImageNet backbone with its final linear replaced by identity.

    Mirrors frozen_distraction.feature_extractor: the number of classes passed
    to BACKBONES is arbitrary and discarded immediately, because the final
    linear it builds is replaced before anything is ever trained on it.
    """
    model = cast(Any, BACKBONES[name](2, pretrained))
    head = model.fc if name == "shufflenet_v2_x0_5" else model.classifier[-1]
    if not isinstance(head, nn.Linear):
        raise TypeError(f"expected the final head of {name} to be nn.Linear")
    dim = int(head.in_features)
    if name == "shufflenet_v2_x0_5":
        model.fc = nn.Identity()
    else:
        model.classifier[-1] = nn.Identity()
    model.eval()
    return cast(nn.Module, model), dim


def _load_crop_archive(crop_root: str | Path, sample_id: str) -> dict[str, np.ndarray]:
    path = (Path(crop_root) / sample_id).with_suffix(".npz")
    with np.load(path) as data:
        return {
            "features": data["features"],
            "crop_rows": data["crop_rows"],
            "crops": data["crops"],
        }


def _nearest_available_row(row: int, crop_rows: np.ndarray) -> int:
    """Nearest row with a crop, guarded against a distant, silent substitution.

    A substitution is expected only when crop_mouth failed on the requested
    row's own frame; the extraction still guarantees a crop within
    MAX_SUBSTITUTION_ROWS of any row that clears the crop gate, and every row
    inside a detection event already clears it. A nearest row further away
    than that means the event and the crop archive do not actually line up,
    so this raises instead of silently pairing the event with a distant frame.
    """
    if crop_rows.size == 0:
        raise ValueError("no crops are available for this video")
    nearest = int(crop_rows[np.argmin(np.abs(crop_rows.astype(int) - row))])
    distance = abs(nearest - row)
    if distance > MAX_SUBSTITUTION_ROWS:
        raise ValueError(
            f"nearest available crop row {nearest} is {distance} rows from requested "
            f"row {row}, exceeding the maximum acceptable substitution distance of "
            f"{MAX_SUBSTITUTION_ROWS} rows"
        )
    return nearest


def _event_crops(
    event: YawnEvent, crop_rows: np.ndarray, crops: np.ndarray, n_frames: int
) -> np.ndarray:
    """Gather the n_frames crops for one event, falling back to the nearest row.

    Every row inside an event clears the detection threshold, which sits above
    the crop gate, so its row is expected to already be in crop_rows. The
    nearest row fallback only covers the rare case where crop_mouth failed on
    a degenerate landmark box for that specific frame.
    """
    position_by_row = {int(row): position for position, row in enumerate(crop_rows)}
    selected: list[np.ndarray] = []
    for row in sample_event_rows(event.start, event.end, n=n_frames):
        actual_row = row if row in position_by_row else _nearest_available_row(row, crop_rows)
        selected.append(crops[position_by_row[actual_row]])
    return np.stack(selected)


def build_event_features(
    samples: Sequence[Sample],
    crop_root: str | Path,
    backbone: nn.Module,
    transform: Callable[[np.ndarray], torch.Tensor],
    *,
    n_frames: int = N_FRAMES,
    events_builder: EventsBuilder = proposal_events,
) -> tuple[np.ndarray, np.ndarray, list[YawnEvent]]:
    """Assemble one pooled feature vector and label per candidate event.

    ``events_builder`` decides which runs become events. The default,
    ``proposal_events``, is the label blind rule a held out or live video must
    use, because at inference time no label exists to select events with; the
    returned label array is then a placeholder the caller ignores. A caller
    that trains on labeled videos must opt into the label aware rule
    explicitly, by passing ``events_builder=training_events``, described
    below. Defaulting to the label blind rule means an omitted keyword
    argument here fails safe: it cannot silently reintroduce the label a real
    detector never sees.

    Event labels come from ``training_events``: the longest event in a
    Yawning or Talking&Yawning video is the sole positive from that video,
    every event in a Normal or Talking video is a negative, and a Yawning
    video's other events are dropped rather than guessed at. The returned
    events are in the same order as the feature and label rows, so a caller
    that also needs per video aggregation (video_scores) can zip them against
    scores without recomputing anything.
    """
    videos: list[tuple[str, str, str, np.ndarray]] = []
    archives: dict[str, dict[str, np.ndarray]] = {}
    for sample in samples:
        archive = _load_crop_archive(crop_root, sample.sample_id)
        archives[sample.sample_id] = archive
        mar = archive["features"][:, MAR_COLUMN]
        videos.append((sample.sample_id, sample.subject_id, sample.label, mar))

    events = events_builder(videos, MAR_YAWN_THRESHOLD)

    rows: list[np.ndarray] = []
    labels: list[int] = []
    for event in events:
        archive = archives[event.sample_id]
        crops = _event_crops(event, archive["crop_rows"], archive["crops"], n_frames)
        rows.append(event_feature_vector(crops, backbone, transform))
        labels.append(event.label)

    features = np.stack(rows).astype(np.float32) if rows else np.empty((0, 0), dtype=np.float32)
    return features, np.array(labels, dtype=np.int64), events


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the frozen backbone yawn event classifier.")
    parser.add_argument(
        "--train-manifest",
        required=True,
        help="the outer 70 subject train manifest; the validation fold is carved from it here",
    )
    parser.add_argument("--crop-root", required=True, help="root of per video crop npz archives")
    parser.add_argument("--out", required=True, help="checkpoint output path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-recall", type=float, default=0.90, help="tau selection recall floor")
    parser.add_argument(
        "--tau-steps",
        type=int,
        default=101,
        help="number of tau values swept between 0.0 and 1.0 inclusive",
    )
    parser.add_argument(
        "--no-class-weighting",
        action="store_true",
        help="disable inverse frequency class weighting in the head's loss",
    )
    args = parser.parse_args(argv)

    size = default_size(BACKBONE_NAME)
    backbone, _dim = _frozen_backbone(BACKBONE_NAME)
    transform = build_transform(train=False, size=size, normalize=True)

    outer_train_samples = read_manifest(args.train_manifest)
    inner_train_samples, val_samples = carve_validation(outer_train_samples, seed=args.seed)

    train_features, train_labels, _train_events = build_event_features(
        inner_train_samples,
        args.crop_root,
        backbone,
        transform,
        events_builder=training_events,
    )

    head = train_yawn_head(
        train_features,
        train_labels,
        epochs=args.epochs,
        seed=args.seed,
        class_weighted=not args.no_class_weighting,
    )

    val_features, _val_labels, val_events = build_event_features(
        val_samples,
        args.crop_root,
        backbone,
        transform,
        events_builder=training_events,
    )
    val_event_scores = score_events(head, val_features)
    val_video_scores = video_scores(val_events, val_event_scores)
    val_truths = {sample.sample_id: is_yawning(sample.label.split("&")) for sample in val_samples}
    taus = np.linspace(0.0, 1.0, args.tau_steps).tolist()
    selection = select_tau(val_video_scores, val_truths, taus, min_recall=args.min_recall)
    tau = cast(float, selection["selected"])
    rows = cast("list[dict[str, object]]", selection["rows"])

    save_yawn_checkpoint(args.out, head, BACKBONE_NAME, CROP_SIZE, N_FRAMES, tau=tau)

    positives = int(train_labels.sum())
    val_row = next(r for r in rows if r["tau"] == tau)
    summary = (
        f"trained on {len(train_labels)} events ({positives} positive) from "
        f"{len({s.subject_id for s in inner_train_samples})} inner train subjects | "
        f"selected tau={tau:.4f} on {len(val_samples)} validation videos from "
        f"{len({s.subject_id for s in val_samples})} subjects "
        f"(precision={val_row['precision']:.4f} recall={val_row['recall']:.4f} "
        f"floor_met={selection['floor_met']}) | checkpoint at {args.out}"
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
