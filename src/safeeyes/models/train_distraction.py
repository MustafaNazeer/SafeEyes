"""Backbone registry and training CLI for the DMD distraction classifier.

Five candidate frame classifiers share one registry: four ImageNet pretrained
torchvision backbones with their heads swapped to the distraction label count,
and the from scratch reference CNN. The training loop minibatches over the frame
dataset so memory stays bounded on a laptop, and the checkpoint carries the
backbone name and label list so evaluation can rebuild the exact model.

Two correctness rails are enforced in the transform. Frames arrive from OpenCV
as BGR, so the transform converts them to RGB before anything else, because the
pretrained backbones expect RGB. And there is no horizontal or vertical flip in
the augmentation, because four of the classes are chirality sensitive (left and
right phone calls and texting), so a flip would silently corrupt the labels.

    python -m safeeyes.models.train_distraction \
        --backbone mobilenet_v3_small \
        --train-manifest splits/dmd/train.csv \
        --frames-root data/dmd/frames \
        --out models/distraction.pt
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from safeeyes.models.distraction_baselines import DistractionScratchCNN
from safeeyes.models.distraction_data import (
    DISTRACTION_LABELS,
    FrameDataset,
    frame_records,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_PRETRAINED_SIZE = 224
_SCRATCH_SIZE = 96


def _swap_head(net: nn.Module, num_classes: int) -> nn.Module:
    module = cast(Any, net)
    if hasattr(net, "fc") and isinstance(module.fc, nn.Linear):
        module.fc = nn.Linear(module.fc.in_features, num_classes)
    else:
        head = module.classifier[-1]
        module.classifier[-1] = nn.Linear(head.in_features, num_classes)
    return net


def _torchvision_builder(
    factory: Callable[..., nn.Module],
) -> Callable[[int, bool], nn.Module]:
    def build(num_classes: int, pretrained: bool) -> nn.Module:
        weights = "DEFAULT" if pretrained else None
        net = factory(weights=weights)
        return _swap_head(net, num_classes)

    return build


def _build_scratch(num_classes: int, pretrained: bool) -> nn.Module:
    return DistractionScratchCNN(num_classes)


BACKBONES: dict[str, Callable[[int, bool], nn.Module]] = {
    "mobilenet_v3_small": _torchvision_builder(models.mobilenet_v3_small),
    "mobilenet_v2": _torchvision_builder(models.mobilenet_v2),
    "efficientnet_b0": _torchvision_builder(models.efficientnet_b0),
    "shufflenet_v2_x0_5": _torchvision_builder(models.shufflenet_v2_x0_5),
    "scratch": _build_scratch,
}


def default_size(backbone: str) -> int:
    return _SCRATCH_SIZE if backbone == "scratch" else _PRETRAINED_SIZE


def head_parameters(model: nn.Module, backbone: str) -> Iterator[nn.Parameter]:
    module = cast(Any, model)
    if backbone == "scratch":
        head = module.classifier
    elif backbone == "shufflenet_v2_x0_5":
        head = module.fc
    else:
        head = module.classifier[-1]
    return iter(head.parameters())


class ImageTransform:
    def __init__(self, *, train: bool, size: int, normalize: bool) -> None:
        self.train = train
        self.size = size
        self.normalize = normalize
        self.augmentations: list[nn.Module] = []
        if train:
            self.augmentations = [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                transforms.RandomRotation(degrees=10),
            ]
        self._augment = transforms.Compose(self.augmentations)
        self._normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        resized = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float().div_(255.0)
        if self.train:
            tensor = self._augment(tensor)
        if self.normalize:
            tensor = self._normalize(tensor)
        result: torch.Tensor = tensor
        return result


def build_transform(*, train: bool, size: int, normalize: bool) -> ImageTransform:
    return ImageTransform(train=train, size=size, normalize=normalize)


def build_model_from_checkpoint(checkpoint_path: str | Path) -> tuple[nn.Module, list[str]]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    backbone = str(checkpoint["backbone"])
    labels = [str(label) for label in checkpoint["labels"]]
    model = BACKBONES[backbone](len(labels), False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, labels


def train_distraction(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    *,
    epochs: int,
    lr: float,
    backbone: str,
    freeze_backbone: bool = False,
) -> list[float]:
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in head_parameters(model, backbone):
            param.requires_grad = True
        trainable = [param for param in model.parameters() if param.requires_grad]
    else:
        trainable = list(model.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable, lr=lr)
    model.train()

    epoch_losses: list[float] = []
    for _ in range(epochs):
        running = 0.0
        seen = 0
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * int(inputs.shape[0])
            seen += int(inputs.shape[0])
        epoch_losses.append(running / seen if seen else 0.0)
    return epoch_losses


def frame_accuracy(model: nn.Module, loader: DataLoader[tuple[torch.Tensor, int]]) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            predictions = model(inputs).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
    return correct / total if total else 0.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the distraction classifier.")
    parser.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    parser.add_argument("--train-manifest", nargs="+", required=True)
    parser.add_argument("--test-manifest", nargs="+", default=None)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--out", required=True, help="checkpoint output path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    size = args.size if args.size is not None else default_size(args.backbone)
    num_classes = len(DISTRACTION_LABELS)

    train_records = frame_records(args.train_manifest, args.frames_root)
    train_ds = FrameDataset(train_records, build_transform(train=True, size=size, normalize=True))
    train_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )

    model = BACKBONES[args.backbone](num_classes, args.pretrained)
    losses = train_distraction(
        model,
        train_loader,
        epochs=args.epochs,
        lr=args.lr,
        backbone=args.backbone,
        freeze_backbone=args.freeze_backbone,
    )

    checkpoint = {
        "state_dict": model.state_dict(),
        "backbone": args.backbone,
        "labels": list(DISTRACTION_LABELS),
        "num_classes": num_classes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)

    summary = (
        f"backbone {args.backbone} | epochs {args.epochs} | "
        f"final train loss {losses[-1]:.4f} | checkpoint at {out}"
    )
    if args.test_manifest:
        test_records = frame_records(args.test_manifest, args.frames_root)
        test_ds = FrameDataset(
            test_records, build_transform(train=False, size=size, normalize=True)
        )
        test_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
            test_ds, batch_size=args.batch_size
        )
        accuracy = frame_accuracy(model, test_loader)
        summary += f" | held out frame accuracy {accuracy:.4f}"
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
