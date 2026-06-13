"""Live drowsiness demo loop.

The integration entry point that drives the tested decision core from a real
camera: each frame is turned into landmarks, then a feature vector, then pushed
through the pipeline, and the resulting alert tier is drawn on screen. This is
the laptop and edge demo runner. It needs a camera and a trained temporal
checkpoint, so it is exercised live rather than in the unit tests; the logic it
orchestrates is tested in isolation.

    python -m safeeyes.alert.run --checkpoint models/temporal.pt --window 150
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch

from safeeyes.alert.hud import draw_hud
from safeeyes.alert.pipeline import DrowsinessPipeline, make_gru_classifier
from safeeyes.alert.state_machine import AlertTier
from safeeyes.perception.facemesh import FaceMeshDetector
from safeeyes.perception.frame import FEATURE_COLUMNS, frame_features
from safeeyes.perception.head_pose import default_camera_matrix
from safeeyes.temporal.model import TemporalGRU


def run(
    checkpoint: str,
    camera_index: int = 0,
    window_capacity: int = 150,
    n_classes: int = 3,
    escalate_steps: int = 5,
    de_escalate_steps: int = 15,
    alarm_after: int = 45,
) -> None:
    import cv2

    n_features = len(FEATURE_COLUMNS)
    classifier = make_gru_classifier(
        _build_model(checkpoint, n_features, n_classes)
    )
    pipeline = DrowsinessPipeline(
        classifier=classifier,
        window_capacity=window_capacity,
        n_features=n_features,
        escalate_steps=escalate_steps,
        de_escalate_steps=de_escalate_steps,
        alarm_after=alarm_after,
    )
    detector = FaceMeshDetector()
    capture = cv2.VideoCapture(camera_index)
    last_tier = AlertTier.NONE
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            landmarks = detector.landmarks(frame)
            ear = None
            if landmarks is not None:
                features = frame_features(landmarks, default_camera_matrix(width, height))
                tier = pipeline.process(features)
                ear = float(features[0])
            else:
                tier = pipeline.current_tier
            if tier != last_tier and tier in (AlertTier.AUDIBLE, AlertTier.ALARM):
                print("\a", end="", flush=True)  # terminal bell as a placeholder chime
            last_tier = tier
            overlay = draw_hud(frame, tier, ear=ear, fatigue_level=pipeline.fatigue_level)
            cv2.imshow("SafeEyes", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        detector.close()
        cv2.destroyAllWindows()


def _build_model(checkpoint: str, n_features: int, n_classes: int) -> TemporalGRU:
    model = TemporalGRU(n_features=n_features, num_classes=n_classes)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the live drowsiness demo.")
    parser.add_argument("--checkpoint", required=True, help="trained temporal model checkpoint")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window", type=int, default=150)
    args = parser.parse_args(argv)
    run(checkpoint=args.checkpoint, camera_index=args.camera, window_capacity=args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
