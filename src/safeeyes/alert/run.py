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
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from safeeyes.alert.hud import draw_hud
from safeeyes.alert.pipeline import DrowsinessPipeline, make_gru_classifier
from safeeyes.alert.state_machine import AlertTier
from safeeyes.observability.session import make_run_observer
from safeeyes.perception.facemesh import FaceMeshDetector
from safeeyes.perception.frame import FEATURE_COLUMNS, frame_features
from safeeyes.perception.head_pose import default_camera_matrix
from safeeyes.temporal.model import TemporalGRU


def run(
    checkpoint: str,
    camera_index: int = 0,
    window_capacity: int = 150,
    n_classes: int = 3,
    escalate_steps: int = 8,
    de_escalate_steps: int = 40,
    alarm_after: int = 45,
    log_file: str | None = None,
    metrics_interval: float = 5.0,
) -> None:
    import cv2

    n_features = len(FEATURE_COLUMNS)
    classifier = make_gru_classifier(_build_model(checkpoint, n_features, n_classes))
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
    observer, log_closer = make_run_observer(log_file=log_file, metrics_interval_s=metrics_interval)
    observer.start(
        {
            "backend": "torch",
            "checkpoint": Path(checkpoint).name,
            "camera": camera_index,
            "window": window_capacity,
            "escalate_steps": escalate_steps,
            "de_escalate_steps": de_escalate_steps,
            "alarm_after": alarm_after,
        }
    )
    last_tier = AlertTier.NONE
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            started = time.perf_counter()
            landmarks = detector.landmarks(frame)
            ear = None
            if landmarks is not None:
                features = frame_features(landmarks, default_camera_matrix(width, height))
                tier = pipeline.process(features)
                ear = float(features[0])
            else:
                tier = pipeline.current_tier
            observer.observe(
                tier=tier.name,
                fatigue=pipeline.fatigue_level,
                face_detected=landmarks is not None,
                latency_s=time.perf_counter() - started,
            )
            if tier != last_tier and tier in (AlertTier.AUDIBLE, AlertTier.ALARM):
                print("\a", end="", flush=True)  # terminal bell as a placeholder chime
            last_tier = tier
            overlay = draw_hud(frame, tier, ear=ear, fatigue_level=pipeline.fatigue_level)
            cv2.imshow("SafeEyes", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        observer.stop()
        if log_closer is not None:
            log_closer.close()
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
    parser.add_argument(
        "--log-file", default=None, help="write structured JSON logs here instead of stderr"
    )
    parser.add_argument(
        "--metrics-interval", type=float, default=5.0, help="seconds between metrics summaries"
    )
    args = parser.parse_args(argv)
    run(
        checkpoint=args.checkpoint,
        camera_index=args.camera,
        window_capacity=args.window,
        log_file=args.log_file,
        metrics_interval=args.metrics_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
