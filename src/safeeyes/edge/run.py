"""Live drowsiness demo on the edge runtime.

The same demo loop as the development runner, but driven by an ONNX model through
ONNX Runtime instead of a PyTorch checkpoint, so it runs on the Raspberry Pi with
the minimal dependency set (no PyTorch). Point it at an exported temporal model,
ideally the int8 one:

    python -m safeeyes.edge.run --model models/edge/temporal.int8.onnx

The classifier construction is a tested seam; the camera capture, landmark
detection, and overlay are integration glue exercised live with a real camera.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from safeeyes.alert.hud import draw_hud
from safeeyes.alert.pipeline import DrowsinessPipeline
from safeeyes.alert.state_machine import AlertTier
from safeeyes.edge.runtime import OnnxModel, make_onnx_window_classifier
from safeeyes.observability.session import make_run_observer
from safeeyes.perception.facemesh import FaceMeshDetector
from safeeyes.perception.frame import FEATURE_COLUMNS, frame_features
from safeeyes.perception.head_pose import default_camera_matrix


def build_onnx_classifier(model_path: str | Path) -> Callable[[np.ndarray], int]:
    return make_onnx_window_classifier(OnnxModel(model_path))


def run(
    model_path: str | Path,
    camera_index: int = 0,
    window_capacity: int = 150,
    escalate_steps: int = 5,
    de_escalate_steps: int = 15,
    alarm_after: int = 45,
    log_file: str | None = None,
    metrics_interval: float = 5.0,
    show_display: bool = True,
) -> None:
    import cv2

    n_features = len(FEATURE_COLUMNS)
    pipeline = DrowsinessPipeline(
        classifier=build_onnx_classifier(model_path),
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
            "backend": "onnx",
            "model": Path(model_path).name,
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
            if show_display:
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
        if show_display:
            cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the live drowsiness demo on the edge runtime."
    )
    parser.add_argument(
        "--model", required=True, help="exported temporal ONNX model (.onnx or .int8.onnx)"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window", type=int, default=150)
    parser.add_argument(
        "--log-file", default=None, help="write structured JSON logs here instead of stderr"
    )
    parser.add_argument(
        "--metrics-interval", type=float, default=5.0, help="seconds between metrics summaries"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="run without the HUD window, for headless or benchmark runs",
    )
    args = parser.parse_args(argv)
    run(
        model_path=args.model,
        camera_index=args.camera,
        window_capacity=args.window,
        log_file=args.log_file,
        metrics_interval=args.metrics_interval,
        show_display=not args.no_display,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
