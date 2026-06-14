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
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from safeeyes.alert.hud import draw_hud
from safeeyes.alert.pipeline import DrowsinessPipeline
from safeeyes.alert.state_machine import AlertTier
from safeeyes.edge.runtime import OnnxModel, make_onnx_window_classifier
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the live drowsiness demo on the edge runtime."
    )
    parser.add_argument(
        "--model", required=True, help="exported temporal ONNX model (.onnx or .int8.onnx)"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window", type=int, default=150)
    args = parser.parse_args(argv)
    run(model_path=args.model, camera_index=args.camera, window_capacity=args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
