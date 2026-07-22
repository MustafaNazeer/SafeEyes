"""Live driver monitor demo on the edge runtime.

The same demo loop as the development runner, but driven by ONNX models through
ONNX Runtime instead of PyTorch checkpoints, so it runs on the Raspberry Pi with
the minimal dependency set (no PyTorch). It always runs the drowsiness track; if a
distraction model is supplied it also runs the distraction track on an Nth-frame
cadence and fuses the two into one alert tier:

    python -m safeeyes.edge.run --model models/edge/temporal.int8.onnx \\
        --distraction-model ~/safeeyes-models/distraction_mobilenet_v3_small.onnx

The classifier and scheduler construction are tested seams; the camera capture,
landmark detection, and overlay are integration glue exercised live with a real
camera.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from safeeyes.alert.gaze_smoothing import GazeZoneSmoother
from safeeyes.alert.gaze_track import EyesOffRoadTrack
from safeeyes.alert.hud import draw_hud
from safeeyes.alert.monitor import DriverMonitorPipeline
from safeeyes.alert.pipeline import DrowsinessPipeline
from safeeyes.alert.state_machine import AlertTier, DistractionAlertTrack, fuse_tiers
from safeeyes.data.dmd_gaze import is_eyes_on_road
from safeeyes.distraction.labels import DISTRACTION_LABELS
from safeeyes.distraction.scheduler import DistractionScheduler
from safeeyes.edge.preprocess import preprocess_distraction_frame
from safeeyes.edge.runtime import (
    OnnxModel,
    make_onnx_distraction_classifier,
    make_onnx_gaze_classifier,
    make_onnx_window_classifier,
)
from safeeyes.observability.session import make_run_observer
from safeeyes.perception.facemesh import FaceMeshDetector
from safeeyes.perception.frame import FEATURE_COLUMNS, frame_features
from safeeyes.perception.gaze_features import gaze_features
from safeeyes.perception.head_pose import default_camera_matrix


def build_onnx_classifier(model_path: str | Path) -> Callable[[np.ndarray], int]:
    return make_onnx_window_classifier(OnnxModel(model_path))


def build_distraction_scheduler(
    model_path: str | Path,
    *,
    every_n: int = 5,
    ema_alpha: float = 0.5,
    size: int = 224,
) -> DistractionScheduler:
    adapter = make_onnx_distraction_classifier(OnnxModel(model_path))

    def classify(frame: np.ndarray) -> np.ndarray:
        return adapter(preprocess_distraction_frame(frame, size))

    return DistractionScheduler(classify, DISTRACTION_LABELS, every_n=every_n, ema_alpha=ema_alpha)


def run(
    model_path: str | Path,
    camera_index: int = 0,
    window_capacity: int = 150,
    escalate_steps: int = 8,
    de_escalate_steps: int = 40,
    alarm_after: int = 45,
    distraction_model: str | None = None,
    distraction_every_n: int = 5,
    distraction_alpha: float = 0.5,
    gaze_model: str | None = None,
    gaze_min_seconds: float = 2.0,
    gaze_audible_seconds: float = 4.0,
    gaze_smoothing_seconds: float = 1.0,
    log_file: str | None = None,
    metrics_interval: float = 5.0,
    show_display: bool = True,
) -> None:
    import cv2

    n_features = len(FEATURE_COLUMNS)
    drowsiness = DrowsinessPipeline(
        classifier=build_onnx_classifier(model_path),
        window_capacity=window_capacity,
        n_features=n_features,
        escalate_steps=escalate_steps,
        de_escalate_steps=de_escalate_steps,
        alarm_after=alarm_after,
    )
    monitor: DriverMonitorPipeline | None = None
    if distraction_model is not None:
        scheduler = build_distraction_scheduler(
            distraction_model, every_n=distraction_every_n, ema_alpha=distraction_alpha
        )
        monitor = DriverMonitorPipeline(
            drowsiness,
            scheduler,
            DistractionAlertTrack(
                escalate_steps=escalate_steps,
                de_escalate_steps=de_escalate_steps,
                audible_after=alarm_after,
            ),
        )

    gaze_classify = None
    gaze_track = None
    gaze_smoother = None
    if gaze_model is not None:
        gaze_classify = make_onnx_gaze_classifier(OnnxModel(gaze_model))
        gaze_smoother = GazeZoneSmoother(window_seconds=gaze_smoothing_seconds)
        gaze_track = EyesOffRoadTrack(
            min_seconds=gaze_min_seconds,
            audible_seconds=gaze_audible_seconds,
        )

    detector = FaceMeshDetector()
    capture = cv2.VideoCapture(camera_index)
    observer, log_closer = make_run_observer(log_file=log_file, metrics_interval_s=metrics_interval)
    observer.start(
        {
            "backend": "onnx",
            "model": Path(model_path).name,
            "distraction_model": Path(distraction_model).name if distraction_model else None,
            "gaze_model": Path(gaze_model).name if gaze_model else None,
            "gaze_min_seconds": gaze_min_seconds if gaze_model else None,
            "gaze_smoothing_seconds": gaze_smoothing_seconds if gaze_model else None,
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
            features: np.ndarray | None = None
            if landmarks is not None:
                features = frame_features(landmarks, default_camera_matrix(width, height))
                ear = float(features[0])

            distraction_activity: str | None = None
            distraction_tier: AlertTier | None = None
            if monitor is not None:
                state = monitor.process(features, frame)
                tier = state.tier
                fatigue_level = state.fatigue_level
                distraction_activity = state.distraction.activity
                distraction_tier = state.distraction_tier
            elif features is not None:
                tier = drowsiness.process(features)
                fatigue_level = drowsiness.fatigue_level
            else:
                tier = drowsiness.current_tier
                fatigue_level = drowsiness.fatigue_level

            gaze_zone: str | None = None
            gaze_tier: AlertTier | None = None
            if gaze_track is not None and gaze_smoother is not None:
                if gaze_classify is not None and features is not None and landmarks is not None:
                    # FEATURE_COLUMNS is (ear, mar, pitch, yaw, roll), so the
                    # pose was already solved for the drowsiness vector. Reusing
                    # it avoids a second solvePnP on the same landmarks.
                    # The raw per frame label flips on classifier noise, so it is
                    # voted over a short window before it can raise an alert.
                    gaze_zone = gaze_smoother.update(
                        gaze_classify(
                            gaze_features(
                                landmarks,
                                default_camera_matrix(width, height),
                                pose=(float(features[2]), float(features[3]), float(features[4])),
                            )
                        )
                    )
                    gaze_tier = gaze_track.update(
                        gaze_zone is not None and not is_eyes_on_road(gaze_zone)
                    )
                else:
                    # No face means no gaze evidence, so the track stands down
                    # rather than holding a stale off road streak.
                    gaze_smoother.update(None)
                    gaze_tier = gaze_track.update(False)
                tier = fuse_tiers(tier, gaze_tier)

            observer.observe(
                tier=tier.name,
                fatigue=fatigue_level,
                face_detected=landmarks is not None,
                latency_s=time.perf_counter() - started,
                gaze_zone=gaze_zone,
            )
            if tier != last_tier and tier in (AlertTier.AUDIBLE, AlertTier.ALARM):
                print("\a", end="", flush=True)  # terminal bell as a placeholder chime
            last_tier = tier
            if show_display:
                overlay = draw_hud(
                    frame,
                    tier,
                    ear=ear,
                    fatigue_level=fatigue_level,
                    distraction_activity=distraction_activity,
                    distraction_tier=distraction_tier,
                    gaze_zone=gaze_zone,
                    gaze_tier=gaze_tier,
                )
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
        description="Run the live driver monitor demo on the edge runtime."
    )
    parser.add_argument(
        "--model", required=True, help="exported temporal ONNX model (.onnx or .int8.onnx)"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window", type=int, default=150)
    parser.add_argument(
        "--distraction-model",
        default=None,
        help="exported distraction ONNX model; omit to run drowsiness only",
    )
    parser.add_argument(
        "--distraction-every-n",
        type=int,
        default=5,
        help="run the distraction CNN every Nth frame",
    )
    parser.add_argument(
        "--distraction-alpha",
        type=float,
        default=0.5,
        help="EMA smoothing factor for distraction probabilities",
    )
    parser.add_argument(
        "--gaze-model",
        default=None,
        help="exported gaze zone ONNX model; omit to run without the eyes off road track",
    )
    parser.add_argument(
        "--gaze-min-seconds",
        type=float,
        default=2.0,
        help="seconds of continuous off road gaze before the track warns",
    )
    parser.add_argument(
        "--gaze-audible-seconds",
        type=float,
        default=4.0,
        help="seconds of continuous off road gaze before the track escalates",
    )
    parser.add_argument(
        "--gaze-smoothing-seconds",
        type=float,
        default=1.0,
        help="seconds of gaze labels voted over before the zone can raise an alert",
    )
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
        distraction_model=args.distraction_model,
        distraction_every_n=args.distraction_every_n,
        distraction_alpha=args.distraction_alpha,
        gaze_model=args.gaze_model,
        gaze_min_seconds=args.gaze_min_seconds,
        gaze_audible_seconds=args.gaze_audible_seconds,
        gaze_smoothing_seconds=args.gaze_smoothing_seconds,
        log_file=args.log_file,
        metrics_interval=args.metrics_interval,
        show_display=not args.no_display,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
