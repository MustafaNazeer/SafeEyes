# Architecture

Public overview of how SafeEyes is built.

## Overview

SafeEyes monitors a driver from a single cabin camera in real time, entirely on device. A perception stage extracts interpretable per frame signals, a temporal stage fuses the fatigue signals over a rolling window, and parallel activity and gaze tracks run beside it. All of them feed one tiered, debounced alert state machine. The second version widened the original drowsiness only pipeline into this three track driver monitor; both versions share the same evaluation discipline and privacy posture.

## Perception (per frame)

Facial landmarks from the camera frame drive several signals in parallel:

- **Eye closure.** Eye aspect ratio plus a trained open or closed eye classifier.
- **Yawning.** Mouth aspect ratio, with a minimum duration requirement that is the accepted yawn signal (a mouth crop classifier was trained and evaluated but did not beat the duration rule, so it was not deployed; see `docs/adr/0006-geometric-duration-rule-over-mouth-crop-cnn.md`).
- **Head pose.** Pitch, yaw, and roll from a landmark to model solve, for nod detection and reused by the gaze track.
- **Gaze zone.** Head pose plus an iris offset geometry feed a compact gradient boosted tree that names the gaze zone. The geometric route cleared its preregistered bar, so the fallback CNN was never built (`docs/adr/0007-geometric-gaze-with-leave-one-subject-out.md`).
- **Distraction activity.** A quantized image classifier labels the driver's activity from the cabin frame. It runs every Nth frame with its class probabilities smoothed over time to protect the frame rate.

## Temporal fusion (rolling window)

Per frame signals accumulate into a fixed length window that a trained temporal classifier maps to a fatigue level (alert, low vigilance, drowsy). The model is trained at the live feature cadence so the window spans the same real time context in training and at run time.

## Alerting

A state machine consumes three parallel tracks and fuses them into one tier by severity:

- **Fatigue**, from the temporal classifier, is the only track that can reach the strongest alarm tier.
- **Eyes off road**, from the gaze zone, warns only after the gaze has stayed off the forward view continuously for a set duration, so mirror and instrument glances do not trip it. It is bounded below the alarm tier. The duration is measured in wall clock time so it holds at any frame rate, and the raw zone label is voted over a short window before it can raise an alert, so classifier noise does not become alerts.
- **Distraction** warns on a sustained non safe activity and is likewise bounded below the alarm tier.

Each track has its own debounce and hysteresis, tuned to fire quickly while keeping false alarms low.

## Training and runtime

Models are trained and evaluated offline on public datasets using subject independent splits, then exported to ONNX, quantized where measurement favors it, and deployed to a Raspberry Pi 4B for real time inference. The runtime performs no network calls and persists no raw video.

## Evaluation

Reported metrics include per class accuracy and AUROC on subject independent splits, false alarm rate reported beside every detection rate, and measured on device latency and frame rate. Every metric regenerates from a committed script and a fixed split. Methodology and model cards live under `docs/ml/`, performance numbers under `docs/perf/`.

## What is demonstrated live versus evaluated offline

The two versions differ in how far each track has been exercised on real hardware with a driver in frame, and the docs keep that distinction explicit:

- **Drowsiness and eyes off road** are demonstrated live on the Pi with measured latency and frame rate.
- **Gaze zone** is evaluated subject independently on the source dataset and its forward versus off road decision is confirmed live, but its fine grained zone labels are specific to the dataset's camera mount and are not claimed to transfer to an arbitrary camera position.
- **Distraction** is evaluated subject independently offline, where it is a deliberately weak signal reported honestly against its class floor. It is integrated into the live loop but is not demonstrated on a face mounted camera, because its training imagery is a body facing view; the model card records this.
