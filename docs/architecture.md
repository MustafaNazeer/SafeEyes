# Architecture

Public overview of how SafeEyes is built. This document fills in as the system comes together.

## Overview

SafeEyes detects driver drowsiness from a single cabin camera in real time, entirely on device. The pipeline is split into a perception stage that extracts interpretable per frame signals and a temporal stage that fuses them over a rolling window into a fatigue estimate, which drives a tiered, debounced alert state machine.

## Stages

1. **Perception (per frame).** Facial landmarks from the camera frame feed eye closure (eye aspect ratio plus a trained open or closed eye classifier), yawning (mouth aspect ratio), and head pose for nod detection.
2. **Temporal fusion (rolling window).** Per frame signals accumulate into features (proportion of eye closure over time, blink dynamics, yawn frequency, head nods) that a trained temporal classifier maps to a fatigue level.
3. **Alerting.** A state machine escalates through visual, audible, and stronger alerts with hysteresis and debounce, tuned to fire quickly while keeping false alarms low.

## Training and runtime

Models are trained and evaluated offline on public datasets using subject independent splits, then quantized and deployed to a Raspberry Pi 4B for real time inference. The runtime performs no network calls and persists no raw video.

## Evaluation

Reported metrics include per class accuracy and AUROC on subject independent splits, false alarm rate, and measured on device latency and frame rate. Methodology and model cards live under `docs/ml/`.
