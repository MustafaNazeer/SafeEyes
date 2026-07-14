# SafeEyes

Real time, on device driver drowsiness detection from a cabin camera.

SafeEyes watches a driver's face through a single camera and raises tiered alerts when it detects the onset of fatigue, eye closure, slow blinks, yawning, and head nodding, before drowsiness leads to an incident. It runs entirely on device with no cloud dependency, and is built to run at real time frame rates on a Raspberry Pi 4B.

## Approach

The detection pipeline has two stages. A perception stage extracts interpretable per frame signals from facial landmarks: eye closure (eye aspect ratio and a trained open or closed eye classifier), yawning (mouth aspect ratio), and head pose. A temporal stage fuses those signals over a rolling window with a trained classifier to estimate a fatigue level, which drives a debounced, tiered alert state machine tuned to fire quickly while keeping false alarms low.

Models are trained and evaluated offline on public driver drowsiness datasets using subject independent splits, then quantized and deployed to the Pi for real time inference.

## Status

The first version is complete: trained eye state and temporal fatigue models with
subject independent evaluation, a tiered alert state machine, and a quantized
deployment running at sustained real time frame rates on a Raspberry Pi 4B. The
measured numbers live in [docs/perf/edge-benchmark.md](docs/perf/edge-benchmark.md)
and the model documentation under [docs/ml/](docs/ml/). A second version is in
development. Setup instructions are in [docs/setup-guide.md](docs/setup-guide.md).

## Scope

The first version focused on drowsiness detection done thoroughly. The second
version, in development, adds distraction detection and eyes off road gaze
estimation. Trip logging and a companion app remain future work.

## A note on intent

SafeEyes is an engineering and research prototype for exploring on device driver monitoring. It is not a medical device and makes no diagnostic or safety of life guarantees.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
