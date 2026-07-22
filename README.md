# SafeEyes

Real time, on device driver monitoring from a cabin camera.

SafeEyes watches a driver through a single camera and raises tiered alerts on the onset of fatigue (eye closure, slow blinks, yawning, head nodding) and on the eyes leaving the road, before a lapse leads to an incident. It runs entirely on device with no cloud dependency, at real time frame rates on a Raspberry Pi 4B.

## Approach

A perception stage extracts interpretable per frame signals from facial landmarks: eye closure (eye aspect ratio and a trained open or closed eye classifier), yawning (mouth aspect ratio), head pose, and a gaze zone from head pose plus iris geometry. A temporal stage fuses the fatigue signals over a rolling window with a trained classifier to estimate a fatigue level. Those feed a debounced, tiered alert state machine that also runs parallel eyes off road and distraction tracks, fused by severity and tuned to fire quickly while keeping false alarms low.

Models are trained and evaluated offline on public driver monitoring datasets using subject independent splits, then quantized and deployed to the Pi for real time inference. Every reported number regenerates from a committed script and a fixed split.

## Status

Complete across two versions. The first delivered trained eye state and temporal
fatigue models with subject independent evaluation, a tiered alert state machine,
and a quantized deployment at sustained real time frame rates on a Raspberry Pi 4B.
The second added an eyes off road gaze track and a distraction activity classifier,
retrained the temporal model at the live feature cadence, measured the alert level
false alarm rate, and validated the yawn signal against a held out dataset. The
measured numbers live in [docs/perf/edge-benchmark.md](docs/perf/edge-benchmark.md)
and the model documentation under [docs/ml/](docs/ml/). Setup instructions are in
[docs/setup-guide.md](docs/setup-guide.md).

## Scope

Drowsiness detection is the core, done thoroughly and demonstrated live. The eyes
off road track is also demonstrated live; its forward versus off road decision is
reliable, while its fine grained gaze zone labels are specific to the source
dataset's camera mount. The distraction classifier is evaluated offline as a
deliberately weak signal reported honestly against its class floor, and is not
demonstrated on a face mounted camera because its training imagery is body facing.
Trip logging and a companion app remain future work.

## A note on intent

SafeEyes is an engineering and research prototype for exploring on device driver monitoring. It is not a medical device and makes no diagnostic or safety of life guarantees.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
