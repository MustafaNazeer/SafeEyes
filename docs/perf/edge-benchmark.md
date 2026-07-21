# Edge export and benchmark

Public note on how the trained models reach the Raspberry Pi 4B and how the
on device latency and frame rate numbers are produced. The runtime on the Pi is
ONNX Runtime, so the laptop side converts the PyTorch checkpoints to ONNX and
quantizes them to int8 before they are copied across.

## Workflow

1. **Train** on the laptop (perception eye state CNN on the eye image split, the
   temporal classifier on the subject independent driving split). This produces
   PyTorch checkpoints.
2. **Export and quantize.** Convert each checkpoint to ONNX and write an int8
   copy next to it:

   ```bash
   python -m safeeyes.edge.export_models \
       --temporal-checkpoint models/temporal.pt --n-features 5 \
       --eye-checkpoint models/eye_state.pt \
       --out-dir models/edge
   ```

   The eye state network is exported through the graph capture path because its
   adaptive pooling layer is not expressible on the older tracing path; the
   temporal recurrent model is exported on the tracing path with a dynamic batch
   and sequence length. Both exports are verified against the PyTorch forward
   pass, and the int8 graphs are verified to carry quantized weights and to make
   the same decisions as the float models.
3. **Copy** the `.int8.onnx` files to the Pi. The Pi environment is the minimal
   one in `requirements-pi.txt` (ONNX Runtime plus the perception dependencies,
   no PyTorch).
4. **Benchmark on the Pi.** Measure latency and throughput against the input
   shape the runtime feeds:

   ```bash
   python -m safeeyes.edge.bench \
       --model models/edge/temporal.int8.onnx --input-shape 1,150,5
   ```

   The harness warms the session, then times single sample inferences and reports
   mean, median, and 95th percentile latency in milliseconds plus the throughput
   implied by the mean.

## Quantization choice

Dynamic quantization is used: weights are stored as int8 and activations are
quantized at inference, which needs no calibration set. For these compact models
the int8 file is not necessarily smaller than the float one, because per tensor
quantization metadata can outweigh the weight savings on a small graph. The
reason for quantizing is the int8 compute path on the device, not the file size.

## Metrics

All figures below are measured on the Raspberry Pi 4B with the production input
shapes. They are recorded here once the trained checkpoints are exported and the
benchmark is run on the device. No numbers are entered until they are measured.

| Stage | Model | Input shape | Mean (ms) | p50 (ms) | p95 (ms) | Throughput (fps) |
|-------|-------|-------------|-----------|----------|----------|------------------|
| Perception eye state | eye_state.onnx (float) | (2, 1, 24, 24) | 0.522 | 0.513 | 0.567 | 1915.9 |
| Perception eye state | eye_state.int8.onnx | (2, 1, 24, 24) | 0.753 | 0.745 | 0.784 | 1327.9 |
| Temporal fatigue | temporal.onnx (float) | (1, 150, 5) | 0.503 | 0.499 | 0.520 | 1988.4 |
| Temporal fatigue | temporal.int8.onnx | (1, 150, 5) | 0.478 | 0.474 | 0.496 | 2092.2 |
| Gaze zone | gaze_zone.onnx | (1, 7) | 0.080 | 0.078 | 0.098 | 12434.9 |
| End to end per frame | full pipeline, HUD on | one camera frame | n/a | 81.9 | 85.5 | 11.1 |
| Integrated loop, no driver in frame | drowsiness only, headless | one camera frame | n/a | 13.4 | 24.5 | 15.0 |
| Integrated loop, no driver in frame | plus distraction every 5th frame, headless | one camera frame | n/a | 13.6 | 50.6 | 14.9 |
| Distraction mobilenet_v3_small | float | (1, 3, 224, 224) | 18.321 | 18.132 | 19.540 | 54.6 |
| Distraction mobilenet_v3_small | int8 | (1, 3, 224, 224) | 62.022 | 61.791 | 63.227 | 16.1 |
| Distraction mobilenet_v2 | float | (1, 3, 224, 224) | 46.598 | 46.142 | 50.156 | 21.5 |
| Distraction mobilenet_v2 | int8 | (1, 3, 224, 224) | 219.954 | 219.342 | 223.723 | 4.5 |
| Distraction efficientnet_b0 | float | (1, 3, 224, 224) | 98.803 | 98.057 | 103.418 | 10.1 |
| Distraction efficientnet_b0 | int8 | (1, 3, 224, 224) | 304.327 | 303.710 | 307.804 | 3.3 |
| Distraction shufflenet_v2_x0_5 | float | (1, 3, 224, 224) | 9.811 | 9.734 | 10.158 | 101.9 |
| Distraction shufflenet_v2_x0_5 | int8 | (1, 3, 224, 224) | 29.917 | 29.690 | 31.534 | 33.4 |

The eye state rows were measured on a Raspberry Pi 4B running 64 bit Raspberry
Pi OS with ONNX Runtime 1.27.0 under Python 3.12 (100 timed runs after warmup;
throughput counts batches per second at the shape shown). The int8 model is
slower than the float export on this device, consistent with the export time
observation on x86: for a network this small, the overhead of dynamic
quantization outweighs the int8 compute savings. The deployed eye state model is
therefore the float export. The board read 44.3 C during these runs with no
throttling; both figures are short burst measurements, and the sustained
thermal picture will be recorded with the end to end pipeline run.

The gaze zone row was measured on the same board (200 timed runs after 20 warmup
runs, board at 35.0 C, throttle state 0x0). At 0.080 ms it is the cheapest model
in the system by a wide margin, roughly six times faster than the temporal GRU
and two orders of magnitude faster than the distraction network, because a
gradient boosted tree ensemble over seven features does far less arithmetic than
any of the neural models. Against a live loop running near 11 frames per second,
the gaze stage consumes well under a tenth of one percent of the frame budget,
so it runs on every frame rather than on a schedule and needs no accelerator.

Two caveats attach to that row. It measures model inference only, not the
feature assembly that precedes it, though the head pose solve and iris offsets
reuse landmarks the drowsiness path already computes, so the marginal perception
cost is close to zero. And no integrated with driver row exists for the gaze
track yet: the board has no capture camera attached, and a faceless run would be
meaningless here because no face means no landmarks and therefore no gaze
inference at all. That measurement is outstanding rather than reported.

The temporal rows were measured on the same board, runtime, and procedure (100
timed runs after warmup, ONNX Runtime 1.27.0, Python 3.12, board at 37.0 C, no
throttling). Here the int8 export is slightly faster than the float one, the
opposite of the eye state result: the GRU is dominated by matrix products large
enough for the int8 compute path to pay for its overhead. The deployed temporal
model is therefore the int8 export, as the setup guide assumes. Both variants
are sub millisecond, and the temporal model runs once per window step rather
than once per frame, so its cost is negligible in the frame budget; the end to
end number will be dominated by the perception stage.

The end to end row is a sustained 5 minute live run (297 s, 3295 frames) on the
same board, with a person in frame (97.8% face detection rate), a USB webcam,
and the HUD rendered to the Pi's local display, the production demo
configuration. The latency figures are the medians of the 10 second interval
p50 and p95 values from the run's structured metrics log and cover landmark
detection through the alert update; frame capture and HUD drawing are excluded
from the latency but included in the throughput, which counts whole loop
iterations. A separately recorded per frame mean was not logged, which is why
that cell is n/a; the loop period (about 90 ms at 11.1 fps) is its upper bound.
Perception dominates the budget: the temporal and eye state models measure
under a millisecond each above, and the rest of the frame is MediaPipe landmark
detection. Thermal, sustained: the board went from 37.9 C at launch to 52.1 C
after 5 minutes with the throttle flags clear (0x0), far from the 80 C
throttling threshold.

On the real time question that gates the Coral fallback: the temporal
classifier was trained on features sampled at every fifth frame of 30 fps
source video, an effective 6 feature updates per second, and the live loop
sustains 11.1, so the deployed pipeline delivers features faster than the rate
the model was trained at, and the drowsiness signals it aggregates (eye
closure proportion, blinks, yawns, nods) evolve over seconds, not frames. The
measured rate is therefore treated as meeting real time for this application,
and the Coral USB accelerator is not needed. That verdict is tied to these
numbers; it would be revisited if the pipeline grew heavier stages.

The distraction rows benchmark the four candidate image backbones at the
production input shape, on the same board and runtime (100 timed runs after
warmup, ONNX Runtime 1.27.0, the board reading 37.9 C at the start of the sweep
and 58.4 C at the end, throttle flags clear at 0x0 throughout). Unlike the
temporal model, the int8 export is three to five times slower than the float one
for every backbone, because these are convolution heavy networks and dynamic
quantization only converts the final linear layer while adding per operator
overhead across the rest of the graph. The deployed distraction model is
therefore the float export, the same conclusion the eye state network reached on
this device. The deployment weighs accuracy against latency: efficientnet_b0 has
the highest balanced accuracy but runs near 10 fps, while mobilenet_v3_small
trails it by a small margin at more than five times the speed (54.6 fps). The
deployed distraction backbone is mobilenet_v3_small (float); its accuracy
comparison, per class recall, and the honesty caveats of the evaluation split
are reported in the distraction model card. Every float candidate clears the
budget the periodic distraction schedule allows, so the Coral accelerator is not
needed for this track either.

The two integrated loop rows measure the whole live loop with the distraction
track wired in, running the distraction backbone on every fifth frame with
exponential moving average smoothing, against a same conditions baseline that runs
the drowsiness track alone. Both were 75 second headless runs on the same board
(no display, ONNX Runtime 1.27.0, board reading 42.8 C afterward, throttle flags
clear at 0x0), and the figures are the mean of the five second interval throughput
and the median of the interval p50 and p95 latencies after dropping the first
interval as warmup. Adding the distraction track leaves sustained throughput
unchanged, 14.9 against 15.0 fps, because the backbone runs on only one frame in
five; its cost shows up in the tail, where p95 rises from 24.5 to 50.6 ms on the
frames that do run it, consistent with the 18.3 ms standalone distraction latency
plus preprocessing. Real time is preserved, so the Coral verdict stands for the
integrated loop as well.

One honesty caveat bounds these two rows: they were recorded with no driver in
the camera frame, so the face detection rate was zero. Without a face the temporal
model is not exercised (its window is fed only when landmarks are present) and
MediaPipe follows its no face path, which is why these throughputs sit above the
11.1 fps of the with driver end to end row rather than beside it. They therefore
measure the added cost of the distraction schedule cleanly, by holding every other
condition equal between the two runs, but they are not a with driver number. A
sustained integrated run with a driver in frame, comparable to the v1 end to end
row, is left for a session at the hardware and will be recorded here when taken.
