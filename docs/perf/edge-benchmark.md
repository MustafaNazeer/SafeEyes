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
       --temporal-checkpoint models/temporal.pt --n-features 8 \
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
       --model models/edge/temporal.int8.onnx --input-shape 1,150,8
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
| Perception eye state | eye_state.int8.onnx | (2, 1, 24, 24) | _to measure_ | | | |
| Temporal fatigue | temporal.int8.onnx | (1, window, features) | _to measure_ | | | |
| End to end per frame | full pipeline | one camera frame | _to measure_ | | | |

Thermal behavior over a sustained run is noted alongside the table when measured.
If the end to end frame rate on the Pi is below real time, the documented
fallback is a Coral USB accelerator; that decision is made on the measured
numbers, not in advance.
