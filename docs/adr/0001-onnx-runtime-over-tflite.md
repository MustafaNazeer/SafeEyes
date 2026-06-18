# ADR 0001: ONNX Runtime over TFLite for edge inference

- Status: Accepted
- Date: 2026-06-18

## Context

The models are trained in PyTorch on the laptop and must run quantized on a
Raspberry Pi 4B for the live demo. Two mature on device runtimes were candidates:
TensorFlow Lite and ONNX Runtime. The choice drives the entire export and
quantization toolchain and the Pi dependency set.

PyTorch exports to ONNX directly. Reaching TFLite from PyTorch instead requires a
conversion chain through an intermediate representation, which adds steps and
failure points and a second framework to keep working.

## Decision

Use ONNX Runtime on the Pi. Models are exported with the PyTorch ONNX exporter,
quantized to int8 with the ONNX quantization tools, and run through ONNX Runtime.
The Pi dependency set (`requirements-pi.txt`) carries `onnxruntime` and excludes
PyTorch and TensorFlow.

## Consequences

- A single, direct export path: PyTorch checkpoint to ONNX to int8 ONNX, all on
  the laptop. See [docs/perf/edge-benchmark.md](../perf/edge-benchmark.md).
- Export fidelity is guarded: exports are checked against the PyTorch forward
  pass, and a pipeline parity test asserts the float ONNX backend makes the same
  alert decisions as the PyTorch reference. Exact parity is not asserted for the
  int8 model, since dynamic quantization is approximate.
- On x86 the int8 model is slower than float, because quantization overhead
  exceeds the compute saved on these small models. The int8 versus float choice
  is therefore deferred to the Pi's ARM CPU, where it is measured rather than
  assumed.
- Tension with the Coral fallback: the Coral EdgeTPU compiles from TFLite, not
  ONNX. If the measured CPU frame rate proves insufficient and a Coral
  accelerator is adopted, a TFLite export path would have to be added for the
  EdgeTPU. That cost is accepted because the Coral fallback is itself contingent
  on evidence that the CPU is not enough, which is not yet established.
