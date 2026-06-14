"""Edge deployment: export, quantize, run, and benchmark the trained models.

The laptop trains the perception and temporal models in PyTorch. This package
turns those checkpoints into ONNX, quantizes them to int8 for the Raspberry Pi,
wraps an ONNX Runtime session as a drop-in classifier for the pipeline, and
measures inference latency and throughput. Everything here is exercised against
the model architectures so the machinery is proven before a trained checkpoint
or the Pi exists; the reported on-device numbers are produced by running the
benchmark on the Pi itself.
"""
