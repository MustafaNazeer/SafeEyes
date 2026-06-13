"""SafeEyes: on device driver drowsiness detection.

A two stage computer vision pipeline. The perception stage extracts per frame
eye, mouth, and head pose signals from a cabin camera. The temporal stage
accumulates those signals over a rolling window and classifies a fatigue level,
which a tiered alert state machine consumes. Models are trained on the laptop
and exported quantized for a Raspberry Pi 4B edge runtime.
"""

__version__ = "0.1.0"
