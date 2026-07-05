"""Structured JSON Lines event emitter.

Writes one JSON object per event to a text stream, each stamped with the wall
clock time. The stream and clock are injected so the output is captured and
timestamps are deterministic under test; in normal use the stream is standard
error (or a log file) and the clock is the wall clock. Only scalar fields are
ever passed in, keeping the log free of frames or personal data.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from typing import TextIO


class JsonlEmitter:
    def __init__(
        self,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._clock = clock

    def emit(self, event: str, **fields: object) -> None:
        record = {"ts": self._clock(), "event": event, **fields}
        self._stream.write(json.dumps(record) + "\n")
        self._stream.flush()
