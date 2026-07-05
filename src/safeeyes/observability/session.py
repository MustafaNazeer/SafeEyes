"""Wire a run observer to its output sink for the live demo loops.

Chooses the log destination (a file when one is named, otherwise standard error)
and assembles the observer over a fresh emitter and metrics window. Returns the
observer together with the handle to close, which is the file when logging to
one and ``None`` for standard error so the caller never closes a shared stream.
"""

from __future__ import annotations

import sys
from typing import TextIO

from safeeyes.observability.emitter import JsonlEmitter
from safeeyes.observability.metrics import RunMetrics
from safeeyes.observability.observer import RunObserver


def make_run_observer(
    log_file: str | None = None,
    metrics_interval_s: float = 5.0,
) -> tuple[RunObserver, TextIO | None]:
    if log_file is None:
        stream: TextIO = sys.stderr
        closer: TextIO | None = None
    else:
        stream = open(log_file, "a", encoding="utf-8")
        closer = stream
    observer = RunObserver(
        JsonlEmitter(stream=stream), RunMetrics(), metrics_interval_s=metrics_interval_s
    )
    return observer, closer
