"""Majority filter over the gaze zone label stream.

The gaze classifier decides per frame from a single frame's geometry, so its
output flips one to two times per second even while the driver holds a steady
gaze. Feeding that straight to the alert track turns classifier noise into
visible alerts. The distraction track already smooths its class probabilities
with an exponential moving average before deciding anything; the gaze path had
no equivalent, which is the gap this closes.

The classifier returns a label rather than probabilities, so the filter is a
majority vote rather than an average. Its window is wall clock seconds and is
measured from an injected clock, never a frame count, for the same reason the
alert duration is: a frame count means a different amount of real time on a
loop running at a different rate.

A frame with no face carries no gaze evidence. That clears the history rather
than voting, so a driver who leaves and re-enters the frame is not judged on a
stale zone from before they left.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Callable


class GazeZoneSmoother:
    def __init__(
        self,
        window_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")

        self.window_seconds = window_seconds
        self._clock = clock
        self._observations: deque[tuple[float, str]] = deque()
        self.zone: str | None = None

    def reset(self) -> None:
        self._observations.clear()
        self.zone = None

    def update(self, zone: str | None) -> str | None:
        if zone is None:
            self.reset()
            return None

        now = self._clock()
        self._observations.append((now, zone))
        cutoff = now - self.window_seconds
        while self._observations and self._observations[0][0] < cutoff:
            self._observations.popleft()

        counts = Counter(z for _, z in self._observations)
        best = max(counts.values())
        # Ties go to whichever tied zone was seen most recently, so a genuine
        # change is never held back by an equally sized stale majority.
        for _, candidate in reversed(self._observations):
            if counts[candidate] == best:
                self.zone = candidate
                break
        return self.zone
