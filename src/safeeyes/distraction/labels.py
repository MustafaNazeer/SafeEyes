"""Distraction class labels, resolved without importing torch.

The training side exposes ``DISTRACTION_LABELS`` from ``models.distraction_data``,
but that module pulls in torch, which the Pi runtime does not have. The labels
derive from the pinned DMD taxonomy the same way, from a torch-free source, so the
edge loop can name the predicted activity. A test pins this tuple to the training
one so they cannot drift.
"""

from __future__ import annotations

from safeeyes.data.dmd_distraction import DMD_DISTRACTION_ACTIONS

DISTRACTION_LABELS: tuple[str, ...] = tuple(
    sorted(action.removeprefix("driver_actions/") for action in DMD_DISTRACTION_ACTIONS)
)
