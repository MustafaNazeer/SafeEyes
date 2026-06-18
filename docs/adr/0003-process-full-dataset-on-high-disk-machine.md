# ADR 0003: Process the full UTA-RLDD dataset on a high disk machine

- Status: Accepted
- Date: 2026-06-18

## Context

The temporal fatigue classifier trains on UTA-RLDD, which is roughly 111 GB of
video. The primary development laptop had only about 22 to 26 GB free, most of
its disk taken by unrelated datasets, so the full dataset could not land there.

Options considered:

- Fold by fold streaming on the laptop: download one fold, extract the compact
  per frame features, delete the video, repeat. Keeps peak disk low but adds
  streaming and delete logic, makes feature re-extraction impossible without
  re-downloading, and depends on the source exposing folds individually.
- An external drive attached to the laptop.
- A cloud VM: pay for compute and storage, move data in and out.
- A separate desktop with ample disk.

Feature extraction itself is CPU bound (MediaPipe over every frame) and a GPU
does not accelerate it, so raw compute was not the deciding factor; disk was.

## Decision

Run the dataset work on a separate desktop with 834 GB free. It holds the full
111 GB at once, so the dataset is downloaded and extracted whole, with no fold by
fold streaming.

Development still happens on the laptop and is published to the shared
repository; the desktop pulls the code and runs the heavy extraction and
training. Only the small trained checkpoints and metrics come back.

## Consequences

- No streaming or delete-as-you-go machinery, and the raw video is retained, so
  features can be re-extracted later without re-downloading.
- The feature extraction pipeline is written to be hardware agnostic and resumable
  so the same code runs on either machine and survives interruption.
- The desktop's CPU is modest, so extraction is a multi hour, unattended job;
  this is accepted because it runs once and the features are then reusable.
- The choice is tied to that machine being available. If it were not, the
  fold by fold streaming path remains the documented fallback for a disk
  constrained environment.
