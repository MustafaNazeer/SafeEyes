# ADR 0005: Interval samples for temporally annotated video

- Status: Accepted
- Date: 2026-07-14

## Context

The first version's datasets attach one label to one file: a UTA-RLDD clip is
alert, low vigilance, or drowsy as a whole, and an MRL image is open or closed.
The split machinery was built around that shape: a sample is a file, a subject
owns samples, and subject independence is enforced when samples are partitioned.

The DMD distraction bundle breaks that shape. A single session video carries
many labeled segments (drink at frames 200 to 380, operate the radio at 900 to
1040, drive safely in between), annotated in OpenLABEL as actions with frame
intervals. Three questions followed: how to represent a labeled segment, which
camera stream to consume, and what to do when an annotation names an action type
the code has never seen.

## Decision

**One sample per labeled action interval.** An interval sample extends the
existing sample with a start and end frame, and its sample id is the video path
plus the frame range (`<video>#<start>-<end>`), so every sample remains uniquely
addressable and traceable to its exact source frames. Because the interval
sample is a plain extension of the existing sample type, the subject independent
split core is reused unchanged: subjects own intervals exactly as they owned
clips, and the leakage guarantees carry over without new machinery. Manifests
gain two columns (start_frame, end_frame) and a dedicated reader and writer that
validate the header.

**The body camera stream.** The distraction bundle ships several synchronized
streams. The body stream is the wide cabin view that frames the driver's torso,
arms, and reach space, which is where activity classification evidence lives.
The face and hands streams are deferred; they can join later as additional
inputs without changing the sample model, since intervals are defined on the
shared frame clock.

**A fail closed action taxonomy.** The set of recognized action types is pinned
in code to exactly the types verified by scanning every annotation file in the
working copy. Intervals labeled `unclassified` are skipped deliberately. Any
other unrecognized type raises an error that names the offending file rather
than being dropped. A silently ignored class would bias every downstream count
and metric while looking healthy; a loud failure costs one code change to
extend the pinned set after the new type is verified.

## Consequences

- Temporally annotated datasets reuse the proven split code path, so subject
  independence holds for intervals with no parallel implementation to keep
  correct.
- Class distribution is counted per interval, not per video, which matches how
  training examples are actually drawn. The split summary, regenerated
  deterministically from the fixed seed, records these counts, and they must be
  shown beside any future accuracy.
- Sample ids embed frame ranges, so a manifest row is auditable back to the
  exact video segment without opening the annotation file.
- A dataset revision that adds new action types halts the manifest build until
  the taxonomy pin is deliberately extended. That friction is accepted as the
  cost of never training on a silently mislabeled or dropped class.
- Frame extraction groups intervals by their source video and decodes each video
  once, so the per interval sample model does not multiply decode cost.
