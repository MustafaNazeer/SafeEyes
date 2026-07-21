# ADR 0007: Geometric gaze zone classification, evaluated leave one subject out

## Status

Accepted.

## Context

The eyes off road alert track needs a gaze signal. The project ethos is
geometric first, with a learned pixel model only where geometry measurably
fails, and that position was recorded in the specification before any gaze work
began, along with a planned fallback to a small convolutional model.

The DMD gaze bundle supplies 14 subjects looking at nine marked zones inside a
stationary car. Three properties of that corpus, all measured before any model
was trained, shaped every decision here:

1. **Roughly 62 percent of frames carry the `not_valid` gaze label**, which
   marks unusable data rather than a direction of gaze.
2. **The corpus contains 207 annotated glances**, each spanning 100 to 230 near
   duplicate frames. The frame count overstates the evidence by more than two
   orders of magnitude.
3. **Forward gaze is rare**, 10 to 17 percent of valid frames, because the
   protocol has drivers looking at marked regions rather than driving.

## Decision

**Head pose plus iris offset, classified by gradient boosted trees, evaluated
leave one subject out, with the deploy bar measured at interval level.**

Four choices are worth recording, each with the reason it beat the alternative.

### Iris offset joins head pose

Head pose alone cannot distinguish a glance made with the eyes from looking
straight ahead, which is exactly what several zone boundaries turn on. The
detector already produces the refined mesh, so iris landmarks were available at
no additional inference cost and were simply unread.

The signal was checked in the raw feature before any model existed: iris offset
orders monotonically with gaze direction, from about +0.21 looking left through
-0.07 forward to -0.27 looking right. The hypothesis was visible in the data
rather than assumed.

### Leave one subject out over a single held out split

A four subject test fold would have rested the eyes off road false alarm rate on
roughly 14 independent forward glances, where one misclassified glance moves the
figure by about seven points. Holding out one subject at a time and pooling puts
every subject in the test position exactly once, so the same rate rests on all
51 forward glances, and per subject variance becomes visible instead of hidden
behind one arbitrary partition.

Nothing is tuned per fold: fixed hyperparameters, no threshold search, and the
binary signal is the nine zone argmax collapsed through one predicate. With no
inner selection loop there is nothing to leak, so pooling held out predictions
is a clean estimate rather than an optimistic one. This is what makes leave one
subject out compatible with a preregistered bar read once.

### The bar is measured at interval level

The alert track applies a duration requirement, but an annotated interval is
already a sustained glance of 100 to 230 frames, so a misclassified interval is
a multi second error the duration gate passes straight through. Gating on a
frame level rate would also have overstated confidence by the frames per
interval.

Every figure is published at both levels regardless, because a model that looks
strong per frame and weak per interval is being flattered by duplication.

### Only forward gaze counts as eyes on road

Mirror checks are legitimate driving glances, but counting them as on road would
bake a driving behaviour judgement into the label that the dataset does not
assert. The duration requirement, not the label, is what keeps normal scanning
from firing the alert.

## Consequences

- **The geometric route is deployed.** It cleared a bar fixed before any model
  was trained: interval level false alarm rate at most 0.10, observed 0.0784
  (4 of 51); detection at least 0.85, observed 0.9872 (154 of 156). The
  convolutional fallback the specification authorized was not triggered and no
  pixel model was built.
- **The false alarm rate passes on the point estimate, not with confidence.**
  Its 95 percent Wilson interval is (0.031, 0.185), so the upper bound sits well
  above the bar. The bar was preregistered as a point estimate and is not
  re-litigated after the fact, but no claim built on this record may state that
  the true rate is below 0.10. Four misread glances out of 51 is the entire
  basis.
- **The signal costs almost nothing on the device.** Inference runs in 0.080 ms,
  the cheapest model in the system, so it runs every frame rather than on a
  schedule and needs no accelerator. The device carries neither scikit-learn nor
  PyTorch; the classifier arrives as ONNX, verified to agree with the trained
  model on all 34,916 rows.
- **Per zone results below about twenty glances are indications, not
  measurements.** `right_mirror` at 0.474 is nine of nineteen and `front_right`
  at 0.609 is fourteen of twenty three. Both are zones adjacent to a neighbour
  in gaze space, the expected failure.
- **This record is about a staged task.** Drivers were instructed to look at
  marked regions in a stationary car, and the geometry is specific to the DMD
  camera mount. A different camera position changes what each zone looks like,
  and nothing here measures how far that transfers. A larger or more natural
  gaze corpus could reasonably revise this decision, and revising it means a new
  record, not an edit to this one.

Full figures, per zone counts and caveats are in
[gaze-model-card.md](../ml/gaze-model-card.md).
