# Model card: gaze zone classifier

A per frame gaze zone classifier over the nine DMD gaze zones, collapsed to a
binary eyes off road signal that drives one of the three parallel alert tracks.
It is **deployed**, having cleared a deploy bar fixed before any model was
trained.

Read the limits below before the numbers. Two of them are large enough to change
what the figures mean.

## What was excluded, and how much

**About 62 percent of every recording is discarded.** DMD annotates a
`not_valid` gaze zone for frames where the driver's gaze cannot be determined,
and those frames are dropped rather than given a class, since the label marks
unusable data rather than a direction of gaze. On the sampled recordings that
covers 3,208 of 5,340 frames for one subject, 3,650 of 5,582 for another, and
3,303 of 5,318 for a third.

Of the frames that survive, a further small fraction is lost where the face
detector finds no face, leaving 34,916 usable rows from 16 recordings. Face
detection ran between 96 and 100 percent on the labelled frames.

**The forward gaze class is rare.** `front` is only 10 to 17 percent of valid
frames, because the recording protocol has drivers looking at nine marked
regions rather than driving. In the binary collapse, eyes off road is roughly 85
to 90 percent of the data, the inverse of real driving. False alarm rate is a
conditional rate, so this skew does not bias it, but it does mean the rate rests
on a small number of forward glances.

## The data is staged, not natural driving

DMD gaze was recorded with drivers **instructed to look at nine marked regions
inside a stationary car**. Nothing here is natural driving behaviour. Every
figure below describes performance on a staged task, and none of it establishes
how the system behaves on a road.

Dataset: Driver Monitoring Dataset (DMD), Vicomtech, gaze bundle, session s6,
RGB face camera. Licensed CC BY-NC-ND 4.0. No frames, clips, annotations or
derivatives are redistributed here.

## Evaluation protocol

Leave one subject out across all 14 subjects, pooled, scored once.

A single held out split was rejected for this corpus. Fourteen subjects carry
only 207 annotated glances, and each glance is 100 to 230 near duplicate frames,
so a four subject test fold would have rested the false alarm rate on roughly 14
independent forward glances. Holding out one subject at a time puts every
subject in the test position exactly once, so the same rate rests on all 51.

Nothing is tuned per fold: fixed hyperparameters, no threshold search, and the
binary signal is the nine zone argmax collapsed through a single predicate. With
no inner selection loop there is nothing to leak, so the pooled held out
predictions are a clean estimate.

**Two subjects have a second recording.** Folds are built on subjects, never on
recordings, so both recordings of one person always sit on the same side.

### Why every number appears twice

Frames inside one annotated glance are near duplicates, so a frame level rate
overstates its own confidence by the number of frames per glance. The annotated
interval is the independent unit. Both levels are reported side by side, and the
interval figures are the ones that carry evidential weight.

## The preregistered bar

Fixed and committed **before any model was trained**, both conditions at
interval level:

| Condition | Threshold | Result | |
|-----------|-----------|--------|---|
| Eyes off road false alarm rate | at most 0.10 | **0.0784** (4 of 51) | pass |
| Eyes off road detection rate | at least 0.85 | **0.9872** (154 of 156) | pass |

**The false alarm rate passes on the point estimate, not with confidence.** Its
95 percent Wilson interval is (0.031, 0.185), so the upper bound sits well above
the 0.10 bar. The bar was preregistered as a point estimate and is not
re-litigated after the fact, but nothing here establishes that the true rate is
below 0.10. Four misread glances out of 51 is the whole basis; one more would
have moved the figure by two points.

Detection is on firmer ground: 154 of 156, with a 95 percent interval of
(0.954, 0.996), comfortably clear of the 0.85 floor.

## Results

| Level | Nine zone accuracy | False alarm rate | Detection rate |
|-------|--------------------|------------------|----------------|
| Interval | 0.8357 (173/207) | 0.0784 (4/51) | 0.9872 (154/156) |
| Frame | 0.7867 (27,469/34,916) | 0.0735 (405/5,512) | 0.9724 (28,592/29,404) |

Interval accuracy exceeds frame accuracy because a majority vote across a whole
glance absorbs transient single frame errors.

### Per zone, at interval level

| Zone | Glances | Correct | Recall |
|------|---------|---------|--------|
| front | 51 | 47 | 0.922 |
| steering_wheel | 24 | 23 | 0.958 |
| front_right | 23 | 14 | 0.609 |
| center_mirror | 21 | 17 | 0.810 |
| right_mirror | 19 | 9 | 0.474 |
| infotainment | 18 | 17 | 0.944 |
| left_mirror | 18 | 16 | 0.889 |
| right | 17 | 15 | 0.882 |
| left | 16 | 15 | 0.938 |

**Counts this small do not support confident per zone claims.** `right_mirror`
at 0.474 is nine correct out of nineteen, and `front_right` at 0.609 is fourteen
of twenty three. Both are zones that sit close to a neighbour in gaze space,
which is the expected failure and is visible in the raw feature: iris offset
alone barely separates `right` from `right_mirror`, leaving head pose to carry
the distinction. A zone level recall built on seventeen to twenty three glances
should be read as an indication, not a measurement.

## Model and features

A scikit-learn gradient boosted classifier over seven geometric features: head
pose pitch, yaw and roll, plus the horizontal and vertical iris offset within
each eye socket, normalized by socket width so it is invariant to face size and
camera distance.

Head pose alone cannot separate a glance made with the eyes from looking
straight ahead, which is exactly what several of these zones turn on. On real
recordings the iris offset orders monotonically with gaze direction, from about
+0.21 looking left through -0.07 forward to -0.27 looking right.

Both signals derive from landmarks the drowsiness path already computes, so the
gaze stage adds no extra landmark inference on the device.

## Eyes on road is `front` only

Only a forward gaze counts as eyes on road. Mirror checks are legitimate driving
glances, but treating them as on road would bake a driving behaviour judgement
into the label that the dataset does not assert. The alert track relies on a
duration requirement rather than the label to avoid firing on normal scanning:
the driver must look away continuously before it warns, and the requirement is
expressed in seconds and converted through the measured loop frame rate.

## On device

The Raspberry Pi 4B runs onnxruntime alone and has neither scikit-learn nor
PyTorch installed, verified on the board. The classifier reaches it as ONNX
through a conversion whose output is checked against the trained model: zero
disagreements across all 34,916 rows.

Inference costs 0.080 ms, the cheapest model in the system. Integrated with a
driver in frame the track costs about 0.9 frames per second. Latency figures and
the caveats attached to them are in
[edge-benchmark.md](../perf/edge-benchmark.md).

## What this model is not

It is not evidence about real driving. It is not a safety device. The deployed
artifact trains on all fourteen subjects, while the figures above come from the
leave one subject out run, so they describe this architecture and feature set
rather than that exact artifact. And the geometry is specific to the DMD camera
mount: a different camera position relative to the driver changes what each zone
looks like, and nothing here measures how far that transfers.

Every number above regenerates from
[gaze-model-metrics.json](gaze-model-metrics.json) and is pinned by the honesty
suite.
