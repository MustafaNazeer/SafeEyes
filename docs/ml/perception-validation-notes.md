# Perception and temporal metric definitions: validation notes

This note reviews the correctness of the geometric feature definitions in the
perception stage and the metric computations in the temporal stage. It is a
sign off on the definitions and on how each number is computed, not on the
metric values themselves: those are pending a training and evaluation run on the
real datasets. For the system overview see
[docs/architecture.md](../architecture.md); for the split correctness review and
the subject independence sign off see
[docs/ml/validation-notes.md](validation-notes.md); for the temporal stage
methodology see
[docs/ml/temporal-classifier-methodology.md](temporal-classifier-methodology.md).

Every definition below is pinned by a unit test. The full perception and
temporal suites pass (54 tests at the time of writing).

## Scope

In review:

- Eye aspect ratio, mouth aspect ratio, the rotation matrix to euler
  decomposition, and the head pose solver.
- Window level features: proportion of eye closure, and the onset based counting
  of blinks, yawns, and nods.
- Evaluation metrics: per class accuracy, macro AUROC, and the false alarm rate.

Out of scope here (covered elsewhere): the subject independent split itself, the
model architectures, and any reported metric value.

## Eye aspect ratio

Verified against the accepted formulation introduced by Soukupova and Cech (2016),
"Real-Time Eye Blink Detection using Facial Landmarks":

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

The implementation takes six ordered points where p1 and p4 are the horizontal
corners and (p2, p6) and (p3, p5) are the two vertical pairs, sums the two
vertical openings, and divides by twice the horizontal width. This is exactly
the accepted ratio: the mean of the two vertical openings over the horizontal
width. A degenerate detection with zero width returns 0.0 rather than dividing by
zero, which downstream is treated as a closed or invalid eye.

The six MediaPipe FaceMesh indices are the standard ones in common use:

- Left eye: 33, 160, 158, 133, 153, 144
- Right eye: 362, 385, 387, 263, 373, 380

In each tuple the first and fourth entries are the eye corners and the remaining
four are the upper and lower lid points, matched to the p1..p6 ordering the ratio
expects. The average eye aspect ratio is the mean of the left and right values.

Verdict: correct. The formula matches the reference and the indices are the
standard FaceMesh six point set.

## Mouth aspect ratio

The ratio reuses the same six point structure as the eye aspect ratio, which is a
reasonable choice: it gives a symmetric measurement with two vertical openings
over the corner to corner width. The chosen FaceMesh indices are:

    (61, 37, 267, 291, 314, 84)

mapped so that 61 and 291 are the mouth corners (the horizontal width), the
vertical pair (37, 84) measures the opening left of center, and the vertical pair
(267, 314) measures the opening right of center. Index 37 is an upper outer lip
point and 84 is the lower outer lip point directly beneath it; 267 and 314 are the
mirrored pair on the other side of center. So the two vertical measurements are
genuinely vertically opposed lip points and the horizontal is corner to corner.
The pairing is sound and symmetric.

A note on standardization: there is no single canonical mouth aspect ratio index
set the way there is for the eye. Published variants differ. Some use a single
inner lip vertical (the center upper and lower inner lip points) over a single
horizontal; others average several verticals. The set chosen here is defensible
and internally consistent, and using the outer lip border makes the opening
during a wide yawn more pronounced than the inner border would. The one change
worth considering, if validation on YawDD shows the signal is noisy, is to switch
the verticals to the center upper and lower lip points (which open most during a
yawn) or to average three verticals instead of two. That is a tuning decision to
be made against real yawn footage, not a correctness defect.

Verdict: defensible as written. Not a blocker. Flagged for a possible tuning
revisit against YawDD if the yawn signal proves weak.

## Rotation matrix to euler decomposition

The decomposition recovers pitch, yaw, and roll in degrees using a standard
Tait-Bryan convention with a gimbal lock fallback when the cosine term is near
zero. Confirmed by direct check that a pure rotation about each axis maps to the
expected single angle:

- A rotation about the x axis yields pitch only.
- A rotation about the y axis yields yaw only.
- A rotation about the z axis yields roll only.

The identity matrix yields all zeros. The gimbal lock branch is the conventional
degenerate case handling.

Verdict: correct.

## Head pose via solvePnP

Head pose fits a generic 3D face model to the detected 2D landmarks with
solvePnP, then converts the recovered rotation vector through Rodrigues into a
rotation matrix and into euler angles using the decomposition above. The model
points are the standard six point generic face geometry (nose tip, chin, the two
eye outer corners, the two mouth corners) and are ordered to match the head pose
landmark indices. When the camera is uncalibrated a reasonable pinhole matrix is
synthesized with the focal length set to the image width and the principal point
at the image center, which is the usual uncalibrated default. The solver raises
on a wrong point count and on a solve failure rather than returning a silent bad
pose.

Verified by projecting the canonical model under a known rotation and confirming
the recovered angles match: a frontal face recovers near zero pitch, yaw, and
roll, and a known yaw rotation is recovered as yaw with negligible leakage into
pitch and roll.

Verdict: correct. The uncalibrated camera matrix is an approximation, as expected
without per camera calibration, so the absolute angles are approximate while the
relative changes used for nod detection are reliable. This is the standard
tradeoff and is acceptable for the nod signal.

## Proportion of eye closure (window feature)

Defined as the fraction of frames in the window whose eye aspect ratio is at or
below a closure threshold. This is the accepted definition: closure is per frame
state and the window feature is the proportion of closed frames. The empty window
returns 0.0. Pinned by tests on known sequences.

Verdict: correct.

## Onset based event counting (blinks, yawns, nods)

Blinks, yawns, and nods are each counted once per onset, where an onset is a
transition into the state. Eye closure and high pitch use a falling or rising
edge into the state below or above a threshold; yawns use a rising edge above a
mouth opening threshold. The implementation marks a frame as an onset when the
frame is in state and the previous frame was not, treating the frame before the
window as not in state, so a state already present at the very first frame counts
as one onset.

This is the correct way to avoid double counting: a sustained closure spanning
many frames is a single blink, not one blink per frame. Confirmed by tests that a
sequence with two separated closures counts as two, that a closure present at the
window start counts as one, and that a window never entering the state counts as
zero. Mean closure duration is the mean run length of consecutive in state
frames, which is the right companion to the onset count.

Verdict: correct. No double counting and no miscounting of boundary states.

## Per class accuracy

Computed per class as the fraction of windows truly of that class that are
predicted as that class. This is recall per class (the diagonal of the row
normalized confusion matrix), which is the right reading of per class accuracy
for an imbalanced three class problem: it does not let a dominant class hide poor
performance on a rare one. A class with no support returns not a number rather
than a misleading zero. Confirmed by a test that reads off the expected per class
recall.

Verdict: correct.

## Macro AUROC

Computed one versus rest and averaged unweighted across the three classes (macro),
using the per class score columns. The binary case is handled separately with the
positive class score column. Confirmed by a test that perfectly separated scores
yield an AUROC of 1.0. The macro average is the right choice for the same reason
as per class accuracy: it weights each fatigue level equally regardless of how
many windows fall into it.

Verdict: correct.

## False alarm rate

Defined as the fraction of genuinely not drowsy windows that are classified as
drowsy. The implementation takes the not drowsy population (every window whose
true label is not the alarm class, which is both the alert and the low vigilance
classes) and reports how many of those were predicted as the alarm class, divided
by the size of that not drowsy population. It is measured on the true negatives,
not on the prediction set and not on training data: the harness is run on the
held out subject independent evaluation split, so the denominator is the genuine
not drowsy windows of unseen subjects. An empty negative population returns 0.0.
Confirmed by a test that one false alarm out of two not drowsy windows reports
0.5.

Verdict: correct. The denominator is the not drowsy population as required, and
because the harness runs on the held out split, the rate is reported on unseen
data rather than on training data.

## Test coverage

Every definition above is pinned by a unit test under the perception and temporal
suites: the eye and mouth aspect ratios with known values and a degenerate case,
the euler decomposition per axis, the head pose solver under known projected
rotations, the proportion of eye closure, onset counting including the start
boundary case, mean closure duration, per class accuracy as recall, macro AUROC,
and the false alarm rate over the not drowsy population. The suites pass.

## Sign off

The geometric definitions (eye aspect ratio, mouth aspect ratio, the euler
decomposition, and the solvePnP head pose), the window features (proportion of
eye closure and onset based event counts), and the evaluation metrics (per class
accuracy, macro AUROC, and the false alarm rate) are correct and are pinned by
tests. The mouth aspect ratio index set is defensible and is flagged only for a
possible tuning revisit against real yawn footage, not as a defect.

This signs off the definitions and the metric computation correctness only. The
metric values are pending a training and evaluation run on the real datasets, and
the false alarm rate, per class accuracy, and macro AUROC numbers must be
regenerated by the evaluation harness on the subject independent split before any
value is published.
