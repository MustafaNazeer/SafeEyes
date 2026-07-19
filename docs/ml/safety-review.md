# Safety and false alarm review

An assessment of whether SafeEyes would help or harm a real driver, whether its
false alarm behavior is acceptable and honestly reported, and whether any public
claim outruns the evidence. For the alert design see
[alert-behavior.md](alert-behavior.md); for the fatigue classifier and its
metrics see
[temporal-classifier-methodology.md](temporal-classifier-methodology.md).

## What was reviewed

- The reported false alarm rate and the cost of a missed detection.
- The alert escalation for nuisance behavior a real driver would not tolerate.
- The public surface, including the README, for diagnostic or safety of life
  language.
- Consistency of the assistive prototype framing.

## False alarm rate

The temporal classifier's per window false alarm rate is 0.100 for the deployed
checkpoint and 0.147 averaged over five training seeds (range 0.100 to 0.211):
between roughly one in ten and one in seven windows that are genuinely not drowsy
(alert or low vigilance) are classified as drowsy. It is reported beside accuracy
in the methodology and recorded in the metrics file, never hidden. That visibility
is correct and is the first thing this review checked.

Two honest qualifications:

- This is a per window classifier metric, not the rate at which the system
  actually raises a nuisance alarm. The alert stage commits a state only after a
  minimum duration and applies hysteresis, so isolated false positive windows do
  not fire an alarm. The effective nuisance rate at the alert level is therefore
  lower than that per window rate.
- That effective alert level rate has now been measured on labeled sequences
  and is reported in [alert-validation.md](alert-validation.md): after tuning
  the debounce and hysteresis on the train subjects under a rule fixed in
  advance, the frozen parameters produced 15.92 false audible alarms per hour on
  the held out not drowsy test footage, beside a 100.0% drowsy clip detection
  rate (9 of 9) with a median 24.2 s to first audible alert. The design
  expectation held in direction (the alert stage absorbs isolated false windows)
  but not in magnitude: the measured rate is orders beyond what a driver would
  tolerate, so the measurement strengthens the case against any reliability
  claim rather than weakening it.

A per window false alarm rate of one in seven is high in absolute terms. It is
acceptable only because the system is presented as an assistive prototype and the
alert stage is explicitly designed to absorb isolated false positives. It would
not be acceptable for anything presented as reliable.

## Cost of a missed detection

Drowsy recall is 0.489 for the deployed checkpoint and 0.485 averaged over seeds:
the classifier catches about half of genuinely drowsy windows, per window. A missed drowsy driver is a more dangerous error than a
false alarm, and the alert stage's hysteresis, quick to warn and slow to stand
down, leans the right way on that asymmetry. Sustained drowsiness, which is what
matters, is more likely to be caught than any single window suggests. Even so,
half of drowsy windows being missed is a real limitation, and it is reported
through the per class recall rather than averaged away into the headline number.

## Nuisance behavior

The four tier escalation, the minimum duration debounce, and the asymmetric
hysteresis are the right structure for keeping nuisance alarms low without
silencing genuine warnings. The honest tension, that suppressing false alarms
also slows the response to real drowsiness, is stated in the alert behavior
document rather than hidden. The thresholds have now been tuned against labeled
footage under a preregistered selection rule, and the resulting alert level
false alarm rate is measured and reported in
[alert-validation.md](alert-validation.md). Tuning halved the train side rate at
an unchanged detection rate, which is real but nowhere near sufficient: debounce
cannot rescue a per window false alarm rate of 0.100 into a usable alert level
rate. The shipped runtime defaults are deliberately unchanged until the tuned
values are reviewed.

## Framing and claims

- No diagnostic, medical, or safety of life language appears in the public
  documents or the README. The README states explicitly that SafeEyes is a
  research prototype, not a medical device, and makes no diagnostic or safety of
  life guarantees.
- The assistive prototype framing is consistent across the model cards, the
  temporal methodology, and the alert behavior document.
- One wording note: the README describes raising alerts "before drowsiness leads
  to an incident", which reads as the system's purpose rather than a performance
  promise. Given the modest recall it should stay framed as intent and never be
  restated as a guarantee. No change is required now, but it is the sentence to
  watch.
- Claim blocked: nothing in the public surface may describe SafeEyes as a
  reliable drowsiness detector. Also blocked is any claim that the trained sequence
  model is decisively better than a simple baseline. Once features are extracted at
  the live cadence and the head pose feature is corrected, the GRU no longer
  robustly beats a gradient boosted baseline on accuracy or macro AUROC and is
  retained only for its lower false alarm rate. The defensible claim is narrower, a
  two stage pipeline with its own trained models, evaluated on subject independent
  splits, with modest and honestly reported per class accuracy, macro AUROC, and
  false alarm rate.

## Verdict

The figures quoted in this review were refreshed when the deployed checkpoint was
retrained on features extracted at the live cadence with the corrected head pose
feature. The false alarm and missed detection conclusions were re-checked against
the new numbers and stand. One finding is new and is recorded above: the trained
model no longer clearly beats its baseline, which strengthens rather than weakens
the case against any reliability claim.

Signed off on honesty and framing. The false alarm rate and the missed detection
cost are both reported and visible, the framing is consistently that of an
assistive prototype, and no unsupported claim survives in the public surface.

The performance is modest and clearly limited, which is acceptable precisely
because it is not overclaimed. This sign off is on the integrity of the reporting
and the framing. It is not an endorsement of the system as fit to be relied on by
a real driver.

Open items for a future pass:

- Decide whether the tuned state machine parameters replace the shipped runtime
  defaults; the measured comparison is in
  [alert-validation.md](alert-validation.md). Either way the alert level false
  alarm rate remains far too high for reliance, and the blocked claims above
  stand on the measured evidence.
- Treat the current figures as a single subject independent split over 48
  subjects (folds 1 to 4 of UTA-RLDD), not a cross validated estimate. The
  alert level test set is 29 clips from 10 subjects; its 100.0% detection rate
  is a small sample statement, not a general claim.
