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

The temporal classifier's false alarm rate is 0.139: about one in seven windows
that are genuinely not drowsy (alert or low vigilance) are classified as drowsy.
It is reported beside accuracy in the methodology and recorded in the metrics
file, never hidden. That visibility is correct and is the first thing this review
checked.

Two honest qualifications:

- This is a per window classifier metric, not the rate at which the system
  actually raises a nuisance alarm. The alert stage commits a state only after a
  minimum duration and applies hysteresis, so isolated false positive windows do
  not fire an alarm. The effective nuisance rate at the alert level is therefore
  lower than 0.139.
- That effective alert level rate has not yet been measured on labeled
  sequences. The debounce and hysteresis thresholds are the knobs that set it,
  and they have not been tuned against real footage. Until that measurement
  exists, 0.139 is the honest number to quote and the lower alert level rate is a
  design expectation, not a measured result. This is recorded here as an open
  item, not a claim.

A per window false alarm rate of one in seven is high in absolute terms. It is
acceptable only because the system is presented as an assistive prototype and the
alert stage is explicitly designed to absorb isolated false positives. It would
not be acceptable for anything presented as reliable.

## Cost of a missed detection

Drowsy recall is 0.532: the classifier catches about half of genuinely drowsy
windows, per window. A missed drowsy driver is a more dangerous error than a
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
document rather than hidden. The thresholds are parameters meant to be tuned
against real footage, which has not happened yet; that tuning, and the measured
alert level false alarm rate it would produce, is the main open item from this
review.

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
  reliable drowsiness detector. The defensible claim is narrower, a two stage
  pipeline with its own trained models, evaluated on subject independent splits,
  with modest and honestly reported per class accuracy, macro AUROC, and false
  alarm rate.

## Verdict

The figures quoted in this review were refreshed when the deployed checkpoint
was retrained from re-extracted features; every conclusion below was re-checked
against the new numbers and stands unchanged.

Signed off on honesty and framing. The false alarm rate and the missed detection
cost are both reported and visible, the framing is consistently that of an
assistive prototype, and no unsupported claim survives in the public surface.

The performance is modest and clearly limited, which is acceptable precisely
because it is not overclaimed. This sign off is on the integrity of the reporting
and the framing. It is not an endorsement of the system as fit to be relied on by
a real driver.

Open items for a future pass:

- Measure the effective alert level false alarm rate on labeled sequences, after
  tuning the debounce and hysteresis thresholds against real footage.
- Treat the current figures as a single subject independent split over 48
  subjects (folds 1 to 4 of UTA-RLDD), not a cross validated estimate.
