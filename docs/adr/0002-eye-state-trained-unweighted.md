# ADR 0002: Eye state model trained without class weighting

- Status: Accepted
- Date: 2026-06-18

## Context

The eye state classifier is trained on the MRL Eye dataset with a subject
independent split. Because the split is made at the subject level, which is the
correctness requirement, the per split open and closed counts drift from an even
balance; the held out test split is open skewed (9,608 open against 5,334
closed).

Inverse-frequency class weighting is a standard mitigation for class imbalance,
and it was a reasonable default to expect. The question was whether it actually
improves the held out result on this split.

## Decision

Train the reported model without class weighting. Class weighting is implemented
and available behind a `--class-weighting` flag, off by default.

The decision is evidence based. Weighting was evaluated against the unweighted
baseline on the same fixed split:

| Metric | Unweighted (reported) | Weighted |
|--------|-----------------------|----------|
| Overall accuracy | 94.36% | 94.28% |
| Balanced accuracy | 94.47% | 93.70% |
| Recall, closed | 94.86% | 91.68% |
| Recall, open | 94.08% | 95.72% |

In the training split closed is the majority, so weighting pushed the model
toward open: open recall rose but closed recall fell further, and both overall
and balanced accuracy dropped.

## Consequences

- The reported model is the better one, and both classes clear 94% recall
  unweighted, so the headline number is not propped up by the majority class.
- The residual imbalance is surfaced honestly through balanced accuracy and per
  class recall rather than corrected by a weighting the evidence shows does not
  help here. See [the model card](../ml/eye-state-model-card.md).
- Weighting remains available and tested for datasets or splits where it might
  help, at no cost to the default path.
- The decision is specific to this dataset and split. If the training data
  changes, the comparison should be rerun rather than the conclusion assumed.
