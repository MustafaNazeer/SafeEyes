# Test plan and regression checklist

How SafeEyes is kept correct: what the automated suite covers, which paths are
treated as correctness critical, and the checklist run before any change ships.
For the system overview see [docs/architecture.md](../architecture.md); for the
evaluation methodology see the documents under [docs/ml/](../ml).

## How to run

The same three commands run locally and in continuous integration on every push
and pull request, so a green local run means a green pipeline:

```
ruff check .
mypy
pytest
```

`pytest` is the source of truth for the current test count. No change merges with
a failing suite, a lint error, or a type error.

## What is tested, by stage

Coverage follows the two stage pipeline plus the data and edge tooling around it.

### Data foundation (`tests/data/`)

Guards the path from raw downloads to fixed, subject independent split manifests:
filename parsing for the MRL Eye and UTA-RLDD datasets, frame sampling, checksum
verification, manifest read and write, and the split generator itself. The split
tests include a deliberately leaked partition to prove the subject independence
check fails loudly when a subject crosses train and test, because a silent leak
would invalidate every downstream number.

### Perception (`tests/perception/`)

Unit tests for the interpretable geometry computed per frame: eye aspect ratio,
mouth aspect ratio, and the head pose solve, including a projection round trip
that recovers the pose it started from. The landmark adapter and the offline
feature extractor are covered with injected detectors so the control flow is
tested without depending on the camera or the landmark model.

### Eye state model (`tests/models/`)

The convolutional classifier definition, its preprocessing, the manifest backed
dataset, the training loop, and the evaluation that regenerates the reported
accuracy, per class recall, and confusion matrix from a saved checkpoint and the
fixed split. The class weighting helper is unit tested, and the evaluation path
is what lets every figure in the model card trace back to a script rather than a
remembered value.

### Temporal fatigue classifier (`tests/temporal/`)

The feature window buffer contract, window feature aggregation (proportion of eye
closure, blink, yawn, and nod events counted once per onset), the evaluation
metrics including the false alarm rate, the recurrent model and the gradient
boosted baseline, the windowing, and the training harness. The harness tests
confirm both models learn a separable signal and that training stays within a
bounded per batch memory footprint.

### Alert stage (`tests/alert/`)

Behavioral tests for the tiered state machine: a sustained drowsy signal
escalates through the tiers, transient noise does not trigger an alert, the
minimum duration debounce and the asymmetric hysteresis hold, and the machine
does not chatter between tiers. These are the tests that protect the false alarm
behavior the project treats as first class.

### Edge tooling (`tests/edge/`)

ONNX export verified against the PyTorch forward pass, int8 dynamic quantization,
the runtime wrapper, the benchmark harness with an injectable clock, and a
pipeline parity test that drives the full decision pipeline through both the
PyTorch and the float ONNX backends and asserts an identical alert tier sequence.
Exact parity is intentionally not asserted for the int8 model, since dynamic
quantization is approximate.

## Correctness critical paths

These are developed test first, because they are easy to get subtly wrong and a
silent error would corrupt a reported result:

- The perception geometry (eye aspect ratio, mouth aspect ratio, head pose).
- The alert state machine (escalation, hysteresis, debounce, no chatter).
- The feature window buffer contract.
- The subject independence of every split.

The evaluation harness is reproducible by construction: every reported metric is
regenerated from a fixed split by a command recorded next to it, so a number can
always be checked rather than trusted.

## Honesty guards

The suite encodes the project's honesty rules as tests and conventions:

- Subject independence is enforced and proven with a leaked split fixture.
- The false alarm rate is a first class metric with its own tests, reported
  beside accuracy and never hidden.
- Per class recall and balanced accuracy are reported alongside overall accuracy
  so a class imbalance cannot inflate a headline number.
- Reported figures regenerate from a script and a fixed split.

## Regression checklist

Run before closing a unit of work or shipping a change:

- [ ] `ruff check .`, `mypy`, and `pytest` all pass with clean output.
- [ ] New behavior was added test first, and each new test was seen to fail
      before it passed.
- [ ] Any reported metric that changed was regenerated from its command on the
      fixed split, and the document that cites it was updated to match.
- [ ] No split was regenerated in a way that lets a subject cross train and test;
      the subject independence check still passes.
- [ ] Correctness critical paths touched by the change (geometry, state machine,
      window buffer, split) still have passing tests covering the new behavior.
- [ ] Edge artifacts, if changed, still pass the export fidelity and pipeline
      parity tests.
- [ ] No public document gained a metric that is not backed by a reproducible run.
