# Alert behavior

How the fatigue level becomes an on screen and audible alert, and why the alert
is designed to be slow to cry wolf. For the system overview see
[docs/architecture.md](../architecture.md); for how the fatigue level itself is
produced see
[docs/ml/temporal-classifier-methodology.md](temporal-classifier-methodology.md).

SafeEyes is an assistive prototype. The alert is a cue to the driver, not a
safety of life intervention, and it makes no diagnostic claim.

## Tiers

The alert escalates through four tiers:

- **None.** The driver reads as alert. No cue.
- **Visual.** A low vigilance state is committed. A quiet on screen banner only.
- **Audible.** A drowsy state is committed. The visual banner plus an audible
  cue.
- **Alarm.** Drowsiness has persisted well past the audible threshold. A stronger
  alarm, the escalation a sustained dangerous state warrants.

## Debounce and hysteresis

A single noisy frame must never trigger an alarm, and a real alert must not
flicker on and off. The state machine enforces both:

- **Minimum duration debounce.** A new fatigue level is only committed once it
  persists for a minimum number of consecutive steps. A lone drowsy frame in an
  otherwise alert stretch is ignored.
- **Hysteresis.** Escalating to a higher tier needs fewer consecutive steps than
  de-escalating back down. The alert is quick to warn and reluctant to stand
  down, so a momentary recovery does not silence a genuine warning.
- **Sustained escalation.** Once drowsiness is committed, continuing to read
  drowsy past a longer threshold escalates from the audible cue to the stronger
  alarm.

These thresholds are parameters of the state machine, so the trade off between
responsiveness and nuisance can be tuned against real footage.

## False alarms are first class

A drowsiness alert that fires when the driver is fine is worse than useless: a
real driver would switch it off. The false alarm rate is therefore reported
beside accuracy for the fatigue classifier (see the temporal methodology), and
the debounce and hysteresis above exist specifically to keep nuisance alerts
low. The honest tension is that suppressing false alarms also slows the response
to real drowsiness; the minimum duration and hysteresis values are where that
balance is set and are tuned, not hidden.
