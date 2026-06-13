# Privacy threat model

SafeEyes processes a live camera feed of a person's face. That makes privacy the central design constraint, not an afterthought. This document records what data exists in the system, where it lives and for how long, what is never stored, the trust boundaries, and the threats considered in and out of scope. The guiding posture is simple: on device processing only, no network egress in the runtime, and no raw video retention. Every mitigation below ties back to a concrete design decision recorded in `docs/architecture.md`.

SafeEyes is an assistive prototype. It makes no diagnostic, medical, or safety of life claim. This document scopes privacy and security risk for that prototype, nothing more.

## Privacy posture in one paragraph

The runtime never sends data off the device and never writes raw camera frames to disk. Frames are read into memory, reduced to interpretable per frame signals, and discarded. Only small derived numbers (eye and mouth geometry, head pose angles, a rolling feature window, the current fatigue level, the alert state) ever persist beyond a single frame, and even those live in memory for the duration of the rolling window, not on disk. Training and evaluation happen offline on the laptop against public datasets; that data never travels to the device and is never collected from a real user.

## Assets

The things worth protecting, in rough order of sensitivity.

1. **Live camera frames of the driver's face.** The most sensitive asset. A face image is biometric and personally identifying. Protected by never persisting it and never transmitting it.
2. **Derived per frame signals.** Facial landmark coordinates, eye aspect ratio, the open or closed eye classification, mouth aspect ratio, and head pose angles. Less sensitive than a raw image but still derived from a face. Held only in memory.
3. **The rolling feature window.** PERCLOS, blink dynamics, yawn frequency, and head nod events accumulated over a fixed length window. Aggregate statistics, not imagery. Held only in memory.
4. **Runtime state.** The current fatigue level and the alert state machine's state. Transient, in memory.
5. **Trained model weights.** The eye state classifier and the temporal fatigue model, quantized for the device. These are artifacts of training on public datasets; they contain no end user data. Protected as code, not as personal data.
6. **Evaluation datasets.** The public datasets used to train and measure the models, plus the fixed splits. They live on the development laptop only, never on the device.

## Data inventory: what exists, where it lives, and for how long

| Data | Location | Lifetime | Persisted to disk |
| --- | --- | --- | --- |
| Raw camera frame | Device memory, during the runtime loop | One frame, then overwritten by the next | Never |
| Facial landmarks and per frame signals | Device memory | One frame | Never |
| Rolling feature window | Device memory | The length of the window, then it slides | Never |
| Fatigue level and alert state | Device memory | Until superseded by the next update | Never |
| HUD overlay buffer | Device memory and the local display only | One frame | Never |
| Quantized model weights | Device filesystem, loaded read only at startup | Until the operator replaces them | Yes, as a model file shipped with the build, no user data inside |
| Public training and evaluation datasets | Development laptop only | Kept for reproducibility of reported metrics | Yes, on the laptop, outside the device entirely |
| Trained checkpoints and exported models | Development laptop, then copied to the device | Kept for reproducibility | Yes, on the laptop |
| Performance and metric logs | Development laptop, and optional local diagnostic counters on the device | Kept for the project record | Counters and timings only, never imagery |

The single most important row is the first one: a raw camera frame exists in memory for exactly as long as it takes to process it, and then it is gone. Nothing in the runtime writes it anywhere.

## What is never stored

These are hard rules, enforced by construction, not by policy alone.

- **Raw video is never persisted.** No frame, no clip, no still image of the driver is ever written to disk, a buffer file, a ring buffer, a temp file, or any other durable store. The runtime has no code path that does this.
- **No frame is ever transmitted.** The runtime makes no network calls. There is no cloud inference, no telemetry upload, no remote logging of imagery.
- **No identity is derived or stored.** SafeEyes does not run face recognition, does not build a face embedding for identification, and does not associate a session with a named person. The eye state classifier answers open or closed, not who.
- **No biometric template is retained.** Landmarks are used for the current frame's geometry and discarded. They are not accumulated into a profile.

If a future feature would break any of these, it is a posture change that has to be reviewed and recorded in an architecture decision record before it ships, not slipped in.

## Trust boundaries

```
+---------------------------- development laptop (offline) ----------------------------+
|  public datasets  ->  training and evaluation  ->  quantized model export            |
+----------------------------------------|---------------------------------------------+
                                          | one way copy of model files at deploy time
                                          v
+------------------------------- device (Raspberry Pi 4B) ------------------------------+
|  camera  ->  in memory runtime loop (perception, fusion, alerting)  ->  local HUD     |
|                          no disk writes of frames, no network egress                  |
+--------------------------------------------------------------------------------------+
```

There are three boundaries that matter.

1. **Camera to runtime.** The frame enters memory and never leaves the device. This is the boundary where the most sensitive asset lives, and it is held entirely inside the device's RAM.
2. **Laptop to device.** The only data that crosses from the training environment to the device is model files and code. It is a one way copy at deploy time. No user data ever flows back across this boundary, because the device collects none.
3. **Device to the outside world.** This boundary is closed in the runtime. There is no outbound connection. The runtime is designed to function with networking disabled entirely (see the hardening checklist).

## Threats considered

| Threat | In scope | Mitigation tied to design |
| --- | --- | --- |
| Camera imagery leaking to disk and later exfiltrated | Yes | No code path writes frames to disk; verified by the runtime posture check and code review |
| Camera imagery sent off device over the network | Yes | Runtime makes no network calls; the demo is expected to run with networking off |
| A compromised dependency exfiltrating frames or phoning home | Yes | Pinned dependencies, supply chain review, and a runtime that has no network code to ride on; see the hardening checklist |
| Sensitive data left in temp files, swap, or core dumps | Yes | No frame is written to a temp file; diagnostic counters store timings and counts only, never imagery |
| Model file tampering causing unexpected behavior | Yes | Model files are loaded read only at startup; integrity is checked against a recorded hash before load |
| Dataset path handling reading or writing outside the intended directory | Yes | Dataset and model paths are validated and confined; see safe file handling in the hardening checklist |
| Logs accidentally capturing imagery or identifying data | Yes | Logging policy forbids frame data; only derived numbers and counters are logged |

## Threats explicitly out of scope

These are real risks for a shipped consumer product but are out of scope for this prototype, and saying so is part of being honest about what SafeEyes is.

- **Physical theft of the device.** If an attacker has physical possession of the device, they have the model files and code. There is no end user data on the device to steal, since none is retained, but the project does not defend against a physical adversary with full hardware access.
- **A hostile operator.** SafeEyes assumes the person running it is not trying to subvert it into a surveillance tool. It does not contain the controls that would be needed to prevent an operator from bolting on their own recording, because building those controls is out of v1 scope.
- **Side channel inference on the live display.** Someone watching the HUD over the driver's shoulder can see the alert state. This is inherent to an on screen cue and is not defended against.
- **Adversarial inputs designed to fool the classifier.** Robustness against deliberately crafted spoof faces is a model quality question, not a privacy question, and is out of scope here.
- **Regulatory certification.** No safety, medical, or automotive certification is claimed or pursued. SafeEyes is an assistive prototype.

## Why this posture holds

The privacy guarantees are credible because they are enforced by the shape of the system, not by a promise. The runtime has no network code, so there is nothing to misconfigure into leaking. The runtime has no frame persistence code, so there is no buffer to accidentally flush to disk. The training data never touches the device, so the device has no corpus of real faces to lose. These properties are restated for public readers in `docs/architecture.md`, and the hardening checklist in this directory turns them into concrete, verifiable controls.
