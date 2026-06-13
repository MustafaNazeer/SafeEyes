# Hardening checklist

Concrete, verifiable controls that keep SafeEyes inside the posture set out in the threat model: on device processing only, no network egress in the runtime, no raw video retention. The threat model in this directory explains the why; this checklist is the what and the how to verify. The public system overview is in `docs/architecture.md`.

Each item is phrased so it can be checked off and re checked. Treat a failed item as a release blocker for the demo build, not a nice to have.

## Dependency hygiene and supply chain

- [ ] Every runtime dependency is pinned to an exact version in a lock file. No floating version ranges in what ships to the device.
- [ ] The dependency set is kept minimal. A library is added only when it earns its place; each new dependency widens the supply chain attack surface.
- [ ] Dependencies are reviewed before they are added: maintenance status, popularity, and whether the library reaches the network or the filesystem in ways the project does not need.
- [ ] A vulnerability scan runs against the locked dependencies in the development workflow, and known high severity advisories are triaged before a build is cut.
- [ ] Training and evaluation tooling lives in a separate dependency group from the runtime, so the device build does not pull in heavyweight training libraries it never uses.
- [ ] Model files copied to the device are recorded with a hash, and the recorded hash is checked at load time so a swapped or corrupted file is detected.

## Least privilege

- [ ] The runtime requests camera access and nothing more. No microphone, no location, no contacts, no broad storage scope.
- [ ] The runtime contains no networking code: no HTTP client, no socket opened for egress, no telemetry SDK, no remote logging target. There is nothing to misconfigure into leaking.
- [ ] The demo is expected to run with device networking disabled. The runtime functions fully offline, and running it offline is the recommended demo configuration because it makes the no egress property observable.
- [ ] The process runs as an unprivileged user. It does not need root, and it does not run as root.
- [ ] The runtime writes only to the small set of paths it genuinely needs (optional local diagnostic counters and logs). It does not write anywhere else on the filesystem.
- [ ] Model files are opened read only. The runtime never writes back to a model file.

## Safe file handling

- [ ] No camera frame is ever written to disk. There is no debug toggle, environment flag, or hidden setting that dumps frames. If a developer needs to inspect imagery during development, that happens on the laptop against public datasets, never against the live device feed, and never in a shipped build.
- [ ] No frame data ends up in a temp file, a ring buffer file, or a crash artifact. Diagnostic output is limited to timings, counts, and derived numbers.
- [ ] Dataset paths and model paths are validated before use: resolved to an absolute path, confined to an expected base directory, and rejected if they escape it. No path is taken from untrusted input and used to open a file unchecked.
- [ ] Model and configuration files are loaded with safe, non executing parsers. No format that can execute arbitrary code on load is used for untrusted or externally sourced files.
- [ ] File permissions on anything the runtime writes (logs, counters) are restrictive by default, readable and writable only by the runtime user.
- [ ] Dataset archives downloaded on the laptop are verified against a recorded checksum before they are unpacked and used, so provenance recorded in `docs/source/` actually matches the bytes on disk.

## Logging and diagnostics

- [ ] Logs never contain imagery, raw landmark dumps that could reconstruct a face, or any identifying data. They contain derived metrics, state transitions, timings, and counts.
- [ ] Log verbosity has a ceiling that cannot be raised into logging frame content. There is no log level that prints a frame.
- [ ] Diagnostic counters on the device are aggregate (frame rate, latency, alert counts), held locally, and never transmitted.

## Runtime posture verification

A short, repeatable check that the live build actually behaves the way the threat model claims. Run it before any demo.

- [ ] **No egress check.** With the runtime active, confirm there are no outbound network connections originating from the process. Running the device with networking disabled and observing that the runtime still works is the strongest version of this check.
- [ ] **No frame on disk check.** After a runtime session, confirm that no image or video file was created anywhere the runtime can write. The set of files the process touched should contain only the expected logs and counters.
- [ ] **Permission check.** Confirm the process is running unprivileged and that it holds only the camera capability it needs.
- [ ] **Model integrity check.** Confirm the loaded model files match their recorded hashes.
- [ ] **Dependency check.** Confirm the installed dependency versions match the lock file, so what is verified is what is running.

These checks are cheap to run and are the difference between claiming the posture and demonstrating it. The threat model in this directory is only as good as the last time these boxes were honestly ticked.
