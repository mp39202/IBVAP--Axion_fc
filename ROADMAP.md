# IBVAP roadmap

This roadmap separates planned work from the features in the current prototype. It is a proposed sequence, not a delivery commitment.

## Priority 1: Validate the current prototype

- Measure AI FPS, alert delay, detection continuity, and resource use on low-cost CPU-only and GPU PCs.
- Measure ANPR precision and recall across plate sizes, angles, motion blur, and day/night footage.
- Test multi-camera operation, reconnect behavior, and long-running stability.
- Add repeatable evaluation clips and record false-positive and false-negative rates.
- Improve setup diagnostics and secure handling of camera URLs and credentials.

## Priority 2: Close SIH capability gaps

- Add optional face detection and, only with explicit authorization, a governed facial-recognition workflow. Define consent, enrollment, access control, retention, audit, and deletion rules before collecting face data.
- Add a dedicated night-movement mode and validate it on low-light CCTV. Consider camera-specific calibration and thermal-camera input where available.
- Extend behavior analytics beyond loitering, starting with explainable event rules and operator review.
- Improve vehicle classification and test a distinct truck category instead of grouping the model's truck class under car.
- Add supported VMS and command-and-control connectors after confirming the target system's API and security requirements.

## Priority 3: Operational hardening

- Support offline installation and model updates for remote sites.
- Add site/camera administration, health monitoring, update signing, backups, and audit export.
- Add encrypted transport and secure secret storage for deployments beyond a local demonstration.
- Evaluate an external append-only audit service or permissioned blockchain only if the deployment needs shared verification across organizations. Keep the distinction clear from the current local hash chain.
- Create deployment guidance, a threat model, and a response runbook with the intended users.

## Acceptance checks before deployment

- Publish test footage composition and per-condition accuracy metrics.
- Measure end-to-end alert latency on the target hardware and camera network.
- Confirm alert snapshots match the source time and include sufficient evidence.
- Review false alerts with operators and tune thresholds by camera.
- Complete a privacy, cybersecurity, and legal review for any face-related capability.
- Validate integrations and recovery behavior in a controlled environment before field use.
