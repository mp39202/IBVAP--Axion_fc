# Contributing and code review

Create a branch for each change and open a pull request into `main`. Request review from the project maintainer or the assigned reviewer in the pull request's **Reviewers** panel. Do not merge until the reviewer has approved the change and the checks pass.

For tracking, inference, alerting, or packaging changes, include the hardware/backend tested, the command used, and the measured AI FPS. Do not report an untested GPU backend or a target FPS as a measured result.

Before requesting review, check that:

- Python source parses and the embedded browser JavaScript passes a syntax check.
- Windows CPU startup still works.
- Optional NVIDIA CUDA, Intel XPU, and AMD DirectML setup paths fail clearly or fall back as documented.
- Camera playback, tracking, alerts, ANPR, and evidence behavior are not unintentionally changed.
- No camera credentials, API keys, private videos, or runtime `ivap_data` are included.
