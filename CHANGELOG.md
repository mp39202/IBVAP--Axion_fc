# V14 targeted fixes

## Optional GPU backends and repository review setup

- Added Intel XPU selection and an opt-in PyTorch XPU install profile (`requirements-intel-xpu.txt`, `python start.py --xpu`).
- Added an opt-in Windows DirectML install profile for AMD/DirectX 12 devices (`requirements-amd-directml.txt`, `python start.py --directml`). This is documented as experimental because the bridge is in preview and some YOLO tracking operations may be unsupported.
- Kept NVIDIA CUDA setup and added backend-aware precision selection; CPU remains the fallback, and AMD ROCm PyTorch builds are detected through PyTorch's CUDA-compatible interface on supported Linux systems.
- Added contribution/review guidance and a pull request checklist. A named CODEOWNER reviewer still needs the GitHub handle to configure automatic review requests.

## Continuous and persistent tracking update

- Fixed overlay expiry to follow completed AI updates instead of the source-frame timestamp, which could make otherwise current boxes disappear after a short processing delay.
- Extended motion/overlap reacquisition for stable local IDs, configured ByteTrack to retain lost tracks longer, and predict missed detections for up to 1 second. Track identity is kept for up to 2 seconds for reacquisition, then discarded to prevent lingering ghosts.
- Added bounded browser staleness handling: the last boxes remain visible through short AI gaps and clear if the AI worker has not completed an update for 2 seconds.
- Confirmed the app supports multiple saved camera sources and shows each as a separate Live View tile. There is no fixed software count; each active source currently loads a separate YOLO model, so available memory and compute set the practical limit.
- Documented packaged GPU support accurately: NVIDIA CUDA acceleration is available with a compatible setup; AMD and Intel graphics use CPU inference because this build does not configure those GPU backends. 30 FPS is not guaranteed.

## Final reliability and dashboard update

- Added a **Daily Limits** tab for operator/admin users to set people, car, bike, and bus daily targets. Dashboard rings compare each class's daily tracked count with its saved target and mark overages; blank/0 disables the target.
- Tightened box tracking to reduce duplicate car boxes and long-lived ghosts: vehicle classes share car association, near-identical car boxes are suppressed, ByteTrack's lost-track buffer is shorter, missed-box prediction is capped at 0.25 seconds, and stale browser overlays expire after 0.75 seconds.
- Video-file camera sources now rewind at end-of-file, with a reopen fallback for codecs that cannot seek. Playback continues without re-adding the source.
- Dashboard object rings now count distinct person, car, bike, and bus track IDs stored for the local calendar day. Counts are per camera and tracker session, so tracker IDs are not claimed to represent unique real-world people or vehicles.
- Daily alert, incident, evidence snapshot, and confirmed plate cards use direct SQLite counts and refresh with the dashboard, instead of estimating totals from the latest limited API results.
- Added an admin **Verify Hash Chain** button that reruns an integrity check and reports verified alert/ledger counts or any broken entries. The check now also confirms every local-ledger entry links to the corresponding alert hash.
- Rechecked the webcam flow from browser permission request through authenticated JPEG frame upload to the temporary YOLO camera and overlay polling. A mocked browser smoke check passed permission, camera-start, frame-creation, and upload steps; this workspace has no OpenCV/Flask/Ultralytics runtime or physical webcam, so backend inference and hardware-level webcam validation remain pending.

## Login, dashboard and form usability

- Added a Show/Hide password control and explicit sign-in feedback. Login trims accidental whitespace from the username while preserving the password exactly as entered.
- Added a guarded `reset_admin.bat` recovery tool for existing local databases. It changes only the admin password and role after the user confirms by typing `RESET`.
- Added circular dashboard analytics for people, cars, bikes, and buses, with daily distinct track counts and an alert activity chart covering the last 12 hours. Counts describe recorded tracks and do not imply unique real-world identities, detector accuracy, or frame rate.
- Reworked source setup, rule creation, user management, and account forms with labeled dark-theme controls and a structured rule builder.
- Added a separate Simulation tab with isolated tamper/signal-loss mock events, browser-permission webcam tracking through the local YOLO path, and temporary video tracking that does not create a persistent camera source.
- Added a simulation video library for local clips and a separate background ANPR-only video scan. It reports scan progress, raw OCR candidates, and confirmed plates in the simulation view without mixing results into operational ANPR history.
- Added safe local video selection from `demo_videos` and uploaded test clips. No copyrighted or operational sample video is bundled.
- Browser webcam frame upload target is selectable from 15 to 30 FPS (20 by default); the backend's AI FPS stays measured and may be lower. ANPR-only scans use a separate temporary YOLO model and can share compute with concurrent camera sources.

- Refreshed the front-end palette with a teal/blue theme and redesigned the login screen with the IBVAP logo and full name at the top; removed bottom-corner branding.
- Removed the separate truck label/category: the detector's truck class is presented as a car, keeping one car category in alerts and boxes.

- Follow-up tracking tune: increased the current-detection weight in box smoothing, refreshed track overlays up to 20 times per second without overlapping polls, allowed a slightly longer bounded display prediction for inference delay, and shortened missed-detection persistence to reduce lagging or ghost boxes.

- Set the AI processing target to 30 FPS by default (configurable from 15 to 120), and removed the shared inference lock. YOLO still analyzes every eligible frame. ANPR OCR is submitted once every six YOLO updates per vehicle. The measured AI output rate remains truthful; no hardware benchmark is implied.
- Added a native Windows WebView2 launcher that opens the existing full-feature command center in an app window and starts YOLO directly in the local backend; browser mode remains available.
- Load model weights and each camera's YOLO model in background threads. Live View streams original decoded camera/video frames as MJPEG; the app window and browser mode both draw boxes and fences separately from track data.
- Keep source JPEG encoding off the detector thread and draw smoothed boxes/fences in a canvas from the tracks API, so UI rendering does not hold up YOLO or re-render the video itself.
- Added a low-resource default (320 inference size, capped CPU thread use, Re-ID off for new installs), while keeping the 30 FPS target and never adding a YOLO frame-stride skip.
- Added a one-time settings migration so older installs move from the former 480/1280/Re-ID-on defaults to 320/960/Re-ID-off. Browser MJPEG encoding is capped at 30 FPS and uses lower JPEG quality to free CPU on high-rate sources.
- Switched to an explicit lightweight ByteTrack config, lowered the detection confidence threshold for recovery, smoothed each tracked box using bounded motion prediction and EMA position/size updates, and retained predictions up to 1.25 seconds across detection gaps. The tracker keeps IDs for up to 90 detector updates. Fence checks use the smoothed bottom-center.
- Added a one-to-one local box association that preserves a UI track ID across short ByteTrack ID changes or missing IDs. Boxes use the stable local ID for smoothing and short prediction.
- Added a named multi-fence table and API with one-time migration from the former per-camera fence column. Intrusion/dwell/exit rules now evaluate each named fence separately; the UI can add and remove fences independently.
- Reduced plate OCR work to the eight largest candidate contours and two preprocessing variants, and bounded the auxiliary job queue with at most one outstanding ANPR job per vehicle track.
- Improved ANPR preprocessing, OCR candidate filtering, plate validation and confidence-weighted voting. Raw reads and crop images remain stored; one candidate per frame contributes to consensus.
- Added one intrusion alert/evidence snapshot on fence entry and one fence-exit alert/evidence snapshot on leaving. A separate loitering alert/evidence snapshot still fires after the configured dwell time. No per-frame intrusion alert or snapshot is generated; the UI suppresses repeat loitering alerts during the same occupancy episode.
- Preserved the SHA-256 alert hash chain and local ledger. The UI and README explicitly identify it as a local tamper-evident ledger, not a blockchain.
- Removed facial recognition end-to-end, including enrollment, matching, settings, routes, dependencies, and previously enrolled templates. Kept the website layout and the remaining rule engine, incident creation, camera controls, ANPR/alerts/evidence pages, Re-ID, floor plan, MQTT and authentication flows.

## Verification

- Python syntax compilation: passed.
- Embedded JavaScript syntax check: passed.
- Focused tests cover stable local IDs, box smoothing and short-gap prediction, ANPR consensus, alerts without rules, evidence and hash-chain persistence, transition-only fence alerts, multiple fences, and stream/UI wiring.
- No live camera/GPU benchmark was run in this environment. Actual AI FPS depends on the source, resolution, GPU and concurrent model load.
