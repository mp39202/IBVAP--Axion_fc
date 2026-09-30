# IBVAP: Intelligent Video Analytics Platform

IBVAP runs as either a Windows desktop window or a browser-based command center. Both use the same local Python/YOLO backend and preserve the same features.

## Run
1. Install Python 3.9 to 3.12 from python.org (tick "Add to PATH" on Windows).
2. On Windows, double-click **run_desktop.bat** to open the native WebView2 desktop window. The first run installs packages and downloads YOLO weights. Windows WebView2 Runtime is required.
3. For browser mode, double-click **run.bat** and open http://127.0.0.1:8000. On Linux/macOS run **./run.sh**.
4. Sign in with username `admin` and password `admin123` (no space). A new installation creates that default account. Existing installations keep the password already stored in their local database. Use **Show** to reveal the password while typing.

If the existing local admin password is unknown, close IBVAP and double-click **reset_admin.bat** in this folder. Type `RESET` when prompted. This resets only the local `admin` account to `admin123`; start IBVAP and change the password under **Account** afterward.

The first run creates a local `.venv`, installs dependencies and fetches the models into `./models`.
This needs internet once. Later runs start in seconds.

## What you get
- **Live**: waits for each camera's AI model to load, then displays the original decoded camera/video feed. Browser streaming is capped at 30 FPS to leave CPU for detection; boxes and fences are drawn in a canvas from the latest smoothed AI tracks, so video is not re-rendered by the detector. Yellow closed fences, **red** = inside fence, **green** = person, **blue** = vehicle. Each tile shows measured capture and AI FPS and flags when AI output is below 30 FPS.
- **Dashboard analytics**: four circular meters show people, cars, bikes, and buses tracked since local midnight. The new **Daily Limits** tab lets an operator or admin set a daily target for each category; each ring shows today's count against its target and turns red when it is exceeded. Blank or 0 means no target. Daily totals are stored per camera and tracker session, while alert, incident, evidence, and confirmed-plate cards use database counts for today instead of capped recent lists. Track totals count distinct track IDs, not guaranteed unique real-world individuals.
- Daily targets guide dashboard reporting only; they do not stop people or vehicles or create alerts.
- **Video replay**: saved video-file sources automatically rewind at end-of-file; if the video backend cannot seek, IBVAP reopens the file and continues playback without requiring the source to be added again.
- **Simulation tab**: clickable camera-tampering and detection-loss scenarios; browser webcam access with local YOLO tracking and smoothed boxes; a temporary video-tracking runner; and an isolated ANPR-only scan for one selected video.
- **Simulation video library**: clips already uploaded in `ivap_data/videos` and preloaded clips placed in `demo_videos` are selectable. You can also add a clip in the Simulation tab. No sample video is bundled, so add a rights-cleared test clip before the demo.
- Browser webcam access is available when the app is opened on `http://127.0.0.1` or HTTPS and the user grants permission. Its resized-frame upload target is selectable from 15 to 30 FPS (20 by default); measured AI FPS is shown separately. A real camera is needed to verify device permissions and image quality on a particular PC.
- Simulation camera-health states and ANPR-only results are isolated from operational alert and plate history. The ANPR-only scan loads its own temporary detector and may share CPU/GPU resources with active cameras.
- Vehicle class 7 detections are shown and logged as **car**; there is no separate truck label or truck category in the interface.
- **Fences**: add any number of named 4-point fences per camera. Each saves on the 4th click, remains visible with its own color, and can be removed individually or cleared together. Existing single fences migrate automatically.
- **ANPR tab**: every raw OCR reading before voting, next to validated, confidence-weighted confirmed plates. OCR uses contrast normalization, scale-up, and multiple thresholding passes.
- **Rules tab** (admin): loitering time, AI rate, image size, stream width, MQTT host, and the rule engine
  ("when *loitering* on *camera X* happens *3* times within *60* s, set severity *critical* and open an incident").
- **Themed forms**: the rule builder, source setup, user management and account controls use labeled fields that match the dark interface.
- **Admin tab**: add RTSP/HTTP/webcam cameras, upload your own video files (they are stored in `ivap_data/videos`, nothing is bundled), manage users (viewer / operator / admin), verify the hash chain.
- **Multiple cameras**: add more than one camera source; each appears as its own Live View tile and has independent fences and tracking. There is no fixed camera-count setting in the app. Each active source starts its own capture, tracking, and stream workers and currently loads its own YOLO model, so the practical number depends on available CPU, GPU/VRAM, and memory; measured AI FPS can fall as sources are added.
- An intrusion creates an alert and evidence snapshot on entry. If that tracked person remains in the fence for the configured dwell time (10 seconds by default), a separate loitering alert and evidence snapshot are created, with a live pop-up.
- Alerts, incidents with timeline, evidence snapshots and zip export, floor plan with camera placement, SQLite storage, HS256 login, and an admin **Verify Hash Chain** button that checks alert hashes, the local ledger chain, and their links. This is a tamper-evident local ledger, not an external blockchain.
- **Login recovery**: `reset_admin.bat` resets only the local admin password after you close the app; it does not delete sources, events, evidence, or settings.
- Simulation-only camera health events stay out of operational alerts. Webcam frame uploads can target 15 to 30 FPS; actual AI FPS is still measured and depends on the computer. ANPR-only results stay in the simulation screen and do not mix with operational plate history.

## Performance design
- The default AI processing target is 30 FPS (configurable from 15 to 120). If the selected hardware cannot sustain 30, try 25, 20 or 15 FPS. The source video display runs independently at capture speed, while AI FPS reports actual detector throughput and flags any shortfall instead of faking the rate.
- The low-resource profile defaults to 320-pixel inference and a one-thread OpenCV / capped PyTorch CPU setup. Re-ID starts off on new installs; enable it in Rules & Settings when the computer has spare capacity. The profile caps browser JPEG encoding at 30 FPS, uses 960-pixel stream width and lower JPEG quality to leave more CPU for detection. Smaller inputs trade some distant-object detail for speed; increase image size for small or far-away vehicles.
- Vehicle/person boxes use ByteTrack IDs plus a local one-to-one motion/overlap association when detector IDs flicker, then low-lag EMA smoothing and bounded motion prediction. Near-identical car/truck-class boxes are collapsed into one car. Missing detections are predicted for up to 1 second, local IDs are retained for up to 2 seconds to allow reacquisition, and the browser only clears the overlay when no successful AI update has arrived for 2 seconds. YOLO is still called for every eligible frame; slow machines may not keep up with every captured frame, and actual AI FPS stays visible.
- Capture continues at the source rate. Capture below the selected AI target or inference/model work slower than the target prevents the detector from processing that many distinct frames per second; optimize model size, image size and auxiliary workload for the actual hardware.
- ANPR OCR is submitted once per six YOLO updates per vehicle, now using at most eight candidate regions and two preprocessing variants. YOLO still analyzes every eligible frame. Re-ID runs in auxiliary background workers without acquiring a YOLO inference lock. OCR and Re-ID can still compete for compute, so measured AI FPS depends on the hardware and video source.
- The server JPEG-encodes the original captured frames only while someone is watching. Boxes and the fence are drawn separately in the browser, so UI rendering does not slow YOLO inference.
- **Graphics support**: NVIDIA CUDA and Intel PyTorch XPU are selected automatically when their compatible PyTorch builds and drivers are installed. Install them with `python start.py --cuda` or `python start.py --xpu`. On Windows, AMD can opt into DirectML with `python start.py --directml` or double-click `run_amd_directml.bat`; pass `--directml` on each DirectML launch. This path is experimental because the DirectML PyTorch bridge is in preview and YOLO tracking operators may not all be supported. AMD Linux users need a matching ROCm PyTorch build; ROCm is detected through PyTorch's CUDA-compatible API. If an accelerator backend is not detected at startup, IBVAP uses CPU inference. A runtime error in the experimental DirectML path may require restarting without `--directml`. Actual AI FPS depends on hardware and is not guaranteed to reach 30.
- Intel XPU requires a supported Intel GPU and driver; current PyTorch XPU support lists Intel Arc and supported Core Ultra graphics. Install the XPU-specific wheels from `requirements-intel-xpu.txt`. AMD DirectML's optional package is `requirements-amd-directml.txt`; do not install it together with the Intel XPU or NVIDIA CUDA PyTorch build. If measured AI FPS is low, leave image size at 320, turn off optional ANPR or Re-ID, close other GPU-heavy apps, or lower the AI target to 25, 20 or 15 FPS.
- Benchmark the selected source, model and GPU in Live View; this package does not claim a benchmark result in advance.

## Options
- NVIDIA GPU: `python start.py --cuda`.
- Intel Arc / supported Core Ultra GPU: `python start.py --xpu`.
- AMD DirectX 12 GPU on Windows: `python start.py --directml` (experimental; see Graphics support).
- Desktop launchers: `run_intel_xpu.bat` installs/uses Intel XPU; `run_amd_directml.bat` installs/uses experimental AMD DirectML.
- AMD ROCm GPU on Linux: install the PyTorch ROCm wheel matching the AMD driver and supported card from the official PyTorch installer, then run `python start.py`; verify that PyTorch detects the GPU before relying on acceleration.
- Pull requests: follow `CONTRIBUTING.md`; the repository includes a review checklist template under `.github`.
- Fully offline package: `python start.py --bundle` saves wheels to `./vendor` (same OS and Python version only). Also zip `./models`.
- MQTT: set the host in the Rules tab (or `MQTT_HOST` env). Every alert is published to `ivap/alerts`.
- LAN access: `HOST=0.0.0.0`. Use HTTPS or a VPN before exposing it beyond a trusted network.

## Notes
- ANPR uses RapidOCR (pip-only), with Tesseract as a fallback if installed.
- Facial recognition and face enrollment are removed; any old enrolled face templates are deleted when this build starts.
- Re-ID and plate thresholds are starting values. Tune them on your footage.

## Publish to GitHub
```
git init && git add . && git commit -m "IVAP"
git branch -M main && git remote add origin https://github.com/YOU/ivap.git && git push -u origin main
```
