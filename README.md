# Vigilo

[![Tauri](https://img.shields.io/badge/Tauri-2.x-24C8DB?logo=tauri&logoColor=white)](https://tauri.app/)
[![Rust](https://img.shields.io/badge/Rust-1.97-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.24-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Frontend](https://img.shields.io/badge/frontend-vanilla%20JS-F7DF1E?logo=javascript&logoColor=black)](dist/)
[![Licence](https://img.shields.io/badge/licence-AGPL--3.0-blue)](LICENSE)

Webcam proctoring for online exams. Opens, turns the camera on, and watches for
the things that matter: nobody in frame, two people in frame, a phone in shot,
a head turned away, eyes off the screen, and whether the person sitting there
is still the person who enrolled.

It reports **violations**, not raw signals — each one has to survive a hold
timer and hysteresis before it appears, so a dropped frame or a blink does not
produce an alert.

## Quick start

```bash
git clone https://github.com/Abdullah-Masood-05/Vigilo
cd Vigilo

bun install          # installs the Tauri CLI (see "About bun" below)
# then fetch the models and ffmpeg, see the two sections below

bun run dev          # run it
bun run build        # build installers
```

If you would rather not use bun, everything works through cargo alone:

```bash
cd src-tauri
cargo run --release
cargo tauri build            # needs: cargo install tauri-cli --version "^2"
```

### About bun

`package.json` exists **only** to install the Tauri CLI. The front end has no
dependencies, no bundler and no build step. It is one HTML file, one CSS file
and one JS file, served as-is. `bun install` will not pull in a framework
because there is nothing to pull in.

If you have npm or pnpm instead, they work identically (`npm install`,
`npm run dev`). Bun is a convenience, not a requirement.

## Requirements

| | Why | How to get it |
|---|---|---|
| **Rust 1.97+** | builds the app | [rustup.rs](https://rustup.rs) |
| **Linker (Windows)** | fast linking with rust-lld | `rustup component add llvm-tools` or [MSVC Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| ffmpeg | reads the webcam via DirectShow (committed to the repo, nothing to fetch) | see "ffmpeg" below |
| WebView2 | renders the UI | preinstalled on Windows 10/11 |
| bun *(optional)* | installs the Tauri CLI | [bun.sh](https://bun.sh) |

**Running the installers** needs, on x86_64, a CPU with AVX2 (`x86-64-v3`:
Intel Haswell / AMD Zen or newer). CI builds with `-C target-cpu=x86-64-v3`
because the preprocessing loops vectorise with it; a local `cargo build` has no
such floor. Apple Silicon builds are unaffected.

### Windows Linker Setup

The project uses `rust-lld` for faster build link times. Windows users can choose either option:

- **Option A (Recommended):** Install LLVM tools through rustup:
  ```powershell
  rustup component add llvm-tools
  ```
- **Option B:** Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

**The installer has no prerequisites.** ffmpeg ships inside it, and the app
prefers its own copy over anything on `PATH` — a stranger's ancient ffmpeg, or
one built without DirectShow, is not a thing worth debugging remotely. If the
bundled copy is missing *and* nothing is on `PATH` (a source checkout, say), the
app opens and shows a full-screen instruction rather than a black window.

Only one process may own a webcam on Windows. If OBS, Teams, Zoom, Discord or a
browser tab is holding it, the app names the likely cause rather than showing a
DirectShow error code.

## ffmpeg

`ffmpeg/ffmpeg.exe` is **committed** (via git-lfs) — a custom minimal build,
camera capture only, ~1.7 MB, fully static (no companion DLLs). `git clone`
gets you a working binary with nothing to fetch and no build toolchain to
install. `cargo tauri build` bundles it as-is.

It configures `--disable-all` and enables exactly: `avdevice`/`dshow` (the
camera), `avcodec` with the `mjpeg`/`rawvideo` decoders, `avformat` with the
`rawvideo` muxer, `avfilter` with only the `scale` filter (needed for the
pixel-format conversion `ffmpeg.exe`'s own CLI machinery requires — the
`swscale` library alone is not enough), and the `pipe` protocol. No network,
no doc, no `ffplay`, no `ffprobe` — the camera path never calls `ffprobe`;
only `file:`/`dir:` replay does, and that is dev-only. `--extra-ldflags=-static`
removes the mingw runtime DLLs (`libwinpthread-1.dll` and friends) a default
build would otherwise depend on.

**LGPL, not GPL.** The default configure is LGPL 2.1+ with none of the GPL/nonfree
flags. LGPL is compatible with this project's AGPL-3.0 licence; a GPL build would
not be. The licence text ships alongside the binary. Full detail, including the
exact configure line and the equivalence measurements against a full LGPL build,
is in `rust_context.md` §22.

Rebuilding it needs MSYS2 with mingw-w64, nasm and pkg-config — a one-time
cost on whoever's machine builds it, not a permanent project dependency, since
the output is committed. For development you do not need any of this: `file:`
and `dir:` replay sources use whatever `ffmpeg`/`ffprobe` are on `PATH`.

## Models

Weights are **not committed** — they are 27 MB and carry their own licences.
Put them in `models/` at the repository root before building; `bun run build`
bundles whatever is there into the installer.

```bash
mkdir -p models && cd models

# Face detection — YuNet, MIT
curl -LO https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# Head pose — MIT
curl -LO https://github.com/yakhyo/head-pose-estimation/releases/download/weights/mobilenetv3_small.onnx
mv mobilenetv3_small.onnx headpose_mobilenetv3_small.onnx

# Gaze + eye-in-head — MobileGaze, MIT
curl -LO https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx

# Prohibited objects — YOLOX-Nano, Apache 2.0
curl -LO https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx

# Identity — ArcFace w600k_mbf, from InsightFace's buffalo_sc pack
curl -LO https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip
unzip -j buffalo_sc.zip w600k_mbf.onnx && rm buffalo_sc.zip
```

You should end up with exactly these five files:

| File | Slot | Size | Licence |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | face + 5 keypoints | 0.2 MB | [MIT](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| `headpose_mobilenetv3_small.onnx` | head pose | 5.8 MB | [MIT](https://github.com/yakhyo/head-pose-estimation) |
| `mobileone_s0_gaze.onnx` | gaze + eye-in-head | 4.7 MB | [MIT](https://github.com/yakhyo/gaze-estimation) |
| `yolox_nano.onnx` | prohibited objects | 3.5 MB | [Apache 2.0](https://github.com/Megvii-BaseDetection/YOLOX) |
| `w600k_mbf.onnx` | identity | 13.0 MB | [InsightFace](https://github.com/deepinsight/insightface) |

Ultralytics' YOLO26n is deliberately **not** used. It measured 2.7x slower than
YOLOX-Nano on this hardware, and YOLOX-Nano's Apache 2.0 licence is simpler to
deal with for downstream users.

## What it detects

## What it detects

| Violation | Default Trigger | Default Severity | Meaning |
|---|---|---|---|
| `NeverSeen` | No face at all in first 10 s | `high` | Candidate failed to position themselves before session started |
| `NoFace` | Face absent continuously for 2.5 s | `high` | Candidate left the camera frame or ducked out of view |
| `MultipleFaces` | 2+ faces present continuously for 2.0 s | `critical` | Unauthorized person entered frame / assisted candidate |
| `HeadTurnedAway` | Head yaw > 30° or pitch > 25° for 1.5 s | `medium` | Looking away toward secondary monitor, notes, or assistant |
| `GazeOffScreen` | Eye gaze > 25° horizontal / 20° vertical for 1.2 s | `low` | Eye wandering away from display boundaries |
| `ProhibitedObject` | Phone confidence > 0.40 for 1.0 s | `high` | Unauthorized mobile phone or handheld screen visible |
| `IdentityMismatch` | 3 consecutive checks with cosine sim < 0.40 | `critical` | Person in chair does not match enrolled reference candidate |
| `SignalLost` | Pose or gaze lost for 5.0 s | `critical` | Camera covered, blocked, or feed frozen (never reads as "clear") |

Every threshold, timer, and severity above is dynamically configurable via the in-app **Settings** panel (or persistent `settings.toml`), and hot-reloads instantly without interrupting camera capture.

**Identity needs enrolment.** Click **Enrol face** once, looking at the camera,
before anything else. Until then the identity slot reports `NotConfigured` and
no mismatch can fire — an unenrolled session is not a verified one.

## Configuration & Settings Guide

Vigilo provides a slide-out **Threshold Settings** UI (and a persistent `settings.toml` configuration file) allowing test administrators and proctors to tailor sensitivity to their testing environment.

### 1. Severity Levels Explained

Every violation is classified under one of four severity tiers. Setting appropriate severities determines how proctoring review workflows triage and act on infractions:

| Severity | Color Code | Proctoring Meaning & Action | Typical Use Cases |
|---|---|---|---|
| **Low** | Blue (`#58a6ff`) | **Informational Anomaly.** Brief posture shift or momentary eye glance. Logged to the timeline for post-exam informational review; does **not** deduct points, flag high-risk warnings, or interrupt the candidate. | `GazeOffScreen` (brief glances, thinking glances) |
| **Medium** | Amber (`#d29922`) | **Suspicious Behavior / Warning.** Sustained pattern of looking off-target, reading off-screen cheat notes, or facing a secondary monitor. Increments cumulative exam risk score and triggers real-time visual warning. | `HeadTurnedAway` (consistent head turning) |
| **High** | Orange (`#f0883e`) | **Significant Misconduct Indicator.** Strong indicator of unauthorized activity or physical absence. Alerts proctor immediately for prioritized real-time video verification. | `NoFace` (candidate left chair), `ProhibitedObject` (smartphone visible) |
| **Critical** | Crimson (`#f85149`) | **Severe Integrity Breach.** Irrefutable or severe compromise of examination integrity. Triggers immediate urgent alarm and provides grounds for instant test pausing, locking, or disqualification. | `MultipleFaces` (second person present), `IdentityMismatch` (impersonation / proxy test taker), `SignalLost` (covered lens / feed tampering) |

---

### 2. User-Configurable Detection Rules

These parameters can be tuned directly from the top section of the in-app **Settings** drawer:

#### A. Presence & Identity
* **No-face hold time (s)** *(Default: 2.5s)*: How many continuous seconds no face can be detected before raising `NoFace`. A buffer of 2.0–3.0s prevents false positives when a candidate sneezes or bends momentarily.
* **Multi-face hold time (s)** *(Default: 2.0s)*: Continuous seconds that multiple faces must remain in frame before triggering `MultipleFaces`. Absorbs brief background passers-by in non-secure rooms.
* **Identity similarity floor (0–1)** *(Default: 0.40)*: ArcFace cosine similarity threshold between the enrolled face embedding and periodic live checks. Embeddings have unit norm ($L_2 = 1.0$), so similarity spans $[-1.0, 1.0]$. Values $\ge 0.40$ represent the same individual with high confidence; scores below indicate an imposter.
* **Identity consecutive failures** *(Default: 3)*: Number of consecutive failed identity checks required before firing `IdentityMismatch`. At the 0.2 Hz identity check interval, 3 failures ensure 15 seconds of confirmed mismatch, eliminating lighting flicker false alarms.

#### B. Head Pose
* **Head yaw threshold (deg)** *(Default: 30°)*: Maximum allowable horizontal head turn (left/right) away from the webcam.
* **Head pitch threshold (deg)** *(Default: 25°)*: Maximum allowable vertical head tilt (up/down).
* **Head turned hold time (s)** *(Default: 1.5s)*: Seconds the head orientation must continuously exceed either yaw or pitch limits before triggering `HeadTurnedAway`.
* *Hysteresis note*: To prevent edge-chatter (rapid firing on/off when hovering near 30°), the pipeline automatically enforces exit bounds: $yaw_{exit} = \max(5.0^\circ, yaw_{enter} - 7.0^\circ)$ and $pitch_{exit} = \max(5.0^\circ, pitch_{enter} - 5.0^\circ)$.

#### C. Gaze Tracking
* **Gaze yaw threshold (deg)** *(Default: 25°)*: Maximum horizontal eye gaze angle off screen center.
* **Gaze pitch threshold (deg)** *(Default: 20°)*: Maximum vertical eye gaze angle off screen center.
* **Gaze off-screen hold time (s)** *(Default: 1.2s)*: Continuous duration looking outside screen bounds before firing `GazeOffScreen`. Tuned to absorb normal physiological blinks and momentary micro-saccades.

#### D. Objects & Hardware Health
* **Phone score threshold (0–1)** *(Default: 0.40)*: YOLOX-Nano confidence floor for mobile phone bounding box detections.
* **Phone hold time (s)** *(Default: 1.0s)*: Seconds of continuous/accumulated visual evidence required before raising `ProhibitedObject`.
* **Signal lost hold time (s)** *(Default: 5.0s)*: Timeout duration when face, pose, or gaze signals are completely missing from the capture pipeline. **Critical security safeguard**: if a candidate covers their camera with tape or disconnects the sensor, Vigilo will never report "all clear" — it raises `SignalLost`.

---

### 3. Developer Options (Internal Tunables)

Clicking **"Show internal tunables"** reveals low-level model and engine parameters:

| Category | Tunable Parameter | Default | Purpose & Impact |
|---|---|---|---|
| **YuNet Face Detector** | `yunet_score_thresh` | `0.60` | Minimum confidence score to accept a detected face candidate. |
| | `yunet_nms_thresh` | `0.30` | Non-Maximum Suppression IoU threshold to deduplicate overlapping face boxes. |
| | `face_crop_margin` | `0.20` | Fractional margin added around the face bounding box before feeding to pose/identity nets. |
| **Smoothing & Filtering** | `pose_ema_alpha` | `0.35` | Exponential Moving Average smoothing factor for head pose angles ($0 < \alpha \le 1.0$). Lower values provide smoother tracking; higher values give faster response. |
| | `gaze_ema_alpha` | `0.35` | EMA smoothing factor for gaze angles. |
| | `blink_ear_cutoff` | `0.18` | Eye Aspect Ratio (EAR) threshold below which an eye is classified as closed (blinking). Blinking frames are ignored in gaze fusion so blinks don't distort gaze angles. |
| **Worker Cadence** | `face_target_fps` | `15.0 Hz` | Target execution rate for face detection, head pose, and gaze tracking worker. |
| | `object_target_fps` | `1.0 Hz` | Target rate for YOLOX-Nano object detection (keeps CPU consumption low). |
| | `identity_interval_sec` | `5.0 s` | Interval between ArcFace identity verification checks (0.2 Hz). |
| **ORT Execution** | `ort_intra_threads` | `1` | Number of parallel intra-op compute threads per ONNX Runtime session. Set to 1 to minimize CPU core thrashing across concurrent workers. |

---

### 4. Persistence & Zero-Overhead Hot-Reloading

1. **Storage Location**: Settings are automatically saved to `settings.toml` in your operating system's standard application data folder:
   - **Windows**: `%APPDATA%\com.deepscreen.viewer\settings.toml`
   - **macOS**: `~/Library/Application Support/com.deepscreen.viewer/settings.toml`
   - **Linux**: `~/.config/com.deepscreen.viewer/settings.toml`
2. **Instant Zero-Cost Updates**:
   - The engine maintains configuration in an atomic `ArcSwap<Config>` pointer.
   - Worker loops check `Arc::ptr_eq` each cycle without acquiring mutex locks.
   - Updating settings in the UI persists to disk and hot-reloads the active detection engine instantaneously without dropped camera frames, video lag, or process restarts.
   - The **"Reset to Defaults"** button instantly restores baseline factory values and removes custom overrides.


## Architecture

```
capture thread ──► ArcSwap<Frame>     latest-frame slot, overwrite, never a queue
                          │
      ┌───────────────────┼───────────────────┐
      ▼                   ▼                   ▼
 face worker 15 Hz   object worker 1 Hz  identity worker 0.2 Hz
 YuNet → pose → gaze   YOLOX-Nano          ArcFace
      └───────────────────┼───────────────────┘
                          ▼
                       Signals ──► fusion ──► Violations ──► events()
```

Rules that hold throughout:

- **One ONNX Runtime session per model, owned by exactly one thread.** No
  `Mutex<Session>` anywhere in the inference path.
- **Every tunable number lives in one `Config` struct.**
- **No detection or decision logic in JavaScript.** No thresholds, no timers,
  no hysteresis. The front end renders what the library sends and computes
  nothing.
- **No frames cross the IPC boundary.** Preview is an MJPEG stream on loopback;
  only a few hundred bytes of JSON per poll go through Tauri.
- **Fusion is a pure function.** No clock reads, no I/O — so a recorded session
  replays to a byte-identical event sequence, which is what makes threshold
  tuning possible at all. 2700 frames replay in 87 ms.

## Development

```bash
cd src-tauri
cargo test --release          # 119 tests, no camera or models required
cargo clippy --all-targets
```

`detect-cli` is a headless harness for the same library code — no window, no
camera needed for most of it. It is **behind a feature flag** so it stays out
of the installer, since it links its own copy of ONNX Runtime:

```bash
cargo run --release --features cli --bin detect-cli -- devices
cargo run --release --features cli --bin detect-cli -- inspect ../models/*.onnx
cargo run --release --features cli --bin detect-cli -- bench --all --iters 50
cargo run --release --features cli --bin detect-cli -- record --source camera:0 --out s.jsonl
cargo run --release --features cli --bin detect-cli -- replay s.jsonl
```

`replay` is the tuning loop: it runs fusion over a recording with **zero
inference**, so changing a threshold is a TOML edit and an 87 ms re-run.

### GPU acceleration

A plain build runs every model on the CPU. GPU inference is a Cargo feature,
one per platform:

| Feature | Platform | Runs on | Needs on the machine |
|---|---|---|---|
| `gpu-directml` | Windows | any DirectX 12 GPU | nothing — `DirectML.dll` ships beside the exe |
| `gpu-coreml` | macOS (Apple Silicon) | GPU / Neural Engine | nothing |
| `gpu-cuda` | Linux (also Windows) | NVIDIA GPUs | NVIDIA driver, CUDA 12 runtime, cuDNN 9 |

```bash
bun run tauri dev --features gpu-directml
bun run tauri build --features gpu-coreml
cargo run --release --features cli,gpu-cuda --bin detect-cli -- bench --all
```

CI builds each installer with its platform's feature. Linux `.deb`, `.rpm` and
Arch packages ship ONNX Runtime's CUDA provider libraries in `/usr/bin` beside
the binary (merged in from `src-tauri/tauri.gpu-cuda.conf.json`); the AppImage
is CPU-only.

Each model tries the GPU and falls back to CPU on its own if the provider will
not start, with a warning in the log. What each model actually got is on the
HUD's `ep` line and in the bench table's `EP` column. `bench-cpu.toml` forces
everything onto the CPU for comparison.

`inspect` earns its keep — it reported that YuNet's released ONNX takes 640×640
and not the 320×320 its docs imply, that the head-pose model returns a 3×3
rotation matrix rather than Euler angles, and that MobileGaze emits two 90-bin
classification heads rather than regressed angles. Guessing any of those
produces plausible-looking output that is quietly wrong.

## Measured

Intel i7-11850H, CPU execution provider, release build.

| | |
|---|---|
| Capture | 30.0 fps sustained |
| Detection | 14.9 fps (15 Hz target) |
| Face + pose + gaze worker | **27.0 ms p50 / 30.9 ms p95** |
| YOLOX-Nano (1 Hz worker) | 11.6 ms p50 |
| Identity worker | no measurable cost |

Measured over a ten-minute live session with a face present throughout. Full
methodology and the per-slot coverage breakdown are in `rust_context.md` §18.

## Known limitations

Stated plainly, because the HUD makes it look further along than it is:

- **Book detection does not work.** COCO's `book` class maxed at 0.149 over
  2700 frames of a book plainly in shot, with zero frames above 0.25. Phone
  detection is validated (peaks 0.80 and 0.86); books are not. Fixing it needs
  fine-tuning on proctoring data. See `rust_context.md` §18.4.
- **Gaze pitch carries a systematic +12–15° offset** because the camera sits
  above the screen. Corrected by a config constant for now; a proper
  per-user calibration step is not built.
- **Severity is a per-rule constant**, not a fused co-occurrence score.
- **Enrolment is in memory only** and lasts as long as the process. Persisting
  a face embedding is a data-protection decision, not a convenience.
- **No session report is written to disk yet**, no evidence capture, no
  calibration UI.
- ffmpeg is a subprocess, not a linked library. Replacing it with a native
  capture crate would drop ~128 MB from the installer and remove a process
  boundary from the capture path.

## Platform support

- **Windows**: `.exe` (NSIS setup) and `.msi` (Wix installer) for x86_64.
- **macOS**: `.dmg` package for Apple Silicon (aarch64).
- **Linux**: Pre-built packages for the three main distribution families:
  - **Debian / Ubuntu**: `.deb` packages via `apt` / `dpkg`
  - **Fedora / RHEL**: `.rpm` packages via `dnf` / `rpm`
  - **Arch Linux**: `.pkg.tar.zst` packages via `pacman`
  - **Generic Linux**: `.AppImage` portable executable

## CI / CD

The GitHub Actions workflow runs on every pull request and push to `main`:

- Runs `cargo clippy --all-targets --release` and `cargo test --release`.
- Builds release packages across Windows, macOS (Apple Silicon), and Linux.
- Packages Debian `.deb`, Fedora `.rpm`, Arch Linux `.pkg.tar.zst`, and `.AppImage` on Linux.
- Packages Windows `.exe` and `.msi` installers.
- Packages macOS `.dmg` disk images.
- Publishes all installers directly to GitHub Releases.

## Licence

AGPL-3.0. Model weights carry their own licences, see the table above.
