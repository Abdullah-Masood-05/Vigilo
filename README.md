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

npm install          # installs the Tauri CLI (bun works too — see "About bun")
# then fetch the models, and on macOS install ffmpeg — see sections below

npm run dev          # run it
npm run build        # build installers
```

If you would rather not use npm, everything works through cargo alone:

```bash
cd src-tauri
cargo run --release
cargo tauri build            # needs: cargo install tauri-cli --version "^2"
```

macOS support is tracked in the [`macos-support`](https://github.com/Abdullah-Masood-05/Vigilo/tree/macos-support) branch. Pre-built macOS builds are available on [EbadJunaid's fork releases](https://github.com/EbadJunaid/deepscreen-viewer/releases).

### macOS

On macOS, the app uses ffmpeg with the **AVFoundation** backend for camera
capture. Install it via Homebrew before running:

```bash
brew install ffmpeg
```

Then from the repository root:

```bash
npm install
cargo run --release                    # opens camera:0
cargo run --release -- --source "file:samples/_smoke_testsrc.mp4"  # test with video
```

Camera access requires macOS permission — the first launch will prompt you to
allow Terminal (or the app) in **System Settings > Privacy & Security > Camera**.

### About bun

`package.json` exists **only** to install the Tauri CLI. The front end has no
dependencies, no bundler and no build step — it is one HTML file, one CSS file
and one JS file, served as-is. `bun install` will not pull in a framework
because there is nothing to pull in.

If you have npm or pnpm instead, they work identically (`npm install`,
`npm run dev`). Bun is a convenience, not a requirement.

## Requirements

| | Why | How to get it |
|---|---|---|
| **Rust 1.97+** | builds the app | [rustup.rs](https://rustup.rs) |
| ffmpeg | reads the webcam — **Windows**: DirectShow (bundled), **macOS**: AVFoundation via Homebrew | see "ffmpeg" below |
| WebView2 / WebKit | renders the UI | preinstalled on Windows 10/11 (WebView2) and macOS (WebKit) |
| bun *(optional)* | installs the Tauri CLI | [bun.sh](https://bun.sh) |

**The installer has no prerequisites.** On Windows, ffmpeg ships inside it, and the app
prefers its own copy over anything on `PATH` — a stranger's ancient ffmpeg, or
one built without DirectShow, is not a thing worth debugging remotely. If the
bundled copy is missing *and* nothing is on `PATH` (a source checkout, say), the
app opens and shows a full-screen instruction rather than a black window.

On macOS, ffmpeg is expected on `PATH` (via Homebrew: `brew install ffmpeg`).

Only one process may own a webcam. If another app is holding it (OBS, Teams,
Zoom, Discord, Photo Booth, FaceTime, or a browser tab), the app names the
likely cause rather than showing a raw error code.

## ffmpeg

Camera capture uses ffmpeg as a subprocess on both platforms:

**Windows** — `ffmpeg/ffmpeg.exe` is **committed** (via git-lfs), a custom
minimal DirectShow build ~1.7 MB, fully static. `git clone` gets you a working
binary with nothing to fetch. `cargo tauri build` bundles it as-is.

It configures `--disable-all` and enables exactly: `avdevice`/`dshow` (the
camera), `avcodec` with the `mjpeg`/`rawvideo` decoders, `avformat` with the
`rawvideo` muxer, `avfilter` with only the `scale` filter, and the `pipe`
protocol. **LGPL, not GPL.** See `rust_context.md` §22 for details.

Rebuilding the Windows binary needs MSYS2 with mingw-w64, nasm and
pkg-config — not needed for development.

**macOS** — the system ffmpeg (from Homebrew) is used with the **AVFoundation**
backend. No custom build is needed:

```bash
brew install ffmpeg
```

The camera source auto-detects the platform and uses the correct ffmpeg input
format (`dshow` on Windows, `avfoundation` on macOS). Device enumeration,
camera index selection, and the RGB pipe infrastructure work identically on
both platforms.

For development, `file:` and `dir:` replay sources use whatever `ffmpeg`/
`ffprobe` are on `PATH` regardless of platform.

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

Ultralytics' YOLO26n is deliberately **not** used. It is AGPL-3.0, which would
require open-sourcing anything shipping it — and it measured 2.7× slower than
YOLOX-Nano on this hardware, so the permissive option was also the quicker one.

## What it detects

| Violation | Fires when |
|---|---|
| `NeverSeen` | no face at all in the first 10 s |
| `NoFace` | face absent for 2.5 s |
| `MultipleFaces` | two or more faces for 2 s |
| `HeadTurnedAway` | smoothed head yaw > 30° or pitch > 25° |
| `GazeOffScreen` | smoothed gaze > 25° off centre |
| `ProhibitedObject` | accumulated evidence for a phone crosses a threshold |
| `IdentityMismatch` | three consecutive checks below the similarity floor |
| `SignalLost` | pose or gaze unavailable for 5 s — a covered camera must not read as "all clear" |

Every number above lives in `Config` and nowhere else. Dump the full set with
`detect-cli config --out dev.toml`.

**Identity needs enrolment.** Click **Enrol face** once, looking at the camera,
before anything else. Until then the identity slot reports `NotConfigured` and
no mismatch can fire — an unenrolled session is not a verified one.

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
cargo test --release          # 121+ tests, no camera or models required
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

## Platform support

- **Linux** — pre-built installers are available for the three main distributions:
  - **Debian/Ubuntu**: `.deb` packages via `apt`/`dpkg`
  - **Fedora**: `.rpm` packages via `dnf`
  - **Arch Linux**: `.tar.zst` packages via `pacman`
  All are built through the GitHub Actions CI workflow (see below).
- **macOS** — pre-built macOS apps are available on [EbadJunaid's fork releases](https://github.com/EbadJunaid/deepscreen-viewer/releases). Support is tracked in the [`macos-support`](https://github.com/Abdullah-Masood-05/Vigilo/tree/macos-support) branch.

## CI / CD

This project uses GitHub Actions for CI. The workflow:

- Runs on every push to `main` and on pull requests
- Checks that `cargo clippy --all-targets` and `cargo test --release` pass
- Builds Tauri installers for Windows, Linux, and macOS
- Publishes Linux `.deb`, `.rpm`, and `.tar.zst` packages to the GitHub Release assets
- Uploads macOS `.dmg` builds

The full `.github/workflows/ci.yml` configuration is available in the repository.

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
- **macOS**: Camera capture requires ffmpeg from Homebrew; the app does not
  bundle a macOS ffmpeg binary. Camera access must be granted in System
  Settings > Privacy & Security > Camera.

## Licence

MIT. Model weights carry their own licences — see the table above.
