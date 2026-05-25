# deepscreen-viewer

[![Tauri](https://img.shields.io/badge/Tauri-2.x-24C8DB?logo=tauri&logoColor=white)](https://tauri.app/)
[![Rust](https://img.shields.io/badge/Rust-1.97-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.24-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Frontend](https://img.shields.io/badge/frontend-vanilla%20JS-F7DF1E?logo=javascript&logoColor=black)](dist/)
[![Licence](https://img.shields.io/badge/licence-MIT-blue)](LICENSE)

The desktop application for exam proctoring: opens, turns the camera on, and
shows the live feed with face boxes, a head-pose gizmo, a gaze ray and a
signals HUD.

One project. The detection code — models, threading, decisions — lives in
`src-tauri/src/lib.rs` and the modules beside it, and knows nothing about
`tauri`: no window, no IPC, no webview import anywhere in it. `tauri::` is used
only in `main.rs` and its two `#[tauri::command]` handlers. That boundary used
to be enforced by a second crate (`deepscreen-detect`, now archived); it is
enforced by module discipline instead, and `cargo test` still runs the
detection code with no window and no camera to prove it.

The lib still has its own front door for exactly that: a `detect-cli` binary
that runs the models from a terminal, against a webcam or a recorded clip, with
no app and no window. It is what caught the INT8 slowdown, three wrong tensor
shapes, and a latency figure that turned out to be measured with no face in
frame — bugs a GUI would have hidden behind "looks fine."

## Screenshot

The HUD reports capture and detection rates separately, because they are
separate threads and the gap between them is the interesting number:

```
camera:0
cap 30.2 fps   det 14.9 fps   skipped 231
detect  p50 5.9 ms   p95 8.0 ms
preview p50 5.0 ms
faces 1   objects 0   seq 455
pose    yaw +2.0  pitch -17.2  roll -17.0
gaze    yaw -4.1  pitch -9.6
eye     yaw -6.1  pitch +7.6
```

## No frames cross the IPC boundary

This is the one design decision worth explaining, because the obvious approach
is a trap.

The previous implementation base64-encoded a JPEG in JavaScript and invoked a
Rust command with it every 300 ms. That is tolerable at 3 Hz and hopeless at
frame rate: base64 inflates by a third on top of an encode and a decode, all on
the UI thread, every frame.

Instead a preview thread downscales to 640×360, JPEG-encodes at quality 70, and
publishes into a lock-free slot. A `tiny_http` server bound to `127.0.0.1` on
an ephemeral port serves `multipart/x-mixed-replace`, and the front end is:

```html
<img src="http://127.0.0.1:PORT/stream">
```

The browser decodes and paints every frame with **zero JavaScript in the loop**.
The only thing crossing the Tauri boundary is a few hundred bytes of JSON per
poll — the signals and the stats, never pixels.

Measured: 5.0 ms to encode a preview frame, ~18 KB per frame, ~290 KB/s over
loopback, and the detection thread's p50 is unchanged whether the preview is
running or not.

Boxes are an SVG layer positioned over the `<img>`, with `viewBox` set to the
source resolution. Signal coordinates go in unmodified and the browser does the
scaling — no scale factors are computed in JavaScript, and nothing is
rasterised into the JPEG.

## No framework

Plain HTML, one CSS file, one JS file, no bundler, no npm, no build step. The
whole front end is under 500 lines.

The previous version's worst bug was frame-rate React re-renders. The most
reliable way not to have that bug is not to have a renderer capable of it. This
also means `cargo run` is the entire dev loop — there is no dev server to start
and nothing to watch for changes.

**No detection or decision logic lives in JavaScript.** No thresholds, no hold
timers, no hysteresis, no smoothing. The front end renders what the library
sends and computes nothing. Duplicating a threshold into the UI is how the
previous system ended up with the same constant defined in three places with
three different values.

## Running it

Clone this one repository. Nothing else is required to build it:

```bash
git clone https://github.com/Abdullah-Masood-05/deepscreen-viewer
```

```bash
# once — fetch the model weights (see "Models" below)
cd src-tauri
cargo run --release                                    # camera:0
cargo run --release -- --source file:../clip.mp4       # a recorded clip
cargo run --release -- --source camera:1 --config dev.toml

# the detection code from a terminal, no window, no camera required for most of it
cargo run --release --bin detect-cli -- devices
cargo run --release --bin detect-cli -- inspect ../models/*.onnx
```

`cargo tauri dev` also works if you have the Tauri CLI, but it is not required.

To build an installer:

```bash
cargo tauri build
```

produces an MSI and an NSIS installer with the models bundled as resources. The
resulting `.exe` resolves them through Tauri's resource directory and runs from
anywhere.

## Models

Weights are **not committed**. Put them in `models/` at the repository root
before building — `cargo tauri build` bundles whatever is there:

```bash
mkdir -p models && cd models
curl -LO https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -LO https://github.com/yakhyo/head-pose-estimation/releases/download/weights/mobilenetv3_small.onnx
mv mobilenetv3_small.onnx headpose_mobilenetv3_small.onnx
curl -LO https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx
curl -LO https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
```

| Slot | Source | Licence |
|---|---|---|
| Face | [opencv/opencv_zoo — YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT |
| Head pose | [yakhyo/head-pose-estimation](https://github.com/yakhyo/head-pose-estimation) | MIT |
| Gaze | [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) | MIT |
| Objects | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache 2.0 |

Ultralytics' YOLO26n is deliberately **not** used. It is AGPL-3.0, which would
require open-sourcing anything shipping it; YOLOX-Nano is Apache 2.0 and also
measured 2.7× faster on this hardware.

## Requirements

- Rust 1.97+, WebView2 (present on Windows 10/11 by default)
- **ffmpeg and ffprobe on `PATH`** — the camera is currently driven through
  ffmpeg's DirectShow input. Replacing this with a native capture crate is
  planned and is what a fully self-contained installer needs.

Only one process may own a webcam on Windows. If OBS, Teams, Zoom, Discord or a
browser tab is holding it, the app fails with a clear message naming the likely
cause rather than a DirectShow error code.

## What is not built yet

Honest list, since the HUD makes it look further along than it is:

- **Fusion.** The flag pills reflect instantaneous signal state — `NO FACE` is
  literally `faces.is_empty()`. There are no hold timers, no hysteresis and no
  violations yet. The UI labels this "raw signals — not violations" precisely
  so it is not mistaken for a verdict.
- **Identity checking.** ArcFace is bundled but no enrolment flow exists.
- **Calibration and liveness.**
- **Session reports.**

## Licence

MIT. Model weights carry their own licences — see the table above.
