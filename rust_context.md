# rust_context.md — what has been built, and what was learned building it

> Companion to `CONTEXT.md` (the old JS/WASM module — history and mistakes) and
> `MODELS.md` (the spec — the plan). **This file is the record of what actually
> happened**, including the places where reality contradicted the plan.
>
> Everything below is measured on the development machine unless marked
> otherwise. Where a number came from a vendor claim or an estimate, it says so.
>
> Development machine: **Intel i7-11850H** (Tiger Lake, 8 physical / 16 logical
> cores, has AVX-512 VNNI), Windows 11 IoT Enterprise LTSC 2024, Rust 1.97.1,
> ffmpeg 8.1.2, integrated webcam ("Integrated Webcam", 1280×720 MJPEG @ 30).

---

## 0. Where things live

```
Update_FYP/
├── CONTEXT.md                  the old JS/WASM module (reference only)
├── MODELS.md                   the spec being built to
├── rust_context.md             ← this file
├── DeepScreen-DesktopApp/      the existing Tauri app — NOT MODIFIED
├── deepscreen-detect/          the detection crate — no UI, no Tauri
└── deepscreen-viewer/          the product: the detection module as a Tauri app
```

**The existing app has not been touched.** Not one file. `git status` inside
`DeepScreen-DesktopApp/` shows exactly what it showed before this work started.
The only interaction has been *reading* it for reference and *copying*
`w600k_mbf.onnx` out of its resources folder.

`deepscreen-detect` deliberately sits outside that repo and has **no `tauri`
dependency**, and never will (MODELS.md §0) — the production Tauri adapter,
written at step 11, depends on this crate, never the reverse.

`deepscreen-viewer` is a *second*, separate Tauri app that depends on
`deepscreen-detect` by path. It began as a test instrument and **is now the
deliverable** — the cheating-detection module itself, shipped standalone.
Integration into `DeepScreen-DesktopApp` is explicitly not this project's job.
See §9 for its architecture and §16 for the change of role.

---

## 1. Build order status — MODELS.md §11

| Step | What | State |
|---|---|---|
| 1 | Crate skeleton, types, config, `detect-cli`, file/dir replay | **done** |
| 2 | Camera capture behind `FrameSource` | **harness path done** (ffmpeg/DirectShow); `crabcamera` still to do |
| 3 | YuNet face detection + baseline bench | **done** — 8.4 ms p50 end-to-end |
| 4 | Threading skeleton, `ArcSwap` frame bus, `Detector` | **done** |
| 5 | DirectML | not started |
| 6 | Pose + gaze | **done** — wired and decoded; gizmo + gaze ray in the app |
| 7 | Objects on their own worker at 1 Hz | **done** — YOLOX-Nano, Apache 2.0; see §17 |
| 8 | Fusion + record/replay tuning | not started |
| 9 | ArcFace identity at 0.2 Hz | model installed, **not wired** |
| 10 | Quantization | INT8 investigated — see §8 |
| 11 | Adapter into `DeepScreen-DesktopApp` | **out of scope** — the viewer is the product now (§16) |

`deepscreen-detect`: 5758 lines of Rust across 20 files, **60 tests**, all
passing. `deepscreen-viewer`: 542 lines of Rust plus a 474-line vanilla
HTML/CSS/JS frontend and no build step. `cargo clippy --all-targets` is silent
in both crates.

---

## 2. What exists

```
deepscreen-detect/
├── models/                     6 .onnx files, ~33 MB, all five slots filled
├── samples/                    _smoke_testsrc.mp4 (generated ffmpeg pattern)
├── src/
│   ├── lib.rs           56     public re-exports + build-status table
│   ├── types.rs        373     Frame, Signals, Violation, Event — the seam
│   ├── config.rs       484     every tunable number, and validation
│   ├── error.rs         68     DetectError + degrade-never-die policy
│   ├── report.rs       210     SessionReport, per-signal liveness, percentiles
│   ├── capture/
│   │   ├── mod.rs      139     FrameSource trait, SourceSpec parsing
│   │   ├── ffmpeg.rs   212     shared raw-rgb24 subprocess pipe
│   │   ├── camera.rs   398     device enumeration, format probing, capture
│   │   └── replay.rs   338     video file + image directory sources
│   ├── models/
│   │   ├── mod.rs      259     session building, inspection, synthetic bench
│   │   └── face.rs     391     YuNet: letterbox, anchor decode, NMS
│   ├── pipeline/
│   │   ├── mod.rs      380     Detector, DetectorBuilder, Detected, Shared state
│   │   ├── frame_bus.rs 158    latest-frame ArcSwap slot
│   │   └── workers.rs  164     capture_loop, detect_loop
│   └── bin/detect-cli.rs  745  the harness
└── tests/fusion_replay.rs 139  replay regression scaffold

deepscreen-viewer/
├── src-tauri/
│   ├── Cargo.toml               depends on ../../deepscreen-detect by path
│   ├── tauri.conf.json          frontendDist -> ../dist, no dev server config
│   └── src/
│       ├── main.rs      255     commands, wiring, arg parsing, error framing
│       └── preview.rs   264     downscale + JPEG encode + loopback MJPEG server
└── dist/                        static frontend, no build step
    ├── index.html        37
    ├── style.css        142
    └── main.js          187
```

`deepscreen-detect` dependencies: `serde`, `serde_json`, `toml`, `thiserror`,
`arc-swap`, `crossbeam-channel`, `clap`, `image`, `ort 2.0.0-rc.12`, `ndarray`,
`fast_image_resize`, `tracing`, `tracing-subscriber`. Dev: `tempfile`.

`ort` is pinned to the same `2.0.0-rc.12` the app already ships, so the two
cannot disagree about the ONNX Runtime binary.

`deepscreen-viewer` adds on top: `tauri`, `tiny_http` (the MJPEG server),
`arc-swap`, `image`, `fast_image_resize`, `tracing`. Nothing else — no HTTP
framework, no async runtime, no bundler.

---

## 3. The seam that everything rests on

`Signals` (stateless, per-frame, produced by models) is kept strictly separate
from `Violation` (stateful, temporal, produced by fusion). `Signals` derives
`Serialize + Deserialize`, so a session can be recorded once and replayed
through fusion thousands of times with **zero inference**.

One addition beyond the spec: **`SignalCoverage`**, a per-frame record of which
model slots actually ran. Without it, `objects: []` is ambiguous between "the
detector looked and saw nothing" and "the detector was never running" — and
MODELS.md §8 makes that distinction a correctness requirement for the product,
not just for logging. This is exactly the ambiguity behind §10's
object-detection gap: `objects` being empty today means "never running", not
"nothing seen", and `SignalCoverage` is what lets that be stated precisely
instead of guessed.

> **⚠ This section claimed "it is now impossible to serialise a `Signals` that
> hides the difference." That was false for about as long as it was written.**
>
> `SignalCoverage` was five booleans, and two of them lied:
>
> - `objects` was fed from a sticky global `AtomicBool` meaning *has ever run*.
>   After the object worker's first result every later frame claimed coverage,
>   including the fourteen in fifteen it never touched. An empty list read as
>   evidence of absence on frames where nothing had looked — the precise
>   ambiguity the field exists to remove.
> - `gaze` was `true` whenever a gaze value came back, but the model returned a
>   **held previous value** when its reliability gate fired. A stale
>   measurement was indistinguishable from a fresh one.
>
> Fixed by making the record per-frame and per-slot: five `SlotState`s
> (`produced` / `skipped_gated` / `skipped_cadence` / `failed` /
> `not_configured`), with the gate reason carried alongside. Gaze returns no
> value at all when gated, and object results are attached to exactly one frame
> and then consumed rather than carried forward. Format version bumped to 2,
> because a v1 recording's `true` cannot be mapped onto `Produced` without
> importing the same ambiguity.
>
> The lesson is not about the booleans. A claim of impossibility written in the
> same commit as the mechanism it describes is an assertion, not a test — and
> this file is the place that is supposed to notice.

The other API decision from §3 is already load-bearing, twice over now: there
is **no way to push continuous values**. `events()` is edge-triggered;
`snapshot()` is polled. The old module's frame-rate React re-render bug is
unrepresentable here because no API expresses it — and `deepscreen-viewer`
(§9) is the second, independent proof of that, not just the crate's own tests.

---

## 4. Capture — what the camera actually does

Live capture currently runs the camera through **ffmpeg's DirectShow input**,
reading raw rgb24 off a pipe, reusing the same machinery as video-file replay.

This is the *harness* path, not the product. It exists because it bought a real
measurement on day one with no new dependency and no `nokhwa` build fight. It
is not what ships: an exam client that spawns ffmpeg has no frame-level control,
an extra process, an extra copy per frame, and a hard dependency on an external
binary. Step 2 proper replaces it with `crabcamera`. Keep this source anyway —
it is a useful reference to benchmark the real path against.

### Measured

```
300 frames, camera:0 at 1280x720 MJPEG
mean 30.05 fps          inter-frame p50 31.23 ms   p95 53.44 ms   max 186.03 ms
79.2 MB/s decoded
```

Capture is not going to be the bottleneck.

### The MJPEG finding

`detect-cli devices --formats` on the built-in webcam:

```
mjpeg      1280x720 @ 30 fps        <- what we use
mjpeg      640x480  @ 30 fps
yuyv422    1280x720 @ 10 fps        <- raw 720p is capped at a THIRD the rate
yuyv422    640x480  @ 30 fps
```

Raw YUYV at 1280×720×30 would be ~55 MB/s over USB, so the camera refuses and
offers 10 fps instead. This is exactly the bandwidth ceiling MODELS.md §12
predicts, now confirmed on real hardware, and it is why `capture.prefer_mjpeg`
defaults to true.

### Two things that cost time

**The camera was held by another process.** Windows lets exactly one process
own a webcam. OBS Studio was running with camera consent, and ffmpeg failed
with *"Could not run graph (sometimes caused by a device already in use by
other application)"*. Worth knowing before blaming the code — and it recurred
while testing the viewer, so `deepscreen-viewer` now detects this failure
string specifically and prints a plain-language explanation instead of the raw
DirectShow error (§9).

**A real bug, found because of that failure.** The first failure reported an
*empty* reason. The stderr drain thread was being read before it had written
anything — the code read the collected message, then reaped the process. Fixed
by reversing the order: `wait()` the child, `join()` the drain, *then* read.
The diagnosis is the whole value of an error in this layer, so losing it was
worse than the original failure.

---

## 5. Models — what is installed, and what they actually look like

All five slots from MODELS.md §5.0 are present, ~33 MB total.

| Slot | File | Real input | Real output | MB | Licence |
|---|---|---|---|---|---|
| Face | `face_detection_yunet_2023mar.onnx` | `1x3x640x640` | 12 tensors, strides 8/16/32 | 0.2 | MIT |
| Face INT8 | `face_detection_yunet_2023mar_int8.onnx` | `1x3x640x640` | same | 0.1 | MIT |
| Head pose | `headpose_mobilenetv3_small.onnx` | `1x3x224x224` | `rotation_matrix [1,3,3]` | 5.8 | MIT |
| Gaze | `mobileone_s0_gaze.onnx` | `1x3x448x448` | `yaw[1,90]`, `pitch[1,90]` | 4.7 | MIT |
| Objects | `yolox_nano.onnx` | `1x3x416x416` | `output [1,3549,85]` | 3.5 | Apache 2.0 |
| Objects (rejected) | `yolo26n.onnx` | `1x3x640x640` | `output0 [1,300,6]` | 9.5 | **AGPL-3.0** |
| Identity | `w600k_mbf.onnx` | `[?,3,112,112]` | `516 [1,512]` | 13.0 | — |

**Corrected.** This table listed `yolo26n.onnx` as *the* object slot long after
§17 replaced it with YOLOX-Nano on licence grounds. `ModelPaths::CONVENTIONAL`
has mapped the slot to `yolox_nano.onnx` since then, and the runtime log
confirms it: `object model ready model=models\yolox_nano.onnx`. `yolo26n.onnx`
stays on disk as benchmark evidence and is excluded from the app bundle —
`deepscreen-viewer/models/` does not contain it.

Sources: OpenCV Zoo (YuNet), `yakhyo/head-pose-estimation` and
`yakhyo/gaze-estimation` GitHub releases, `ultralytics/assets` release `v8.4.0`,
and the app's own resources folder (ArcFace, copied out).

**`yolo26n.onnx` ships pre-exported.** No Python, no torch, no export step —
which was the reason it had initially been deferred. That assumption was wrong
and cost a round trip. Having the file is not the same as having it wired in —
see §10.

### `detect-cli inspect` earned its keep immediately

A command that reports a model's real tensor interface was written *before* any
pre- or post-processing, on the principle that a published shape and an exported
shape are not reliably the same thing. It caught three discrepancies in
MODELS.md §5.0 straight away:

1. **YuNet's input is 640×640, not 320×320**, and the head emits twelve
   separate tensors with no NMS in the graph.
2. **The head-pose model outputs a 3×3 rotation matrix, not Euler angles**, at
   224×224 rather than 60×60. (It is `yakhyo/head-pose-estimation`, MIT —
   `head-pose-estimation-adas-0001` is published only as OpenVINO IR, and the
   PINTO mirror is dead.) Postprocessing must convert the matrix to yaw/pitch/roll.
3. **MobileGaze is 4.97 MB, not ~8 MB**, takes 448×448, and emits two 90-bin
   *classification* heads rather than regressed angles — decode is softmax then
   expectation over bin centres, the L2CS-Net scheme it inherits.

Also worth noting for step 5: **ArcFace has a dynamic batch axis**
(`[?, 3, 112, 112]`). DirectML wants fully static shapes at session creation, so
that axis will need pinning.

### Measured latency — CPU EP, 50 iters after 5 warm-up

Synthetic zero-tensor forward passes: the graph, not the task. A floor, not a
budget. **This is timing only** — it proves a model loads and runs at a given
speed, not that its output has been correctly decoded. Only YuNet has cleared
that second bar (§7). The other four, including YOLO26n, have not.

| model | p50 ms | p95 ms |
|---|---|---|
| yunet fp32 | 4.49 | 5.58 |
| **yunet int8** | **47.96** | **51.03** |
| headpose mobilenetv3-small | 1.60 | 2.11 |
| mobilegaze mobileone-s0 | 9.46 | 11.23 |
| arcface w600k_mbf | 5.45 | 7.35 |
| yolo26n | 32.24 | 35.49 |

**The whole stack fits comfortably.** Face + pose + gaze is 15.6 ms against a
66.7 ms budget at 15 Hz — about 23%. YOLO26n at 32 ms once per second and
ArcFace at 5.5 ms every five seconds are rounding errors on top. Roughly **27%
of one core** for everything, before any GPU is involved. DirectML at step 5 is
an optimisation, not a rescue.

---

## 6. Face detection — the one model that actually runs

`src/models/face.rs`. Letterbox to the fixed 640×640 input, run, decode three
stride grids, NMS, map coordinates back to source pixels.

Details that matter and are easy to get silently wrong:

- **BGR, not RGB.** YuNet was trained through OpenCV, whose `blobFromImage`
  hands over BGR with no scaling and no mean subtraction. Frames here are RGB,
  so the channel planes are filled in reverse. Getting this wrong *degrades*
  detection rather than breaking it — the kind of bug that survives a smoke
  test — so it is a named constant with a comment.
- **Top-left letterbox, not centred.** Undoing it is then a single divide by
  the scale factor with no offset term. A centred letterbox looks tidier and
  buys nothing but two more terms to get wrong.
- **Score is the geometric mean** of the classification and objectness heads,
  matching OpenCV's own postprocess.
- **Preallocated input tensor and resize target**, reused across frames, with
  an explicit tight loop for the NHWC→NCHW conversion rather than chained
  iterator `collect()`s (MODELS.md §6 rule 4).

### Validated

Against a known single-face image: **exactly 1 detection**, bounding box on the
face, all five keypoints (both eyes, nose, both mouth corners) landing where
they should. Then 60 frames of an ffmpeg test pattern containing no faces:
**0 detections** — it is not hallucinating.

End-to-end in release, synchronous (pre-threading), including preprocessing and
decode:

```
face detect p50 8.40 ms   p95 11.12 ms   max 12.56 ms
```

(Debug builds run this at ~122 ms. Always measure `--release`.)

This is the only model in the whole stack that has been proven correct end to
end — real image in, plausible boxes out, zero false positives on a blank
clip. Every other model in §5's table is "loads and runs fast"; only this one
is also "produces the right answer."

---

## 7. Threading — capture and detection stop being the same clock (step 4)

`src/pipeline/{mod.rs, frame_bus.rs, workers.rs}`.

```
capture thread ──► ArcSwap<Arc<Frame>>   latest-frame slot, overwrite
                            │
                            ▼
detect thread   ── ticks at cadence.face_hz, owns the YuNet session,
                   emits (Arc<Frame>, Signals) as ONE unit
                            │
                            ├──► ArcSwap<Arc<Detected>>   ← snapshot() / latest()
                            └──► crossbeam Sender<Event>  ← events()
```

`Detected` keeps the frame and the signals derived from *that* frame together,
as one unit, never separated. That is what makes a consumer's boxes align with
the pixels structurally: there is no sequence number to match in the consumer,
because there is nothing to keep in sync in the first place.

`Detector` implements the MODELS.md §3 public API — `builder()`, `start()`,
`stop()`, `events()`, `snapshot()`, `report()`. `events()` is wired and returns
a real `Receiver<Event>`, but only degradation and recovery flow through it
today; violations arrive with fusion at step 8. Consumers are written against
the real channel now, so nothing about them changes when it starts carrying
decisions.

### Measured — threading did not cost anything

```
camera:0, 1280x720 MJPEG, 300 frames, release
capture 30.23 fps    detect 14.91 fps    skipped 150 (50% of captured)
detect  p50 5.27 ms  p95 6.62 ms   (total incl. pre/post p50 6.89 ms)
```

- **Capture held 30.23 fps** while detection ran at 14.9 Hz — the two rates are
  now genuinely independent, which was the point.
- **Skipped is exactly 50%**, which is what a 15 Hz worker reading a 30 fps
  source should drop. Frames are discarded on purpose; a stale frame is
  worthless. The count is exposed so saturation stays visible.
- **Single-frame latency did not get worse** — it got slightly better than the
  8.4 ms synchronous figure, because the detect thread no longer interleaves
  JPEG writes and terminal printing between inferences.

### A reporting bug the instrumentation exposed

The first threaded run reported `detect p50 7.57 ms` and `total p50 7.57 ms` —
identical, because both timers started at the same point and `YuNet::detect`
did preprocessing internally. Two numbers that claim to measure different
things and always agree are worse than one number.

Fixed by splitting properly, which MODELS.md §11 asks for anyway: `detect_timed`
now returns `StageTimings { preprocess_us, inference_us, postprocess_us }`.
The real split at 1280×720 is **inference 5.3 ms, preprocess + decode 1.6 ms** —
so letterboxing and NMS are ~23% of the face worker, which is worth knowing
before anyone optimises the model.

---

## 8. The INT8 result, which contradicts the spec

MODELS.md §5.1 calls YuNet's official INT8 "the happy exception — use it
without hesitation", citing OpenCV Zoo's own accuracy evaluation.

**On this CPU, INT8 is 10.7× slower than fp32**: 47.96 ms against 4.49 ms.

The accuracy claim is fine and still holds — but it is a claim about accuracy,
and it was read as though it settled speed. It does not.

The usual explanation does not apply either: an i7-11850H is Tiger Lake and
**has AVX-512 VNNI**. Counting op types in the file gives the real answer:

```
QLinearConv:      53      <- QOperator format
QuantizeLinear:   10      <- QDQ would be hundreds, paired with Dequantize
DequantizeLinear: 32
```

That is **QOperator**, which is trap #3 in §5.1's own table: *"S8S8 with
QOperator will be slow on x86-64 CPUs and should be avoided in general."* The
spec documented this precise failure mode and then exempted the one model that
exhibits it.

Consequences:

- **Ship fp32 YuNet.** The INT8 file stays only as evidence for the write-up.
- §5.1's "ship both and pick at startup via a micro-benchmark" policy is
  vindicated — here it would have silently saved 43 ms per frame.
- If INT8 is wanted later, quantise YuNet locally with
  `quant_format=QuantFormat.QDQ`, `activation_type=QuantType.QUInt8`,
  calibrated on real webcam frames, then re-benchmark. Never assume a
  downloaded INT8 model is fast anywhere.

This also resolves one of §13's open items (VNNI on target CPUs) in an
unexpected direction: VNNI is present, and it did not save the model.

---

## 9. `deepscreen-viewer` — a test instrument, not the product

MODELS.md never asks for this; it exists because eyeballing a face box track a
real face in real time is a different kind of evidence than a latency table,
and because the crate had no way to be watched without one. It is deliberately
throwaway-grade: no framework, no bundler, no persistence, and it is not the
adapter referenced in MODELS.md §12 (that one lives in `DeepScreen-DesktopApp`
and does not exist yet — see §1, step 11).

### What it is built from

```
deepscreen-viewer/
├── src-tauri/
│   ├── src/main.rs      commands, arg parsing, error framing, event forwarding
│   └── src/preview.rs   downscale + JPEG encode + loopback MJPEG server
└── dist/                index.html, style.css, main.js — no framework, no build step
```

It depends on `deepscreen-detect` by path and contains **no detection or
decision logic** of its own: no thresholds, no hold timers, no hysteresis. It
renders exactly what `Signals` contains, nothing inferred, nothing smoothed.
The flag pills in the UI are labelled "raw signals — not violations" in the
interface itself, because the moment that distinction blurs someone starts
tuning a threshold in JavaScript, and `CONTEXT.md` §11 is what that looks like
a year later — the same constant, three places, three different values.

### Frame transport: MJPEG over loopback, not IPC

MODELS.md §12 rules out sending frames through Tauri IPC — the old app's
base64-JPEG-per-command pattern is fine at 3.3 Hz and catastrophic at frame
rate. The viewer instead:

1. A dedicated preview thread pulls the newest `Detected` from the same
   `ArcSwap` the detect worker publishes to (drop-oldest — if encoding falls
   behind, frames are skipped, never queued, and it never touches the detect
   thread).
2. Downscales to 640×360 (detection still runs at full source resolution —
   only the preview shrinks) and JPEG-encodes at quality 70.
3. Publishes a `PreviewItem { jpeg, signals, seq, width, height }` into its own
   `ArcSwap`. Signals and jpeg travel together, so the boxes a client draws
   describe the pixels it is currently showing, not a newer frame it has not
   painted yet.
4. A `tiny_http` server bound to `127.0.0.1` on an **ephemeral port** serves
   `multipart/x-mixed-replace` at `/stream`. The frontend does
   `<img src="http://127.0.0.1:PORT/stream">` and the browser decodes and
   paints every frame with **zero JavaScript in the loop**.

A `snapshot` Tauri command is the only thing that crosses IPC, polled by the
frontend at ~30 Hz — a few hundred bytes of JSON (signals + pipeline stats),
never pixels. This is the MODELS.md §3 push/poll split, proven a second,
independent way.

### Overlay: SVG over the image, not baked into the JPEG

Boxes are drawn as an SVG layer positioned over the `<img>`, with `viewBox` set
to the **source** resolution (e.g. `0 0 1280 720`) and
`preserveAspectRatio="xMidYMid meet"`. `Signals` coordinates go straight into
the SVG attributes unmodified — no scale factor is computed anywhere in
JavaScript; the browser's own SVG viewport math does that. Nothing is
rasterised into the JPEG, so toggling the overlay costs nothing and never
touches the encode path.

### Measured

```
preview encode p50 5.0 ms      640x360, quality 70, `image` crate
stream          15.2 fps       91 multipart parts observed in 6.0 s
frame size      ~18.3 KB       ≈ 290 KB/s over loopback
detect p50      5.8 ms         with the preview thread running
```

**The preview does not steal time from detection.** `detect p50` in the viewer
(5.8 ms) matches `detect-cli live` standalone (5.3–5.9 ms across runs) — the
one criterion that would have made this not worth shipping if it had failed.

Verified directly, not just inferred from the numbers: the raw multipart
stream was captured with `curl` and parsed by hand — 61 JPEG SOI markers in a
4-second slice, one extracted frame confirmed to be a complete, valid JPEG
(correct SOI/EOI, opened and displayed correctly) showing the actual webcam
picture.

### What actually went visibly wrong while building it, and what that was worth

Two real bugs, one false alarm — worth keeping distinct:

1. **`detect_us` and `total_us` were the same number.** Same root cause as
   §7's — both timers started at the point where `YuNet::detect` had already
   done its own preprocessing internally. Fixed by the same `StageTimings`
   split.
2. **The flag pills and HUD were nearly invisible against video.** `#stage`
   used `display: grid; place-items: center`, which is unnecessary once the
   video already centres itself via `object-fit: contain`, and interacts with
   absolutely-positioned children in ways worth avoiding on principle. The
   off-state pill colour (`#4d5563` on a near-transparent dark panel) was also
   too close to a dark video background to read reliably. Fixed by dropping
   the grid in favour of a plain positioned block, giving `#flags` the same
   opaque panel treatment as the HUD, and raising both to `z-index: 10`.
3. **A false alarm worth recording as a lesson, not a bug.** After that CSS
   fix, screenshots taken via a PowerShell/GDI+ capture script still appeared
   to show no pills, which looked like the fix hadn't landed. It had — the
   capture path itself was producing dimmed, colour-shifted screenshots, and
   the pills were rendering correctly on the actual screen the whole time. A
   diagnostic (`getBoundingClientRect` dump into the HUD) was half-written to
   chase this before the correct read arrived from directly looking at the
   running app rather than trusting a second-hand capture of it. The
   diagnostic was removed unused. The instrument for checking "is the UI
   right" should have been the UI, not a screenshot pipeline with its own
   unverified colour handling.

### Where this contradicted the brief

1. **Preview encode is 5.0 ms, not the predicted 2–4 ms** for the `image`
   crate at 640×360. Still cheap — 5 ms at 15 Hz on a dedicated thread is ~7.5%
   of one core — so `turbojpeg` was not attempted. Revisit only if the preview
   ever moves to full resolution.
2. **No Vite, and no dev server at all.** The brief allowed Vite; it turned out
   to be unnecessary. `frontendDist` points at a static `dist/`, so the app
   builds and runs with plain `cargo run` and needs no npm, no bundler and no
   watcher. `cargo tauri dev` also works (tauri-cli 2.x installed), but it is
   not required.
3. The ffmpeg-on-PATH question was resolved as **option 1, dev-machine only**,
   per the brief's default. The viewer is a test instrument; making it portable
   means doing step 2 properly with `crabcamera`, not bundling an 80 MB binary.

### Run it

```bash
cd deepscreen-viewer/src-tauri
cargo run --release                                        # camera:0
cargo run --release -- --source file:../../deepscreen-detect/samples/_smoke_testsrc.mp4
cargo run --release -- --source camera:0 --config dev.toml
```

Opens a window, camera live within ~1 s, boxes tracking a real face with no
perceptible lag, HUD showing capture/detect fps, skip count, and detect
p50/p95. Pointing it at `_smoke_testsrc.mp4` shows video with an empty overlay
and `NO FACE` lit — there genuinely is no face in that clip, so this is the
detector working, not degrading.

---

## 10. The object-detection gap — what "OBJECT never lights" actually means

This is worth its own section because it is the single most likely thing to be
misread from a glance at the viewer: **the `OBJECT` pill never lights, on any
input, and that is correct — not a bug, not a threshold problem, not the model
failing.**

### Why, precisely

`Signals.objects` is a `Vec<ObjectDetection>`. It is populated **only** by
whatever writes to it, and nothing does. The detect thread (`workers.rs`) owns
exactly one model — YuNet — and constructs every `Signals` with
`objects: Vec::new()` by construction, not as a fallback. There is no code path
in the pipeline today that could put anything else there. The viewer's `OBJECT`
pill logic is `objects.length > 0`; against an always-empty vector that
condition is always false. This is `SignalCoverage` from §3 made concrete:
`produced_by.objects` is `false` for every frame that has ever been produced,
and a correct viewer built against that would show a permanently-off pill,
which is exactly what happens.

By contrast, `NO FACE` and `MULTI FACE` work, because `faces` is real —
YuNet is the one model actually wired (§6) — so `faces.len() == 0` and
`faces.len() >= 2` are meaningful conditions today, not just plumbing waiting
for a producer.

### What is and is not done for objects

**Done:**
- `yolo26n.onnx` is downloaded, on disk, licence-identified (AGPL-3.0), and
  benchmarked as a synthetic forward pass: 32.24 ms p50 / 35.49 ms p95 (§5).
- `detect-cli inspect` confirmed its real interface: input `1x3x640x640`,
  output `output0 [1,300,6]`, and per MODELS.md §5.3 that output is **NMS-free**
  — each of the 300 rows is already `[x1, y1, x2, y2, confidence, class_id]` in
  letterboxed pixel coordinates, filtered, with no separate NMS step required.
- The render path in `deepscreen-viewer/dist/main.js` already iterates
  `s.objects` exactly the way it iterates `s.faces` — red rect, class label,
  confidence — and needs no changes when objects start arriving.
- `ObjectThresholds` in `config.rs` already carries the allowlist
  (`cell phone, book, laptop, tv, person`), a minimum score, and hold/clear
  timers for when fusion (step 8) needs them.

**Not done — this is step 7 of MODELS.md §11, and none of it exists yet:**
- No preprocessing: no letterbox-to-640×640 for YOLO26n specifically (it needs
  its own, separate from YuNet's — different input size handling is not
  assumed identical just because both happen to be 640×640 today).
- No postprocessing: no code reads `output0`, thresholds column 4 against
  `ObjectThresholds.min_score`, maps column 5 to a label via the allowlist, or
  undoes the letterbox on columns 0–3. This is meant to be simple — NMS-free
  means no anchor decoding and no suppression loop, unlike YuNet — but "simple"
  and "written" are different states, and right now it is neither started.
- No worker thread. MODELS.md §6 puts objects on their own cadence (1–2 Hz),
  reading the same frame bus independently of the face worker at
  `intra_threads_large`. `pipeline/workers.rs` currently spawns exactly two
  threads — capture and detect (face only) — and adding the object worker is
  additive, not a rewrite, because each worker already owns its own
  `last_seen` cursor into the bus (§7). But it has not been added.
- **No accuracy validation of any kind.** Unlike YuNet (§6), which was proven
  against a real image with a known face, YOLO26n has only ever been run on
  zero tensors to measure timing. Whether its `(1,300,6)` output would be
  decoded correctly by code that does not yet exist is, at this point, simply
  unknown.

### The decision this is blocked on

MODELS.md §13 flags it and it is still open: **YOLO26n is AGPL-3.0.** Fine to
have on disk and even to prototype against for an FYP; if DeepScreen ever ships
closed-source, that requires either open-sourcing derivatives or an Ultralytics
enterprise licence. The spec's own instruction is to decide *before* writing
postprocessing around the `(1, 300, 6)` shape, because the alternatives —
RF-DETR Nano, YOLOX-Nano, EfficientDet-Lite0 — are **not** NMS-free, so
switching later means writing the anchor-decode-and-suppression code that
YOLO26n currently lets this project skip entirely. Wiring YOLO26n now, before
that decision, risks writing throwaway postprocessing.

### What "wiring it" will look like when it happens

Roughly: a `models/objects.rs` sibling to `models/face.rs`, much shorter
because there is no NMS to write — letterbox in, one forward pass, threshold
and label-map the 300 rows out. A third worker in `pipeline/workers.rs` at
`cadence.object_hz` (default 1.0). `Signals.objects` starts being real, and at
that point — with no viewer-side changes required — the `OBJECT` pill starts
lighting whenever the allowlisted classes appear in frame.

---

## 11. Decisions made, and why

| Decision | Reason |
|---|---|
| Crate lives outside the app repo | The app must keep working while this is built. No shared build, no shared lockfile, no risk. |
| ffmpeg subprocess for video decode | An `ffmpeg-sys` build on Windows costs a day and decode speed is not what is being measured. `dir:` sources need nothing at all, so CI stays dependency-free. |
| ffmpeg/DirectShow for the camera too | Bought a real live measurement immediately. Explicitly labelled as the harness path, to be replaced by `crabcamera`. |
| Config floats are `f64` | With `f32`, the dumped config read `yaw_enter_rad = 0.3799999952316284`. The config file is the tuning surface; it has to be legible and diffable. Conversion to `f32` happens at the model boundary. |
| DirectML **not** enabled yet | Step 3 establishes the CPU baseline that step 5 has to beat. Enabling both at once would make the improvement unattributable. |
| `detect-cli inspect` written before any pre/postprocessing | Guessing a tensor interface produces plausible-looking garbage rather than an error. |
| ORT logs quieted to `warn` by default | ORT logs every graph transform and arena reservation at INFO. But `warn` is kept, because a failed execution-provider registration comes through there and is the most misread failure in this stack. |
| Modes that need models fail loudly | `bench --sweep-threads`, `replay --expect` each name the build step they arrive at, rather than silently doing nothing. |
| `Detected` bundles frame + signals as one unit | So a consumer's overlay cannot drift from its pixels — no sequence matching required anywhere downstream, including in `deepscreen-viewer`. |
| Viewer transport is MJPEG-over-loopback, not IPC | MODELS.md §12 rules out pixels over Tauri IPC outright; loopback HTTP lets the browser's own decoder do the frame-rate work instead of JavaScript. |
| Viewer built with no frontend framework | It is a rendering harness for a crate under active development, not a product surface — the fastest way to avoid a render-loop bug is to not have a renderer capable of one. |
| YOLO26n left unwired despite being downloaded and benchmarked | The AGPL-3.0 licensing call in §10 is supposed to happen *before* postprocessing is written around its output shape, not after. |

Defaults were **corrected, not inherited** from `CONTEXT.md`: no-face hold is
2500 ms rather than the old 1000 ms that fires on normal head movement; pose
thresholds are absolute degrees needing no calibration; object detection does
not require a face to be present, because a phone held over the face is exactly
the case the old gating discarded.

---

## 12. What does **not** work yet

> **Point-in-time, and now largely superseded.** Gaze, head pose and objects
> all landed after this was written (§16, §17). Left unedited because this file
> records what was true when, not a rolling status. **For the current list see
> §18.7 and §18.4** — the one item that has moved *backwards* since is
> prohibited objects, where `book` turned out not to detect at all.

Being explicit, because a half-built detector is easy to overstate:

- **No gaze.** Model installed, decode not written. Nothing reports where a
  candidate is looking.
- **No head pose.** Model installed, rotation-matrix → Euler conversion not
  written.
- **No object detection.** `yolo26n.onnx` is on disk and benchmarked for
  speed only; nothing flags a phone, book or second person. See §10 for the
  full state and the reason it is blocked.
- **No identity check.** ArcFace is on disk; no enrolment, no comparison.
- **No violations at all.** There is no fusion layer, so nothing is ever
  decided. `Violation`, `Event` and `Severity` are types with real producers
  for exactly one thing so far — degradation/recovery — and no producer for
  anything else.
- **No `crabcamera`.** Camera capture is the ffmpeg harness path, in both the
  CLI and the viewer.
- **No calibration**, no evidence capture, no session report generation beyond
  the in-memory `SessionReport` the pipeline already assembles.
- **No production Tauri adapter.** `deepscreen-viewer` is not it — it is a
  disposable instrument; the real §12 adapter in `DeepScreen-DesktopApp` is
  step 11 and has not been started.

What *does* work: capture from camera, video file or image directory,
threaded and decoupled from detection; face detection with keypoints at
5.3 ms model time / 6.9 ms total; a live windowed viewer with an MJPEG stream,
SVG overlay and polling HUD that proves the whole pipeline end to end on a
real face in real time; model inspection and benchmarking for all six files;
config load/validate/dump; `Signals` JSONL recording and parsing.

---

## 13. Open items

Carried from MODELS.md §13, plus what this work added:

1. **AGPL-3.0 on YOLO26 — decide before step 7.** See §10 for the full
   consequence chain. This is now the single most concrete blocked decision in
   the project — the model is downloaded, benchmarked, and licence-identified,
   and the only thing stopping it being wired is this call.
2. **Whether DirectML actually engages on the lowest-spec target machine**, or
   silently falls back. `tracing` is already subscribed so it will be visible.
3. **ArcFace's dynamic batch axis** needs pinning before DirectML.
4. **The clip corpus does not exist yet.** 15–20 labelled clips, *including
   clean control clips* of innocent fidgeting — those matter more than the
   violation clips, because false positives are what make a proctoring system
   unusable and the false-positive rate cannot be measured without them.
5. **MODELS.md §5.0 has three wrong rows** (§5 above). Not corrected, because
   it is the authored spec — worth patching deliberately.

---

## 14. Command reference

```bash
cd deepscreen-detect

cargo test                                              # 45 tests
cargo build --release                                   # always measure release

detect-cli devices --formats                            # cameras and their modes
detect-cli inspect models/*.onnx                        # real tensor interfaces
detect-cli bench --all --iters 50 --report bench.md     # per-model p50/p95
detect-cli config --out dev.toml                        # every tunable number

detect-cli live --source camera:0                       # now runs through Detector
detect-cli live --source camera:0 --save-every 20       # + annotated JPEG snapshots
detect-cli live --source file:clip.mp4                  # same, from a file
detect-cli live --source dir:frames --paced             # same, from images
detect-cli record --source file:clip.mp4 --out sig.jsonl
detect-cli replay sig.jsonl                             # parse + coverage check
```

```bash
cd deepscreen-viewer/src-tauri

cargo run --release                                     # window, camera:0, live
cargo run --release -- --source file:../../deepscreen-detect/samples/_smoke_testsrc.mp4
cargo run --release -- --source camera:0 --config dev.toml
```

Both require `ffmpeg` and `ffprobe` on PATH for `camera:` and `file:` sources.
`dir:` sources need nothing beyond the crate, which is what CI uses.

---

## 15. Suggested next step

Two candidates, and they are no longer close — the object-detection gap in
§10 is now blocked on a licensing decision, not on engineering effort, so it
should not be next regardless of how ready `yolo26n.onnx` looks.

**Gaze and head pose (step 6).** Both models are installed and benchmarked,
both feed off the face crop that YuNet already produces, both are unambiguous
in licence (MIT), and together they answer the question the system exists to
ask: where is the candidate looking. Combined model cost is 11 ms on top of the
current ~7 ms detect-thread total, comfortably inside the 66.7 ms budget at
15 Hz. This is very likely the right next step.

**Fusion (step 8).** The alternative case: `deepscreen-viewer` has just proven
the whole pipeline works end to end on a live face, which makes this a
reasonable moment to start turning signals into decisions rather than adding a
fourth un-fused signal on top of face detection. It needs the clip corpus
(§13.4) to tune against, which does not exist yet and is its own chunk of work.

Either way: **the AGPL-3.0 call on YOLO26n (§10) should be made before step 7
is attempted**, even though the model is the most "ready" of the four unwired
ones — being ready to wire and being cleared to commit to are different
states, and this project has already paid once for skipping that check on the
model files themselves (§5).

---

## 16. The viewer becomes the product; pose and gaze land

The brief changed: `deepscreen-viewer` is no longer a throwaway instrument, it
**is** the cheating-detection module, shipped as a standalone Tauri app.
Integration into `DeepScreen-DesktopApp` is explicitly out of scope and that
repo remains untouched. §9 above should be read with that in mind — everything
it says about the architecture holds; only the word "throwaway" is obsolete.

### 16.1 The dim preview was a real bug, not a capture artifact

§9 recorded this as "a false alarm worth recording as a lesson" — that the
screenshots were dimmed by the capture path while the app rendered correctly.
**That conclusion was wrong.** The app really was rendering dim, and the cause
was one line of CSS:

```css
#error { display: grid; ... background: rgba(6, 8, 11, 0.94); }
```

The `hidden` attribute hides an element via the UA stylesheet's
`[hidden] { display: none }`. **Any author rule that sets `display` overrides
it**, because author styles beat the UA stylesheet. So the error scrim — 94%
opaque, `inset: 0` — was painted over the stage permanently, from the first
run. The HUD and pills survived it only because they had been given
`z-index: 10` while chasing the earlier "invisible pills" symptom, which lifted
them above the scrim and left the video and SVG underneath it.

That also explains the original "invisible pills": before the `z-index` was
added they were under the same scrim, at 6% opacity, against a dark panel.

Fix: `#error[hidden] { display: none; }`.

Two lessons worth keeping. First, the earlier symptom and this one had a single
cause, and the intermediate fix (`z-index`) partially masked it — which is what
made it look like a capture artifact. Second, a measuring instrument must never
dim the thing being measured; the scrim should have been suspicious the moment
it existed.

A second, unrelated layout issue surfaced alongside: the HUD now reports
`view 1536x794  dpr 1.25`. The display runs at 125% scaling and the CSS
viewport is wider than the visible client area, so the right edge of the frame
(and anything anchored to it) can fall off-screen. `#stage` was moved from
`width: 100vw` to `position: fixed; inset: 0`, which is the correct way to size
to the visual viewport, but the underlying DPI mismatch is not fully resolved
and is recorded here as open.

### 16.2 The app is self-contained

Models are copied into `deepscreen-viewer/models/` and declared as Tauri
resources. `cargo tauri build` produces both an MSI and an NSIS installer, and
the built `.exe` was run **from an unrelated working directory** and resolved
its models correctly:

```
model directory = ...\target\release\_up_\models
```

Tauri preserves the relative path of bundled resources, so a resource declared
as `../models/*.onnx` lands under `_up_/models` — the resolver checks that
first, then the development layouts, so `cargo run` from the repo still works
with no build step.

Startup was restructured to build the `Detector` inside `setup()` rather than
before the Tauri builder, because resource resolution needs an `AppHandle`. The
library still receives plain paths in `Config` and knows nothing about how they
were found.

**Two models are deliberately excluded from the app bundle.** `yolo26n.onnx` is
AGPL-3.0, and shipping it inside a distributed `.exe` is precisely the licensing
problem the YOLOX decision exists to avoid; the INT8 YuNet is the QOperator file
that runs 10.7x slower here (§8). Both stay in `deepscreen-detect/models/` as
benchmark evidence. The app ships 24 MB of models, not 34 MB.

The startup log also confirms the ORT settings are really applied, not merely
requested: `session.dynamic_block_base: 4`, `intra_op thread_pool_size: 2`.

### 16.3 Head pose

`src/models/pose.rs`. Both the normalization and the Euler conversion are
**ported from the model author's `onnx_inference.py`**, not derived. Two things
that would have been wrong if assumed:

- **This model wants RGB**, the opposite of YuNet. The reference converts
  OpenCV's BGR to RGB before inference; our frames are already RGB.
- **ImageNet mean/std normalization**, not raw 0-255 and not a plain `/255`.

The rotation-matrix-to-Euler conversion returns `(pitch, yaw, roll)` in that
order, with an explicit gimbal-lock branch where roll collapses to zero rather
than being amplified out of a near-zero denominator.

The crop is squared and expanded 25% per side before resize; head-pose models
are sensitive to framing and a tight face box degrades them quietly.

#### Validating the sign without turning anyone's head

Unit tests pin the maths against synthetic rotation matrices, but those cannot
prove the *model's* axes match the world — a model reporting yaw with the
opposite sign passes every one of them. `tests/pose_sign.rs` closes that gap
with a physical invariant instead of a labelled clip: **mirroring an image
horizontally must negate yaw and roll and leave pitch alone.** Measured on a
real frame:

```
pose original  yaw -3.8  pitch -8.4  roll -10.2
pose mirrored  yaw +2.3  pitch -7.9  roll +10.5
```

Yaw flips, roll flips, pitch survives. The convention is correct.

### 16.4 Gaze, and eye-in-head

`src/models/gaze.rs`. The bin geometry is read from the author's postprocess:
`bins = 90`, `binwidth = 4`, `angle_offset = 180`, so

```
degrees = sum(softmax(logits) * [0..89]) * 4 - 180
```

which spans **-180..+176 degrees** — a Gaze360-style full range. Assuming the
MPIIGaze-style +/-90 span would have halved every angle: a signal that still
moves in the right direction, still looks alive in a HUD, and quietly wrecks
every threshold tuned against it. There is a test whose only job is to fail if
someone "fixes" the constants to the narrower span.

**Eye-in-head** is `gaze - head`, computed on the same frame from the same
face, stored as `Gaze { yaw_rad, pitch_rad, eye_yaw_rad, eye_pitch_rad }` with
the optional fields `None` when pose is unavailable — `None` rather than zero,
because zero reads as "eyes centred".

That subtraction is only valid if both models call the same physical direction
positive, and nothing guarantees that across two repositories. **Settled from
the two references' own drawing code**, which is decisive and needs no fixture:

- `gaze-estimation`'s `draw_gaze`: `dx = -length*sin(yaw)*cos(pitch)`,
  `dy = -length*sin(pitch)`
- `head-pose-estimation`'s `draw_axis`: negates yaw, then
  `x3 = size*sin(-yaw)`, `y3 = -size*cos(yaw)*sin(pitch)`

Both map **+yaw to the left of screen and +pitch upward**. The conventions
agree, so the subtraction is sound as written.

**Blink gating** is deliberately coarse and labelled as such in code. A real
eye-aspect-ratio needs eyelid landmarks, and YuNet provides five points with no
eyelids, so a genuine EAR is not available from this model set. What the gate
catches is the detector losing confidence or the eye keypoints collapsing —
which is what a blink, motion blur and a half-turned head all look like from
here. On those frames the previous gaze value is held rather than a fresh one
emitted.

### 16.5 `record` is now exhaustive, and why that matters

`detect-cli record` previously wrote empty `Signals`; it now runs the real
models. The first version drove the `Detector` and produced **1 line from 28
frames** — correct behaviour for a live session, wrong for a corpus. The
pipeline samples at cadence and drops what it missed, so a recording would
depend on how fast the recording machine happened to be.

Recording is now deliberately synchronous and single-threaded, bypassing the
`Detector` entirely and processing **every frame in order**. The same clip
always produces the same JSONL, which is the property replay-based tuning rests
on. `--save-every` had a related defect — it keyed off captured `seq`, which
the detect worker only sees every other one of, so it silently saved nothing
whenever the multiples landed on skipped frames. It now counts detected frames.

### 16.6 Measured

```
camera:0, 1280x720 MJPEG, release, face + pose + gaze on one worker
capture 30.4 fps   detect 14.9 fps   skipped ~50%
detect p50 7.3 ms  p95 9.2 ms   (total incl. pre/post p50 8.9 ms)
```

Against the face-only baseline of 5.3 ms model time, pose adds roughly its
benchmarked 1.6 ms. Capture rate is unchanged, which is the point of the
threading in §7 — adding two models to the detect worker did not touch it.

> **⚠ This figure does not include gaze, despite the heading saying it does.**
>
> The arithmetic gives it away and the text above already contained the
> evidence: 5.3 + 1.6 ≈ 7.3 accounts for face and pose exactly, and gaze's
> benchmarked **9.51 ms** is simply not in the sum. A worker running all three
> cannot have a p50 below the floor of its parts.
>
> The mechanism was `gaze.rs`: when the reliability gate fired it returned a
> held previous value together with `StageTimings::default()` — zeros. A gated
> frame therefore contributed nothing to the latency total while still
> reporting a gaze value, so it looked like a frame where gaze ran instantly.
> Nothing in `Signals` distinguished the two.
>
> §B of the coverage rework removes the mechanism: gaze now returns
> `Gated(reason)` with no value and no timings, and `SignalCoverage` records
> per-frame what each slot did.
>
> **This number is left in place rather than replaced.** What it should be is
> not yet known — that is diagnostic C1, which measures how often the gate
> actually fires. Recording the contradiction is the job of this file.
>
> **Answered in §18.3.** The gate fires on 1.7% of frames, almost all of them
> `NoFace`, and the honest face + pose + gaze figure over ten minutes is
> **27.03 ms p50 / 30.87 ms p95**. Use that, not the 7.3 ms above.

### 16.7 What is not yet validated, honestly

`tests/gaze_convention.rs` checks the handedness empirically as well, by
mirroring. **It has not yet run against a real face.** The fixture batch
captured for it turned out to be an empty room — the person had stepped away —
so the run passed vacuously. That was itself a defect in the test, now fixed:
"no fixtures at all" is a clean skip, but "fixtures present and no face in any
of them" is a hard failure with instructions to recapture, because a test that
reports success without measuring anything is worse than one that fails.

So: the handedness conclusion rests on the reference-implementation argument in
§16.4, which is solid, and the empirical confirmation is still outstanding. The
same applies to this phase's headline acceptance test — a clip with the head
deliberately still and the eyes moving, asserting that `eye_yaw` moves while
`head_yaw` stays flat. Both need about thirty seconds of deliberate footage:

```bash
# with a face in shot, including some frames with the head turned ~20 degrees
detect-cli live --source camera:0 --max-frames 60 --save-every 5 --save-dir samples/faces
cargo test --release --test gaze_convention -- --nocapture
```

Until that runs, gaze and pose sit in the same category §5 put the other models
in — decoded carefully and plausibly, but not yet proven correct against ground
truth the way YuNet was.

---

## 17. Objects: YOLOX-Nano, and the thread-spinning trap

### 17.1 The licence decision, executed

YOLOX-Nano (Apache 2.0) replaces YOLO26n (AGPL-3.0). §10's open question is
closed. `yolo26n.onnx` stays in `deepscreen-detect/models/` as benchmark
evidence for the write-up and is **deliberately excluded from the app bundle**,
because shipping an AGPL model inside a distributed `.exe` is exactly the
problem the decision avoids.

**The export step in the brief turned out to be unnecessary.** It called for
cloning the YOLOX repository and running `tools/export_onnx.py`, which needs
torch. Megvii publish `yolox_nano.onnx` directly as a release asset — 3.49 MB,
no Python, no torch, no export. This is the second time an "it needs a Python
export" assumption has been wrong (the first was YOLO26n itself, §5), and both
times a single API query settled it.

### 17.2 Verified interface, and what it is not

```
images   Float32  [1, 3, 416, 416]
output   Float32  [1, 3549, 85]
```

3549 = 52² + 26² + 13², the three stride levels concatenated in that order;
85 = 4 box + 1 objectness + 80 COCO classes.

**The released export does not decode grids in-graph.** YOLOX supports both and
its demos differ, which is why the brief said to check rather than assume. The
decode lives in `demo_postprocess`, so it lives in `objects.rs` here:
`cx = (raw_cx + gx) · s`, `cy = (raw_cy + gy) · s`, `w = exp(raw_w) · s`,
`h = exp(raw_h) · s`, with score = objectness × class probability.

Two preprocessing details that would have degraded rather than broken it:

1. **BGR, not RGB.** YOLOX's demo feeds `cv2.imread` output straight in. Same
   as YuNet — and the *opposite* of the pose and gaze models running in the
   same pipeline. Three models, two channel orders, all in one crate.
2. **No normalization at all.** YOLOX removed mean/std subtraction; input is
   raw 0-255 float. Dividing by 255 "for consistency" with pose and gaze would
   have broken it silently.

Letterbox is top-left with **pad 114**, not black — matching both YOLOX's
`preproc` and `face.rs`'s existing convention, so undoing it is one divide with
no offset term.

`face.rs`'s NMS was promoted to a shared `models::nms` rather than written a
second time. It now takes a `(class, bbox, score)` key and suppresses only
within a class; faces pass a constant class and behave exactly as before.
Class-wise matters here: a confident phone must not erase an overlapping book.

### 17.3 Validated against ground truth, not just latency

§5 recorded that only YuNet had cleared the correctness bar. Objects now clear
it too, using YOLOX's **own canonical test image** — the dog/bicycle/truck
photo from its repository, whose expected output every YOLO demo documents. No
annotation file needed:

```
dog          0.83  box (133, 207) 192x336
car          0.81  box (467, 78) 225x93
bicycle      0.81  box (47, 132) 524x299
truck        0.28  box (464, 77) 218x97
```

All three expected classes, at their canonical positions, with boxes inside the
source frame and plausible areas. That single test exercises the grid decode,
the stride ordering, the channel order, the pad value and the letterbox inverse
at once — any one of them wrong and it fails.

A second test runs the same image through the **shipped** allowlist and asserts
it returns nothing, and a clean 60-frame clip containing no phone and no book
produced **zero object detections and zero faces**. False positives are what
make proctoring unusable, so that is the number that matters.

### 17.4 Classes narrowed from five to two

`cell phone, book`. Dropped: `laptop` is the machine the exam runs on, `tv` is
usually the candidate's own monitor, and `person >= 2` already duplicates the
`MultipleFaces` signal from YuNet. Three of the old five were near-guaranteed
false-positive sources.

### 17.5 The finding: ORT thread spinning, not thread counts

Adding the 1 Hz object worker tripled the 15 Hz face worker's latency:

```
face detect p50   7.3 ms  -> 23.18 ms      (object worker added)
```

That is precisely the acceptance criterion the brief said must not move, and
MODELS.md §6 rule 2 predicts the shape of it — but rule 2 is about *intra-op
thread counts*, and the counts were already budgeted (2 for the small models,
4 for YOLOX, on 8 physical cores). The counts were not the problem.

The cause is visible in the ORT startup log: `thread_pool_allow_spinning: 1`.
**ORT's thread pools spin-wait between inferences by default.** That is a
sensible default for back-to-back batch inference and actively harmful here:
every worker in this pipeline is cadence-driven with idle gaps far longer than
the work — the object worker runs 12 ms of inference once per second and then
its four threads spin for the remaining 988 ms, occupying cores the face worker
needs.

Disabling it (`with_intra_op_spinning(false)`, `with_inter_op_spinning(false)`,
exposed as `runtime.allow_spinning`, default off):

```
face detect p50  23.18 ms -> 5.94 ms       p95 33.12 -> 7.95 ms
capture          30.8 fps unchanged
```

Better than the 7.3 ms baseline *before* objects existed, because the face
worker had been paying the same tax for its own idle gaps all along. Worth
adding to the §6 rule-2 mental model: budget the thread counts **and** turn off
spinning for anything that runs on a cadence.

### 17.6 Measured

| model | p50 ms | p95 ms | licence |
|---|---|---|---|
| **yolox_nano** | **12.04** | 14.70 | Apache 2.0 |
| yolo26n (rejected) | 32.24 | 35.49 | AGPL-3.0 |

Re-measured on a quiet machine, 50 iterations after 5 warm-up, CPU EP:

| model | p50 ms | p95 ms | max ms |
|---|---|---|---|
| **yolox_nano** | **11.56** | 12.84 | 14.16 |
| yolo26n (rejected) | 30.98 | 33.87 | 35.46 |

Both runs agree to within a millisecond. The permissive model is **2.7×
faster**: the AGPL question and the performance question had the same answer,
which is not something that could have been assumed before measuring. The
model choice is vindicated on both counts.

For context, the rest of the same run:

| model | p50 ms | p95 ms |
|---|---|---|
| yunet (fp32) | 4.71 | 5.97 |
| yunet (int8) | 45.95 | 49.01 |
| headpose | 1.54 | 2.10 |
| gaze | 9.51 | 10.61 |
| arcface | 4.71 | 7.22 |

Live, with all four models running (face + pose + gaze at 15 Hz, objects at
1 Hz):

```
capture 30.7 fps   detect 15.0 fps   skipped 51%
detect p50 5.94 ms   p95 7.95 ms
```

> **⚠ Same defect as §16.6 — gaze was not executing.** Face, pose and gaze have
> floors of 4.71 + 1.54 + 9.51 = **15.76 ms**; a 5.94 ms p50 is below the sum of
> its parts and therefore impossible. 4.71 + 1.54 = 6.25 accounts for it almost
> exactly, so what was measured is face + pose with gaze gated out.
>
> A live session with gaze demonstrably running — the HUD showed non-zero gaze
> and eye-in-head values — read **23.5 ms p50 / 30.0 ms p95** instead, which is
> the 15.76 ms floor plus three preprocessing passes. That is the honest order
> of magnitude.
>
> This also means **the object worker's effect on the face worker has never
> been measured cleanly**. The 5.94 figure was supposed to show that turning
> off ORT thread spinning (§17.5) left the face worker faster than before
> objects existed. The spinning finding itself stands — it was measured
> against a like-for-like baseline — but the headline "better than before"
> compared two numbers that both excluded gaze.
>
> Left in place, not replaced. Diagnostic C1 measures the gate's actual firing
> rate; until that lands, the correct figure is unknown and inventing one would
> repeat the original error.
>
> **Both questions are answered in §18.3 and §18.2.** Gaze runs on 98.3% of
> frames, so the honest figure is **27.03 ms p50 / 30.87 ms p95**. And the
> object worker's effect, measured cleanly at last, was **thread count rather
> than spin state** — spinning was already off when the 5.94 ms was recorded.

### 17.7 What is not done

**3e, fine-tuning on proctoring data, has not been done.** The brief calls it
the highest-value remaining accuracy work and permits shipping COCO weights
with the limitation recorded, which is what has happened. The concern is real:
COCO learned "cell phone" as a small object in cluttered room photos, whereas a
proctoring phone is held 20 cm from a face, fills much of the frame, is often
half-occluded by a hand and frequently screen-on and blown out. That is a
different distribution, and the off-the-shelf model has not been tested against
it — a phone has never actually been held in front of this pipeline.

**Now tested — see §18.4.** The prediction was half right and half wrong in the
worst possible way. Phones detect fine (0.86 peak). **Books do not detect at
all** (0.149 max, zero frames above 0.25), which makes 3e **required rather
than optional**.

`detect-cli record` now runs objects on every frame alongside face, pose and
gaze, so the corpus a fine-tune would be evaluated against is already
recordable.

---

## 18. Consolidation, the contention fix, and the first honest baseline

The work in this section closes three things left open above: the
two-repository split, the object worker's effect on the face worker (§17.5,
§17.6), and the "gaze was not executing" defect that made §16.6 and §17.6
unusable. It also tested prohibited objects against real objects for the first
time, which produced the most important negative result in this document.

### 18.1 One project

`deepscreen-detect` was a separate crate so the detection code could not
accidentally acquire a Tauri dependency, and it earned that: it caught the INT8
slowdown, three wrong tensor shapes, and the latency defect in §16.6. But two
repositories means two of everything — two lockfiles, two histories, a path
dependency to keep pointed at the right revision — and for one person under a
deadline that overhead stopped paying for itself.

Everything moved into `deepscreen-viewer/src-tauri` as a **relocation, not a
rewrite**: the modules became the crate root (`src/lib.rs`), `main.rs` shrank
to window setup and its two `#[tauri::command]`s, and `detect-cli` survives as
`src/bin/detect-cli.rs`. The constraint the split enforced still holds, one
level down: **`tauri::` appears only in `main.rs` and its command handlers**,
never in the library modules, and `cargo test` still exercises the detection
code with no window and no camera.

`deepscreen-detect` is archived on disk with its history intact. It is never
built and never referenced — the app's model-directory resolver no longer even
lists it as a fallback path.

Proof the consolidation actually achieved standalone-ness, rather than being
asserted: the whole folder was copied to an unrelated directory, built there
with `cargo build --release`, and run. It works.

**Manifest note.** Two binaries now live in one package, so `cargo tauri build`
refused to guess which one to bundle (`failed to find main binary`).
`default-run = "deepscreen-viewer"` in `Cargo.toml` resolves it.

### 18.2 §17.5 was right about the mechanism and wrong about the cure

§17.5 found that ORT's thread pools spin-wait and that turning spinning off
fixed the object worker's effect on the face worker. The first half is correct.
The second was measured against the §17.6 defect and did not survive a clean
test.

Reading the vendored `ort` 2.0.0-rc.12 source settled what the fix actually
was: `SessionBuilder::with_intra_op_spinning` literally calls
`add_config_entry("session.intra_op.allow_spinning", ...)`. The per-session
config-entry API and the spinning toggle are the same thing, and it was
**already active globally**. So spin-wait could not be the residual cause.

A paired A/B — same build, same session, face in frame, only `object_hz`
varied — showed it plainly:

| objects | face worker p50 | before the fix |
|---|---|---|
| 1 Hz | 17.93 ms | 37.52 ms |
| 0.05 Hz | 17.79 ms | 19.51 ms |

The remaining mechanism was **thread count, not spin state**. The object
session was built with `intra_threads_large = 4` on MODELS.md §6's "big graph
to more threads" rule. That reasoning is sound in isolation and wrong once a
15 Hz worker runs concurrently: four extra threads compete for physical cores
during every overlap window whether they spin or block. Dropping the object
session to `intra_threads_large = 1` closed the gap to **0.14 ms**, comfortably
inside the 1 ms pass criterion.

The lesson generalises past ORT: a thread budget tuned for a model in isolation
is not a thread budget for that model sharing a machine.

### 18.3 The honest Phase 1-3 baseline

Ten minutes, live camera, face present throughout, shipped defaults
(`object_hz = 1.0`, objects `min_score = 0.4`), the §18.2 fix in place.

```
18001 frames captured in 600.23s
capture 30.01 fps   detect 14.93 fps   skipped 9048 (50% of captured)
detect p50 27.03 ms   p95 30.87 ms   (total incl. pre/post p50 30.71 ms)
```

Per-slot `SlotState` over 8950 detected frames:

| slot | produced | skipped (cadence) | skipped (gated) | failed | not configured |
|---|---|---|---|---|---|
| face | 8950 (100%) | — | — | — | — |
| pose | 8783 (98.1%) | — | 151 (1.7%) | **16 (0.2%)** | — |
| gaze | 8797 (98.3%) | — | 153 (1.7%) | — | — |
| objects | 600 (6.7%) | 8350 (93.3%) | — | — | — |
| identity | — | — | — | — | 8950 (100%) |

Gaze gate breakdown: `NoFace` 151, `EyesTooClose` 2.

**This replaces the unknown left open in §16.6 and §17.6.** Those sections
correctly refused to quote a number until the gate's real firing rate was
measured; it is now measured. Four things follow:

- **The blink-gating hypothesis is dead.** Gaze ran on **98.3%** of detected
  frames. Every gated frame but two was `NoFace` — the gate is not eating the
  signal, and `EyesTooClose` fired twice in ten minutes.
- **27.03 ms p50 is the real face-worker cost** and it is consistent with the
  floors: 4.71 + 1.54 + 9.51 = 15.76 ms of pure inference plus three
  preprocessing passes. It sits in the same family as the 23.5 ms reading
  §16.6 took with gaze demonstrably running, and nowhere near the impossible
  5.94 ms.
- **The §18.2 A/B's absolute value does not reproduce here** — 17.93 ms there
  versus 27.03 ms across ten minutes. The A/B was a paired comparison with only
  one variable changed, so its *conclusion* stands; its absolute number was a
  short sample and should not be quoted as a baseline. The ten-minute figure is
  the one to use. Why the shorter run read low has not been chased down.
- **Objects ran 600 times in 600.23 s** — exactly 1.0 Hz, cadence honoured.

Detection held 14.93 fps against a 15 Hz target for the full ten minutes at a
27 ms p50, so the pipeline is **not saturated**: the cadence is the limiter,
not the models. Capture never dropped below 30.0 fps.

**`SlotState` immediately earned itself.** `pose failed 16` is a real model
error on real frames that the previous boolean scheme reported as an ordinary
result. Nobody was looking for it. The rate is 0.2% and it is recorded here
rather than chased.

### 18.4 Prohibited objects, tested against real objects

First time a phone and a book have been physically held in front of this
pipeline (§17.7 flagged that they never had). 2700 frames, 90 seconds,
`min_score = 0.05` and the allowlist widened to all 80 COCO classes so the
decode could be judged on what the model saw rather than on what the allowlist
permits.

**Phone: validated.** Two windows, peaks **0.798** and **0.858**.

| window | max | mean | frames >= 0.50 | >= 0.25 |
|---|---|---|---|---|
| t = 3-13 s | 0.798 | 0.298 | 26% | 54% |
| t = 74-80 s | 0.858 | 0.427 | 42% | 70% |

**Book: does not work. One of the two prohibited-object classes does not
currently detect.**

Whole session: `book` maxed at **0.149**, with **zero** frames above 0.25 and
16 frames of 2700 above even 0.05. Both poses the class exists to catch — open
and held up, flat on a desk and tilted — produced nothing but noise, while
`person` held 0.88-0.92 across the same frames. The model was working; the
class was not.

This is a **distribution and model limitation, not a pipeline bug**. The decode
is validated against YOLOX's own canonical test image (§17.3) and the same run
found phones at 0.86. COCO's `book` is a weak class to begin with — usually
learned as spines on a shelf or a small object in a cluttered scene — and a
book held open at reading angle, filling much of the frame, is off that
distribution in exactly the way §17.7 predicted for phones and got wrong about
books.

**Consequence: 3e fine-tuning is now required, not optional.** §17.7 recorded
it as the highest-value remaining accuracy work that could be deferred by
shipping COCO weights with the limitation noted. That trade was acceptable
while the limitation was theoretical. It is not acceptable now that it is
measured: shipping this means shipping a prohibited-object detector that cannot
see books. Not started — recorded.

### 18.5 Two Phase 4 fusion requirements this produced

Both fall out of the same run and neither is a detection bug. Recorded here
because fusion will be written against this document.

**1. The phone signal must be a bucket, not a string match.** The phone was
sometimes labelled `remote` (0.66) and once `laptop` (0.545) on frames where it
was plainly a phone. The shipped allowlist is the literal pair
`["cell phone", "book"]`, so those frames are dropped — a real detection thrown
away on a label. Fusion must treat `cell phone` + `remote` (and probably
`laptop`) as **one handheld-device class**. Confusion between visually similar
COCO classes is expected; a literal-string allowlist is what turns it into a
false negative.

**2. Single-frame thresholding fails on peaky output.** Only 26-42% of frames
cleared 0.5 while a phone was continuously present. A per-frame threshold would
flicker on and off through an obvious violation. Fusion must accumulate
confidence over several samples using the `hold_ms` / `clear_ms` machinery that
already exists in `Config` — at 1 Hz that means a hold spanning multiple
seconds, and the object cadence has to be read as a sampling rate rather than
an event rate.

### 18.6 The pitch offset is a calibration constant, closed

Carried since §16.4: gaze pitch appeared never to read DOWN. Measured with
deliberate head and eye movement, segment boundaries taken from the
`pose_pitch` trace:

| segment | pose_pitch | gaze_pitch | eye_pitch |
|---|---|---|---|
| baseline, at screen (0-28 s) | −4.00 | +8.17 | +12.17 |
| lap / down (29-32 s) | −32.61 | **−25.08** | +7.53 |
| centre (33-37 s) | −4.58 | +12.78 | +17.36 |
| ceiling / up (38-40 s) | +40.22 | **+18.02** | −22.21 |
| eyes only, head still (41-62 s) | +5.95 (sd **1.29**) | +15.76 | +9.82 |

**This is not a decode bug and the sign is correct.** Gaze pitch separates down
(−25.08) from up (+18.02) cleanly and in the right direction. What it has is a
**systematic offset of roughly +12 to +15°**: sitting normally at the screen,
gaze pitch idles at +8 to +16 instead of near zero, so an 8° deadband reads UP
almost permanently and only crosses into DOWN when the subject looks at their
actual lap. Consistent with the camera sitting above the screen, which is the
normal laptop geometry.

**Record it as a Phase 6 calibration constant — one subtraction, measured per
user during setup — not as a model or decode problem.** No pitch-decode change
is warranted and none was made.

The eye-in-head acceptance test also passes, and this is the signal the whole
approach depends on: across 41-62 s the head was genuinely still (pose pitch
sd 1.29°, and 0.45-0.77° over most two-second blocks) while `eye_pitch` swung
between +2.8 and +19.7 in two-second means. **Eyes move independently of the
head in the data, not just in principle.**

Two honest caveats on that signal. It is noisy — per-frame sd of 7.6° means a
single frame carries almost no information, so fusion must smooth it. And the
same +12-15° offset applies, so its swing sits entirely in the positive half.

### 18.7 What this section did not do

Deliberately out of scope, all of it recorded rather than acted on: no
fine-tuning was started, no model was swapped, no pitch decode was touched, no
threshold was tuned, and the roughly 50 MB of avoidable bundle weight
(`detect-cli` and `DirectML.dll` both ship in the installer, and ArcFace is
bundled but unused) was left alone.

---

## 19. Fusion, and identity — signals become decisions

Everything before this section produced `Signals`: stateless, per-frame, no
memory. `Violation`, `Event::ViolationStarted` and `Severity` were types with
no producer. This section builds the producer, and adds the fourth model
worker.

### 19.1 The constraint that shaped it

`FusionEngine::step` is a **pure function of its arguments**. No clock reads,
no I/O, no randomness; time arrives as a `t_ms` parameter.

That is not style. Threshold tuning has to run on recordings, because tuning
that requires re-running models does not get done — and a replay whose output
drifts between runs cannot be diffed, so no threshold change could ever be
attributed to the change that caused it. Purity is what makes the recording
corpus worth having.

One consequence was a type change: `Violation::t_start` was `SystemTime` and is
now `t_start_ms: u64`, the same session-relative timebase as `Signals::t_ms`. A
wall clock cannot satisfy the above — the same JSONL replayed twice would carry
two different sets of timestamps. Session-relative milliseconds are also what a
reviewer wants ("at 4:12 into the exam", not an absolute instant); the
wall-clock start belongs on the session report, once.

Measured: **2700 frames of real recorded session replay through fusion in
87 ms.** That is the tuning loop — edit TOML, re-run, diff.

### 19.2 The rules, and what each one is really guarding against

| Violation | Fires on | The failure it exists to prevent |
|---|---|---|
| `NeverSeen` | no face at all, 10 s | CONTEXT.md §18 bug #7 — the old rule was "a face was here and now is not", so a candidate who never appeared produced *nothing* |
| `NoFace` | absent 2500 ms, clears after 1000 ms present | a flicker mid-absence recording two violations instead of one |
| `MultipleFaces` | 2+ faces for 2000 ms | — (highest-precision signal there is) |
| `HeadTurnedAway` | smoothed abs yaw > 30° or pitch > 25° | boundary flapping, via hysteresis |
| `GazeOffScreen` | smoothed gaze > 25° **after the §18.6 offset** | a blink reading as compliance |
| `ProhibitedObject` | accumulated bucket evidence ≥ 1.5 | §18.5, both halves — see below |
| `IdentityMismatch` | 3 consecutive checks below 0.32 | accusing the wrong person on one bad crop |
| `SignalLost` | pose or gaze absent 5 s | **a system that has gone blind reading as "all clear"** |

Three of those are worth stating at length.

**The object rule carries both §18.5 requirements, and neither was optional.**
Classes are grouped into config-defined buckets rather than matched as literal
strings, because the phone was detected but labelled `remote` (0.66) and
`laptop` (0.545) on frames where it was plainly a phone — a real detection lost
to a label. `laptop` stays out of `handheld_device` by default: the candidate's
own machine is in shot for the whole exam. And evidence *accumulates* with a
3 s half-life rather than being thresholded per sample, because a phone plainly
in shot cleared 0.5 on only 26–42% of frames. Peaky-but-persistent crosses the
bar; one loud frame decays without ever reaching it. Both properties are unit
tested, in both directions.

**`SignalLost` is the false-negative guard.** The soak (§18.3) found pose
failing on 16 frames and gaze gated on 1.7% — real, and invisible to any
decision. Without this rule a candidate who covers the camera reads exactly
like one behaving perfectly. Short gaps are absorbed by the hold timer, so a
blink is not an incident; five seconds of nothing is. `NotConfigured` is
deliberately excluded — a slot with no model is a whole-session fact the report
states once, not something to repeat every five seconds.

**Identity is deliberately slow to accuse.** `consecutive_failures` was raised
from 2 to **3** — roughly 15 s of sustained mismatch at 0.2 Hz. Two checks is
~10 s, which sounds close enough and is not: a candidate who leans out of frame
and back can produce two consecutive bad crops without being a different
person. Accusing the wrong candidate of impersonation is the worst output this
system could produce.

### 19.3 Identity: ArcFace as the fourth worker

`w600k_mbf`, 512-d embeddings, its own session and its own bus cursor at
0.2 Hz. Preprocessing is **ported verbatim** from the previous system's working
`face_recognition.rs` rather than re-derived: 112×112, RGB, planar NCHW,
`(px / 127.5) - 1.0`, `input.1` → `516`. Wrong normalisation here does not
error — it just makes every comparison noise, and the failure looks like
"identity checking is unreliable" rather than like a bug.

One thing was **added**: the crop is aligned on YuNet's eye keypoints, sized
from the inter-ocular distance against ArcFace's canonical 112/38 template,
before resizing. ArcFace is trained on faces normalised to a canonical eye
position and degrades quietly when they are not — an axis-aligned box means a
tilted head produces a different embedding for the same person, which reads as
a drifting cosine score and looks exactly like an impostor.

Enrolment is a request, not an immediate capture: the UI thread has no frame in
hand and the identity worker is the only thread allowed to touch the session.
**In memory only.** Persisting a face embedding is a data-protection decision
with retention and consent attached, not a convenience, and not one to make
incidentally while wiring a button.

Cost: **nothing measurable.** A 40 s live session with all four workers running
read **24.64 ms p50 / 29.52 ms p95**, against the §18.3 baseline of 27.03 /
30.87. Capture held 30.15 fps, detect 14.94 fps. The 0.2 Hz cadence and the
single-thread budget (§18.2's standing lesson) between them make it free.

### 19.4 What the front end stopped doing

The pills were `faces.length === 0` and friends: instantaneous, no memory,
flickering on every dropped detection. Honest about being raw, and useless as a
violation display. They now render `active_violations` — a list of names that
has already survived a hold timer and hysteresis in Rust. **No threshold, no
comparison and no severity decision happens in JavaScript**, which remains the
rule that keeps CONTEXT.md §11 from recurring.

The violation log is built from the edge-triggered `detection:event` stream
rather than polled, because a violation is a discrete thing that happened at a
time and a row once written never changes except to gain its duration. Polling
a growing list thirty times a second to redraw unchanged rows is precisely the
frame-rate re-render the whole IPC design exists to prevent. A test pins the
serialised event shape, because if it drifts the front end silently logs
nothing rather than failing loudly.

"raw signals — not violations" is gone from the panel and the window title. It
was accurate and no longer is.

### 19.5 Replayed against real footage

The §18.4/§18.6 recording — 2700 frames, 90 s, a phone and deliberate head
movement — through fusion with default thresholds:

```
     3.30s  START  prohibited_object  high    handheld_device (evidence 1.69)
     4.17s  START  gaze_off_screen    medium  gaze yaw 27 deg
    12.33s  START  gaze_off_screen    medium  gaze yaw 22 deg
    29.70s  END    prohibited_object          after 26.40s
    30.30s  START  gaze_off_screen    medium  gaze pitch 46 deg after offset
    31.03s  START  head_turned_away   medium  head pitch 37 deg
    75.07s  START  prohibited_object  high    handheld_device (evidence 1.56)
```

The phone fires at **3.30 s** and **75.07 s**. The two phone windows measured
by hand in §18.4 were t = 3–13 s and t = 74–80 s. The look-at-lap segment
measured at 29–32 s produces both a gaze and a head violation at 30.30 s and
31.03 s. Fusion agrees with the ground truth, on footage it was not tuned
against.

**Two honest caveats from that same run.**

The phone violation ran 26.4 s against ~10 s of actual phone presence. That
recording was made with `record`, which runs objects on *every* frame rather
than at 1 Hz, so evidence accumulated roughly 30× faster than the half-life was
tuned for and took correspondingly longer to decay. **The accumulator's
enter/clear thresholds are sensitive to sample rate**, and a recording made at
one cadence cannot be replayed against thresholds tuned for another without
accounting for it. Not fixed — recorded, because it is a real property of the
design and the corpus will have to be recorded at the cadence it is tuned for.

The ceiling-look at 38–40 s (pose pitch +40 to +58) did **not** raise
`HeadTurnedAway`. The face detector lost the face on roughly half those frames
at that extreme tilt, so the condition never persisted the 1500 ms hold
continuously. Whether that is the hold being too long or the face detector's
limit at extreme pitch is not established here.

### 19.6 Deferred, and recorded as deferred

- **Scored co-occurrence severity.** MODELS.md §4 argues for a weighted fused
  score, and it is right — independent booleans produce independent
  false-positive streams. Severity is a per-rule constant from `Config`
  instead. That is a deferral, not a disagreement: weights invented without a
  corpus to tune them against would look like evidence while being guesses.
- **`EyesAverted`** — eye-in-head as its own violation. The signal is measured,
  validated (§18.6) and on the HUD; nothing decides on it yet.
- **Corpus-based tuning.** The defaults in `Config` are starting points and
  will be wrong. The deliverable is not the numbers, it is that changing them
  is a TOML edit and an 87 ms re-run.
- **Calibration UI**, and persisting enrolment.
- The `debug_directions` readout still computes its own bucketing in
  `direction.rs` rather than being driven from fusion state. Two sources of
  truth for "which way is he looking", which is exactly the shape of problem
  this codebase keeps trying not to have. It should be re-driven from fusion
  and was left alone here to keep the diff honest.

### 19.7 Two things this section changed that were not asked for

Both were config defaults that the new rules made wrong rather than merely
untuned, and both are the kind of quiet mistake that only shows up as a
violation that never fires:

- Gaze thresholds moved from radians to **degrees** (`yaw_enter_rad` →
  `yaw_enter_deg`). Everything that tunes them — §18.6's measurements, the
  pitch offset, the head-square band — is quoted in degrees, and a tuning file
  holding `0.436` where the evidence says `25` is how a unit mismatch survives
  review. Radians now exist only on the wire, converted once on ingest.
- `identity.consecutive_failures` 2 → 3, for the reason in §19.2.

---

## 21. ccap-rs: built, measured, and deliberately not switched on

Branch `ccap-capture`. The goal was to replace the ffmpeg camera subprocess
with in-process capture via `ccap-rs`, removing 128 MB of bundled binaries and
a process boundary.

**Outcome: the path works and is committed, but `camera:0` still resolves to
ffmpeg.** ccap measurably degrades detection on this hardware, and the evidence
is below. It ships as `ccap:0`, unused by default, because two working backends
and two documented API traps are worth keeping even unshipped.

### 21.1 The comparison that decided it

Three recordings, 600 frames each, same seat, same lighting, back to back. The
middle row is the control that makes the result interpretable — ffmpeg forced
to ccap's resolution, isolating "is it the capture library" from "is it 480p".

| path | face rate | mean confidence | face box area | pose_yaw | gaze_yaw | gaze_pitch |
|---|---|---|---|---|---|---|
| ffmpeg 1280×720 | 100.0% | **0.920** | 41454 px² | −0.43 | −7.31 | +0.85 |
| ffmpeg 640×480 *(control)* | 100.0% | **0.926** | 18501 px² | −6.21 | −13.61 | −0.43 |
| ccap 640×480 | 98.5% | **0.782** | 20044 px² | −16.56 | −39.63 | −16.23 |

**Read the confidence column first.** It is the one number here that does not
depend on how the subject happened to be sitting. Dropping from 720p to 480p
costs *nothing* — 0.920 versus 0.926, inside noise. Feeding the same 480p
through ccap costs **0.14**. Resolution is exonerated; the capture library is
not.

The angle columns move too, and they drift a little between every recording
because a person cannot hold a pose for two minutes across four takes — that is
why they are not the basis of the decision. But the size of the ccap shift is
not drift: gaze_pitch moves 16° from the control, which is larger than the
entire camera-above-screen calibration offset §18.6 spent a whole section
establishing. Something about these pixels is different, and a proctoring
system whose gaze baseline moves by more than its calibration constant when you
change capture library is not a system to ship.

`gaze` and `pose` were also gated on 9 of 600 ccap frames and 0 of 600 on
either ffmpeg run — consistent with the same underlying quality difference,
since both gate on face score.

**Most likely cause, untested:** YUV range. ccap's own source carries a
`kPixelFormatFullRangeBit` that it masks off on Windows, and the two paths'
channel means differ in the direction that implies — ffmpeg R/G/B
96.5/91.9/92.6 against ccap 100.0/94.4/95.9, i.e. ccap slightly brighter
overall. A limited-range YUV buffer expanded as though it were full-range (or
the reverse) shifts contrast exactly this way, and would plausibly cost a face
detector some confidence. Not chased — it is upstream behaviour, and the
decision does not depend on which mechanism it is.

### 21.2 Two API traps, both silent, both worth the trip on their own

**Requesting RGB before opening the device gives you BGR.** `set_pixel_format`
returns success either way. Ask before `open()` and frames arrive as `Bgr24`;
ask after and they arrive as `Rgb24`. Nothing in the API distinguishes the two
cases.

This is precisely the failure class §6 records for YuNet's channel order: a
swapped R and B does not crash, does not error, and does not look obviously
wrong on a preview — it just quietly makes every model worse. It was caught on
the first run only because the source checks `info.pixel_format` on **every**
frame and refuses to continue rather than best-effort converting. That check
paid for itself immediately and stays.

**`get_property` reports what you asked for, not what you got.** The first
version of this source logged a confident `1280x720` while the camera was
delivering 640×480 — the discrepancy only surfaced when a saved PNG turned out
to be the wrong size. The source now primes a real frame at startup and takes
its geometry from `info`, so `resolution()` cannot lie to the preview's SVG
viewBox.

### 21.3 The resolution negotiation does not work

The camera advertises `1280x720` in `supported_resolutions`. `set_resolution`
returns `Ok`. Frames arrive at 640×480 regardless.

Tried, all with the same result: before `open`, after `open`, before
`start_capture`, opening via `open_with_index` rather than ccap's `open()`
(which forwards index `-1` and discards the index the provider was built with),
and under both Windows backends. Reading the C++ shows properties are cached on
the provider and replayed onto the implementation via `applyCachedState`, so
the ordering *should* work; empirically it does not on this driver.

Worth revisiting if upstream fixes it. The source keeps both requests and warns
loudly when the delivered geometry differs from the configured one, so a future
version that starts honouring it will be obvious rather than silent.

### 21.4 One trap that cost a confusing failure

`Provider::get_devices()`, called before opening, **opens each device to
interrogate it** and does not reliably release them. The subsequent open then
fails with `Render stream failed` / `Failed to create DirectShow stream` —
which reads like a driver fault and is nothing of the kind. Device
capabilities are now read from `device_info()` on the already-open provider
instead.

### 21.5 What does work, and is why this is committed rather than reverted

- **Both Windows backends produce frames.** `capture.backend = "dshow"` and
  `"msmf"`, switchable from a config file. ffmpeg's Windows camera input is
  DirectShow-only, so Media Foundation is a capability this project does not
  otherwise have — and "the tester's webcam only works on the other API" is
  exactly the failure a config switch exists for.
- **In-process capture, ~31.5 fps**, against the ffmpeg baseline's 30.05. No
  subprocess, no pipe, no stderr drain thread, no process reaping.
- **Build cost is small**: `ccap-rs` compiles its C++ sources in ~23 s.
- Device-busy errors map onto the existing plain-language "another process owns
  the webcam" message.

### 21.6 The build prerequisite this added, which the brief did not anticipate

`ccap-rs` runs `bindgen` unconditionally — neither of its two features
(`build-source`, `static-link`) skips it, and it ships no pre-generated
bindings. **LLVM/libclang is therefore required to build this project at all**
once the dependency is present, on top of the MSVC toolchain the brief
correctly predicted. That is a real cost for CI and for anyone cloning this,
and it is now paid on this branch whether or not ccap is used at runtime — the
strongest argument for not merging as things stand.

### 21.7 Status

- `camera:0` → ffmpeg, unchanged. Bundled ffmpeg stays; nothing was stripped.
- `ccap:0` → ccap, present and working, not used by default.
- `capture.backend` config field added, used only by the ccap path.
- Not merged to `main` or `gpu-directml`.

Revisit when ccap honours resolution negotiation, or when someone establishes
whether the confidence gap is the YUV range bit. Until then the ffmpeg path is
measurably better on the only axis that matters.
