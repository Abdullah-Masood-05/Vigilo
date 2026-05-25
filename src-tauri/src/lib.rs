//! `deepscreen-viewer` — camera frames in, proctoring signals out, one project.
//!
//! This started as two repositories: a headless detection library
//! (`deepscreen-detect`) and a thin Tauri app consuming it by path dependency.
//! That split earned its keep early — it let every model be benchmarked
//! outside a window manager, and `detect-cli` caught the INT8 slowdown, three
//! wrong tensor shapes, and a faceless-benchmark error that a GUI would have
//! hidden behind "seems fine." But two projects also meant two of everything:
//! two locks, a commit-hash pin that had to be re-pinned on every library
//! edit, two READMEs to keep in sync. For one person under deadline that
//! overhead outweighed the isolation. `deepscreen-detect/` is retired,
//! left on disk as an archive.
//!
//! **The constraint that mattered survives, just enforced at a different
//! grain.** It used to be "this crate has no `tauri` dependency" (MODELS.md
//! §0). Now the crate does — `preview` and the Tauri commands are part of it —
//! so the constraint is: `tauri::` is used only in `main.rs` and its command
//! handlers. Every module below knows nothing about windows, IPC, or the
//! webview, and is exercised the same way it always was — through
//! `detect-cli`, and through `cargo test` with no camera and no window.
//!
//! # Build status
//!
//! | Step | What | State |
//! |---|---|---|
//! | 1 | Crate skeleton, types, config, `detect-cli`, file replay | **done** |
//! | 2 | Camera capture behind `FrameSource` | **done** |
//! | 3 | YuNet face detection + baseline bench | **done** |
//! | 4 | Threading skeleton, `ArcSwap` frame bus, `Detector` | **done** |
//! | 5 | DirectML | not started |
//! | 6 | Pose + gaze | **done** |
//! | 7 | Objects (YOLOX-Nano) on its own worker | **done** |
//! | 8 | Fusion, record/replay tuning | not started |
//! | 9 | ArcFace identity | model wired, not yet used |
//! | 10 | Quantization | not started |
//! | 11 | Tauri adapter | **done** — this crate |

pub mod capture;
pub mod config;
pub mod direction;
pub mod error;
pub mod models;
pub mod pipeline;
pub mod preview;
pub mod report;
pub mod types;

pub use capture::{FrameSource, SourceSpec};
pub use config::Config;
pub use direction::{Axes, DebugDirections, DirectionTracker, FrameOfReference, Horizontal, Vertical};
pub use error::{DetectError, Result};
pub use pipeline::{Detected, Detector, DetectorBuilder};
pub use report::{FrameStats, Latencies, LatencySummary, SessionReport, SignalStatus};
pub use types::{
    BBox, Contribution, DegradeReason, DetectorState, Event, EyeAspect, FaceDetection,
    FaceKeypoints, Frame, GateReason, Gaze, HeadPose, ObjectDetection, Severity, SignalCoverage,
    SignalSource, Signals, SlotState, Violation, ViolationKind,
};

/// Version of the `Signals` JSONL format. Bump when a change would make an
/// old recording replay to different violations — recordings are the
/// regression corpus, and silently reinterpreting them would be worse than
/// refusing to read them.
///
/// **2**: `SignalCoverage` went from five booleans to five [`SlotState`]s. A
/// v1 recording says `"face": true`, which will not deserialise, so old
/// recordings are refused rather than half-read.
///
/// Unchanged by the migration into this crate — the wire format is the
/// contract with recordings on disk, not with the crate that reads them.
pub const SIGNALS_FORMAT_VERSION: u32 = 2;
