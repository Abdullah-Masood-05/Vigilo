//! DeepScreen Viewer — the exam proctoring desktop app. This is the Tauri
//! entry point: window setup, the two `#[tauri::command]`s, and nothing else.
//! The detection code this app is built on lives in the crate root
//! (`src/lib.rs`) and knows nothing about `tauri`.
//!
//! Three rules this layer does not break:
//!
//! 1. **No detection or decision logic lives here.** It renders what `Signals`
//!    contains and nothing more. No thresholds, no hold timers, no hysteresis.
//!    `CONTEXT.md` §11 is what happens when the same constant lives in three
//!    places with three different values — one source of truth, in `Config`.
//! 2. **No frames over IPC.** Preview is an MJPEG stream on loopback; the only
//!    thing crossing the Tauri boundary is a few hundred bytes of JSON per poll.
//! 3. **`snapshot` is polled, never pushed.** This is MODELS.md §3's design,
//!    and proving it works here means the production integration is already
//!    validated.

// Release builds should not pop a console window behind the app.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::atomic::Ordering;
use std::sync::Arc;

use arc_swap::ArcSwap;
use vigilo::config::Config;
use vigilo::preview::{self, PreviewSlot, PreviewStats};
use vigilo::types::{PipelineStats, Signals};
use vigilo::{Detector, SourceSpec};
use serde::Serialize;
use tauri::{Emitter, Manager, State};

struct ViewerState {
    detector: Option<Arc<Detector>>,
    preview: PreviewSlot,
    preview_stats: Arc<PreviewStats>,
    stream_port: u16,
    source: String,
    /// Set when startup failed. The window still opens and shows this, because
    /// the diagnosis is the whole value of an error in this layer
    /// (MODELS.md §8).
    startup_error: Option<String>,
    /// A missing prerequisite rather than a failure — the app cannot run at
    /// all until the user installs something. Rendered as a full-screen
    /// instruction instead of an error strip over a dead video pane.
    setup_blocked: bool,
}

/// Everything the HUD and overlay need, in one poll.
#[derive(Serialize, Default)]
struct SnapshotDto {
    seq: u64,
    /// Source resolution — the SVG viewBox matches this, so the frontend never
    /// computes a scale factor.
    width: u32,
    height: u32,
    signals: Signals,
    stats: PipelineStats,
    preview_p50_us: u64,
    preview_encoded: u64,
    running: bool,
    source: String,
    degraded: Vec<String>,
    error: Option<String>,
    /// Whether the picture on screen is flipped. Reported so that a direction
    /// label disagreeing with the video is diagnosable rather than a guess —
    /// see [`preview::MIRRORED`].
    preview_mirrored: bool,
    /// Violations fusion currently has raised, by their `as_str` names.
    ///
    /// Level-triggered, for the pills. The scrolling log is built from the
    /// edge-triggered `detection:event` stream instead — polling a growing
    /// list every 33 ms to redraw rows that have not changed is the shape of
    /// the bug the whole IPC design exists to avoid.
    active_violations: Vec<String>,
    /// Whether a reference face has been enrolled this session.
    enrolled: bool,
    /// Show `error` as a full-screen setup instruction, not an error banner.
    setup_blocked: bool,
}

#[tauri::command]
fn stream_port(state: State<ViewerState>) -> u16 {
    state.stream_port
}

#[tauri::command]
fn snapshot(state: State<ViewerState>) -> SnapshotDto {
    let Some(detector) = state.detector.as_ref() else {
        return SnapshotDto {
            source: state.source.clone(),
            error: state.startup_error.clone(),
            // Set explicitly rather than left to `Default`, which would happen
            // to be right today only because `false` is both.
            preview_mirrored: preview::MIRRORED,
            setup_blocked: state.setup_blocked,
            ..Default::default()
        };
    };

    // Read the signals from the same item the stream is serving, so the boxes
    // describe the pixels currently on screen rather than a newer frame the
    // browser has not painted yet.
    let item = state.preview.load_full().as_ref().clone();
    let detector_state = detector.snapshot();

    let (seq, width, height, signals) = match item.as_ref() {
        Some(p) => (p.seq, p.width, p.height, p.signals.clone()),
        None => (0, 0, 0, Signals::default()),
    };

    SnapshotDto {
        seq,
        width,
        height,
        signals,
        stats: detector_state.stats,
        preview_p50_us: state.preview_stats.encode_p50_us.load(Ordering::Relaxed),
        preview_encoded: state.preview_stats.encoded.load(Ordering::Relaxed),
        running: detector.is_running(),
        source: state.source.clone(),
        degraded: detector_state.degraded.iter().map(|d| format!("{d:?}")).collect(),
        error: detector.error().or_else(|| state.startup_error.clone()),
        preview_mirrored: preview::MIRRORED,
        active_violations: detector_state
            .active_violations
            .iter()
            .map(|v| v.as_str().to_string())
            .collect(),
        enrolled: detector.is_enrolled(),
        setup_blocked: state.setup_blocked,
    }
}

/// Capture the next usable face as the identity reference.
///
/// In memory only, for this process. Persisting a face embedding is a data
/// protection decision with retention and consent attached, not a convenience
/// — it is not one to make incidentally while wiring a button.
#[tauri::command]
fn enrol(state: State<ViewerState>) -> bool {
    match state.detector.as_ref() {
        Some(detector) => {
            detector.enrol();
            true
        }
        None => false,
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,ort=warn")),
        )
        .with_target(false)
        .init();

    let args = Args::parse();
    tracing::info!(source = %args.source, "starting viewer");

    tauri::Builder::default()
        // Everything is built inside `setup` rather than before the builder,
        // because model paths come from Tauri's resource resolver and that
        // needs an `AppHandle`. The library still receives plain paths in
        // `Config` and knows nothing about how they were found.
        .setup(move |app| {
            let handle = app.handle().clone();
            let model_dir = resolve_model_dir(&handle);
            tracing::info!(dir = ?model_dir, "model directory");

            // Point capture at our own ffmpeg before anything opens a source.
            // The library resolves nothing itself; the app knows where the
            // resource directory is and tells it (MODELS.md §9), exactly as
            // it does for models.
            let ffmpeg_dir = resolve_ffmpeg_dir(&handle);
            tracing::info!(dir = ?ffmpeg_dir, "ffmpeg directory");
            vigilo::capture::set_ffmpeg_dir(ffmpeg_dir);

            let preview_slot: PreviewSlot = Arc::new(ArcSwap::from_pointee(None));
            let preview_stats = Arc::new(PreviewStats::default());

            // Preflight before anything expensive. A missing prerequisite is
            // not a capture failure and must not be reported as one — the
            // person reading this has never seen the app and needs an
            // instruction, not a diagnosis.
            let (detector, startup_error, blocked) =
                if args.source.starts_with("dir:") || vigilo::capture::ffmpeg_available()
                {
                    match start_detector(&args, model_dir.as_deref()) {
                        Ok(d) => (Some(Arc::new(d)), None, false),
                        Err(e) => {
                            tracing::error!(error = %e, "could not start detection");
                            (None, Some(explain(&e)), false)
                        }
                    }
                } else {
                    tracing::error!("ffmpeg is not on PATH");
                    (
                        None,
                        Some(vigilo::capture::FFMPEG_MISSING_HELP.to_string()),
                        true,
                    )
                };

            // The stream server runs whether or not detection started: an empty
            // stream is better than a broken page, and the error is shown over
            // it (MODELS.md §8 — the diagnosis is the value).
            let port = preview::serve(Arc::clone(&preview_slot)).unwrap_or(0);
            tracing::info!(port, "preview stream on http://127.0.0.1:{port}/stream");

            if let Some(detector) = detector.clone() {
                let slot = Arc::clone(&preview_slot);
                let stats = Arc::clone(&preview_stats);
                std::thread::Builder::new()
                    .name("ds-preview".into())
                    .spawn(move || preview::preview_loop(detector, slot, stats))
                    .expect("spawning preview thread");
            }

            // Forward decisions to the webview. Edge-triggered and low-rate,
            // the opposite of the polled snapshot.
            if let Some(detector) = detector.clone() {
                let emit_handle = handle.clone();
                std::thread::spawn(move || {
                    for event in detector.events() {
                        let _ = emit_handle.emit("detection:event", &event);
                    }
                });
            }

            app.manage(ViewerState {
                setup_blocked: blocked,
                detector,
                preview: preview_slot,
                preview_stats,
                stream_port: port,
                source: args.source.clone(),
                startup_error,
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![stream_port, snapshot, enrol])
        .run(tauri::generate_context!())
        .expect("error while running viewer");
}

/// Where the bundled models live.
///
/// Resource directory first — that is where they are once the app is
/// installed — then the development layouts, so `cargo run` from the repo
/// still works without a build step.
fn resolve_model_dir(handle: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(resource) = handle.path().resource_dir() {
        candidates.push(resource.join("models"));
        // Tauri preserves the relative path of bundled resources, so a
        // resource declared as `../models/*` lands under `_up_/models`.
        candidates.push(resource.join("_up_").join("models"));
    }
    candidates.push(std::path::PathBuf::from("models"));
    candidates.push(std::path::PathBuf::from("../models"));
    candidates.push(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../models"));

    candidates.into_iter().find(|p| p.join("face_detection_yunet_2023mar.onnx").exists())
}

/// Where the bundled ffmpeg helpers live.
///
/// Same search order as [`resolve_model_dir`], and for the same reason: the
/// resource directory once installed, then the development layouts so
/// `cargo run` from the repo works with no build step.
///
/// `None` means fall back to whatever is on `PATH` — which is what a source
/// checkout without a downloaded `ffmpeg/` folder does, and what the startup
/// check then reports on if there is nothing there either.
fn resolve_ffmpeg_dir(handle: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(resource) = handle.path().resource_dir() {
        candidates.push(resource.join("ffmpeg"));
        // Tauri preserves the relative path of bundled resources, so a
        // resource declared as `../ffmpeg/*` lands under `_up_/ffmpeg`.
        candidates.push(resource.join("_up_").join("ffmpeg"));
    }
    candidates.push(std::path::PathBuf::from("ffmpeg"));
    candidates.push(std::path::PathBuf::from("../ffmpeg"));
    candidates.push(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../ffmpeg"));

    candidates.into_iter().find(|p| p.join("ffmpeg.exe").exists())
}

fn start_detector(
    args: &Args,
    model_dir: Option<&std::path::Path>,
) -> Result<Detector, vigilo::DetectError> {
    let mut config = match &args.config {
        Some(path) => Config::load(path)?,
        None => Config::default(),
    };

    // The crate takes model paths from `Config` and resolves nothing itself.
    // Finding the directory is the app's job.
    match model_dir {
        Some(dir) => config.models.fill_missing_from_dir(dir),
        None => {
            return Err(vigilo::DetectError::Config(
                "no model directory found — the app could not locate its bundled models".into(),
            ))
        }
    }

    let spec: SourceSpec = args.source.parse()?;
    let source = spec.open(&config.capture, args.paced)?;

    let mut detector = Detector::builder().config(config).build()?;
    detector.start(source)?;
    Ok(detector)
}

/// Turn a `DetectError` into something a person can act on.
fn explain(e: &vigilo::DetectError) -> String {
    let base = e.to_string();
    if base.contains("Could not run graph") || base.contains("already in use") {
        format!(
            "{base}\n\nWindows lets exactly one process own a webcam. \
             Close OBS, Teams, Zoom, Discord, a browser tab, or a running \
             detect-cli, then restart the viewer."
        )
    } else if base.contains("could not start ffmpeg") || base.contains("could not run ffprobe") {
        format!("{base}\n\nffmpeg and ffprobe must be on PATH for camera: and file: sources.")
    } else {
        base
    }
}

struct Args {
    source: String,
    config: Option<std::path::PathBuf>,
    paced: bool,
}

impl Args {
    fn parse() -> Self {
        let mut args = Args { source: "camera:0".into(), config: None, paced: false };
        let mut argv = std::env::args().skip(1);
        while let Some(arg) = argv.next() {
            match arg.as_str() {
                "--source" => {
                    if let Some(v) = argv.next() {
                        args.source = v;
                    }
                }
                "--config" => args.config = argv.next().map(Into::into),
                // Replay sources run flat out otherwise, which is wrong for
                // eyeballing a clip.
                "--paced" => args.paced = true,
                _ => {}
            }
        }
        // A file or directory source is only watchable in real time.
        if !args.source.starts_with("camera:") {
            args.paced = true;
        }
        args
    }
}
