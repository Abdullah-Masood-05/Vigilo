//! `detect-cli` — the standalone harness (MODELS.md §10).
//!
//! Built first, before any model integration, because the old module could not
//! be tested without a browser, a camera and a running React app — which is
//! why it was impossible to make fast. You could never measure one thing in
//! isolation.
//!
//! Four modes. At build step 1 the ones that need models say so plainly rather
//! than pretending:
//!
//! ```text
//! detect-cli live   --source file:samples/phone_01.mp4     # frame count + FPS
//! detect-cli record --source file:clip.mp4 --out sig.jsonl # Signals -> JSONL
//! detect-cli bench  --source file:clip.mp4 --iters 200     # p50/p95
//! detect-cli replay sig.jsonl --expect phone@2.1s          # fusion only
//! detect-cli config --out dev.toml                         # resolved defaults
//! ```

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};

use deepscreen_viewer::capture::camera;
use deepscreen_viewer::config::Config;
use deepscreen_viewer::models::face::YuNet;
use deepscreen_viewer::models::gaze::GazeNet;
use deepscreen_viewer::models::objects::YoloxNano;
use deepscreen_viewer::models::pose::HeadPoseNet;
use deepscreen_viewer::error::{DetectError, Result};
use deepscreen_viewer::report::Latencies;
use std::collections::BTreeMap;

use deepscreen_viewer::models::gaze::GazeOutcome;
use deepscreen_viewer::types::{FaceDetection, GateReason, SignalCoverage, Signals, SlotState};
use deepscreen_viewer::{Detector, SourceSpec};

#[derive(Parser, Debug)]
#[command(
    name = "detect-cli",
    about = "Harness for deepscreen-viewer's detection modules. No window, no camera required.",
    version
)]
struct Cli {
    /// Config file (TOML or JSON). Defaults are used when omitted.
    #[arg(long, global = true, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Log at debug level. Overridden by RUST_LOG if set.
    #[arg(long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// List capture devices and the modes they support.
    Devices {
        /// Also probe each device for its supported formats. Slower — it opens
        /// every device in turn.
        #[arg(long)]
        formats: bool,
    },

    /// Pull frames from a source and report throughput. Eyeball it.
    Live {
        /// camera:<index> | file:<path> | dir:<path>
        #[arg(long)]
        source: String,
        /// Draw a preview window. Arrives with the real capture path, step 2.
        #[arg(long)]
        overlay: bool,
        /// Sleep to the source's nominal frame rate instead of running flat out.
        #[arg(long)]
        paced: bool,
        /// Stop after this many frames.
        #[arg(long, value_name = "N")]
        max_frames: Option<u64>,
        /// Save every Nth frame as a JPEG — how you eyeball a live camera
        /// before there is a preview window.
        #[arg(long, value_name = "N")]
        save_every: Option<u64>,
        /// Where those JPEGs go.
        #[arg(long, value_name = "DIR", default_value = "snapshots")]
        save_dir: PathBuf,
    },

    /// Run models over a clip and dump Signals as JSONL. No decisions.
    Record {
        #[arg(long)]
        source: String,
        #[arg(long, value_name = "PATH")]
        out: PathBuf,
        #[arg(long, value_name = "N")]
        max_frames: Option<u64>,
    },

    /// Per-stage p50/p95, EP verification, thread sweep.
    Bench {
        /// Benchmark the capture path against this source.
        #[arg(long)]
        source: Option<String>,
        /// Benchmark one model by name. Arrives with the models, step 3+.
        #[arg(long)]
        model: Option<String>,
        /// Benchmark every model.
        #[arg(long)]
        all: bool,
        /// Sweep intra-op thread counts 1..cores (MODELS.md §6 rule 2).
        #[arg(long)]
        sweep_threads: bool,
        #[arg(long, default_value_t = 200)]
        iters: u64,
        /// Write a markdown summary here.
        #[arg(long, value_name = "PATH")]
        report: Option<PathBuf>,
    },

    /// Push recorded Signals through fusion only. Zero inference.
    Replay {
        /// A JSONL file produced by `record`.
        path: PathBuf,
        /// Expected violation, as `kind@seconds`. Repeatable.
        #[arg(long, value_name = "KIND@SECS")]
        expect: Vec<String>,
    },

    /// Report a model's real tensor interface. Run this before writing any
    /// pre- or post-processing against a published shape.
    Inspect {
        /// One or more .onnx files.
        #[arg(value_name = "PATH", required = true)]
        models: Vec<PathBuf>,
    },

    /// Print the resolved config — every tunable number, in one place.
    Config {
        /// Write here instead of stdout.
        #[arg(long, value_name = "PATH")]
        out: Option<PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    if let Err(err) = run(cli) {
        eprintln!("\nerror: {err}");
        let mut source = std::error::Error::source(&err);
        while let Some(e) = source {
            eprintln!("  caused by: {e}");
            source = e.source();
        }
        std::process::exit(1);
    }
}

/// Subscribe to `tracing` from the start. `ort` logs EP selection failures
/// through it, and a silent DirectML fallback otherwise looks exactly like
/// "the GPU didn't help" when it never engaged (MODELS.md §5.2).
fn init_tracing(verbose: bool) {
    // ORT logs every graph transform and arena reservation at INFO, which
    // buries our own output. Quiet it by default but keep WARN, because a
    // failed execution-provider registration comes through at that level and
    // is the single most misread failure in this whole stack (MODELS.md §5.2).
    let default = if verbose { "debug,ort=info" } else { "info,ort=warn" };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default));
    tracing_subscriber::fmt().with_env_filter(filter).with_target(false).without_time().init();
}

fn run(cli: Cli) -> Result<()> {
    let cfg = match &cli.config {
        Some(path) => {
            let cfg = Config::load(path)?;
            tracing::info!(path = %path.display(), "config loaded");
            cfg
        }
        None => Config::default(),
    };

    match cli.cmd {
        Cmd::Devices { formats } => cmd_devices(formats),
        Cmd::Live { source, overlay, paced, max_frames, save_every, save_dir } => {
            cmd_live(&cfg, &source, LiveOpts { overlay, paced, max_frames, save_every, save_dir })
        }
        Cmd::Record { source, out, max_frames } => cmd_record(&cfg, &source, &out, max_frames),
        Cmd::Bench { source, model, all, sweep_threads, iters, report } => {
            cmd_bench(&cfg, source.as_deref(), model.as_deref(), all, sweep_threads, iters, report)
        }
        Cmd::Replay { path, expect } => cmd_replay(&path, &expect),
        Cmd::Inspect { models } => cmd_inspect(&models),
        Cmd::Config { out } => cmd_config(&cfg, out),
    }
}

// ---------------------------------------------------------------------------
// devices
// ---------------------------------------------------------------------------

fn cmd_devices(formats: bool) -> Result<()> {
    let devices = camera::list_devices()?;
    println!("{} capture device(s):\n", devices.len());

    for dev in &devices {
        println!("  camera:{}  {}", dev.index, dev.name);
        if let Some(alt) = &dev.alt_name {
            println!("            {alt}");
        }

        if formats {
            match camera::list_formats(dev) {
                Ok(modes) if modes.is_empty() => {
                    println!("            (no modes reported)");
                }
                Ok(modes) => {
                    // Highest frame rate first, then largest frame — the
                    // interesting question is what this camera can sustain.
                    let mut modes = modes;
                    modes.sort_by(|a, b| {
                        b.fps
                            .partial_cmp(&a.fps)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then((b.width * b.height).cmp(&(a.width * a.height)))
                    });
                    modes.dedup_by(|a, b| a == b);
                    for m in &modes {
                        println!("            {m}");
                    }
                }
                Err(e) => println!("            could not probe: {e}"),
            }
        }
        println!();
    }

    println!("Use one with:  detect-cli live --source camera:<index>");
    Ok(())
}

// ---------------------------------------------------------------------------
// live
// ---------------------------------------------------------------------------

/// Per-slot `SlotState` counts over a live run.
///
/// `live` reported latency and frame counts but nothing about *which signals
/// actually existed* on those frames — so a session where gaze was gated for
/// nine minutes read exactly like one where it ran throughout. Counting the
/// states the pipeline already publishes is the difference between a latency
/// number and a baseline.
#[derive(Default)]
struct CoverageTally {
    slots: BTreeMap<&'static str, BTreeMap<&'static str, u64>>,
    gates: BTreeMap<GateReason, u64>,
    frames: u64,
}

impl CoverageTally {
    fn observe(&mut self, c: &SignalCoverage) {
        self.frames += 1;
        for (name, state) in [
            ("face", c.face),
            ("pose", c.pose),
            ("gaze", c.gaze),
            ("objects", c.objects),
            ("identity", c.identity),
        ] {
            *self.slots.entry(name).or_default().entry(state.as_str()).or_default() += 1;
        }
        if let Some(reason) = c.gaze_gate {
            *self.gates.entry(reason).or_default() += 1;
        }
    }

    fn print(&self) {
        if self.frames == 0 {
            return;
        }
        println!("\nsignal coverage over {} detected frame(s):", self.frames);
        for (slot, states) in &self.slots {
            let breakdown: Vec<String> = states
                .iter()
                .map(|(s, n)| {
                    format!("{s} {n} ({:.1}%)", 100.0 * *n as f64 / self.frames as f64)
                })
                .collect();
            println!("  {slot:<9} {}", breakdown.join("   "));
        }
        if self.gates.is_empty() {
            println!("  gaze gate  never fired");
        } else {
            let breakdown: Vec<String> =
                self.gates.iter().map(|(r, n)| format!("{r:?} {n}")).collect();
            println!("  gaze gate  {}", breakdown.join("   "));
        }
    }
}

struct LiveOpts {
    overlay: bool,
    paced: bool,
    max_frames: Option<u64>,
    save_every: Option<u64>,
    save_dir: PathBuf,
}

fn cmd_live(cfg: &Config, source: &str, opts: LiveOpts) -> Result<()> {
    let LiveOpts { overlay, paced, max_frames, save_every, save_dir } = opts;

    if overlay {
        tracing::warn!(
            "--overlay needs a preview window; use `deepscreen-viewer` for that, \
             or --save-every N to eyeball frames here"
        );
    }
    if save_every.is_some() {
        std::fs::create_dir_all(&save_dir).map_err(|e| DetectError::io(&save_dir, e))?;
    }

    let spec: SourceSpec = source.parse()?;
    let src = spec.open(&cfg.capture, paced)?;
    let (w, h) = src.resolution();
    println!(
        "source {}  {}x{}  nominal {}",
        src.name(),
        w,
        h,
        src.nominal_fps().map(|f| format!("{f:.2} fps")).unwrap_or_else(|| "unknown".into())
    );

    let mut cfg = cfg.clone();
    cfg.models.fill_missing_from_dir("../models");
    for (slot, path) in [
        ("pose", &cfg.models.pose),
        ("gaze", &cfg.models.gaze),
        ("objects", &cfg.models.objects),
        ("identity", &cfg.models.identity),
    ] {
        if let Some(p) = path {
            println!("{slot} model {}", p.display());
        }
    }
    match &cfg.models.face {
        Some(p) => println!("face model {}", p.display()),
        None => {
            return Err(DetectError::Config(
                "no face model: set models.face in config, or put \
                 face_detection_yunet_2023mar.onnx in models/"
                    .into(),
            ))
        }
    }

    // Everything below runs through the real pipeline — capture on its own
    // thread, detection at its own cadence, this loop only observing. Nothing
    // here can slow capture down, which is the whole point of step 4.
    let mut det = Detector::builder().config(cfg).build()?;
    det.start(src)?;

    let start = Instant::now();
    let mut saved = 0u64;
    let mut seen = 0u64;
    let mut last_saved_seq = u64::MAX;
    let mut last_report = Instant::now();
    let mut coverage = CoverageTally::default();

    while det.is_running() {
        if let Some(d) = det.latest() {
            if d.frame.seq != last_saved_seq {
                // Count *detected* frames, not captured sequence numbers. The
                // detect worker only sees every Nth captured frame, so keying
                // off `seq % n` silently saves nothing whenever the multiples
                // land on skipped frames.
                let nth = seen;
                seen += 1;
                last_saved_seq = d.frame.seq;
                coverage.observe(&d.signals.produced_by);
                if save_every.is_some_and(|n| n > 0 && nth.is_multiple_of(n)) {
                    save_frame(&d.frame, &d.signals.faces, &save_dir)?;
                    saved += 1;
                }
            }
        }

        if last_report.elapsed().as_secs_f32() >= 1.0 {
            let s = det.snapshot();
            println!(
                "  cap {:>6} @ {:5.1} fps   det {:>6} @ {:5.1} fps   skipped {:>6}   {} face(s)",
                s.stats.frames_captured,
                s.stats.capture_fps,
                s.stats.frames_detected,
                s.stats.detect_fps,
                s.stats.frames_skipped,
                s.face_count,
            );
            last_report = Instant::now();
        }

        if max_frames.is_some_and(|m| det.snapshot().stats.frames_captured >= m) {
            break;
        }

        // Polling cadence of an observer, not a worker. `snapshot()` is
        // level-triggered by design (MODELS.md §3) — nothing is missed by
        // looking less often, only reported later.
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    det.stop();
    let s = det.snapshot();
    let elapsed = start.elapsed().as_secs_f32();

    println!("\n{} frames captured in {elapsed:.2}s", s.stats.frames_captured);
    if s.stats.frames_captured == 0 {
        if let Some(err) = det.error() {
            return Err(DetectError::Camera(err));
        }
        println!("no frames — source produced nothing");
        return Ok(());
    }

    println!(
        "capture {:.2} fps   detect {:.2} fps   skipped {} ({:.0}% of captured)",
        s.stats.capture_fps,
        s.stats.detect_fps,
        s.stats.frames_skipped,
        100.0 * s.stats.frames_skipped as f32 / s.stats.frames_captured as f32
    );
    println!(
        "detect p50 {:.2} ms   p95 {:.2} ms   (total incl. pre/post p50 {:.2} ms)",
        s.stats.detect_p50_us as f32 / 1000.0,
        s.stats.detect_p95_us as f32 / 1000.0,
        s.stats.total_p50_us as f32 / 1000.0,
    );
    coverage.print();
    if saved > 0 {
        println!("saved {saved} frame(s) to {}", save_dir.display());
    }
    if let Some(err) = det.error() {
        println!("ended with: {err}");
    }
    Ok(())
}

/// Write one frame as a JPEG so a live camera can be eyeballed without a
/// preview window. Deliberately outside any hot path — this encodes on the
/// capture thread and is only ever driven by an explicit `--save-every`.
fn save_frame(
    frame: &deepscreen_viewer::Frame,
    detections: &[FaceDetection],
    dir: &std::path::Path,
) -> Result<()> {
    let path = dir.join(format!("frame_{:06}.jpg", frame.seq));
    let mut img = image::RgbImage::from_raw(frame.width, frame.height, frame.data.to_vec())
        .ok_or_else(|| DetectError::Config("frame buffer did not match its dimensions".into()))?;

    for d in detections {
        draw_rect(&mut img, &d.bbox, [0, 255, 0]);
        if let Some(k) = &d.keypoints {
            // Eyes green, nose yellow, mouth corners red — enough to tell at a
            // glance whether the keypoints are landing where they should.
            for (p, colour) in [
                (k.right_eye, [0u8, 255, 0]),
                (k.left_eye, [0, 255, 0]),
                (k.nose, [255, 255, 0]),
                (k.right_mouth, [255, 0, 0]),
                (k.left_mouth, [255, 0, 0]),
            ] {
                draw_cross(&mut img, p.0, p.1, colour);
            }
        }
    }

    img.save(&path).map_err(|e| DetectError::source(path.display().to_string(), e.to_string()))
}

fn put(img: &mut image::RgbImage, x: i64, y: i64, colour: [u8; 3]) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, image::Rgb(colour));
    }
}

fn draw_rect(img: &mut image::RgbImage, b: &deepscreen_viewer::BBox, colour: [u8; 3]) {
    let (x0, y0) = (b.x.round() as i64, b.y.round() as i64);
    let (x1, y1) = ((b.x + b.w).round() as i64, (b.y + b.h).round() as i64);
    // Two pixels thick, so it survives JPEG at a glance.
    for t in 0..2 {
        for x in x0..=x1 {
            put(img, x, y0 + t, colour);
            put(img, x, y1 - t, colour);
        }
        for y in y0..=y1 {
            put(img, x0 + t, y, colour);
            put(img, x1 - t, y, colour);
        }
    }
}

fn draw_cross(img: &mut image::RgbImage, cx: f32, cy: f32, colour: [u8; 3]) {
    let (cx, cy) = (cx.round() as i64, cy.round() as i64);
    for d in -3..=3i64 {
        put(img, cx + d, cy, colour);
        put(img, cx, cy + d, colour);
    }
}

// ---------------------------------------------------------------------------
// record
// ---------------------------------------------------------------------------

/// Record `Signals` for **every** frame of a source.
///
/// Deliberately synchronous and single-threaded, bypassing the `Detector`.
/// The live pipeline samples at cadence and drops whatever it missed, which is
/// correct for a session and wrong for a corpus: a recording that depends on
/// how fast the recording machine was is not a regression fixture. Here every
/// frame is processed, in order, so the same clip always produces the same
/// JSONL — which is the property replay-based tuning rests on (MODELS.md §2).
fn cmd_record(cfg: &Config, source: &str, out: &PathBuf, max_frames: Option<u64>) -> Result<()> {
    let spec: SourceSpec = source.parse()?;
    let mut src = spec.open(&cfg.capture, false)?;
    let fps = src.nominal_fps().unwrap_or(cfg.capture.fps as f32).max(1.0);

    let mut cfg = cfg.clone();
    cfg.models.fill_missing_from_dir("../models");
    let face_path = cfg.models.face.clone().ok_or_else(|| {
        DetectError::Config("no face model: put models in models/ or set models.face".into())
    })?;

    let mut face = YuNet::load(&face_path, &cfg)?;
    let mut pose = match &cfg.models.pose {
        Some(p) => Some(HeadPoseNet::load(p, &cfg)?),
        None => None,
    };
    let mut gaze = match &cfg.models.gaze {
        Some(p) => Some(GazeNet::load(p, &cfg)?),
        None => None,
    };
    // Objects run on every frame here too. In the live pipeline they are a
    // 1 Hz worker; for a corpus, every frame gets a real answer so replay is
    // deterministic.
    let mut objects = match &cfg.models.objects {
        Some(p) => Some(YoloxNano::load(p, &cfg)?),
        None => None,
    };

    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| DetectError::io(parent, e))?;
    }
    let file = std::fs::File::create(out).map_err(|e| DetectError::io(out, e))?;
    let mut writer = std::io::BufWriter::new(file);

    let start = Instant::now();
    let mut written = 0u64;

    while let Some(frame) = src.next_frame()? {
        let faces = face.detect(&frame)?;

        let mut pose_state = SlotState::NotConfigured;
        let head_pose = match (pose.as_mut(), faces.first()) {
            (Some(model), Some(primary)) => match model.estimate(&frame, primary) {
                Ok((p, _)) => {
                    pose_state = SlotState::Produced;
                    Some(p)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "head pose failed");
                    pose_state = SlotState::Failed;
                    None
                }
            },
            (Some(_), None) => {
                pose_state = SlotState::SkippedGated;
                None
            }
            _ => None,
        };

        let mut gaze_state = SlotState::NotConfigured;
        let mut gaze_gate = None;
        let gaze_signal = match (gaze.as_mut(), faces.first()) {
            (Some(model), Some(primary)) => {
                match model.estimate(&frame, primary, head_pose) {
                    Ok(GazeOutcome::Produced { gaze: g, .. }) => {
                        gaze_state = SlotState::Produced;
                        Some(g)
                    }
                    Ok(GazeOutcome::Gated(reason)) => {
                        gaze_state = SlotState::SkippedGated;
                        gaze_gate = Some(reason);
                        None
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "gaze failed");
                        gaze_state = SlotState::Failed;
                        None
                    }
                }
            }
            (Some(_), None) => {
                gaze_state = SlotState::SkippedGated;
                gaze_gate = Some(GateReason::NoFace);
                None
            }
            _ => None,
        };

        // `record` runs every model on every frame, so objects never skip for
        // cadence here — that is the whole point of the command.
        let mut objects_state = SlotState::NotConfigured;
        let detected_objects = match objects.as_mut() {
            Some(model) => match model.detect(&frame) {
                Ok((o, _)) => {
                    objects_state = SlotState::Produced;
                    o
                }
                Err(e) => {
                    tracing::warn!(error = %e, "object detection failed");
                    objects_state = SlotState::Failed;
                    Vec::new()
                }
            },
            None => Vec::new(),
        };

        let signals = Signals {
            seq: frame.seq,
            // Frame-derived, not wall clock: the recording must mean the same
            // thing on any machine.
            t_ms: (frame.seq as f64 * 1000.0 / fps as f64).round() as u64,
            faces,
            head_pose,
            gaze: gaze_signal,
            objects: detected_objects,
            produced_by: SignalCoverage {
                face: SlotState::Produced,
                pose: pose_state,
                gaze: gaze_state,
                objects: objects_state,
                identity: SlotState::NotConfigured,
                gaze_gate,
            },
            ..Default::default()
        };

        serde_json::to_writer(&mut writer, &signals)
            .map_err(|e| DetectError::Config(format!("writing {}: {e}", out.display())))?;
        writer.write_all(b"\n").map_err(|e| DetectError::io(out, e))?;
        written += 1;

        if max_frames.is_some_and(|m| written >= m) {
            break;
        }
    }

    writer.flush().map_err(|e| DetectError::io(out, e))?;
    println!(
        "wrote {written} Signals to {} in {:.2}s ({:.1} fps timebase, every frame)",
        out.display(),
        start.elapsed().as_secs_f32(),
        fps
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------

fn cmd_bench(
    cfg: &Config,
    source: Option<&str>,
    model: Option<&str>,
    all: bool,
    sweep_threads: bool,
    iters: u64,
    report: Option<PathBuf>,
) -> Result<()> {
    if sweep_threads {
        return Err(DetectError::Config(
            "--sweep-threads arrives with the threading skeleton at build step 4".into(),
        ));
    }

    if all || model.is_some() {
        return bench_models(cfg, model, iters, report);
    }

    let source = source.ok_or_else(|| {
        DetectError::Config("bench needs --source, --model <path> or --all".into())
    })?;

    let spec: SourceSpec = source.parse()?;
    let mut src = spec.open(&cfg.capture, false)?;
    let (w, h) = src.resolution();

    let mut decode = Latencies::with_capacity(iters as usize);
    let mut frames = 0u64;
    let overall = Instant::now();

    while frames < iters {
        let t = Instant::now();
        let Some(frame) = src.next_frame()? else { break };
        decode.record(t.elapsed());
        std::hint::black_box(&frame.data[0]);
        frames += 1;
    }

    let elapsed = overall.elapsed().as_secs_f32();
    let summary = decode.summary();

    let mut out = String::new();
    out.push_str("# detect-cli bench\n\n");
    out.push_str(&format!("- source: `{}`\n", src.name()));
    out.push_str(&format!("- resolution: {w}x{h}\n"));
    out.push_str(&format!("- frames: {frames}\n"));
    out.push_str(&format!("- wall: {elapsed:.2}s ({:.1} fps)\n", frames as f32 / elapsed));
    if let Some(s) = summary {
        out.push_str("\n| stage | p50 (ms) | p95 (ms) | max (ms) | n |\n");
        out.push_str("|---|---|---|---|---|\n");
        out.push_str(&format!(
            "| capture | {:.2} | {:.2} | {:.2} | {} |\n",
            s.p50_us as f32 / 1000.0,
            s.p95_us as f32 / 1000.0,
            s.max_us as f32 / 1000.0,
            s.samples
        ));
    }
    out.push_str(
        "\n_Capture path only. Per-model latency, EP verification and the intra-op \
         thread sweep arrive with build steps 3-5._\n",
    );

    print!("{out}");
    if let Some(path) = report {
        std::fs::write(&path, &out).map_err(|e| DetectError::io(&path, e))?;
        println!("written to {}", path.display());
    }
    Ok(())
}

/// Benchmark model graphs on synthetic input.
fn bench_models(
    cfg: &Config,
    model: Option<&str>,
    iters: u64,
    report: Option<PathBuf>,
) -> Result<()> {
    let paths: Vec<PathBuf> = match model {
        Some(m) => vec![PathBuf::from(m)],
        None => {
            let dir = PathBuf::from("../models");
            let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
                .map_err(|e| DetectError::io(&dir, e))?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "onnx"))
                .collect();
            found.sort();
            found
        }
    };

    if paths.is_empty() {
        return Err(DetectError::Config("no .onnx files found in models/".into()));
    }

    let mut out = String::from("# detect-cli bench --all\n\n");
    out.push_str(&format!(
        "Synthetic zero-tensor forward passes, {iters} iterations after {} warm-up. \
         CPU execution provider. No preprocessing, no decode — a floor, not a budget.\n\n",
        cfg.runtime.warmup_iters
    ));
    out.push_str("| model | MB | input | load ms | p50 ms | p95 ms | max ms |\n");
    out.push_str("|---|---|---|---|---|---|---|\n");

    for path in &paths {
        match deepscreen_viewer::models::bench_model(path, &cfg.runtime, iters as u32) {
            Ok(r) => {
                let shape = r
                    .input_shapes
                    .first()
                    .map(|s| s.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x"))
                    .unwrap_or_default();
                out.push_str(&format!(
                    "| {} | {:.1} | {} | {:.0} | {:.2} | {:.2} | {:.2} |\n",
                    r.name,
                    r.size_bytes as f64 / 1_048_576.0,
                    shape,
                    r.load_ms,
                    r.latency.p50_us as f64 / 1000.0,
                    r.latency.p95_us as f64 / 1000.0,
                    r.latency.max_us as f64 / 1000.0,
                ));
            }
            Err(e) => {
                // A model that will not load is exactly what this command is
                // for finding, so report it in the table rather than aborting.
                out.push_str(&format!(
                    "| {} | — | — | — | — | — | **failed: {e}** |\n",
                    path.file_name().unwrap_or_default().to_string_lossy()
                ));
            }
        }
    }

    print!("{out}");
    if let Some(path) = report {
        std::fs::write(&path, &out).map_err(|e| DetectError::io(&path, e))?;
        println!("\nwritten to {}", path.display());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// replay
// ---------------------------------------------------------------------------

fn cmd_replay(path: &PathBuf, expect: &[String]) -> Result<()> {
    let text = std::fs::read_to_string(path).map_err(|e| DetectError::io(path, e))?;

    let mut count = 0u64;
    let mut faces = 0u64;
    let mut objects = 0u64;
    let mut first_t = None;
    let mut last_t = 0u64;
    // Per-slot tallies, keyed by state. A slot that "ever ran" says nothing
    // useful; what matters is the proportion of frames each state accounts for,
    // and for gaze, which gate condition caused each skip.
    let mut slots: BTreeMap<&'static str, BTreeMap<&'static str, u64>> = BTreeMap::new();
    let mut gates: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut frames_with_face = 0u64;

    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let s: Signals = serde_json::from_str(line).map_err(|e| {
            DetectError::Config(format!(
                "{}:{}: not valid Signals JSON: {e}",
                path.display(),
                i + 1
            ))
        })?;
        count += 1;
        faces += s.faces.len() as u64;
        objects += s.objects.len() as u64;
        first_t.get_or_insert(s.t_ms);
        last_t = s.t_ms;
        if !s.faces.is_empty() {
            frames_with_face += 1;
        }
        let c = &s.produced_by;
        for (slot, state) in [
            ("face", c.face),
            ("pose", c.pose),
            ("gaze", c.gaze),
            ("objects", c.objects),
            ("identity", c.identity),
        ] {
            *slots.entry(slot).or_default().entry(state.as_str()).or_default() += 1;
        }
        if let Some(reason) = c.gaze_gate {
            *gates.entry(reason.as_str()).or_default() += 1;
        }
    }

    let span_ms = last_t.saturating_sub(first_t.unwrap_or(0));
    println!("{count} Signals over {:.2}s", span_ms as f32 / 1000.0);
    println!("  face detections: {faces}");
    println!("  object detections: {objects}");
    println!("  frames with a face: {frames_with_face} of {count}");

    println!("\n  per-slot state, frames (share of all frames):");
    for (slot, states) in &slots {
        let parts: Vec<String> = states
            .iter()
            .map(|(state, n)| {
                format!("{state}={n} ({:.0}%)", 100.0 * *n as f32 / count.max(1) as f32)
            })
            .collect();
        println!("    {slot:<9} {}", parts.join("  "));
    }

    // The number that decides whether the gate is doing its job or eating the
    // signal. Denominated in frames that had a face, because a frame with no
    // face was never a candidate for gaze in the first place.
    if let Some(gaze_states) = slots.get("gaze") {
        let produced = gaze_states.get("produced").copied().unwrap_or(0);
        if frames_with_face > 0 {
            println!(
                "\n  gaze ran on {produced} of {frames_with_face} frames with a face ({:.1}%)",
                100.0 * produced as f32 / frames_with_face as f32
            );
        }
    }
    if !gates.is_empty() {
        println!("  gate reasons:");
        let total: u64 = gates.values().sum();
        for (reason, n) in &gates {
            println!("    {reason:<16} {n} ({:.1}%)", 100.0 * *n as f32 / total as f32);
        }
    }

    if !expect.is_empty() {
        return Err(DetectError::Config(
            "--expect needs the fusion layer, which arrives at build step 8; \
             the recording above parsed cleanly"
                .into(),
        ));
    }
    println!("\n(fusion arrives at build step 8 — this is a parse and coverage check)");
    Ok(())
}

// ---------------------------------------------------------------------------
// inspect
// ---------------------------------------------------------------------------

fn cmd_inspect(paths: &[PathBuf]) -> Result<()> {
    for path in paths {
        let info = deepscreen_viewer::models::inspect(path)?;
        println!("\n{}", info.path.display());
        println!("  {:.1} KB", info.size_bytes as f64 / 1024.0);

        println!("  inputs:");
        for t in &info.inputs {
            println!("    {t}");
        }
        println!("  outputs:");
        for t in &info.outputs {
            println!("    {t}");
        }

        if info.has_dynamic_axes() {
            println!(
                "  note: dynamic axes present — DirectML wants fully static shapes at \
                 session creation (step 5)"
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

fn cmd_config(cfg: &Config, out: Option<PathBuf>) -> Result<()> {
    let text = cfg.to_toml()?;
    match out {
        Some(path) => {
            std::fs::write(&path, &text).map_err(|e| DetectError::io(&path, e))?;
            println!("wrote {}", path.display());
        }
        None => print!("{text}"),
    }
    Ok(())
}
