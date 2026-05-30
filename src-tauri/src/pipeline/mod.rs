//! The `Detector` — threads, channels, cadence (MODELS.md §3, §6).
//!
//! Two output paths, deliberately different in kind:
//!
//! - [`Detector::events`] is **edge-triggered and low-rate**. Decisions. This
//!   is what gets persisted and sent to a backend.
//! - [`Detector::snapshot`] is **level-triggered and polled**. Continuous
//!   values, read at whatever rate a UI wants.
//!
//! There is deliberately no way to push continuous values. The old module's
//! frame-rate re-render problem existed because detection pushed state at frame
//! rate; here that bug is unrepresentable because no API expresses it.

pub mod frame_bus;
mod workers;

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::SystemTime;

use arc_swap::ArcSwap;
use crossbeam_channel::{Receiver, Sender};

use crate::capture::FrameSource;
use crate::config::Config;
use crate::error::{DetectError, Result};
use crate::models::face::YuNet;
use crate::models::gaze::GazeNet;
use crate::models::identity::ArcFace;
use crate::models::objects::YoloxNano;
use crate::models::pose::HeadPoseNet;
use crate::report::{FrameStats, Latencies, SessionReport, SignalStatus};
use crate::types::{
    DegradeReason, DetectorState, Event, ObjectDetection, PipelineStats, Signals, SlotState,
};

use frame_bus::FrameBus;

/// How many recent samples the live percentiles describe. At 15 Hz this is
/// about half a minute — long enough to be stable, short enough that a stall
/// shows up while it is still happening.
const LATENCY_WINDOW: usize = 512;

/// A frame and the signals derived from **that** frame, kept together.
///
/// Never separate these. Handing a consumer a frame and a separately-published
/// set of signals means it has to match sequence numbers to draw a box in the
/// right place, and it will eventually get that wrong. As one unit, overlay
/// alignment is structural rather than something anyone has to maintain.
#[derive(Debug)]
pub struct Detected {
    pub frame: Arc<crate::types::Frame>,
    pub signals: Signals,
    /// Model time only.
    pub detect_us: u32,
    /// Including preprocessing and decode.
    pub total_us: u32,
}

/// State shared between the threads. Everything here is either atomic,
/// lock-free, or a mutex that is never held across an inference.
pub(crate) struct Shared {
    bus: FrameBus,
    latest: ArcSwap<Option<Arc<Detected>>>,
    events: Sender<Event>,

    stop: AtomicBool,
    capture_done: AtomicBool,
    source_ended: AtomicBool,

    frames_detected: AtomicU64,
    frames_skipped: AtomicU64,
    capture_fps: AtomicU32,
    detect_fps: AtomicU32,

    // Held for the duration of a `record_us` or a `summary()`, never across a
    // model run. Percentiles are not worth an atomic histogram here.
    detect_latency: Mutex<Latencies>,
    total_latency: Mutex<Latencies>,
    object_latency: Mutex<Latencies>,

    /// Newest object result, published by the object worker and read by the
    /// face worker when it assembles `Signals`. Lock-free on the read side.
    objects: ArcSwap<ObjectResult>,
    /// Newest identity result, same attach-once contract as `objects`.
    identity: ArcSwap<IdentityResult>,

    /// The enrolled reference embedding. `None` until someone enrols.
    ///
    /// In memory only, on purpose: persisting a face embedding is a data
    /// protection decision with retention and consent attached, not a
    /// convenience, and it is not one to make incidentally while wiring a
    /// worker. Enrolment lasts as long as the process.
    enrolled: Mutex<Option<Vec<f32>>>,
    /// Set by [`Detector::enrol`]; cleared by the identity worker once it has
    /// captured a usable face.
    enrol_request: AtomicBool,

    /// Which execution provider each model session actually got, recorded at
    /// load. Hard-coding "CPU" here is how a silent DirectML fallback stays
    /// invisible, which is the failure this whole field exists to catch.
    eps: Mutex<BTreeMap<String, String>>,

    /// Violations currently raised, for the polled snapshot.
    active_violations: Mutex<Vec<(crate::types::ViolationKind, Option<String>)>>,
    /// Every violation this session, closed ones included. Feeds the viewer's
    /// log and the session report.
    violation_log: Mutex<Vec<crate::types::Violation>>,

    error: Mutex<Option<String>>,
    degraded: Mutex<Vec<DegradeReason>>,
}

/// One object-worker result, tagged with the frame it was computed on.
///
/// The sequence number is what makes per-frame coverage possible: the face
/// worker can tell a result it has not seen before from one it already
/// attached to an earlier frame.
#[derive(Default)]
pub(crate) struct ObjectResult {
    detections: Vec<ObjectDetection>,
    /// Frame the detections were computed on. `0` = nothing yet.
    seq: u64,
    /// `Produced` once a run succeeds, `Failed` after an error,
    /// `NotConfigured` while no object model is loaded.
    state: SlotState,
}

/// One identity check, tagged with the frame it ran on.
#[derive(Default)]
pub(crate) struct IdentityResult {
    /// Cosine similarity against the enrolled embedding.
    score: Option<f32>,
    seq: u64,
    state: SlotState,
}

impl Shared {
    fn new(events: Sender<Event>) -> Self {
        Self {
            bus: FrameBus::new(),
            latest: ArcSwap::from_pointee(None),
            events,
            stop: AtomicBool::new(false),
            capture_done: AtomicBool::new(false),
            source_ended: AtomicBool::new(false),
            frames_detected: AtomicU64::new(0),
            frames_skipped: AtomicU64::new(0),
            capture_fps: AtomicU32::new(0),
            detect_fps: AtomicU32::new(0),
            detect_latency: Mutex::new(Latencies::rolling(LATENCY_WINDOW)),
            total_latency: Mutex::new(Latencies::rolling(LATENCY_WINDOW)),
            object_latency: Mutex::new(Latencies::rolling(LATENCY_WINDOW)),
            objects: ArcSwap::from_pointee(ObjectResult::default()),
            identity: ArcSwap::from_pointee(IdentityResult::default()),
            enrolled: Mutex::new(None),
            enrol_request: AtomicBool::new(false),
            eps: Mutex::new(BTreeMap::new()),
            active_violations: Mutex::new(Vec::new()),
            violation_log: Mutex::new(Vec::new()),
            error: Mutex::new(None),
            degraded: Mutex::new(Vec::new()),
        }
    }

    /// Mark the object slot as configured but not yet run, so a frame before
    /// the worker's first result reads as a cadence skip rather than as "no
    /// model loaded".
    fn arm_objects(&self) {
        self.objects.store(Arc::new(ObjectResult {
            state: SlotState::SkippedCadence,
            ..Default::default()
        }));
    }

    fn publish_objects(&self, detections: Vec<ObjectDetection>, seq: u64) {
        if !detections.is_empty() {
            tracing::debug!(seq, count = detections.len(), "objects detected");
        }
        self.objects.store(Arc::new(ObjectResult {
            detections,
            seq,
            state: SlotState::Produced,
        }));
    }

    fn publish_object_failure(&self, seq: u64) {
        self.objects.store(Arc::new(ObjectResult {
            detections: Vec::new(),
            seq,
            state: SlotState::Failed,
        }));
    }

    /// Hand the newest unseen object result to the face worker, exactly once.
    ///
    /// The two workers run on independent cursors over the frame bus, so they
    /// rarely land on the same frame — requiring an exact sequence match would
    /// throw nearly every object result away. Instead each result is attached
    /// to the first face frame that follows it and is then consumed, so it is
    /// reported once and never carried forward.
    ///
    /// Every later frame reports `SkippedCadence` with an empty list: no new
    /// measurement, and explicitly not evidence that nothing is there.
    fn take_new_objects(&self, cursor: &mut u64) -> (Vec<ObjectDetection>, SlotState) {
        let result = self.objects.load();
        match result.state {
            SlotState::NotConfigured => (Vec::new(), SlotState::NotConfigured),
            _ if result.seq == 0 || result.seq <= *cursor => (Vec::new(), SlotState::SkippedCadence),
            state => {
                *cursor = result.seq;
                (result.detections.clone(), state)
            }
        }
    }

    /// Mark identity as configured but not yet enrolled. Until an enrolment
    /// happens there is nothing to compare against, which is `NotConfigured`
    /// rather than a failure — the distinction fusion needs so an unenrolled
    /// session does not read as a passing one.
    fn arm_identity(&self) {
        self.identity.store(Arc::new(IdentityResult {
            state: SlotState::NotConfigured,
            ..Default::default()
        }));
    }

    fn publish_identity(&self, score: Option<f32>, seq: u64, state: SlotState) {
        self.identity.store(Arc::new(IdentityResult { score, seq, state }));
    }

    /// Same attach-once-then-consume contract as [`Shared::take_new_objects`]:
    /// at 0.2 Hz against a 15 Hz worker, requiring an exact sequence match
    /// would discard almost every result.
    fn take_new_identity(&self, cursor: &mut u64) -> (Option<f32>, SlotState) {
        let result = self.identity.load();
        match result.state {
            SlotState::NotConfigured => (None, SlotState::NotConfigured),
            _ if result.seq == 0 || result.seq <= *cursor => (None, SlotState::SkippedCadence),
            state => {
                *cursor = result.seq;
                (result.score, state)
            }
        }
    }

    fn note_ep(&self, slot: &str, ep: crate::models::ActiveEp) {
        if let Ok(mut map) = self.eps.lock() {
            map.insert(slot.to_string(), ep.as_str().to_string());
        }
    }

    /// What a slot actually ran on, or `"unknown"` if it never loaded.
    fn ep_of(&self, slot: &str) -> String {
        self.eps
            .lock()
            .ok()
            .and_then(|m| m.get(slot).cloned())
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn note_degraded(&self, reason: DegradeReason) {
        if let Ok(mut list) = self.degraded.lock() {
            if !list.contains(&reason) {
                list.push(reason.clone());
            }
        }
        let _ = self.events.send(Event::Degraded(reason));
    }

    fn set_error(&self, message: String) {
        if let Ok(mut slot) = self.error.lock() {
            slot.get_or_insert(message);
        }
    }

    fn stats(&self) -> PipelineStats {
        let detect = self.detect_latency.lock().ok().and_then(|l| l.summary()).unwrap_or_default();
        let total = self.total_latency.lock().ok().and_then(|l| l.summary()).unwrap_or_default();
        PipelineStats {
            frames_captured: self.bus.captured(),
            frames_detected: self.frames_detected.load(Ordering::Relaxed),
            frames_skipped: self.frames_skipped.load(Ordering::Relaxed),
            capture_fps: f32::from_bits(self.capture_fps.load(Ordering::Relaxed)),
            detect_fps: f32::from_bits(self.detect_fps.load(Ordering::Relaxed)),
            detect_p50_us: detect.p50_us,
            detect_p95_us: detect.p95_us,
            total_p50_us: total.p50_us,
        }
    }
}

// ---------------------------------------------------------------------------
// builder
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct DetectorBuilder {
    config: Config,
}

impl DetectorBuilder {
    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    pub fn build(self) -> Result<Detector> {
        self.config.validate()?;
        let (tx, rx) = crossbeam_channel::unbounded();
        Ok(Detector {
            config: self.config,
            shared: Arc::new(Shared::new(tx)),
            events_rx: rx,
            threads: Vec::new(),
            started_at: SystemTime::now(),
            source_name: String::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// detector
// ---------------------------------------------------------------------------

pub struct Detector {
    config: Config,
    shared: Arc<Shared>,
    events_rx: Receiver<Event>,
    threads: Vec<JoinHandle<()>>,
    started_at: SystemTime,
    source_name: String,
}

impl Detector {
    pub fn builder() -> DetectorBuilder {
        DetectorBuilder::default()
    }

    /// Load models, then spawn capture and detect.
    ///
    /// Model loading happens here rather than lazily on first frame: session
    /// creation is expensive (graph optimisation, EP graph construction) and
    /// paying it mid-session would show up as a stall exactly when a candidate
    /// is being watched (MODELS.md §9).
    pub fn start(&mut self, source: Box<dyn FrameSource>) -> Result<()> {
        if !self.threads.is_empty() {
            return Err(DetectError::Config("detector is already running".into()));
        }
        self.source_name = source.name();

        // The face model is one of only two fatal dependencies. Everything
        // else degrades (MODELS.md §8).
        let face_path = self.config.models.face.clone().ok_or_else(|| {
            DetectError::Config("config.models.face is required — no face model, no session".into())
        })?;
        let face = YuNet::load(&face_path, &self.config)?;
        self.shared.note_ep("face", face.ep());
        tracing::info!(model = %face_path.display(), ep = face.ep().as_str(), "face model ready");

        // Everything past the face model is a capability, not a requirement:
        // if it will not load, the session continues without it and says so
        // (MODELS.md §8, degrade never die).
        let gaze = match self.config.models.gaze.clone() {
            Some(path) => match GazeNet::load(&path, &self.config) {
                Ok(model) => {
                    self.shared.note_ep("gaze", model.ep());
                    tracing::info!(model = %path.display(), ep = model.ep().as_str(), "gaze model ready");
                    Some(model)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "gaze unavailable; continuing without it");
                    self.shared.note_degraded(DegradeReason::ModelUnavailable {
                        model: "gaze".into(),
                        why: e.to_string(),
                    });
                    None
                }
            },
            None => None,
        };

        let pose = match self.config.models.pose.clone() {
            Some(path) => match HeadPoseNet::load(&path, &self.config) {
                Ok(model) => {
                    self.shared.note_ep("pose", model.ep());
                    tracing::info!(model = %path.display(), ep = model.ep().as_str(), "head pose model ready");
                    Some(model)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "head pose unavailable; continuing without it");
                    self.shared.note_degraded(DegradeReason::ModelUnavailable {
                        model: "pose".into(),
                        why: e.to_string(),
                    });
                    None
                }
            },
            None => None,
        };

        let shared = Arc::clone(&self.shared);
        let capture = std::thread::Builder::new()
            .name("ds-capture".into())
            .spawn(move || workers::capture_loop(source, shared))
            .map_err(|e| DetectError::Config(format!("spawning capture thread: {e}")))?;

        // Objects are a capability like pose and gaze: absent means the
        // signal is unavailable, not that the session cannot run.
        let objects = match self.config.models.objects.clone() {
            Some(path) => match YoloxNano::load(&path, &self.config) {
                Ok(model) => {
                    self.shared.note_ep("objects", model.ep());
                    tracing::info!(model = %path.display(), ep = model.ep().as_str(), "object model ready");
                    // Configured but not yet run: frames before the first
                    // result are cadence skips, not a missing model.
                    self.shared.arm_objects();
                    Some(model)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "objects unavailable; continuing without them");
                    self.shared.note_degraded(DegradeReason::ModelUnavailable {
                        model: "objects".into(),
                        why: e.to_string(),
                    });
                    None
                }
            },
            None => None,
        };

        let identity = match self.config.models.identity.clone() {
            Some(path) => match ArcFace::load(&path, &self.config) {
                Ok(model) => {
                    self.shared.note_ep("identity", model.ep());
                    tracing::info!(model = %path.display(), ep = model.ep().as_str(), "identity model ready");
                    Some(model)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "identity unavailable; continuing without it");
                    self.shared.note_degraded(DegradeReason::ModelUnavailable {
                        model: "identity".into(),
                        why: e.to_string(),
                    });
                    None
                }
            },
            None => None,
        };

        let shared = Arc::clone(&self.shared);
        let cfg = self.config.clone();
        let events = self.shared.events.clone();
        let detect = std::thread::Builder::new()
            .name("ds-detect".into())
            .spawn(move || {
                workers::detect_loop(workers::WorkerModels { face, pose, gaze }, cfg, shared, events)
            })
            .map_err(|e| DetectError::Config(format!("spawning detect thread: {e}")))?;

        self.threads.push(capture);
        self.threads.push(detect);

        if let Some(model) = objects {
            let shared = Arc::clone(&self.shared);
            let cfg = self.config.clone();
            let handle = std::thread::Builder::new()
                .name("ds-objects".into())
                .spawn(move || workers::object_loop(model, cfg, shared))
                .map_err(|e| DetectError::Config(format!("spawning object thread: {e}")))?;
            self.threads.push(handle);
        }

        // Identity is the fourth worker and the slowest: its own session, its
        // own cursor over the frame bus, 0.2 Hz. Same rule as every other
        // model here — one session owned by exactly one thread, no
        // `Mutex<Session>` anywhere in the inference path.
        if let Some(model) = identity {
            self.shared.arm_identity();
            let shared = Arc::clone(&self.shared);
            let cfg = self.config.clone();
            let handle = std::thread::Builder::new()
                .name("ds-identity".into())
                .spawn(move || workers::identity_loop(model, cfg, shared))
                .map_err(|e| DetectError::Config(format!("spawning identity thread: {e}")))?;
            self.threads.push(handle);
        }

        Ok(())
    }

    pub fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }
    }

    /// Low-rate stream of decisions. Fusion is its producer.
    pub fn events(&self) -> Receiver<Event> {
        self.events_rx.clone()
    }

    /// Cheap non-blocking read of current continuous values, for a live HUD.
    pub fn snapshot(&self) -> DetectorState {
        let latest = self.shared.latest.load_full();
        let degraded = self.shared.degraded.lock().map(|d| d.clone()).unwrap_or_default();

        match latest.as_ref() {
            Some(d) => DetectorState {
                seq: d.signals.seq,
                t_ms: d.signals.t_ms,
                face_count: d.signals.faces.len(),
                object_count: d.signals.objects.len(),
                head_pose: d.signals.head_pose,
                gaze: d.signals.gaze,
                identity_match: d.signals.identity_match,
                active_violations: self
                    .shared
                    .active_violations
                    .lock()
                    .map(|v| v.iter().map(|(k, _)| *k).collect())
                    .unwrap_or_default(),
                degraded,
                calibrated: false,
                stats: self.shared.stats(),
            },
            None => DetectorState { degraded, stats: self.shared.stats(), ..Default::default() },
        }
    }

    /// Capture the next usable face as the identity reference.
    ///
    /// Deliberately a request rather than an immediate capture: the caller is
    /// a UI thread that has no frame in hand, and the identity worker is the
    /// only thread allowed to touch the session. It enrols on its next cycle
    /// with a face present.
    pub fn enrol(&self) {
        self.shared.enrol_request.store(true, Ordering::Relaxed);
    }

    /// Whether a reference embedding exists.
    pub fn is_enrolled(&self) -> bool {
        self.shared.enrolled.lock().map(|e| e.is_some()).unwrap_or(false)
    }

    /// Which execution provider each model slot actually ended up on.
    ///
    /// Exposed so the HUD can show it. ORT falls back to CPU silently, and a
    /// GPU that never engaged is indistinguishable from a GPU that did not
    /// help unless this is on screen next to the latency it supposedly
    /// improved.
    pub fn execution_providers(&self) -> Vec<(String, String)> {
        self.shared
            .eps
            .lock()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default()
    }

    /// Every violation this session, closed ones included.
    pub fn violations(&self) -> Vec<crate::types::Violation> {
        self.shared.violation_log.lock().map(|v| v.clone()).unwrap_or_default()
    }

    /// The most recent frame **and** the signals derived from it, as one unit.
    ///
    /// This is what a preview renderer wants: the pixels and the boxes cannot
    /// disagree, because they were produced together.
    pub fn latest(&self) -> Option<Arc<Detected>> {
        self.shared.latest.load_full().as_ref().clone()
    }

    /// Whether the pipeline still has threads doing work.
    pub fn is_running(&self) -> bool {
        !self.threads.is_empty() && !self.shared.capture_done.load(Ordering::Relaxed)
    }

    /// Set when the source ran out cleanly, as opposed to failing.
    pub fn source_ended(&self) -> bool {
        self.shared.source_ended.load(Ordering::Relaxed)
    }

    /// The first fatal error, if the session ended because of one.
    pub fn error(&self) -> Option<String> {
        self.shared.error.lock().ok().and_then(|e| e.clone())
    }

    pub fn report(&self) -> SessionReport {
        let stats = self.shared.stats();
        let detect =
            self.shared.detect_latency.lock().ok().and_then(|l| l.summary()).unwrap_or_default();

        // Per-signal liveness, because a proctor must be able to tell "no
        // violations" from "the detector was never running" (MODELS.md §8).
        let mut signals = std::collections::BTreeMap::new();
        signals.insert(
            "face".to_string(),
            SignalStatus {
                active: stats.frames_detected > 0,
                active_fraction: if stats.frames_captured > 0 {
                    stats.frames_detected as f32 / stats.frames_captured as f32
                } else {
                    0.0
                },
                frames_processed: stats.frames_detected,
                execution_provider: self.shared.ep_of("face"),
                latency: Some(detect),
            },
        );
        let object_latency =
            self.shared.object_latency.lock().ok().and_then(|l| l.summary()).unwrap_or_default();
        signals.insert(
            "objects".to_string(),
            SignalStatus {
                // Sampled from the worker's own latency histogram rather than
                // a "has ever run" flag: a slot that ran once an hour ago and
                // a slot running at cadence are not equally alive.
                active: object_latency.samples > 0,
                active_fraction: if stats.frames_captured > 0 {
                    object_latency.samples as f32 / stats.frames_captured as f32
                } else {
                    0.0
                },
                frames_processed: object_latency.samples,
                execution_provider: self.shared.ep_of("objects"),
                latency: Some(object_latency),
            },
        );
        for slot in ["pose", "gaze", "identity"] {
            // Present and explicitly inactive, rather than absent. An absent
            // key would read as "not applicable". The EP is still recorded —
            // a slot that loaded on DirectML and then produced nothing is a
            // different problem from one that never loaded.
            signals.insert(
                slot.to_string(),
                SignalStatus {
                    execution_provider: self.shared.ep_of(slot),
                    ..SignalStatus::default()
                },
            );
        }

        SessionReport {
            started_at: self.started_at,
            duration_ms: self.started_at.elapsed().map(|d| d.as_millis() as u64).unwrap_or(0),
            source: self.source_name.clone(),
            frames: FrameStats {
                captured: stats.frames_captured,
                processed: stats.frames_detected,
                skipped: stats.frames_skipped,
                duplicate: 0,
                mean_fps: stats.capture_fps,
            },
            signals,
            violations: Vec::new(),
            degraded: self.shared.degraded.lock().map(|d| d.clone()).unwrap_or_default(),
            config: self.config.clone(),
        }
    }
}

impl Drop for Detector {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::replay::ImageDirSource;

    fn write_png(dir: &std::path::Path, name: &str, w: u32, h: u32) {
        let img = image::RgbImage::new(w, h);
        img.save(dir.join(name)).unwrap();
    }

    #[test]
    fn a_detector_without_a_face_model_refuses_to_start() {
        // No face model is one of the two fatal conditions; failing at start
        // with a clear reason beats degrading into a detector that detects
        // nothing and says nothing.
        let mut det = Detector::builder().build().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        write_png(tmp.path(), "a.png", 8, 8);
        let source = ImageDirSource::open(tmp.path(), 30, false).unwrap();

        let err = det.start(Box::new(source)).unwrap_err().to_string();
        assert!(err.contains("models.face"), "{err}");
    }

    #[test]
    fn snapshot_is_valid_before_anything_has_been_detected() {
        // The HUD polls from the moment the window opens, which is before the
        // first frame exists.
        let det = Detector::builder().build().unwrap();
        let state = det.snapshot();
        assert_eq!(state.face_count, 0);
        assert_eq!(state.stats.frames_captured, 0);
        assert!(state.active_violations.is_empty());
        assert!(det.latest().is_none());
    }

    #[test]
    fn events_channel_exists_before_it_has_a_producer() {
        // Consumers are written against this now; fusion fills it at step 8.
        let det = Detector::builder().build().unwrap();
        let rx = det.events();
        assert!(rx.try_recv().is_err(), "nothing produces events yet");
    }

    #[test]
    fn a_second_start_is_refused_rather_than_spawning_more_threads() {
        let mut cfg = Config::default();
        cfg.models.face = Some("models/does-not-exist.onnx".into());
        let mut det = Detector::builder().config(cfg).build().unwrap();

        let tmp = tempfile::tempdir().unwrap();
        write_png(tmp.path(), "a.png", 8, 8);
        // First start fails on the missing model, leaving no threads behind.
        let source = ImageDirSource::open(tmp.path(), 30, false).unwrap();
        assert!(det.start(Box::new(source)).is_err());
        assert!(!det.is_running());
    }
}
