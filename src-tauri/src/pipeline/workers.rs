//! The worker loops (MODELS.md §6).
//!
//! Four threads: capture, detect (face -> pose -> gaze at 15 Hz), objects at
//! 1 Hz, and identity at 0.2 Hz. Each of the three model workers owns its own
//! `last_seen` cursor over the same latest-frame bus and its own sessions, so
//! adding one never changed the others — which is why the topology in
//! MODELS.md §6 could be built in three separate steps.
//!
//! Every worker obeys the same three rules: it owns its sessions outright, it
//! reads the latest frame rather than a queue, and it never blocks capture.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::Sender;

use crate::capture::FrameSource;
use crate::config::Config;
use crate::direction::DirectionTracker;
use crate::models::face::YuNet;
use crate::models::gaze::{GazeNet, GazeOutcome};
use crate::models::identity::{cosine_similarity, ArcFace};
use crate::models::objects::YoloxNano;
use crate::models::pose::HeadPoseNet;
use crate::types::{DegradeReason, Event, GateReason, SignalCoverage, Signals, SlotState};

use super::{Detected, Shared};

/// The sessions the face worker owns outright.
///
/// One thread, several models, run sequentially — which is correct rather than
/// a compromise: they are a dependency chain (face -> crop -> pose/gaze) at the
/// same cadence, so there is nothing to parallelise (MODELS.md §6 rule 1).
/// Each is optional because only the face model is fatal to lose (§8).
pub(super) struct WorkerModels {
    pub face: YuNet,
    pub pose: Option<HeadPoseNet>,
    pub gaze: Option<GazeNet>,
}

/// Pull frames as fast as the source provides them and publish each to the bus.
///
/// This thread does nothing else. It never waits on a worker, never encodes,
/// never runs a model — if it did, capture rate would become a function of
/// inference rate, which is the coupling this whole design exists to avoid.
pub(super) fn capture_loop(mut source: Box<dyn FrameSource>, shared: Arc<Shared>) {
    let started = Instant::now();

    loop {
        if shared.stop.load(Ordering::Relaxed) {
            break;
        }

        match source.next_frame() {
            Ok(Some(frame)) => {
                shared.bus.publish(frame);
                let n = shared.bus.captured();
                shared.capture_fps.store(fps_bits(n, started), Ordering::Relaxed);
            }
            Ok(None) => {
                // A file or directory ran out. Not an error — the session is
                // simply over. A camera never returns this.
                tracing::info!("frame source exhausted");
                shared.source_ended.store(true, Ordering::Relaxed);
                break;
            }
            Err(e) => {
                // Losing the camera is one of only two fatal conditions
                // (MODELS.md §8), but even here the process must not die: the
                // session ends, having said why.
                tracing::error!(error = %e, "capture failed");
                let _ = shared.events.send(Event::Degraded(DegradeReason::CameraLost(e.to_string())));
                shared.set_error(e.to_string());
                break;
            }
        }
    }

    shared.capture_done.store(true, Ordering::Relaxed);
    tracing::debug!(frames = shared.bus.captured(), "capture thread finished");
}

/// Run the face model at its configured cadence against whatever the bus holds.
///
/// Owns the YuNet session outright — no `Mutex<Session>`, which is both a
/// throughput matter at 15 Hz and a hard requirement under DirectML, where only
/// one thread may call `Run()` on a session.
pub(super) fn detect_loop(
    mut models: WorkerModels,
    cfg: Config,
    shared: Arc<Shared>,
    events: Sender<Event>,
) {
    let period = Duration::from_secs_f64(1.0 / cfg.cadence.face_hz.max(0.1));
    let started = Instant::now();
    let mut last_seen = 0u64;
    let mut consecutive_failures = 0u32;
    let mut degraded = false;
    // Hysteresis state for the debug direction readout. Owned by this thread
    // because it is the only one that writes it, and updated in frame order.
    let mut directions = DirectionTracker::new(&cfg.thresholds.debug_direction);
    // Cursor over object results, so each is reported on exactly one frame.
    let mut last_object_seq = 0u64;
    let mut last_identity_seq = 0u64;
    let mut fusion = crate::fusion::FusionEngine::new(&cfg);

    loop {
        if shared.stop.load(Ordering::Relaxed) {
            break;
        }
        // Capture finished and there is nothing new left to process.
        if shared.capture_done.load(Ordering::Relaxed) && shared.bus.latest().seq <= last_seen {
            break;
        }

        let tick = Instant::now();

        if let Some((frame, missed)) = shared.bus.take_new(&mut last_seen) {
            if missed > 0 {
                shared.frames_skipped.fetch_add(missed, Ordering::Relaxed);
            }

            match models.face.detect_timed(&frame) {
                Ok((faces, mut timings)) => {
                    if degraded {
                        // Recovered: say so, because a report that shows a
                        // degraded window without a recovery reads as though
                        // the signal never came back.
                        let _ = events.send(Event::Recovered);
                        degraded = false;
                    }
                    consecutive_failures = 0;

                    // Pose runs on the primary face only. YuNet returns boxes
                    // sorted by score, so index 0 is the most confident — with
                    // two people in frame, pose describes the candidate, not
                    // whoever wandered past behind them.
                    let mut head_pose = None;
                    // `NotConfigured` unless a model exists; a face-less frame
                    // is a gated skip, because there was nothing to crop from.
                    let mut pose_state = match models.pose {
                        Some(_) if faces.is_empty() => SlotState::SkippedGated,
                        Some(_) => SlotState::NotConfigured,
                        None => SlotState::NotConfigured,
                    };
                    if let (Some(pose_model), Some(primary)) =
                        (models.pose.as_mut(), faces.first())
                    {
                        match pose_model.estimate(&frame, primary) {
                            Ok((p, pose_timings)) => {
                                head_pose = Some(p);
                                pose_state = SlotState::Produced;
                                timings.preprocess_us += pose_timings.preprocess_us;
                                timings.inference_us += pose_timings.inference_us;
                                timings.postprocess_us += pose_timings.postprocess_us;
                            }
                            Err(e) => {
                                // Losing pose is a degraded capability, not a
                                // dead session — the face signal is unaffected.
                                tracing::warn!(error = %e, "head pose failed");
                                pose_state = SlotState::Failed;
                            }
                        }
                    }

                    // Gaze runs after pose, on the same face and the same
                    // frame, so eye-in-head is a difference of two
                    // measurements of the same instant rather than of two
                    // things that happened to be nearby in time.
                    let mut gaze = None;
                    let mut gaze_state = SlotState::NotConfigured;
                    let mut gaze_gate = None;
                    if models.gaze.is_some() && faces.is_empty() {
                        gaze_state = SlotState::SkippedGated;
                        gaze_gate = Some(GateReason::NoFace);
                    }
                    if let (Some(gaze_model), Some(primary)) =
                        (models.gaze.as_mut(), faces.first())
                    {
                        match gaze_model.estimate(&frame, primary, head_pose) {
                            Ok(GazeOutcome::Produced { gaze: g, timings: gaze_timings }) => {
                                gaze = Some(g);
                                gaze_state = SlotState::Produced;
                                timings.preprocess_us += gaze_timings.preprocess_us;
                                timings.inference_us += gaze_timings.inference_us;
                                timings.postprocess_us += gaze_timings.postprocess_us;
                            }
                            // Gated: no value and no timings. Nothing is added
                            // to `timings`, so a skipped frame cannot read as a
                            // frame where gaze ran unusually fast.
                            Ok(GazeOutcome::Gated(reason)) => {
                                gaze_state = SlotState::SkippedGated;
                                gaze_gate = Some(reason);
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "gaze failed");
                                gaze_state = SlotState::Failed;
                            }
                        }
                    }

                    // Objects run on their own thread at their own rate. Each
                    // result is attached to the first face frame after it and
                    // consumed; later frames get an empty list and
                    // `SkippedCadence`, never a carried-forward value.
                    let (objects, objects_state) = shared.take_new_objects(&mut last_object_seq);
                    let (identity_match, identity_state) =
                        shared.take_new_identity(&mut last_identity_seq);

                    let signals = Signals {
                        seq: frame.seq,
                        t_ms: started.elapsed().as_millis() as u64,
                        faces,
                        head_pose,
                        gaze,
                        objects,
                        identity_match,
                        produced_by: SignalCoverage {
                            face: SlotState::Produced,
                            pose: pose_state,
                            gaze: gaze_state,
                            objects: objects_state,
                            identity: identity_state,
                            gaze_gate,
                        },
                        // Bucketed here, on the same frame's angles, so the
                        // label and the number beside it can never disagree.
                        debug_directions: Some(directions.update(head_pose, gaze)),
                        ..Default::default()
                    };

                    // Fusion runs here, on the detect thread, immediately after
                    // the signals it reads are assembled. It is single-threaded
                    // by construction — one owner, no locks — and costs
                    // microseconds of arithmetic, so it does not belong on a
                    // thread of its own and must not run on two.
                    for event in fusion.step(&signals, signals.t_ms) {
                        if let Event::ViolationStarted(v) | Event::ViolationEnded(v) = &event {
                            if let Ok(mut log) = shared.violation_log.lock() {
                                // A start is replaced by its own end rather
                                // than appended twice, so the log holds one
                                // row per violation with its final duration.
                                if let Some(existing) = log.iter_mut().find(|e| {
                                    e.kind == v.kind
                                        && e.subject == v.subject
                                        && e.t_start_ms == v.t_start_ms
                                }) {
                                    *existing = v.clone();
                                } else {
                                    log.push(v.clone());
                                }
                            }
                        }
                        let _ = events.send(event);
                    }
                    if let Ok(mut active) = shared.active_violations.lock() {
                        *active = fusion.active_detail();
                    }

                    let detected = Detected {
                        frame,
                        signals,
                        detect_us: timings.inference_us,
                        total_us: timings.total_us(),
                    };

                    let n = shared.frames_detected.fetch_add(1, Ordering::Relaxed) + 1;
                    shared.detect_fps.store(fps_bits(n, started), Ordering::Relaxed);
                    if let Ok(mut lat) = shared.detect_latency.lock() {
                        lat.record_us(timings.inference_us as u64);
                    }
                    if let Ok(mut lat) = shared.total_latency.lock() {
                        lat.record_us(detected.total_us as u64);
                    }
                    shared.latest.store(Arc::new(Some(Arc::new(detected))));
                }
                Err(e) => {
                    // Degrade, never die. One bad frame is not a reason to end
                    // an exam; a model that fails every frame is worth saying
                    // out loud, once.
                    consecutive_failures += 1;
                    tracing::warn!(error = %e, consecutive_failures, "face inference failed");
                    if !degraded && consecutive_failures >= 3 {
                        degraded = true;
                        let _ = events.send(Event::Degraded(DegradeReason::InferenceFailing {
                            model: "yunet".into(),
                            why: e.to_string(),
                        }));
                    }
                }
            }
        }

        // Hold the cadence. Running flat out would burn a core for frames the
        // camera has not produced yet.
        if let Some(rest) = period.checked_sub(tick.elapsed()) {
            std::thread::sleep(rest);
        }
    }

    // Close anything still open. A session that ends mid-violation would
    // otherwise record a start with no end, and the report could not say how
    // long it lasted — or whether it ever finished.
    let final_t = started.elapsed().as_millis() as u64;
    for event in fusion.finish(final_t) {
        if let Event::ViolationEnded(v) = &event {
            if let Ok(mut log) = shared.violation_log.lock() {
                if let Some(existing) = log
                    .iter_mut()
                    .find(|e| e.kind == v.kind && e.subject == v.subject && e.t_start_ms == v.t_start_ms)
                {
                    *existing = v.clone();
                }
            }
        }
        let _ = events.send(event);
    }
    if let Ok(mut active) = shared.active_violations.lock() {
        active.clear();
    }

    tracing::debug!(frames = shared.frames_detected.load(Ordering::Relaxed), "detect thread finished");
}

/// Prohibited-object detection on its own thread and its own cadence.
///
/// **Deliberately independent of face presence.** `CONTEXT.md` §18 bug #1: the
/// old module discarded object results whenever no face was detected, which
/// threw away the single case it most needed to catch — a phone held up over
/// the candidate's face, hiding it. Objects are published even when the face
/// worker sees nothing at all.
///
/// It reads the same latest-frame bus with its own cursor, so it skips
/// independently of the face worker and neither can stall the other.
pub(super) fn object_loop(mut model: YoloxNano, cfg: Config, shared: Arc<Shared>) {
    let period = Duration::from_secs_f64(1.0 / cfg.cadence.object_hz.max(0.05));
    let mut last_seen = 0u64;

    loop {
        if shared.stop.load(Ordering::Relaxed) {
            break;
        }
        if shared.capture_done.load(Ordering::Relaxed) && shared.bus.latest().seq <= last_seen {
            break;
        }

        let tick = Instant::now();
        if let Some((frame, _missed)) = shared.bus.take_new(&mut last_seen) {
            match model.detect(&frame) {
                Ok((objects, timings)) => {
                    shared.publish_objects(objects, frame.seq);
                    if let Ok(mut lat) = shared.object_latency.lock() {
                        lat.record_us(timings.inference_us as u64);
                    }
                }
                Err(e) => {
                    // Published as a failure rather than dropped, so the face
                    // worker reports `Failed` for one frame instead of the
                    // silence that a cadence skip looks like.
                    tracing::warn!(error = %e, "object detection failed");
                    shared.publish_object_failure(frame.seq);
                }
            }
        }

        if let Some(rest) = period.checked_sub(tick.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    tracing::debug!("object thread finished");
}

/// Identity, at 0.2 Hz — the slowest worker, and the one allowed to be.
///
/// A face embedding takes ~5 ms and a candidate's identity does not change
/// between frames, so anything faster spends CPU to learn nothing. The cadence
/// is also what makes the fusion rule sane: three consecutive failing checks
/// is roughly fifteen seconds of sustained mismatch, which no single bad crop
/// can fake.
pub(super) fn identity_loop(mut model: ArcFace, cfg: Config, shared: Arc<Shared>) {
    let period = Duration::from_secs_f64(1.0 / cfg.cadence.identity_hz.max(0.05));
    let mut last_seen = 0u64;

    loop {
        if shared.stop.load(Ordering::Relaxed) {
            break;
        }
        if shared.capture_done.load(Ordering::Relaxed) && shared.bus.latest().seq <= last_seen {
            break;
        }

        let tick = Instant::now();
        if let Some((frame, _missed)) = shared.bus.take_new(&mut last_seen) {
            // The identity worker runs its own face detection nowhere: it
            // reuses whatever the detect worker last saw. Running YuNet twice
            // on the same picture to save a lock would cost more than the lock.
            let face = shared
                .latest
                .load_full()
                .as_ref()
                .as_ref()
                .and_then(|d| d.signals.faces.first().cloned());

            if let Some(face) = face {
                // A low-scoring box is a bad crop, and a bad crop is where
                // false mismatches come from. Skipping is not a failure.
                if (face.score as f64) < cfg.thresholds.face.min_score {
                    shared.publish_identity(None, frame.seq, SlotState::SkippedGated);
                } else {
                    match model.embed(&frame, &face) {
                        Ok((embedding, _timings)) => {
                            let enrolling = shared.enrol_request.swap(false, Ordering::Relaxed);
                            if enrolling {
                                if let Ok(mut slot) = shared.enrolled.lock() {
                                    tracing::info!("enrolled reference face");
                                    *slot = Some(embedding.clone());
                                }
                            }
                            let reference =
                                shared.enrolled.lock().ok().and_then(|e| e.clone());
                            match reference {
                                Some(reference) => {
                                    let score = cosine_similarity(&reference, &embedding);
                                    shared.publish_identity(
                                        Some(score),
                                        frame.seq,
                                        SlotState::Produced,
                                    );
                                }
                                // Nobody has enrolled, so there is nothing to
                                // compare against. `NotConfigured` rather than
                                // a pass — an unenrolled session must not read
                                // as a verified one.
                                None => shared.publish_identity(
                                    None,
                                    frame.seq,
                                    SlotState::NotConfigured,
                                ),
                            }
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "identity embedding failed");
                            shared.publish_identity(None, frame.seq, SlotState::Failed);
                        }
                    }
                }
            } else {
                shared.publish_identity(None, frame.seq, SlotState::SkippedGated);
            }
        }

        if let Some(rest) = period.checked_sub(tick.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    tracing::debug!("identity thread finished");
}

/// Frames per second as `f32` bits, for lock-free publication through an
/// `AtomicU32`.
fn fps_bits(count: u64, since: Instant) -> u32 {
    let secs = since.elapsed().as_secs_f32();
    let fps = if secs > 0.0 { count as f32 / secs } else { 0.0 };
    fps.to_bits()
}
