//! The worker loops (MODELS.md §6).
//!
//! Two threads today: capture, and detect. The spec's topology has object and
//! identity workers alongside detect, at their own cadences, reading the same
//! bus — those arrive at steps 7 and 9 and slot in without changing anything
//! here, because each worker owns its own `last_seen` cursor and its own
//! sessions.
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

                    let signals = Signals {
                        seq: frame.seq,
                        t_ms: started.elapsed().as_millis() as u64,
                        faces,
                        head_pose,
                        gaze,
                        objects,
                        produced_by: SignalCoverage {
                            face: SlotState::Produced,
                            pose: pose_state,
                            gaze: gaze_state,
                            objects: objects_state,
                            identity: SlotState::NotConfigured,
                            gaze_gate,
                        },
                        // Bucketed here, on the same frame's angles, so the
                        // label and the number beside it can never disagree.
                        debug_directions: Some(directions.update(head_pose, gaze)),
                        ..Default::default()
                    };

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

/// Frames per second as `f32` bits, for lock-free publication through an
/// `AtomicU32`.
fn fps_bits(count: u64, since: Instant) -> u32 {
    let secs = since.elapsed().as_secs_f32();
    let fps = if secs > 0.0 { count as f32 / secs } else { 0.0 };
    fps.to_bits()
}
