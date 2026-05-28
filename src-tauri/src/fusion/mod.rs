//! Fusion: the thing that turns [`Signals`] into decisions.
//!
//! Everything upstream of here is stateless and per-frame. A face is in this
//! frame or it is not; a gaze angle describes this instant and nothing else.
//! None of that is a violation. A violation is a *temporal* claim — "absent
//! for three seconds", "a phone has been in shot repeatedly" — and this module
//! is the only place in the codebase allowed to make one.
//!
//! # The one hard rule
//!
//! [`FusionEngine::step`] is a **pure function of its arguments**. It reads no
//! clock, touches no filesystem, and contains no randomness. Time arrives as a
//! `t_ms` parameter. That is not stylistic: it is what makes a recorded
//! session replay to a byte-identical event sequence every time, which is what
//! makes threshold tuning possible at all. Tuning that requires re-running
//! models does not get done, so it has to run on recordings — and a replay
//! whose output drifts between runs cannot be diffed.
//!
//! # What is deliberately not here yet
//!
//! MODELS.md §4 argues for a weighted fused score with co-occurrence
//! escalation, and it is right: independent booleans produce independent
//! false-positive streams. Severity here is a per-rule constant from `Config`
//! instead. That is a deferral, not a disagreement — weights invented without
//! a corpus to tune them against would look like evidence while being
//! guesses. `EyesAverted` (eye-in-head as its own violation) is deferred for
//! the same reason, and the eye-in-head signal itself is still measured and
//! still on the HUD.

pub mod temporal;

use std::collections::BTreeMap;

use crate::config::Config;
use crate::types::{
    Contribution, Event, Severity, SignalSource, Signals, SlotState, Violation, ViolationKind,
};

use temporal::{DecayingScore, Edge, Ema, HoldTimer, Hysteresis};

/// A violation currently open, and what it looked like at its worst.
#[derive(Debug, Clone)]
struct Open {
    t_start_ms: u64,
    severity: Severity,
    confidence: f32,
    contributions: Vec<Contribution>,
}

/// One rule's conclusion about one frame, ready to be turned into an event.
///
/// A struct rather than seven positional arguments: `apply(kind, subject,
/// edge, confidence, contributions, ...)` is a call where transposing two
/// arguments compiles cleanly and produces a violation attributed to the wrong
/// signal.
struct Verdict {
    kind: ViolationKind,
    subject: Option<String>,
    edge: Edge,
    confidence: f32,
    contributions: Vec<Contribution>,
}

/// Identifies one violation stream. The subject distinguishes two object
/// buckets, or two lost slots, that are otherwise the same kind — without it
/// a phone appearing while a book is already flagged would look like the same
/// violation continuing.
type Key = (ViolationKind, Option<String>);

/// Everything fusion remembers between frames.
pub struct FusionEngine {
    cfg: Config,

    // face presence
    presence: Presence,
    no_face: HoldTimer,
    never_seen: HoldTimer,
    multi_face: HoldTimer,

    // head pose
    head_yaw: Ema,
    head_pitch: Ema,
    head_yaw_gate: Hysteresis,
    head_pitch_gate: Hysteresis,
    head_hold: HoldTimer,

    // gaze
    gaze_yaw: Ema,
    gaze_pitch: Ema,
    gaze_yaw_gate: Hysteresis,
    gaze_pitch_gate: Hysteresis,
    gaze_hold: HoldTimer,

    // objects, one accumulator per configured bucket
    buckets: BTreeMap<String, DecayingScore>,
    bucket_active: BTreeMap<String, bool>,
    class_to_bucket: BTreeMap<String, String>,

    // signal loss
    pose_lost: HoldTimer,
    gaze_lost: HoldTimer,

    // identity
    identity_failures: u32,
    identity_active: bool,

    open: BTreeMap<Key, Open>,
}

/// Where the session is in its "has anyone shown up" lifecycle.
///
/// Three states rather than a boolean, because "nobody ever appeared" and
/// "someone left" are different events with different severities, and the old
/// module could express only the second. CONTEXT.md §18 bug #7: the rule was
/// written as "a face was here and now is not", so a candidate who never
/// appeared at all produced no violation and the empty session was found only
/// on review.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Presence {
    NeverSeen,
    Present,
    Absent,
}

impl FusionEngine {
    pub fn new(cfg: &Config) -> Self {
        let t = &cfg.thresholds;
        let gap = t.fusion.signal_lost_ms;

        let mut buckets = BTreeMap::new();
        let mut bucket_active = BTreeMap::new();
        let mut class_to_bucket = BTreeMap::new();
        for (name, bucket) in &t.objects.buckets {
            buckets.insert(name.clone(), DecayingScore::new(t.objects.score_half_life_ms));
            bucket_active.insert(name.clone(), false);
            for class in &bucket.classes {
                class_to_bucket.insert(class.clone(), name.clone());
            }
        }

        Self {
            presence: Presence::NeverSeen,
            no_face: HoldTimer::new(t.face.no_face_hold_ms, t.face.no_face_clear_ms),
            // Once a face appears the condition is over for good, so there is
            // nothing for a clear hold to debounce.
            never_seen: HoldTimer::new(t.face.never_seen_ms, 0),
            multi_face: HoldTimer::new(t.face.multi_face_hold_ms, t.face.multi_face_clear_ms),

            head_yaw: Ema::new(t.pose.ema_alpha, gap),
            head_pitch: Ema::new(t.pose.ema_alpha, gap),
            head_yaw_gate: Hysteresis::new(t.pose.yaw_enter_deg, t.pose.yaw_exit_deg),
            head_pitch_gate: Hysteresis::new(t.pose.pitch_enter_deg, t.pose.pitch_exit_deg),
            head_hold: HoldTimer::new(t.pose.hold_ms, t.pose.clear_ms),

            gaze_yaw: Ema::new(t.gaze.ema_alpha, gap),
            gaze_pitch: Ema::new(t.gaze.ema_alpha, gap),
            gaze_yaw_gate: Hysteresis::new(t.gaze.yaw_enter_deg, t.gaze.yaw_exit_deg),
            gaze_pitch_gate: Hysteresis::new(t.gaze.pitch_enter_deg, t.gaze.pitch_exit_deg),
            gaze_hold: HoldTimer::new(t.gaze.hold_ms, t.gaze.clear_ms),

            buckets,
            bucket_active,
            class_to_bucket,

            pose_lost: HoldTimer::new(t.fusion.signal_lost_ms, t.fusion.signal_lost_clear_ms),
            gaze_lost: HoldTimer::new(t.fusion.signal_lost_ms, t.fusion.signal_lost_clear_ms),

            identity_failures: 0,
            identity_active: false,

            open: BTreeMap::new(),
            cfg: cfg.clone(),
        }
    }

    /// Advance by one frame of signals. **Pure**: no clock, no I/O.
    pub fn step(&mut self, s: &Signals, t_ms: u64) -> Vec<Event> {
        let mut events = Vec::new();
        self.step_face(s, t_ms, &mut events);
        self.step_head(s, t_ms, &mut events);
        self.step_gaze(s, t_ms, &mut events);
        self.step_objects(s, t_ms, &mut events);
        self.step_signal_loss(s, t_ms, &mut events);
        self.step_identity(s, t_ms, &mut events);
        events
    }

    /// Kinds currently raised, for the polled HUD snapshot.
    pub fn active(&self) -> Vec<ViolationKind> {
        let mut kinds: Vec<ViolationKind> = self.open.keys().map(|(k, _)| *k).collect();
        kinds.dedup();
        kinds
    }

    /// Open violations, with their subjects, for the viewer's live panel.
    pub fn active_detail(&self) -> Vec<(ViolationKind, Option<String>)> {
        self.open.keys().cloned().collect()
    }

    /// Close everything still open, as a session ends. Without this a session
    /// that ends mid-violation records a start with no end, and the report
    /// cannot say how long it lasted.
    pub fn finish(&mut self, t_ms: u64) -> Vec<Event> {
        let keys: Vec<Key> = self.open.keys().cloned().collect();
        let mut events = Vec::new();
        for key in keys {
            if let Some(open) = self.open.remove(&key) {
                events.push(Event::ViolationEnded(finished(&key, &open, t_ms)));
            }
        }
        events
    }

    // -- rule helpers -------------------------------------------------------

    fn severity_for(&self, kind: ViolationKind) -> Severity {
        self.cfg
            .thresholds
            .fusion
            .severity
            .get(kind.as_str())
            .copied()
            .unwrap_or(Severity::Medium)
    }

    /// Apply one rule's edge, emitting at most one event.
    ///
    /// This is the only place `ViolationStarted` and `ViolationEnded` are
    /// constructed, which is how the "once on each edge, never per frame"
    /// contract holds structurally rather than by everyone remembering.
    fn apply(&mut self, verdict: Verdict, t_ms: u64, events: &mut Vec<Event>) {
        let Verdict { kind, subject, edge, confidence, contributions } = verdict;
        let key = (kind, subject.clone());
        match edge {
            Edge::Rose => {
                let open = Open {
                    t_start_ms: t_ms,
                    severity: self.severity_for(kind),
                    confidence,
                    contributions,
                };
                events.push(Event::ViolationStarted(Violation {
                    kind,
                    severity: open.severity,
                    confidence: open.confidence,
                    t_start_ms: t_ms,
                    t_end_ms: None,
                    subject: subject.clone(),
                    evidence: None,
                    contributing: open.contributions.clone(),
                }));
                self.open.insert(key, open);
            }
            Edge::Fell => {
                if let Some(open) = self.open.remove(&key) {
                    events.push(Event::ViolationEnded(finished(&key, &open, t_ms)));
                }
            }
            Edge::Steady => {
                // Keep the live confidence current so the HUD can show a
                // violation strengthening, but emit nothing.
                if let Some(open) = self.open.get_mut(&key) {
                    open.confidence = confidence;
                }
            }
        }
    }

    // -- rules --------------------------------------------------------------

    fn step_face(&mut self, s: &Signals, t_ms: u64, events: &mut Vec<Event>) {
        let count = s.faces.len();
        let present = count > 0;
        match (self.presence, present) {
            (Presence::NeverSeen | Presence::Absent, true) => self.presence = Presence::Present,
            (Presence::Present, false) => self.presence = Presence::Absent,
            _ => {}
        }

        // Never-seen only runs before the first face. After that an absence is
        // an absence, and firing both would double-count one silence.
        let edge = self.never_seen.update(self.presence == Presence::NeverSeen, t_ms);
        self.apply(
            Verdict {
                kind: ViolationKind::NeverSeen,
                subject: None,
                edge,
                confidence: 1.0,
                contributions: vec![contribution(
                SignalSource::Face,
                1.0,
                format!("no face has ever been detected, {}s in", t_ms / 1000),
            )],
            },
            t_ms,
            events,
        );

        let edge = self.no_face.update(self.presence == Presence::Absent, t_ms);
        self.apply(
            Verdict {
                kind: ViolationKind::NoFace,
                subject: None,
                edge,
                confidence: 1.0,
                contributions: vec![contribution(SignalSource::Face, 1.0, "no face in frame")],
            },
            t_ms,
            events,
        );

        let edge = self.multi_face.update(count >= self.cfg.thresholds.face.multi_face_count, t_ms);
        self.apply(
            Verdict {
                kind: ViolationKind::MultipleFaces,
                subject: None,
                edge,
                confidence: 1.0,
                contributions: vec![contribution(SignalSource::Face, 1.0, format!("{count} faces in frame"))],
            },
            t_ms,
            events,
        );
    }

    fn step_head(&mut self, s: &Signals, t_ms: u64, events: &mut Vec<Event>) {
        // A gated or failed pose slot is not a pose of zero degrees. Feeding
        // the filter a zero would smooth the head back toward square during
        // exactly the frames where it could not be measured — inventing the
        // innocent answer, which is the failure mode `SlotState` exists to
        // prevent.
        let Some(pose) = s.head_pose.filter(|_| s.produced_by.pose.produced()) else {
            self.head_yaw.miss(t_ms);
            self.head_pitch.miss(t_ms);
            let edge = self.head_hold.update(false, t_ms);
            self.apply(
            Verdict {
                kind: ViolationKind::HeadTurnedAway,
                subject: None,
                edge,
                confidence: 0.0,
                contributions: vec![],
            },
            t_ms,
            events,
        );
            return;
        };

        let yaw = self.head_yaw.update(pose.yaw_deg as f64, t_ms).abs();
        let pitch = self.head_pitch.update(pose.pitch_deg as f64, t_ms).abs();
        let turned = self.head_yaw_gate.update(yaw) | self.head_pitch_gate.update(pitch);

        let mut contributions = Vec::new();
        if self.head_yaw_gate.active() {
            contributions.push(contribution(
                SignalSource::Pose,
                self.head_yaw_gate.intensity(yaw),
                format!("head yaw {yaw:.0} deg"),
            ));
        }
        if self.head_pitch_gate.active() {
            contributions.push(contribution(
                SignalSource::Pose,
                self.head_pitch_gate.intensity(pitch),
                format!("head pitch {pitch:.0} deg"),
            ));
        }
        let confidence = self
            .head_yaw_gate
            .intensity(yaw)
            .max(self.head_pitch_gate.intensity(pitch));

        let edge = self.head_hold.update(turned, t_ms);
        self.apply(
            Verdict {
                kind: ViolationKind::HeadTurnedAway,
                subject: None,
                edge,
                confidence,
                contributions,
            },
            t_ms,
            events,
        );
    }

    fn step_gaze(&mut self, s: &Signals, t_ms: u64, events: &mut Vec<Event>) {
        // A blink-gated frame contributes nothing — not a zero, not a
        // carried-forward value. `EyesTooClose` is the model declining to
        // guess, and reading it as "gaze was centred" would make blinking look
        // like compliance.
        let Some(gaze) = s.gaze.filter(|_| s.produced_by.gaze.produced()) else {
            self.gaze_yaw.miss(t_ms);
            self.gaze_pitch.miss(t_ms);
            let edge = self.gaze_hold.update(false, t_ms);
            self.apply(
            Verdict {
                kind: ViolationKind::GazeOffScreen,
                subject: None,
                edge,
                confidence: 0.0,
                contributions: vec![],
            },
            t_ms,
            events,
        );
            return;
        };

        // Degrees from here down; radians exist only on the wire. The offset
        // is §18.6's: the camera sits above the screen, so gaze pitch idles
        // around +12 to +15 when the candidate is looking straight at it.
        // Subtracting it is a frame-of-reference correction, not a decode fix.
        let offset = self.cfg.thresholds.gaze.pitch_offset_deg;
        let yaw = self.gaze_yaw.update(gaze.yaw_rad.to_degrees() as f64, t_ms).abs();
        let pitch =
            self.gaze_pitch.update(gaze.pitch_rad.to_degrees() as f64 - offset, t_ms).abs();
        let off = self.gaze_yaw_gate.update(yaw) | self.gaze_pitch_gate.update(pitch);

        let mut contributions = Vec::new();
        if self.gaze_yaw_gate.active() {
            contributions.push(contribution(
                SignalSource::Gaze,
                self.gaze_yaw_gate.intensity(yaw),
                format!("gaze yaw {yaw:.0} deg"),
            ));
        }
        if self.gaze_pitch_gate.active() {
            contributions.push(contribution(
                SignalSource::Gaze,
                self.gaze_pitch_gate.intensity(pitch),
                format!("gaze pitch {pitch:.0} deg after offset"),
            ));
        }
        let confidence = self
            .gaze_yaw_gate
            .intensity(yaw)
            .max(self.gaze_pitch_gate.intensity(pitch));

        let edge = self.gaze_hold.update(off, t_ms);
        self.apply(
            Verdict {
                kind: ViolationKind::GazeOffScreen,
                subject: None,
                edge,
                confidence,
                contributions,
            },
            t_ms,
            events,
        );
    }

    /// The two §18.5 requirements, and the only place they live.
    ///
    /// **Buckets, not literal class names.** The phone was detected but
    /// labelled `remote` (0.66) and `laptop` (0.545) on frames where it was
    /// plainly a phone; matching strings threw those away. `laptop` stays out
    /// of `handheld_device` by default because the candidate's own machine is
    /// in shot for the whole exam.
    ///
    /// **Accumulation, not single-frame thresholding.** A phone plainly in
    /// shot cleared 0.5 on only 26–42% of frames, so any per-sample threshold
    /// misses half the seconds it is there. Samples above the floor add
    /// confidence, the total decays with a half-life, and the violation fires
    /// on the total — peaky-but-persistent crosses, one loud frame does not.
    fn step_objects(&mut self, s: &Signals, t_ms: u64, events: &mut Vec<Event>) {
        let o = &self.cfg.thresholds.objects;
        let (enter, clear, floor) = (o.enter_score, o.clear_score, o.min_score);

        // Only a frame the object worker actually ran on carries information.
        // Decaying on cadence-skipped frames would bleed the score away at the
        // face worker's rate rather than at the configured half-life.
        if s.produced_by.objects.produced() {
            let mut best: BTreeMap<String, f64> = BTreeMap::new();
            for det in &s.objects {
                if (det.score as f64) < floor {
                    continue;
                }
                if let Some(bucket) = self.class_to_bucket.get(&det.label) {
                    // Best per bucket per frame, not the sum: three boxes on
                    // one phone is still one phone.
                    let slot = best.entry(bucket.clone()).or_insert(0.0);
                    *slot = slot.max(det.score as f64);
                }
            }
            for (name, score) in &mut self.buckets {
                match best.get(name) {
                    Some(v) => {
                        score.add(*v, t_ms);
                    }
                    None => {
                        score.decay_to(t_ms);
                    }
                }
            }
        }

        let names: Vec<String> = self.buckets.keys().cloned().collect();
        for name in names {
            let score = self.buckets[&name].score();
            let was = self.bucket_active[&name];
            let now = if was { score >= clear } else { score >= enter };
            self.bucket_active.insert(name.clone(), now);

            let edge = match (was, now) {
                (false, true) => Edge::Rose,
                (true, false) => Edge::Fell,
                _ => Edge::Steady,
            };
            let confidence = (((score - enter) / enter) as f32).clamp(0.0, 1.0);
            self.apply(
            Verdict {
                kind: ViolationKind::ProhibitedObject,
                subject: Some(name.clone()),
                edge,
                confidence,
                contributions: vec![contribution(
                    SignalSource::Objects,
                    confidence.max(0.01),
                    format!("{name} accumulated evidence {score:.2}"),
                )],
            },
            t_ms,
            events,
        );
        }
    }

    /// "The system cannot currently see" as a reportable condition.
    ///
    /// The soak found pose failing on 16 frames and gaze gated on 1.7% — both
    /// real, both invisible to any decision until now. Without this a
    /// candidate who covers the camera reads exactly like one behaving
    /// perfectly, which is the false negative worth more than all the others
    /// combined. Short gaps are absorbed by the hold timer: a blink is not an
    /// incident.
    fn step_signal_loss(&mut self, s: &Signals, t_ms: u64, events: &mut Vec<Event>) {
        let gate = s.produced_by.gaze_gate;
        for (slot, state, source) in [
            ("pose", s.produced_by.pose, SignalSource::Pose),
            ("gaze", s.produced_by.gaze, SignalSource::Gaze),
        ] {
            // `NotConfigured` is excluded on purpose: a slot with no model is
            // unavailable for the whole session, which the session report
            // states once. Repeating it every five seconds would bury the
            // slots that genuinely went dark mid-session.
            let lost = matches!(state, SlotState::SkippedGated | SlotState::Failed);
            let timer = if slot == "pose" { &mut self.pose_lost } else { &mut self.gaze_lost };
            let edge = timer.update(lost, t_ms);
            let detail = match (state, gate) {
                (SlotState::Failed, _) => format!("{slot}: model failing"),
                (_, Some(reason)) if slot == "gaze" => format!("{slot}: {reason:?}"),
                _ => format!("{slot}: gated"),
            };
            self.apply(
            Verdict {
                kind: ViolationKind::SignalLost,
                subject: Some(slot.to_string()),
                edge,
                confidence: 1.0,
                contributions: vec![contribution(source, 1.0, detail)],
            },
            t_ms,
            events,
        );
        }
    }

    /// Identity, deliberately slow to accuse.
    ///
    /// One bad crop — a half-turned head, motion blur, a hand across the face
    /// — drops cosine similarity well below any sane threshold. Requiring
    /// several consecutive failing checks at 0.2 Hz means roughly 15 seconds
    /// of sustained mismatch before anything is claimed. Accusing the wrong
    /// candidate of impersonation on one blurry frame is the worst output this
    /// system could produce.
    fn step_identity(&mut self, s: &Signals, t_ms: u64, events: &mut Vec<Event>) {
        if !s.produced_by.identity.produced() {
            return; // no check this frame; the counter holds its value
        }
        let Some(score) = s.identity_match else { return };
        let id = &self.cfg.thresholds.identity;

        if (score as f64) < id.cosine_enter {
            self.identity_failures += 1;
        } else if (score as f64) >= id.cosine_exit {
            self.identity_failures = 0;
        }

        let was = self.identity_active;
        let now = if was {
            // Clears only on a check that recovers past the higher bar, which
            // `cosine_exit` above already reset the counter for.
            self.identity_failures > 0
        } else {
            self.identity_failures >= id.consecutive_failures
        };
        self.identity_active = now;

        let edge = match (was, now) {
            (false, true) => Edge::Rose,
            (true, false) => Edge::Fell,
            _ => Edge::Steady,
        };
        self.apply(
            Verdict {
                kind: ViolationKind::IdentityMismatch,
                subject: None,
                edge,
                confidence: 1.0,
                contributions: vec![contribution(
                SignalSource::Identity,
                1.0,
                format!(
                    "cosine {score:.2} below {:.2} on {} consecutive checks",
                    id.cosine_enter, self.identity_failures
                ),
            )],
            },
            t_ms,
            events,
        );
    }
}

fn finished(key: &Key, open: &Open, t_ms: u64) -> Violation {
    Violation {
        kind: key.0,
        severity: open.severity,
        confidence: open.confidence,
        t_start_ms: open.t_start_ms,
        t_end_ms: Some(t_ms),
        subject: key.1.clone(),
        evidence: None,
        contributing: open.contributions.clone(),
    }
}

fn contribution(signal: SignalSource, weight: f32, detail: impl Into<String>) -> Contribution {
    Contribution { signal, weight, detail: detail.into() }
}

/// Run fusion over a whole recording. The replay entry point.
///
/// Takes a fresh engine, so two calls with the same input produce the same
/// output — the property `detect-cli replay` and the determinism test both
/// rest on.
pub fn replay(signals: &[Signals], cfg: &Config) -> Vec<Event> {
    let mut engine = FusionEngine::new(cfg);
    let mut events = Vec::new();
    let mut last_t = 0;
    for s in signals {
        last_t = s.t_ms;
        events.extend(engine.step(s, s.t_ms));
    }
    events.extend(engine.finish(last_t));
    events
}
