//! Replay regression suite (MODELS.md §10).
//!
//! The eventual shape: each labelled clip has a recorded `Signals` JSONL and an
//! expected violation list, and this file asserts that fusion turns one into
//! the other — with no camera, no GPU, and no models. Fusion lands at build
//! step 8; until then these tests lock down the thing fusion depends on, which
//! is that a recording survives the round trip through JSONL unchanged.
//!
//! The control clips matter more than the violation clips. False positives are
//! what make a proctoring system unusable, and you cannot measure a
//! false-positive rate without footage of innocent behaviour.

use vigilo::types::{
    BBox, EyeAspect, FaceDetection, Gaze, HeadPose, ObjectDetection, SignalCoverage, Signals,
    SlotState,
};
use vigilo::Config;

/// A synthetic clip: someone looking progressively further left while a phone
/// enters frame. Stands in for real footage until the corpus is recorded.
fn synthetic_look_away_with_phone(frames: u64, fps: f32) -> Vec<Signals> {
    // The same tracker the detect worker owns, driven over the same sweep, so
    // the recording carries the labels a real session would rather than a stub
    // that happens to serialize.
    let mut directions =
        vigilo::DirectionTracker::new(&Config::default().thresholds.debug_direction);

    (0..frames)
        .map(|seq| {
            let t_ms = (seq as f64 * 1000.0 / fps as f64).round() as u64;
            let yaw = -(seq as f32) * 1.5; // turning left over time
            let head_pose = Some(HeadPose { yaw_deg: yaw, pitch_deg: 2.0, roll_deg: 0.5 });
            let gaze = Some(Gaze {
                pitch_rad: 0.05,
                yaw_rad: yaw.to_radians() * 0.8,
                eye_yaw_rad: None,
                eye_pitch_rad: None,
            });
            Signals {
                seq,
                t_ms,
                faces: vec![FaceDetection {
                    bbox: BBox { x: 400.0, y: 200.0, w: 220.0, h: 260.0 },
                    score: 0.95,
                    keypoints: None,
                }],
                head_pose,
                gaze,
                objects: if t_ms >= 2000 {
                    vec![ObjectDetection {
                        class_id: 67,
                        label: "cell phone".into(),
                        score: 0.71,
                        bbox: BBox { x: 700.0, y: 380.0, w: 90.0, h: 170.0 },
                    }]
                } else {
                    vec![]
                },
                identity_match: Some(0.68),
                eye_aspect: Some(EyeAspect { left: 0.29, right: 0.30 }),
                produced_by: SignalCoverage {
                    face: SlotState::Produced,
                    pose: SlotState::Produced,
                    gaze: SlotState::Produced,
                    // Before the phone appears the object worker has run and
                    // seen nothing; `Produced` with an empty list. That is the
                    // distinction the old boolean could not express.
                    objects: if t_ms >= 2000 {
                        SlotState::Produced
                    } else {
                        SlotState::SkippedCadence
                    },
                    identity: if seq % 75 == 0 {
                        SlotState::Produced
                    } else {
                        SlotState::SkippedCadence
                    },
                    gaze_gate: None,
                },
                debug_directions: Some(directions.update(head_pose, gaze)),
            }
        })
        .collect()
}

fn to_jsonl(signals: &[Signals]) -> String {
    signals.iter().map(|s| serde_json::to_string(s).unwrap()).collect::<Vec<_>>().join("\n")
}

fn from_jsonl(text: &str) -> Vec<Signals> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn a_recording_survives_the_jsonl_round_trip_exactly() {
    // This is the property every replay-based tuning session rests on: if the
    // round trip is lossy, tuned thresholds do not mean what they claim.
    let original = synthetic_look_away_with_phone(90, 15.0);
    let replayed = from_jsonl(&to_jsonl(&original));
    assert_eq!(original, replayed);
}

#[test]
fn the_timebase_is_frame_derived_not_wall_clock() {
    // Replay runs flat out, so timestamps must come from the frame index and
    // the declared rate. Otherwise a fast machine and a slow one would tune to
    // different thresholds from the same clip.
    let signals = synthetic_look_away_with_phone(45, 15.0);
    assert_eq!(signals[0].t_ms, 0);
    assert_eq!(signals[15].t_ms, 1000);
    assert_eq!(signals[44].t_ms, 2933);
    assert!(signals.windows(2).all(|w| w[1].t_ms > w[0].t_ms), "timestamps must be monotonic");
}

#[test]
fn coverage_distinguishes_absent_from_never_ran() {
    let signals = synthetic_look_away_with_phone(45, 15.0);

    let early = &signals[0];
    assert!(early.objects.is_empty());
    assert!(
        !early.produced_by.objects.produced(),
        "an empty list from a frame the object worker never touched is not \
         evidence that nothing was there"
    );

    let late = signals.iter().find(|s| s.t_ms >= 2000).unwrap();
    assert!(late.produced_by.objects.produced());
    assert_eq!(late.objects[0].label, "cell phone");
}

#[test]
fn a_recording_survives_the_round_trip_with_slot_states_intact() {
    // The states are the part fusion reads to decide whether a signal is
    // absent or merely quiet, so they have to survive serialisation exactly.
    // Long enough to cross the 2 s mark where the phone appears; a shorter
    // clip is all cadence skips and would assert nothing.
    let signals = synthetic_look_away_with_phone(45, 15.0);
    let back = from_jsonl(&to_jsonl(&signals));
    assert_eq!(signals, back);

    let states: Vec<_> = back.iter().map(|s| s.produced_by.objects).collect();
    assert!(
        states.contains(&SlotState::SkippedCadence) && states.contains(&SlotState::Produced),
        "the clip should exercise both a cadence skip and a real result"
    );
}

#[test]
fn a_steady_turn_changes_its_label_exactly_once() {
    // The property hysteresis exists to provide, over a realistic sweep rather
    // than a hand-picked pair of numbers: the head turns steadily from square
    // to well past the threshold, so the label must go CENTER -> LEFT and stay
    // there. Any flapping around the boundary shows up as extra transitions.
    use vigilo::Horizontal;

    let signals = synthetic_look_away_with_phone(45, 15.0);
    let labels: Vec<Horizontal> = signals
        .iter()
        .map(|s| s.debug_directions.unwrap().head.unwrap().horizontal)
        .collect();

    assert_eq!(labels.first(), Some(&Horizontal::Center), "starts square to the camera");
    assert_eq!(labels.last(), Some(&Horizontal::Left), "ends well past the threshold");

    let transitions = labels.windows(2).filter(|w| w[0] != w[1]).count();
    assert_eq!(
        transitions, 1,
        "a monotonic turn should cross once; {transitions} crossings means the \
         label is flapping on the boundary. Labels: {labels:?}"
    );

    // And the sign is the one the module documents: negative yaw is the
    // subject's left. If this fails, the readout is mirrored.
    assert!(signals.last().unwrap().head_pose.unwrap().yaw_deg < 0.0);
}

#[test]
fn old_recordings_still_parse_after_signal_slots_are_added() {
    // The corpus is the regression suite; it must outlive schema growth.
    let line = r#"{"seq":7,"t_ms":466,"faces":[],"head_pose":null}"#;
    let s: Signals = serde_json::from_str(line).unwrap();
    assert_eq!(s.seq, 7);
    assert!(s.eye_aspect.is_none());
    assert_eq!(s.produced_by.face, SlotState::NotConfigured);
}

#[test]
fn tuned_thresholds_load_from_a_partial_config_file() {
    // Tuning loop ergonomics: change one number, rerun replay. If a tuning
    // file had to restate the whole config, nobody would iterate.
    let tuned = r#"
[thresholds.pose]
yaw_enter_deg = 30.0
yaw_exit_deg = 22.0
hold_ms = 1200
"#;
    let cfg: Config = toml::from_str(tuned).unwrap();
    cfg.validate().unwrap();
    assert_eq!(cfg.thresholds.pose.yaw_enter_deg, 30.0);
    assert_eq!(cfg.thresholds.pose.hold_ms, 1200);
    // Untouched signals keep their defaults.
    assert_eq!(cfg.thresholds.face.no_face_hold_ms, 2500);
    assert_eq!(cfg.cadence.face_hz, 15.0);
}

// TODO(step 8): replace the synthetic clip above with the recorded corpus and
// assert violations, e.g.
//
//   let events = fusion::replay(&signals, &cfg);
//   assert_violation(&events, ViolationKind::HeadTurnedAway, 1.6..2.4);
//   assert_violation(&events, ViolationKind::ProhibitedObject, 3.9..4.3);
//   assert_no_violations(&control_clip_events);
