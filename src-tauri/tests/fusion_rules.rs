//! Fusion behaviour, on hand-built signal sequences.
//!
//! Separate from `fusion_replay.rs`, which pins the recording format these
//! rest on. Five properties matter here and they run in milliseconds: the
//! peaky phone fires, single-frame noise does not, a steady turn crosses once,
//! a calm session is silent, and the same input twice gives the same output.
//!
//! Hand-built rather than recorded on purpose. A corpus of real clips is
//! better evidence and is being gathered separately; these exist so a
//! regression is caught by `cargo test` rather than by a person staring at a
//! HUD wondering why nothing fired.

use deepscreen_viewer::fusion;
use deepscreen_viewer::types::{
    BBox, Event, FaceDetection, GateReason, Gaze, HeadPose, ObjectDetection, SignalCoverage,
    Signals, SlotState, ViolationKind,
};
use deepscreen_viewer::Config;

/// A frame with a face present and nothing going on.
fn calm_frame(seq: u64, t_ms: u64) -> Signals {
    Signals {
        seq,
        t_ms,
        faces: vec![FaceDetection {
            bbox: BBox { x: 500.0, y: 250.0, w: 210.0, h: 250.0 },
            score: 0.95,
            keypoints: None,
        }],
        head_pose: Some(HeadPose { yaw_deg: 1.0, pitch_deg: -2.0, roll_deg: 0.5 }),
        gaze: Some(Gaze {
            // 12.5 degrees in radians — exactly the configured offset, so this
            // is a candidate looking straight at the screen once corrected.
            pitch_rad: 0.218,
            yaw_rad: 0.02,
            eye_yaw_rad: None,
            eye_pitch_rad: None,
        }),
        objects: vec![],
        identity_match: None,
        eye_aspect: None,
        produced_by: SignalCoverage {
            face: SlotState::Produced,
            pose: SlotState::Produced,
            gaze: SlotState::Produced,
            objects: SlotState::SkippedCadence,
            identity: SlotState::NotConfigured,
            gaze_gate: None,
        },
        debug_directions: None,
    }
}

fn phone(score: f32) -> ObjectDetection {
    ObjectDetection {
        class_id: 67,
        label: "cell phone".into(),
        score,
        bbox: BBox { x: 700.0, y: 380.0, w: 90.0, h: 170.0 },
    }
}

fn labelled(label: &str, class_id: u32, score: f32) -> ObjectDetection {
    ObjectDetection {
        class_id,
        label: label.into(),
        score,
        bbox: BBox { x: 700.0, y: 380.0, w: 90.0, h: 170.0 },
    }
}

fn starts(events: &[Event], kind: ViolationKind) -> Vec<f32> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::ViolationStarted(v) if v.kind == kind => Some(v.t_start_ms as f32 / 1000.0),
            _ => None,
        })
        .collect()
}

/// 15 Hz for `secs` seconds, with a hook to modify individual frames.
fn sequence(secs: u64, mut f: impl FnMut(&mut Signals, u64)) -> Vec<Signals> {
    (0..secs * 15)
        .map(|i| {
            let t_ms = i * 1000 / 15;
            let mut s = calm_frame(i, t_ms);
            f(&mut s, t_ms);
            s
        })
        .collect()
}

/// True on exactly the one frame per second the 1 Hz object worker runs on.
///
/// `t_ms` is `i * 1000 / 15`, so a whole second lands exactly on i = 0, 15,
/// 30... A range test like `t_ms % 1000 < 67` looks equivalent and matches
/// *two* frames per second, which silently doubles the sample rate and makes
/// every accumulator assertion below mean something other than it says.
fn object_tick(t_ms: u64) -> bool {
    t_ms.is_multiple_of(1000)
}

#[test]
fn a_peaky_phone_fires_despite_never_being_confidently_seen() {
    // The §18.5 case, and the reason the accumulator exists. A phone plainly in
    // shot cleared 0.5 on only 26-42% of frames, so a per-sample threshold
    // misses most of the seconds it is there. These are real-shaped scores:
    // sampled at 1 Hz, mostly unconvincing, never absent.
    let scores = [0.62, 0.30, 0.71, 0.28, 0.80, 0.33, 0.77, 0.29, 0.68, 0.31];
    let signals = sequence(20, |s, t_ms| {
        if object_tick(t_ms) {
            if let Some(score) = scores.get((t_ms / 1000) as usize) {
                s.objects = vec![phone(*score)];
                s.produced_by.objects = SlotState::Produced;
            }
        }
    });

    let events = fusion::replay(&signals, &Config::default());
    let fired = starts(&events, ViolationKind::ProhibitedObject);
    assert_eq!(
        fired.len(),
        1,
        "a phone present for ten seconds must fire exactly once, got {fired:?}. \
         Single-frame thresholding is what this test exists to prevent."
    );

    let subject = events
        .iter()
        .find_map(|e| match e {
            Event::ViolationStarted(v) if v.kind == ViolationKind::ProhibitedObject => {
                v.subject.clone()
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(subject, "handheld_device", "report the bucket, not the COCO class");
}

#[test]
fn a_phone_the_detector_called_remote_still_counts() {
    // §18.5's other half: the phone was labelled `remote` at 0.66 on frames
    // where it was plainly a phone. A literal-string allowlist throws those
    // away — a real detection lost to a label.
    let signals = sequence(20, |s, t_ms| {
        if object_tick(t_ms) && t_ms < 10_000 {
            s.objects = vec![labelled("remote", 65, 0.66)];
            s.produced_by.objects = SlotState::Produced;
        }
    });

    let events = fusion::replay(&signals, &Config::default());
    assert_eq!(
        starts(&events, ViolationKind::ProhibitedObject).len(),
        1,
        "a phone the detector called `remote` is still a phone"
    );
}

#[test]
fn a_single_noisy_detection_never_fires() {
    // The other half of the accumulator's job, and the one that decides whether
    // the system is usable at all: one spurious frame must raise nothing.
    let signals = sequence(20, |s, t_ms| {
        if t_ms == 0 {
            s.objects = vec![phone(0.9)];
            s.produced_by.objects = SlotState::Produced;
        }
    });

    let events = fusion::replay(&signals, &Config::default());
    assert!(
        starts(&events, ViolationKind::ProhibitedObject).is_empty(),
        "a lone detection is noise, not evidence: {events:?}"
    );
}

#[test]
fn a_laptop_does_not_count_as_a_handheld_device() {
    // Deliberate policy rather than an oversight: the candidate's own machine
    // is in shot for the whole exam, so putting `laptop` in the bucket would
    // fire continuously for every honest candidate.
    let signals = sequence(20, |s, t_ms| {
        if object_tick(t_ms) {
            s.objects = vec![labelled("laptop", 63, 0.95)];
            s.produced_by.objects = SlotState::Produced;
        }
    });

    let events = fusion::replay(&signals, &Config::default());
    assert!(starts(&events, ViolationKind::ProhibitedObject).is_empty());
}

#[test]
fn a_steady_head_turn_raises_exactly_one_violation() {
    // Hysteresis' purpose at the fusion level rather than the primitive level:
    // a head turning steadily past the threshold produces one violation, not
    // one per frame and not a burst around the boundary.
    let signals = sequence(10, |s, t_ms| {
        let yaw = (t_ms as f32 / 1000.0) * 6.0; // 0 -> 60 degrees over ten seconds
        s.head_pose = Some(HeadPose { yaw_deg: yaw, pitch_deg: 0.0, roll_deg: 0.0 });
    });

    let events = fusion::replay(&signals, &Config::default());
    let fired = starts(&events, ViolationKind::HeadTurnedAway);
    assert_eq!(fired.len(), 1, "a monotonic turn crosses once; got {fired:?}");
}

#[test]
fn a_calm_session_produces_no_violations_at_all() {
    // The acceptance test that matters most. False positives are what make a
    // proctoring system unusable — a candidate sitting still and behaving must
    // produce silence.
    let signals = sequence(120, |_, _| {});
    let events = fusion::replay(&signals, &Config::default());
    let raised: Vec<_> =
        events.iter().filter(|e| matches!(e, Event::ViolationStarted(_))).collect();
    assert!(raised.is_empty(), "two calm minutes must be silent, got {raised:?}");
}

#[test]
fn leaving_frame_raises_no_face_and_returning_clears_it() {
    let signals = sequence(20, |s, t_ms| {
        if (5_000..12_000).contains(&t_ms) {
            s.faces.clear();
            s.head_pose = None;
            s.gaze = None;
            s.produced_by.pose = SlotState::SkippedGated;
            s.produced_by.gaze = SlotState::SkippedGated;
            s.produced_by.gaze_gate = Some(GateReason::NoFace);
        }
    });

    let events = fusion::replay(&signals, &Config::default());
    let fired = starts(&events, ViolationKind::NoFace);
    assert_eq!(fired.len(), 1, "one absence, one violation: {fired:?}");
    // The face goes at 5 s and the hold is 2500 ms.
    assert!((fired[0] - 7.5).abs() < 0.2, "fired at {}s, expected ~7.5s", fired[0]);

    let ended = events.iter().any(|e| {
        matches!(e, Event::ViolationEnded(v)
            if v.kind == ViolationKind::NoFace && v.t_end_ms.is_some())
    });
    assert!(ended, "returning to frame must close it, with a duration");
}

#[test]
fn a_session_where_nobody_ever_appears_is_its_own_violation() {
    // CONTEXT.md §18 bug #7: the old rule was "a face was here and now is not",
    // so a candidate who never showed up produced nothing at all, and the empty
    // session was discovered only on review.
    let signals = sequence(20, |s, _| {
        s.faces.clear();
        s.head_pose = None;
        s.gaze = None;
        s.produced_by.pose = SlotState::NotConfigured;
        s.produced_by.gaze = SlotState::NotConfigured;
    });

    let events = fusion::replay(&signals, &Config::default());
    let fired = starts(&events, ViolationKind::NeverSeen);
    assert_eq!(fired.len(), 1, "expected NeverSeen, got {events:?}");
    assert!((fired[0] - 10.0).abs() < 0.2, "fired at {}s, expected ~10s", fired[0]);
    assert!(
        starts(&events, ViolationKind::NoFace).is_empty(),
        "NeverSeen owns this silence; firing NoFace as well would double-count it"
    );
}

#[test]
fn a_sustained_gate_reports_the_signal_as_lost() {
    // The soak found pose failing on 16 frames and gaze gated on 1.7%, both
    // invisible to any decision until now. A covered camera must not read as
    // "all clear".
    let signals = sequence(20, |s, t_ms| {
        if t_ms >= 4_000 {
            s.gaze = None;
            s.produced_by.gaze = SlotState::Failed;
        }
    });

    let events = fusion::replay(&signals, &Config::default());
    let fired = starts(&events, ViolationKind::SignalLost);
    assert_eq!(fired.len(), 1, "expected one SignalLost, got {fired:?}");
    assert!((fired[0] - 9.0).abs() < 0.2, "fired at {}s, expected ~9s", fired[0]);
}

#[test]
fn blinking_is_absorbed_rather_than_reported() {
    // Gaze gated for a few frames every couple of seconds is a person blinking.
    // If that reached the violation log the log would be useless.
    let signals = sequence(60, |s, t_ms| {
        if (t_ms / 100).is_multiple_of(20) {
            s.gaze = None;
            s.produced_by.gaze = SlotState::SkippedGated;
            s.produced_by.gaze_gate = Some(GateReason::EyesTooClose);
        }
    });

    let events = fusion::replay(&signals, &Config::default());
    assert!(
        starts(&events, ViolationKind::SignalLost).is_empty(),
        "blinks must be absorbed by the hold timer, got {events:?}"
    );
    assert!(
        starts(&events, ViolationKind::GazeOffScreen).is_empty(),
        "and a blink is certainly not gaze going off-screen"
    );
}

#[test]
fn the_same_recording_replays_to_the_same_events_every_time() {
    // The property everything else rests on. If replay is not deterministic, a
    // diff of two tuning runs is meaningless and no threshold change can be
    // attributed to the change that caused it.
    let signals = sequence(30, |s, t_ms| {
        if object_tick(t_ms) && (3_000..15_000).contains(&t_ms) {
            s.objects = vec![phone(0.55)];
            s.produced_by.objects = SlotState::Produced;
        }
        if (20_000..25_000).contains(&t_ms) {
            s.faces.clear();
        }
    });

    let cfg = Config::default();
    let first = fusion::replay(&signals, &cfg);
    let second = fusion::replay(&signals, &cfg);
    assert_eq!(first, second, "replay must be a pure function of its input");
    assert!(!first.is_empty(), "this clip should produce something worth comparing");
}

#[test]
fn thresholds_retune_from_a_partial_toml_without_touching_code() {
    // The deliverable is not the numbers — they are wrong and known to be. It
    // is that changing them is a TOML edit and a re-run.
    let signals = sequence(20, |s, t_ms| {
        if object_tick(t_ms) && t_ms < 6_000 {
            s.objects = vec![phone(0.30)];
            s.produced_by.objects = SlotState::Produced;
        }
    });

    let default_events = fusion::replay(&signals, &Config::default());
    assert!(starts(&default_events, ViolationKind::ProhibitedObject).is_empty());

    // Both ends, because `validate` rejects an exit threshold above an enter
    // one — lowering `enter_score` alone would leave clear (0.6) above enter
    // (0.5) and the violation could never clear. Getting that error here rather
    // than a latched-on violation in a live session is the point of it.
    let tuned: Config =
        toml::from_str("[thresholds.objects]\nenter_score = 0.5\nclear_score = 0.2\n").unwrap();
    tuned.validate().unwrap();
    let tuned_events = fusion::replay(&signals, &tuned);
    assert_eq!(
        starts(&tuned_events, ViolationKind::ProhibitedObject).len(),
        1,
        "lowering enter_score in TOML alone must change the outcome"
    );
}

#[test]
fn identity_needs_several_consecutive_failures_before_accusing() {
    // One bad crop — a half-turned head, motion blur, a hand across the face —
    // drops cosine similarity below any sane threshold. Accusing the wrong
    // candidate of impersonation on one blurry frame is the worst output this
    // system could produce.
    let one_bad = sequence(60, |s, t_ms| {
        if t_ms == 10_000 {
            s.identity_match = Some(0.10);
            s.produced_by.identity = SlotState::Produced;
        }
    });
    let events = fusion::replay(&one_bad, &Config::default());
    assert!(
        starts(&events, ViolationKind::IdentityMismatch).is_empty(),
        "a single failing check must never accuse: {events:?}"
    );

    // Sustained is different: three checks at 0.2 Hz is ~15 s of mismatch.
    // A check every 5 s, i.e. the real 0.2 Hz cadence.
    let sustained = sequence(60, |s, t_ms| {
        if t_ms >= 10_000 && t_ms.is_multiple_of(5_000) {
            s.identity_match = Some(0.10);
            s.produced_by.identity = SlotState::Produced;
        }
    });
    let events = fusion::replay(&sustained, &Config::default());
    assert_eq!(
        starts(&events, ViolationKind::IdentityMismatch).len(),
        1,
        "sustained mismatch must fire exactly once, got {events:?}"
    );
}

#[test]
fn an_unenrolled_session_never_claims_a_match_or_a_mismatch() {
    // `NotConfigured` means nobody enrolled, so there is nothing to compare
    // against. It must not read as a pass — an unverified session is not a
    // verified one.
    let signals = sequence(60, |s, _| {
        s.identity_match = None;
        s.produced_by.identity = SlotState::NotConfigured;
    });
    let events = fusion::replay(&signals, &Config::default());
    assert!(starts(&events, ViolationKind::IdentityMismatch).is_empty());
}

#[test]
fn a_violation_event_serialises_to_the_shape_the_front_end_reads() {
    // `dist/main.js` switches on `payload.event` and then reads `kind`,
    // `severity`, `subject`, `t_start_ms` and `t_end_ms` from the same object.
    // The enum is internally tagged, so those fields are flattened alongside
    // the tag — and if that ever changes, the front end silently logs nothing
    // rather than failing loudly. This is the test that makes it fail loudly.
    let signals = sequence(20, |s, t_ms| {
        if (5_000..12_000).contains(&t_ms) {
            s.faces.clear();
        }
    });
    let events = fusion::replay(&signals, &Config::default());

    let started = events
        .iter()
        .find(|e| matches!(e, Event::ViolationStarted(_)))
        .expect("this clip raises a violation");
    let json = serde_json::to_value(started).unwrap();

    assert_eq!(json["event"], "violation_started");
    assert_eq!(json["kind"], "no_face");
    assert!(json["severity"].is_string(), "severity must be a plain string");
    assert!(json["t_start_ms"].is_number());
    assert!(json.get("subject").is_some(), "subject must be present, even as null");

    let ended = events
        .iter()
        .find(|e| matches!(e, Event::ViolationEnded(_)))
        .expect("and closes it");
    let json = serde_json::to_value(ended).unwrap();
    assert_eq!(json["event"], "violation_ended");
    assert!(json["t_end_ms"].is_number(), "a closed violation must carry its end");
}
