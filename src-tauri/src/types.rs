//! Core types (MODELS.md §2).
//!
//! The seam that matters: [`Signals`] is stateless per-frame data produced by
//! models; [`Violation`] is a temporal decision produced only by fusion.
//! `Signals` is `Serialize + Deserialize` so a session can be recorded once and
//! replayed through fusion thousands of times with zero inference.

use std::sync::Arc;
use std::time::{Instant, SystemTime};

use serde::{Deserialize, Serialize};

/// Raw camera frame. RGB8, tightly packed. Shared, never copied.
#[derive(Clone)]
pub struct Frame {
    pub data: Arc<[u8]>,
    pub width: u32,
    pub height: u32,
    pub seq: u64,
    pub captured_at: Instant,
}

impl Frame {
    /// Sentinel for the initial state of the frame bus. `seq == 0` and empty
    /// data; workers skip it because it never matches a new sequence number.
    pub fn empty() -> Self {
        Self {
            data: Arc::from(Vec::new()),
            width: 0,
            height: 0,
            seq: 0,
            captured_at: Instant::now(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || self.data.is_empty()
    }

    /// Bytes a tightly packed RGB8 buffer of this size should occupy.
    pub fn expected_len(&self) -> usize {
        self.width as usize * self.height as usize * 3
    }
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Frame")
            .field("seq", &self.seq)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bytes", &self.data.len())
            .finish()
    }
}

/// Axis-aligned box in **source frame pixel coordinates** — letterboxing is
/// undone by each model's postprocess before it ever reaches this type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl BBox {
    pub fn area(&self) -> f32 {
        (self.w * self.h).max(0.0)
    }

    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w * 0.5, self.y + self.h * 0.5)
    }

    pub fn intersection(&self, other: &BBox) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.w).min(other.x + other.w);
        let y2 = (self.y + self.h).min(other.y + other.h);
        ((x2 - x1).max(0.0)) * ((y2 - y1).max(0.0))
    }

    pub fn iou(&self, other: &BBox) -> f32 {
        let inter = self.intersection(other);
        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

/// YuNet's five keypoints, in source frame pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FaceKeypoints {
    pub right_eye: (f32, f32),
    pub left_eye: (f32, f32),
    pub nose: (f32, f32),
    pub right_mouth: (f32, f32),
    pub left_mouth: (f32, f32),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaceDetection {
    pub bbox: BBox,
    pub score: f32,
    pub keypoints: Option<FaceKeypoints>,
}

/// Absolute head pose. No calibration baseline subtracted — the regressor
/// emits absolute angles, which is why pose needs no calibration step
/// (MODELS.md §4).
///
/// Units are in the field names on purpose. Head pose is degrees and gaze is
/// radians, and fusion subtracts one from the other to get eye-in-head
/// (`CONTEXT.md` §19 item 17). A silent unit mismatch there would produce a
/// signal that moves in the right direction with the wrong magnitude, which
/// survives every smoke test and then quietly wrecks threshold tuning.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HeadPose {
    /// Positive = turned toward the **subject's own right**, which an
    /// unmirrored camera draws on the *left* of the picture.
    ///
    /// Stated from the subject's side because "left" is meaningless otherwise,
    /// and the two readings are opposites. The convention is not a choice made
    /// here — it follows from the author's own `draw_axis`, which projects the
    /// nose axis to `sin(-yaw_deg)`. See [`crate::direction`] for the full
    /// derivation and the tests that pin it.
    pub yaw_deg: f32,
    /// Positive = looking up.
    pub pitch_deg: f32,
    pub roll_deg: f32,
}

/// Gaze direction in **radians**, camera coordinate frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Gaze {
    /// Combined head + eye direction, which is what the model regresses.
    pub yaw_rad: f32,
    pub pitch_rad: f32,
    /// Eye-in-head: gaze minus head pose, i.e. where the eyes point
    /// *independently of where the head points*. `None` until head pose is
    /// available for the same frame. This is the signal the old module could
    /// not express at all.
    pub eye_yaw_rad: Option<f32>,
    pub eye_pitch_rad: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectDetection {
    pub class_id: u32,
    pub label: String,
    pub score: f32,
    pub bbox: BBox,
}

/// Eye aspect ratio, used to suppress "gaze off" during blinks (MODELS.md §4).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EyeAspect {
    pub left: f32,
    pub right: f32,
}

impl EyeAspect {
    pub fn mean(&self) -> f32 {
        (self.left + self.right) * 0.5
    }
}

/// Everything the models saw in one frame. Pure data, no history, no decisions.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Signals {
    pub seq: u64,
    /// Milliseconds since session start. Monotonic, not wall clock.
    pub t_ms: u64,
    pub faces: Vec<FaceDetection>,
    pub head_pose: Option<HeadPose>,
    pub gaze: Option<Gaze>,
    /// Only populated on frames where `produced_by.objects` is
    /// [`SlotState::Produced`]. Empty otherwise — an object result is never
    /// carried forward onto a later frame, because "there was a phone 900 ms
    /// ago" is an inference and inference belongs to fusion, not here.
    pub objects: Vec<ObjectDetection>,
    /// Cosine similarity against the enrolled embedding.
    pub identity_match: Option<f32>,
    pub eye_aspect: Option<EyeAspect>,
    /// Which signal slots actually ran for this frame. A signal that is
    /// `None` because its model is degraded must not be read as "absent".
    pub produced_by: SignalCoverage,
    /// Plain-language direction labels for the live HUD.
    ///
    /// **Temporary, and named to say so.** It rides along with the rest of the
    /// signals rather than being derived by whoever draws them, so the labels
    /// always describe the same frame as the angles printed beside them. When
    /// fusion lands, its smoothed state replaces this and the field goes away
    /// — see [`crate::direction`].
    pub debug_directions: Option<crate::direction::DebugDirections>,
}

/// What one model slot did on one frame.
///
/// A boolean could only say "ran" or "didn't", and the interesting cases all
/// live inside "didn't". Fusion has to tell an expected skip from a failure
/// from a model that was never loaded, because reading an absent signal as a
/// benign one is a false-negative generator — the worst failure mode a
/// proctoring system has.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotState {
    /// Ran on this frame. Its value in [`Signals`] is a measurement — including
    /// when that measurement is "nothing there".
    Produced,
    /// Deliberately not run: the input did not meet the model's entry
    /// conditions. A blink, a half-turned head, or no face to crop from.
    /// Expected, and not a fault.
    SkippedGated,
    /// Runs slower than the face worker and had no new result for this frame.
    /// Expected: the object worker is 1 Hz against a 15 Hz face worker.
    SkippedCadence,
    /// The model was loaded and erred on this frame. Degraded.
    Failed,
    /// No model in this slot. The signal is unavailable for the whole session,
    /// not absent on this frame.
    #[default]
    NotConfigured,
}

impl SlotState {
    /// True only for [`Self::Produced`]. Anything else means the matching field
    /// in [`Signals`] carries no measurement.
    pub fn produced(self) -> bool {
        matches!(self, Self::Produced)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Produced => "produced",
            Self::SkippedGated => "gated",
            Self::SkippedCadence => "cadence",
            Self::Failed => "failed",
            Self::NotConfigured => "absent",
        }
    }
}

/// Why the gaze model declined to run.
///
/// Carried per-frame so the skip rate can be attributed rather than guessed.
/// A gate that fires constantly for one reason is a threshold problem; one
/// that fires evenly across reasons is a framing problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateReason {
    /// No face detected on this frame, so there was nothing to crop.
    NoFace,
    /// The detector was barely holding the face.
    LowFaceScore,
    /// Eye keypoints collapsed together — a blink, motion blur, or a profile.
    EyesTooClose,
    /// Eye keypoints implausibly far apart for the box; the keypoints are not
    /// describing a forward-facing face.
    EyesTooFar,
    /// Degenerate face box, so the ratio test could not be computed.
    DegenerateBox,
}

impl GateReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoFace => "no_face",
            Self::LowFaceScore => "low_face_score",
            Self::EyesTooClose => "eyes_too_close",
            Self::EyesTooFar => "eyes_too_far",
            Self::DegenerateBox => "degenerate_box",
        }
    }
}

/// Per-frame record of what each model slot did.
///
/// **Per frame, not per session.** The previous version was a set of booleans
/// where `objects` was a sticky global meaning "has ever run", so a frame the
/// object worker never touched was indistinguishable from one where it looked
/// and found nothing. That is precisely the ambiguity this type exists to
/// remove, and it did not remove it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SignalCoverage {
    pub face: SlotState,
    pub pose: SlotState,
    pub gaze: SlotState,
    pub objects: SlotState,
    pub identity: SlotState,
    /// Set when `gaze` is [`SlotState::SkippedGated`], so the skip rate can be
    /// broken down by cause.
    pub gaze_gate: Option<GateReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViolationKind {
    /// No face in frame for longer than the hold.
    NoFace,
    /// A face was never seen at all — distinct from, and worse than, `NoFace`.
    NeverSeen,
    MultipleFaces,
    HeadTurnedAway,
    GazeOffScreen,
    ProhibitedObject,
    /// Cosine similarity against the enrolled embedding fell through the floor.
    IdentityDrift,
}

impl ViolationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ViolationKind::NoFace => "no_face",
            ViolationKind::NeverSeen => "never_seen",
            ViolationKind::MultipleFaces => "multiple_faces",
            ViolationKind::HeadTurnedAway => "head_turned_away",
            ViolationKind::GazeOffScreen => "gaze_off_screen",
            ViolationKind::ProhibitedObject => "prohibited_object",
            ViolationKind::IdentityDrift => "identity_drift",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

/// A downscaled JPEG kept out of the hot path and rate limited. Stored as a
/// path once written to disk, so `Violation` stays cheap to clone and send.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub path: std::path::PathBuf,
    pub seq: u64,
}

/// A decision. Produced only by the fusion layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Violation {
    pub kind: ViolationKind,
    pub severity: Severity,
    pub confidence: f32,
    pub t_start: SystemTime,
    /// `None` = still ongoing.
    pub t_end: Option<SystemTime>,
    pub evidence: Option<EvidenceRef>,
    /// Which signals argued for this violation, and how strongly. This is what
    /// lets a proctor reading the report see *why* (MODELS.md §4).
    #[serde(default)]
    pub contributing: Vec<Contribution>,
}

/// Which model slot an argument came from. An enum rather than a string so a
/// typo in fusion is a compile error and a recorded report stays readable
/// after the signal set changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalSource {
    Face,
    Pose,
    Gaze,
    Objects,
    Identity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contribution {
    pub signal: SignalSource,
    pub weight: f32,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "reason", content = "detail")]
pub enum DegradeReason {
    CameraLost(String),
    ModelUnavailable { model: String, why: String },
    InferenceFailing { model: String, why: String },
    ExecutionProviderFallback { model: String, from: String, to: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "event")]
pub enum Event {
    ViolationStarted(Violation),
    ViolationEnded(Violation),
    CalibrationProgress { pct: f32 },
    CalibrationComplete,
    Degraded(DegradeReason),
    Recovered,
}

/// Level-triggered view for the HUD. Polled by the UI at whatever rate it
/// wants, never pushed (MODELS.md §3). There is deliberately no API that would
/// let continuous values be pushed at frame rate.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DetectorState {
    pub seq: u64,
    pub t_ms: u64,
    pub face_count: usize,
    /// Smoothed, not raw — this is what the HUD should draw.
    pub head_pose: Option<HeadPose>,
    pub gaze: Option<Gaze>,
    pub identity_match: Option<f32>,
    pub active_violations: Vec<ViolationKind>,
    pub degraded: Vec<DegradeReason>,
    pub calibrated: bool,
    pub object_count: usize,
    pub stats: PipelineStats,
}

/// Throughput and latency of the running pipeline.
///
/// Skipped frames are expected and healthy — the detect worker runs slower
/// than capture and deliberately drops whatever it missed rather than queueing
/// it. A stale frame is worthless. What matters is that the number is
/// *visible*: a sudden climb means saturation (MODELS.md §6 rule 3).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PipelineStats {
    pub frames_captured: u64,
    pub frames_detected: u64,
    pub frames_skipped: u64,
    pub capture_fps: f32,
    pub detect_fps: f32,
    /// Model time only.
    pub detect_p50_us: u64,
    pub detect_p95_us: u64,
    /// Including preprocess and decode.
    pub total_p50_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signals_roundtrip_through_json() {
        let s = Signals {
            seq: 42,
            t_ms: 1400,
            faces: vec![FaceDetection {
                bbox: BBox { x: 10.0, y: 20.0, w: 100.0, h: 120.0 },
                score: 0.97,
                keypoints: None,
            }],
            head_pose: Some(HeadPose { yaw_deg: -31.5, pitch_deg: 4.0, roll_deg: 1.0 }),
            gaze: Some(Gaze { pitch_rad: 0.1, yaw_rad: -0.4, eye_yaw_rad: None, eye_pitch_rad: None }),
            objects: vec![],
            identity_match: Some(0.61),
            eye_aspect: Some(EyeAspect { left: 0.28, right: 0.30 }),
            produced_by: SignalCoverage {
                face: SlotState::Produced,
                pose: SlotState::Produced,
                gaze: SlotState::SkippedGated,
                gaze_gate: Some(GateReason::EyesTooClose),
                ..Default::default()
            },
            debug_directions: Some(crate::direction::DebugDirections {
                head: Some(crate::direction::Axes {
                    horizontal: crate::direction::Horizontal::Left,
                    vertical: crate::direction::Vertical::Center,
                }),
                gaze: None,
                eye: None,
                frame_of_reference: crate::direction::FrameOfReference::Subject,
            }),
        };
        let line = serde_json::to_string(&s).unwrap();
        assert_eq!(s, serde_json::from_str::<Signals>(&line).unwrap());
    }

    #[test]
    fn signals_tolerates_missing_fields() {
        // Forward compatibility: an old JSONL recording must still replay
        // after new signal slots are added.
        let s: Signals = serde_json::from_str(r#"{"seq":1,"t_ms":66}"#).unwrap();
        assert_eq!(s.seq, 1);
        assert!(s.faces.is_empty());
        // The default is `NotConfigured`, not `Produced`: a recording that
        // says nothing about a slot must not be read as that slot having run.
        assert_eq!(s.produced_by.objects, SlotState::NotConfigured);
        assert!(!s.produced_by.objects.produced());
    }

    #[test]
    fn an_empty_object_list_is_only_a_measurement_when_the_slot_produced() {
        // The whole point of the type. Both frames below carry
        // `objects: []`, and they mean opposite things.
        let looked_and_saw_nothing = Signals {
            produced_by: SignalCoverage { objects: SlotState::Produced, ..Default::default() },
            ..Default::default()
        };
        let never_ran = Signals {
            produced_by: SignalCoverage {
                objects: SlotState::SkippedCadence,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(looked_and_saw_nothing.objects, never_ran.objects);
        assert!(looked_and_saw_nothing.produced_by.objects.produced());
        assert!(
            !never_ran.produced_by.objects.produced(),
            "a cadence skip must never read as evidence of absence"
        );
    }

    #[test]
    fn iou_of_identical_boxes_is_one() {
        let b = BBox { x: 0.0, y: 0.0, w: 10.0, h: 10.0 };
        assert!((b.iou(&b) - 1.0).abs() < 1e-6);
        let far = BBox { x: 100.0, y: 100.0, w: 10.0, h: 10.0 };
        assert_eq!(b.iou(&far), 0.0);
    }
}
