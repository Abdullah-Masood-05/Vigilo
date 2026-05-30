//! Gaze estimation, and the eye-in-head signal (build step 6b).
//!
//! `mobileone_s0_gaze.onnx` from `yakhyo/gaze-estimation` (MIT). `detect-cli
//! inspect` reports:
//!
//! ```text
//! input   Float32  [1, 3, 448, 448]
//! yaw     Float32  [1, 90]
//! pitch   Float32  [1, 90]
//! ```
//!
//! Those are **90-bin classification heads, not regressed angles** — the
//! L2CS-Net scheme this model inherits. Decode is softmax over the bins, then
//! the expectation over bin centres.
//!
//! The bin geometry is read from the author's own postprocess, not assumed:
//! `bins = 90`, `binwidth = 4`, `angle_offset = 180`, so
//!
//! ```text
//! degrees = sum(softmax(logits) * [0..89]) * 4 - 180
//! ```
//!
//! which spans **-180..+176 degrees**, a Gaze360-style full range. Assuming the
//! MPIIGaze-style +/-90 span would have halved every angle — a signal that
//! still moves in the right direction, still looks alive in a HUD, and
//! quietly wrecks every threshold tuned against it.
//!
//! Preprocessing matches the head-pose model: RGB (the reference converts from
//! OpenCV's BGR, ours is already RGB), /255, then ImageNet mean/std.

use std::path::Path;

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use ort::session::Session;

use crate::config::{Config, ModelSlot};
use crate::error::{DetectError, Result};
use crate::types::{FaceDetection, Frame, GateReason, Gaze, HeadPose};

use super::{build_session_for, ActiveEp, inference_error, nchw_input, StageTimings};

pub const INPUT_SIZE: u32 = 448;

const BINS: usize = 90;
const BIN_WIDTH_DEG: f32 = 4.0;
const ANGLE_OFFSET_DEG: f32 = 180.0;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// What one call to [`GazeNet::estimate`] did.
///
/// Replaces returning a held previous value with zeroed timings. That was
/// wrong twice over: the caller could not tell a fresh measurement from a
/// stale one, and the zeroed timings made a skipped frame look like a frame
/// where gaze ran in no time at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GazeOutcome {
    Produced { gaze: Gaze, timings: StageTimings },
    /// Not run, and why. There is no gaze value at all — not a stale one, and
    /// not a zero, which would read as "looking straight ahead".
    Gated(GateReason),
}

pub struct GazeNet {
    /// Which execution provider this session actually got — reported,
    /// never inferred from timing.
    ep: ActiveEp,
    session: Session,
    resizer: Resizer,
    scaled: Image<'static>,
    tensor: Vec<f32>,
    min_face_score: f32,
}

impl GazeNet {
    /// Which execution provider this session is actually running on.
    ///
    /// Read from the session that was built, not from what config asked
    /// for — those differ whenever DirectML registration failed and the
    /// CPU fallback took over.
    pub fn ep(&self) -> ActiveEp {
        self.ep
    }

    pub fn load(path: impl AsRef<Path>, cfg: &Config) -> Result<Self> {
        let (session, ep) = build_session_for(
            path,
            &cfg.runtime,
            false,
            ModelSlot::Gaze,
            &[],
        )?;
        let side = INPUT_SIZE as usize;

        let mut model = Self {
            ep,
            session,
            resizer: Resizer::new(),
            scaled: Image::new(INPUT_SIZE, INPUT_SIZE, PixelType::U8x3),
            tensor: vec![0.0; 3 * side * side],
            min_face_score: cfg.thresholds.gaze.min_face_score as f32,
        };
        model.warm_up(cfg.runtime.warmup_iters)?;
        Ok(model)
    }

    fn warm_up(&mut self, iters: u32) -> Result<()> {
        for _ in 0..iters {
            let input = nchw_input(&self.tensor, INPUT_SIZE, "gaze/warmup")?;
            self.session
                .run(ort::inputs!["input" => input])
                .map_err(|e| inference_error("gaze/warmup", e))?;
        }
        Ok(())
    }

    /// Estimate gaze, and eye-in-head when head pose is available for the same
    /// frame.
    pub fn estimate(
        &mut self,
        frame: &Frame,
        face: &FaceDetection,
        head_pose: Option<HeadPose>,
    ) -> Result<GazeOutcome> {
        // Gaze during a blink, or off a face the detector is barely holding
        // on to, is not a measurement — it is noise that looks like a
        // measurement. Say so, rather than emitting anything.
        if let Some(reason) = self.gate_reason(face) {
            return Ok(GazeOutcome::Gated(reason));
        }

        let t0 = std::time::Instant::now();
        self.preprocess(frame, face)?;
        let preprocess_us = t0.elapsed().as_micros() as u32;

        let t1 = std::time::Instant::now();
        let input = nchw_input(&self.tensor, INPUT_SIZE, "gaze")?;
        let outputs = self
            .session
            .run(ort::inputs!["input" => input])
            .map_err(|e| inference_error("gaze", e))?;
        let inference_us = t1.elapsed().as_micros() as u32;

        let t2 = std::time::Instant::now();
        let yaw_deg = decode_bins(extract(&outputs, "yaw")?);
        let pitch_deg = decode_bins(extract(&outputs, "pitch")?);
        let (yaw, pitch) = (yaw_deg.to_radians(), pitch_deg.to_radians());

        Ok(GazeOutcome::Produced {
            gaze: assemble(yaw, pitch, head_pose),
            timings: StageTimings {
                preprocess_us,
                inference_us,
                postprocess_us: t2.elapsed().as_micros() as u32,
            },
        })
    }

    /// A coarse gate, deliberately.
    ///
    /// **This is not a blink detector.** A real eye-aspect-ratio needs eyelid
    /// landmarks, and YuNet gives five points with no eyelids — so a genuine
    /// EAR is not available from this model set. What this catches is the
    /// detector losing confidence or the eye keypoints collapsing, which is
    /// what a blink, motion blur and a half-turned head all look like from
    /// here. Documented as a proxy so nobody later reads it as more than it is.
    /// `None` means run it. `Some(reason)` means don't.
    pub fn gate_reason(&self, face: &FaceDetection) -> Option<GateReason> {
        gate_reason(face, self.min_face_score)
    }

    /// The reference feeds the detector's face box directly, with no
    /// expansion — unlike the pose model, which wants a wider crop.
    fn preprocess(&mut self, frame: &Frame, face: &FaceDetection) -> Result<()> {
        let b = &face.bbox;
        let x0 = b.x.max(0.0);
        let y0 = b.y.max(0.0);
        let x1 = (b.x + b.w).min(frame.width as f32);
        let y1 = (b.y + b.h).min(frame.height as f32);
        let (w, h) = ((x1 - x0).max(1.0), (y1 - y0).max(1.0));

        let src = ImageRef::new(frame.width, frame.height, &frame.data, PixelType::U8x3)
            .map_err(|e| DetectError::Config(format!("source image: {e}")))?;

        self.resizer
            .resize(
                &src,
                &mut self.scaled,
                &ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(FilterType::Bilinear))
                    .crop(x0 as f64, y0 as f64, w as f64, h as f64),
            )
            .map_err(|e| DetectError::Config(format!("gaze crop/resize: {e}")))?;

        let plane = (INPUT_SIZE * INPUT_SIZE) as usize;
        let px = self.scaled.buffer();
        for i in 0..plane {
            let s = i * 3;
            for c in 0..3 {
                self.tensor[c * plane + i] = ((px[s + c] as f32 / 255.0) - MEAN[c]) / STD[c];
            }
        }
        Ok(())
    }
}

/// Should gaze run on this face?
///
/// `None` means run it. `Some(reason)` means don't, and says which test failed
/// — the breakdown is what turns a bare skip rate into a diagnosis.
///
/// A free function so it is testable without a loaded ONNX session. The
/// previous test reimplemented this logic inline and therefore proved only
/// that the test agreed with itself.
///
/// **This is not a blink detector.** A real eye-aspect-ratio needs eyelid
/// landmarks, and YuNet gives five points with no eyelids — so a genuine EAR
/// is not available from this model set. What this catches is the detector
/// losing confidence or the eye keypoints collapsing, which is what a blink,
/// motion blur and a half-turned head all look like from here. Documented as a
/// proxy so nobody later reads it as more than it is.
pub fn gate_reason(face: &FaceDetection, min_face_score: f32) -> Option<GateReason> {
    if face.score < min_face_score {
        return Some(GateReason::LowFaceScore);
    }
    // Without keypoints the ratio test cannot run, so gaze goes ahead. The `?`
    // returns "no gate reason", which is the permissive answer here.
    let k = face.keypoints?;
    let dx = k.left_eye.0 - k.right_eye.0;
    let dy = k.left_eye.1 - k.right_eye.1;
    let inter_eye = (dx * dx + dy * dy).sqrt();
    if face.bbox.w <= 0.0 {
        return Some(GateReason::DegenerateBox);
    }
    // Eyes sit at roughly a quarter to a half of face width apart. Outside
    // that band the keypoints are not describing a forward-facing face.
    let ratio = inter_eye / face.bbox.w;
    if ratio < 0.15 {
        Some(GateReason::EyesTooClose)
    } else if ratio > 0.75 {
        Some(GateReason::EyesTooFar)
    } else {
        None
    }
}

/// Combine raw gaze with head pose to get eye-in-head.
///
/// `eye = gaze - head`, in radians, which is only meaningful because both are
/// in the same camera frame and the same sign convention. Head pose is stored
/// in degrees, so the conversion happens here rather than being assumed
/// anywhere else (`CONTEXT.md` §19 item 17).
fn assemble(yaw_rad: f32, pitch_rad: f32, head: Option<HeadPose>) -> Gaze {
    let (eye_yaw, eye_pitch) = match head {
        Some(h) => (
            Some(yaw_rad - h.yaw_deg.to_radians()),
            Some(pitch_rad - h.pitch_deg.to_radians()),
        ),
        None => (None, None),
    };
    Gaze { yaw_rad, pitch_rad, eye_yaw_rad: eye_yaw, eye_pitch_rad: eye_pitch }
}

fn extract<'a>(outputs: &'a ort::session::SessionOutputs, name: &str) -> Result<&'a [f32]> {
    let value = outputs
        .get(name)
        .ok_or_else(|| DetectError::Config(format!("gaze model produced no `{name}` output")))?;
    let (_, data) = value.try_extract_tensor::<f32>().map_err(|e| inference_error("gaze", e))?;
    Ok(data)
}

/// Softmax over the bins, then expectation over bin centres, in degrees.
fn decode_bins(logits: &[f32]) -> f32 {
    let n = logits.len().min(BINS);
    if n == 0 {
        return 0.0;
    }
    let max = logits[..n].iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    let mut weighted = 0.0;
    for (i, &logit) in logits[..n].iter().enumerate() {
        let e = (logit - max).exp();
        sum += e;
        weighted += e * i as f32;
    }
    if sum <= 0.0 {
        return 0.0;
    }
    (weighted / sum) * BIN_WIDTH_DEG - ANGLE_OFFSET_DEG
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BBox, FaceKeypoints};

    fn one_hot(bin: usize) -> Vec<f32> {
        let mut v = vec![-30.0; BINS];
        v[bin] = 30.0;
        v
    }

    #[test]
    fn a_single_dominant_bin_decodes_to_its_own_centre() {
        // Bin 45 is the midpoint: 45 * 4 - 180 = 0 degrees, i.e. straight ahead.
        assert!(decode_bins(&one_hot(45)).abs() < 0.5);
        // Bin 0 is the far end of the span.
        assert!((decode_bins(&one_hot(0)) + 180.0).abs() < 0.5);
        // Bin 89 is the other end: 89 * 4 - 180 = 176.
        assert!((decode_bins(&one_hot(89)) - 176.0).abs() < 0.5);
    }

    #[test]
    fn the_span_is_plus_minus_180_not_plus_minus_90() {
        // The assumption this test exists to prevent. If someone "fixes" the
        // constants to an MPIIGaze-style span, this fails loudly instead of
        // silently halving every angle.
        let full_span = decode_bins(&one_hot(89)) - decode_bins(&one_hot(0));
        assert!(full_span > 300.0, "span was {full_span}, expected ~356 degrees");
    }

    #[test]
    fn a_flat_distribution_decodes_to_the_middle() {
        assert!(decode_bins(&vec![1.0; BINS]).abs() < 2.5);
    }

    #[test]
    fn eye_in_head_is_the_difference_and_is_absent_without_pose() {
        let head = HeadPose { yaw_deg: 30.0, pitch_deg: 0.0, roll_deg: 0.0 };
        // Gaze pointing exactly where the head points means the eyes are
        // centred in their sockets: eye-in-head must be ~zero.
        let g = assemble(30f32.to_radians(), 0.0, Some(head));
        assert!(g.eye_yaw_rad.unwrap().abs() < 1e-5);

        // Head straight, gaze off to the side: that is pure eye movement, and
        // it is the signal the old module could not express.
        let g = assemble(20f32.to_radians(), 0.0, Some(HeadPose {
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
        }));
        assert!((g.eye_yaw_rad.unwrap() - 20f32.to_radians()).abs() < 1e-5);

        // Without pose there is no eye-in-head to report — `None`, not zero,
        // because zero would read as "eyes centred".
        let g = assemble(0.3, 0.1, None);
        assert!(g.eye_yaw_rad.is_none() && g.eye_pitch_rad.is_none());
    }

    fn face(score: f32, inter_eye: f32, width: f32) -> FaceDetection {
        FaceDetection {
            bbox: BBox { x: 0.0, y: 0.0, w: width, h: width },
            score,
            keypoints: Some(FaceKeypoints {
                right_eye: (0.0, 0.0),
                left_eye: (inter_eye, 0.0),
                nose: (0.0, 0.0),
                right_mouth: (0.0, 0.0),
                left_mouth: (0.0, 0.0),
            }),
        }
    }

    #[test]
    fn the_reliability_gate_rejects_low_scores_and_collapsed_eyes() {
        let min = Config::default().thresholds.gaze.min_face_score as f32;

        assert_eq!(gate_reason(&face(0.95, 40.0, 100.0), min), None, "a normal face should pass");
        assert_eq!(
            gate_reason(&face(0.10, 40.0, 100.0), min),
            Some(GateReason::LowFaceScore),
            "a barely-held face should not"
        );
        assert_eq!(
            gate_reason(&face(0.95, 2.0, 100.0), min),
            Some(GateReason::EyesTooClose),
            "collapsed eye keypoints should not"
        );
        assert_eq!(
            gate_reason(&face(0.95, 90.0, 100.0), min),
            Some(GateReason::EyesTooFar),
            "eyes wider apart than the face is not a forward-facing face"
        );
    }

    #[test]
    fn each_gate_reason_is_reachable_and_distinct() {
        // A breakdown is only useful if the causes can actually be told apart.
        // If two tests collapsed onto one reason, the C1 diagnostic would
        // report a cause that never fires and hide one that does.
        let min = Config::default().thresholds.gaze.min_face_score as f32;
        let mut degenerate = face(0.95, 40.0, 100.0);
        degenerate.bbox.w = 0.0;

        let reasons = [
            gate_reason(&face(0.10, 40.0, 100.0), min),
            gate_reason(&face(0.95, 2.0, 100.0), min),
            gate_reason(&face(0.95, 90.0, 100.0), min),
            gate_reason(&degenerate, min),
        ];
        let distinct: std::collections::HashSet<_> = reasons.iter().collect();
        assert_eq!(distinct.len(), 4, "reasons collapsed: {reasons:?}");
    }
}
