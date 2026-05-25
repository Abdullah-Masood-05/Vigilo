//! Head pose estimation (MODELS.md §5.3, build step 6a).
//!
//! `headpose_mobilenetv3_small.onnx` from `yakhyo/head-pose-estimation` (MIT).
//! `detect-cli inspect` reports:
//!
//! ```text
//! input             Float32  [1, 3, 224, 224]
//! rotation_matrix   Float32  [1, 3, 3]
//! ```
//!
//! Note what that is **not**: it is not the 60x60 `head-pose-estimation-adas-0001`
//! in MODELS.md §5.0, and it does not emit Euler angles. The ortho6D
//! representation is decoded to a rotation matrix inside the graph; converting
//! that to yaw/pitch/roll is our job, and the axis convention is not
//! guessable — a wrong one produces angles that look plausible and are subtly
//! wrong, which is the worst possible failure here.
//!
//! Both the normalization and the Euler extraction below are ported from the
//! model author's own `onnx_inference.py` rather than derived. Two things that
//! would have been wrong if assumed:
//!
//! 1. **This model wants RGB**, unlike YuNet which wants BGR. The reference
//!    does `cvtColor(BGR2RGB)` before inference because OpenCV hands it BGR;
//!    our frames are already RGB, so they go in unchanged.
//! 2. **ImageNet normalization**, not raw 0-255 and not a plain /255. The
//!    reference divides by 255 and then applies mean/std per channel.
//!
//! Angles are absolute, so there is no calibration step and none of
//! `CONTEXT.md` §12's `x2` fudge factors or resolution dependence.

use std::path::Path;

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use ort::session::Session;

use crate::config::Config;
use crate::error::{DetectError, Result};
use crate::types::{BBox, FaceDetection, Frame, HeadPose};

use super::{build_session, inference_error, nchw_input, StageTimings};

/// Fixed by the exported graph.
pub const INPUT_SIZE: u32 = 224;

/// ImageNet statistics, as used by the reference implementation.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct HeadPoseNet {
    session: Session,
    resizer: Resizer,
    scaled: Image<'static>,
    tensor: Vec<f32>,
    crop_expand: f32,
}

impl HeadPoseNet {
    pub fn load(path: impl AsRef<Path>, cfg: &Config) -> Result<Self> {
        let session = build_session(path, &cfg.runtime, false)?;
        let side = INPUT_SIZE as usize;

        let mut model = Self {
            session,
            resizer: Resizer::new(),
            scaled: Image::new(INPUT_SIZE, INPUT_SIZE, PixelType::U8x3),
            tensor: vec![0.0; 3 * side * side],
            crop_expand: cfg.thresholds.pose.crop_expand as f32,
        };
        model.warm_up(cfg.runtime.warmup_iters)?;
        Ok(model)
    }

    fn warm_up(&mut self, iters: u32) -> Result<()> {
        for _ in 0..iters {
            let input = nchw_input(&self.tensor, INPUT_SIZE, "headpose/warmup")?;
            self.session
                .run(ort::inputs!["input" => input])
                .map_err(|e| inference_error("headpose/warmup", e))?;
        }
        Ok(())
    }

    /// Estimate pose from a face box in the source frame.
    pub fn estimate(
        &mut self,
        frame: &Frame,
        face: &FaceDetection,
    ) -> Result<(HeadPose, StageTimings)> {
        let t0 = std::time::Instant::now();
        self.preprocess(frame, &face.bbox)?;
        let preprocess_us = t0.elapsed().as_micros() as u32;

        let t1 = std::time::Instant::now();
        let input = nchw_input(&self.tensor, INPUT_SIZE, "headpose")?;
        let outputs = self
            .session
            .run(ort::inputs!["input" => input])
            .map_err(|e| inference_error("headpose", e))?;
        let inference_us = t1.elapsed().as_micros() as u32;

        let t2 = std::time::Instant::now();
        let value = outputs.get("rotation_matrix").ok_or_else(|| {
            DetectError::Config("head pose model produced no `rotation_matrix` output".into())
        })?;
        let (_, data) =
            value.try_extract_tensor::<f32>().map_err(|e| inference_error("headpose", e))?;
        if data.len() < 9 {
            return Err(DetectError::Config(format!(
                "rotation_matrix has {} elements, expected 9",
                data.len()
            )));
        }
        let pose = rotation_matrix_to_euler(data);

        Ok((
            pose,
            StageTimings {
                preprocess_us,
                inference_us,
                postprocess_us: t2.elapsed().as_micros() as u32,
            },
        ))
    }

    /// Square, expanded crop around the face, resized to the model input.
    ///
    /// Head-pose models are sensitive to how the head sits in the frame: a
    /// tight face box crops the skull and jaw the model was trained to see,
    /// and accuracy degrades quietly rather than obviously. The reference
    /// expands by 0.2; this is configurable and defaults to 0.25.
    fn preprocess(&mut self, frame: &Frame, bbox: &BBox) -> Result<()> {
        if frame.data.len() != frame.expected_len() {
            return Err(DetectError::Config(format!(
                "frame {} is {} bytes, expected {}",
                frame.seq,
                frame.data.len(),
                frame.expected_len()
            )));
        }

        let (left, top, width, height) =
            square_crop(bbox, frame.width, frame.height, self.crop_expand);

        let src = ImageRef::new(frame.width, frame.height, &frame.data, PixelType::U8x3)
            .map_err(|e| DetectError::Config(format!("source image: {e}")))?;

        // Crop and resize in one pass — no intermediate buffer for the crop.
        self.resizer
            .resize(
                &src,
                &mut self.scaled,
                &ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(FilterType::Bilinear))
                    .crop(left, top, width, height),
            )
            .map_err(|e| DetectError::Config(format!("pose crop/resize: {e}")))?;

        let side = INPUT_SIZE as usize;
        let plane = side * side;
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

/// The expanded, squared, frame-clamped crop region in source pixels.
///
/// Squared before expansion so the aspect ratio the model sees does not depend
/// on how tall the detector happened to make the box.
fn square_crop(bbox: &BBox, frame_w: u32, frame_h: u32, expand: f32) -> (f64, f64, f64, f64) {
    let (cx, cy) = bbox.center();
    let side = bbox.w.max(bbox.h) * (1.0 + 2.0 * expand);
    let half = side * 0.5;

    // Clamp to the frame. A face at the edge yields an off-centre crop, which
    // biases the estimate slightly — better than feeding the model garbage
    // padding, and it only happens when the candidate is half out of shot.
    let x0 = (cx - half).max(0.0);
    let y0 = (cy - half).max(0.0);
    let x1 = (cx + half).min(frame_w as f32);
    let y1 = (cy + half).min(frame_h as f32);

    let w = (x1 - x0).max(1.0);
    let h = (y1 - y0).max(1.0);
    (x0 as f64, y0 as f64, w as f64, h as f64)
}

/// Rotation matrix to Euler angles, in degrees.
///
/// Ported from the model author's `rotation_matrix_to_euler`, including the
/// singular (gimbal-lock) branch. Row-major `[r00 r01 r02 r10 r11 r12 r20 r21 r22]`.
///
/// The reference returns `(pitch, yaw, roll)` in that order — pitch from the
/// x-axis, yaw from the y-axis, roll from the z-axis.
fn rotation_matrix_to_euler(r: &[f32]) -> HeadPose {
    let (r00, r01, r02) = (r[0], r[1], r[2]);
    let (r10, r11, r12) = (r[3], r[4], r[5]);
    let (r20, r21, r22) = (r[6], r[7], r[8]);
    let _ = r01;
    let _ = r02;

    let sy = (r00 * r00 + r10 * r10).sqrt();
    let singular = sy < 1e-6;

    let (pitch, roll) = if singular {
        // Gimbal lock: yaw near +/-90 degrees collapses one degree of freedom,
        // so roll is unrecoverable and is reported as zero rather than as
        // noise amplified by a near-zero denominator.
        ((-r12).atan2(r11), 0.0)
    } else {
        (r21.atan2(r22), r10.atan2(r00))
    };
    let yaw = (-r20).atan2(sy);

    HeadPose {
        pitch_deg: pitch.to_degrees(),
        yaw_deg: yaw.to_degrees(),
        roll_deg: roll.to_degrees(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Row-major rotation matrices for known rotations, so the convention is
    /// pinned by a test rather than by a comment that can drift.
    fn ry(deg: f32) -> [f32; 9] {
        let (s, c) = deg.to_radians().sin_cos();
        [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]
    }
    fn rx(deg: f32) -> [f32; 9] {
        let (s, c) = deg.to_radians().sin_cos();
        [1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c]
    }
    fn rz(deg: f32) -> [f32; 9] {
        let (s, c) = deg.to_radians().sin_cos();
        [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]
    }

    #[test]
    fn identity_is_level() {
        let p = rotation_matrix_to_euler(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(p.yaw_deg.abs() < 1e-3);
        assert!(p.pitch_deg.abs() < 1e-3);
        assert!(p.roll_deg.abs() < 1e-3);
    }

    #[test]
    fn each_axis_maps_to_its_own_angle_with_the_right_sign() {
        // The whole point of porting rather than deriving: these three
        // assertions are what a wrong axis order would break.
        let yawed = rotation_matrix_to_euler(&ry(30.0));
        assert!((yawed.yaw_deg - 30.0).abs() < 1e-2, "yaw was {}", yawed.yaw_deg);
        assert!(yawed.pitch_deg.abs() < 1e-2);
        assert!(yawed.roll_deg.abs() < 1e-2);

        let pitched = rotation_matrix_to_euler(&rx(20.0));
        assert!((pitched.pitch_deg - 20.0).abs() < 1e-2, "pitch was {}", pitched.pitch_deg);
        assert!(pitched.yaw_deg.abs() < 1e-2);

        let rolled = rotation_matrix_to_euler(&rz(15.0));
        assert!((rolled.roll_deg - 15.0).abs() < 1e-2, "roll was {}", rolled.roll_deg);
        assert!(rolled.yaw_deg.abs() < 1e-2);
    }

    #[test]
    fn negative_rotations_keep_their_sign() {
        assert!(rotation_matrix_to_euler(&ry(-25.0)).yaw_deg < -20.0);
        assert!(rotation_matrix_to_euler(&rx(-25.0)).pitch_deg < -20.0);
    }

    #[test]
    fn gimbal_lock_does_not_produce_garbage() {
        // Yaw at exactly -90 degrees makes sy ~ 0; roll must go to zero rather
        // than to whatever atan2(0, 0) happens to return.
        let p = rotation_matrix_to_euler(&ry(90.0));
        assert!((p.yaw_deg.abs() - 90.0).abs() < 1e-2, "yaw was {}", p.yaw_deg);
        assert!(p.roll_deg.abs() < 1e-3, "roll should collapse to 0, got {}", p.roll_deg);
        assert!(p.pitch_deg.is_finite());
    }

    #[test]
    fn crop_is_square_expanded_and_clamped() {
        let bbox = BBox { x: 400.0, y: 200.0, w: 100.0, h: 140.0 };
        let (x, y, w, h) = square_crop(&bbox, 1280, 720, 0.25);
        // Longest side 140, expanded by 25% each side -> 210.
        assert!((w - 210.0).abs() < 1.0, "w was {w}");
        assert!((h - 210.0).abs() < 1.0, "h was {h}");
        // Centred on the face centre (450, 270).
        assert!((x + w / 2.0 - 450.0).abs() < 1.0);
        assert!((y + h / 2.0 - 270.0).abs() < 1.0);
    }

    #[test]
    fn a_face_at_the_frame_edge_still_yields_a_valid_crop() {
        // Clamping must never produce a zero or negative region — that would
        // be a resize error at exactly the moment someone leaves the frame.
        let bbox = BBox { x: 0.0, y: 0.0, w: 80.0, h: 80.0 };
        let (x, y, w, h) = square_crop(&bbox, 1280, 720, 0.25);
        assert!(x >= 0.0 && y >= 0.0);
        assert!(w > 0.0 && h > 0.0);
        assert!(x + w <= 1280.0 && y + h <= 720.0);
    }
}
