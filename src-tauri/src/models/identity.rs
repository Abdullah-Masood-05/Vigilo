//! Face recognition — ArcFace `w600k_mbf` (MODELS.md §5.5, build step 9).
//!
//! `detect-cli inspect` reports:
//!
//! ```text
//! input.1   Float32  [?, 3, 112, 112]     <- dynamic batch
//! 516       Float32  [?, 512]
//! ```
//!
//! The tensor names are `input.1` and `516` — ONNX node numbers, not chosen
//! names. Guessing `input`/`output` here fails at run time with a name lookup
//! error, which is the good outcome; what would be worse is guessing the
//! *normalisation* and getting embeddings that are subtly wrong. A wrong
//! preprocessing here does not error, it just makes every comparison noise,
//! and the failure looks like "identity checking is unreliable" rather than
//! like a bug.
//!
//! So the preprocessing below is **ported verbatim**, not re-derived, from the
//! previous system's working `face_recognition.rs`: 112x112, Triangle resize,
//! RGB channel order, NCHW planar layout, and `(px / 127.5) - 1.0` per
//! channel. That implementation was known to produce usable embeddings; this
//! one has to match it byte for byte in intent or the comparison against it is
//! meaningless.
//!
//! What is **added** here, and was not in the reference: the crop is aligned
//! on YuNet's eye keypoints before resizing. ArcFace is trained on faces
//! normalised to a canonical eye position and is unusually sensitive to it —
//! feeding it an axis-aligned box means a tilted head produces a different
//! embedding for the same person, which shows up as a drifting cosine score
//! and looks exactly like an impostor.

use std::path::Path;

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use ort::session::Session;

use crate::config::{Config, ModelSlot};
use crate::error::{DetectError, Result};
use crate::types::{FaceDetection, Frame};

use super::{build_session_for, ActiveEp, inference_error, StageTimings};

/// Fixed by the exported graph.
pub const INPUT_SIZE: u32 = 112;
/// Embedding width.
pub const EMBEDDING_DIM: usize = 512;

const INPUT_NAME: &str = "input.1";
const OUTPUT_NAME: &str = "516";

/// Pin the one dynamic axis in the whole model set.
///
/// This is the only model of the five that does not ship fully static:
/// `input.1` is `[?, 3, 112, 112]`. DirectML wants every shape known when the
/// session is created, and an axis left free means the EP quietly leaves that
/// subgraph on the CPU — which reads as "DirectML didn't help" rather than as
/// "DirectML never ran here".
///
/// The axis is literally named `None`, which is what PyTorch's ONNX exporter
/// writes for an unnamed batch dimension. That is confirmed from
/// `detect-cli inspect`, not guessed — overriding a name the graph does not
/// use is a silent no-op, so guessing would have looked like it worked.
///
/// Overriding at load rather than re-exporting the weights keeps the model
/// file exactly as downloaded, so there is nothing to re-do when it is
/// fetched fresh.
pub(crate) const ARCFACE_DIMS: &[(&str, i64)] = &[("None", 1)];

pub struct ArcFace {
    /// Which execution provider this session actually got — reported,
    /// never inferred from timing.
    ep: ActiveEp,
    session: Session,
    resizer: Resizer,
    scaled: Image<'static>,
    tensor: Vec<f32>,
}

impl ArcFace {
    /// Which execution provider this session is actually running on.
    ///
    /// Read from the session that was built, not from what config asked
    /// for — those differ whenever DirectML registration failed and the
    /// CPU fallback took over.
    pub fn ep(&self) -> ActiveEp {
        self.ep
    }

    pub fn load(path: impl AsRef<Path>, cfg: &Config) -> Result<Self> {
        // `false` = the small-model thread budget. This is a 13 MB graph run
        // at 0.2 Hz; §18.2 is the standing lesson that a thread budget tuned
        // for a model in isolation is not a budget for one sharing a machine
        // with a 15 Hz worker. One extra idle pool is exactly what tripled the
        // face worker's latency last time.
        let (session, ep) = build_session_for(
            path,
            &cfg.runtime,
            false,
            ModelSlot::Identity,
            ARCFACE_DIMS,
        )?;
        let side = INPUT_SIZE as usize;
        let mut model = Self {
            ep,
            session,
            resizer: Resizer::new(),
            scaled: Image::new(INPUT_SIZE, INPUT_SIZE, PixelType::U8x3),
            tensor: vec![0.0; 3 * side * side],
        };
        model.warm_up(cfg.runtime.warmup_iters)?;
        Ok(model)
    }

    fn warm_up(&mut self, iters: u32) -> Result<()> {
        for _ in 0..iters {
            let input = batch_of_one(&self.tensor)?;
            self.session
                .run(ort::inputs![INPUT_NAME => input])
                .map_err(|e| inference_error("arcface/warmup", e))?;
        }
        Ok(())
    }

    /// A 512-d embedding for one face, L2-normalised so comparison is a dot
    /// product.
    pub fn embed(
        &mut self,
        frame: &Frame,
        face: &FaceDetection,
    ) -> Result<(Vec<f32>, StageTimings)> {
        let t0 = std::time::Instant::now();
        self.preprocess(frame, face)?;
        let preprocess_us = t0.elapsed().as_micros() as u32;

        let t1 = std::time::Instant::now();
        let input = batch_of_one(&self.tensor)?;
        let outputs = self
            .session
            .run(ort::inputs![INPUT_NAME => input])
            .map_err(|e| inference_error("arcface", e))?;
        let inference_us = t1.elapsed().as_micros() as u32;

        let t2 = std::time::Instant::now();
        let value = outputs.get(OUTPUT_NAME).ok_or_else(|| {
            DetectError::Config(format!("arcface produced no `{OUTPUT_NAME}` output"))
        })?;
        let (_, data) =
            value.try_extract_tensor::<f32>().map_err(|e| inference_error("arcface", e))?;
        if data.len() != EMBEDDING_DIM {
            return Err(DetectError::Config(format!(
                "arcface returned {} values, expected {EMBEDDING_DIM}",
                data.len()
            )));
        }
        let embedding = l2_normalise(data);

        Ok((
            embedding,
            StageTimings {
                preprocess_us,
                inference_us,
                postprocess_us: t2.elapsed().as_micros() as u32,
            },
        ))
    }

    /// Crop, align, resize, normalise.
    ///
    /// The normalisation is the reference's, unchanged: RGB, planar NCHW,
    /// `(px / 127.5) - 1.0`. Our frames are already RGB, so unlike the
    /// reference there is no colour conversion to undo.
    fn preprocess(&mut self, frame: &Frame, face: &FaceDetection) -> Result<()> {
        if frame.data.len() != frame.expected_len() {
            return Err(DetectError::Config(format!(
                "frame {} is {} bytes, expected {}",
                frame.seq,
                frame.data.len(),
                frame.expected_len()
            )));
        }

        let (left, top, width, height) = aligned_crop(face, frame.width, frame.height);

        let src = ImageRef::new(frame.width, frame.height, &frame.data, PixelType::U8x3)
            .map_err(|e| DetectError::Config(format!("source image: {e}")))?;

        // The reference used `image`'s Triangle filter; Bilinear here is the
        // same kernel under a different name, and it crops and resizes in one
        // pass rather than materialising the crop first.
        self.resizer
            .resize(
                &src,
                &mut self.scaled,
                &ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(FilterType::Bilinear))
                    .crop(left, top, width, height),
            )
            .map_err(|e| DetectError::Config(format!("arcface crop/resize: {e}")))?;

        let side = INPUT_SIZE as usize;
        let plane = side * side;
        let pixels = self.scaled.buffer();
        for i in 0..plane {
            let px = &pixels[i * 3..i * 3 + 3];
            self.tensor[i] = (px[0] as f32 / 127.5) - 1.0;
            self.tensor[plane + i] = (px[1] as f32 / 127.5) - 1.0;
            self.tensor[2 * plane + i] = (px[2] as f32 / 127.5) - 1.0;
        }
        Ok(())
    }
}

/// Square crop centred between the eyes, sized from the inter-ocular distance.
///
/// ArcFace's training data is aligned so the eyes sit at a canonical position,
/// and it degrades quietly when they do not. Deriving the crop from the
/// keypoints rather than the box means a head that is tilted or sitting low in
/// its bounding box still presents the eyes where the model expects them.
///
/// Falls back to the bounding box when YuNet gave no keypoints — a slightly
/// worse embedding beats no embedding, and the caller cannot tell the
/// difference anyway.
fn aligned_crop(face: &FaceDetection, frame_w: u32, frame_h: u32) -> (f64, f64, f64, f64) {
    let (cx, cy, side) = match &face.keypoints {
        Some(k) => {
            let (lx, ly) = k.left_eye;
            let (rx, ry) = k.right_eye;
            let eye_cx = (lx + rx) * 0.5;
            let eye_cy = (ly + ry) * 0.5;
            let interocular = ((lx - rx).powi(2) + (ly - ry).powi(2)).sqrt().max(1.0);
            // ArcFace's canonical 112x112 template puts the eyes about 38 px
            // apart, i.e. roughly a third of the crop width. Inverting that
            // gives the crop side, and the centre drops below the eye line by
            // the same proportion so the mouth is included.
            let side = interocular * (112.0 / 38.0);
            (eye_cx, eye_cy + side * 0.12, side)
        }
        None => {
            let (cx, cy) = face.bbox.center();
            (cx, cy, face.bbox.w.max(face.bbox.h) * 1.15)
        }
    };

    let half = side * 0.5;
    let x0 = (cx - half).max(0.0);
    let y0 = (cy - half).max(0.0);
    let x1 = (cx + half).min(frame_w as f32);
    let y1 = (cy + half).min(frame_h as f32);
    ((x0 as f64), (y0 as f64), (x1 - x0).max(1.0) as f64, (y1 - y0).max(1.0) as f64)
}

/// The batch axis is dynamic in the graph; pinning it to 1 means ORT never has
/// to reallocate for a shape change, and a batch of anything other than one
/// face is not something this pipeline can produce.
///
/// A free function rather than a method so the borrow of the tensor and the
/// mutable borrow of the session do not overlap — the same reason
/// [`super::nchw_input`] is one.
fn batch_of_one(tensor: &[f32]) -> Result<ort::value::TensorRef<'_, f32>> {
    ort::value::TensorRef::from_array_view((
        vec![1i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64],
        tensor,
    ))
    .map_err(|e| inference_error("arcface", e))
}

fn l2_normalise(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Cosine similarity between two embeddings.
///
/// Both are L2-normalised by [`ArcFace::embed`], so this is a plain dot
/// product. Kept as a named function anyway because "why is this not divided
/// by the magnitudes" is a question worth answering once, here, rather than at
/// every call site.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>().clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalised_vectors_compare_to_one_with_themselves() {
        let v = l2_normalise(&[3.0, 4.0, 0.0]);
        assert!((v.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn opposite_vectors_compare_to_minus_one() {
        let a = l2_normalise(&[1.0, 0.0]);
        let b = l2_normalise(&[-1.0, 0.0]);
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn mismatched_lengths_do_not_panic() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0]), 0.0);
    }

    #[test]
    fn the_aligned_crop_is_square_and_inside_the_frame() {
        use crate::types::{BBox, FaceKeypoints};
        let face = FaceDetection {
            bbox: BBox { x: 100.0, y: 100.0, w: 200.0, h: 240.0 },
            score: 0.95,
            keypoints: Some(FaceKeypoints {
                right_eye: (160.0, 180.0),
                left_eye: (240.0, 180.0),
                nose: (200.0, 230.0),
                right_mouth: (170.0, 270.0),
                left_mouth: (230.0, 270.0),
            }),
        };
        let (x, y, w, h) = aligned_crop(&face, 1280, 720);
        assert!(x >= 0.0 && y >= 0.0);
        assert!(x + w <= 1280.0 && y + h <= 720.0);
        // 80 px between the eyes, scaled by 112/38, is a ~236 px crop.
        assert!((w - 235.8).abs() < 1.0, "unexpected crop width {w}");
    }

    #[test]
    fn a_face_without_keypoints_still_produces_a_crop() {
        use crate::types::BBox;
        let face = FaceDetection {
            bbox: BBox { x: 10.0, y: 10.0, w: 100.0, h: 100.0 },
            score: 0.9,
            keypoints: None,
        };
        let (_, _, w, h) = aligned_crop(&face, 640, 480);
        assert!(w > 0.0 && h > 0.0);
    }
}
