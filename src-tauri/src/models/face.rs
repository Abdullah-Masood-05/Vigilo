//! YuNet face detection (MODELS.md §5.3).
//!
//! 75,856 parameters against RetinaFace's 27.3 M, MIT licensed, and it emits
//! five keypoints in the same pass — which is what gives pose and gaze their
//! crops for free.
//!
//! # What `detect-cli inspect` says, versus what the spec assumed
//!
//! The released `face_detection_yunet_2023mar.onnx` has a **fixed 640x640
//! input**, not the 320x320 in MODELS.md §5.0, and twelve outputs rather than
//! one packed tensor:
//!
//! ```text
//! input                Float32  [1, 3, 640, 640]
//! cls_8/16/32          Float32  [1, 6400|1600|400, 1]
//! obj_8/16/32          Float32  [1, 6400|1600|400, 1]
//! bbox_8/16/32         Float32  [1, 6400|1600|400, 4]
//! kps_8/16/32          Float32  [1, 6400|1600|400, 10]
//! ```
//!
//! OpenCV's DNN module silently reshapes the input to whatever you ask for;
//! `ort` will not. So we letterbox to 640x640 and undo it afterwards. This is
//! the "classic YuNet-outside-OpenCV stumble" the spec budgets an hour for.
//!
//! There is no NMS in the graph either — unlike YOLO26's end-to-end head — so
//! anchor decoding and suppression live here.

use std::path::Path;

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use ort::session::Session;

use crate::config::Config;
use crate::error::{DetectError, Result};
use crate::types::{BBox, FaceDetection, FaceKeypoints, Frame};

use super::{build_session, inference_error, nchw_input, StageTimings};

/// The model's fixed input side. Not configurable — it is baked into the file.
pub const INPUT_SIZE: u32 = 640;

/// Feature-map strides the head predicts at.
const STRIDES: [u32; 3] = [8, 16, 32];

/// YuNet was trained through OpenCV, whose `blobFromImage` hands over **BGR**
/// with no scaling and no mean subtraction. Our frames are RGB, so the channel
/// planes are filled in reverse. Flip this and detection degrades rather than
/// vanishes, which is exactly the kind of bug that survives a smoke test —
/// hence the named constant and this comment.
const CHANNEL_ORDER_BGR: bool = true;

pub struct YuNet {
    session: Session,
    score_threshold: f32,
    nms_threshold: f32,
    top_k: usize,
    resizer: Resizer,
    /// Reused resize target — allocating per frame is pure hot-loop pressure.
    scaled: Image<'static>,
    /// Reused NCHW input tensor.
    tensor: Vec<f32>,
    /// Letterbox state from the most recent preprocess.
    letterbox: Letterbox,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Letterbox {
    scale: f32,
    width: u32,
    height: u32,
}

impl YuNet {
    pub fn load(path: impl AsRef<Path>, cfg: &Config) -> Result<Self> {
        let session = build_session(path, &cfg.runtime, false)?;
        let side = INPUT_SIZE as usize;

        let mut model = Self {
            session,
            score_threshold: cfg.thresholds.face.min_score as f32,
            nms_threshold: cfg.thresholds.face.nms_threshold as f32,
            top_k: cfg.thresholds.face.top_k,
            resizer: Resizer::new(),
            scaled: Image::new(INPUT_SIZE, INPUT_SIZE, PixelType::U8x3),
            tensor: vec![0.0; 3 * side * side],
            letterbox: Letterbox::default(),
        };
        model.warm_up(cfg.runtime.warmup_iters)?;
        Ok(model)
    }

    /// Run a few inferences on zero tensors so the first real frame is not an
    /// outlier (MODELS.md §9).
    fn warm_up(&mut self, iters: u32) -> Result<()> {
        for _ in 0..iters {
            // Borrow the two fields separately: the tensor view is alive
            // across the run, so going through `&self` would conflict.
            let input = nchw_input(&self.tensor, INPUT_SIZE, "yunet/warmup")?;
            self.session
                .run(ort::inputs!["input" => input])
                .map_err(|e| inference_error("yunet/warmup", e))?;
        }
        Ok(())
    }

    pub fn detect(&mut self, frame: &Frame) -> Result<Vec<FaceDetection>> {
        self.detect_timed(frame).map(|(faces, _)| faces)
    }

    /// Detection with per-stage timings.
    ///
    /// Split three ways because that is the only way to know which stage to
    /// attack (MODELS.md §11: instrument from day one). Preprocessing is not
    /// free at 15 Hz, and lumping it into "inference" hides the cost that is
    /// usually easiest to remove.
    pub fn detect_timed(&mut self, frame: &Frame) -> Result<(Vec<FaceDetection>, StageTimings)> {
        let t0 = std::time::Instant::now();
        self.preprocess(frame)?;
        let preprocess_us = t0.elapsed().as_micros() as u32;

        let t1 = std::time::Instant::now();
        let input = nchw_input(&self.tensor, INPUT_SIZE, "yunet")?;
        let outputs = self
            .session
            .run(ort::inputs!["input" => input])
            .map_err(|e| inference_error("yunet", e))?;
        let inference_us = t1.elapsed().as_micros() as u32;

        let t2 = std::time::Instant::now();
        let mut candidates = Vec::new();
        for stride in STRIDES {
            decode_stride(
                stride,
                extract(&outputs, &format!("cls_{stride}"))?,
                extract(&outputs, &format!("obj_{stride}"))?,
                extract(&outputs, &format!("bbox_{stride}"))?,
                extract(&outputs, &format!("kps_{stride}"))?,
                self.score_threshold,
                self.letterbox,
                &mut candidates,
            );
        }

        let faces = nms(candidates, self.nms_threshold, self.top_k);
        Ok((
            faces,
            StageTimings {
                preprocess_us,
                inference_us,
                postprocess_us: t2.elapsed().as_micros() as u32,
            },
        ))
    }

    /// Letterbox into the fixed input, then write planar BGR.
    ///
    /// The scaled image is placed at the **top left** and the remainder left
    /// black, so undoing it is a single divide by `scale` with no offset. A
    /// centred letterbox looks tidier and buys nothing but two more terms to
    /// get wrong.
    fn preprocess(&mut self, frame: &Frame) -> Result<()> {
        if frame.data.len() != frame.expected_len() {
            return Err(DetectError::Config(format!(
                "frame {} is {} bytes, expected {} for {}x{} RGB8",
                frame.seq,
                frame.data.len(),
                frame.expected_len(),
                frame.width,
                frame.height
            )));
        }

        let scale =
            (INPUT_SIZE as f32 / frame.width as f32).min(INPUT_SIZE as f32 / frame.height as f32);
        let new_w = ((frame.width as f32 * scale).round() as u32).clamp(1, INPUT_SIZE);
        let new_h = ((frame.height as f32 * scale).round() as u32).clamp(1, INPUT_SIZE);
        self.letterbox = Letterbox { scale, width: new_w, height: new_h };

        let src = ImageRef::new(frame.width, frame.height, &frame.data, PixelType::U8x3)
            .map_err(|e| DetectError::Config(format!("source image: {e}")))?;

        if self.scaled.width() != new_w || self.scaled.height() != new_h {
            self.scaled = Image::new(new_w, new_h, PixelType::U8x3);
        }

        // Bilinear, not Lanczos3: this is a downscale feeding a detector, and
        // the extra taps cost more than they buy.
        self.resizer
            .resize(
                &src,
                &mut self.scaled,
                &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
            )
            .map_err(|e| DetectError::Config(format!("resize: {e}")))?;

        // Explicit tight loop over a preallocated buffer — not chained
        // iterator `collect()`s (MODELS.md §6 rule 4).
        self.tensor.fill(0.0);
        let side = INPUT_SIZE as usize;
        let plane = side * side;
        let scaled = self.scaled.buffer();
        for y in 0..new_h as usize {
            let src_row = y * new_w as usize * 3;
            let dst_row = y * side;
            for x in 0..new_w as usize {
                let s = src_row + x * 3;
                let d = dst_row + x;
                let (r, g, b) = (scaled[s] as f32, scaled[s + 1] as f32, scaled[s + 2] as f32);
                if CHANNEL_ORDER_BGR {
                    self.tensor[d] = b;
                    self.tensor[plane + d] = g;
                    self.tensor[2 * plane + d] = r;
                } else {
                    self.tensor[d] = r;
                    self.tensor[plane + d] = g;
                    self.tensor[2 * plane + d] = b;
                }
            }
        }

        Ok(())
    }
}

fn extract<'a>(outputs: &'a ort::session::SessionOutputs, name: &str) -> Result<&'a [f32]> {
    let value = outputs
        .get(name)
        .ok_or_else(|| DetectError::Config(format!("YuNet produced no output named {name}")))?;
    let (_, data) = value.try_extract_tensor::<f32>().map_err(|e| inference_error("yunet", e))?;
    Ok(data)
}

/// Decode one stride's anchor grid.
///
/// YuNet is anchor-free: each cell predicts a centre offset from its own
/// top-left corner in cell units, plus log-space width and height. Confidence
/// is the geometric mean of the classification and objectness scores, which is
/// what OpenCV's own postprocess computes.
#[allow(clippy::too_many_arguments)]
fn decode_stride(
    stride: u32,
    cls: &[f32],
    obj: &[f32],
    bbox: &[f32],
    kps: &[f32],
    score_threshold: f32,
    lb: Letterbox,
    out: &mut Vec<FaceDetection>,
) {
    let cols = (INPUT_SIZE / stride) as usize;
    let n = cls.len().min(obj.len()).min(bbox.len() / 4).min(kps.len() / 10);
    let stride = stride as f32;
    let inv_scale = if lb.scale > 0.0 { 1.0 / lb.scale } else { 1.0 };

    for i in 0..n {
        let score = (cls[i].clamp(0.0, 1.0) * obj[i].clamp(0.0, 1.0)).sqrt();
        if score < score_threshold {
            continue;
        }

        let (row, col) = ((i / cols) as f32, (i % cols) as f32);
        let b = &bbox[i * 4..i * 4 + 4];
        let cx = (col + b[0]) * stride;
        let cy = (row + b[1]) * stride;
        let w = b[2].exp() * stride;
        let h = b[3].exp() * stride;

        // Straight back to source pixels: a top-left letterbox means no offset
        // term, just the inverse scale.
        let k = &kps[i * 10..i * 10 + 10];
        let point = |j: usize| {
            (((col + k[j * 2]) * stride) * inv_scale, ((row + k[j * 2 + 1]) * stride) * inv_scale)
        };

        out.push(FaceDetection {
            bbox: BBox {
                x: (cx - w * 0.5) * inv_scale,
                y: (cy - h * 0.5) * inv_scale,
                w: w * inv_scale,
                h: h * inv_scale,
            },
            score,
            keypoints: Some(FaceKeypoints {
                right_eye: point(0),
                left_eye: point(1),
                nose: point(2),
                right_mouth: point(3),
                left_mouth: point(4),
            }),
        });
    }
}

/// Faces are a single class, so the shared suppressor is handed a constant
/// class key and behaves class-agnostically.
fn nms(boxes: Vec<FaceDetection>, iou_threshold: f32, top_k: usize) -> Vec<FaceDetection> {
    super::nms(boxes, iou_threshold, top_k, |f| (0, f.bbox, f.score))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, w: f32, h: f32, score: f32) -> FaceDetection {
        FaceDetection { bbox: BBox { x, y, w, h }, score, keypoints: None }
    }

    #[test]
    fn nms_keeps_the_best_of_an_overlapping_cluster() {
        let boxes = vec![
            det(10.0, 10.0, 100.0, 100.0, 0.80),
            det(12.0, 11.0, 98.0, 102.0, 0.95), // same face, higher score
            det(400.0, 300.0, 90.0, 90.0, 0.70), // a genuinely different face
        ];
        let kept = nms(boxes, 0.3, 100);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].score, 0.95, "highest score must survive its cluster");
        assert_eq!(kept[1].bbox.x, 400.0);
    }

    #[test]
    fn nms_respects_top_k_before_suppressing() {
        let boxes: Vec<_> =
            (0..10).map(|i| det(i as f32 * 500.0, 0.0, 50.0, 50.0, i as f32 / 10.0)).collect();
        assert_eq!(nms(boxes, 0.3, 3).len(), 3);
    }

    #[test]
    fn decode_maps_a_hit_back_through_the_letterbox() {
        // One anchor at grid (row 1, col 2), stride 32, on a frame that was
        // downscaled by half: the box must come back in source pixels, doubled.
        let cols = (INPUT_SIZE / 32) as usize;
        let n = cols * cols;
        let mut cls = vec![0.0; n];
        let mut obj = vec![0.0; n];
        let mut bbox = vec![0.0; n * 4];
        let kps = vec![0.0; n * 10];

        let i = cols + 2;
        cls[i] = 1.0;
        obj[i] = 1.0;
        bbox[i * 4..i * 4 + 4].copy_from_slice(&[0.0, 0.0, 0.0, 0.0]);

        let mut out = Vec::new();
        let lb = Letterbox { scale: 0.5, width: 640, height: 360 };
        decode_stride(32, &cls, &obj, &bbox, &kps, 0.5, lb, &mut out);

        assert_eq!(out.len(), 1);
        let d = &out[0];
        assert!((d.score - 1.0).abs() < 1e-6);
        // cx = (2 + 0) * 32 = 64, w = e^0 * 32 = 32 -> x = 48; /0.5 -> 96
        assert!((d.bbox.x - 96.0).abs() < 1e-3, "got {}", d.bbox.x);
        assert!((d.bbox.y - 32.0).abs() < 1e-3, "got {}", d.bbox.y);
        assert!((d.bbox.w - 64.0).abs() < 1e-3, "got {}", d.bbox.w);
    }

    #[test]
    fn decode_drops_everything_below_threshold() {
        let cols = (INPUT_SIZE / 32) as usize;
        let n = cols * cols;
        let mut out = Vec::new();
        decode_stride(
            32,
            &vec![0.4; n],
            &vec![0.4; n],
            &vec![0.0; n * 4],
            &vec![0.0; n * 10],
            0.5,
            Letterbox { scale: 1.0, width: 640, height: 640 },
            &mut out,
        );
        assert!(out.is_empty(), "score 0.4 must not survive a 0.5 threshold");
    }
}
