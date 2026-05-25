//! Prohibited-object detection with YOLOX-Nano (build step 7).
//!
//! **Licence decision: YOLOX-Nano, Apache 2.0.** Not YOLO26n, which is
//! AGPL-3.0 and therefore cannot ship inside a distributed binary without
//! open-sourcing derivatives or buying an enterprise licence. YOLO26n stays on
//! disk in `deepscreen-detect/models/` as benchmark evidence only, and is
//! deliberately excluded from the app bundle.
//!
//! The other Apache-2.0 candidate, RF-DETR Nano, is NMS-free but measured
//! ~180 ms on CPU at 320x320 with INT8 giving no improvement. YOLOX-Nano is
//! 0.91M parameters and a pure conv graph, so INT8 and DirectML both have a
//! real path later. The deployment target is a low-end candidate laptop.
//!
//! # Verified interface
//!
//! `detect-cli inspect models/yolox_nano.onnx`:
//!
//! ```text
//! images   Float32  [1, 3, 416, 416]
//! output   Float32  [1, 3549, 85]
//! ```
//!
//! 3549 = 52^2 + 26^2 + 13^2, the three stride levels concatenated in that
//! order, and 85 = 4 box + 1 objectness + 80 COCO classes.
//!
//! **The released export does not decode grids in-graph.** YOLOX supports
//! both and the demos differ; this one leaves the decode to
//! `demo_postprocess`, which is why it happens here.
//!
//! # The two things that would silently degrade this
//!
//! 1. **BGR, not RGB.** YOLOX's demo feeds `cv2.imread` output straight in, so
//!    the model was trained on BGR. Same as YuNet, and the opposite of the
//!    pose and gaze models in this same pipeline.
//! 2. **No normalization at all.** YOLOX removed mean/std subtraction; the
//!    input is raw 0-255 as float. Dividing by 255 "for consistency" with the
//!    other models here would break it quietly.

use std::path::Path;

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use ort::session::Session;

use crate::config::Config;
use crate::error::{DetectError, Result};
use crate::types::{BBox, Frame, ObjectDetection};

use super::{build_session, inference_error, nchw_input, StageTimings};

pub const INPUT_SIZE: u32 = 416;

/// Strides, in the order the graph concatenates them.
const STRIDES: [u32; 3] = [8, 16, 32];

/// YOLOX pads with 114, not black.
const PAD_VALUE: u8 = 114;

/// See the module comment — this is not an oversight.
const CHANNEL_ORDER_BGR: bool = true;

const NUM_CLASSES: usize = 80;
/// 4 box + 1 objectness + 80 classes.
const STRIDE_ELEMS: usize = 5 + NUM_CLASSES;

/// COCO class names in the order YOLOX emits them.
pub const COCO_CLASSES: [&str; NUM_CLASSES] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

pub struct YoloxNano {
    session: Session,
    resizer: Resizer,
    scaled: Image<'static>,
    tensor: Vec<f32>,
    score_threshold: f32,
    nms_threshold: f32,
    /// Class ids kept after decode, resolved once from the config allowlist.
    allowed: Vec<u32>,
    letterbox_scale: f32,
}

impl YoloxNano {
    pub fn load(path: impl AsRef<Path>, cfg: &Config) -> Result<Self> {
        // Big graph, low rate: this one gets the larger intra-op budget
        // (MODELS.md §6 rule 2).
        let session = build_session(path, &cfg.runtime, true)?;
        let side = INPUT_SIZE as usize;

        let allowed = resolve_allowlist(&cfg.thresholds.objects.allowlist);
        if allowed.is_empty() {
            return Err(DetectError::Config(
                "objects.allowlist matched no COCO class — check the spelling".into(),
            ));
        }

        let mut model = Self {
            session,
            resizer: Resizer::new(),
            scaled: Image::new(INPUT_SIZE, INPUT_SIZE, PixelType::U8x3),
            tensor: vec![0.0; 3 * side * side],
            score_threshold: cfg.thresholds.objects.min_score as f32,
            nms_threshold: cfg.thresholds.objects.nms_threshold as f32,
            allowed,
            letterbox_scale: 1.0,
        };
        model.warm_up(cfg.runtime.warmup_iters)?;
        Ok(model)
    }

    fn warm_up(&mut self, iters: u32) -> Result<()> {
        for _ in 0..iters {
            let input = nchw_input(&self.tensor, INPUT_SIZE, "yolox/warmup")?;
            self.session
                .run(ort::inputs!["images" => input])
                .map_err(|e| inference_error("yolox/warmup", e))?;
        }
        Ok(())
    }

    pub fn detect(&mut self, frame: &Frame) -> Result<(Vec<ObjectDetection>, StageTimings)> {
        let t0 = std::time::Instant::now();
        self.preprocess(frame)?;
        let preprocess_us = t0.elapsed().as_micros() as u32;

        let t1 = std::time::Instant::now();
        let input = nchw_input(&self.tensor, INPUT_SIZE, "yolox")?;
        let outputs = self
            .session
            .run(ort::inputs!["images" => input])
            .map_err(|e| inference_error("yolox", e))?;
        let inference_us = t1.elapsed().as_micros() as u32;

        let t2 = std::time::Instant::now();
        let value = outputs
            .get("output")
            .ok_or_else(|| DetectError::Config("YOLOX produced no `output` tensor".into()))?;
        let (_, data) = value.try_extract_tensor::<f32>().map_err(|e| inference_error("yolox", e))?;

        let candidates =
            decode(data, self.score_threshold, self.letterbox_scale, &self.allowed);
        let objects = super::nms(candidates, self.nms_threshold, 100, |o| {
            (o.class_id, o.bbox, o.score)
        });

        Ok((
            objects,
            StageTimings {
                preprocess_us,
                inference_us,
                postprocess_us: t2.elapsed().as_micros() as u32,
            },
        ))
    }

    /// Letterbox top-left with pad 114, matching YOLOX's `preproc`.
    ///
    /// Top-left placement is also the convention `face.rs` uses, so undoing it
    /// is a single divide by the scale with no offset term.
    fn preprocess(&mut self, frame: &Frame) -> Result<()> {
        if frame.data.len() != frame.expected_len() {
            return Err(DetectError::Config(format!(
                "frame {} is {} bytes, expected {}",
                frame.seq,
                frame.data.len(),
                frame.expected_len()
            )));
        }

        let scale =
            (INPUT_SIZE as f32 / frame.width as f32).min(INPUT_SIZE as f32 / frame.height as f32);
        let new_w = ((frame.width as f32 * scale) as u32).clamp(1, INPUT_SIZE);
        let new_h = ((frame.height as f32 * scale) as u32).clamp(1, INPUT_SIZE);
        self.letterbox_scale = scale;

        let src = ImageRef::new(frame.width, frame.height, &frame.data, PixelType::U8x3)
            .map_err(|e| DetectError::Config(format!("source image: {e}")))?;

        if self.scaled.width() != new_w || self.scaled.height() != new_h {
            self.scaled = Image::new(new_w, new_h, PixelType::U8x3);
        }
        self.resizer
            .resize(
                &src,
                &mut self.scaled,
                &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
            )
            .map_err(|e| DetectError::Config(format!("object resize: {e}")))?;

        // Fill with the pad value first, then write the scaled image into the
        // top-left corner. Padding with black instead would put a hard edge
        // where the model expects neutral grey.
        self.tensor.fill(PAD_VALUE as f32);
        let side = INPUT_SIZE as usize;
        let plane = side * side;
        let px = self.scaled.buffer();
        for y in 0..new_h as usize {
            let src_row = y * new_w as usize * 3;
            let dst_row = y * side;
            for x in 0..new_w as usize {
                let s = src_row + x * 3;
                let d = dst_row + x;
                let (r, g, b) = (px[s] as f32, px[s + 1] as f32, px[s + 2] as f32);
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

/// Map allowlist names to COCO class ids, ignoring case and unknown names.
fn resolve_allowlist(names: &[String]) -> Vec<u32> {
    names
        .iter()
        .filter_map(|name| {
            let wanted = name.trim().to_ascii_lowercase();
            COCO_CLASSES.iter().position(|c| *c == wanted).map(|i| i as u32)
        })
        .collect()
}

/// Grid decode, per YOLOX's `demo_postprocess`.
///
/// For anchor `i` at grid `(gx, gy)` with stride `s`:
/// `cx = (raw_cx + gx) * s`, `cy = (raw_cy + gy) * s`,
/// `w = exp(raw_w) * s`, `h = exp(raw_h) * s`.
///
/// Score is objectness times class probability, and the allowlist is applied
/// here — trivial across 3,549 anchors, unlike the old EfficientDet path's
/// 19,206 x 90.
fn decode(
    data: &[f32],
    score_threshold: f32,
    letterbox_scale: f32,
    allowed: &[u32],
) -> Vec<ObjectDetection> {
    let mut out = Vec::new();
    let inv_scale = if letterbox_scale > 0.0 { 1.0 / letterbox_scale } else { 1.0 };

    let mut offset = 0usize;
    for stride in STRIDES {
        let cells = (INPUT_SIZE / stride) as usize;
        let stride_f = stride as f32;

        for i in 0..cells * cells {
            let base = (offset + i) * STRIDE_ELEMS;
            if base + STRIDE_ELEMS > data.len() {
                return out;
            }
            let objectness = data[base + 4];
            if objectness < score_threshold {
                continue; // cheap reject before scanning 80 classes
            }

            // Grid is row-major: meshgrid(arange(w), arange(h)) stacked.
            let (gy, gx) = ((i / cells) as f32, (i % cells) as f32);

            let mut best_class = 0usize;
            let mut best_prob = 0.0f32;
            for c in 0..NUM_CLASSES {
                let p = data[base + 5 + c];
                if p > best_prob {
                    best_prob = p;
                    best_class = c;
                }
            }

            let score = objectness * best_prob;
            if score < score_threshold {
                continue;
            }
            let class_id = best_class as u32;
            if !allowed.contains(&class_id) {
                continue;
            }

            let cx = (data[base] + gx) * stride_f;
            let cy = (data[base + 1] + gy) * stride_f;
            let w = data[base + 2].exp() * stride_f;
            let h = data[base + 3].exp() * stride_f;

            out.push(ObjectDetection {
                class_id,
                label: COCO_CLASSES[best_class].to_string(),
                score,
                bbox: BBox {
                    x: (cx - w * 0.5) * inv_scale,
                    y: (cy - h * 0.5) * inv_scale,
                    w: w * inv_scale,
                    h: h * inv_scale,
                },
            });
        }
        offset += cells * cells;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank() -> Vec<f32> {
        vec![0.0; 3549 * STRIDE_ELEMS]
    }

    #[test]
    fn the_anchor_count_matches_the_declared_output() {
        let total: usize =
            STRIDES.iter().map(|s| ((INPUT_SIZE / s) as usize).pow(2)).sum();
        assert_eq!(total, 3549, "anchor layout disagrees with the inspected shape");
    }

    #[test]
    fn allowlist_resolves_to_the_right_coco_ids() {
        let ids = resolve_allowlist(&["cell phone".into(), "book".into()]);
        assert_eq!(ids, vec![67, 73]);
        assert_eq!(COCO_CLASSES[67], "cell phone");
        assert_eq!(COCO_CLASSES[73], "book");
    }

    #[test]
    fn unknown_allowlist_names_are_dropped_not_guessed() {
        let ids = resolve_allowlist(&["cellphone".into(), "book".into()]);
        assert_eq!(ids, vec![73], "a near-miss name must not silently match");
    }

    #[test]
    fn a_planted_anchor_decodes_to_the_right_box_and_class() {
        let mut data = blank();
        // Stride 8 grid is 52x52; put a detection at cell (row 3, col 2).
        let cells = 52usize;
        let i = 3 * cells + 2;
        let base = i * STRIDE_ELEMS;
        data[base] = 0.0; // raw cx offset
        data[base + 1] = 0.0; // raw cy offset
        data[base + 2] = 0.0; // exp(0) * 8 = 8 wide
        data[base + 3] = 0.0;
        data[base + 4] = 0.9; // objectness
        data[base + 5 + 67] = 0.8; // cell phone

        let out = decode(&data, 0.3, 1.0, &[67]);
        assert_eq!(out.len(), 1);
        let d = &out[0];
        assert_eq!(d.label, "cell phone");
        assert!((d.score - 0.72).abs() < 1e-5, "score should be obj * class");
        // cx = (0 + 2) * 8 = 16, w = 8 -> x = 12
        assert!((d.bbox.x - 12.0).abs() < 1e-3, "x was {}", d.bbox.x);
        assert!((d.bbox.y - 20.0).abs() < 1e-3, "y was {}", d.bbox.y);
        assert!((d.bbox.w - 8.0).abs() < 1e-3);
    }

    #[test]
    fn the_letterbox_is_undone_by_a_single_divide() {
        let mut data = blank();
        let base = (3 * 52 + 2) * STRIDE_ELEMS;
        data[base + 4] = 0.9;
        data[base + 5 + 67] = 0.8;

        let out = decode(&data, 0.3, 0.5, &[67]);
        // Same anchor at half scale must land at twice the coordinates.
        assert!((out[0].bbox.x - 24.0).abs() < 1e-3, "x was {}", out[0].bbox.x);
    }

    #[test]
    fn classes_outside_the_allowlist_are_dropped() {
        let mut data = blank();
        let base = (3 * 52 + 2) * STRIDE_ELEMS;
        data[base + 4] = 0.99;
        data[base + 5] = 0.99; // class 0 = person, deliberately not allowlisted

        assert!(decode(&data, 0.3, 1.0, &[67, 73]).is_empty());
    }

    #[test]
    fn a_blank_output_yields_no_detections() {
        // The number that matters: false positives are what make proctoring
        // unusable.
        assert!(decode(&blank(), 0.3, 1.0, &[67, 73]).is_empty());
    }

    #[test]
    fn class_wise_nms_keeps_an_overlapping_phone_and_book() {
        let obj = |cls: u32, label: &str, score: f32| ObjectDetection {
            class_id: cls,
            label: label.into(),
            score,
            bbox: BBox { x: 10.0, y: 10.0, w: 100.0, h: 100.0 },
        };
        // Same box, different classes: suppressing across classes would erase
        // the book because the phone scored higher.
        let kept = super::super::nms(
            vec![obj(67, "cell phone", 0.9), obj(73, "book", 0.7)],
            0.45,
            100,
            |o| (o.class_id, o.bbox, o.score),
        );
        assert_eq!(kept.len(), 2);
    }
}
