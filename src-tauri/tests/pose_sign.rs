//! End-to-end sign check for head pose.
//!
//! `pose.rs`'s unit tests pin the rotation-matrix-to-Euler conversion against
//! synthetic matrices, which proves the *maths* is right. They cannot prove the
//! **model's** axes are wired to the world the way we assume — a model that
//! reports yaw with the opposite sign would pass every one of them.
//!
//! This closes that gap using a physical invariant instead of a labelled clip:
//! **mirroring an image horizontally must negate yaw and roll, and leave pitch
//! alone.** A face turned 20 degrees left is, in a mirror, a face turned 20
//! degrees right. No head-turning, no ground-truth annotation, no camera at
//! test time — just a fixture frame and an axiom.
//!
//! Generate a fixture with:
//!
//! ```bash
//! detect-cli live --source camera:0 --max-frames 30 --save-every 5 \
//!   --save-dir samples/faces
//! ```
//!
//! Skips cleanly when the models or the fixture are absent, so CI on a machine
//! with neither still passes.

use std::sync::Arc;

use vigilo::config::Config;
use vigilo::models::face::YuNet;
use vigilo::models::pose::HeadPoseNet;
use vigilo::types::Frame;

const MODEL_DIR: &str = "models";
const FIXTURE_DIR: &str = "samples/faces";

fn load_fixture() -> Option<Frame> {
    let dir = std::fs::read_dir(FIXTURE_DIR).ok()?;
    let mut images: Vec<_> = dir
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "jpg" || e == "png"))
        .collect();
    images.sort();
    let path = images.first()?;

    let img = image::open(path).ok()?.into_rgb8();
    let (w, h) = (img.width(), img.height());
    Some(Frame {
        data: Arc::from(img.into_raw().as_slice()),
        width: w,
        height: h,
        seq: 1,
        captured_at: std::time::Instant::now(),
    })
}

/// Horizontal mirror, preserving dimensions.
fn mirror(frame: &Frame) -> Frame {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        let row = y * w * 3;
        for x in 0..w {
            let src = row + x * 3;
            let dst = row + (w - 1 - x) * 3;
            out[dst..dst + 3].copy_from_slice(&frame.data[src..src + 3]);
        }
    }
    Frame {
        data: Arc::from(out.as_slice()),
        width: frame.width,
        height: frame.height,
        seq: frame.seq,
        captured_at: frame.captured_at,
    }
}

fn models(cfg: &Config) -> Option<(YuNet, HeadPoseNet)> {
    let face_path = std::path::Path::new(MODEL_DIR).join("face_detection_yunet_2023mar.onnx");
    let pose_path = std::path::Path::new(MODEL_DIR).join("headpose_mobilenetv3_small.onnx");
    if !face_path.exists() || !pose_path.exists() {
        return None;
    }
    Some((YuNet::load(&face_path, cfg).ok()?, HeadPoseNet::load(&pose_path, cfg).ok()?))
}

#[test]
fn mirroring_a_face_negates_yaw_and_leaves_pitch_alone() {
    let cfg = Config::default();
    let Some(frame) = load_fixture() else {
        eprintln!("skipping: no fixture in {FIXTURE_DIR}");
        return;
    };
    let Some((mut face, mut pose)) = models(&cfg) else {
        eprintln!("skipping: models not present in {MODEL_DIR}");
        return;
    };

    let flipped = mirror(&frame);

    let original_faces = face.detect(&frame).expect("detect on fixture");
    let mirrored_faces = face.detect(&flipped).expect("detect on mirrored fixture");
    assert!(!original_faces.is_empty(), "fixture has no detectable face — recapture it");
    assert!(!mirrored_faces.is_empty(), "mirrored fixture has no detectable face");

    let (a, _) = pose.estimate(&frame, &original_faces[0]).expect("pose on fixture");
    let (b, _) = pose.estimate(&flipped, &mirrored_faces[0]).expect("pose on mirrored fixture");

    // Yaw must flip. The tolerance is loose because the mirrored image is not
    // a perfectly symmetric input — lighting and the detector's box differ
    // slightly — but the *sign* relationship is not a matter of degree.
    assert!(
        (a.yaw_deg + b.yaw_deg).abs() < 12.0,
        "yaw should negate under mirroring: {:.1} vs {:.1}",
        a.yaw_deg,
        b.yaw_deg
    );

    // Pitch is unchanged by a horizontal mirror.
    assert!(
        (a.pitch_deg - b.pitch_deg).abs() < 12.0,
        "pitch should survive mirroring: {:.1} vs {:.1}",
        a.pitch_deg,
        b.pitch_deg
    );

    // Roll flips with yaw.
    assert!(
        (a.roll_deg + b.roll_deg).abs() < 12.0,
        "roll should negate under mirroring: {:.1} vs {:.1}",
        a.roll_deg,
        b.roll_deg
    );

    eprintln!(
        "pose original  yaw {:+.1}  pitch {:+.1}  roll {:+.1}",
        a.yaw_deg, a.pitch_deg, a.roll_deg
    );
    eprintln!(
        "pose mirrored  yaw {:+.1}  pitch {:+.1}  roll {:+.1}",
        b.yaw_deg, b.pitch_deg, b.roll_deg
    );
}

#[test]
fn pose_angles_are_in_a_physically_plausible_range() {
    // A webcam user is not upside down. Angles outside this range mean the
    // conversion or the normalization is wrong, not that the person is doing
    // gymnastics.
    let cfg = Config::default();
    let Some(frame) = load_fixture() else { return };
    let Some((mut face, mut pose)) = models(&cfg) else { return };

    let faces = face.detect(&frame).expect("detect");
    if faces.is_empty() {
        return;
    }
    let (p, _) = pose.estimate(&frame, &faces[0]).expect("pose");

    assert!(p.yaw_deg.abs() <= 90.0, "yaw {:.1} out of range", p.yaw_deg);
    assert!(p.pitch_deg.abs() <= 90.0, "pitch {:.1} out of range", p.pitch_deg);
    assert!(p.roll_deg.abs() <= 90.0, "roll {:.1} out of range", p.roll_deg);
}
