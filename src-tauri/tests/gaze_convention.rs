//! Does gaze agree with head pose about which way is positive?
//!
//! `eye_yaw = gaze_yaw - head_yaw` is only meaningful if both models call the
//! same physical direction positive. They come from different repositories,
//! trained on different datasets, and nothing guarantees it. If the handedness
//! were opposite, eye-in-head would report roughly *double* the head rotation
//! instead of the eye's own offset — a signal that looks alive, tracks head
//! movement convincingly, and means nothing.
//!
//! Two independent lines of evidence:
//!
//! 1. **From the reference implementations' own drawing code**, which is
//!    decisive and needs no fixture. `yakhyo/gaze-estimation`'s `draw_gaze`
//!    projects `dx = -length * sin(yaw) * cos(pitch)`, `dy = -length *
//!    sin(pitch)`. `yakhyo/head-pose-estimation`'s `draw_axis` first negates
//!    yaw, then draws the face-forward axis at `x3 = size * sin(-yaw)`,
//!    `y3 = -size * cos(yaw) * sin(pitch)`. Both therefore map **+yaw to the
//!    left of screen and +pitch upward**. The conventions agree, so
//!    `gaze - head` is a valid subtraction.
//!
//! 2. **Empirically, by mirroring** — yaw must negate for both models, so the
//!    two deltas must share a sign. This only decides anything on a fixture
//!    with real rotation in it; on a near-square head both deltas are noise,
//!    and the test says so rather than deciding on it.
//!
//! Skips cleanly when models or fixtures are absent.

use std::sync::Arc;

use vigilo::config::Config;
use vigilo::models::face::YuNet;
use vigilo::models::gaze::{GazeNet, GazeOutcome};
use vigilo::models::pose::HeadPoseNet;
use vigilo::types::Frame;

const MODEL_DIR: &str = "models";
const FIXTURE_DIR: &str = "samples/faces";

fn fixtures() -> Vec<Frame> {
    let Ok(dir) = std::fs::read_dir(FIXTURE_DIR) else { return Vec::new() };
    let mut paths: Vec<_> = dir
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "jpg" || e == "png"))
        .collect();
    paths.sort();

    paths
        .iter()
        .filter_map(|p| {
            let img = image::open(p).ok()?.into_rgb8();
            let (w, h) = (img.width(), img.height());
            Some(Frame {
                data: Arc::from(img.into_raw().as_slice()),
                width: w,
                height: h,
                seq: 1,
                captured_at: std::time::Instant::now(),
            })
        })
        .collect()
}

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

struct Models {
    face: YuNet,
    pose: HeadPoseNet,
    gaze: GazeNet,
}

fn models(cfg: &Config) -> Option<Models> {
    let dir = std::path::Path::new(MODEL_DIR);
    let (f, p, g) = (
        dir.join("face_detection_yunet_2023mar.onnx"),
        dir.join("headpose_mobilenetv3_small.onnx"),
        dir.join("mobileone_s0_gaze.onnx"),
    );
    if !f.exists() || !p.exists() || !g.exists() {
        return None;
    }
    Some(Models {
        face: YuNet::load(&f, cfg).ok()?,
        pose: HeadPoseNet::load(&p, cfg).ok()?,
        gaze: GazeNet::load(&g, cfg).ok()?,
    })
}

/// Head yaw and gaze yaw, in degrees, for one frame.
fn measure(m: &mut Models, frame: &Frame) -> Option<(f32, f32)> {
    let faces = m.face.detect(frame).ok()?;
    let primary = faces.first()?;
    let (pose, _) = m.pose.estimate(frame, primary).ok()?;
    // A gated frame yields no measurement at all, which is the point: pairing
    // a head angle with a held-over gaze angle would compare two instants.
    let GazeOutcome::Produced { gaze, .. } = m.gaze.estimate(frame, primary, Some(pose)).ok()?
    else {
        return None;
    };
    Some((pose.yaw_deg, gaze.yaw_rad.to_degrees()))
}

#[test]
fn gaze_and_head_pose_share_a_handedness() {
    let cfg = Config::default();
    let frames = fixtures();
    if frames.is_empty() {
        eprintln!("skipping: no fixtures in {FIXTURE_DIR}");
        return;
    }
    let Some(mut m) = models(&cfg) else {
        eprintln!("skipping: models not present in {MODEL_DIR}");
        return;
    };

    // Near-square heads produce deltas indistinguishable from noise, and a
    // handedness question decided by noise is worse than one left open. Judge
    // on whichever fixture carries the most rotation.
    let mut samples: Vec<(f32, f32)> = Vec::new();
    for frame in &frames {
        let flipped = mirror(frame);
        let Some((head_a, gaze_a)) = measure(&mut m, frame) else { continue };
        let Some((head_b, gaze_b)) = measure(&mut m, &flipped) else { continue };
        samples.push((head_a - head_b, gaze_a - gaze_b));
    }

    // "No fixtures" is a clean skip. "Fixtures that contain no face" is not —
    // that would let this report success without measuring anything, which is
    // worse than failing.
    assert!(
        !samples.is_empty(),
        "{} fixture(s) in {FIXTURE_DIR} but no face detected in any of them.          Recapture with someone in shot:  detect-cli live --source camera:0          --max-frames 60 --save-every 5 --save-dir {FIXTURE_DIR}",
        frames.len()
    );

    samples.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap());
    for (h, g) in samples.iter().take(5) {
        eprintln!("head delta {h:+8.2}   gaze delta {g:+8.2}");
    }

    let (head_delta, gaze_delta) = samples[0];

    // Below this, the fixture simply does not carry a direction to compare.
    const DECISIVE_DEG: f32 = 8.0;
    if head_delta.abs() < DECISIVE_DEG || gaze_delta.abs() < 2.0 {
        eprintln!(
            "inconclusive from fixtures: strongest carries only {:.1} deg head / {:.1} deg gaze.              Handedness is instead established from the two reference implementations              (see the module comment); recapture with a deliberate ~20 deg head turn to              confirm empirically.",
            head_delta.abs(),
            gaze_delta.abs()
        );
        return;
    }

    assert!(
        head_delta.signum() == gaze_delta.signum(),
        "gaze and head pose disagree on which direction is positive:          head delta {head_delta:+.2}, gaze delta {gaze_delta:+.2}.          eye_yaw = gaze_yaw - head_yaw is meaningless until one is negated."
    );
}

#[test]
fn mirroring_negates_gaze_yaw() {
    let cfg = Config::default();
    let frames = fixtures();
    if frames.is_empty() {
        return;
    }
    let Some(mut m) = models(&cfg) else { return };

    let frame = &frames[0];
    let flipped = mirror(frame);
    let (Some((_, gaze_a)), Some((_, gaze_b))) =
        (measure(&mut m, frame), measure(&mut m, &flipped))
    else {
        return;
    };

    assert!(
        (gaze_a + gaze_b).abs() < 15.0,
        "gaze yaw should negate under mirroring: {gaze_a:+.1} vs {gaze_b:+.1}"
    );
}
