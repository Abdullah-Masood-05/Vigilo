//! Does the YOLOX decode actually produce correct boxes and classes?
//!
//! `rust_context.md` §5 is explicit that only YuNet had been validated for
//! correctness and the rest were merely "loads and runs fast". This closes
//! that for objects, using YOLOX's **own canonical test image** — the classic
//! dog/bicycle/truck photo from its repository, whose expected output is
//! documented by every YOLO demo ever written.
//!
//! That makes it ground truth without needing an annotation file: if the grid
//! decode, the stride ordering, the BGR channel order, the pad value or the
//! letterbox inverse were wrong, this would not find those three classes at
//! sensible positions. A wrong channel order in particular *degrades* rather
//! than breaks detection, which is exactly the failure this catches.
//!
//! Skips cleanly when the model or the image is absent.

use std::sync::Arc;

use vigilo::config::Config;
use vigilo::models::objects::YoloxNano;
use vigilo::types::Frame;

// Models live one level above the crate root (`deepscreen-viewer/models`),
// matching the layout the app's own resource resolver and `tauri.conf.json`
// already use — not duplicated into `src-tauri/models` just for tests.
const MODEL: &str = "../models/yolox_nano.onnx";
const IMAGE: &str = "samples/objects/dog.jpg";

fn load_image(path: &str) -> Option<Frame> {
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

/// A config that keeps every class, so the decode can be judged on what the
/// model actually saw rather than on what the proctoring allowlist permits.
fn wide_open() -> Config {
    let mut cfg = Config::default();
    cfg.thresholds.objects.allowlist = vigilo::models::objects::COCO_CLASSES
        .iter()
        .map(|s| s.to_string())
        .collect();
    cfg.thresholds.objects.min_score = 0.25;
    cfg
}

#[test]
fn yolox_finds_the_expected_objects_in_its_own_test_image() {
    let Some(frame) = load_image(IMAGE) else {
        eprintln!("skipping: {IMAGE} not present");
        return;
    };
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("skipping: {MODEL} not present");
        return;
    }

    let cfg = wide_open();
    let mut model = YoloxNano::load(MODEL, &cfg).expect("load yolox");
    let (objects, timings) = model.detect(&frame).expect("detect");

    for o in &objects {
        eprintln!(
            "{:<12} {:.2}  box ({:.0}, {:.0}) {:.0}x{:.0}",
            o.label, o.score, o.bbox.x, o.bbox.y, o.bbox.w, o.bbox.h
        );
    }
    eprintln!("timings: {timings:?}");

    let found: Vec<&str> = objects.iter().map(|o| o.label.as_str()).collect();
    for expected in ["dog", "bicycle", "truck"] {
        assert!(
            found.contains(&expected),
            "expected a {expected} in YOLOX's own test image, found {found:?}. \
             A wrong channel order, pad value, stride order or grid decode all \
             show up exactly like this."
        );
    }

    // Boxes must land inside the source image, not in letterbox space. If the
    // inverse scale were missing, everything would be squeezed into the
    // top-left 416x416 corner of a much larger photo.
    for o in &objects {
        assert!(
            o.bbox.x >= -5.0
                && o.bbox.y >= -5.0
                && o.bbox.x + o.bbox.w <= frame.width as f32 + 5.0
                && o.bbox.y + o.bbox.h <= frame.height as f32 + 5.0,
            "{} box {:?} escapes the {}x{} frame — letterbox inverse is wrong",
            o.label,
            o.bbox,
            frame.width,
            frame.height
        );
    }

    // The dog is the large foreground subject; a decode that produced
    // plausible labels at nonsense scales would still pass the checks above.
    let dog = objects.iter().find(|o| o.label == "dog").unwrap();
    let frame_area = (frame.width * frame.height) as f32;
    let ratio = dog.bbox.area() / frame_area;
    assert!(
        (0.02..0.6).contains(&ratio),
        "dog occupies {:.1}% of the frame, which is not a plausible size",
        ratio * 100.0
    );
}

#[test]
fn the_proctoring_allowlist_filters_that_same_image_to_nothing() {
    // The shipped allowlist is cell phone + book. The test image contains a
    // dog, a bicycle and a truck, so a correctly-filtered run must return
    // nothing at all — which is also the false-positive property that matters
    // most in proctoring.
    let Some(frame) = load_image(IMAGE) else { return };
    if !std::path::Path::new(MODEL).exists() {
        return;
    }

    let cfg = Config::default(); // cell phone + book
    let mut model = YoloxNano::load(MODEL, &cfg).expect("load yolox");
    let (objects, _) = model.detect(&frame).expect("detect");

    assert!(
        objects.is_empty(),
        "allowlist should have filtered everything, got {:?}",
        objects.iter().map(|o| &o.label).collect::<Vec<_>>()
    );
}
