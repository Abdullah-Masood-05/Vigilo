//! Preview encoding and the loopback MJPEG server.
//!
//! # Why an HTTP stream rather than IPC
//!
//! MODELS.md §12 is explicit that frames must not cross the Tauri IPC
//! boundary. The old app base64-encoded a JPEG and invoked a command every
//! 300 ms — fine at 3.3 Hz, catastrophic at frame rate, because base64 is ~33%
//! size inflation on top of an encode and a decode, on the UI thread, per
//! frame.
//!
//! Serving `multipart/x-mixed-replace` over loopback instead means the browser
//! decodes and paints frames with **zero JavaScript per frame**. The kernel
//! loopback path costs ~0.1 ms. Nothing in the render loop is ours.
//!
//! # Why this thread never touches the detect thread
//!
//! It reads the same `ArcSwap` the detect worker publishes to, with
//! drop-oldest semantics: if encoding falls behind, frames are skipped, never
//! queued. Encoding is 2–4 ms at 640×360 and must not be charged to the
//! detection budget — if `detect p50` in the viewer were materially worse than
//! `detect-cli bench`, this thread would be the reason.

use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use deepscreen_detect::types::Signals;
use deepscreen_detect::Detector;
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

/// Preview is capped regardless of detection rate. Eyeballing does not need
/// more than this, and it halves encode cost.
const PREVIEW_FPS: f64 = 15.0;

/// Detection still runs at full source resolution — only the preview shrinks.
/// This is the single biggest win in the whole preview path: 4× fewer pixels
/// to encode than 1280×720.
const PREVIEW_WIDTH: u32 = 640;

/// Above ~70 you are spending milliseconds on artefacts nobody can see in a
/// test window.
const JPEG_QUALITY: u8 = 70;

/// Whether the preview is horizontally flipped before it reaches the screen.
///
/// **It is not**, and this constant is here so that claim is checkable in one
/// place instead of inferred from the absence of code. Nothing in [`encode`]
/// flips, the capture path passes no `hflip` to ffmpeg, and the stylesheet
/// applies no `scaleX(-1)`. What the window shows is what the camera sent.
///
/// This matters far more than it looks. Direction labels describe the subject's
/// own left and right; a mirrored preview would show them moving the opposite
/// way, and "the label is backwards" and "the picture is flipped" are
/// indistinguishable symptoms unless one of them is stated outright. So it is
/// reported in the HUD.
///
/// Flipping the preview later means setting this to `true` **and** nothing
/// else: the labels are about the person, not the picture, so they do not
/// change.
pub const MIRRORED: bool = false;

/// A JPEG and the signals for **that** JPEG.
///
/// Carrying both together is what bounds box/pixel skew at one preview frame,
/// normally zero, with no sequence matching in JavaScript.
pub struct PreviewItem {
    pub jpeg: Vec<u8>,
    pub signals: Signals,
    pub seq: u64,
    pub width: u32,
    pub height: u32,
}

#[derive(Default)]
pub struct PreviewStats {
    /// Rolling mean of encode time, in microseconds.
    pub encode_p50_us: AtomicU64,
    pub encoded: AtomicU64,
    pub dropped: AtomicU64,
}

pub type PreviewSlot = Arc<ArcSwap<Option<Arc<PreviewItem>>>>;

/// Pull the newest `Detected`, downscale, encode, publish. Drop-oldest.
pub fn preview_loop(detector: Arc<Detector>, slot: PreviewSlot, stats: Arc<PreviewStats>) {
    let period = Duration::from_secs_f64(1.0 / PREVIEW_FPS);
    let mut resizer = Resizer::new();
    let mut scaled: Option<Image<'static>> = None;
    let mut last_seq = u64::MAX;
    let mut samples: Vec<u64> = Vec::with_capacity(128);

    loop {
        let tick = Instant::now();

        if let Some(detected) = detector.latest() {
            if detected.frame.seq != last_seq {
                last_seq = detected.frame.seq;
                let t = Instant::now();

                match encode(&detected, &mut resizer, &mut scaled) {
                    Ok(item) => {
                        let us = t.elapsed().as_micros() as u64;
                        samples.push(us);
                        if samples.len() > 128 {
                            samples.remove(0);
                        }
                        let mut sorted = samples.clone();
                        sorted.sort_unstable();
                        stats
                            .encode_p50_us
                            .store(sorted[sorted.len() / 2], Ordering::Relaxed);
                        stats.encoded.fetch_add(1, Ordering::Relaxed);
                        slot.store(Arc::new(Some(Arc::new(item))));
                    }
                    Err(e) => {
                        stats.dropped.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(error = %e, "preview encode failed");
                    }
                }
            }
        }

        // The detector owns the lifetime; when it stops, so does the preview.
        if !detector.is_running() && detector.latest().is_none() {
            break;
        }

        if let Some(rest) = period.checked_sub(tick.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    tracing::debug!("preview thread finished");
}

fn encode(
    detected: &deepscreen_detect::Detected,
    resizer: &mut Resizer,
    scaled: &mut Option<Image<'static>>,
) -> Result<PreviewItem, String> {
    let frame = &detected.frame;
    let (sw, sh) = (frame.width, frame.height);
    if sw == 0 || sh == 0 {
        return Err("empty frame".into());
    }

    // Preserve aspect; keep both dimensions even, which JPEG chroma
    // subsampling prefers.
    let target_w = PREVIEW_WIDTH.min(sw);
    let target_h = (((sh as f32 * target_w as f32 / sw as f32).round() as u32).max(2)) & !1;

    let src = ImageRef::new(sw, sh, &frame.data, PixelType::U8x3).map_err(|e| e.to_string())?;
    let dst = match scaled {
        Some(img) if img.width() == target_w && img.height() == target_h => img,
        slot => {
            *slot = Some(Image::new(target_w, target_h, PixelType::U8x3));
            slot.as_mut().unwrap()
        }
    };

    resizer
        .resize(&src, dst, &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)))
        .map_err(|e| e.to_string())?;

    let mut jpeg = Vec::with_capacity(48 * 1024);
    let mut encoder =
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, JPEG_QUALITY);
    encoder
        .encode(dst.buffer(), target_w, target_h, image::ExtendedColorType::Rgb8)
        .map_err(|e| e.to_string())?;

    Ok(PreviewItem {
        jpeg,
        signals: detected.signals.clone(),
        seq: frame.seq,
        // Source dimensions, not preview dimensions: `Signals` coordinates are
        // in source pixels, and the SVG viewBox has to match them.
        width: sw,
        height: sh,
    })
}

// ---------------------------------------------------------------------------
// http
// ---------------------------------------------------------------------------

const BOUNDARY: &str = "deepscreenframe";

/// Bind loopback on an ephemeral port and serve the stream. Returns the port.
pub fn serve(slot: PreviewSlot) -> io::Result<u16> {
    // Loopback only. This is a debug stream of someone's webcam; it has no
    // business being reachable from the network.
    let server = tiny_http::Server::http("127.0.0.1:0")
        .map_err(|e| io::Error::other(e.to_string()))?;
    let port = server.server_addr().to_ip().map(|a| a.port()).unwrap_or(0);

    std::thread::Builder::new().name("ds-preview-http".into()).spawn(move || {
        for request in server.incoming_requests() {
            let url = request.url().to_string();
            let slot = Arc::clone(&slot);
            // One thread per connection: the stream response never returns, so
            // it cannot share the accept loop.
            std::thread::spawn(move || match url.as_str() {
                "/stream" => {
                    let header = tiny_http::Header::from_bytes(
                        &b"Content-Type"[..],
                        format!("multipart/x-mixed-replace; boundary={BOUNDARY}").as_bytes(),
                    )
                    .expect("static header");
                    let reader = MjpegReader::new(slot);
                    let response =
                        tiny_http::Response::new(200.into(), vec![header], reader, None, None);
                    let _ = request.respond(response);
                }
                "/health" => {
                    let _ = request.respond(tiny_http::Response::from_string("ok"));
                }
                _ => {
                    let _ = request.respond(tiny_http::Response::from_string("not found").with_status_code(404));
                }
            });
        }
    })?;

    Ok(port)
}

/// Turns the preview slot into an endless multipart body.
struct MjpegReader {
    slot: PreviewSlot,
    last_seq: u64,
    buf: Vec<u8>,
    pos: usize,
}

impl MjpegReader {
    fn new(slot: PreviewSlot) -> Self {
        Self { slot, last_seq: u64::MAX, buf: Vec::new(), pos: 0 }
    }

    /// Block until a frame newer than `last_seq` appears, then stage one
    /// multipart part. Returns false if nothing arrives for a while, which
    /// ends the response cleanly rather than hanging the connection open —
    /// the browser simply reconnects.
    fn stage_next_part(&mut self) -> bool {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(item) = self.slot.load_full().as_ref().clone() {
                if item.seq != self.last_seq {
                    self.last_seq = item.seq;
                    self.buf.clear();
                    let _ = write!(
                        self.buf,
                        "\r\n--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                        item.jpeg.len()
                    );
                    self.buf.extend_from_slice(&item.jpeg);
                    self.pos = 0;
                    return true;
                }
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(4));
        }
    }
}

impl Read for MjpegReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.buf.len() && !self.stage_next_part() {
            return Ok(0); // EOF: no frames for 10 s
        }
        let n = out.len().min(self.buf.len() - self.pos);
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}
