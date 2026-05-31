//! In-process camera capture via `ccap-rs`.
//!
//! The alternative to [`super::camera`], which drives ffmpeg as a subprocess
//! and reads raw frames off a pipe. Same DirectShow backend underneath, so the
//! pixels should be identical — that equivalence is measured rather than
//! assumed, and the numbers are in `rust_context.md` §21.
//!
//! What this removes: a spawned process, a pipe read per frame, a stderr drain
//! thread, process reaping, and ~128 MB of bundled binaries. Also a whole class
//! of bug that only exists when your camera driver is another program — the
//! stderr drain that was read before ffmpeg had written to it, and reported an
//! empty error message, being the one already hit.
//!
//! What it keeps: `file:` and `dir:` replay still go through ffmpeg, because
//! decoding arbitrary video is exactly the job ffmpeg is good at and those
//! sources are development-only.

use std::time::Instant;

use ccap::{PixelFormat, PropertyName, Provider};

use crate::config::{CaptureBackend, CaptureConfig};
use crate::error::{DetectError, Result};
use crate::types::Frame;

use super::FrameSource;

/// How long to wait for a frame before giving up on one grab.
///
/// Generous relative to a 33 ms frame interval: a camera that has genuinely
/// stalled should surface as an error, but a driver hiccup during startup
/// should not end the session.
const GRAB_TIMEOUT_MS: u32 = 3000;

pub struct CcapSource {
    provider: Provider,
    width: u32,
    height: u32,
    fps: f32,
    index: u32,
    device_name: String,
    seq: u64,
    /// Reusable RGB buffer, so a frame does not allocate 2.7 MB per grab.
    ///
    /// Only used on the stride-padded path; when the source rows are tightly
    /// packed the frame data is copied straight out and this stays empty.
    packed: Vec<u8>,
}

impl CcapSource {
    pub fn open(cfg: &CaptureConfig) -> Result<Self> {
        // Backend selection is config, not a rebuild. Webcam vendors are
        // inconsistent about which Windows capture API their driver actually
        // works with, and we ship to machines whose cameras nobody here has
        // seen. `dshow` is the default because it is what the ffmpeg path used,
        // which is what makes the two comparable.
        let extra = match cfg.backend {
            CaptureBackend::Dshow => "dshow",
            CaptureBackend::Msmf => "msmf",
        };

        // Ordering here is empirical, and both halves of it matter.
        //
        // Pixel format must be requested **after** the device is open. Asking
        // before open returns success and then delivers BGR24 — the exact
        // silent channel swap that degrades detection instead of breaking it,
        // caught only because every frame is checked below.
        //
        // Resolution is requested in both positions and honoured in neither on
        // this camera; see the warning further down and rust_context.md §21.
        let mut provider = Provider::with_device_and_extra_info(cfg.device_index as i32, Some(extra))
            .map_err(|e| busy_or(format!("could not create capture provider: {e}")))?;

        if let Err(e) = provider.set_resolution(cfg.width, cfg.height) {
            tracing::warn!(error = %e, "pre-open set_resolution rejected");
        }

        provider
            .open_with_index(cfg.device_index as i32, false)
            .map_err(|e| busy_or(format!("could not open camera {}: {e}", cfg.device_index)))?;

        if let Err(e) = provider.set_resolution(cfg.width, cfg.height) {
            tracing::warn!(error = %e, "post-open set_resolution rejected");
        }
        if let Err(e) = provider.set_frame_rate(cfg.fps as f64) {
            tracing::warn!(error = %e, "set_frame_rate rejected");
        }
        // `Frame` carries RGB end to end. ccap will just as happily hand back
        // BGR, and getting that backwards is the same silent failure class as
        // the YuNet channel order in §6.
        if let Err(e) = provider.set_pixel_format(PixelFormat::Rgb24) {
            tracing::warn!(error = %e, "set_pixel_format rejected");
        }

        provider
            .start_capture()
            .map_err(|e| busy_or(format!("could not start capture on camera {}: {e}", cfg.device_index)))?;

        if !provider.is_started() {
            return Err(busy_or(format!("camera {} opened but did not start", cfg.device_index)));
        }

        // Learn the real geometry from an actual frame, not from
        // `get_property` — that reports what was *asked for*, so a driver that
        // silently downgraded reads back as success. Priming one frame here
        // also gets the negotiation done before the pipeline starts timing
        // anything.
        let primed = provider
            .grab_frame(GRAB_TIMEOUT_MS)
            .map_err(|e| busy_or(format!("could not read a first frame: {e}")))?
            .ok_or_else(|| {
                busy_or(format!("camera {} produced no frame after starting", cfg.device_index))
            })?;
        let (width, height, format) = {
            let info = primed
                .info()
                .map_err(|e| DetectError::Camera(format!("frame info unavailable: {e}")))?;
            (info.width, info.height, info.pixel_format)
        };
        let fps = provider.get_property(PropertyName::FrameRate).unwrap_or(cfg.fps as f64) as f32;

        if (width, height) != (cfg.width, cfg.height) {
            // Not fatal — a camera that only does 640x480 should still work —
            // but it changes how large a face is in frame, and every threshold
            // was tuned at 720p. Silence here would make that invisible.
            tracing::warn!(
                requested = format!("{}x{}", cfg.width, cfg.height),
                got = format!("{width}x{height}"),
                "camera did not honour the requested resolution"
            );
        }

        // Supported list comes from the already-open device. Enumerating with
        // `Provider::get_devices()` before opening looked like the obvious
        // place for this and is a trap: it opens each device to interrogate it
        // and leaves the camera locked, so the subsequent open fails with
        // "Render stream failed" — which reads like a driver fault and is not.
        if let Ok(info) = provider.device_info() {
            tracing::debug!(
                resolutions = ?info.supported_resolutions,
                formats = ?info.supported_pixel_formats,
                "ccap device capabilities"
            );
        }

        let device_name = provider
            .device_info()
            .ok()
            .map(|d| d.name.clone())
            .unwrap_or_else(|| format!("device {}", cfg.device_index));

        tracing::info!(
            device = %device_name,
            index = cfg.device_index,
            backend = extra,
            width,
            height,
            fps,
            format = ?format,
            "ccap capture started"
        );

        Ok(Self {
            provider,
            width,
            height,
            fps,
            index: cfg.device_index,
            device_name,
            seq: 0,
            packed: Vec::new(),
        })
    }
}

/// Map a ccap failure onto the plain-language cause when it looks like one.
///
/// Windows lets exactly one process own a webcam, and that single fact
/// accounts for most "the camera does not work" reports — §4 records it
/// costing real time. The ffmpeg path already said so in words; losing that
/// message on the way to a nicer capture API would be a downgrade.
fn busy_or(detail: String) -> DetectError {
    let lower = detail.to_ascii_lowercase();
    let looks_busy = lower.contains("in use")
        || lower.contains("busy")
        || lower.contains("access")
        || lower.contains("denied")
        || lower.contains("0x800700aa");
    if looks_busy {
        DetectError::Camera(format!(
            "{detail}\n\nWindows lets exactly one process own a webcam. Close OBS, Teams, \
             Zoom, Discord, a browser tab, or another copy of this app, then try again."
        ))
    } else {
        DetectError::Camera(detail)
    }
}

impl FrameSource for CcapSource {
    fn next_frame(&mut self) -> Result<Option<Frame>> {
        let grabbed = self
            .provider
            .grab_frame(GRAB_TIMEOUT_MS)
            .map_err(|e| DetectError::Camera(format!("grab failed: {e}")))?;

        let Some(video) = grabbed else {
            // Timed out. A live camera has no end-of-stream, so this is a
            // stall rather than completion — `None` would tell the pipeline
            // the session ended cleanly, which it did not.
            return Err(DetectError::Camera(format!(
                "camera {} produced no frame within {GRAB_TIMEOUT_MS} ms",
                self.index
            )));
        };

        let info = video
            .info()
            .map_err(|e| DetectError::Camera(format!("frame info unavailable: {e}")))?;

        if info.pixel_format != PixelFormat::Rgb24 {
            // Deliberately fatal rather than best-effort. Every downstream
            // model assumes RGB, and a wrong channel order degrades detection
            // silently instead of failing — far better to stop here and say
            // so than to ship subtly worse accuracy nobody can see.
            return Err(DetectError::Camera(format!(
                "camera returned {:?}, expected Rgb24. The pipeline is RGB end to end and a \
                 silent channel swap degrades detection rather than breaking it.",
                info.pixel_format
            )));
        }

        let (w, h) = (info.width, info.height);
        let stride = info.strides[0] as usize;
        let row_bytes = w as usize * 3;
        let src = info.data_planes[0].ok_or_else(|| {
            DetectError::Camera("frame carried no pixel plane".into())
        })?;

        // Rows may be padded to a hardware-friendly stride. `Frame` is tightly
        // packed, and handing a padded buffer downstream shears the image by a
        // few pixels per row — which looks like a corrupted camera rather than
        // a layout bug.
        let data: std::sync::Arc<[u8]> = if stride == row_bytes {
            std::sync::Arc::from(src)
        } else {
            self.packed.clear();
            self.packed.reserve(row_bytes * h as usize);
            for y in 0..h as usize {
                let start = y * stride;
                self.packed.extend_from_slice(&src[start..start + row_bytes]);
            }
            std::sync::Arc::from(self.packed.as_slice())
        };

        self.seq += 1;
        Ok(Some(Frame {
            data,
            width: w,
            height: h,
            seq: self.seq,
            captured_at: Instant::now(),
        }))
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn name(&self) -> String {
        format!("ccap:{} ({})", self.index, self.device_name)
    }

    fn nominal_fps(&self) -> Option<f32> {
        Some(self.fps)
    }
}
