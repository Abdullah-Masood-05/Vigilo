//! Live camera capture.
//!
//! **This is the harness path, not the production path.** It drives the camera
//! through ffmpeg's DirectShow input and reads rgb24 off a pipe, exactly like
//! the video-file source. That buys a real live test today, with no new
//! dependency and no `nokhwa` build fight, which is what you need to answer
//! "does my webcam actually deliver 30 fps at 720p" before any model exists.
//!
//! Build step 2 replaces this with `crabcamera` (MIT, Tauri v2 plugin, also
//! usable standalone via `PlatformCamera`), because shipping an exam client
//! that spawns ffmpeg is not sane: no frame-level control, an extra process, an
//! extra copy, and a hard dependency on an external binary. Keep this source
//! afterwards anyway — it is a useful reference to benchmark the real capture
//! path against.
//!
//! Two things this path makes visible immediately, both from MODELS.md §12:
//!
//! - **Ask for MJPEG.** Raw YUYV at 1280x720x30 is ~55 MB/s over USB. Most
//!   webcams simply refuse it and cap the frame rate instead.
//! - **There is no back pressure here.** ffmpeg keeps decoding whether or not
//!   anything reads, so a slow consumer gets *stale* frames rather than fewer
//!   frames. That is precisely why the real pipeline uses a latest-frame
//!   `ArcSwap` slot and never a queue (§6 rule 3).


use crate::config::CaptureConfig;
use crate::error::{DetectError, Result};
use crate::types::Frame;

use super::ffmpeg::RawRgbPipe;
use super::FrameSource;

/// A capture device as the platform reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraDevice {
    /// Position in the enumeration — this is what `camera:<index>` selects.
    pub index: usize,
    pub name: String,
    /// The unambiguous device path. Preferred over `name`, which is not
    /// guaranteed unique when two identical webcams are plugged in.
    pub alt_name: Option<String>,
}

impl CameraDevice {
    /// What to hand ffmpeg after `video=`.
    #[allow(dead_code)]
    fn selector(&self) -> &str {
        self.alt_name.as_deref().unwrap_or(&self.name)
    }
}

/// One mode the camera claims to support.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraFormat {
    /// `mjpeg`, `yuyv422`, ...
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
}

impl std::fmt::Display for CameraFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:<10} {}x{} @ {} fps", self.codec, self.width, self.height, self.fps)
    }
}

/// Enumerate video capture devices.
pub fn list_devices() -> Result<Vec<CameraDevice>> {
    #[cfg(windows)]
    {
        // ffmpeg prints the device list to stderr and then exits non-zero because
        // the dummy input cannot be opened. That is the documented way to do this.
        let out = super::quiet_command("ffmpeg")
            .args(["-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])
            .output()
            .map_err(|e| {
                DetectError::Camera(format!("could not run ffmpeg ({e}); it must be on PATH"))
            })?;

        let devices = parse_dshow_devices(&String::from_utf8_lossy(&out.stderr));
        if devices.is_empty() {
            return Err(DetectError::Camera(
                "no video capture devices found. Check that the camera is not in use by another \
                 app, and that Windows camera privacy settings allow desktop apps"
                    .into(),
            ));
        }
        Ok(devices)
    }
    #[cfg(target_os = "macos")]
    {
        let out = super::quiet_command("ffmpeg")
            .args(["-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""])
            .output()
            .map_err(|e| {
                DetectError::Camera(format!("could not run ffmpeg ({e}); it must be on PATH"))
            })?;

        let devices = parse_avfoundation_devices(&String::from_utf8_lossy(&out.stderr));
        if devices.is_empty() {
            return Err(DetectError::Camera(
                "no video capture devices found. Check camera permissions in macOS System Settings"
                    .into(),
            ));
        }
        Ok(devices)
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let devices = list_v4l2_devices();
        if devices.is_empty() {
            return Err(DetectError::Camera(
                "no video capture devices found in /dev/video*. Check that camera is plugged in and accessible"
                    .into(),
            ));
        }
        Ok(devices)
    }
}

#[cfg(not(any(windows, target_os = "macos")))]
fn list_v4l2_devices() -> Vec<CameraDevice> {
    let mut devices = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/sys/class/video4linux") {
        let mut paths: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        paths.sort_by_key(|e| e.file_name());
        for entry in paths {
            let filename = entry.file_name();
            let name_str = filename.to_string_lossy();
            if let Some(idx_str) = name_str.strip_prefix("video") {
                if let Ok(index) = idx_str.parse::<usize>() {
                    let dev_path = format!("/dev/{name_str}");
                    if std::path::Path::new(&dev_path).exists() {
                        let name = std::fs::read_to_string(entry.path().join("name"))
                            .map(|s| s.trim().to_string())
                            .unwrap_or_else(|_| format!("Camera {index}"));
                        devices.push(CameraDevice {
                            index,
                            name,
                            alt_name: Some(dev_path),
                        });
                    }
                }
            }
        }
    }
    if devices.is_empty() {
        for i in 0..16 {
            let p = format!("/dev/video{i}");
            if std::path::Path::new(&p).exists() {
                devices.push(CameraDevice {
                    index: i,
                    name: format!("Video Device {i}"),
                    alt_name: Some(p),
                });
            }
        }
    }
    devices
}

#[cfg(target_os = "macos")]
fn parse_avfoundation_devices(text: &str) -> Vec<CameraDevice> {
    let mut devices = Vec::new();
    let mut in_video_section = false;
    for line in text.lines() {
        if line.contains("AVFoundation video devices:") {
            in_video_section = true;
            continue;
        }
        if line.contains("AVFoundation audio devices:") {
            break;
        }
        if in_video_section {
            if let Some(start_bracket) = line.rfind('[') {
                if let Some(end_bracket) = line[start_bracket..].find(']') {
                    let idx_str = &line[start_bracket + 1..start_bracket + end_bracket];
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        let name = line[start_bracket + end_bracket + 1..].trim();
                        if !name.is_empty() {
                            devices.push(CameraDevice {
                                index: idx,
                                name: name.to_string(),
                                alt_name: None,
                            });
                        }
                    }
                }
            }
        }
    }
    devices
}

/// Modes a device claims to support.
pub fn list_formats(device: &CameraDevice) -> Result<Vec<CameraFormat>> {
    #[cfg(windows)]
    {
        let out = super::quiet_command("ffmpeg")
            .args(["-hide_banner", "-f", "dshow", "-list_options", "true", "-i"])
            .arg(format!("video={}", device.selector()))
            .output()
            .map_err(|e| DetectError::Camera(format!("could not run ffmpeg ({e})")))?;
        Ok(parse_dshow_formats(&String::from_utf8_lossy(&out.stderr)))
    }
    #[cfg(not(windows))]
    {
        let _ = device;
        Ok(vec![
            CameraFormat {
                codec: "raw".into(),
                width: 1280,
                height: 720,
                fps: 30.0,
            },
            CameraFormat {
                codec: "raw".into(),
                width: 640,
                height: 480,
                fps: 30.0,
            },
        ])
    }
}

/// Parse the device list out of ffmpeg's stderr.
///
/// Audio devices are dropped. Devices ffmpeg tags `(none)` are kept — virtual
/// cameras (OBS and friends) report that way, and excluding them would make
/// `camera:<index>` mean something different from what the user just read.
#[allow(dead_code)]
fn parse_dshow_devices(text: &str) -> Vec<CameraDevice> {
    let mut devices: Vec<CameraDevice> = Vec::new();
    let mut last_was_kept = false;

    for line in text.lines() {
        let line = strip_ffmpeg_prefix(line);

        if let Some(alt) = line.strip_prefix("Alternative name ") {
            if last_was_kept {
                if let Some(name) = unquote(alt.trim()) {
                    if let Some(dev) = devices.last_mut() {
                        dev.alt_name = Some(name.to_string());
                    }
                }
            }
            continue;
        }

        // `"Integrated Webcam" (video)`
        let Some(close) = line.rfind('"') else {
            last_was_kept = false;
            continue;
        };
        if !line.starts_with('"') || close == 0 {
            last_was_kept = false;
            continue;
        }
        let name = &line[1..close];
        let kind = line[close + 1..].trim();

        if kind == "(audio)" {
            last_was_kept = false;
            continue;
        }
        if kind != "(video)" && kind != "(none)" && !kind.is_empty() {
            last_was_kept = false;
            continue;
        }

        devices.push(CameraDevice { index: devices.len(), name: name.to_string(), alt_name: None });
        last_was_kept = true;
    }

    devices
}

/// Parse `-list_options` output. Lines look like:
///
/// ```text
/// vcodec=mjpeg  min s=1280x720 fps=30 max s=1280x720 fps=30
/// pixel_format=yuyv422  min s=640x480 fps=30 max s=640x480 fps=30
/// ```
///
/// The `max` figures are the ones worth reporting — that is the ceiling the
/// mode actually offers.
#[allow(dead_code)]
fn parse_dshow_formats(text: &str) -> Vec<CameraFormat> {
    let mut formats = Vec::new();

    for line in text.lines() {
        let line = strip_ffmpeg_prefix(line);
        let codec = field_after(line, "vcodec=").or_else(|| field_after(line, "pixel_format="));
        let Some(codec) = codec else { continue };

        // Take everything from `max ` so a mode with differing min/max is
        // reported at its ceiling rather than its floor.
        let tail = line.find("max ").map(|i| &line[i..]).unwrap_or(line);
        let Some(size) = field_after(tail, "s=") else { continue };
        let Some((w, h)) = size.split_once('x') else { continue };
        let (Ok(width), Ok(height)) = (w.parse::<u32>(), h.parse::<u32>()) else { continue };
        let fps = field_after(tail, "fps=").and_then(|f| f.parse::<f32>().ok()).unwrap_or(0.0);

        formats.push(CameraFormat { codec: codec.to_string(), width, height, fps });
    }

    formats
}

/// Strip the `[in#0 @ 0x...]` / `[dshow @ 0x...]` prefix ffmpeg puts on every
/// line. The tag has changed across ffmpeg versions, so match the shape.
#[allow(dead_code)]
fn strip_ffmpeg_prefix(line: &str) -> &str {
    let trimmed = line.trim();
    if trimmed.starts_with('[') {
        if let Some(close) = trimmed.find("] ") {
            return trimmed[close + 2..].trim();
        }
    }
    trimmed
}

#[allow(dead_code)]
fn field_after<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    Some(&rest[..end]).filter(|s| !s.is_empty())
}

#[allow(dead_code)]
fn unquote(s: &str) -> Option<&str> {
    s.strip_prefix('"')?.strip_suffix('"')
}

// ---------------------------------------------------------------------------
// the source
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct CameraSource {
    device: CameraDevice,
    width: u32,
    height: u32,
    fps: f32,
    mjpeg: bool,
    pipe: RawRgbPipe,
    /// The frame read during open, to prove the camera really delivers.
    primed: Option<Frame>,
}

impl CameraSource {
    pub fn open(cfg: &CaptureConfig) -> Result<Self> {
        let devices = list_devices()?;
        let device =
            devices.iter().find(|d| d.index == cfg.device_index as usize).cloned().ok_or_else(
                || {
                    let listed = devices
                        .iter()
                        .map(|d| format!("  camera:{}  {}", d.index, d.name))
                        .collect::<Vec<_>>()
                        .join("\n");
                    DetectError::Camera(format!(
                        "no device at index {}. Available:\n{listed}",
                        cfg.device_index
                    ))
                },
            )?;

        let mut cmd = super::quiet_command("ffmpeg");
        #[cfg(windows)]
        {
            cmd.args(["-v", "error", "-nostdin", "-f", "dshow"])
                // dshow drops frames and warns loudly without a real-time buffer.
                .args(["-rtbufsize", "128M"]);
            if cfg.prefer_mjpeg {
                cmd.args(["-vcodec", "mjpeg"]);
            }
            cmd.args(["-video_size", &format!("{}x{}", cfg.width, cfg.height)])
                .args(["-framerate", &cfg.fps.to_string()])
                .arg("-i")
                .arg(format!("video={}", device.selector()));
        }
        #[cfg(target_os = "macos")]
        {
            cmd.args(["-v", "error", "-nostdin", "-f", "avfoundation"])
                .args(["-framerate", &cfg.fps.to_string()])
                .args(["-video_size", &format!("{}x{}", cfg.width, cfg.height)])
                .arg("-i")
                .arg(format!("{}:none", device.index));
        }
        #[cfg(not(any(windows, target_os = "macos")))]
        {
            cmd.args(["-v", "error", "-nostdin", "-f", "v4l2"]);
            if cfg.prefer_mjpeg {
                cmd.args(["-input_format", "mjpeg"]);
            }
            let default_dev = format!("/dev/video{}", device.index);
            let dev = device.alt_name.as_deref().unwrap_or(&default_dev);
            cmd.args(["-video_size", &format!("{}x{}", cfg.width, cfg.height)])
                .args(["-framerate", &cfg.fps.to_string()])
                .arg("-i")
                .arg(dev);
        }

        let label = format!("camera:{} ({})", device.index, device.name);
        let mut pipe = RawRgbPipe::spawn(label, cfg.width, cfg.height, cmd, true)?;

        // Read one frame here rather than at first use. A camera that will not
        // start, or a mode it does not support, should fail at open with
        // ffmpeg's own explanation — not thirty seconds into a session.
        let primed = pipe.next_frame().map_err(|e| {
            DetectError::Camera(format!(
                "{e}\n\n\
                 Three things cause this, in order of likelihood:\n\
                 1. Another app is holding the camera — OBS, Teams, Zoom, Discord, or a browser \
                    tab. Windows lets exactly one process own a webcam. Close it and retry.\n\
                 2. The requested mode is not offered. You asked for {}{}x{} @ {} fps; \
                    run `detect-cli devices --formats` and pick a listed combination.\n\
                 3. Camera access is off for desktop apps in Windows privacy settings.",
                if cfg.prefer_mjpeg { "mjpeg " } else { "raw " },
                cfg.width,
                cfg.height,
                cfg.fps
            ))
        })?;

        Ok(Self {
            device,
            width: cfg.width,
            height: cfg.height,
            fps: cfg.fps as f32,
            mjpeg: cfg.prefer_mjpeg,
            pipe,
            primed,
        })
    }

    pub fn device(&self) -> &CameraDevice {
        &self.device
    }

    pub fn is_mjpeg(&self) -> bool {
        self.mjpeg
    }
}

impl FrameSource for CameraSource {
    fn next_frame(&mut self) -> Result<Option<Frame>> {
        if let Some(frame) = self.primed.take() {
            return Ok(Some(frame));
        }
        self.pipe.next_frame()
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn name(&self) -> String {
        format!("camera:{} ({})", self.device.index, self.device.name)
    }

    fn nominal_fps(&self) -> Option<f32> {
        Some(self.fps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real output from ffmpeg 8.1.2 on the development machine.
    const DEVICE_LIST: &str = r#"
[in#0 @ 0000000000717100] "Integrated Webcam" (video)
[in#0 @ 0000000000717100]   Alternative name "@device_pnp_\\?\usb#vid_0c45&pid_6a09&mi_00#6&47d1d30&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"
[in#0 @ 0000000000717100] "OBS Virtual Camera" (none)
[in#0 @ 0000000000717100]   Alternative name "@device_sw_{860BB310-5D01-11D0-BD3B-00A0C911CE86}\{A3FCE0F5-3493-419F-958A-ABA1250EC20B}"
[in#0 @ 0000000000717100] "Microphone Array (Realtek(R) Audio)" (audio)
[in#0 @ 0000000000717100]   Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{FCA63AC5-3CA6-4BEE-AD70-C0E92BCE028E}"
Error opening input file dummy.
"#;

    const FORMAT_LIST: &str = r#"
[in#0 @ 0000000000674b00]   vcodec=mjpeg  min s=1280x720 fps=30 max s=1280x720 fps=30
[in#0 @ 0000000000674b00]   vcodec=mjpeg  min s=640x480 fps=30 max s=640x480 fps=30
[in#0 @ 0000000000674b00]   pixel_format=yuyv422  min s=1280x720 fps=10 max s=1280x720 fps=10
[in#0 @ 0000000000674b00]   pixel_format=yuyv422  min s=320x180 fps=30 max s=320x180 fps=30
"#;

    #[test]
    fn parses_devices_and_drops_the_microphone() {
        let devices = parse_dshow_devices(DEVICE_LIST);
        assert_eq!(devices.len(), 2, "audio devices must not take up a camera index");
        assert_eq!(devices[0].index, 0);
        assert_eq!(devices[0].name, "Integrated Webcam");
        assert!(devices[0].alt_name.as_deref().unwrap().starts_with("@device_pnp_"));
        // A virtual camera reports as (none) but is still selectable.
        assert_eq!(devices[1].name, "OBS Virtual Camera");
        assert_eq!(devices[1].index, 1);
    }

    #[test]
    fn an_audio_alternative_name_is_not_attached_to_the_previous_camera() {
        // The mic's alt-name line follows the last kept device; attaching it
        // would silently point the camera at a microphone.
        let devices = parse_dshow_devices(DEVICE_LIST);
        assert!(devices[1].alt_name.as_deref().unwrap().starts_with("@device_sw_"));
    }

    #[test]
    fn empty_output_yields_no_devices_rather_than_a_bogus_one() {
        assert!(parse_dshow_devices("").is_empty());
        assert!(parse_dshow_devices("Error opening input file dummy.").is_empty());
    }

    #[test]
    fn parses_formats_including_the_yuyv_frame_rate_cliff() {
        let formats = parse_dshow_formats(FORMAT_LIST);
        assert_eq!(formats.len(), 4);
        assert_eq!(
            formats[0],
            CameraFormat { codec: "mjpeg".into(), width: 1280, height: 720, fps: 30.0 }
        );
        // This is the reason `prefer_mjpeg` defaults to true: the same
        // resolution over raw YUYV is capped at a third of the frame rate.
        let raw_720p = formats.iter().find(|f| f.codec == "yuyv422" && f.height == 720).unwrap();
        assert_eq!(raw_720p.fps, 10.0);
    }

    #[test]
    fn device_selector_prefers_the_unambiguous_path() {
        let with_alt = CameraDevice {
            index: 0,
            name: "Integrated Webcam".into(),
            alt_name: Some("@device_pnp_x".into()),
        };
        assert_eq!(with_alt.selector(), "@device_pnp_x");

        let without = CameraDevice { index: 0, name: "Integrated Webcam".into(), alt_name: None };
        assert_eq!(without.selector(), "Integrated Webcam");
    }
}
