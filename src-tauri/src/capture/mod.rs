//! Frame sources (MODELS.md §3).
//!
//! `FrameSource` is a trait so the same `Detector` runs against a live camera,
//! a video file, or a directory of PNGs with no code change. That is what
//! makes the module testable headless, in CI, without a camera — which is the
//! whole reason this crate has no Tauri dependency.

pub mod camera;
mod ffmpeg;
pub mod replay;

use std::str::FromStr;

use crate::config::CaptureConfig;
use crate::error::{DetectError, Result};
use crate::types::Frame;

pub trait FrameSource: Send {
    /// `Ok(None)` means the source is exhausted — end of file, end of
    /// directory. A live camera never returns `None`; it blocks.
    fn next_frame(&mut self) -> Result<Option<Frame>>;

    fn resolution(&self) -> (u32, u32);

    /// Human-readable identity, recorded in `SessionReport` so a stored report
    /// says what it was actually looking at.
    fn name(&self) -> String {
        "unknown".to_string()
    }

    /// Declared frame rate, when the source knows one.
    fn nominal_fps(&self) -> Option<f32> {
        None
    }
}

/// How a source is named on the command line and in config.
///
/// ```text
/// camera:0                 device index 0
/// file:samples/phone.mp4   video file, decoded via ffmpeg
/// dir:samples/frames       directory of images, sorted by filename
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceSpec {
    Camera { index: u32 },
    File { path: std::path::PathBuf },
    Dir { path: std::path::PathBuf },
}

impl FromStr for SourceSpec {
    type Err = DetectError;

    fn from_str(s: &str) -> Result<Self> {
        let (kind, rest) = s.split_once(':').ok_or_else(|| {
            DetectError::source(s, "expected one of camera:<index>, file:<path>, dir:<path>")
        })?;
        match kind {
            "camera" | "cam" => {
                let index = rest.parse::<u32>().map_err(|_| {
                    DetectError::source(s, format!("camera index must be a number, got {rest:?}"))
                })?;
                Ok(SourceSpec::Camera { index })
            }
            "file" => Ok(SourceSpec::File { path: rest.into() }),
            "dir" => Ok(SourceSpec::Dir { path: rest.into() }),
            other => Err(DetectError::source(
                s,
                format!("unknown source kind {other:?}; expected camera, file or dir"),
            )),
        }
    }
}

impl SourceSpec {
    /// Open the source this spec names. `paced` makes replay sources sleep to
    /// their nominal frame rate instead of running flat out — off for
    /// benchmarking and recording, on when eyeballing a clip live.
    pub fn open(&self, capture: &CaptureConfig, paced: bool) -> Result<Box<dyn FrameSource>> {
        match self {
            SourceSpec::Camera { index } => {
                let mut cfg = capture.clone();
                cfg.device_index = *index;
                camera::CameraSource::open(&cfg).map(|s| Box::new(s) as Box<dyn FrameSource>)
            }
            SourceSpec::File { path } => replay::VideoFileSource::open(path, paced)
                .map(|s| Box::new(s) as Box<dyn FrameSource>),
            SourceSpec::Dir { path } => {
                replay::ImageDirSource::open(path, capture.fps.max(1), paced)
                    .map(|s| Box::new(s) as Box<dyn FrameSource>)
            }
        }
    }
}

impl std::fmt::Display for SourceSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceSpec::Camera { index } => write!(f, "camera:{index}"),
            SourceSpec::File { path } => write!(f, "file:{}", path.display()),
            SourceSpec::Dir { path } => write!(f, "dir:{}", path.display()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_spec_form() {
        assert_eq!("camera:0".parse::<SourceSpec>().unwrap(), SourceSpec::Camera { index: 0 });
        assert_eq!("cam:2".parse::<SourceSpec>().unwrap(), SourceSpec::Camera { index: 2 });
        assert_eq!(
            "file:samples/a b.mp4".parse::<SourceSpec>().unwrap(),
            SourceSpec::File { path: "samples/a b.mp4".into() }
        );
        // Windows paths contain a colon; only the first one is the separator.
        assert_eq!(
            r"file:C:\clips\a.mp4".parse::<SourceSpec>().unwrap(),
            SourceSpec::File { path: r"C:\clips\a.mp4".into() }
        );
    }

    #[test]
    fn rejects_nonsense_with_a_useful_message() {
        let err = "webcam".parse::<SourceSpec>().unwrap_err().to_string();
        assert!(err.contains("camera:<index>"), "{err}");
        assert!("camera:left".parse::<SourceSpec>().is_err());
        assert!("rtsp:whatever".parse::<SourceSpec>().is_err());
    }

    #[test]
    fn display_roundtrips() {
        for s in ["camera:1", "file:x.mp4", "dir:frames"] {
            assert_eq!(s.parse::<SourceSpec>().unwrap().to_string(), s);
        }
    }
}
