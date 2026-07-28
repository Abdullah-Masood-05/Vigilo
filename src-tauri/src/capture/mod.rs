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

/// Spawn a helper process without letting Windows pop a console window for it.
///
/// `ffmpeg` and `ffprobe` are console subsystem executables. The app is a GUI
/// process (`windows_subsystem = "windows"`), so it has no console of its own
/// — and when a GUI process starts a console child, Windows helpfully
/// **allocates a new console window** for it. The result is a black terminal
/// appearing next to the app, once per probe and once per capture session,
/// which looks exactly like a crash to anyone who did not write this.
///
/// `CREATE_NO_WINDOW` suppresses that. Every `Command` in this module must go
/// through here; one that does not is a stray window on a tester's screen.
///
/// No-op on non-Windows, where the whole problem does not exist.
/// Directory holding the bundled `ffmpeg.exe` / `ffprobe.exe`, once located.
///
/// A process-wide static rather than something threaded through `Config`,
/// because this is not a tunable — it is a fact about where this installation
/// put its files, discovered once at startup by whoever knows about resource
/// directories. The library still resolves nothing itself (MODELS.md §9): the
/// app tells it, exactly as it does for model paths.
static FFMPEG_DIR: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();

/// Point capture at a directory containing `ffmpeg.exe` and `ffprobe.exe`.
///
/// Call once, before opening any source. Later calls are ignored, so a stray
/// second call cannot silently repoint a running pipeline at different
/// binaries.
pub fn set_ffmpeg_dir(dir: Option<std::path::PathBuf>) {
    let _ = FFMPEG_DIR.set(dir);
}

/// Absolute path to a bundled helper, or `None` to fall back to `PATH`.
///
/// Bundled wins deliberately. A tester may well have some other ffmpeg on
/// `PATH` — an ancient one, or a build without DirectShow — and "works on my
/// machine but not theirs" traced to a stranger's ffmpeg version is not a bug
/// anyone wants to debug remotely. The copy shipped alongside is the one that
/// was tested.
fn bundled(program: &str) -> Option<std::path::PathBuf> {
    let dir = FFMPEG_DIR.get()?.as_ref()?;
    let exe = dir.join(if cfg!(windows) {
        format!("{program}.exe")
    } else {
        program.to_string()
    });
    exe.exists().then_some(exe)
}

pub(crate) fn quiet_command(program: &str) -> std::process::Command {
    #[allow(unused_mut)]
    let mut cmd = match bundled(program) {
        Some(path) => std::process::Command::new(path),
        None => std::process::Command::new(program),
    };
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        /// `CREATE_NO_WINDOW`, from `winbase.h`. Spelled out rather than
        /// pulling in the `windows-sys` crate for one constant.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Is `ffmpeg` itself usable — bundled or on `PATH`?
///
/// **Deliberately does not check `ffprobe`.** The camera path — what every
/// installed copy of this app actually uses — never calls it; only `file:`
/// replay does, to read a clip's dimensions, and that is a development-only
/// source never exposed through the installed app's normal launch. Since
/// §22's minimal build ships `ffmpeg.exe` alone, requiring `ffprobe` here
/// would fail every fresh install for a binary the camera path never touches.
///
/// Still checked even though the installer ships its own `ffmpeg.exe`: a build
/// run from a source tree has no bundle, and an install with a deleted or
/// blocked `ffmpeg/` folder should say so rather than fail at the first frame.
pub fn ffmpeg_available() -> bool {
    binary_runs("ffmpeg")
}

fn binary_runs(name: &str) -> bool {
    quiet_command(name)
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .stdin(std::process::Stdio::null())
        .status()
        .is_ok()
}

/// What to tell someone who does not have it.
///
/// Lives here, next to the check, rather than in the UI layer: the fact that
/// this app needs ffmpeg is a property of the capture implementation, and when
/// capture stops shelling out this string should disappear with it.
pub const FFMPEG_MISSING_HELP: &str = "\
This app needs a free tool called ffmpeg to read your webcam, and it is not \
installed yet.

To install it, open the Start menu, type \"Terminal\" or \"PowerShell\", open \
it, then copy and paste this line and press Enter:

    winget install --id Gyan.FFmpeg -e

If that does not work, download it from https://www.gyan.dev/ffmpeg/builds/ \
(choose \"release essentials\"), unzip it, and add its \"bin\" folder to your \
PATH.

When it has finished installing, close this app and open it again.";

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
