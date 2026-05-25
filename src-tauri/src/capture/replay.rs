//! Replay sources: a video file, or a directory of images.
//!
//! This is the source that makes the crate testable — headless, in CI, with no
//! camera and no browser. Tuning happens against recorded footage, so the
//! replay path is not a test convenience bolted on afterwards; it is the
//! primary development surface (MODELS.md §0, §10).
//!
//! Video decoding shells out to `ffmpeg`, which is decoded once to raw rgb24
//! and piped in. That avoids linking a C decoder into the crate — an
//! `ffmpeg-sys` build on Windows costs a day and buys nothing here, because
//! decoding speed is not what we are measuring.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use crate::error::{DetectError, Result};
use crate::types::Frame;

use super::ffmpeg::{Pacer, RawRgbPipe};
use super::FrameSource;

// ---------------------------------------------------------------------------
// video file
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct VideoFileSource {
    path: PathBuf,
    width: u32,
    height: u32,
    fps: f32,
    pipe: RawRgbPipe,
    pacer: Pacer,
}

impl VideoFileSource {
    pub fn open(path: impl AsRef<Path>, paced: bool) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(DetectError::source(path.display().to_string(), "file does not exist"));
        }

        let probe = probe_video(&path)?;

        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error", "-nostdin", "-i"]).arg(&path);

        // End of file is the expected way a clip finishes, so EOF is not an
        // error here — unlike a camera, where it means the device vanished.
        let pipe = RawRgbPipe::spawn(
            format!("file:{}", path.display()),
            probe.width,
            probe.height,
            cmd,
            false,
        )?;

        Ok(Self {
            path,
            width: probe.width,
            height: probe.height,
            fps: probe.fps,
            pipe,
            pacer: Pacer::new(probe.fps, paced),
        })
    }
}

impl FrameSource for VideoFileSource {
    fn next_frame(&mut self) -> Result<Option<Frame>> {
        self.pacer.wait_for(self.pipe.seq());
        self.pipe.next_frame()
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn name(&self) -> String {
        format!("file:{}", self.path.display())
    }

    fn nominal_fps(&self) -> Option<f32> {
        Some(self.fps)
    }
}

struct VideoProbe {
    width: u32,
    height: u32,
    fps: f32,
}

fn probe_video(path: &Path) -> Result<VideoProbe> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .map_err(|e| {
            DetectError::source(
                path.display().to_string(),
                format!("could not run ffprobe ({e}); ffmpeg must be on PATH to replay video"),
            )
        })?;

    if !out.status.success() {
        return Err(DetectError::source(
            path.display().to_string(),
            format!("ffprobe failed: {}", String::from_utf8_lossy(&out.stderr).trim()),
        ));
    }

    let text = String::from_utf8_lossy(&out.stdout);
    let mut width = None;
    let mut height = None;
    let mut fps = 30.0f32;
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else { continue };
        match key.trim() {
            "width" => width = value.trim().parse::<u32>().ok(),
            "height" => height = value.trim().parse::<u32>().ok(),
            "avg_frame_rate" => {
                if let Some((num, den)) = value.trim().split_once('/') {
                    let num = num.parse::<f32>().unwrap_or(30.0);
                    let den = den.parse::<f32>().unwrap_or(1.0);
                    if den > 0.0 && num > 0.0 {
                        fps = num / den;
                    }
                }
            }
            _ => {}
        }
    }

    match (width, height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => Ok(VideoProbe { width: w, height: h, fps }),
        _ => Err(DetectError::source(
            path.display().to_string(),
            "ffprobe reported no video stream dimensions",
        )),
    }
}

// ---------------------------------------------------------------------------
// image directory
// ---------------------------------------------------------------------------

/// A directory of images, sorted by filename. Zero external dependencies, so
/// this is the source CI uses — it works on a machine with no ffmpeg and no
/// camera. Name frames zero-padded (`frame_0001.png`) so the lexicographic
/// sort is also the temporal one.
#[derive(Debug)]
pub struct ImageDirSource {
    dir: PathBuf,
    files: Vec<PathBuf>,
    next: usize,
    width: u32,
    height: u32,
    fps: f32,
    seq: u64,
    pacer: Pacer,
}

const IMAGE_EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "bmp", "webp"];

impl ImageDirSource {
    pub fn open(dir: impl AsRef<Path>, fps: u32, paced: bool) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let entries = std::fs::read_dir(&dir).map_err(|e| DetectError::io(&dir, e))?;

        let mut files: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
                    .unwrap_or(false)
            })
            .collect();
        files.sort();

        if files.is_empty() {
            return Err(DetectError::source(
                dir.display().to_string(),
                format!("no images found (looked for {})", IMAGE_EXTENSIONS.join(", ")),
            ));
        }

        // Dimensions come from the first image; the rest must match, because a
        // mid-stream resolution change would silently corrupt every model's
        // letterboxing.
        let first = image::open(&files[0])
            .map_err(|e| DetectError::source(files[0].display().to_string(), e.to_string()))?;
        let (width, height) = (first.width(), first.height());

        Ok(Self {
            dir,
            files,
            next: 0,
            width,
            height,
            fps: fps as f32,
            seq: 0,
            pacer: Pacer::new(fps as f32, paced),
        })
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }
}

impl FrameSource for ImageDirSource {
    fn next_frame(&mut self) -> Result<Option<Frame>> {
        let Some(path) = self.files.get(self.next).cloned() else {
            return Ok(None);
        };
        self.next += 1;

        let img = image::open(&path)
            .map_err(|e| DetectError::source(path.display().to_string(), e.to_string()))?;
        if img.width() != self.width || img.height() != self.height {
            return Err(DetectError::source(
                path.display().to_string(),
                format!(
                    "resolution changed mid-sequence: expected {}x{}, got {}x{}",
                    self.width,
                    self.height,
                    img.width(),
                    img.height()
                ),
            ));
        }

        let rgb = img.into_rgb8();
        self.pacer.wait_for(self.seq);
        let frame = Frame {
            data: Arc::from(rgb.into_raw().as_slice()),
            width: self.width,
            height: self.height,
            seq: self.seq,
            captured_at: Instant::now(),
        };
        self.seq += 1;
        Ok(Some(frame))
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn name(&self) -> String {
        format!("dir:{}", self.dir.display())
    }

    fn nominal_fps(&self) -> Option<f32> {
        Some(self.fps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_png(dir: &Path, name: &str, w: u32, h: u32, fill: [u8; 3]) {
        let mut img = image::RgbImage::new(w, h);
        for px in img.pixels_mut() {
            *px = image::Rgb(fill);
        }
        img.save(dir.join(name)).unwrap();
    }

    #[test]
    fn image_dir_yields_frames_in_filename_order_then_stops() {
        let tmp = tempfile::tempdir().unwrap();
        write_png(tmp.path(), "frame_0002.png", 8, 4, [0, 255, 0]);
        write_png(tmp.path(), "frame_0001.png", 8, 4, [255, 0, 0]);
        std::fs::write(tmp.path().join("notes.txt"), "ignored").unwrap();

        let mut src = ImageDirSource::open(tmp.path(), 30, false).unwrap();
        assert_eq!(src.len(), 2);
        assert_eq!(src.resolution(), (8, 4));

        let f0 = src.next_frame().unwrap().unwrap();
        assert_eq!(f0.seq, 0);
        assert_eq!(f0.data.len(), f0.expected_len());
        assert_eq!(&f0.data[0..3], &[255, 0, 0], "first file should sort first");

        let f1 = src.next_frame().unwrap().unwrap();
        assert_eq!(f1.seq, 1);
        assert_eq!(&f1.data[0..3], &[0, 255, 0]);

        assert!(src.next_frame().unwrap().is_none());
        assert!(src.next_frame().unwrap().is_none(), "exhausted source stays exhausted");
    }

    #[test]
    fn image_dir_rejects_a_resolution_change_mid_sequence() {
        let tmp = tempfile::tempdir().unwrap();
        write_png(tmp.path(), "a.png", 8, 4, [1, 2, 3]);
        write_png(tmp.path(), "b.png", 16, 4, [1, 2, 3]);

        let mut src = ImageDirSource::open(tmp.path(), 30, false).unwrap();
        assert!(src.next_frame().unwrap().is_some());
        let err = src.next_frame().unwrap_err().to_string();
        assert!(err.contains("resolution changed"), "{err}");
    }

    #[test]
    fn empty_dir_fails_with_a_message_that_says_what_it_wanted() {
        let tmp = tempfile::tempdir().unwrap();
        let err = ImageDirSource::open(tmp.path(), 30, false).unwrap_err().to_string();
        assert!(err.contains("no images found"), "{err}");
        assert!(err.contains("png"), "{err}");
    }

    #[test]
    fn missing_video_file_is_reported_before_ffmpeg_is_involved() {
        let err = VideoFileSource::open("does/not/exist.mp4", false).unwrap_err().to_string();
        assert!(err.contains("does not exist"), "{err}");
    }
}
