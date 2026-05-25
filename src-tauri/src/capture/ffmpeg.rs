//! Shared plumbing: run ffmpeg, read raw rgb24 frames off its stdout.
//!
//! Used by both the video-file replay source and the interim camera source.
//! Decoding in a subprocess rather than linking a C decoder is deliberate — an
//! `ffmpeg-sys` build on Windows costs a day, and decode speed is not what any
//! of this measures.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::error::{DetectError, Result};
use crate::types::Frame;

/// Sleeps so frames arrive at roughly `fps`, when pacing is on.
#[derive(Debug)]
pub(crate) struct Pacer {
    fps: f32,
    start: Instant,
    enabled: bool,
}

impl Pacer {
    pub(crate) fn new(fps: f32, enabled: bool) -> Self {
        Self { fps: if fps > 0.0 { fps } else { 30.0 }, start: Instant::now(), enabled }
    }

    pub(crate) fn wait_for(&self, seq: u64) {
        if !self.enabled {
            return;
        }
        let due = Duration::from_secs_f32(seq as f32 / self.fps);
        let elapsed = self.start.elapsed();
        if due > elapsed {
            std::thread::sleep(due - elapsed);
        }
    }
}

/// A running ffmpeg process emitting tightly packed rgb24 frames of a known
/// size. Owns the child and kills it on drop.
pub(crate) struct RawRgbPipe {
    label: String,
    width: u32,
    height: u32,
    child: Option<Child>,
    stderr: Arc<Mutex<String>>,
    /// Held so the drain can be joined before its output is read. Without
    /// that, a process that dies instantly reports an empty reason — the
    /// thread has not appended anything yet — which is exactly the case where
    /// ffmpeg's message is the whole diagnosis.
    stderr_drain: Option<std::thread::JoinHandle<()>>,
    buf: Vec<u8>,
    seq: u64,
    done: bool,
    /// A file ending is normal; a camera ending is a fault.
    eof_is_error: bool,
}

impl RawRgbPipe {
    /// `cmd` should carry the input arguments only — output arguments, pipe
    /// wiring and stdio are set here so every caller gets the same contract.
    pub(crate) fn spawn(
        label: impl Into<String>,
        width: u32,
        height: u32,
        mut cmd: Command,
        eof_is_error: bool,
    ) -> Result<Self> {
        let label = label.into();
        let mut child = cmd
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-sn", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .spawn()
            .map_err(|e| {
                DetectError::source(
                    label.clone(),
                    format!("could not start ffmpeg ({e}); it must be on PATH"),
                )
            })?;

        // Drain stderr on its own thread. ffmpeg with `-v error` says little,
        // but a full pipe buffer would deadlock the decode.
        let stderr = Arc::new(Mutex::new(String::new()));
        let stderr_drain = child.stderr.take().map(|mut err| {
            let sink = Arc::clone(&stderr);
            std::thread::spawn(move || {
                let mut s = String::new();
                let _ = err.read_to_string(&mut s);
                if let Ok(mut guard) = sink.lock() {
                    guard.push_str(&s);
                }
            })
        });

        Ok(Self {
            label,
            width,
            height,
            child: Some(child),
            stderr,
            stderr_drain,
            buf: vec![0u8; width as usize * height as usize * 3],
            seq: 0,
            done: false,
            eof_is_error,
        })
    }

    pub(crate) fn ffmpeg_said(&self) -> String {
        self.stderr.lock().map(|s| s.trim().to_string()).unwrap_or_default()
    }

    pub(crate) fn seq(&self) -> u64 {
        self.seq
    }

    pub(crate) fn next_frame(&mut self) -> Result<Option<Frame>> {
        if self.done {
            return Ok(None);
        }
        let child = match self.child.as_mut() {
            Some(c) => c,
            None => return Ok(None),
        };
        let stdout = child.stdout.as_mut().expect("stdout was piped");

        // read_exact, but distinguishing "clean end of stream" from "the
        // decoder died mid-frame" — those need different reactions.
        let mut filled = 0usize;
        while filled < self.buf.len() {
            match stdout.read(&mut self.buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    self.done = true;
                    return Err(DetectError::source(self.label.clone(), e.to_string()));
                }
            }
        }

        if filled == 0 {
            self.done = true;
            // Order matters: reap the process, then join the drain, and only
            // then read what it collected. Reading first races the thread and
            // loses the one piece of information worth having.
            let status = self.child.as_mut().and_then(|c| c.wait().ok());
            if let Some(handle) = self.stderr_drain.take() {
                let _ = handle.join();
            }
            let msg = self.ffmpeg_said();
            let failed = status.map(|s| !s.success()).unwrap_or(false);
            if failed || self.eof_is_error {
                return Err(DetectError::source(
                    self.label.clone(),
                    if msg.is_empty() {
                        "stream ended unexpectedly".to_string()
                    } else {
                        format!("stream ended: {msg}")
                    },
                ));
            }
            return Ok(None);
        }

        if filled < self.buf.len() {
            self.done = true;
            tracing::warn!(
                source = %self.label,
                got = filled,
                want = self.buf.len(),
                "stream ended mid-frame; discarding the partial frame"
            );
            return Ok(None);
        }

        let frame = Frame {
            data: Arc::from(self.buf.as_slice()),
            width: self.width,
            height: self.height,
            seq: self.seq,
            captured_at: Instant::now(),
        };
        self.seq += 1;
        Ok(Some(frame))
    }
}

impl Drop for RawRgbPipe {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl std::fmt::Debug for RawRgbPipe {
    /// Hand-written so a panic message never dumps a megabyte of pixels.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawRgbPipe")
            .field("label", &self.label)
            .field("resolution", &(self.width, self.height))
            .field("seq", &self.seq)
            .field("done", &self.done)
            .finish()
    }
}
