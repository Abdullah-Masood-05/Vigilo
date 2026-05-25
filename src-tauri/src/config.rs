//! One injected struct, one source of truth (MODELS.md §7).
//!
//! Every tunable number lives here and **nowhere else**. The old module had
//! the same constants duplicated across store, hooks and worker with three
//! different values for the same thing; a single `Config` is what prevents
//! that. Defaults are seeded from `CONTEXT.md`'s measured numbers, corrected
//! per MODELS.md §4 where the old value was known to be wrong.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{DetectError, Result};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub capture: CaptureConfig,
    pub models: ModelPaths,
    pub cadence: CadenceConfig,
    pub thresholds: Thresholds,
    pub runtime: RuntimeConfig,
}

impl Config {
    /// Load from TOML or JSON, chosen by file extension.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| DetectError::io(path, e))?;
        let cfg: Config = match path.extension().and_then(|e| e.to_str()) {
            Some("json") => serde_json::from_str(&text)
                .map_err(|e| DetectError::Config(format!("{}: {e}", path.display())))?,
            _ => toml::from_str(&text)
                .map_err(|e| DetectError::Config(format!("{}: {e}", path.display())))?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).map_err(|e| DetectError::Config(e.to_string()))
    }

    /// Catch the config mistakes that would otherwise show up as mysterious
    /// runtime behaviour — inverted hysteresis being the classic one.
    pub fn validate(&self) -> Result<()> {
        let t = &self.thresholds;

        // Hysteresis: exit must be easier to satisfy than enter, or the
        // violation latches on and never clears.
        check_hysteresis("pose.yaw", t.pose.yaw_enter_deg, t.pose.yaw_exit_deg)?;
        check_hysteresis("pose.pitch", t.pose.pitch_enter_deg, t.pose.pitch_exit_deg)?;
        check_hysteresis("gaze.yaw", t.gaze.yaw_enter_rad, t.gaze.yaw_exit_rad)?;
        check_hysteresis("gaze.pitch", t.gaze.pitch_enter_rad, t.gaze.pitch_exit_rad)?;
        check_hysteresis(
            "debug_direction",
            t.debug_direction.enter_deg,
            t.debug_direction.exit_deg,
        )?;

        // Identity runs the other way round: similarity *below* enter trips it.
        if t.identity.cosine_exit < t.identity.cosine_enter {
            return Err(DetectError::Config(format!(
                "identity.cosine_exit ({}) must be >= cosine_enter ({}) — \
                 similarity has to recover past a higher bar than it fell through",
                t.identity.cosine_exit, t.identity.cosine_enter
            )));
        }

        for (name, hz) in [
            ("face", self.cadence.face_hz),
            ("objects", self.cadence.object_hz),
            ("identity", self.cadence.identity_hz),
        ] {
            if hz <= 0.0 {
                return Err(DetectError::Config(format!(
                    "cadence.{name}_hz must be > 0, got {hz}"
                )));
            }
        }

        if self.capture.width == 0 || self.capture.height == 0 {
            return Err(DetectError::Config("capture resolution must be non-zero".into()));
        }

        for (name, alpha) in
            [("pose.ema_alpha", t.pose.ema_alpha), ("gaze.ema_alpha", t.gaze.ema_alpha)]
        {
            if !(0.0..=1.0).contains(&alpha) {
                return Err(DetectError::Config(format!("{name} must be in 0..=1, got {alpha}")));
            }
        }

        Ok(())
    }
}

fn check_hysteresis(name: &str, enter: f64, exit: f64) -> Result<()> {
    if exit > enter {
        return Err(DetectError::Config(format!(
            "{name}_exit ({exit}) must be <= {name}_enter ({enter}) — \
             an exit threshold above the enter threshold means the signal flaps"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// capture
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CaptureConfig {
    pub device_index: u32,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// Ask for MJPEG when the camera offers it. Raw YUYV at 1280x720x30 is
    /// ~55 MB/s over USB and caps you at the bus, not the model
    /// (MODELS.md §12).
    pub prefer_mjpeg: bool,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self { device_index: 0, width: 1280, height: 720, fps: 30, prefer_mjpeg: true }
    }
}

// ---------------------------------------------------------------------------
// models
// ---------------------------------------------------------------------------

/// Resolved paths only. The crate never learns how Tauri resolves a resource
/// directory — the adapter resolves and passes paths in (MODELS.md §9).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelPaths {
    pub face: Option<PathBuf>,
    pub pose: Option<PathBuf>,
    pub gaze: Option<PathBuf>,
    pub objects: Option<PathBuf>,
    pub identity: Option<PathBuf>,
    /// Which precision to load when both are shipped.
    pub variant: Variant,
}

impl ModelPaths {
    /// Conventional filename for each slot, as downloaded.
    pub const CONVENTIONAL: [(&'static str, &'static str); 5] = [
        ("face", "face_detection_yunet_2023mar.onnx"),
        ("pose", "headpose_mobilenetv3_small.onnx"),
        ("gaze", "mobileone_s0_gaze.onnx"),
        ("objects", "yolox_nano.onnx"),
        ("identity", "w600k_mbf.onnx"),
    ];

    /// Fill any slot that config left unset, from a directory of models.
    ///
    /// This is a convenience for callers, not the crate resolving its own
    /// assets: an explicit path in `Config` always wins, and nothing here
    /// knows about Tauri, resource bundles or the working directory
    /// (MODELS.md §9). A binary decides *which* directory; this only knows
    /// what the files are conventionally called.
    pub fn fill_missing_from_dir(&mut self, dir: impl AsRef<Path>) {
        let dir = dir.as_ref();
        for (slot, filename) in Self::CONVENTIONAL {
            let candidate = dir.join(filename);
            if !candidate.exists() {
                continue;
            }
            let target = match slot {
                "face" => &mut self.face,
                "pose" => &mut self.pose,
                "gaze" => &mut self.gaze,
                "objects" => &mut self.objects,
                "identity" => &mut self.identity,
                _ => continue,
            };
            if target.is_none() {
                *target = Some(candidate);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Variant {
    Fp32,
    Int8,
    /// Micro-benchmark both at startup and keep the winner (MODELS.md §5.1).
    /// The result genuinely flips depending on CPU, so this is the default.
    #[default]
    Auto,
}

// ---------------------------------------------------------------------------
// cadence
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CadenceConfig {
    /// YuNet -> pose -> gaze, one worker, sequential dependency chain.
    pub face_hz: f64,
    /// A phone does not appear for 400 ms and the hold is 2 s.
    pub object_hz: f64,
    pub identity_hz: f64,
}

impl Default for CadenceConfig {
    fn default() -> Self {
        Self { face_hz: 15.0, object_hz: 1.0, identity_hz: 0.2 }
    }
}

// ---------------------------------------------------------------------------
// thresholds
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Thresholds {
    pub face: FaceThresholds,
    pub pose: PoseThresholds,
    pub gaze: GazeThresholds,
    pub objects: ObjectThresholds,
    pub identity: IdentityThresholds,
    pub fusion: FusionWeights,
    pub debug_direction: DebugDirectionThresholds,
}

/// Buckets for the temporary plain-language direction readout
/// (see [`crate::direction`]).
///
/// Deliberately its own group rather than borrowed from `pose`/`gaze`: those
/// thresholds decide *violations* and will be tuned against the clip corpus,
/// while these only decide what a debug label says. Sharing them would mean
/// tuning one silently changed the other, and would make this readout hard to
/// delete when fusion replaces it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugDirectionThresholds {
    /// Degrees away from centre before a direction is claimed.
    pub enter_deg: f64,
    /// Degrees it must fall back through before the claim is released. Lower
    /// than `enter_deg`, which is what stops the label flickering on the
    /// boundary.
    pub exit_deg: f64,
}

impl Default for DebugDirectionThresholds {
    fn default() -> Self {
        Self { enter_deg: 8.0, exit_deg: 5.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FaceThresholds {
    /// YuNet's own demo defaults to 0.9 and the old MediaPipe detector used
    /// 0.5; they are not the same scale. This is a starting point to tune
    /// against the clip corpus, not a value inherited from either.
    pub min_score: f64,
    /// IoU above which the lower-scoring box is suppressed.
    pub nms_threshold: f64,
    /// Boxes considered after sorting. A webcam frame never holds thousands
    /// of faces; this only bounds pathological cases.
    pub top_k: usize,
    /// The old 1 s (CONTEXT.md) fires on normal head movement. Raised, and
    /// paired with an explicit clear hold.
    pub no_face_hold_ms: u64,
    pub no_face_clear_ms: u64,
    /// `NeverSeen` past this is its own, more serious violation — the old
    /// module could never fire before the first face was seen at all.
    pub never_seen_ms: u64,
    pub multi_face_count: usize,
    pub multi_face_hold_ms: u64,
    pub multi_face_clear_ms: u64,
}

impl Default for FaceThresholds {
    fn default() -> Self {
        Self {
            min_score: 0.6,
            nms_threshold: 0.3,
            top_k: 200,
            no_face_hold_ms: 2500,
            no_face_clear_ms: 500,
            never_seen_ms: 10_000,
            multi_face_count: 2,
            multi_face_hold_ms: 2000,
            multi_face_clear_ms: 1000,
        }
    }
}

/// Absolute degrees from an absolute regressor — no calibration baseline is
/// subtracted, so these numbers are meaningful on their own (MODELS.md §4).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PoseThresholds {
    pub yaw_enter_deg: f64,
    pub yaw_exit_deg: f64,
    pub pitch_enter_deg: f64,
    pub pitch_exit_deg: f64,
    pub hold_ms: u64,
    pub clear_ms: u64,
    /// EMA smoothing on yaw/pitch. Lower = smoother, more lag.
    pub ema_alpha: f64,
    /// How far past the face box to crop, per side, before feeding the pose
    /// model. Head-pose models are sensitive to framing and a tight face box
    /// degrades them quietly. The reference implementation uses 0.2.
    pub crop_expand: f64,
}

impl Default for PoseThresholds {
    fn default() -> Self {
        Self {
            yaw_enter_deg: 25.0,
            yaw_exit_deg: 18.0,
            pitch_enter_deg: 20.0,
            pitch_exit_deg: 14.0,
            hold_ms: 1500,
            clear_ms: 700,
            ema_alpha: 0.35,
            crop_expand: 0.25,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GazeThresholds {
    /// One short calibration step, not two (MODELS.md §4).
    pub calibration_ms: u64,
    pub calibration_min_samples: usize,
    /// Reject and retry if the candidate moved during calibration.
    pub calibration_variance_ceiling: f64,
    /// Widen thresholds proportionally for noisy setups.
    pub variance_widening: f64,
    pub yaw_enter_rad: f64,
    pub yaw_exit_rad: f64,
    pub pitch_enter_rad: f64,
    pub pitch_exit_rad: f64,
    pub hold_ms: u64,
    pub clear_ms: u64,
    /// Below this EAR the eyes are closing — suppress gaze rather than
    /// letting a blink produce a false "gaze off" event.
    pub blink_ear_floor: f64,
    pub ema_alpha: f64,
    /// Below this face score, gaze is held rather than emitted. A coarse
    /// proxy for "the detector is struggling", which is what a blink, motion
    /// blur and a half-turned head all look like from here — YuNet has no
    /// eyelid landmarks, so a true eye-aspect-ratio is not available.
    pub min_face_score: f64,
}

impl Default for GazeThresholds {
    fn default() -> Self {
        Self {
            calibration_ms: 3000,
            calibration_min_samples: 30,
            calibration_variance_ceiling: 0.02,
            variance_widening: 1.5,
            yaw_enter_rad: 0.38,
            yaw_exit_rad: 0.26,
            pitch_enter_rad: 0.33,
            pitch_exit_rad: 0.22,
            hold_ms: 1000,
            clear_ms: 500,
            blink_ear_floor: 0.18,
            ema_alpha: 0.4,
            min_face_score: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ObjectThresholds {
    pub min_score: f64,
    /// IoU above which a lower-scoring box of the **same class** is dropped.
    /// YOLOX's own demo uses 0.45.
    pub nms_threshold: f64,
    pub hold_ms: u64,
    pub clear_ms: u64,
    /// Narrowed from the old module's five classes to two.
    ///
    /// `laptop` is the machine the exam runs on, `tv` is usually the
    /// candidate's own monitor, and `person >= 2` already duplicates the
    /// `MultipleFaces` signal from YuNet. Three of the old five were
    /// near-guaranteed false-positive sources, and a false positive is what
    /// makes a proctoring system unusable.
    pub allowlist: Vec<String>,
    pub person_count: usize,
    /// Objects are detected independently of face presence. A phone held over
    /// the face is exactly the case the old gating discarded (MODELS.md §4).
    pub require_face_present: bool,
}

impl Default for ObjectThresholds {
    fn default() -> Self {
        Self {
            min_score: 0.4,
            nms_threshold: 0.45,
            hold_ms: 2000,
            clear_ms: 1000,
            allowlist: ["cell phone", "book"].iter().map(|s| s.to_string()).collect(),
            person_count: 2,
            require_face_present: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IdentityThresholds {
    /// Similarity falling below this trips the violation.
    pub cosine_enter: f64,
    /// It has to recover past this higher bar to clear.
    pub cosine_exit: f64,
    /// Consecutive failing checks required. At 0.2 Hz each one is ~5 s.
    pub consecutive_failures: u32,
}

impl Default for IdentityThresholds {
    fn default() -> Self {
        Self { cosine_enter: 0.32, cosine_exit: 0.42, consecutive_failures: 2 }
    }
}

/// Fuse, don't OR (MODELS.md §4). Five independent booleans produce five
/// independent false-positive streams; a weighted score with the contributing
/// signals attached is both stronger evidence and reviewable.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FusionWeights {
    pub no_face: f64,
    pub never_seen: f64,
    pub multiple_faces: f64,
    pub head_turned: f64,
    pub gaze_off: f64,
    pub prohibited_object: f64,
    pub identity_drift: f64,
    /// Score at or above which a fused violation is escalated to High.
    pub high_severity_score: f64,
    /// ...and to Critical.
    pub critical_severity_score: f64,
}

impl Default for FusionWeights {
    fn default() -> Self {
        Self {
            no_face: 0.6,
            never_seen: 0.9,
            multiple_faces: 1.0,
            head_turned: 0.5,
            gaze_off: 0.45,
            prohibited_object: 0.9,
            identity_drift: 1.0,
            high_severity_score: 1.2,
            critical_severity_score: 1.8,
        }
    }
}

// ---------------------------------------------------------------------------
// runtime
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeConfig {
    /// Small models, high rate: small graphs parallelise badly and sync
    /// overhead exceeds the win. Sweep 1..cores and pick on p50/p95
    /// (MODELS.md §6 rule 2).
    pub intra_threads_small: usize,
    /// YOLO's graph is big and its rate is low, so it can use more.
    pub intra_threads_large: usize,
    pub inter_threads: usize,
    /// ORT's constant-cost parallelism model causes high latency variance;
    /// this switches to decreasing-granularity work claiming.
    pub dynamic_block_base: usize,
    pub execution_provider: ExecutionProviderPref,
    /// First inference is much slower than steady state. Without warm-up the
    /// first real frame is an outlier and calibration starts on garbage timing.
    pub warmup_iters: u32,
    /// Iterations per variant when `ModelPaths::variant` is `Auto`.
    pub variant_bench_iters: u32,
    /// Let ORT's thread pools spin-wait between inferences.
    ///
    /// Off, deliberately. Spinning is a throughput optimisation for
    /// back-to-back inference; every worker here runs on a cadence with idle
    /// gaps far longer than the work, so spinning pools just occupy cores.
    /// With it on, adding the 1 Hz object worker tripled the 15 Hz face
    /// worker's p50.
    pub allow_spinning: bool,
    /// Cap on evidence JPEGs written per minute, so the hot path never
    /// becomes an encoder.
    pub evidence_per_minute: u32,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            intra_threads_small: 2,
            intra_threads_large: 4,
            inter_threads: 1,
            dynamic_block_base: 4,
            execution_provider: ExecutionProviderPref::default(),
            warmup_iters: 5,
            variant_bench_iters: 20,
            allow_spinning: false,
            evidence_per_minute: 6,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProviderPref {
    /// DirectML first, CPU fallback. Runs on any DirectX 12 device, which
    /// includes the Intel/AMD integrated graphics an exam candidate actually
    /// has (MODELS.md §5.2).
    #[default]
    DirectMlThenCpu,
    CpuOnly,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn default_config_survives_a_toml_roundtrip() {
        let text = Config::default().to_toml().unwrap();
        let back: Config = toml::from_str(&text).unwrap();
        back.validate().unwrap();
        assert_eq!(back.cadence.face_hz, 15.0);
        // Narrowed to cell phone + book; see ObjectThresholds::allowlist.
        assert_eq!(back.thresholds.objects.allowlist.len(), 2);
    }

    #[test]
    fn partial_toml_fills_the_rest_from_defaults() {
        // Tuning a single number must not require restating the whole file.
        let cfg: Config = toml::from_str("[thresholds.pose]\nyaw_enter_deg = 30.0\n").unwrap();
        assert_eq!(cfg.thresholds.pose.yaw_enter_deg, 30.0);
        assert_eq!(cfg.thresholds.pose.yaw_exit_deg, 18.0);
        assert_eq!(cfg.capture.width, 1280);
    }

    #[test]
    fn inverted_hysteresis_is_rejected() {
        let mut cfg = Config::default();
        cfg.thresholds.pose.yaw_exit_deg = 40.0;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("pose.yaw_exit"), "unhelpful message: {err}");
    }

    #[test]
    fn inverted_identity_hysteresis_is_rejected() {
        let mut cfg = Config::default();
        cfg.thresholds.identity.cosine_exit = 0.1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn zero_cadence_is_rejected() {
        let mut cfg = Config::default();
        cfg.cadence.object_hz = 0.0;
        assert!(cfg.validate().is_err());
    }
}
