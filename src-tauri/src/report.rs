//! Session report (MODELS.md §8, §11).
//!
//! A proctor reading a clean report must be able to tell the difference
//! between "no violations detected" and "the object detector was never
//! running". That is a correctness requirement for the product, not hygiene,
//! so per-signal liveness is part of the report type itself rather than
//! something a caller is trusted to log.

use std::collections::BTreeMap;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::types::{DegradeReason, Violation};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionReport {
    pub started_at: SystemTime,
    pub duration_ms: u64,
    /// What the session was actually looking at — `camera:0`, `file:clip.mp4`.
    pub source: String,
    pub frames: FrameStats,
    /// Keyed by signal name: face, pose, gaze, objects, identity.
    pub signals: BTreeMap<String, SignalStatus>,
    pub violations: Vec<Violation>,
    pub degraded: Vec<DegradeReason>,
    /// The resolved config, dumped verbatim. When a session produces odd
    /// results you need to know exactly which thresholds were live.
    pub config: Config,
}

impl SessionReport {
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// True when any signal the config asked for was not running for the whole
    /// session — i.e. when "no violations" cannot be read at face value.
    pub fn has_coverage_gap(&self) -> bool {
        self.signals.values().any(|s| !s.active || s.active_fraction < 0.99)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FrameStats {
    pub captured: u64,
    pub processed: u64,
    /// Frames the workers never saw because a newer one had already replaced
    /// them in the bus. Expected and healthy; a spike means saturation.
    pub skipped: u64,
    /// Ticks where the bus held nothing new.
    pub duplicate: u64,
    pub mean_fps: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SignalStatus {
    pub active: bool,
    /// Fraction of the session this signal was producing values.
    pub active_fraction: f32,
    pub frames_processed: u64,
    /// Which EP this model actually ran on. A silent DirectML fallback looks
    /// exactly like "the GPU didn't help" unless this is surfaced
    /// (MODELS.md §5.2).
    pub execution_provider: String,
    pub latency: Option<LatencySummary>,
}

/// Per-stage timings, in microseconds. Kept separate because preprocessing is
/// not free at 15 Hz and lumping it into "inference" hides the cost that is
/// actually easiest to remove (MODELS.md §6 rule 4).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LatencySummary {
    pub p50_us: u64,
    pub p95_us: u64,
    pub max_us: u64,
    pub mean_us: u64,
    pub samples: u64,
}

/// Accumulates raw samples and reduces them to percentiles.
///
/// p95, not just the median: ORT's constant-cost parallelism model produces
/// high latency variance, and the tail is what a candidate actually
/// experiences as a stutter.
#[derive(Debug, Default, Clone)]
pub struct Latencies {
    samples: std::collections::VecDeque<u64>,
    /// When set, the oldest sample is dropped once this many are held. A
    /// long-running session would otherwise accumulate a sample per frame
    /// forever, and `summary()` sorts a copy every time it is polled.
    window: Option<usize>,
}

impl Latencies {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(n: usize) -> Self {
        Self { samples: std::collections::VecDeque::with_capacity(n), window: None }
    }

    /// Keep only the most recent `n` samples — a rolling view, which is what
    /// a live HUD wants. Percentiles then describe recent behaviour rather
    /// than the whole session average.
    pub fn rolling(n: usize) -> Self {
        Self { samples: std::collections::VecDeque::with_capacity(n), window: Some(n) }
    }

    pub fn record_us(&mut self, us: u64) {
        if let Some(window) = self.window {
            while self.samples.len() >= window {
                self.samples.pop_front();
            }
        }
        self.samples.push_back(us);
    }

    pub fn record(&mut self, d: std::time::Duration) {
        self.record_us(d.as_micros() as u64);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn summary(&self) -> Option<LatencySummary> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted: Vec<u64> = self.samples.iter().copied().collect();
        sorted.sort_unstable();
        let sum: u64 = sorted.iter().sum();
        Some(LatencySummary {
            p50_us: percentile(&sorted, 0.50),
            p95_us: percentile(&sorted, 0.95),
            max_us: *sorted.last().unwrap(),
            mean_us: sum / sorted.len() as u64,
            samples: sorted.len() as u64,
        })
    }
}

/// Nearest-rank percentile over an ascending slice.
fn percentile(sorted: &[u64], q: f32) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (q * sorted.len() as f32).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_on_a_known_distribution() {
        let mut l = Latencies::new();
        for v in 1..=100u64 {
            l.record_us(v);
        }
        let s = l.summary().unwrap();
        assert_eq!(s.p50_us, 50);
        assert_eq!(s.p95_us, 95);
        assert_eq!(s.max_us, 100);
        assert_eq!(s.samples, 100);
    }

    #[test]
    fn single_sample_is_its_own_every_percentile() {
        let mut l = Latencies::new();
        l.record_us(7);
        let s = l.summary().unwrap();
        assert_eq!((s.p50_us, s.p95_us, s.max_us), (7, 7, 7));
    }

    #[test]
    fn empty_summarises_to_nothing_rather_than_zero() {
        // Zeroes would read as "instant", which is worse than absent.
        assert!(Latencies::new().summary().is_none());
    }

    #[test]
    fn a_disabled_signal_counts_as_a_coverage_gap() {
        let report = SessionReport {
            started_at: SystemTime::UNIX_EPOCH,
            duration_ms: 1000,
            source: "file:x.mp4".into(),
            frames: FrameStats::default(),
            signals: BTreeMap::from([(
                "objects".to_string(),
                SignalStatus { active: false, ..Default::default() },
            )]),
            violations: vec![],
            degraded: vec![],
            config: Config::default(),
        };
        assert!(report.has_coverage_gap());
    }
}
