//! Time-domain primitives: smoothing, hysteresis, hold timers.
//!
//! Three small pieces, each doing one thing, each testable without a camera, a
//! model or a clock. Every one of them takes time as a **parameter**. Nothing
//! in this module reads the system clock, which is what lets a recorded
//! session replay to a byte-identical event sequence on any machine at any
//! speed.
//!
//! They exist separately because the old module inlined all three behaviours
//! into each rule, so "does gaze flap on the boundary?" and "does head pose
//! flap on the boundary?" were separate questions with separate answers. Here
//! there is one implementation of each and it is tested once.

use serde::{Deserialize, Serialize};

/// Exponential moving average over a signal that can disappear.
///
/// The interesting part is not the smoothing, it is the gaps. Pose and gaze
/// are absent on roughly 1.7% of frames (§18.3) — a blink, a turned head, a
/// frame with no face. Smoothing straight across a gap treats the value before
/// it and the value after it as adjacent samples, which quietly invents a
/// trajectory that was never measured.
///
/// **Decision: gaps longer than `reset_after_ms` reset the filter.** The next
/// sample is adopted whole rather than blended with a stale one. Shorter gaps
/// are blended normally, because a single dropped frame really is
/// "approximately continuous" and resetting on every blink would throw away
/// the smoothing entirely.
///
/// The alternative — decaying the old value toward nothing — was rejected
/// because it produces a number that is neither the last measurement nor the
/// current one, and every consumer would have to know that.
#[derive(Debug, Clone)]
pub struct Ema {
    alpha: f64,
    reset_after_ms: u64,
    value: Option<f64>,
    last_t_ms: Option<u64>,
}

impl Ema {
    pub fn new(alpha: f64, reset_after_ms: u64) -> Self {
        Self { alpha, reset_after_ms, value: None, last_t_ms: None }
    }

    /// Feed a measurement taken at `t_ms`. Returns the smoothed value.
    pub fn update(&mut self, sample: f64, t_ms: u64) -> f64 {
        let stale = match self.last_t_ms {
            // Replay and live both hand timestamps in monotonically, but
            // `saturating_sub` means a non-monotonic one degrades to "no gap"
            // rather than to a panic or a wildly wrong gap.
            Some(prev) => t_ms.saturating_sub(prev) > self.reset_after_ms,
            None => true,
        };
        let next = match self.value {
            Some(prev) if !stale => self.alpha * sample + (1.0 - self.alpha) * prev,
            _ => sample,
        };
        self.value = Some(next);
        self.last_t_ms = Some(t_ms);
        next
    }

    /// Note that no measurement arrived at `t_ms`.
    ///
    /// Deliberately does **not** advance `last_t_ms`: the gap is measured from
    /// the last real sample, so a long absence is still a long absence however
    /// many times it is observed.
    pub fn miss(&mut self, _t_ms: u64) {}

    /// The current smoothed value, or `None` if nothing has been fed yet.
    pub fn get(&self) -> Option<f64> {
        self.value
    }

    /// Forget everything. Used when a signal's slot goes `NotConfigured` or a
    /// session restarts.
    pub fn reset(&mut self) {
        self.value = None;
        self.last_t_ms = None;
    }
}

/// A two-threshold latch.
///
/// One threshold flaps: a value sitting on the boundary crosses it many times
/// a second and every crossing is an event. Enter high, leave low, and the
/// boundary has to actually be traversed before anything changes.
///
/// Deliberately has no notion of time — combining hysteresis and hold timing
/// in one type is how the old module ended up unable to say which of the two
/// was responsible for a missed event.
#[derive(Debug, Clone)]
pub struct Hysteresis {
    enter: f64,
    exit: f64,
    active: bool,
}

impl Hysteresis {
    /// `exit` must be at or below `enter`; `Config::validate` enforces it, and
    /// this asserts the same in debug builds so a hand-built test cannot get
    /// it backwards silently.
    pub fn new(enter: f64, exit: f64) -> Self {
        debug_assert!(exit <= enter, "hysteresis exit ({exit}) must be <= enter ({enter})");
        Self { enter, exit, active: false }
    }

    /// Feed a magnitude. Returns whether the latch is now set.
    pub fn update(&mut self, value: f64) -> bool {
        if self.active {
            if value < self.exit {
                self.active = false;
            }
        } else if value >= self.enter {
            self.active = true;
        }
        self.active
    }

    pub fn active(&self) -> bool {
        self.active
    }

    /// How far past the enter threshold the value sits, on [0, 1].
    ///
    /// Zero below the threshold, ramping to one at twice it. This is the
    /// per-rule contribution that Part 3's scoring fuses — a rule that only
    /// ever answered "yes" or "no" could not distinguish a head barely past
    /// the line from one turned completely away.
    pub fn intensity(&self, value: f64) -> f32 {
        if value < self.enter || self.enter <= 0.0 {
            return 0.0;
        }
        (((value - self.enter) / self.enter) as f32).clamp(0.0, 1.0)
    }

    pub fn reset(&mut self) {
        self.active = false;
    }
}

/// What a [`HoldTimer`] did on one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Edge {
    /// Held long enough; this is the rising edge and fires exactly once.
    Rose,
    /// Absent long enough; falling edge, fires exactly once.
    Fell,
    /// No change worth reporting.
    Steady,
}

/// A condition must persist before it counts, and stop persisting before it
/// stops counting.
///
/// Both directions matter and the old module only had the first. Without a
/// clear hold, a candidate who leaves frame and whose face is redetected for a
/// single frame mid-absence produces two violations instead of one.
///
/// Time arrives as a parameter. There is no `Instant` in this type.
#[derive(Debug, Clone)]
pub struct HoldTimer {
    hold_ms: u64,
    clear_ms: u64,
    fired: bool,
    /// When the condition first became true in its current run, or `None` if
    /// it is currently false.
    true_since: Option<u64>,
    /// Likewise for false.
    false_since: Option<u64>,
}

impl HoldTimer {
    pub fn new(hold_ms: u64, clear_ms: u64) -> Self {
        Self { hold_ms, clear_ms, fired: false, true_since: None, false_since: None }
    }

    /// Advance to `t_ms` with the condition currently `condition`.
    pub fn update(&mut self, condition: bool, t_ms: u64) -> Edge {
        if condition {
            self.false_since = None;
            let since = *self.true_since.get_or_insert(t_ms);
            if !self.fired && t_ms.saturating_sub(since) >= self.hold_ms {
                self.fired = true;
                return Edge::Rose;
            }
        } else {
            self.true_since = None;
            let since = *self.false_since.get_or_insert(t_ms);
            if self.fired && t_ms.saturating_sub(since) >= self.clear_ms {
                self.fired = false;
                return Edge::Fell;
            }
        }
        Edge::Steady
    }

    /// Whether the timer is currently latched on.
    pub fn fired(&self) -> bool {
        self.fired
    }

    /// Force the latch off, reporting whether that was a falling edge. Used
    /// when a session ends with violations still open.
    pub fn force_clear(&mut self) -> Edge {
        self.true_since = None;
        self.false_since = None;
        if std::mem::take(&mut self.fired) {
            Edge::Fell
        } else {
            Edge::Steady
        }
    }
}

/// A decaying sum of evidence.
///
/// §18.5 is the reason this exists. A phone plainly in shot cleared a 0.5
/// per-sample threshold on only 26–42% of frames, so thresholding any single
/// sample misses roughly half the seconds the phone is there. Accumulating
/// instead: every sample above the floor adds its confidence, the total decays
/// with a half-life, and the violation fires on the total.
///
/// Peaky-but-persistent crosses; one loud frame does not.
#[derive(Debug, Clone)]
pub struct DecayingScore {
    half_life_ms: u64,
    score: f64,
    last_t_ms: Option<u64>,
}

impl DecayingScore {
    pub fn new(half_life_ms: u64) -> Self {
        Self { half_life_ms, score: 0.0, last_t_ms: None }
    }

    /// Decay to `t_ms` without adding anything.
    pub fn decay_to(&mut self, t_ms: u64) -> f64 {
        if let Some(prev) = self.last_t_ms {
            let dt = t_ms.saturating_sub(prev) as f64;
            if dt > 0.0 && self.half_life_ms > 0 {
                self.score *= 0.5f64.powf(dt / self.half_life_ms as f64);
            }
        }
        self.last_t_ms = Some(t_ms);
        self.score
    }

    /// Decay to `t_ms`, then add `amount`.
    pub fn add(&mut self, amount: f64, t_ms: u64) -> f64 {
        self.decay_to(t_ms);
        self.score += amount;
        self.score
    }

    pub fn score(&self) -> f64 {
        self.score
    }

    pub fn reset(&mut self) {
        self.score = 0.0;
        self.last_t_ms = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_smooths_toward_a_step() {
        let mut ema = Ema::new(0.5, 1000);
        assert_eq!(ema.update(10.0, 0), 10.0, "the first sample is adopted whole");
        assert_eq!(ema.update(20.0, 100), 15.0);
        assert_eq!(ema.update(20.0, 200), 17.5);
        assert!(ema.get().unwrap() < 20.0, "smoothing lags, by design");
    }

    #[test]
    fn ema_does_not_interpolate_across_a_long_gap() {
        // The property the doc comment promises: a value measured before a
        // long absence must not be blended with one measured after it. Those
        // are two different moments, not two adjacent samples.
        let mut ema = Ema::new(0.5, 500);
        ema.update(10.0, 0);
        let after_gap = ema.update(30.0, 5_000);
        assert_eq!(
            after_gap, 30.0,
            "a 5 s gap must reset the filter; blending would report {} — a \
             value that was never measured",
            0.5 * 30.0 + 0.5 * 10.0
        );
    }

    #[test]
    fn ema_blends_across_a_short_gap() {
        // A single dropped frame at 15 Hz is ~67 ms. That really is
        // approximately continuous, and resetting on every blink would discard
        // the smoothing altogether.
        let mut ema = Ema::new(0.5, 500);
        ema.update(10.0, 0);
        assert_eq!(ema.update(30.0, 67), 20.0);
    }

    #[test]
    fn a_miss_does_not_restart_the_gap_clock() {
        // Observing an absence repeatedly must not make the absence look
        // shorter than it is.
        let mut ema = Ema::new(0.5, 500);
        ema.update(10.0, 0);
        for t in [100, 200, 300, 400, 500, 600] {
            ema.miss(t);
        }
        assert_eq!(ema.update(30.0, 700), 30.0, "700 ms of misses is still a 700 ms gap");
    }

    #[test]
    fn hysteresis_crosses_once_on_a_monotonic_ramp() {
        let mut h = Hysteresis::new(25.0, 18.0);
        let mut transitions = 0;
        let mut last = false;
        for i in 0..100 {
            let now = h.update(i as f64 * 0.5);
            if now != last {
                transitions += 1;
            }
            last = now;
        }
        assert_eq!(transitions, 1, "a monotonic ramp must cross exactly once");
    }

    #[test]
    fn hysteresis_does_not_flap_on_the_boundary() {
        // This is the whole reason the type exists. A value dithering either
        // side of `enter` produces one event, not dozens.
        let mut h = Hysteresis::new(25.0, 18.0);
        assert!(h.update(26.0), "crosses in");
        for v in [24.9, 25.1, 24.0, 25.5, 19.0, 24.0] {
            assert!(h.update(v), "still latched at {v} — above the exit threshold");
        }
        assert!(!h.update(17.9), "only releases below the exit threshold");
    }

    #[test]
    fn hysteresis_intensity_scales_past_the_threshold() {
        let h = Hysteresis::new(20.0, 14.0);
        assert_eq!(h.intensity(10.0), 0.0, "below the threshold contributes nothing");
        assert_eq!(h.intensity(20.0), 0.0, "exactly at the threshold is the bottom of the ramp");
        assert!((h.intensity(30.0) - 0.5).abs() < 1e-6);
        assert_eq!(h.intensity(60.0), 1.0, "saturates rather than growing without bound");
    }

    #[test]
    fn hold_timer_waits_for_the_hold_then_fires_once() {
        let mut t = HoldTimer::new(2500, 1000);
        assert_eq!(t.update(true, 0), Edge::Steady);
        assert_eq!(t.update(true, 2499), Edge::Steady, "not yet");
        assert_eq!(t.update(true, 2500), Edge::Rose);
        for ms in [2600, 3000, 9000] {
            assert_eq!(t.update(true, ms), Edge::Steady, "must not re-emit every frame");
        }
    }

    #[test]
    fn hold_timer_needs_the_clear_hold_before_releasing() {
        let mut t = HoldTimer::new(1000, 1000);
        t.update(true, 0);
        assert_eq!(t.update(true, 1000), Edge::Rose);
        assert_eq!(t.update(false, 1100), Edge::Steady, "a momentary reappearance is not a clear");
        assert_eq!(t.update(true, 1200), Edge::Steady, "and it does not re-fire either");
        assert_eq!(t.update(false, 1300), Edge::Steady);
        assert_eq!(t.update(false, 2300), Edge::Fell);
    }

    #[test]
    fn a_flicker_mid_absence_does_not_produce_two_violations() {
        // The concrete bug this prevents: someone leaves frame, one frame
        // happens to redetect them, and the session records two separate
        // absences instead of one.
        let mut t = HoldTimer::new(1000, 1000);
        let mut rises = 0;
        let mut falls = 0;
        for step in 0..100u64 {
            let ms = step * 100;
            // Absent throughout, except one lone frame at 3 s.
            let present = ms == 3000;
            match t.update(!present, ms) {
                Edge::Rose => rises += 1,
                Edge::Fell => falls += 1,
                Edge::Steady => {}
            }
        }
        assert_eq!((rises, falls), (1, 0), "one absence, still ongoing");
    }

    #[test]
    fn decaying_score_halves_over_one_half_life() {
        let mut s = DecayingScore::new(3000);
        s.add(1.0, 0);
        assert!((s.decay_to(3000) - 0.5).abs() < 1e-9);
        assert!((s.decay_to(6000) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn peaky_but_persistent_evidence_accumulates_past_the_bar() {
        // §18.5's phone: sampled at 1 Hz, present throughout, scores that only
        // sometimes look convincing. Must cross an enter bar of 1.5.
        let mut s = DecayingScore::new(3000);
        let samples = [0.62, 0.30, 0.71, 0.28, 0.80, 0.33, 0.77];
        let mut peak: f64 = 0.0;
        for (i, v) in samples.iter().enumerate() {
            peak = peak.max(s.add(*v, i as u64 * 1000));
        }
        assert!(peak >= 1.5, "peaky-but-persistent evidence must fire; peaked at {peak:.2}");
    }

    #[test]
    fn one_noisy_sample_decays_without_firing() {
        // The other half of the same property, and the one that keeps the
        // system usable: a single spurious detection must never fire.
        let mut s = DecayingScore::new(3000);
        s.add(0.9, 0);
        let mut peak: f64 = s.score();
        for step in 1..30 {
            peak = peak.max(s.decay_to(step * 1000));
        }
        assert!(peak < 1.5, "a lone sample must not reach the bar; peaked at {peak:.2}");
        assert!(s.score() < 0.01, "and it must fade to nothing");
    }
}
