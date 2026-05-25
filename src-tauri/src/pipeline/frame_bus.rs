//! Latest-frame slot (MODELS.md §6 rule 3).
//!
//! **Never a queue.** A stale frame is worthless: by the time a worker gets to
//! a frame that has been sitting behind three others, the candidate has moved
//! and the decision would be about the past. So capture overwrites a single
//! slot, and each worker reads whatever is there when it ticks.
//!
//! The consequence is that frames are dropped, on purpose, whenever detection
//! is slower than capture — which it always is. That is the design, not a
//! failure, and the drop count is exposed so saturation stays visible.
//!
//! `Arc<[u8]>` inside `Frame` means the capture thread decodes once and every
//! reader shares those bytes with zero copies.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::types::Frame;

pub struct FrameBus {
    slot: ArcSwap<Frame>,
    captured: AtomicU64,
}

impl FrameBus {
    pub fn new() -> Self {
        Self {
            // The sentinel has seq 0 and no pixels. Readers start with
            // `last_seen = 0`, so it is never handed out as a real frame.
            slot: ArcSwap::from_pointee(Frame::empty()),
            captured: AtomicU64::new(0),
        }
    }

    /// Called by the capture thread. Overwrites whatever was there.
    pub fn publish(&self, frame: Frame) {
        self.slot.store(Arc::new(frame));
        self.captured.fetch_add(1, Ordering::Relaxed);
    }

    /// Whatever is in the slot right now, new or not.
    pub fn latest(&self) -> Arc<Frame> {
        self.slot.load_full()
    }

    /// The current frame, but only if it is newer than what this reader last
    /// saw. Returns how many frames were missed in between, which is the
    /// skip count the caller accumulates.
    ///
    /// Each worker owns its own `last_seen`, so workers at different cadences
    /// skip independently.
    pub fn take_new(&self, last_seen: &mut u64) -> Option<(Arc<Frame>, u64)> {
        let frame = self.slot.load_full();
        if frame.seq <= *last_seen || frame.is_empty() {
            return None;
        }
        // Everything strictly between the last frame we processed and this one
        // was published and then overwritten before we got to it.
        let missed = frame.seq - *last_seen - 1;
        *last_seen = frame.seq;
        Some((frame, missed))
    }

    pub fn captured(&self) -> u64 {
        self.captured.load(Ordering::Relaxed)
    }
}

impl Default for FrameBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn frame(seq: u64) -> Frame {
        Frame {
            data: Arc::from(vec![0u8; 4 * 2 * 3].as_slice()),
            width: 4,
            height: 2,
            seq,
            captured_at: Instant::now(),
        }
    }

    #[test]
    fn empty_bus_hands_out_nothing() {
        let bus = FrameBus::new();
        let mut last_seen = 0;
        assert!(bus.take_new(&mut last_seen).is_none(), "the sentinel is not a frame");
        assert_eq!(bus.captured(), 0);
    }

    #[test]
    fn a_new_frame_is_taken_exactly_once() {
        let bus = FrameBus::new();
        let mut last_seen = 0;

        bus.publish(frame(1));
        let (f, missed) = bus.take_new(&mut last_seen).expect("frame 1 is new");
        assert_eq!(f.seq, 1);
        assert_eq!(missed, 0);

        // Same frame still in the slot — a second tick must not reprocess it.
        assert!(bus.take_new(&mut last_seen).is_none(), "duplicate seq must be skipped");
    }

    #[test]
    fn publishing_overwrites_and_reports_what_was_missed() {
        let bus = FrameBus::new();
        let mut last_seen = 0;

        // Capture runs ahead while the reader is busy inferring.
        for seq in 1..=5 {
            bus.publish(frame(seq));
        }

        let (f, missed) = bus.take_new(&mut last_seen).unwrap();
        assert_eq!(f.seq, 5, "the reader must get the newest frame, not the oldest");
        assert_eq!(missed, 4, "frames 1-4 were overwritten before being seen");
        assert_eq!(bus.captured(), 5);
    }

    #[test]
    fn two_readers_skip_independently() {
        // A 15 Hz worker and a 1 Hz worker must not interfere: taking a frame
        // is not consuming it.
        let bus = FrameBus::new();
        let (mut fast, mut slow) = (0, 0);

        bus.publish(frame(1));
        assert!(bus.take_new(&mut fast).is_some());
        bus.publish(frame(2));
        assert!(bus.take_new(&mut fast).is_some());

        let (f, missed) = bus.take_new(&mut slow).unwrap();
        assert_eq!(f.seq, 2);
        assert_eq!(missed, 1, "the slow reader missed frame 1, the fast one did not");
    }

    #[test]
    fn readers_share_pixels_without_copying() {
        let bus = FrameBus::new();
        bus.publish(frame(1));
        let a = bus.latest();
        let b = bus.latest();
        // Two readers, one buffer: the guarantee is that handing a frame out
        // does not copy its pixels.
        assert_eq!(a.data.as_ptr(), b.data.as_ptr());
        assert!(Arc::ptr_eq(&a, &b));
    }
}
