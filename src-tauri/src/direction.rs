//! Plain-language direction labels for the live HUD.
//!
//! **This is a validation aid and it is temporary.** It exists to answer one
//! question that raw angles cannot answer at a glance: when the candidate looks
//! left, does the signal actually say left? Reading `yaw -0.7` off a HUD and
//! deciding whether that is correct requires holding a sign convention in your
//! head; reading `GAZE LEFT` does not.
//!
//! When fusion lands it must drive this readout from its own smoothed,
//! hysteresis-gated state, and this module goes away. Two independent answers
//! to "is the gaze off" is exactly the `CONTEXT.md` §11 failure — the same
//! constant living in three places with three different values.
//!
//! # Frame of reference
//!
//! Every label here is **from the subject's point of view**. "RIGHT" means the
//! candidate turned toward their own right shoulder, not that they moved toward
//! the right-hand side of the picture. Those are opposite directions in an
//! unmirrored camera feed, which is why [`FrameOfReference`] travels in the
//! payload rather than being assumed by whatever draws it.
//!
//! # Which way is positive
//!
//! Both models were ported with their authors' own drawing code, and that code
//! is what pins the convention:
//!
//! - The head-pose gizmo projects its nose axis to `size * sin(-yaw_deg)`
//!   (`draw_axis`, negation included). Positive yaw therefore draws the nose
//!   toward **smaller x**, i.e. the left of the image.
//! - The gaze ray projects to `dx = -sin(yaw) * cos(pitch)`. Positive yaw again
//!   points toward the left of the image.
//! - Both project positive pitch to **negative y**, which is up.
//!
//! The two agree, which is the precondition for `eye = gaze - head` meaning
//! anything at all.
//!
//! An unmirrored camera shows the subject as though you were facing them, so
//! the left of the image is the subject's own right. Hence:
//!
//! ```text
//! yaw   > 0  ->  subject's RIGHT  (drawn toward the left of the picture)
//! pitch > 0  ->  UP
//! ```
//!
//! If the preview is ever mirrored, the *display* changes and these labels do
//! not: they describe the person, not the picture. The viewer reports its own
//! mirroring state separately so the two can never be silently confused.

use serde::{Deserialize, Serialize};

use crate::config::DebugDirectionThresholds;
use crate::types::{Gaze, HeadPose};

/// Horizontal direction from the subject's point of view.
///
/// Serialized in upper case because the HUD prints it verbatim — the front end
/// does no mapping, which keeps the sign convention in exactly one language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Horizontal {
    Left,
    Center,
    Right,
}

/// Vertical direction. Up is up for everyone, so this one has no ambiguity to
/// resolve — it is separate from [`Horizontal`] only so a yaw value can never
/// be bucketed into `UP` by a transposed argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Vertical {
    Up,
    Center,
    Down,
}

/// Whose left is "LEFT".
///
/// Both variants exist even though this crate only ever emits [`Self::Subject`],
/// because naming the alternative is what stops a reader assuming there isn't
/// one. It is carried in the payload and printed in the UI header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameOfReference {
    /// The candidate's own left and right.
    #[serde(rename = "subject POV")]
    Subject,
    /// Left and right as they appear on the displayed picture.
    #[serde(rename = "screen POV")]
    Screen,
}

/// One signal bucketed on both axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Axes {
    pub horizontal: Horizontal,
    pub vertical: Vertical,
}

/// The whole readout for one frame.
///
/// Each row is `None` when the signal behind it did not run for this frame,
/// which the UI shows as a dash. `None` and `CENTER` mean very different things
/// and must not render the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugDirections {
    /// Where the head is pointing.
    pub head: Option<Axes>,
    /// Where the eyes are pointing in total — head rotation included.
    pub gaze: Option<Axes>,
    /// Eye-in-head: where the eyes point *within their sockets*. `None` until
    /// head pose is available for the same frame, because it is a difference of
    /// the two and a missing operand is not a zero.
    pub eye: Option<Axes>,
    pub frame_of_reference: FrameOfReference,
}

/// Hysteresis on one axis.
///
/// A plain threshold makes the label flicker every frame while the angle sits
/// on the boundary, which reads as a broken detector when it is a correctly
/// working one. Entering a direction takes a larger angle than staying in it.
#[derive(Debug, Clone, Copy, Default)]
struct AxisTracker {
    /// `-1`, `0`, `+1`. Persists across frames — that persistence *is* the
    /// hysteresis.
    state: i8,
}

impl AxisTracker {
    fn update(&mut self, deg: f32, enter: f32, exit: f32) -> i8 {
        // What the angle would say with no history at all.
        let fresh = if deg >= enter {
            1
        } else if deg <= -enter {
            -1
        } else {
            0
        };

        self.state = match self.state {
            // Nothing latched: only a full `enter` crossing commits.
            0 => fresh,
            // Latched positive: hold while still past the lower `exit` bar.
            // A hard swing to the far side wins immediately rather than
            // detouring through `CENTER` for a frame.
            1 if deg >= exit => 1,
            1 => fresh,
            // Latched negative, mirrored.
            _ if deg <= -exit => -1,
            _ => fresh,
        };
        self.state
    }
}

/// Per-session state for the readout. Lives on the detect worker, which is the
/// only thread that writes it.
#[derive(Debug, Clone)]
pub struct DirectionTracker {
    head_yaw: AxisTracker,
    head_pitch: AxisTracker,
    gaze_yaw: AxisTracker,
    gaze_pitch: AxisTracker,
    eye_yaw: AxisTracker,
    eye_pitch: AxisTracker,
    enter_deg: f32,
    exit_deg: f32,
}

impl DirectionTracker {
    pub fn new(t: &DebugDirectionThresholds) -> Self {
        Self {
            head_yaw: AxisTracker::default(),
            head_pitch: AxisTracker::default(),
            gaze_yaw: AxisTracker::default(),
            gaze_pitch: AxisTracker::default(),
            eye_yaw: AxisTracker::default(),
            eye_pitch: AxisTracker::default(),
            enter_deg: t.enter_deg as f32,
            exit_deg: t.exit_deg as f32,
        }
    }

    /// Bucket one frame.
    ///
    /// When a signal is absent the corresponding tracker is left untouched
    /// rather than reset, so a single dropped pose estimate does not make a
    /// steadily-held head snap back to `CENTER` and then have to re-cross the
    /// enter threshold. The row reports `None` for that frame regardless.
    pub fn update(&mut self, pose: Option<HeadPose>, gaze: Option<Gaze>) -> DebugDirections {
        let (enter, exit) = (self.enter_deg, self.exit_deg);

        let head = pose.map(|p| Axes {
            horizontal: horizontal(self.head_yaw.update(p.yaw_deg, enter, exit)),
            vertical: vertical(self.head_pitch.update(p.pitch_deg, enter, exit)),
        });

        // Gaze is radians and pose is degrees. Converting here, once, is the
        // whole reason this is not done in the front end — `CONTEXT.md` §19
        // item 17 is a unit mismatch that survived every smoke test.
        let gaze_axes = gaze.map(|g| Axes {
            horizontal: horizontal(self.gaze_yaw.update(g.yaw_rad.to_degrees(), enter, exit)),
            vertical: vertical(self.gaze_pitch.update(g.pitch_rad.to_degrees(), enter, exit)),
        });

        let eye = gaze.and_then(|g| match (g.eye_yaw_rad, g.eye_pitch_rad) {
            (Some(yaw), Some(pitch)) => Some(Axes {
                horizontal: horizontal(self.eye_yaw.update(yaw.to_degrees(), enter, exit)),
                vertical: vertical(self.eye_pitch.update(pitch.to_degrees(), enter, exit)),
            }),
            _ => None,
        });

        DebugDirections {
            head,
            gaze: gaze_axes,
            eye,
            frame_of_reference: FrameOfReference::Subject,
        }
    }
}

/// Positive yaw is the subject's right. See the module docs for why — this one
/// line is the entire sign convention and everything else follows from it.
fn horizontal(state: i8) -> Horizontal {
    match state {
        1 => Horizontal::Right,
        -1 => Horizontal::Left,
        _ => Horizontal::Center,
    }
}

/// Positive pitch is up.
fn vertical(state: i8) -> Vertical {
    match state {
        1 => Vertical::Up,
        -1 => Vertical::Down,
        _ => Vertical::Center,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker() -> DirectionTracker {
        DirectionTracker::new(&DebugDirectionThresholds::default())
    }

    fn pose(yaw: f32, pitch: f32) -> Option<HeadPose> {
        Some(HeadPose { yaw_deg: yaw, pitch_deg: pitch, roll_deg: 0.0 })
    }

    /// Gaze built from degrees, so the tests read in the same units as the
    /// thresholds even though the type is radians.
    fn gaze_deg(yaw: f32, pitch: f32) -> Option<Gaze> {
        Some(Gaze {
            yaw_rad: yaw.to_radians(),
            pitch_rad: pitch.to_radians(),
            eye_yaw_rad: None,
            eye_pitch_rad: None,
        })
    }

    #[test]
    fn positive_yaw_is_the_subjects_right_and_positive_pitch_is_up() {
        // The entire sign convention, pinned. If someone flips these to "fix"
        // a readout that looked backwards, this fails and points them at the
        // module docs and the two drawing functions instead.
        let mut t = tracker();
        let d = t.update(pose(30.0, 20.0), None).head.unwrap();
        assert_eq!(d.horizontal, Horizontal::Right);
        assert_eq!(d.vertical, Vertical::Up);

        let mut t = tracker();
        let d = t.update(pose(-30.0, -20.0), None).head.unwrap();
        assert_eq!(d.horizontal, Horizontal::Left);
        assert_eq!(d.vertical, Vertical::Down);
    }

    #[test]
    fn small_angles_are_centre() {
        let mut t = tracker();
        let d = t.update(pose(3.0, -4.0), None).head.unwrap();
        assert_eq!(d.horizontal, Horizontal::Center);
        assert_eq!(d.vertical, Vertical::Center);
    }

    #[test]
    fn gaze_is_read_as_radians_not_degrees() {
        // The trap this exists to catch: 0.3 is a substantial angle in radians
        // (17 degrees) and a rounding error in degrees. Skipping the
        // conversion would bucket a clear look to the side as CENTER, and the
        // HUD would look calm and considered while being wrong.
        let mut t = tracker();
        let g = Some(Gaze {
            yaw_rad: 0.3,
            pitch_rad: 0.0,
            eye_yaw_rad: None,
            eye_pitch_rad: None,
        });
        assert_eq!(t.update(None, g).gaze.unwrap().horizontal, Horizontal::Right);
    }

    #[test]
    fn a_label_does_not_flicker_while_the_angle_sits_on_the_boundary() {
        // 8 degrees to enter, 5 to leave. An angle wobbling between 6 and 9
        // crosses the enter threshold once and must then stay put — without
        // hysteresis this alternates RIGHT/CENTER every frame and reads as a
        // broken detector.
        let mut t = tracker();
        assert_eq!(
            t.update(pose(9.0, 0.0), None).head.unwrap().horizontal,
            Horizontal::Right
        );
        for deg in [6.0, 9.0, 5.5, 8.5, 6.2, 7.0] {
            assert_eq!(
                t.update(pose(deg, 0.0), None).head.unwrap().horizontal,
                Horizontal::Right,
                "released at {deg} degrees, which is still above the exit bar"
            );
        }
        // Below the exit bar it finally lets go.
        assert_eq!(
            t.update(pose(4.0, 0.0), None).head.unwrap().horizontal,
            Horizontal::Center
        );
    }

    #[test]
    fn entering_takes_more_than_holding() {
        // Approaching from centre, 6 degrees is not enough to commit — the
        // asymmetry is the point.
        let mut t = tracker();
        assert_eq!(
            t.update(pose(6.0, 0.0), None).head.unwrap().horizontal,
            Horizontal::Center
        );
        assert_eq!(
            t.update(pose(8.5, 0.0), None).head.unwrap().horizontal,
            Horizontal::Right
        );
    }

    #[test]
    fn a_hard_swing_crosses_straight_over_without_a_centre_frame() {
        let mut t = tracker();
        t.update(pose(20.0, 0.0), None);
        assert_eq!(
            t.update(pose(-20.0, 0.0), None).head.unwrap().horizontal,
            Horizontal::Left,
            "a decisive turn should not spend a frame reading CENTER"
        );
    }

    #[test]
    fn eye_in_head_is_absent_rather_than_centre_when_pose_is_missing() {
        // `None` and `CENTER` are different claims: one says "not measured",
        // the other says "measured, and the eyes are straight ahead".
        let mut t = tracker();
        let out = t.update(None, gaze_deg(20.0, 0.0));
        assert!(out.head.is_none(), "no pose means no head row");
        assert!(out.eye.is_none(), "eye-in-head needs pose to exist");
        assert!(out.gaze.is_some(), "gaze itself was measured");
    }

    #[test]
    fn eye_in_head_moves_while_the_head_stays_centre() {
        // The acceptance test in miniature, and the only thing that
        // distinguishes a working eye-in-head subtraction from two numbers
        // that merely look alive: head square to the camera, eyes off to the
        // subject's left.
        let mut t = tracker();
        let g = Some(Gaze {
            yaw_rad: (-18.0f32).to_radians(),
            pitch_rad: 0.0,
            eye_yaw_rad: Some((-18.0f32).to_radians()),
            eye_pitch_rad: Some(0.0),
        });
        let out = t.update(pose(0.0, 0.0), g);
        assert_eq!(out.head.unwrap().horizontal, Horizontal::Center);
        assert_eq!(out.eye.unwrap().horizontal, Horizontal::Left);
    }

    #[test]
    fn a_dropped_pose_frame_does_not_reset_a_held_direction() {
        let mut t = tracker();
        t.update(pose(20.0, 0.0), None);
        // Pose missing for one frame: the row goes away, the latch does not.
        assert!(t.update(None, None).head.is_none());
        // Back at an angle that could hold but not enter.
        assert_eq!(
            t.update(pose(6.0, 0.0), None).head.unwrap().horizontal,
            Horizontal::Right
        );
    }

    #[test]
    fn the_frame_of_reference_is_stated_and_serializes_as_prose() {
        let mut t = tracker();
        let out = t.update(pose(0.0, 0.0), None);
        assert_eq!(out.frame_of_reference, FrameOfReference::Subject);
        // The UI prints this verbatim; it must not arrive as `Subject`.
        let json = serde_json::to_string(&out.frame_of_reference).unwrap();
        assert_eq!(json, "\"subject POV\"");
    }

    #[test]
    fn labels_serialize_in_upper_case_for_direct_display() {
        assert_eq!(serde_json::to_string(&Horizontal::Left).unwrap(), "\"LEFT\"");
        assert_eq!(serde_json::to_string(&Vertical::Down).unwrap(), "\"DOWN\"");
        assert_eq!(serde_json::to_string(&Horizontal::Center).unwrap(), "\"CENTER\"");
    }

    #[test]
    fn the_wire_shape_is_exactly_what_the_front_end_reads() {
        // Nothing checks the boundary between this struct and the JavaScript
        // that renders it — not the compiler, not the type system, nothing.
        // Rename a field here and the HUD silently shows dashes forever, which
        // looks like "the signal isn't running" rather than "the name moved".
        //
        // These are the exact paths in `drawDirections`:
        //   snap.signals.debug_directions.head.horizontal
        //   snap.signals.debug_directions.frame_of_reference
        let mut t = tracker();
        let g = Some(Gaze {
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            eye_yaw_rad: Some(0.0),
            eye_pitch_rad: Some(0.0),
        });
        let json = serde_json::to_string(&t.update(pose(-30.0, 0.0), g)).unwrap();

        assert_eq!(
            json,
            r#"{"head":{"horizontal":"LEFT","vertical":"CENTER"},"#.to_owned()
                + r#""gaze":{"horizontal":"CENTER","vertical":"CENTER"},"#
                + r#""eye":{"horizontal":"CENTER","vertical":"CENTER"},"#
                + r#""frame_of_reference":"subject POV"}"#
        );

        // An absent row must arrive as `null`, which the front end renders as a
        // dash. Omitting the key entirely would read the same in JavaScript
        // today and stop doing so the moment anyone adds `skip_serializing_if`.
        let mut t = tracker();
        let json = serde_json::to_string(&t.update(None, None)).unwrap();
        assert!(json.contains(r#""head":null"#), "got {json}");
        assert!(json.contains(r#""eye":null"#), "got {json}");
    }
}
