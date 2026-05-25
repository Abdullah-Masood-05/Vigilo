# samples/

Labelled clips for the regression suite (MODELS.md §10).

Record 15–20 short clips: phone visible, book open, second person entering
frame, looking left/right/down, leaving frame, sustained off-screen gaze —
**plus clean control clips** where nothing happens but you fidget, scratch your
face, drink water, adjust your chair.

The control clips matter more than the violation clips. False positives are
what make a proctoring system unusable, and you cannot measure a false-positive
rate without footage of innocent behaviour.

Store the expected outcome next to each clip:

```
samples/
  phone_01.mp4
  phone_01.expect.toml     # kind + time window per expected violation
  control_fidget_01.mp4
  control_fidget_01.expect.toml   # expects nothing
```

`tests/fusion_replay.rs` asserts these. That suite is what lets the FYP defence
answer "how accurate is it?" with a number instead of a vibe.

Clips are committed through git-lfs (see `../.gitattributes`) or not at all.
`_smoke_testsrc.mp4` is a generated ffmpeg test pattern, not footage — it exists
only to smoke-test the plumbing.
