# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-04

### Added
- **Dynamic Threshold Configuration System**:
  - Centralized, strongly-typed configuration DTOs (`UserThresholds`, `DevThresholds`, and `SettingsPayload`) with comprehensive boundary validation.
  - Zero-overhead atomic hot-reloading using `ArcSwap<Config>` in detection and identity pipeline worker threads without frame drops or capture interruptions.
  - Persistent disk storage in `settings.toml` located in the OS application data directory (`%APPDATA%` on Windows, `~/.config` on Linux, `~/Library/Application Support` on macOS).
  - Backend Tauri IPC commands: `get_thresholds`, `set_thresholds`, and `reset_thresholds`.
- **Modernized Settings UI**:
  - Slide-out glassmorphic settings drawer with backdrop blur and responsive layout.
  - Grouped controls for Presence & Identity, Head Pose, Gaze Tracking, Objects & Hardware Health, and Rule Severities.
  - Interactive In-App Guide accordion (`.settings-guide-card`) explaining all rule behaviors and severity tiers.
  - Developer Options toggle concealing low-level internal engine tunables (YuNet NMS/score, YOLOX thresholds, EMA smoothing alphas, blink EAR cutoff, loop cadences, ORT thread counts).
  - One-click "Reset to Defaults" button to restore factory settings.
- **Severity Tiers & Documentation**:
  - Formalized four proctoring severity tiers: **Low** (Informational anomaly), **Medium** (Suspicious warning), **High** (Probable misconduct), and **Critical** (Severe integrity breach).
  - Added comprehensive `Configuration & Settings Guide` to `README.md` detailing every threshold parameter, hysteresis calculation, developer tunable, and persistence behavior.
- **Integration Tests**:
  - Added `tests/threshold_settings.rs` testing settings roundtrip serialization, file corruption resilience, input validation, and runtime hot-reloading.

### Changed
- Refactored `detect_loop` to detect configuration pointer changes via `Arc::ptr_eq` and cleanly reconstruct `FusionEngine` and `DirectionTracker` on the fly.
- Updated release tags and packaging configuration to `1.0.0` across `Cargo.toml`, `tauri.conf.json`, `package.json`, `PKGBUILD`, and GitHub Actions CI.

---

## [0.1.0] - 2026-08-22

### Added
- Multi-model ONNX Runtime inference pipeline:
  - Face detection & 5 facial landmarks via YuNet.
  - Head pose estimation (yaw, pitch, roll) via MobileNetV3.
  - Gaze and eye-in-head tracking via MobileGaze.
  - Prohibited handheld device detection via YOLOX-Nano.
  - Candidate identity verification via ArcFace embeddings.
- DirectShow webcam capture on Windows using bundled, minimal static `ffmpeg.exe` (1.7 MB).
- Pure-function temporal fusion engine with hold timers and hysteresis for violation event generation (`NoFace`, `MultipleFaces`, `HeadTurnedAway`, `GazeOffScreen`, `ProhibitedObject`, `IdentityMismatch`, `SignalLost`).
- Zero-dependency vanilla HTML/JS/CSS frontend with loopback MJPEG video feed and real-time event HUD.
- Multi-platform packaging and installer pipelines for Windows (`.exe`, `.msi`), macOS (`.dmg`), and Linux (`.deb`, `.rpm`, `.pkg.tar.zst`, `.AppImage`).
