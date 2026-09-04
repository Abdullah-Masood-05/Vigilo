use tempfile::tempdir;
use vigilo::config::Config;
use vigilo::config_store;
use vigilo::pipeline::Detector;
use vigilo::types::Severity;
use vigilo::SettingsPayload;

#[test]
fn settings_store_roundtrip_works() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path();

    // No file initially
    assert!(config_store::load(dir).is_none());

    let cfg = Config::default();
    let mut payload = SettingsPayload::from_config(&cfg);
    payload.user.no_face_hold_ms = 5250;
    payload.user.pose_yaw_enter_deg = 42.0;
    payload.user.pose_yaw_exit_deg = 32.0;
    payload.user.severity.insert("no_face".to_string(), Severity::Low);

    config_store::save(dir, &payload).expect("saving settings");

    let loaded = config_store::load(dir).expect("loading settings");
    assert_eq!(loaded.user.no_face_hold_ms, 5250);
    assert_eq!(loaded.user.pose_yaw_enter_deg, 42.0);
    assert_eq!(loaded.user.pose_yaw_exit_deg, 32.0);
    assert_eq!(loaded.user.severity.get("no_face"), Some(&Severity::Low));
}

#[test]
fn corrupt_settings_file_is_ignored_and_treated_as_absent() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path();
    let file = config_store::settings_path(dir);
    std::fs::write(&file, "not a valid toml = [[[{").unwrap();

    let loaded = config_store::load(dir);
    assert!(loaded.is_none());
}

#[test]
fn apply_to_updates_and_validates_config() {
    let mut cfg = Config::default();
    let mut payload = SettingsPayload::from_config(&cfg);
    payload.user.no_face_hold_ms = 7500;
    payload.user.pose_yaw_enter_deg = 35.0;
    payload.user.pose_yaw_exit_deg = 25.0;
    payload.apply_to(&mut cfg).expect("valid settings apply");

    assert_eq!(cfg.thresholds.face.no_face_hold_ms, 7500);
    assert_eq!(cfg.thresholds.pose.yaw_enter_deg, 35.0);
    assert_eq!(cfg.thresholds.pose.yaw_exit_deg, 25.0);

    // Invalid hysteresis: enter < exit fails validation
    payload.user.pose_yaw_enter_deg = 20.0;
    payload.user.pose_yaw_exit_deg = 30.0;
    assert!(payload.apply_to(&mut cfg).is_err());
}

#[test]
fn detector_hot_reloads_config() {
    let det = Detector::builder().build().unwrap();
    let initial = det.current_config();
    assert_eq!(initial.thresholds.face.no_face_hold_ms, 2500);

    let mut new_cfg = (*initial).clone();
    new_cfg.thresholds.face.no_face_hold_ms = 8000;
    det.update_config(new_cfg).expect("update succeeds");

    let updated = det.current_config();
    assert_eq!(updated.thresholds.face.no_face_hold_ms, 8000);
}

