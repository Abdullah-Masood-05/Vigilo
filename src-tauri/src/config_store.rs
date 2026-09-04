//! Persists user-configured thresholds to disk.
//!
//! The settings file lives in Tauri's app data directory, which survives
//! reinstalls and is editable by hand. The format is TOML, consistent with
//! the existing `--config` CLI flag.
//!
//! This module knows nothing about Tauri — it takes a directory path and
//! reads or writes a file. The Tauri layer resolves the directory and calls
//! in.

use std::path::{Path, PathBuf};

use crate::config::SettingsPayload;

const SETTINGS_FILE: &str = "settings.toml";

/// Resolved path to the settings file within a directory.
pub fn settings_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(SETTINGS_FILE)
}

/// Load persisted settings, or `None` if the file does not exist or is
/// unreadable. A corrupt file is logged and treated as absent — the user
/// gets defaults and can save again.
pub fn load(app_data_dir: &Path) -> Option<SettingsPayload> {
    let path = settings_path(app_data_dir);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "could not read settings file");
            return None;
        }
    };
    match toml::from_str::<SettingsPayload>(&text) {
        Ok(s) => Some(s),
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "corrupt settings file, using defaults");
            None
        }
    }
}

/// Persist settings to disk. Creates the directory if needed.
pub fn save(app_data_dir: &Path, settings: &SettingsPayload) -> Result<(), String> {
    let path = settings_path(app_data_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("could not create settings directory: {e}"))?;
    }
    let text = toml::to_string_pretty(settings)
        .map_err(|e| format!("could not serialize settings: {e}"))?;
    std::fs::write(&path, text)
        .map_err(|e| format!("could not write settings file: {e}"))?;
    tracing::info!(path = %path.display(), "settings saved");
    Ok(())
}
