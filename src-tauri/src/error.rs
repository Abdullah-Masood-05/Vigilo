//! Errors and the degradation policy (MODELS.md §8).
//!
//! Policy: **degrade, never die.** A proctoring session must not crash
//! mid-exam. Only two conditions are fatal — no camera, and no face model.
//! Everything else is a capability the session continues without, having
//! emitted `Event::Degraded` and recorded the loss in the session report.

use std::path::PathBuf;

#[derive(thiserror::Error, Debug)]
pub enum DetectError {
    #[error("camera: {0}")]
    Camera(String),

    #[error("model load {path}: {source}")]
    ModelLoad {
        path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("inference in {model}: {source}")]
    Inference {
        model: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("config: {0}")]
    Config(String),

    #[error("frame source {source_name}: {detail}")]
    Source { source_name: String, detail: String },

    #[error("io {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl DetectError {
    /// Whether this error ends the session. Everything else degrades.
    pub fn is_fatal(&self) -> bool {
        match self {
            DetectError::Camera(_) => true,
            // A model load failure is fatal only for the face model; the
            // caller knows which slot it was loading and decides. Default to
            // survivable so an unexpected slot never kills an exam.
            DetectError::ModelLoad { .. } => false,
            DetectError::Inference { .. } => false,
            DetectError::Config(_) => true,
            DetectError::Source { .. } => true,
            DetectError::Io { .. } => false,
        }
    }

    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        DetectError::Io { path: path.into(), source }
    }

    pub fn source(name: impl Into<String>, detail: impl Into<String>) -> Self {
        DetectError::Source { source_name: name.into(), detail: detail.into() }
    }
}

pub type Result<T> = std::result::Result<T, DetectError>;
