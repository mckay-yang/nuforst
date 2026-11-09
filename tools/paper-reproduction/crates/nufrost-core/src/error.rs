use thiserror::Error;

/// Error types for the NUFROST core library.
///
/// Uses [`thiserror`] for ergonomic `Display` and `Error` implementations.
/// All public error variants carry structured context rather than bare strings.
#[derive(Error, Debug)]
pub enum NufrostError {
    /// A timestamp string could not be parsed in any supported format.
    #[error("invalid timestamp: '{0}'")]
    InvalidTimestamp(String),

    /// A required field was missing from a JSON configuration.
    ///
    /// The string names the missing field (e.g. `"modes"`, `"nof"`).
    #[error("missing required config field: '{0}'")]
    MissingConfigField(String),

    /// A config field was present but its value is semantically invalid.
    #[error("invalid value for config field '{field}': {reason}")]
    InvalidConfigValue {
        /// Name of the config key.
        field: String,
        /// Human-readable reason.
        reason: String,
    },

    /// The algorithm selected by the user does not match any recognised variant.
    #[error("unknown algorithm: '{0}' (expected one of 'nufrost', 'hants', 'zhu2015')")]
    UnknownAlgorithm(String),

    /// No valid observations remain after applying the valid mask.
    #[error("no valid observations after filtering")]
    NoValidObservations,

    /// I/O error forwarded from [`std::io`].
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON (de)serialisation error forwarded from [`serde_json`].
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
