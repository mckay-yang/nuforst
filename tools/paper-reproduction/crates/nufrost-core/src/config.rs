use serde::{Deserialize, Serialize};

use crate::error::NufrostError;

/// NUFROST reconstruction parameters.
///
/// Field names match the Python `config/nufrost.json` keys exactly,
/// except `ridge_lam` which maps to Python's `"ridge"` key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NufrostConfig {
    pub modes: u32,
    pub eps: f64,
    pub num_peaks: u32,
    pub power_cum: f64,
    pub ignore_dc_hz: f64,
    #[serde(default = "default_frequency_selection")]
    pub frequency_selection: String,
    #[serde(default = "default_empty_string")]
    pub preferred_periods_days: String,
    #[serde(default = "default_preferred_top_k")]
    pub preferred_top_k: u32,
    #[serde(default = "default_spectral_top_k")]
    pub spectral_top_k: u32,
    #[serde(default = "default_spectral_merge_tol")]
    pub spectral_merge_tol: f64,
    /// Max private frequencies per band (in addition to shared).
    /// Private frequencies are band-specific peaks not present in shared.
    #[serde(default = "default_private_top_k_per_band")]
    pub private_top_k_per_band: usize,
    /// Ridge penalty multiplier for private (band-specific) frequency columns.
    /// Values > 1.0 increase regularization on private frequencies.
    #[serde(default = "default_private_freq_penalty_mult")]
    pub private_freq_penalty_mult: f64,
    pub refine_peaks: bool,
    pub include_trend: bool,
    /// Rust field name is `ridge_lam`; JSON key is `ridge` (matching Python config).
    #[serde(alias = "ridge_lam", alias = "ridge")]
    pub ridge_lam: f64,
    pub freq_weight: f64,
    /// Extra ridge penalty on band-specific coefficient deviations from the
    /// across-band mean trajectory.  This is a multiband-periodogram style
    /// shrinkage term; 0 disables it.
    #[serde(default = "default_multiband_shrinkage")]
    pub multiband_shrinkage: f64,
    /// Multi-band fitting normalization. `robust` uses per-pixel band median/MAD;
    /// `reflectance` uses fixed Sentinel-2 reflectance scaling Y / 10000;
    /// `centered_reflectance` subtracts the per-pixel band median and then
    /// applies fixed Sentinel-2 reflectance scaling.
    #[serde(default = "default_normalization_mode")]
    pub normalization_mode: String,
    pub huber_iters: u32,
    pub huber_delta: f64,
    pub min_obs: u32,
    #[serde(default = "default_outlier_sigma")]
    pub outlier_sigma: f64,
    #[serde(default = "default_outlier_reject_iters")]
    pub outlier_reject_iters: u32,
    #[serde(default = "default_outlier_sigma")]
    pub outlier_reject_sigma: f64,
    #[serde(default = "default_outlier_reject_max_fraction")]
    pub outlier_reject_max_fraction: f64,
    #[serde(default = "default_lambda_step")]
    pub lambda_step: f64,
    #[serde(default = "default_lambda_high")]
    pub lambda_high: f64,
    #[serde(default = "default_low_freq_period_days")]
    pub low_freq_period_days: f64,
    #[serde(default = "default_step_dt_weighting")]
    pub step_dt_weighting: bool,
    #[serde(default = "default_max_outer_iter")]
    pub max_outer_iter: u32,
    #[serde(default = "default_outer_tol")]
    pub outer_tol: f64,
    #[serde(default = "default_joint_outlier")]
    pub joint_outlier: bool,
    #[serde(default = "default_outlier_sigma")]
    pub joint_outlier_sigma: f64,
    #[serde(default = "default_admm_rho")]
    pub admm_rho: f64,
    #[serde(default = "default_admm_max_iter")]
    pub admm_max_iter: u32,
    #[serde(default = "default_admm_tol")]
    pub admm_tol: f64,
}

// Default helpers — match Python config/nufrost.json defaults.
fn default_frequency_selection() -> String {
    "spectral".into()
}

fn default_empty_string() -> String {
    String::new()
}
fn default_preferred_top_k() -> u32 {
    4
}
fn default_spectral_top_k() -> u32 {
    8
}
fn default_spectral_merge_tol() -> f64 {
    0.15
}
fn default_private_top_k_per_band() -> usize {
    2
}
fn default_private_freq_penalty_mult() -> f64 {
    1.5
}
fn default_outlier_sigma() -> f64 {
    2.5
}
fn default_multiband_shrinkage() -> f64 {
    0.0
}
fn default_normalization_mode() -> String {
    "robust".into()
}
fn default_outlier_reject_iters() -> u32 {
    2
}
fn default_outlier_reject_max_fraction() -> f64 {
    0.35
}
fn default_lambda_step() -> f64 {
    1e30
}
fn default_lambda_high() -> f64 {
    0.005
}
fn default_max_outer_iter() -> u32 {
    5
}
fn default_outer_tol() -> f64 {
    1e-3
}
fn default_admm_rho() -> f64 {
    1.0
}
fn default_admm_max_iter() -> u32 {
    80
}
fn default_admm_tol() -> f64 {
    1e-4
}
fn default_low_freq_period_days() -> f64 {
    60.0
}
fn default_step_dt_weighting() -> bool {
    true
}
fn default_joint_outlier() -> bool {
    true
}

impl NufrostConfig {
    /// Load from a JSON byte slice.
    pub fn from_json(data: &[u8]) -> Result<Self, NufrostError> {
        serde_json::from_slice(data).map_err(NufrostError::Json)
    }

    /// Validate that required numeric fields are in sensible ranges.
    ///
    /// Returns `Ok(())` or an [`NufrostError::InvalidConfigValue`].
    pub fn validate(&self) -> Result<(), NufrostError> {
        if self.modes == 0 {
            return Err(NufrostError::InvalidConfigValue {
                field: "modes".into(),
                reason: "must be > 0".into(),
            });
        }
        if self.num_peaks == 0 {
            return Err(NufrostError::InvalidConfigValue {
                field: "num_peaks".into(),
                reason: "must be > 0".into(),
            });
        }
        if self.eps <= 0.0 {
            return Err(NufrostError::InvalidConfigValue {
                field: "eps".into(),
                reason: "must be > 0".into(),
            });
        }
        if self.outlier_reject_max_fraction < 0.0 || self.outlier_reject_max_fraction >= 1.0 {
            return Err(NufrostError::InvalidConfigValue {
                field: "outlier_reject_max_fraction".into(),
                reason: "must be in [0, 1)".into(),
            });
        }
        if self.multiband_shrinkage < 0.0 {
            return Err(NufrostError::InvalidConfigValue {
                field: "multiband_shrinkage".into(),
                reason: "must be >= 0".into(),
            });
        }
        if self.normalization_mode != "robust"
            && self.normalization_mode != "reflectance"
            && self.normalization_mode != "centered_reflectance"
        {
            return Err(NufrostError::InvalidConfigValue {
                field: "normalization_mode".into(),
                reason: "must be robust, reflectance, or centered_reflectance".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ────────────────────────────────────────────────────────────

    const NUFROST_JSON: &str = r#"{
        "modes": 4096,
        "eps": 1e-12,
        "num_peaks": 10,
        "power_cum": 0.7,
        "ignore_dc_hz": 1e-10,
        "refine_peaks": true,
        "include_trend": true,
        "ridge_lam": 0.005,
        "freq_weight": 2.0,
        "huber_iters": 3,
        "huber_delta": 0.05,
        "min_obs": 12
    }"#;

    #[test]
    fn parse_nufrost_config() {
        let cfg = NufrostConfig::from_json(NUFROST_JSON.as_bytes()).unwrap();
        assert_eq!(cfg.modes, 4096);
        assert_eq!(cfg.num_peaks, 10);
        assert!((cfg.eps - 1e-12).abs() < 1e-20);
        assert!((cfg.ridge_lam - 0.005).abs() < 1e-20);
        assert_eq!(cfg.min_obs, 12);
        // Defaults kick in for omitted fields
        assert_eq!(cfg.frequency_selection, "spectral");
        assert_eq!(cfg.multiband_shrinkage, 0.0);
        assert_eq!(cfg.normalization_mode, "robust");
        assert_eq!(cfg.outlier_sigma, 2.5);
        assert_eq!(cfg.outlier_reject_iters, 2);
    }

    #[test]
    fn nufrost_validate_zero_modes() {
        let mut cfg = NufrostConfig::from_json(NUFROST_JSON.as_bytes()).unwrap();
        cfg.modes = 0;
        let err = cfg.validate().unwrap_err();
        assert!(format!("{err}").contains("modes"));
    }

    #[test]
    fn missing_required_field_is_error() {
        // "ridge_lam" is required (deny_unknown_fields, no default) but
        // "ridge" alias exists.  Drop "modes" which has no default.
        let bad_json = r#"{"eps": 1e-12, "num_peaks": 10}"#;
        let result = NufrostConfig::from_json(bad_json.as_bytes());
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("missing"),
            "expected missing field error, got: {msg}"
        );
    }

    #[test]
    fn parse_ridge_alias() {
        // Python config uses "ridge", Rust struct uses "ridge_lam" with alias.
        let json_with_ridge = r#"{
            "modes": 4096,
            "eps": 1e-12,
            "num_peaks": 10,
            "power_cum": 0.7,
            "ignore_dc_hz": 1e-10,
            "refine_peaks": true,
            "include_trend": true,
            "ridge": 0.005,
            "freq_weight": 2.0,
            "huber_iters": 3,
            "huber_delta": 0.05,
            "min_obs": 12
        }"#;
        let cfg = NufrostConfig::from_json(json_with_ridge.as_bytes()).unwrap();
        assert!((cfg.ridge_lam - 0.005).abs() < 1e-10);
    }

    #[test]
    fn centered_reflectance_normalization_is_valid() {
        let mut cfg = NufrostConfig::from_json(NUFROST_JSON.as_bytes()).unwrap();
        cfg.normalization_mode = "centered_reflectance".into();
        cfg.validate().unwrap();
    }
}
