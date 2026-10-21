use serde::{Deserialize, Serialize};

use crate::error::NufrostError;

/// How NUFROST chooses the shared harmonic frequencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrequencySelection {
    Spectral,
    Preferred,
    #[serde(alias = "shared_spectral")]
    Hybrid,
    All,
}

impl FrequencySelection {
    /// Mode string consumed by [`crate::nufrost::select_frequencies`].
    pub fn select_mode(self) -> &'static str {
        match self {
            Self::Spectral => "spectral",
            Self::Preferred => "preferred",
            Self::Hybrid => "hybrid",
            Self::All => "all",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "spectral" => Some(Self::Spectral),
            "preferred" => Some(Self::Preferred),
            "hybrid" | "shared_spectral" => Some(Self::Hybrid),
            "all" => Some(Self::All),
            _ => None,
        }
    }
}

/// How a multi-band observation is centered and scaled before fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Normalization {
    /// Per-band median and MAD.
    Robust,
    /// Fixed Sentinel-2 reflectance scale, `y / 10000`.
    Reflectance,
    /// Per-band median, then the fixed Sentinel-2 reflectance scale.
    CenteredReflectance,
}

impl Normalization {
    pub fn uses_reflectance_scale(self) -> bool {
        matches!(self, Self::Reflectance)
    }
}

/// Shared-frequency discovery for one pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralSettings {
    pub modes: u32,
    pub num_peaks: u32,
    pub power_cum: f64,
    pub ignore_dc_hz: f64,
    pub selection: FrequencySelection,
    pub preferred_periods_days: String,
    pub preferred_top_k: u32,
    pub spectral_top_k: u32,
    pub merge_tolerance: f64,
    pub refine_peaks: bool,
}

/// Harmonic ridge penalty shared by the vector fit and the single-band fit.
#[derive(Debug, Clone, PartialEq)]
pub struct RidgeSettings {
    pub lambda: f64,
    pub freq_weight: f64,
    pub multiband_shrinkage: f64,
    pub lambda_high: f64,
    pub low_freq_period_days: f64,
    pub include_trend: bool,
}

/// Standardization, Huber weights, and date-level outlier rejection.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustFit {
    pub normalization: Normalization,
    pub huber_iters: u32,
    pub huber_delta: f64,
    pub min_obs: u32,
    pub outlier_sigma: f64,
    pub reject_iters: u32,
    pub reject_sigma: f64,
    pub reject_max_fraction: f64,
    pub joint_outlier: bool,
}

/// Piecewise step term used by the single-band `nufrost` command.
#[derive(Debug, Clone, PartialEq)]
pub struct StepFit {
    pub lambda: f64,
    pub weight_by_gap: bool,
    pub max_outer_iter: u32,
    pub outer_tol: f64,
}

/// NUFROST reconstruction parameters.
///
/// JSON stays a flat object. `ridge` and `ridge_lam` both set [`RidgeSettings::lambda`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    from = "FlatNufrostConfig",
    into = "FlatNufrostConfig"
)]
pub struct NufrostConfig {
    pub spectral: SpectralSettings,
    pub ridge: RidgeSettings,
    pub robust: RobustFit,
    pub step: StepFit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FlatNufrostConfig {
    modes: u32,
    num_peaks: u32,
    power_cum: f64,
    ignore_dc_hz: f64,
    #[serde(default = "default_frequency_selection")]
    frequency_selection: FrequencySelection,
    #[serde(default = "default_empty_string")]
    preferred_periods_days: String,
    #[serde(default = "default_preferred_top_k")]
    preferred_top_k: u32,
    #[serde(default = "default_spectral_top_k")]
    spectral_top_k: u32,
    #[serde(default = "default_spectral_merge_tol")]
    spectral_merge_tol: f64,
    refine_peaks: bool,
    include_trend: bool,
    #[serde(rename = "ridge_lam", alias = "ridge")]
    ridge_lam: f64,
    freq_weight: f64,
    #[serde(default = "default_multiband_shrinkage")]
    multiband_shrinkage: f64,
    #[serde(default = "default_normalization")]
    normalization_mode: Normalization,
    huber_iters: u32,
    huber_delta: f64,
    min_obs: u32,
    #[serde(default = "default_outlier_sigma")]
    outlier_sigma: f64,
    #[serde(default = "default_outlier_reject_iters")]
    outlier_reject_iters: u32,
    #[serde(default = "default_outlier_sigma")]
    outlier_reject_sigma: f64,
    #[serde(default = "default_outlier_reject_max_fraction")]
    outlier_reject_max_fraction: f64,
    #[serde(default = "default_lambda_step")]
    lambda_step: f64,
    #[serde(default = "default_lambda_high")]
    lambda_high: f64,
    #[serde(default = "default_low_freq_period_days")]
    low_freq_period_days: f64,
    #[serde(default = "default_step_dt_weighting")]
    step_dt_weighting: bool,
    #[serde(default = "default_max_outer_iter")]
    max_outer_iter: u32,
    #[serde(default = "default_outer_tol")]
    outer_tol: f64,
    #[serde(default = "default_joint_outlier")]
    joint_outlier: bool,
}

fn default_frequency_selection() -> FrequencySelection {
    FrequencySelection::Spectral
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
fn default_outlier_sigma() -> f64 {
    2.5
}
fn default_multiband_shrinkage() -> f64 {
    0.0
}
fn default_normalization() -> Normalization {
    Normalization::Robust
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
fn default_low_freq_period_days() -> f64 {
    60.0
}
fn default_step_dt_weighting() -> bool {
    true
}
fn default_max_outer_iter() -> u32 {
    5
}
fn default_outer_tol() -> f64 {
    1e-3
}
fn default_joint_outlier() -> bool {
    true
}

impl From<FlatNufrostConfig> for NufrostConfig {
    fn from(flat: FlatNufrostConfig) -> Self {
        Self {
            spectral: SpectralSettings {
                modes: flat.modes,
                num_peaks: flat.num_peaks,
                power_cum: flat.power_cum,
                ignore_dc_hz: flat.ignore_dc_hz,
                selection: flat.frequency_selection,
                preferred_periods_days: flat.preferred_periods_days,
                preferred_top_k: flat.preferred_top_k,
                spectral_top_k: flat.spectral_top_k,
                merge_tolerance: flat.spectral_merge_tol,
                refine_peaks: flat.refine_peaks,
            },
            ridge: RidgeSettings {
                lambda: flat.ridge_lam,
                freq_weight: flat.freq_weight,
                multiband_shrinkage: flat.multiband_shrinkage,
                lambda_high: flat.lambda_high,
                low_freq_period_days: flat.low_freq_period_days,
                include_trend: flat.include_trend,
            },
            robust: RobustFit {
                normalization: flat.normalization_mode,
                huber_iters: flat.huber_iters,
                huber_delta: flat.huber_delta,
                min_obs: flat.min_obs,
                outlier_sigma: flat.outlier_sigma,
                reject_iters: flat.outlier_reject_iters,
                reject_sigma: flat.outlier_reject_sigma,
                reject_max_fraction: flat.outlier_reject_max_fraction,
                joint_outlier: flat.joint_outlier,
            },
            step: StepFit {
                lambda: flat.lambda_step,
                weight_by_gap: flat.step_dt_weighting,
                max_outer_iter: flat.max_outer_iter,
                outer_tol: flat.outer_tol,
            },
        }
    }
}

impl From<NufrostConfig> for FlatNufrostConfig {
    fn from(config: NufrostConfig) -> Self {
        Self {
            modes: config.spectral.modes,
            num_peaks: config.spectral.num_peaks,
            power_cum: config.spectral.power_cum,
            ignore_dc_hz: config.spectral.ignore_dc_hz,
            frequency_selection: config.spectral.selection,
            preferred_periods_days: config.spectral.preferred_periods_days,
            preferred_top_k: config.spectral.preferred_top_k,
            spectral_top_k: config.spectral.spectral_top_k,
            spectral_merge_tol: config.spectral.merge_tolerance,
            refine_peaks: config.spectral.refine_peaks,
            include_trend: config.ridge.include_trend,
            ridge_lam: config.ridge.lambda,
            freq_weight: config.ridge.freq_weight,
            multiband_shrinkage: config.ridge.multiband_shrinkage,
            lambda_high: config.ridge.lambda_high,
            low_freq_period_days: config.ridge.low_freq_period_days,
            normalization_mode: config.robust.normalization,
            huber_iters: config.robust.huber_iters,
            huber_delta: config.robust.huber_delta,
            min_obs: config.robust.min_obs,
            outlier_sigma: config.robust.outlier_sigma,
            outlier_reject_iters: config.robust.reject_iters,
            outlier_reject_sigma: config.robust.reject_sigma,
            outlier_reject_max_fraction: config.robust.reject_max_fraction,
            lambda_step: config.step.lambda,
            step_dt_weighting: config.step.weight_by_gap,
            max_outer_iter: config.step.max_outer_iter,
            outer_tol: config.step.outer_tol,
            joint_outlier: config.robust.joint_outlier,
        }
    }
}

impl NufrostConfig {
    /// Load from a JSON byte slice.
    pub fn from_json(data: &[u8]) -> Result<Self, NufrostError> {
        serde_json::from_slice(data).map_err(NufrostError::Json)
    }

    /// Validate that required numeric fields are in sensible ranges.
    pub fn validate(&self) -> Result<(), NufrostError> {
        if self.spectral.modes == 0 {
            return Err(NufrostError::InvalidConfigValue {
                field: "modes".into(),
                reason: "must be > 0".into(),
            });
        }
        if self.spectral.num_peaks == 0 {
            return Err(NufrostError::InvalidConfigValue {
                field: "num_peaks".into(),
                reason: "must be > 0".into(),
            });
        }
        if self.robust.reject_max_fraction < 0.0 || self.robust.reject_max_fraction >= 1.0 {
            return Err(NufrostError::InvalidConfigValue {
                field: "outlier_reject_max_fraction".into(),
                reason: "must be in [0, 1)".into(),
            });
        }
        if self.ridge.multiband_shrinkage < 0.0 {
            return Err(NufrostError::InvalidConfigValue {
                field: "multiband_shrinkage".into(),
                reason: "must be >= 0".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NUFROST_JSON: &str = r#"{
        "modes": 4096,
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
        assert_eq!(cfg.spectral.modes, 4096);
        assert_eq!(cfg.spectral.num_peaks, 10);
        assert!((cfg.ridge.lambda - 0.005).abs() < 1e-20);
        assert_eq!(cfg.robust.min_obs, 12);
        assert_eq!(cfg.spectral.selection, FrequencySelection::Spectral);
        assert_eq!(cfg.ridge.multiband_shrinkage, 0.0);
        assert_eq!(cfg.robust.normalization, Normalization::Robust);
        assert_eq!(cfg.robust.outlier_sigma, 2.5);
        assert_eq!(cfg.robust.reject_iters, 2);
    }

    #[test]
    fn shared_spectral_alias_is_hybrid() {
        let json = r#"{
            "modes": 64,
            "num_peaks": 4,
            "power_cum": 0.7,
            "ignore_dc_hz": 1e-10,
            "frequency_selection": "shared_spectral",
            "refine_peaks": true,
            "include_trend": true,
            "ridge": 0.05,
            "freq_weight": 1.0,
            "huber_iters": 1,
            "huber_delta": 1.0,
            "min_obs": 8
        }"#;
        let cfg = NufrostConfig::from_json(json.as_bytes()).unwrap();
        assert_eq!(cfg.spectral.selection, FrequencySelection::Hybrid);
    }

    #[test]
    fn nufrost_validate_zero_modes() {
        let mut cfg = NufrostConfig::from_json(NUFROST_JSON.as_bytes()).unwrap();
        cfg.spectral.modes = 0;
        let err = cfg.validate().unwrap_err();
        assert!(format!("{err}").contains("modes"));
    }

    #[test]
    fn missing_required_field_is_error() {
        let bad_json = r#"{"num_peaks": 10}"#;
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
        let json_with_ridge = r#"{
            "modes": 4096,
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
        assert!((cfg.ridge.lambda - 0.005).abs() < 1e-10);
    }

    #[test]
    fn centered_reflectance_normalization_is_valid() {
        let mut cfg = NufrostConfig::from_json(NUFROST_JSON.as_bytes()).unwrap();
        cfg.robust.normalization = Normalization::CenteredReflectance;
        cfg.validate().unwrap();
    }

    #[test]
    fn flat_json_round_trip_keeps_groups() {
        let cfg = NufrostConfig::from_json(NUFROST_JSON.as_bytes()).unwrap();
        let encoded = serde_json::to_vec(&cfg).unwrap();
        let decoded = NufrostConfig::from_json(&encoded).unwrap();
        assert_eq!(cfg, decoded);
    }
}
