//! Opt-in supported multi-frequency reconstruction. Original entrypoints are unchanged.
//! V1 uses scaled reflectance (1 unit = 1/10000 reflectance) and a relative-day axis.
//! Constraints are training-derived regularization, never additional observed dates.
mod seasonal;
mod year_analog;

use crate::nufrost::{
    fit_vector_with_options, nufrost_fit_pixel_vector, NufrostVectorResult, VectorFitOptions,
};
use crate::NufrostConfig;
use seasonal::SeasonalSupport;
use year_analog::YearAnalog;

pub const MODEL_VERSION: &str = "nufrost-supported-v1";

#[derive(Debug, thiserror::Error)]
pub enum SupportedInputError {
    #[error("each band must have one value per timestamp, and input must be nonempty")]
    MalformedInput,
    #[error("supported V1 requires finite relative days, absolute values <=100000 and span <=10000 days")]
    InvalidTimeAxis,
}

#[derive(Debug, Clone)]
pub struct SupportedVectorResult {
    pub model: NufrostVectorResult,
    pub version: &'static str,
    pub training_dates: usize,
    /// Number of regularization constraints; not an observation count.
    pub constraint_count: usize,
}

/// Fits one joint Fourier coefficient matrix; predict with `nufrost_predict_vector`.
/// Hidden values must be NaN before calling this function. All bands share valid dates.
pub fn fit_supported_vector(
    days: &[f64],
    observations: &[Vec<f64>],
    config: &NufrostConfig,
) -> Result<SupportedVectorResult, SupportedInputError> {
    if days.is_empty()
        || observations.is_empty()
        || observations.iter().any(|b| b.len() != days.len())
    {
        return Err(SupportedInputError::MalformedInput);
    }
    let low = days.iter().copied().fold(f64::INFINITY, f64::min);
    let high = days.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if days.iter().any(|t| !t.is_finite() || t.abs() > 100_000.) || high - low > 10_000. {
        return Err(SupportedInputError::InvalidTimeAxis);
    }
    let train: Vec<_> = (0..days.len())
        .filter(|&i| observations.iter().all(|b| b[i].is_finite()))
        .collect();
    let original = nufrost_fit_pixel_vector(days, observations, config);
    let mut result = SupportedVectorResult {
        model: original,
        version: MODEL_VERSION,
        training_dates: train.len(),
        constraint_count: 0,
    };
    if !result.model.valid {
        return Ok(result);
    }
    let order = if train.len() < 18 {
        1
    } else if train.len() < 24 {
        2
    } else {
        3
    };
    let annual: Vec<_> = (1..=order).map(|k| k as f64 / (365.25 * 86400.)).collect();
    let seasonal_options = VectorFitOptions {
        fixed_frequencies: Some(annual.clone()),
        unpenalized_frequencies: Some(annual),
        ..Default::default()
    };
    let seasonal = fit_vector_with_options(days, observations, config, &seasonal_options);
    if !seasonal.valid {
        result.model = seasonal;
        return Ok(result);
    }
    let base = SeasonalSupport {
        times: days,
        values: observations,
        train: &train,
        seasonal: &seasonal,
        phase_transfer: false,
    };
    let analog = YearAnalog::new(base, 2, 200.);
    let constraints = analog.rows(6.);
    result.constraint_count = constraints.len();
    let options = VectorFitOptions {
        fixed_frequencies: Some(result.model.freqs.clone()),
        constraints,
        ..Default::default()
    };
    result.model = fit_vector_with_options(days, observations, config, &options);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_mismatched_bands_and_second_scale_axes() {
        let config =
            NufrostConfig::from_json(include_bytes!("../../../config/nufrost.json")).unwrap();
        assert!(matches!(
            fit_supported_vector(&[0., 1.], &[vec![1.]], &config),
            Err(SupportedInputError::MalformedInput)
        ));
        assert!(matches!(
            fit_supported_vector(&[0., 86400.], &[vec![1., 2.]], &config),
            Err(SupportedInputError::InvalidTimeAxis)
        ));
    }
    #[test]
    fn insufficient_observations_do_not_create_pseudo_truth() {
        let result = fit_supported_vector(
            &[0., 1.],
            &[vec![f64::NAN; 2]],
            &NufrostConfig::from_json(include_bytes!("../../../config/nufrost.json")).unwrap(),
        )
        .unwrap();
        assert!(!result.model.valid);
        assert_eq!(result.training_dates, 0);
        assert_eq!(result.constraint_count, 0);
    }
}
