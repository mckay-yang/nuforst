//! Vector-valued NUFROST fit: one shared frequency set and one date-level weight
//! sequence for every band of a pixel.

use ndarray::Array2;

use crate::config::{Normalization, NufrostConfig};
use crate::nufft::{type1_vector_power_kb, NufftOptions};

use super::{
    make_design_matrix, maybe_days_to_seconds, multi_output_tiered_ridge_solve, nanmean, nanmedian,
    parse_preferred_frequencies, select_frequencies, select_phenology_frequencies,
    NufrostVectorResult,
};

const MAD_CONSISTENCY: f64 = 1.4826;
const SCALE_FLOOR: f64 = 1e-6;
const REFLECTANCE_SCALE: f64 = 10_000.0;
const HUBER_WEIGHT_DAMPING: f64 = 0.5;
const HUBER_MIN_WEIGHT: f64 = 0.03;
const MAX_HUBER_PASSES: usize = 5;
const ROBUST_DELTA_FLOOR: f64 = 1.5;
const REFLECTANCE_DELTA_FLOOR: f64 = 1e-6;

/// One jointly valid date after robust or reflectance standardization.
struct StandardizedObservations {
    row_indices: Vec<usize>,
    t_min: f64,
    t_rel: Vec<f64>,
    t_rel_mean: f64,
    centers: Vec<f64>,
    scales: Vec<f64>,
    values: Array2<f64>,
}

struct FittedDates {
    row_indices: Vec<usize>,
    design: Array2<f64>,
    standardized: Array2<f64>,
    result: NufrostVectorResult,
}

/// Fit one vector-valued NUFROST model.
///
/// `observations[band][time]` shares `ts_days`. Dates that are non-finite in
/// any band are dropped together.
pub fn fit_pixel(
    ts_days: &[f64],
    observations: &[Vec<f64>],
    config: &NufrostConfig,
) -> NufrostVectorResult {
    let n_times = ts_days.len();
    let n_bands = observations.len();
    let result = NufrostVectorResult {
        n_bands,
        include_trend: config.ridge.include_trend,
        ..NufrostVectorResult::default()
    };
    if n_times == 0 || n_bands == 0 || observations.iter().any(|band| band.len() != n_times) {
        return result;
    }

    let (ts_sec, _) = maybe_days_to_seconds(ts_days, ts_days[0]);
    let mut active = jointly_valid_mask(&ts_sec, observations);
    let reject_passes = rejection_passes(config);
    let mut final_fit = None;

    for pass in 0..=reject_passes {
        let Some(fit) = fit_active(&ts_sec, observations, &active, config) else {
            break;
        };
        if pass >= reject_passes {
            final_fit = Some(fit);
            break;
        }
        let Some(next_active) = reject_outlier_dates(&fit, &active, config) else {
            final_fit = Some(fit);
            break;
        };
        active = next_active;
    }

    final_fit.map(|fit| fit.result).unwrap_or(result)
}

fn rejection_passes(config: &NufrostConfig) -> usize {
    if config.robust.joint_outlier {
        (config.robust.reject_iters as usize).min(MAX_HUBER_PASSES)
    } else {
        0
    }
}

fn jointly_valid_mask(times_seconds: &[f64], observations: &[Vec<f64>]) -> Vec<bool> {
    (0..times_seconds.len())
        .map(|i| {
            times_seconds[i].is_finite() && observations.iter().all(|band| band[i].is_finite())
        })
        .collect()
}

fn fit_active(
    times_seconds: &[f64],
    observations: &[Vec<f64>],
    active: &[bool],
    config: &NufrostConfig,
) -> Option<FittedDates> {
    let standardized = standardize(times_seconds, observations, active, config)?;
    let (frequencies, phenology) = select_shared_frequencies(&standardized, config)?;
    let design = make_design_matrix(
        &standardized.t_rel,
        &frequencies,
        config.ridge.include_trend,
        true,
    );
    if design.ncols() == 0 {
        return None;
    }
    let theta = fit_huber(
        &design,
        &standardized.values,
        &frequencies,
        &phenology,
        config,
    )?;
    Some(FittedDates {
        row_indices: standardized.row_indices.clone(),
        design,
        standardized: standardized.values.clone(),
        result: NufrostVectorResult {
            valid: true,
            n_bands: observations.len(),
            freqs: frequencies,
            t_min: standardized.t_min,
            t_rel_mean: standardized.t_rel_mean,
            centers: standardized.centers,
            scales: standardized.scales,
            theta,
            include_trend: config.ridge.include_trend,
        },
    })
}

fn standardize(
    times_seconds: &[f64],
    observations: &[Vec<f64>],
    active: &[bool],
    config: &NufrostConfig,
) -> Option<StandardizedObservations> {
    let row_indices: Vec<usize> = active
        .iter()
        .enumerate()
        .filter_map(|(i, &keep)| keep.then_some(i))
        .collect();
    let n_use = row_indices.len();
    let min_obs = (config.robust.min_obs as usize).max(3);
    if n_use < min_obs {
        return None;
    }

    let t_use: Vec<f64> = row_indices.iter().map(|&i| times_seconds[i]).collect();
    let t_min = t_use.iter().copied().fold(f64::INFINITY, f64::min);
    let t_rel: Vec<f64> = t_use.iter().map(|&t| t - t_min).collect();
    let t_rel_mean = nanmean(&t_rel);
    let tspan = t_rel.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - t_rel.iter().copied().fold(f64::INFINITY, f64::min);
    if !tspan.is_finite() || tspan <= 0.0 {
        return None;
    }

    let n_bands = observations.len();
    let mut centers = vec![0.0; n_bands];
    let mut scales = vec![1.0; n_bands];
    let mut values = Array2::<f64>::zeros((n_use, n_bands));
    for band in 0..n_bands {
        let column: Vec<f64> = row_indices.iter().map(|&i| observations[band][i]).collect();
        let (center, scale) = center_and_scale(&column, config.robust.normalization);
        centers[band] = center;
        scales[band] = scale;
        for (row, value) in column.iter().enumerate() {
            values[[row, band]] = (value - center) / scale;
        }
    }

    Some(StandardizedObservations {
        row_indices,
        t_min,
        t_rel,
        t_rel_mean,
        centers,
        scales,
        values,
    })
}

fn center_and_scale(values: &[f64], normalization: Normalization) -> (f64, f64) {
    match normalization {
        Normalization::Reflectance => (0.0, REFLECTANCE_SCALE),
        Normalization::CenteredReflectance => (finite_center(values), REFLECTANCE_SCALE),
        Normalization::Robust => robust_center_and_scale(values),
    }
}

fn finite_center(values: &[f64]) -> f64 {
    let center = nanmedian(values);
    if center.is_finite() {
        center
    } else {
        nanmean(values)
    }
}

fn robust_center_and_scale(values: &[f64]) -> (f64, f64) {
    let center = finite_center(values);
    let deviations: Vec<f64> = values
        .iter()
        .map(|&value| {
            if value.is_finite() {
                (value - center).abs()
            } else {
                f64::NAN
            }
        })
        .collect();
    let mut scale = MAD_CONSISTENCY * nanmedian(&deviations);
    if !scale.is_finite() || scale <= SCALE_FLOOR {
        scale = standard_deviation(values);
    }
    (center, scale.max(SCALE_FLOOR))
}

fn standard_deviation(values: &[f64]) -> f64 {
    let mean = nanmean(values);
    let variance = values
        .iter()
        .filter(|value| value.is_finite())
        .map(|&value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() as f64).max(1.0);
    variance.sqrt()
}

fn select_shared_frequencies(
    observations: &StandardizedObservations,
    config: &NufrostConfig,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n_bands = observations.values.ncols();
    let spectrum_dims: Vec<Vec<f64>> = (0..n_bands)
        .map(|band| observations.values.column(band).iter().copied().collect())
        .collect();
    let (vector_freqs, vector_power) = type1_vector_power_kb(
        &observations.t_rel,
        &spectrum_dims,
        config.spectral.modes as usize,
        NufftOptions::default(),
    );
    if vector_freqs.is_empty() || vector_power.is_empty() {
        return None;
    }

    let fmax = nyquist_hz(&observations.t_rel);
    let preferred = parse_preferred_frequencies(&config.spectral.preferred_periods_days);
    let phenology = select_phenology_frequencies(
        &vector_freqs,
        &vector_power,
        fmax,
        &preferred,
        config.spectral.preferred_top_k as usize,
        config.spectral.ignore_dc_hz,
        config.spectral.merge_tolerance,
    );
    let selected = select_frequencies(
        &vector_freqs,
        &vector_power,
        fmax,
        config.spectral.selection.select_mode(),
        &preferred,
        config.spectral.preferred_top_k as usize,
        config.spectral.num_peaks as usize,
        config.spectral.spectral_top_k as usize,
        config.spectral.merge_tolerance,
        config.spectral.power_cum,
        config.spectral.ignore_dc_hz,
        config.spectral.refine_peaks,
    );
    if selected.is_empty() {
        None
    } else {
        Some((selected, phenology))
    }
}

fn nyquist_hz(t_rel: &[f64]) -> f64 {
    let mut sorted = t_rel.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let gaps: Vec<f64> = sorted
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .filter(|&gap| gap > 0.0)
        .collect();
    let tspan = sorted.last().copied().unwrap_or(0.0) - sorted.first().copied().unwrap_or(0.0);
    let dt_med = if gaps.is_empty() {
        tspan / t_rel.len() as f64
    } else {
        nanmedian(&gaps)
    };
    0.5 / dt_med.max(1e-12)
}

fn fit_huber(
    design: &Array2<f64>,
    values: &Array2<f64>,
    frequencies: &[f64],
    phenology: &[f64],
    config: &NufrostConfig,
) -> Option<Array2<f64>> {
    let n_rows = values.nrows();
    let mut weights = vec![1.0; n_rows];
    let delta = huber_delta(config);
    let passes = (config.robust.huber_iters as usize).clamp(1, MAX_HUBER_PASSES);
    let mut theta = solve_ridge(design, values, frequencies, &weights, phenology, config)?;

    for _ in 0..passes {
        let fitted = design.dot(&theta);
        weights = huber_date_weights(values, &fitted, &weights, delta);
        if let Some(next) = solve_ridge(design, values, frequencies, &weights, phenology, config) {
            theta = next;
        }
    }
    Some(theta)
}

fn solve_ridge(
    design: &Array2<f64>,
    values: &Array2<f64>,
    frequencies: &[f64],
    weights: &[f64],
    phenology: &[f64],
    config: &NufrostConfig,
) -> Option<Array2<f64>> {
    multi_output_tiered_ridge_solve(
        design,
        values,
        frequencies,
        weights,
        config.ridge.lambda,
        config.ridge.lambda_high,
        config.ridge.low_freq_period_days,
        config.ridge.freq_weight,
        true,
        config.ridge.include_trend,
        phenology,
        config.ridge.multiband_shrinkage,
    )
}

fn huber_delta(config: &NufrostConfig) -> f64 {
    let floor = if config.robust.normalization.uses_reflectance_scale() {
        REFLECTANCE_DELTA_FLOOR
    } else {
        ROBUST_DELTA_FLOOR
    };
    config.robust.huber_delta.max(floor)
}

fn huber_date_weights(
    values: &Array2<f64>,
    fitted: &Array2<f64>,
    previous: &[f64],
    delta: f64,
) -> Vec<f64> {
    (0..values.nrows())
        .map(|row| {
            let error = date_rms(values, fitted, row, 0.0);
            let huber = if error <= delta {
                1.0
            } else {
                delta / (error + 1e-12)
            };
            (HUBER_WEIGHT_DAMPING * previous[row] + (1.0 - HUBER_WEIGHT_DAMPING) * huber)
                .max(HUBER_MIN_WEIGHT)
        })
        .collect()
}

/// Root-mean-square residual of one date across bands.
fn date_rms(values: &Array2<f64>, fitted: &Array2<f64>, row: usize, empty: f64) -> f64 {
    let mut sum_squares = 0.0;
    let mut count = 0usize;
    for band in 0..values.ncols() {
        let residual = values[[row, band]] - fitted[[row, band]];
        if residual.is_finite() {
            sum_squares += residual * residual;
            count += 1;
        }
    }
    if count == 0 {
        empty
    } else {
        (sum_squares / count as f64).sqrt()
    }
}

/// Returns the next active mask, or `None` when no date should be removed.
fn reject_outlier_dates(
    fit: &FittedDates,
    active: &[bool],
    config: &NufrostConfig,
) -> Option<Vec<bool>> {
    let fitted = fit.design.dot(&fit.result.theta);
    let rms: Vec<f64> = (0..fit.standardized.nrows())
        .map(|row| date_rms(&fit.standardized, &fitted, row, f64::NAN))
        .collect();
    let threshold = outlier_threshold(&rms, huber_delta(config), config.robust.reject_sigma);
    let mut candidates: Vec<(usize, f64)> = rms
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, error)| error.is_finite() && *error > threshold)
        .collect();
    if candidates.is_empty() {
        return None;
    }
    candidates.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_rows = fit.row_indices.len();
    let min_remaining = (config.robust.min_obs as usize)
        .max(fit.design.ncols() + 2)
        .min(n_rows);
    let removable = n_rows.saturating_sub(min_remaining);
    let max_by_fraction = ((n_rows as f64) * config.robust.reject_max_fraction).ceil() as usize;
    let n_remove = candidates.len().min(removable).min(max_by_fraction.max(1));
    if n_remove == 0 {
        return None;
    }

    let mut next = active.to_vec();
    for (row, _) in candidates.into_iter().take(n_remove) {
        next[fit.row_indices[row]] = false;
    }
    Some(next)
}

fn outlier_threshold(rms: &[f64], delta_floor: f64, sigma: f64) -> f64 {
    let center = nanmedian(rms);
    let deviations: Vec<f64> = rms.iter().map(|&value| (value - center).abs()).collect();
    let mut scale = MAD_CONSISTENCY * nanmedian(&deviations);
    if !scale.is_finite() || scale <= 1e-12 {
        scale = standard_deviation(rms);
    }
    (center + sigma.max(0.0) * scale).max(delta_floor)
}
