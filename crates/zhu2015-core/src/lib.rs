//! Zhu et al. (2015) synthetic Landsat reconstruction baseline.
//!
//! # Paper reference
//! Zhu et al. 2015 — "Generating synthetic Landsat images based on all
//! available Landsat data: Predicting Landsat surface reflectance at any
//! given time."
//!
//! # Algorithm summary
//!
//! 1. Select harmonic model order from valid observation count:
//!    - < 6 valid → median fallback (order 0)
//!    - 6–17     → simple   (order 1, 4 coefs)
//!    - 18–23    → advanced (order 2, 6 coefs)
//!    - ≥ 24     → full     (order 3, 8 coefs)
//! 2. Build harmonic + linear-trend design matrix.
//! 3. Fit via LASSO (L1-regularised) coordinate descent.
//! 4. Predict at target time.
//! 5. Segment / break detection (full paper algorithm, single-segment
//!    for parity with Python reference).
//!
//! # Solver: custom coordinate descent
//!
//! We implement LASSO coordinate descent directly rather than pulling
//! in an external solver crate:
//!
//! - **Deterministic parity**: sklearn's `Lasso(alpha=..., fit_intercept=True,
//!   max_iter=1000)` uses unnormalized coordinate descent with intercept
//!   handled by centering. External solver crates use different formulations
//!   that produce different coefficient paths.
//!
//! - **No heavy dependency**: ~50 lines of Rust vs pulling in BLAS/LAPACK
//!   via ndarray-linalg.
//!
//! - **Paper fidelity**: the paper explicitly uses LASSO (not ridge, OLS).
//!
//! ## Coordinate descent update rule
//!
//! For unnormalized data with intercept handled by centering:
//!
//! ```text
//! X_c  = X - mean(X, axis=0),  y_c  = y - mean(y)
//! For each feature j:
//!   ρ_j = (X_cj' @ r) / n  +  w_j * ||X_cj||² / n
//!   w_j = soft_threshold(ρ_j, α) / (||X_cj||² / n)
//! intercept = mean(y) - mean(X) · w
//! ```
//!
//! # QA band encoding
//!
//! | QA | Meaning         | Valid obs |
//! |----|-----------------|-----------|
//! | 0  | Median fallback | < 6       |
//! | 1  | Simple model    | 6–17      |
//! | 2  | Advanced model  | 18–23     |
//! | 3  | Full model      | ≥ 24      |
//!
//! Matches the Python reference's simplified model-order encoding.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub enum Zhu2015ConfigError {
    Json(serde_json::Error),
    InvalidConfigValue { field: String, reason: String },
}

impl std::fmt::Display for Zhu2015ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(err) => write!(f, "json parse error: {err}"),
            Self::InvalidConfigValue { field, reason } => {
                write!(f, "invalid config value for '{field}': {reason}")
            }
        }
    }
}

impl std::error::Error for Zhu2015ConfigError {}

/// Zhu2015 reconstruction parameters.
///
/// Field names match `config/zhu2015.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Zhu2015Config {
    pub lasso_alpha: f64,
}

impl Zhu2015Config {
    /// Load from a JSON byte slice.
    pub fn from_json(data: &[u8]) -> Result<Self, Zhu2015ConfigError> {
        serde_json::from_slice(data).map_err(Zhu2015ConfigError::Json)
    }

    /// Validate required fields.
    pub fn validate(&self) -> Result<(), Zhu2015ConfigError> {
        if self.lasso_alpha < 0.0 {
            return Err(Zhu2015ConfigError::InvalidConfigValue {
                field: "lasso_alpha".into(),
                reason: "must be >= 0".into(),
            });
        }
        Ok(())
    }
}

// ── Constants ──────────────────────────────────────────────────────────────

pub const DAYS_PER_YEAR: f64 = 365.25;
pub const DEFAULT_LASSO_ALPHA: f64 = 0.1;
pub const MIN_OBS_FOR_FIT: usize = 6;

// ── Model order selection ──────────────────────────────────────────────────

/// Select harmonic model order from valid observation count.
///
/// Paper thresholds: `<6→0, 6-17→1, 18-23→2, ≥24→3`.
#[inline]
pub fn select_model_order(n_valid: usize) -> u32 {
    if n_valid < 6 {
        0
    } else if n_valid < 18 {
        1
    } else if n_valid < 24 {
        2
    } else {
        3
    }
}

// ── Design matrix ─────────────────────────────────────────────────────────

/// Build the harmonic + linear-trend design matrix.
///
/// For harmonic order `k` (1..=order):
///   - column(2k-2) = cos(k * ω * x)
///   - column(2k-1) = sin(k * ω * x)
/// Last column = `x - ref_x_mean` (or `x` if `ref_x_mean` is `None`).
pub fn make_design_matrix(x: &[f64], order: u32, ref_x_mean: Option<f64>) -> Array2<f64> {
    let n = x.len();
    let n_cols = (2 * order + 1) as usize;
    let w = 2.0 * std::f64::consts::PI / DAYS_PER_YEAR;

    let mut data: Vec<f64> = Vec::with_capacity(n * n_cols);
    for &xi in x {
        for k in 1..=order {
            let kwx = (k as f64) * w * xi;
            data.push(kwx.cos());
            data.push(kwx.sin());
        }
        let trend = match ref_x_mean {
            Some(m) => xi - m,
            None => xi,
        };
        data.push(trend);
    }
    Array2::from_shape_vec((n, n_cols), data).unwrap()
}

// ── Coordinate descent LASSO solver ────────────────────────────────────────

#[inline]
fn soft_threshold(z: f64, t: f64) -> f64 {
    if z > t {
        z - t
    } else if z < -t {
        z + t
    } else {
        0.0
    }
}

/// LASSO coordinate descent matching sklearn's `Lasso(fit_intercept=True)`.
///
/// Minimizes: `(1/(2n))·‖y − X·w − intercept‖² + α·‖w‖₁`.
///
/// # Panics
/// Panics if `X.nrows() == 0` or `X.nrows() != y.len()`.
pub fn lasso_fit(
    x: &Array2<f64>,
    y: &Array1<f64>,
    alpha: f64,
    max_iter: u32,
    tol: f64,
) -> (Array1<f64>, f64) {
    let (n, p) = (x.nrows(), x.ncols());
    debug_assert!(n > 0, "X must have at least one row");
    debug_assert_eq!(n, y.len(), "X and y must have same number of rows");

    let x_mean = x.mean_axis(Axis(0)).unwrap();
    let y_mean = y.mean().unwrap();
    let x_c = x - &x_mean;
    let y_c = y - y_mean;

    let col_norms_sq: Vec<f64> = x_c
        .columns()
        .into_iter()
        .map(|col| col.dot(&col) / (n as f64))
        .collect();

    let mut w = Array1::<f64>::zeros(p);
    let mut residual = y_c.clone();

    for _iter in 0..max_iter {
        let mut max_delta: f64 = 0.0;
        for j in 0..p {
            let w_old = w[j];
            let norm_sq = col_norms_sq[j];
            if norm_sq < 1e-15 {
                w[j] = 0.0;
                continue;
            }

            let col = x_c.column(j);
            let rho = col.dot(&residual) / (n as f64) + w_old * norm_sq;
            let w_new = soft_threshold(rho, alpha) / norm_sq;

            w[j] = w_new;
            let delta = (w_new - w_old).abs();
            if delta > 0.0 {
                let signed_delta = w_new - w_old;
                ndarray::Zip::from(&mut residual)
                    .and(&col)
                    .for_each(|r, &xij| *r -= signed_delta * xij);
            }
            if delta > max_delta {
                max_delta = delta;
            }
        }
        if max_delta < tol {
            break;
        }
    }

    let intercept = y_mean - x_mean.dot(&w);
    (w, intercept)
}

/// Predict using LASSO coefficients.
pub fn lasso_predict(x: &Array2<f64>, coef: &Array1<f64>, intercept: f64) -> Array1<f64> {
    x.dot(coef).mapv(|v| v + intercept)
}

// ── Single-pixel interface ─────────────────────────────────────────────────

/// Result of a Zhu2015 per-pixel LASSO fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Zhu2015Result {
    pub prediction: f64,
    pub qa: u32,
}

/// Fit a Zhu2015 LASSO model on a single pixel and predict at `target_t_day`.
///
/// Equivalent to Python `fit_predict_pixel()`.
pub fn fit_predict_pixel(
    t_days: &[f64],
    y: &[f64],
    target_t_day: f64,
    lasso_alpha: f64,
) -> Zhu2015Result {
    debug_assert_eq!(t_days.len(), y.len());

    let mut t_valid: Vec<f64> = Vec::with_capacity(t_days.len());
    let mut y_valid: Vec<f64> = Vec::with_capacity(y.len());
    for (i, &obs) in y.iter().enumerate() {
        if obs.is_finite() {
            t_valid.push(t_days[i]);
            y_valid.push(obs);
        }
    }

    let n_valid = t_valid.len();
    if n_valid == 0 {
        return Zhu2015Result {
            prediction: f64::NAN,
            qa: 0,
        };
    }

    if n_valid < MIN_OBS_FOR_FIT {
        let mut y_sorted = y_valid.clone();
        y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if y_sorted.len() % 2 == 1 {
            y_sorted[y_sorted.len() / 2]
        } else {
            let mid = y_sorted.len() / 2;
            (y_sorted[mid - 1] + y_sorted[mid]) / 2.0
        };
        return Zhu2015Result {
            prediction: median,
            qa: 0,
        };
    }

    let order = select_model_order(n_valid);
    debug_assert!(order >= 1);

    let x_mean = t_valid.iter().sum::<f64>() / (t_valid.len() as f64);
    let x_train = make_design_matrix(&t_valid, order, Some(x_mean));
    let y_train = Array1::from_vec(y_valid);

    let (coef, intercept) = lasso_fit(&x_train, &y_train, lasso_alpha, 1000, 1e-12);
    let x_target = make_design_matrix(&[target_t_day], order, Some(x_mean));
    let pred = x_target.dot(&coef)[0] + intercept;

    Zhu2015Result {
        prediction: pred,
        qa: order,
    }
}

/// Reconstruct a full raster using Zhu2015.
///
/// GDAL handles raster I/O and per-pixel traversal; this crate supplies the
/// Zhu2015 pixel model.
#[cfg(feature = "raster")]
pub fn reconstruct_zhu2015_geotiff<P: AsRef<std::path::Path>>(
    reader: &gdal::RasterReader,
    timestamps_days: &[f64],
    target_t_day: f64,
    lasso_alpha: f64,
    output_path: P,
    metadata: &gdal::RasterMetadata,
) -> anyhow::Result<()> {
    let cube = gdal::read_all_bands(reader)?;
    gdal::reconstruct_single_band(
        &cube,
        timestamps_days,
        target_t_day,
        output_path,
        metadata,
        |ts, obs, targ| {
            let result = fit_predict_pixel(ts, obs, targ, lasso_alpha);
            if result.prediction.is_finite() {
                result.prediction
            } else {
                f64::NAN
            }
        },
    )
}

// ── Segment-aware fitting ──────────────────────────────────────────────────

/// A fitted model segment.
#[derive(Debug, Clone)]
pub struct SegmentModel {
    pub coef: Array1<f64>,
    pub intercept: f64,
    pub order: u32,
    pub rmse: f64,
    pub x_mean: f64,
    pub t_start: f64,
    pub t_end: f64,
}

/// Fit a single Zhu2015 segment on `t_days[start..end]`.
///
/// Returns `None` if fewer than 6 valid observations.
pub fn fit_segment(
    t_days: &[f64],
    y: &[f64],
    start: usize,
    end: usize,
    lasso_alpha: f64,
) -> Option<SegmentModel> {
    debug_assert!(start <= end && end <= t_days.len());

    let slice = &t_days[start..end];
    let y_slice = &y[start..end];

    let mut t_valid = Vec::with_capacity(slice.len());
    let mut y_valid = Vec::with_capacity(slice.len());
    for (i, &obs) in y_slice.iter().enumerate() {
        if obs.is_finite() {
            t_valid.push(slice[i]);
            y_valid.push(obs);
        }
    }

    let n_valid = t_valid.len();
    if n_valid < MIN_OBS_FOR_FIT {
        return None;
    }

    let order = select_model_order(n_valid);
    if order == 0 {
        return None;
    }

    let x_mean = t_valid.iter().sum::<f64>() / (n_valid as f64);
    let x_train = make_design_matrix(&t_valid, order, Some(x_mean));
    let y_train = Array1::from_vec(y_valid);

    let (coef, intercept) = lasso_fit(&x_train, &y_train, lasso_alpha, 1000, 1e-12);

    let y_pred = lasso_predict(&x_train, &coef, intercept);
    let residuals = &y_train - &y_pred;
    let rmse = (residuals.mapv(|r| r * r).mean().unwrap_or(0.0)).sqrt();

    let t_start = t_valid[0];
    let t_end = t_valid[n_valid - 1];

    Some(SegmentModel {
        coef,
        intercept,
        order,
        rmse,
        x_mean,
        t_start,
        t_end,
    })
}

/// Predict using a fitted segment model.
pub fn segment_predict(model: &SegmentModel, target_t_day: f64) -> f64 {
    let x_target = make_design_matrix(&[target_t_day], model.order, Some(model.x_mean));
    x_target.dot(&model.coef)[0] + model.intercept
}

/// Temporally-adjusted RMSE: nearest N neighbors by day-of-year.
///
/// Paper rule: when clear obs > 24, use nearest 24 observations by DOY
/// for seasonal RMSE adjustment.
pub fn temporal_rmse(
    t_days: &[f64],
    y: &[f64],
    target_doy: f64,
    n_neighbors: usize,
) -> Option<f64> {
    let doy_period = DAYS_PER_YEAR;
    let target = target_doy % doy_period;
    let mut doy_dists: Vec<(f64, f64)> = Vec::new();

    for (i, &obs) in y.iter().enumerate() {
        if !obs.is_finite() {
            continue;
        }
        let doy = t_days[i] % doy_period;
        let mut dist = (doy - target).abs();
        if dist > doy_period / 2.0 {
            dist = doy_period - dist;
        }
        doy_dists.push((dist, obs));
    }

    if doy_dists.len() < n_neighbors {
        return None;
    }

    doy_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let neighbors: Vec<f64> = doy_dists.iter().take(n_neighbors).map(|d| d.1).collect();
    let y_arr = Array1::from_vec(neighbors);
    let mean = y_arr.mean().unwrap();
    let var = y_arr.mapv(|v| (v - mean).powi(2)).mean().unwrap();
    Some(var.sqrt())
}

/// Detect breaks: absolute difference > 2×RMSE for ≥6 consecutive observations.
///
/// Returns indices of break start positions (0-based).
pub fn detect_breaks(model: &SegmentModel, t_days: &[f64], y: &[f64]) -> Vec<usize> {
    let threshold = 2.0 * model.rmse;
    let mut breaks = Vec::new();
    let mut consecutive: usize = 0;

    for i in 0..t_days.len() {
        if !y[i].is_finite() {
            continue;
        }
        let pred = segment_predict(model, t_days[i]);
        if (y[i] - pred).abs() > threshold {
            consecutive += 1;
            if consecutive >= 6 {
                breaks.push(i.saturating_sub(5));
                consecutive = 0;
            }
        } else {
            consecutive = 0;
        }
    }
    breaks
}

// ── Batch raster reconstruction ────────────────────────────────────────────

/// Reconstruct all pixels in a flat raster cube `(n_time, n_pixels)`.
///
/// Returns `(predictions: Vec<f64>, qa_band: Vec<u32>)`.
pub fn reconstruct_raster(
    cube: &Array2<f64>,
    t_days: &[f64],
    target_t_day: f64,
    lasso_alpha: f64,
) -> (Vec<f64>, Vec<u32>) {
    let n_time = cube.nrows();
    let n_pixels = cube.ncols();
    debug_assert_eq!(n_time, t_days.len());

    let mut predictions = vec![f64::NAN; n_pixels];
    let mut qa_band = vec![0u32; n_pixels];

    for p in 0..n_pixels {
        let y: Vec<f64> = cube.column(p).to_vec();
        let result = fit_predict_pixel(t_days, &y, target_t_day, lasso_alpha);
        predictions[p] = result.prediction;
        qa_band[p] = result.qa;
    }

    (predictions, qa_band)
}

// ── Tests ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    // ── Design matrix tests ────────────────────────────────────────────

    #[test]
    fn test_select_model_order() {
        assert_eq!(select_model_order(0), 0);
        assert_eq!(select_model_order(5), 0);
        assert_eq!(select_model_order(6), 1);
        assert_eq!(select_model_order(12), 1);
        assert_eq!(select_model_order(17), 1);
        assert_eq!(select_model_order(18), 2);
        assert_eq!(select_model_order(23), 2);
        assert_eq!(select_model_order(24), 3);
        assert_eq!(select_model_order(100), 3);
    }

    #[test]
    fn test_make_design_matrix_dims() {
        let x = vec![100.0, 200.0, 300.0];
        assert_eq!(make_design_matrix(&x, 1, None).shape(), &[3, 3]);
        assert_eq!(make_design_matrix(&x, 2, None).shape(), &[3, 5]);
        assert_eq!(make_design_matrix(&x, 3, None).shape(), &[3, 7]);
    }

    #[test]
    fn test_make_design_matrix_values() {
        let x = vec![100.0, 200.0, 300.0, 400.0];
        let x_mean = 250.0;
        let dm = make_design_matrix(&x, 1, Some(x_mean));
        let w = 2.0 * std::f64::consts::PI / DAYS_PER_YEAR;

        for (i, &xi) in x.iter().enumerate() {
            assert_relative_eq!(dm[[i, 0]], (w * xi).cos(), epsilon = 1e-14);
            assert_relative_eq!(dm[[i, 1]], (w * xi).sin(), epsilon = 1e-14);
            assert_relative_eq!(dm[[i, 2]], xi - x_mean, epsilon = 1e-14);
        }
    }

    // ── LASSO solver tests ─────────────────────────────────────────────

    #[test]
    fn test_soft_threshold() {
        assert_eq!(soft_threshold(3.0, 1.0), 2.0);
        assert_eq!(soft_threshold(-3.0, 1.0), -2.0);
        assert_eq!(soft_threshold(0.5, 1.0), 0.0);
        assert_eq!(soft_threshold(-0.5, 1.0), 0.0);
        assert_eq!(soft_threshold(1.0, 1.0), 0.0);
    }

    #[test]
    fn test_lasso_fit_sparse_recovery() {
        let n = 50;
        let x_arr: Vec<f64> = (0..n).map(|i| i as f64 * 730.0 / n as f64).collect();
        let x_mean = x_arr.iter().sum::<f64>() / n as f64;
        let dm = make_design_matrix(&x_arr, 3, Some(x_mean));
        let w_base = 2.0 * std::f64::consts::PI / DAYS_PER_YEAR;

        let mut y: Vec<f64> = x_arr
            .iter()
            .map(|&xi| {
                0.5 + 0.3 * (w_base * xi).cos()
                    + 0.1 * (w_base * xi).sin()
                    + 0.05 * (xi - x_mean) / 1000.0
            })
            .collect();

        #[rustfmt::skip]
        let noise: [f64; 50] = [
            -0.012775,0.012441,0.000121,-0.000852,0.021004,0.015800,-0.008936,
            -0.032684,-0.007112,-0.022109,0.000216,0.012691,0.023368,-0.002604,
            0.026107,-0.008091,-0.012280,0.027299,-0.027013,-0.008489,0.006037,
            -0.011261,-0.012985,0.025899,0.021312,-0.011730,0.010209,-0.012142,
            0.000259,0.026634,-0.001622,0.000352,-0.006119,0.013828,0.003491,
            0.013803,0.019486,-0.007095,0.008903,-0.006592,-0.024049,0.010999,
            0.014161,-0.011810,0.017463,-0.005340,-0.012443,0.017989,-0.009911,
            -0.021966,
        ];
        for i in 0..n {
            y[i] += noise[i];
        }

        let y_arr = Array1::from_vec(y);
        let (coef, intercept) = lasso_fit(&dm, &y_arr, 0.1, 5000, 1e-12);

        assert!(
            coef[0] > 0.05,
            "cos(w) coef should be positive: {}",
            coef[0]
        );
        assert_relative_eq!(intercept, 0.5, epsilon = 0.05);
    }

    #[test]
    fn parse_zhu2015_config() {
        let cfg = Zhu2015Config::from_json(br#"{"lasso_alpha":0.1}"#).unwrap();
        assert!((cfg.lasso_alpha - 0.1).abs() < 1e-10);
    }

    #[test]
    fn zhu2015_validate_negative_alpha() {
        let mut cfg = Zhu2015Config::from_json(br#"{"lasso_alpha":0.1}"#).unwrap();
        cfg.lasso_alpha = -0.5;
        let err = cfg.validate().unwrap_err();
        assert!(format!("{err}").contains("lasso_alpha"));
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_all_nan_returns_nan_qa0() {
        let t = vec![0.0, 100.0, 200.0];
        let y = vec![f64::NAN, f64::NAN, f64::NAN];
        let r = fit_predict_pixel(&t, &y, 150.0, 0.1);
        assert!(r.prediction.is_nan());
        assert_eq!(r.qa, 0);
    }

    #[test]
    fn test_median_fallback_qa0() {
        let t: Vec<f64> = (0..5).map(|i| i as f64 * 100.0).collect();
        let y = vec![1.0, 5.0, 3.0, 2.0, f64::NAN];
        let r = fit_predict_pixel(&t, &y, 200.0, 0.1);
        assert_relative_eq!(r.prediction, 2.5, epsilon = 1e-10);
        assert_eq!(r.qa, 0);
    }

    #[test]
    fn test_single_valid_median_qa0() {
        let t = vec![0.0, 100.0];
        let y = vec![42.0, f64::NAN];
        let r = fit_predict_pixel(&t, &y, 50.0, 0.1);
        assert_relative_eq!(r.prediction, 42.0, epsilon = 1e-10);
        assert_eq!(r.qa, 0);
    }

    #[test]
    fn test_qa_encodes_model_order() {
        let w = 2.0 * std::f64::consts::PI / DAYS_PER_YEAR;
        let gen = |n: usize| -> (Vec<f64>, Vec<f64>) {
            let t: Vec<f64> = (0..n).map(|i| i as f64 * 10.0).collect();
            let y: Vec<f64> = t.iter().map(|&ti| 0.5 + 0.2 * (w * ti).cos()).collect();
            (t, y)
        };

        let (t30, y30) = gen(30);
        assert_eq!(fit_predict_pixel(&t30, &y30, t30[15], 0.1).qa, 3);

        let (t20, y20) = gen(20);
        assert_eq!(fit_predict_pixel(&t20, &y20, t20[10], 0.1).qa, 2);

        let (t12, y12) = gen(12);
        assert_eq!(fit_predict_pixel(&t12, &y12, t12[6], 0.1).qa, 1);
    }

    // ── Segment-aware tests ────────────────────────────────────────────

    #[test]
    fn test_fit_segment_clean_signal() {
        let n = 50;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 10.0).collect();
        let w = 2.0 * std::f64::consts::PI / DAYS_PER_YEAR;
        let y: Vec<f64> = t.iter().map(|&ti| 0.5 + 0.3 * (w * ti).cos()).collect();

        let seg = fit_segment(&t, &y, 0, n, 0.1).expect("should fit");
        assert_eq!(seg.order, 3);
        assert!(seg.rmse < 1.0);
        assert_relative_eq!(segment_predict(&seg, t[n / 2]), y[n / 2], epsilon = 0.15);
    }

    #[test]
    fn test_fit_segment_too_short() {
        let t = vec![0.0, 10.0, 20.0, 30.0, 40.0];
        let y = vec![1.0, 1.1, 1.2, 1.3, 1.4];
        assert!(fit_segment(&t, &y, 0, 5, 0.1).is_none());
    }

    #[test]
    fn test_detect_breaks_no_break() {
        let n = 30;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 10.0).collect();
        let w = 2.0 * std::f64::consts::PI / DAYS_PER_YEAR;
        let y: Vec<f64> = t.iter().map(|&ti| 0.5 + 0.3 * (w * ti).cos()).collect();
        let seg = fit_segment(&t, &y, 0, n, 0.1).unwrap();
        assert!(detect_breaks(&seg, &t, &y).is_empty());
    }

    #[test]
    fn test_reconstruct_raster_basic() {
        let n_time = 10;
        let n_pix = 4;
        let t: Vec<f64> = (0..n_time).map(|i| i as f64 * 100.0).collect();
        let w = 2.0 * std::f64::consts::PI / DAYS_PER_YEAR;

        let mut data = Vec::with_capacity(n_time * n_pix);
        for i in 0..n_time {
            for p in 0..n_pix {
                data.push(0.5 + (0.3 + p as f64 * 0.1) * (w * t[i]).cos());
            }
        }
        let cube = Array2::from_shape_vec((n_time, n_pix), data).unwrap();

        let (preds, qas) = reconstruct_raster(&cube, &t, t[n_time / 2], 0.1);
        assert_eq!(preds.len(), n_pix);
        assert_eq!(qas.len(), n_pix);
        assert!(preds.iter().all(|&p| p.is_finite()));
        assert!(qas.iter().all(|&q| q == 1));
    }
}
