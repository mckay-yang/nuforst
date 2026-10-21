// HANTS (Harmonic ANalysis of Time Series) — paper-faithful Rust port.
//
// Based on Roerink et al. (2000).
// Preserves paper-mandated semantics: NOF includes zero-frequency mean,
// SF controls directional outlier suppression, DOD enforces minimum retained
// observations, and FET gates the iterative rejection loop.
//
// ── Parameter quick-reference (paper names) ─────────────────────────────
//   NOF  : number of frequencies including zero frequency (mean)
//   SF   : Hi/Lo suppression flag → "low", "high", or "none"
//   IDRT : (not directly used; Python impl uses FET on residuals)
//   FET  : fit error tolerance — stop when all residuals ≤ FET in SF direction
//   DOD  : degree of overdeterminedness — keep at least (2*nof-1 + dod) points

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub enum HantsConfigError {
    Json(serde_json::Error),
    InvalidConfigValue { field: String, reason: String },
}

impl std::fmt::Display for HantsConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(err) => write!(f, "json parse error: {err}"),
            Self::InvalidConfigValue { field, reason } => {
                write!(f, "invalid config value for '{field}': {reason}")
            }
        }
    }
}

impl std::error::Error for HantsConfigError {}

/// HANTS reconstruction parameters.
///
/// Field names match `config/hants.json` exactly, with `Option<f64>` for the
/// optional `valid_min` / `valid_max` thresholds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HantsConfig {
    pub nof: u32,
    pub sf: String,
    pub fet: f64,
    pub dod: u32,
    pub valid_min: Option<f64>,
    pub valid_max: Option<f64>,
    pub period: f64,
}

impl HantsConfig {
    /// Load from a JSON byte slice.
    pub fn from_json(data: &[u8]) -> Result<Self, HantsConfigError> {
        serde_json::from_slice(data).map_err(HantsConfigError::Json)
    }

    /// Validate required fields.
    pub fn validate(&self) -> Result<(), HantsConfigError> {
        if self.nof == 0 {
            return Err(HantsConfigError::InvalidConfigValue {
                field: "nof".into(),
                reason: "number of frequencies must be > 0".into(),
            });
        }
        if self.dod == 0 {
            return Err(HantsConfigError::InvalidConfigValue {
                field: "dod".into(),
                reason: "degree of over-determination must be > 0".into(),
            });
        }
        Ok(())
    }
}

/// Result of a single-pixel HANTS fit.
#[derive(Debug, Clone)]
pub struct HantsResult {
    /// Whether the fit produced usable coefficients.
    pub valid: bool,
    /// Number of frequencies (NOF) used.
    pub nof: u32,
    /// Base period in days (typically 365.25).
    pub period: f64,
    /// Fitted harmonic coefficients.  Length = 2*nof - 1.
    /// When `valid` is false, contains NaN.
    pub coeffs: Vec<f64>,
    /// Fallback value: median of valid observations, or NaN.
    pub fill_value: f64,
    /// Number of outlier-rejection iterations performed.
    pub n_iterations: u32,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Design matrix
// ═══════════════════════════════════════════════════════════════════════════

/// Build the harmonic design matrix.
///
/// Column 0 is the constant term (ones).  For each frequency `f` in
/// `frequencies`, columns `cos(2π·f·t)` and `sin(2π·f·t)` are appended.
///
/// With NOF = 3 the matrix has 1 + 2·(3-1) = 5 columns, matching the
/// paper description that NOF includes the zero-frequency term.
pub fn make_design_matrix(t: &[f64], frequencies: &[f64]) -> Array2<f64> {
    let n = t.len();
    let n_cols = 1 + 2 * frequencies.len();
    let mut x = Array2::<f64>::zeros((n, n_cols));

    // column 0: ones (mean / intercept)
    x.column_mut(0).fill(1.0);

    // pairs of cos(ω·t), sin(ω·t) for each frequency
    for (k, &f) in frequencies.iter().enumerate() {
        if f == 0.0 {
            continue;
        }
        let omega = 2.0 * std::f64::consts::PI * f;
        let col_cos = 1 + 2 * k;
        let col_sin = 2 + 2 * k;
        for (i, &ti) in t.iter().enumerate() {
            x[[i, col_cos]] = (omega * ti).cos();
            x[[i, col_sin]] = (omega * ti).sin();
        }
    }

    x
}

// ═══════════════════════════════════════════════════════════════════════════
//  Linear solver — Gaussian elimination with partial pivoting
// ═══════════════════════════════════════════════════════════════════════════

/// Solve `A · x = b` via Gaussian elimination with partial pivoting.
///
/// `a` must be square.  Returns `None` when the matrix is numerically
/// singular (pivot < 1e-14).  This matches the Python behaviour where
/// `np.linalg.solve` raises `LinAlgError` for singular matrices.
fn gauss_solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    debug_assert_eq!(a.ncols(), n);
    debug_assert_eq!(b.len(), n);

    if n == 0 {
        return Some(Array1::zeros(0));
    }

    // augmented matrix [A | b]
    let mut aug = Array2::<f64>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // forward elimination with partial pivoting
    for col in 0..n {
        // find pivot row
        let mut pivot_row = col;
        let mut pivot_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }

        if pivot_val < 1e-14 {
            return None; // singular
        }

        // swap rows
        if pivot_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = tmp;
            }
        }

        // eliminate below
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / aug[[col, col]];
            for j in col..=n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    // back substitution
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Some(x)
}

/// Solve normal equations XᵀX · β = Xᵀy.
///
/// Returns `None` when X has fewer rows than columns or the system is singular.
fn solve_normal_equations(x: &Array2<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
    if x.nrows() == 0 || x.nrows() < x.ncols() {
        return None;
    }
    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);
    gauss_solve(&xtx, &xty)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Directional outlier detection
// ═══════════════════════════════════════════════════════════════════════════

/// Mark outliers based on the **SF** (suppression flag) semantics.
///
/// - `"low"`  → reject observations whose residual < -fet (too low)
/// - `"high"` → reject observations whose residual > +fet (too high)
/// - anything else (e.g. `"none"`) → reject `|residual| > fet`
fn detect_outliers(residuals: &Array1<f64>, fet: f64, sf: &str) -> Vec<bool> {
    residuals
        .iter()
        .map(|&r| match sf {
            "low" => r < -fet,
            "high" => r > fet,
            _ => r.abs() > fet,
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
//  HANTS single-pixel fit
// ═══════════════════════════════════════════════════════════════════════════

/// Fit harmonic coefficients to a single pixel's time series using the
/// paper-faithful HANTS iterative outlier rejection algorithm.
///
/// # Parameters
/// - `t` : time points (typically days since first observation)
/// - `y` : observed values (may contain NaN)
/// - `nof` : number of frequencies including zero-frequency (e.g. 3)
/// - `sf` : suppression flag — `"low"`, `"high"`, or `"none"` for no direction
/// - `valid_min` / `valid_max` : clamp valid range (excluded before fitting)
/// - `fet` : fit error tolerance — stop when all retained residuals ≤ fet
/// - `dod` : degree of overdeterminedness — final fit needs ≥ (2·nof-1 + dod) points
/// - `period` : base period in days (typically 365.25)
pub fn hants_fit(
    t: &[f64],
    y: &[f64],
    nof: u32,
    sf: &str,
    valid_min: Option<f64>,
    valid_max: Option<f64>,
    fet: f64,
    dod: u32,
    period: f64,
) -> HantsResult {
    let coeff_count = 1 + 2 * (nof.max(1) - 1) as usize;
    let num_params = coeff_count as u32;
    let min_obs = num_params + dod;

    // compute fill_value: nanmedian of valid observations
    let fill_value = nanmedian(y);

    let mut result = HantsResult {
        valid: false,
        nof,
        period,
        coeffs: vec![f64::NAN; coeff_count],
        fill_value,
        n_iterations: 0,
    };

    // ── initial valid mask (finite + range check) ─────────────────────────
    let valid_mask: Vec<bool> = y
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                return false;
            }
            if let Some(vmin) = valid_min {
                if v < vmin {
                    return false;
                }
            }
            if let Some(vmax) = valid_max {
                if v > vmax {
                    return false;
                }
            }
            true
        })
        .collect();

    let n_initial_valid = valid_mask.iter().filter(|&&b| b).count();
    if n_initial_valid == 0 {
        return result;
    }

    // gather initial working arrays
    let mut t_curr: Vec<f64> = t
        .iter()
        .zip(valid_mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&ti, _)| ti)
        .collect();
    let mut y_curr: Vec<f64> = y
        .iter()
        .zip(valid_mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&vi, _)| vi)
        .collect();

    if (y_curr.len() as u32) < min_obs {
        return result;
    }

    // frequencies for harmonic terms (excluding zero-frequency)
    let freqs: Vec<f64> = (1..nof).map(|i| i as f64 / period).collect();
    let max_iter = (y_curr.len()).min(50);

    // ── iterative outlier rejection ──────────────────────────────────────
    let mut coeffs: Option<Array1<f64>> = None;
    for it in 0..max_iter {
        let n_obs = y_curr.len();
        if (n_obs as u32) < min_obs {
            break;
        }

        let x_design = make_design_matrix(&t_curr, &freqs);
        let y_arr = Array1::from_vec(y_curr.clone());

        let coeffs_curr = match solve_normal_equations(&x_design, &y_arr) {
            Some(c) => c,
            None => break,
        };

        let y_pred = x_design.dot(&coeffs_curr);
        let residuals = &y_arr - &y_pred;

        let bad = detect_outliers(&residuals, fet, sf);
        let any_bad = bad.iter().any(|&b| b);

        result.n_iterations = (it + 1) as u32;

        if !any_bad {
            coeffs = Some(coeffs_curr);
            break; // converged
        }

        // keep only non-outlier observations
        let mut new_t = Vec::with_capacity(n_obs);
        let mut new_y = Vec::with_capacity(n_obs);
        for i in 0..n_obs {
            if !bad[i] {
                new_t.push(t_curr[i]);
                new_y.push(y_curr[i]);
            }
        }
        t_curr = new_t;
        y_curr = new_y;

        // if we hit the last iteration, the last computed coeffs ARE the
        // result (Python saves them even though loop ends on break below)
        if it == max_iter - 1 && !y_curr.is_empty() {
            let x_final = make_design_matrix(&t_curr, &freqs);
            let y_final = Array1::from_vec(y_curr.clone());
            coeffs = solve_normal_equations(&x_final, &y_final);
        }
    }

    // ── final validation ──────────────────────────────────────────────────
    if (y_curr.len() as u32) < min_obs {
        return result;
    }

    if coeffs.is_none() && !y_curr.is_empty() {
        let x_final = make_design_matrix(&t_curr, &freqs);
        let y_final = Array1::from_vec(y_curr.clone());
        coeffs = solve_normal_equations(&x_final, &y_final);
    }

    let coeffs = match coeffs {
        Some(c) => c,
        None => return result,
    };

    result.valid = true;
    for (i, &c) in coeffs.iter().enumerate() {
        if i < result.coeffs.len() {
            result.coeffs[i] = c;
        }
    }

    result
}

/// Predict the HANTS-reconstructed value at a single target time.
pub fn hants_predict(result: &HantsResult, target_t: f64) -> f64 {
    if !result.valid {
        return result.fill_value;
    }
    let nof = result.nof;
    let period = result.period;
    let freqs: Vec<f64> = (1..nof).map(|i| i as f64 / period).collect();
    let x_target = make_design_matrix(&[target_t], &freqs);
    let coeffs_arr = Array1::from_vec(result.coeffs.clone());
    let pred = x_target.dot(&coeffs_arr);
    pred[0]
}

/// Predict the HANTS-reconstructed curve at multiple target times.
pub fn hants_predict_curve(result: &HantsResult, target_t: &[f64]) -> Vec<f64> {
    target_t.iter().map(|&t| hants_predict(result, t)).collect()
}

/// Convenience: fit + predict in one call.
///
/// This is the Rust equivalent of Python's `hants_pixel()`.
pub fn hants_pixel(
    t: &[f64],
    y: &[f64],
    target_t: f64,
    nof: u32,
    sf: &str,
    valid_min: Option<f64>,
    valid_max: Option<f64>,
    fet: f64,
    dod: u32,
    period: f64,
) -> f64 {
    let result = hants_fit(t, y, nof, sf, valid_min, valid_max, fet, dod, period);
    hants_predict(&result, target_t)
}

/// Reconstruct a full raster using HANTS.
///
/// GDAL handles raster I/O and per-pixel traversal; this crate supplies the
/// HANTS pixel model.
#[cfg(feature = "raster")]
pub fn reconstruct_hants_geotiff<P: AsRef<std::path::Path>>(
    reader: &gdal::RasterReader,
    timestamps_days: &[f64],
    target_t_day: f64,
    nof: u32,
    sf: &str,
    valid_min: Option<f64>,
    valid_max: Option<f64>,
    fet: f64,
    dod: u32,
    period: f64,
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
            let pred = hants_pixel(
                ts, obs, targ, nof, sf, valid_min, valid_max, fet, dod, period,
            );
            if pred.is_finite() {
                pred
            } else {
                f64::NAN
            }
        },
    )
}

/// Convenience: fit + predict curve in one call.
pub fn hants_curve_pixel(
    t: &[f64],
    y: &[f64],
    target_t: &[f64],
    nof: u32,
    sf: &str,
    valid_min: Option<f64>,
    valid_max: Option<f64>,
    fet: f64,
    dod: u32,
    period: f64,
) -> Vec<f64> {
    let result = hants_fit(t, y, nof, sf, valid_min, valid_max, fet, dod, period);
    hants_predict_curve(&result, target_t)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Utility: NaN-aware median
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the median of finite values, ignoring NaN.
/// Returns NaN if no finite values are present.
fn nanmedian(data: &[f64]) -> f64 {
    let mut finite: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = finite.len();
    if n % 2 == 1 {
        finite[n / 2]
    } else {
        (finite[n / 2 - 1] + finite[n / 2]) / 2.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Design matrix shape ───────────────────────────────────────────────

    #[test]
    fn design_matrix_shape_nof3() {
        let t = vec![0.0, 1.0];
        let freqs = vec![1.0 / 365.25, 2.0 / 365.25];
        let x = make_design_matrix(&t, &freqs);
        // 1 (mean) + 2*2 = 5 columns
        assert_eq!(x.shape(), &[2, 5]);
    }

    #[test]
    fn design_matrix_first_column_is_ones() {
        let t = vec![10.0, 20.0, 30.0];
        let freqs = vec![1.0 / 365.25];
        let x = make_design_matrix(&t, &freqs);
        for i in 0..3 {
            assert!((x[[i, 0]] - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn design_matrix_cos_col_near_one_at_t_zero() {
        let t = vec![0.0];
        let freqs = vec![1.0 / 365.25];
        let x = make_design_matrix(&t, &freqs);
        // cos(0) = 1, sin(0) = 0
        assert!((x[[0, 1]] - 1.0).abs() < 1e-15);
        assert!(x[[0, 2]].abs() < 1e-15);
    }

    // ── Gaussian solver ───────────────────────────────────────────────────

    #[test]
    fn gauss_solve_2x2() {
        // [2 1] [x0] = [5]  →  x0=2, x1=1
        // [1 3] [x1]   [5]
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 3.0]).unwrap();
        let b = Array1::from_vec(vec![5.0, 5.0]);
        let x = gauss_solve(&a, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-12);
        assert!((x[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gauss_solve_identity() {
        let a = Array2::eye(3);
        let b = Array1::from_vec(vec![7.0, 8.0, 9.0]);
        let x = gauss_solve(&a, &b).unwrap();
        assert!((x[0] - 7.0).abs() < 1e-12);
        assert!((x[1] - 8.0).abs() < 1e-12);
        assert!((x[2] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn gauss_solve_singular_returns_none() {
        // singular matrix
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        let b = Array1::from_vec(vec![1.0, 2.0]);
        assert!(gauss_solve(&a, &b).is_none());
    }

    // ── SF semantics ──────────────────────────────────────────────────────

    #[test]
    fn detect_outliers_low() {
        let residuals = Array1::from_vec(vec![-0.5, -0.01, 0.0, 0.01, 0.5]);
        let bad = detect_outliers(&residuals, 0.1, "low");
        // low: only residuals < -0.1 are bad
        assert_eq!(bad, vec![true, false, false, false, false]);
    }

    #[test]
    fn detect_outliers_high() {
        let residuals = Array1::from_vec(vec![-0.5, -0.01, 0.0, 0.01, 0.5]);
        let bad = detect_outliers(&residuals, 0.1, "high");
        // high: only residuals > +0.1 are bad
        assert_eq!(bad, vec![false, false, false, false, true]);
    }

    #[test]
    fn detect_outliers_none() {
        let residuals = Array1::from_vec(vec![-0.5, -0.01, 0.0, 0.01, 0.5]);
        let bad = detect_outliers(&residuals, 0.1, "none");
        // none: abs(residual) > 0.1 → both tails
        assert_eq!(bad, vec![true, false, false, false, true]);
    }

    // ── SF is one-sided ───────────────────────────────────────────────────

    #[test]
    fn sf_is_one_sided_low_rejects_low_outliers() {
        let t: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![1.0, 1.0, 1.0, 10.0];
        // "low": reject low outliers; the 10.0 (high outlier) stays
        let low_pred = hants_pixel(&t, &y, 1.5, 1, "low", None, None, 0.1, 0, 365.25);
        assert!(
            low_pred > 2.0,
            "low SF should keep high values, got {low_pred}"
        );

        // "high": reject high outliers; the 10.0 is removed → fit ≈ 1.0
        let high_pred = hants_pixel(&t, &y, 1.5, 1, "high", None, None, 0.1, 0, 365.25);
        assert!(
            (high_pred - 1.0).abs() < 1e-6,
            "high SF should remove 10.0, got {high_pred}"
        );
    }

    // ── valid_min / valid_max filtering ───────────────────────────────────

    #[test]
    fn valid_min_filters_before_fit() {
        let t: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![0.2, 0.4, 0.8, 0.9];
        let result = hants_fit(&t, &y, 1, "low", Some(0.7), None, 0.1, 0, 365.25);
        assert!(result.valid);
        // only 0.8 and 0.9 survive valid_min=0.7 → mean ≈ 0.85
        let pred = hants_predict(&result, 1.5);
        assert!((pred - 0.85).abs() < 1e-6, "expected ~0.85, got {pred}");
    }

    // ── NOF=3 uses 5 parameters ───────────────────────────────────────────

    #[test]
    fn nof_3_uses_5_parameters() {
        let t: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = vec![1.0; 20];
        let result = hants_fit(&t, &y, 3, "low", None, None, 0.1, 5, 365.25);
        assert!(result.valid);
        assert_eq!(result.coeffs.len(), 5);
        // 1 + 2*(3-1) = 5
    }

    // ── Too few observations after rejection → invalid ────────────────────

    #[test]
    fn too_few_observations_returns_invalid() {
        let t: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![0.0, 0.0, 10.0, 10.0];
        // nof=1 → 1 param, dod=3 → need 4 observations
        // After rejection of outliers at fet=0.1 (none direction), only 2 remain
        let result = hants_fit(&t, &y, 1, "none", None, None, 0.1, 3, 365.25);
        assert!(!result.valid);
    }

    // ── FET stops iteration ───────────────────────────────────────────────

    #[test]
    fn stops_when_all_residuals_within_fet() {
        let t: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let period = 365.25;
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti / period).sin() * 0.1 + 0.5)
            .collect();
        let result = hants_fit(&t, &y, 1, "low", None, None, 1.0, 0, period);
        assert!(result.valid);
    }

    // ── Multiple outliers removed iteratively ─────────────────────────────

    #[test]
    fn iterative_rejection_removes_multiple_outliers() {
        let t: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut y: Vec<f64> = vec![0.5; 100];
        y[20] = -10.0;
        y[40] = -10.0;
        y[60] = -10.0;
        let result = hants_fit(&t, &y, 1, "low", None, None, 0.1, 0, 365.25);
        assert!(result.valid);
        let pred = hants_predict(&result, 50.0);
        assert!((pred - 0.5).abs() < 0.5);
        assert!(result.n_iterations >= 1);
    }

    // ── Iteration cap ─────────────────────────────────────────────────────

    #[test]
    fn converges_with_iteration_cap() {
        let t: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let mut y: Vec<f64> = vec![0.5; 200];
        for i in (0..200).step_by(7) {
            y[i] = -5.0;
        }
        let result = hants_fit(&t, &y, 1, "low", None, None, 0.1, 0, 365.25);
        assert!(result.valid);
        assert!(result.n_iterations >= 1);
        assert!(result.n_iterations <= 50); // cap
        let pred = hants_predict(&result, 100.0);
        assert!((pred - 0.5).abs() < 0.5);
    }

    // ── Default FET handles DN scale ──────────────────────────────────────

    #[test]
    fn default_hants_fet_handles_dn_scale_without_median_fallback() {
        let t: Vec<f64> = (0..80).map(|i| i as f64 * 30.0).collect();
        let period = 365.25;
        let y: Vec<f64> = t
            .iter()
            .enumerate()
            .map(|(i, &ti)| {
                let val = 1800.0 + 400.0 * (2.0 * std::f64::consts::PI * ti / period).sin();
                if i % 13 == 0 {
                    val - 1200.0
                } else {
                    val
                }
            })
            .collect();
        let result = hants_fit(&t, &y, 3, "high", None, None, 500.0, 5, period);
        assert!(result.valid);
        let curve = hants_predict_curve(&result, &t);
        let mean_curve: f64 = curve.iter().sum::<f64>() / curve.len() as f64;
        let var: f64 =
            curve.iter().map(|&v| (v - mean_curve).powi(2)).sum::<f64>() / curve.len() as f64;
        let std_dev = var.sqrt();
        assert!(
            std_dev > 100.0,
            "curve std dev should be > 100, got {std_dev}"
        );
        let max_val = curve.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!(
            max_val < 5000.0,
            "max curve value should be < 5000, got {max_val}"
        );
    }

    // ── NaN handling ──────────────────────────────────────────────────────

    #[test]
    fn nanmedian_returns_nan_for_all_nan() {
        let data = vec![f64::NAN, f64::NAN];
        assert!(nanmedian(&data).is_nan());
    }

    #[test]
    fn nanmedian_skips_nan() {
        let data = vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0];
        let m = nanmedian(&data);
        assert!((m - 3.0).abs() < 1e-15);
    }

    #[test]
    fn nanmedian_even_count() {
        let data = vec![1.0, 3.0, f64::NAN, 5.0, 7.0, f64::NAN];
        let m = nanmedian(&data);
        assert!(
            (m - 4.0).abs() < 1e-15,
            "even median should be (3+5)/2=4, got {m}"
        );
    }

    // ── Empty / all-invalid input ─────────────────────────────────────────

    #[test]
    fn all_nan_yields_invalid() {
        let t = vec![0.0, 1.0, 2.0];
        let y = vec![f64::NAN; 3];
        let result = hants_fit(&t, &y, 1, "none", None, None, 0.1, 0, 365.25);
        assert!(!result.valid);
        assert!(result.fill_value.is_nan());
    }

    // ── hants_pixel convenience ───────────────────────────────────────────

    #[test]
    fn hants_pixel_returns_finite() {
        let t: Vec<f64> = (0..23).map(|i| i as f64 * 16.0).collect();
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| 0.25 + 0.05 * (2.0 * std::f64::consts::PI * ti / 365.25).sin())
            .collect();
        let pred = hants_pixel(&t, &y, t[4], 3, "none", None, None, 500.0, 1, 365.25);
        assert!(pred.is_finite(), "prediction should be finite, got {pred}");
    }

    // ── hants_curve_pixel returns correct shape ────────────────────────────

    #[test]
    fn hants_curve_pixel_returns_correct_shape() {
        let t: Vec<f64> = (0..23).map(|i| i as f64 * 16.0).collect();
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| 0.25 + 0.05 * (2.0 * std::f64::consts::PI * ti / 365.25).sin())
            .collect();
        let target: Vec<f64> = t[..3].to_vec();
        let curve = hants_curve_pixel(&t, &y, &target, 3, "none", None, None, 500.0, 1, 365.25);
        assert_eq!(curve.len(), 3);
        assert!(curve.iter().all(|&v| v.is_finite()));
    }

    const HANTS_JSON: &str = r#"{
        "nof": 3,
        "sf": "high",
        "fet": 500.0,
        "dod": 5,
        "valid_min": null,
        "valid_max": null,
        "period": 365.25
    }"#;

    #[test]
    fn parse_hants_config() {
        let cfg = HantsConfig::from_json(HANTS_JSON.as_bytes()).unwrap();
        assert_eq!(cfg.nof, 3);
        assert_eq!(cfg.sf, "high");
        assert!((cfg.fet - 500.0).abs() < 1e-10);
        assert_eq!(cfg.dod, 5);
        assert!(cfg.valid_min.is_none());
        assert!(cfg.valid_max.is_none());
        assert!((cfg.period - 365.25).abs() < 1e-10);
    }

    #[test]
    fn hants_validate_zero_nof() {
        let mut cfg = HantsConfig::from_json(HANTS_JSON.as_bytes()).unwrap();
        cfg.nof = 0;
        let err = cfg.validate().unwrap_err();
        assert!(format!("{err}").contains("nof"));
    }
}
